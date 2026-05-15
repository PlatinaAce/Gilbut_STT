import os
import warnings
import logging
import asyncio
import json
import re
import time
import numpy as np
import sounddevice as sd
import torch
import websockets
import winsound
import threading
from TTS_wd import start_tts_server
from collections import deque
from transformers import WhisperForConditionalGeneration, WhisperProcessor

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

# transformers 경고 로그 숨기기
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.generation").setLevel(logging.ERROR)
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

# ===============================설정값============================

# 오디오 설정
SAMPLE_RATE = 16000
CHUNK_SIZE = 512
CHUNK_DURATION = CHUNK_SIZE / SAMPLE_RATE
INPUT_DEVICE_INDEX = None

# PC 스피커 알림음 설정
BEEP_FREQUENCY = 1000
BEEP_DURATION = 200
AFTER_BEEP_DELAY = 0.2

# tiny 모델 경로
WAKE_MODEL_PATH = r"F:\dataset\tinyfinal\outputs\whisper-tiny-gilbut"

# turbo 모델 경로
COMMAND_MODEL_PATH = r"F:\dataset\turbo\outputs\whisper-large-turbo-lora-gilbut"

# 만약 LoRA adapter만 저장한 경우 True로 바꾸고 BASE_MODEL_NAME을 지정해야 함.
# full fine-tuned model이면 False.
USE_WAKE_PEFT = False
WAKE_BASE_MODEL_NAME = "openai/whisper-tiny"

USE_COMMAND_PEFT = False
COMMAND_BASE_MODEL_NAME = "openai/whisper-large-v3-turbo"


# Spring Boot WebSocket 설정
SPRING_WS_URL = "ws://localhost:8080/whisper"


# VAD 설정
VAD_THRESHOLD = 0.5 # 말소리로 판단할 확률 
PRE_ROLL_SECONDS = 0.4 # 발화 앞뒤로 저장할 시간(초 단위)

# Tiny 모델 설정
WAKE_MIN_SPEECH_SECONDS = 0.3 # 발화 최소 길이(이 이하로 짧은건 무시)
WAKE_MAX_SPEECH_SECONDS = 3.0 # 발화 최대 길이(이 이상으로 길면 강제 종료)
WAKE_SILENCE_SECONDS = 0.5 # 발화중 침묵 최대 길이(이 이상으로 침묵 지속되면 발화 종료로 간주)
WAKE_WAIT_TIMEOUT = None  # wake는 계속 기다리는 구조라 timeout 없음

# Turbo 모델 설정
COMMAND_MIN_SPEECH_SECONDS = 0.7 # 발화 최소 길이(이 이하로 짧은건 무시)
COMMAND_MAX_SPEECH_SECONDS = 8.0 # 발화 최대 길이(이 이상으로 길면 강제 종료)
COMMAND_SILENCE_SECONDS = 0.9 # 발화중 침묵 최대 길이(이 이상으로 침묵 지속되면 발화 종료로 간주)
COMMAND_WAIT_TIMEOUT = 6.0  # 띠링 후 6초 안에 말 안 하면 다시 wake 대기

# Tiny 모델이 wake word로 인식할 수 있는 후보들
# 추가할 만한 부분이 있으면 넣으시면 됩니다
WAKE_KEYWORDS = [
    "길벗",
    "길벗아",
    "길버",
    "길벋",
    "길봇",
    "길버사"
]

# Turbo 모델 초기 프롬프트
# 추가할 만한 명령어나 키워드가 있으면 넣으시면 됩니다
COMMAND_INITIAL_PROMPT = (
    "가천대학교 안내로봇 명령어입니다. "
    "비전타워, 중앙도서관, 학생회관, 가천관, 글로벌센터, "
    "공과대학, 산학협력관, 기숙사, 정문, 후문, "
    "AI공학관, 바이오나노대학, 전자정보도서관, "
    "안내해줘, 데려다줘, 가자, 어디야."
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Silero VAD 모델 저장 변수
vad_model = None

# Tiny 모델 processor와 model 저장 변수
wake_processor = None
wake_model = None

# Turbo 모델 processor와 model 저장 변수
command_processor = None
command_model = None

# TTS가 말하는 중인지 STT.py와 tts_server.py가 공유하는 이벤트
tts_playing_event = threading.Event()

# ================================함수===============================

# 오디오 출력 function: 현재 PC에서 잡히는 오디오 장치 목록 출력
def print_audio_devices():
    print("\n========== Audio Devices ==========")
    print(sd.query_devices())
    print("===================================\n")


# 정규화 function: 한국어 텍스트에서 공백과 특수문자 제거, 영어와 숫자는 유지
def normalize_korean_text(text: str) -> str:
    if text is None:
        return ""

    text = text.strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^가-힣a-zA-Z0-9]", "", text)
    return text


# 비프음 function: 길벗이 호출된 것을 소리로 알림
def play_ready_sound():
    print("[INFO] Playing ready sound...")
    winsound.Beep(BEEP_FREQUENCY, BEEP_DURATION)


# ============================모델 로딩============================

# VAD 모델 로딩 function: torch.hub로 Silero VAD 모델 불러옮
def load_silero_vad():
    print("[INFO] Loading Silero VAD...")

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True
    )

    model.to("cpu")
    model.eval()

    print("[INFO] Silero VAD loaded.")
    return model


# Whisper 모델 로딩 function: Hugging Face Transformers로 Whisper 모델과 processor 불러옮
def load_whisper_model(model_path: str, base_model_name: str, use_peft: bool):
    print(f"[INFO] Loading Whisper model from: {model_path}")

    if use_peft: # Case 1: Tiny 모델
        if PeftModel is None:
            raise RuntimeError(
                "peft가 설치되어 있지 않습니다. LoRA adapter를 쓰려면 pip install peft 하세요."
            )

        print(f"[INFO] PEFT mode. Base model: {base_model_name}")

        processor = WhisperProcessor.from_pretrained(model_path)

        base_model = WhisperForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        )

        model = PeftModel.from_pretrained(base_model, model_path)

        # 추론 속도와 단순화를 위해 adapter를 base model에 merge
        # merge가 불가능한 환경이면 아래 줄에서 에러가 날 수 있음
        model = model.merge_and_unload()

    else: # Case 2: Turbo 모델
        processor = WhisperProcessor.from_pretrained(model_path)

        model = WhisperForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        )

    model.to(DEVICE)
    model.eval()

    # 추론용 설정
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    print(f"[INFO] Whisper model loaded on {DEVICE}.")
    return processor, model


# 모델 로딩 function: VAD, Tiny, Turbo 모델 모두 로딩해서 전역 변수에 저장
def load_all_models():
    global vad_model
    global wake_processor, wake_model
    global command_processor, command_model

    print(f"[INFO] Device: {DEVICE}")

    vad_model = load_silero_vad()

    print("[INFO] Loading wake model...")
    wake_processor, wake_model = load_whisper_model(
        model_path=WAKE_MODEL_PATH,
        base_model_name=WAKE_BASE_MODEL_NAME,
        use_peft=USE_WAKE_PEFT
    )

    print("[INFO] Loading command model...")
    command_processor, command_model = load_whisper_model(
        model_path=COMMAND_MODEL_PATH,
        base_model_name=COMMAND_BASE_MODEL_NAME,
        use_peft=USE_COMMAND_PEFT
    )

    print("[INFO] All models loaded.")

# ================================여기까지 모델로딩 ====================




# ================================VAD 관련 함수===============================

# VAD 설정 function: 호출부분과 명령 부분의 설정값 다르게 가져오기
def get_vad_config(mode: str):
    if mode == "wake":
        return {
            "min_speech_seconds": WAKE_MIN_SPEECH_SECONDS,
            "max_speech_seconds": WAKE_MAX_SPEECH_SECONDS,
            "silence_seconds": WAKE_SILENCE_SECONDS,
            "wait_timeout": WAKE_WAIT_TIMEOUT,
            "log_prefix": "Wake",
        }

    if mode == "command":
        return {
            "min_speech_seconds": COMMAND_MIN_SPEECH_SECONDS,
            "max_speech_seconds": COMMAND_MAX_SPEECH_SECONDS,
            "silence_seconds": COMMAND_SILENCE_SECONDS,
            "wait_timeout": COMMAND_WAIT_TIMEOUT,
            "log_prefix": "Command",
        }

    raise ValueError(f"Unknown VAD mode: {mode}")


# VAD 확률 계산 function
def get_speech_probability(audio_chunk: np.ndarray) -> float:
    if audio_chunk.ndim != 1:
        audio_chunk = audio_chunk.reshape(-1)

    audio_tensor = torch.from_numpy(audio_chunk).float()

    with torch.no_grad():
        prob = vad_model(audio_tensor, SAMPLE_RATE).item()

    return prob

#===============================여기까지 VAD 관련 함수==========================


# 발화 녹음 function: 발화를 잘라내서 반환함
def record_one_utterance(mode: str = "wake", wait_timeout=None):
    # 호출 부분과 명령어 부분 각각에 맞는 VAD 설정값 가져오기
    config = get_vad_config(mode)

    # VAD 설정값 변수로 풀기
    min_speech_seconds = config["min_speech_seconds"]
    max_speech_seconds = config["max_speech_seconds"]
    silence_seconds = config["silence_seconds"]
    log_prefix = config["log_prefix"]

    if wait_timeout is None:
        wait_timeout = config["wait_timeout"]

    pre_roll_chunks = max(1, int(PRE_ROLL_SECONDS / CHUNK_DURATION))
    silence_limit_chunks = max(1, int(silence_seconds / CHUNK_DURATION))

    # 말 시작 전의 오디오를 조금 저장해두기 위한 버퍼
    # VAD가 말 시작을 늦게 감지해도 앞부분이 잘리지 않게 해줌
    pre_roll_buffer = deque(maxlen=pre_roll_chunks)

    # 실제 발화 저장 버퍼
    speech_buffer = []

    is_speaking = False
    silence_count = 0
    speech_start_time = None
    wait_start_time = time.time()

    # Silero VAD 내부 상태 초기화
    if hasattr(vad_model, "reset_states"):
        vad_model.reset_states()

    print(f"[INFO] Waiting for {mode} speech...")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            device=INPUT_DEVICE_INDEX
        ) as stream:

            while True:
                # TTS가 말하기 시작하면 현재 STT 녹음을 즉시 중단
                if tts_playing_event.is_set():
                    print(f"[INFO] TTS is playing. Stop {mode} recording.")
                    return None

                # 아직 말이 시작되지 않은 상태에서 timeout 검사
                if not is_speaking and wait_timeout is not None:
                    elapsed_wait = time.time() - wait_start_time
                    if elapsed_wait > wait_timeout:
                        print(f"[WARN] {log_prefix} wait timeout.")
                        return None

                # 마이크에서 chunk 하나 읽기
                audio_chunk, overflowed = stream.read(CHUNK_SIZE)

                if overflowed:
                    print(f"[WARN] Audio input overflowed in {mode} mode.")

                # shape: (CHUNK_SIZE, 1) -> (CHUNK_SIZE,)
                audio_chunk = audio_chunk.reshape(-1).astype(np.float32)

                # 현재 chunk가 말소리인지 확률 계산
                speech_prob = get_speech_probability(audio_chunk)
                is_speech_chunk = speech_prob >= VAD_THRESHOLD

                if not is_speaking:
                    # 아직 말 시작 전이면 pre-roll에 계속 저장
                    pre_roll_buffer.append(audio_chunk)

                    if is_speech_chunk:
                        # 말 시작 감지
                        is_speaking = True
                        speech_start_time = time.time()
                        silence_count = 0

                        # 말 시작 직전 pre-roll까지 포함해서 저장
                        speech_buffer = list(pre_roll_buffer)
                        speech_buffer.append(audio_chunk)

                        print(f"[INFO] {log_prefix} speech started. prob={speech_prob:.3f}")

                else:
                    # 말하는 중이면 계속 저장
                    speech_buffer.append(audio_chunk)

                    if is_speech_chunk:
                        silence_count = 0
                    else:
                        silence_count += 1

                    speech_duration = time.time() - speech_start_time

                    # 너무 길어지면 강제 종료
                    if speech_duration >= max_speech_seconds:
                        print(f"[WARN] {log_prefix} max length reached. Forced stop.")
                        break

                    # 침묵이 일정 시간 지속되면 발화 종료
                    if silence_count >= silence_limit_chunks:
                        print(f"[INFO] {log_prefix} speech ended.")
                        break

    except Exception as e:
        print(f"[ERROR] Audio recording failed in {mode} mode: {e}")
        return None

    if not speech_buffer:
        return None

    speech_audio = np.concatenate(speech_buffer).astype(np.float32)
    speech_seconds = len(speech_audio) / SAMPLE_RATE

    if speech_seconds < min_speech_seconds:
        print(f"[INFO] {log_prefix} speech too short. Ignored: {speech_seconds:.2f} sec")
        return None

    print(f"[INFO] {log_prefix} utterance length: {speech_seconds:.2f} sec")
    return speech_audio




def transcribe_with_whisper(
    audio: np.ndarray,
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    initial_prompt: str = None,
    max_new_tokens: int = 128,
    num_beams: int = 5
) -> str:
    
    if audio is None or len(audio) == 0:
        return ""

    audio = audio.astype(np.float32).reshape(-1)

    inputs = processor(
    audio,
    sampling_rate=SAMPLE_RATE,
    return_tensors="pt",
    return_attention_mask=True
    )

    input_features = inputs.input_features.to(DEVICE)

    attention_mask = None
    
    if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
        attention_mask = inputs.attention_mask.to(DEVICE)

    if DEVICE == "cuda":
        input_features = input_features.half()

    # 한국어 transcribe 고정
    # 짧은 명령어에서 language auto detection을 쓰면 흔들릴 수 있으므로 고정하는 게 좋음
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="ko",
        task="transcribe"
    )

    generate_kwargs = {
    "input_features": input_features,
    "forced_decoder_ids": forced_decoder_ids,
    "max_new_tokens": max_new_tokens,
    "num_beams": num_beams,
    "do_sample": False,
    }

    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask

    # initial_prompt는 transformers 버전에 따라 prompt_ids 지원 여부가 다를 수 있음
    # 지원되면 사용하고, 안 되면 그냥 prompt 없이 진행
    if initial_prompt:
        try:
            prompt_ids = processor.get_prompt_ids(
                initial_prompt,
                return_tensors="pt"
            ).to(DEVICE)
            generate_kwargs["prompt_ids"] = prompt_ids
        except Exception:
            pass

    with torch.no_grad():
        predicted_ids = model.generate(**generate_kwargs)

    text = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return text.strip()


# 호출 감지 function: 길벗을 불렀는지 확인함
def detect_wake_word(wake_audio: np.ndarray):
    
    wake_text = transcribe_with_whisper(
        audio=wake_audio,
        processor=wake_processor,
        model=wake_model,
        initial_prompt="길벗, 길벗아, 가천대학교 안내로봇 호출어입니다.",
        max_new_tokens=32,
        num_beams=3
    )

    normalized = normalize_korean_text(wake_text)

    print(f"[INFO] Wake ASR text: {wake_text}")
    print(f"[INFO] Wake normalized: {normalized}")

    # 10글자까지만 추출. 이 안에 길벗이 있는지 확인
    head = normalized[:10]

    for keyword in WAKE_KEYWORDS:
        if keyword in head:
            return True, wake_text

    return False, wake_text



def transcribe_command(command_audio: np.ndarray) -> str:

    raw_command_text = transcribe_with_whisper(
        audio=command_audio,
        processor=command_processor,
        model=command_model,
        initial_prompt=COMMAND_INITIAL_PROMPT,
        max_new_tokens=96,
        num_beams=5
    )

    print(f"[INFO] Raw command text: {raw_command_text}")
    return raw_command_text


def clean_command_text(raw_text: str) -> str:
    if raw_text is None:
        return ""

    text = raw_text.strip()

    # 공백 정리
    text = re.sub(r"\s+", " ", text)

    # 앞쪽에 들어간 wake word 후보 제거
    # 예: "길벗아 비전타워까지 안내해줘" -> "비전타워까지 안내해줘"
    wake_remove_patterns = [
        r"^길벗아?\s*",
        r"^길버\s*",
        r"^길벋\s*",
        r"^길봇\s*",
    ]

    for pattern in wake_remove_patterns:
        text = re.sub(pattern, "", text)

    # 불필요한 기호 제거
    text = re.sub(r"[\"'`]", "", text)
    text = text.strip()

    return text



# 전송 function: Spring Boot WebSocket 서버로 명령어 텍스트 전송
async def send_to_spring_async(payload: dict):
    async with websockets.connect(SPRING_WS_URL) as websocket:
        await websocket.send(json.dumps(payload, ensure_ascii=False))


def send_to_spring(command_text: str) -> bool:
    payload = {
        "status": "SUCCESS",
        "text": command_text,
        "time": int(time.time() * 1000),
    }

    print("[INFO] Sending payload to Spring Boot:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    try:
        asyncio.run(send_to_spring_async(payload))
        print("[INFO] Sent to Spring Boot server.")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to send to Spring Boot: {e}")
        return False



# 메인함수
def main():
    print_audio_devices()

    load_all_models()

    start_tts_server(tts_playing_event)

    print("\n[INFO] Ready. Waiting for wake word...\n")

    while True:
        try:
            if tts_playing_event.is_set():
                print("[INFO] TTS is playing. Main loop is waiting.")
                time.sleep(0.1)
                continue
            
            # 호출어 부분
            print("\n[INFO] Waiting for wake word...")

            wake_audio = record_one_utterance(mode="wake")

            if wake_audio is None:
                # wake 모드에서는 거의 안 나오지만, 안전 처리
                continue

            wake_detected, wake_text = detect_wake_word(wake_audio)

            if not wake_detected:
                print("[INFO] Wake word not detected. Ignored.")
                continue
            
            # 호출이 감지
            print("[INFO] Wake word detected\n")
            play_ready_sound()

            # 비프음이 인식에 들어가는 것을 방지
            time.sleep(AFTER_BEEP_DELAY)

            # 명령어 부분
            print("[INFO] Waiting for command...")

            command_audio = record_one_utterance(
                mode="command",
                wait_timeout=COMMAND_WAIT_TIMEOUT
            )

            if command_audio is None:
                print("[WARN] No command detected. Back to wake mode.")
                continue
            
            # 발화가 감지
            raw_command_text = transcribe_command(command_audio)
            
            # Spring Boot로 보내기 전에 명령어 텍스트 정리(길벗 정리)
            command_text = clean_command_text(raw_command_text)

            print(f"[INFO] Command text: {command_text}")

            # 너무 짧은 명령어는 무시 (예: "응", "네", "그래")
            if len(command_text) < 2:
                print("[WARN] Command text too short. Ignored.")
                continue


            # Spring Boot 서버로 전송
            send_to_spring(
                command_text = command_text
            )
            
            print("[INFO] Command process finished. Back to wake mode.")

        except KeyboardInterrupt:
            print("\n[INFO] Program terminated by user.")
            break

        except Exception as e:
            print(f"[ERROR] Main loop error: {e}")
            print("[INFO] Recovering. Back to wake mode.")
            time.sleep(1.0)


if __name__ == "__main__":
    main()