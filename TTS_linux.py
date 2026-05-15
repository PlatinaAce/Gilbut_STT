import asyncio
import json
import os
import time
import threading
import websockets
import pyttsx3


# TTS 서버 설정
# 같은 Linux PC에서 Spring Boot가 접속하면 127.0.0.1도 가능.
# 다른 PC/Spring Boot 컨테이너에서 접속할 수 있게 하려면 0.0.0.0 사용.
# 예: TTS_HOST=0.0.0.0 python STT_linux.py
TTS_HOST = os.getenv("TTS_HOST", "0.0.0.0")
TTS_PORT = int(os.getenv("TTS_PORT", "7070"))

# TTS가 끝난 후 스피커 잔향 방지 시간
TTS_END_DELAY = float(os.getenv("TTS_END_DELAY", "0.5"))

# TTS가 말하는 속도
TTS_RATE = int(os.getenv("TTS_RATE", "180"))

# TTS 볼륨
TTS_VOLUME = float(os.getenv("TTS_VOLUME", "1.0"))

# TTS가 여러 개 동시에 들어왔을 때 겹쳐서 말하지 않도록 잠금
_tts_lock = threading.Lock()


# 한국어 음성 선택 function
def select_korean_voice(engine):
    try:
        voices = engine.getProperty("voices")
    except Exception as e:
        print(f"[TTS][WARN] Failed to get voices: {e}")
        return

    selected_voice_id = None

    for voice in voices:
        voice_id = str(getattr(voice, "id", "")).lower()
        voice_name = str(getattr(voice, "name", "")).lower()

        if (
            "ko" in voice_id
            or "korean" in voice_name
            or "heami" in voice_name
        ):
            selected_voice_id = voice.id
            break

    if selected_voice_id is not None:
        engine.setProperty("voice", selected_voice_id)
        print(f"[TTS] Korean voice selected: {selected_voice_id}")
    else:
        print("[TTS][WARN] Korean voice not found. Using default voice.")


# TTS 재생 function
def speak_text(text: str):
    if text is None:
        return

    text = str(text).strip()

    if len(text) == 0:
        return

    print(f"[TTS] Speaking: {text}")

    engine = None

    try:
        # TTS 엔진 생성
        # Linux에서는 espeak/espeak-ng 패키지가 설치되어 있어야 pyttsx3가 동작함.
        engine = pyttsx3.init()

        # 한국어 음성 선택 시도
        select_korean_voice(engine)

        # 속도 설정
        engine.setProperty("rate", TTS_RATE)

        # 볼륨 설정
        engine.setProperty("volume", TTS_VOLUME)

        # 말할 문장 등록
        engine.say(text)

        # 실제로 말하기 시작
        # 이 함수는 TTS가 끝날 때까지 blocking 됨
        engine.runAndWait()

    except Exception as e:
        print(f"[TTS][ERROR] Failed to speak text: {e}")

    finally:
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

    print("[TTS] Speaking finished.")



# JSON 처리 function
def handle_tts_payload(raw_message: str, tts_playing_event: threading.Event):
    try:
        payload = json.loads(raw_message)

    except json.JSONDecodeError:
        print("[TTS][ERROR] Invalid JSON received.")
        print(f"[TTS][ERROR] Raw message: {raw_message}")
        return

    print("[TTS] Received payload:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    status = str(payload.get("status", "")).upper()

    if status != "SUCCESS":
        print(f"[TTS][WARN] status is not SUCCESS: {status}")
        return

    tts_message = (
        payload.get("tts_message")
        or payload.get("ttsMessage")
        or payload.get("message")
        or payload.get("text")
        or ""
    )

    tts_message = str(tts_message).strip()

    if len(tts_message) == 0:
        print("[TTS][WARN] Empty tts_message. Ignored.")
        return


    # TTS 재생
    # 여러 메시지가 거의 동시에 들어와도 TTS가 겹치지 않도록 lock 사용
    with _tts_lock:
        try:
            # 여기서 STT.py에게 "지금 TTS 말하는 중"이라고 알림
            tts_playing_event.set()
            print("[TTS] tts_playing_event SET. STT should pause.")

            # 실제 TTS 재생
            speak_text(tts_message)

            # 스피커 잔향 방지 대기
            time.sleep(TTS_END_DELAY)

        finally:
            # TTS가 끝났으므로 STT 재개 허용
            tts_playing_event.clear()
            print("[TTS] tts_playing_event CLEARED. STT can resume.")



# websocket handler function
async def tts_websocket_handler(websocket, tts_playing_event: threading.Event):
    """
    Spring Boot가 ws://localhost:7070/tts 로 접속하면 실행되는 함수.

    들어오는 메시지를 계속 기다리다가,
    메시지가 오면 handle_tts_payload()로 처리함.
    """

    print("[TTS] Spring Boot connected to TTS server.")

    try:
        async for message in websocket:
            # TTS 재생은 blocking 작업이라 별도 thread에서 실행
            # 그래야 WebSocket 이벤트 루프가 완전히 멈추지 않음
            await asyncio.to_thread(
                handle_tts_payload,
                message,
                tts_playing_event
            )

    except websockets.exceptions.ConnectionClosed:
        print("[TTS] Spring Boot connection closed.")

    except Exception as e:
        print(f"[TTS][ERROR] WebSocket handler error: {e}")



# 서버 실행 function
async def run_tts_server(tts_playing_event: threading.Event):
    
    print("[TTS] Starting TTS WebSocket server...")
    print(f"[TTS] URL: ws://{TTS_HOST}:{TTS_PORT}/tts")

    async def handler(websocket):
        await tts_websocket_handler(websocket, tts_playing_event)

    async with websockets.serve(handler, TTS_HOST, TTS_PORT):
        print("[TTS] TTS WebSocket server is running.")
        await asyncio.Future()



# STT에서 TTS 서버 호출하는 function
def start_tts_server(tts_playing_event: threading.Event):

    def server_thread_main():
        try:
            asyncio.run(run_tts_server(tts_playing_event))

        except Exception as e:
            print(f"[TTS][ERROR] TTS server failed: {e}")

    thread = threading.Thread(
        target=server_thread_main,
        daemon=True
    )

    thread.start()

    print("[TTS] TTS server thread started.")

    return thread



# 단독 실행 테스트용
if __name__ == "__main__":
    test_tts_event = threading.Event()

    print("[TTS] Running TTS_linux.py directly.")
    print("[TTS] Press Ctrl+C to stop.")

    try:
        asyncio.run(run_tts_server(test_tts_event))

    except KeyboardInterrupt:
        print("\n[TTS] TTS server stopped by user.")