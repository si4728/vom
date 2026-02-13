import time
import json
import base64
import queue
import argparse

import numpy as np
import sounddevice as sd
import paho.mqtt.client as mqtt


def pcm16_bytes_to_float32(pcm: bytes) -> np.ndarray:
    """PCM16(bytes) -> float32 [-1,1]"""
    audio_i16 = np.frombuffer(pcm, dtype=np.int16)
    audio_f32 = (audio_i16.astype(np.float32) / 32767.0)
    return audio_f32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="218.146.225.166")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--topic", default="audio/pcm16/mono")
    ap.add_argument("--qos", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--client-id", default=f"audio-sub-{int(time.time())}")
    ap.add_argument("--jitter-ms", type=int, default=80, help="초기 버퍼(ms) (끊김 방지용)")
    args = ap.parse_args()

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)

    stream = None
    stream_rate = None

    def start_stream(rate: int):
        nonlocal stream, stream_rate
        if stream is not None and stream_rate == rate:
            return
        if stream is not None:
            stream.stop()
            stream.close()

        stream_rate = rate

        def callback(outdata, frames, time_info, status):
            if status:
                pass
            # frames 만큼 채워야 함
            needed = frames
            out = np.zeros((frames,), dtype=np.float32)
            filled = 0

            while filled < needed:
                try:
                    chunk = audio_q.get_nowait()
                except queue.Empty:
                    break

                take = min(len(chunk), needed - filled)
                out[filled:filled + take] = chunk[:take]
                filled += take

                # chunk가 남으면 다시 큐 앞에 넣기(간단 처리)
                if take < len(chunk):
                    rest = chunk[take:]
                    try:
                        audio_q.put_nowait(rest)
                    except queue.Full:
                        pass

            outdata[:] = out.reshape(-1, 1)

        stream = sd.OutputStream(
            channels=1,
            samplerate=rate,
            dtype="float32",
            callback=callback,
            blocksize=0,  # 자동
        )
        stream.start()

    def on_message(client, userdata, msg):
        nonlocal stream
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            rate = int(payload["rate"])
            data_b64 = payload["data_b64"]
            pcm = base64.b64decode(data_b64)
            audio = pcm16_bytes_to_float32(pcm)

            # 스트림 시작(또는 rate 변경 시 재시작)
            start_stream(rate)

            # 큐에 넣기
            try:
                audio_q.put_nowait(audio)
            except queue.Full:
                # 밀리면 오래된 거 버리고 최신 유지
                try:
                    _ = audio_q.get_nowait()
                    audio_q.put_nowait(audio)
                except queue.Empty:
                    pass
        except Exception:
            # 깨진 패킷/포맷 오류는 무시
            return

    client = mqtt.Client(client_id=args.client_id, clean_session=True)
    client.on_message = on_message
    client.connect(args.host, args.port, keepalive=30)
    client.subscribe(args.topic, qos=args.qos)

    print(f"[SUB] host={args.host}:{args.port} topic={args.topic}")
    print("[SUB] 수신 대기 (Ctrl+C 종료)")

    # 초기 지터 버퍼(수신 조금 모은 뒤 재생 시작하면 끊김이 덜함)
    t0 = time.time()
    while (time.time() - t0) * 1000 < args.jitter_ms:
        client.loop(timeout=0.05)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[SUB] 종료")
    finally:
        try:
            if stream is not None:
                stream.stop()
                stream.close()
        except Exception:
            pass
        client.disconnect()


if __name__ == "__main__":
    main()
