import time
import json
import base64
import queue
import argparse

import numpy as np
import sounddevice as sd
import paho.mqtt.client as mqtt


def pcm16_bytes_from_float32(audio_f32: np.ndarray) -> bytes:
    """sounddevice는 float32 [-1,1]을 주는 경우가 많아 PCM16으로 변환"""
    audio_f32 = np.clip(audio_f32, -1.0, 1.0)
    audio_i16 = (audio_f32 * 32767.0).astype(np.int16)
    return audio_i16.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="218.146.225.166")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--topic", default="audio/pcm16/mono")
    ap.add_argument("--rate", type=int, default=16000, help="sample rate")
    ap.add_argument("--block", type=int, default=1024, help="frames per chunk")
    ap.add_argument("--qos", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--client-id", default=f"audio-pub-{int(time.time())}")
    args = ap.parse_args()

    q: "queue.Queue[bytes]" = queue.Queue(maxsize=50)

    def audio_callback(indata, frames, time_info, status):
        if status:
            # 드롭/언더런 등 상태가 뜰 수 있음
            pass
        # indata: shape (frames, channels), float32
        mono = indata[:, 0].copy()
        pcm = pcm16_bytes_from_float32(mono)
        try:
            q.put_nowait(pcm)
        except queue.Full:
            # 전송이 밀리면 오래된 프레임 버리고 최신 유지
            try:
                _ = q.get_nowait()
                q.put_nowait(pcm)
            except queue.Empty:
                pass

    client = mqtt.Client(client_id=args.client_id, clean_session=True)
    client.connect(args.host, args.port, keepalive=30)
    client.loop_start()

    seq = 0
    print(f"[PUB] host={args.host}:{args.port} topic={args.topic}")
    print("[PUB] 마이크 캡처 시작 (Ctrl+C 종료)")

    try:
        with sd.InputStream(
            channels=1,
            samplerate=args.rate,
            blocksize=args.block,
            dtype="float32",
            callback=audio_callback,
        ):
            while True:
                pcm = q.get()  # bytes
                payload = {
                    "ver": 1,
                    "ts": time.time(),
                    "seq": seq,
                    "rate": args.rate,
                    "channels": 1,
                    "sample_width": 2,  # bytes (16-bit)
                    "codec": "pcm_s16le",
                    "data_b64": base64.b64encode(pcm).decode("ascii"),
                }
                client.publish(args.topic, json.dumps(payload), qos=args.qos)
                seq += 1

    except KeyboardInterrupt:
        print("\n[PUB] 종료")

    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
