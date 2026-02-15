# file: mqtt_audio_pub_opus.py
import time
import json
import base64
import queue
import argparse

import numpy as np
import sounddevice as sd
import paho.mqtt.client as mqtt
from opuslib import Encoder, APPLICATION_AUDIO


def float_to_int16_interleaved(audio_f32: np.ndarray) -> np.ndarray:
    # audio_f32 shape: (frames, channels), range [-1,1]
    audio_f32 = np.clip(audio_f32, -1.0, 1.0)
    audio_i16 = (audio_f32 * 32767.0).astype(np.int16)
    # already interleaved if 2D (sounddevice provides interleaved memory)
    return audio_i16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="218.146.225.166")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--topic", default="stream/live1")  # base topic
    ap.add_argument("--rate", type=int, default=48000, help="sample rate for Opus (8k~48k, 48k 추천)")
    ap.add_argument("--channels", type=int, default=1, choices=[1, 2])
    ap.add_argument("--frame-ms", type=int, default=20, choices=[10, 20, 40, 60])
    ap.add_argument("--bitrate", type=int, default=64000, help="bits per second (예: 64000, 96000, 128000)")
    ap.add_argument("--qos", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--client-id", default=f"opus-pub-{int(time.time())}")
    ap.add_argument("--latency", default="low", help="sounddevice latency hint ('low' 권장)")
    ap.add_argument("--username")
    ap.add_argument("--password")
    ap.add_argument("--tls", action="store_true")
    args = ap.parse_args()

    frame_size = int(args.rate * args.frame_ms / 1000)  # samples per channel
    topic_audio = f"{args.topic}/audio"
    topic_meta = f"{args.topic}/meta"
    topic_status = f"{args.topic}/status"

    # --- MQTT setup ---
    client = mqtt.Client(client_id=args.client_id, clean_session=True)
    if args.username:
        client.username_pw_set(args.username, args.password or None)
    if args.tls:
        client.tls_set()

    # LWT
    lwt = {"type": "lwt", "status": "offline", "ts": time.time(), "client_id": args.client_id}
    client.will_set(topic_status, json.dumps(lwt).encode("utf-8"), qos=1, retain=True)

    reconnecting = {"flag": False, "delay": 1.0}

    def on_connect(c, u, f, rc, p=None):
        print(f"[MQTT] connected rc={rc}")
        reconnecting["flag"] = False
        reconnecting["delay"] = 1.0
        online = {"type": "status", "status": "online", "ts": time.time(), "client_id": args.client_id}
        c.publish(topic_status, json.dumps(online), qos=1, retain=True)

        # 세션 메타(구독자 초기화용)
        meta = {
            "ver": 1,
            "codec": "opus",
            "rate": args.rate,
            "channels": args.channels,
            "frame_ms": args.frame_ms,
            "bitrate": args.bitrate,
            "start_ts": time.time(),
        }
        c.publish(topic_meta, json.dumps(meta), qos=1, retain=True)

    def on_disconnect(c, u, rc, p=None):
        print(f"[MQTT] disconnected rc={rc}")
        if rc != 0:
            reconnecting["flag"] = True

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect

    client.connect(args.host, args.port, keepalive=25)
    client.loop_start()

    # --- Opus encoder ---
    enc = Encoder(args.rate, args.channels, APPLICATION_AUDIO)
    enc.bitrate = args.bitrate  # bps
    # enc.vbr = True  # 필요 시 가변비트레이트

    # --- Audio capture queue ---
    max_queue_packets = max(3, int(np.ceil(0.2 * (1000 / args.frame_ms))))  # ~200ms 버퍼
    q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=max_queue_packets)

    def audio_callback(indata, frames, time_info, status):
        # indata: float32, shape (frames, channels)
        if status:
            # XRuns log만 필요 시 활성화
            # print("[AUDIO] status:", status)
            pass
        try:
            q.put_nowait(indata.copy())
        except queue.Full:
            # 오래된 프레임 drop -> 지연 누적 방지
            try:
                _ = q.get_nowait()
                q.put_nowait(indata.copy())
            except queue.Empty:
                pass

    seq = 0
    last_stat = time.time()
    sent = 0

    try:
        with sd.InputStream(
            channels=args.channels,
            samplerate=args.rate,
            blocksize=frame_size,  # 프레임과 동일하게 맞춤
            dtype="float32",
            latency=args.latency,
            callback=audio_callback,
        ):
            print(f"[PUB] Opus live: {args.rate}Hz, ch={args.channels}, {args.frame_ms}ms, {args.bitrate/1000:.0f}kbps")
            while True:
                # 재접속 백오프
                if reconnecting["flag"]:
                    delay = reconnecting["delay"]
                    print(f"[MQTT] reconnecting in {delay:.1f}s ...")
                    time.sleep(delay)
                    try:
                        client.reconnect()
                        reconnecting["delay"] = min(30.0, delay * 2.0)
                        continue
                    except Exception:
                        reconnecting["delay"] = min(30.0, delay * 2.0)
                        continue

                buf = q.get()  # float32 (frames, ch)
                i16 = float_to_int16_interleaved(buf)  # int16 ndarray
                packet = enc.encode(i16.tobytes(), frame_size)  # returns bytes (Opus frame)

                payload = {
                    "ver": 1,
                    "ts": time.time(),
                    "seq": seq,
                    "data_b64": base64.b64encode(packet).decode("ascii"),
                }
                client.publish(topic_audio, json.dumps(payload), qos=args.qos)
                seq += 1
                sent += 1

                now = time.time()
                if now - last_stat > 2.0:
                    pps = sent / (now - last_stat)
                    print(f"[STAT] tx={pps:.1f} pkt/s  (target ~{1000/args.frame_ms:.1f})")
                    sent = 0
                    last_stat = now

    except KeyboardInterrupt:
        print("\n[PUB] 종료")
    finally:
        client.loop_stop()
        try:
            client.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()