import time
import json
import base64
import queue
import argparse
from typing import Tuple

import numpy as np
import sounddevice as sd
import paho.mqtt.client as mqtt


def pcm16_bytes_from_float32(audio_f32: np.ndarray) -> bytes:
    """sounddevice float32[-1,1] -> PCM16 LE bytes"""
    audio_f32 = np.clip(audio_f32, -1.0, 1.0)
    audio_i16 = (audio_f32 * 32767.0).astype(np.int16)
    return audio_i16.tobytes()


class RateAverager:
    def __init__(self, win_sec: float = 5.0):
        self.win_sec = win_sec
        self.ts = []

    def tick(self, t: float):
        self.ts.append(t)
        cutoff = t - self.win_sec
        while self.ts and self.ts[0] < cutoff:
            self.ts.pop(0)

    def rate(self) -> float:
        if len(self.ts) < 2:
            return 0.0
        dur = self.ts[-1] - self.ts[0]
        return (len(self.ts) - 1) / dur if dur > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description="PCM16 live publisher over MQTT")
    ap.add_argument("--host", default="218.146.225.166")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--topic", default="stream/pcm1", help="base topic (meta,audio,status 하위 토픽 사용)")
    ap.add_argument("--rate", type=int, default=16000, help="sample rate (Hz)")
    ap.add_argument("--channels", type=int, default=1, choices=[1, 2])
    ap.add_argument("--block", type=int, default=512, help="frames per packet (예: 256/512/1024)")
    ap.add_argument("--qos", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--client-id", default=f"pcm-pub-{int(time.time())}")
    ap.add_argument("--latency", default="low", help="sounddevice latency hint ('low' 권장)")
    ap.add_argument("--queue-ms", type=int, default=200, help="오디오 큐 목표 지연(ms)")
    ap.add_argument("--username")
    ap.add_argument("--password")
    ap.add_argument("--tls", action="store_true")
    args = ap.parse_args()

    topic_audio = f"{args.topic}/audio"
    topic_meta = f"{args.topic}/meta"
    topic_status = f"{args.topic}/status"

    # Queue capacity from time budget
    packets_per_sec = args.rate / args.block
    max_packets = max(3, int(np.ceil((args.queue_ms / 1000.0) * packets_per_sec)))
    q: "queue.Queue[Tuple[bytes, float]]" = queue.Queue(maxsize=max_packets)

    # MQTT client setup
    client = mqtt.Client(client_id=args.client_id, clean_session=True)
    if args.username:
        client.username_pw_set(args.username, args.password or None)
    if args.tls:
        client.tls_set()

    # LWT for presence
    lwt = {"type": "lwt", "status": "offline", "ts": time.time(), "client_id": args.client_id}
    client.will_set(topic_status, json.dumps(lwt), qos=1, retain=True)

    reconnecting = {"flag": False, "delay": 1.0}

    def on_connect(c, u, f, rc, p=None):
        print(f"[MQTT] connected rc={rc}")
        reconnecting["flag"] = False
        reconnecting["delay"] = 1.0

        # online + meta (retain)
        c.publish(topic_status, json.dumps({"type": "status", "status": "online", "ts": time.time()}), qos=1, retain=True)
        meta = {
            "ver": 1,
            "codec": "pcm_s16le",
            "rate": args.rate,
            "channels": args.channels,
            "block": args.block,
            "sample_width": 2,
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

    def audio_callback(indata, frames, time_info, status):
        if status:
            # XRuns 등 상태 표시 필요시 로그
            # print("[AUDIO]", status)
            pass
        pcm = pcm16_bytes_from_float32(indata.copy())  # bytes (int16 interleaved)
        produced_ts = time.time()
        try:
            q.put_nowait((pcm, produced_ts))
        except queue.Full:
            # 오래된 프레임 drop -> 지연 누적 방지
            try:
                _ = q.get_nowait()
                q.put_nowait((pcm, produced_ts))
            except queue.Empty:
                pass

    seq = 0
    rate_avg = RateAverager(5.0)
    last_stat = time.time()

    print(f"[PUB] host={args.host}:{args.port} topic={args.topic}")
    print(f"[PUB] rate={args.rate}Hz ch={args.channels} block={args.block} qos={args.qos} queue={max_packets} (≈{args.queue_ms}ms)")
    try:
        with sd.InputStream(
            channels=args.channels,
            samplerate=args.rate,
            blocksize=args.block,
            dtype="float32",
            latency=args.latency,
            callback=audio_callback,
        ):
            print("[PUB] 마이크 캡처 시작 (Ctrl+C 종료)")
            while True:
                # reconnect backoff
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

                pcm, produced_ts = q.get()
                now = time.time()
                q_delay_ms = (now - produced_ts) * 1000.0

                payload = {
                    "ver": 1,
                    "ts": now,
                    "seq": seq,
                    "codec": "pcm_s16le",
                    "rate": args.rate,
                    "channels": args.channels,
                    "block": args.block,
                    "sample_width": 2,
                    "data_b64": base64.b64encode(pcm).decode("ascii"),
                    "q_delay_ms": round(q_delay_ms, 1),
                }
                data = json.dumps(payload)
                client.publish(topic_audio, data, qos=args.qos)
                seq += 1

                rate_avg.tick(now)
                if now - last_stat > 2.0:
                    print(f"[STAT] tx_rate={rate_avg.rate():.2f} pkt/s  q_delay≈{q_delay_ms:.0f} ms  payload≈{len(data)} B")
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