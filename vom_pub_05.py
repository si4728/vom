import time
import json
import queue
import argparse
from typing import Tuple, Optional

import numpy as np
import sounddevice as sd
import paho.mqtt.client as mqtt
import audioop


def main():
    ap = argparse.ArgumentParser(description="8kHz + μ-law(G.711) live publisher over MQTT (binary payload)")
    ap.add_argument("--host", default="218.146.225.166")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--topic", default="stream/mulaw8k1", help="base topic (meta,audio,status 하위 토픽 사용)")

    # 캡처(입력) 설정
    ap.add_argument("--in-rate", type=int, default=16000, help="input capture sample rate (Hz). 보통 16000 권장")
    ap.add_argument("--channels", type=int, default=1, choices=[1, 2])
    ap.add_argument("--block-in", type=int, default=320, help="input frames per callback (예: 16000Hz에서 20ms=320)")
    ap.add_argument("--latency", default="low", help="sounddevice latency hint ('low' 권장)")

    # 출력(전송) 설정: 8kHz + μ-law
    ap.add_argument("--out-rate", type=int, default=8000, help="output sample rate (Hz). 기본 8000")
    ap.add_argument("--qos-audio", type=int, default=0, choices=[0, 1, 2], help="audio QoS (보통 0, 불안정하면 1)")
    ap.add_argument("--queue-ms", type=int, default=200, help="오디오 큐 목표 지연(ms)")

    ap.add_argument("--client-id", default=f"mulaw8k-pub-{int(time.time())}")
    ap.add_argument("--username")
    ap.add_argument("--password")
    ap.add_argument("--tls", action="store_true")
    args = ap.parse_args()

    topic_audio = f"{args.topic}/audio"   # 바이너리 μ-law(8bit) payload
    topic_meta = f"{args.topic}/meta"     # JSON retain
    topic_status = f"{args.topic}/status" # JSON retain

    # 큐 용량 (입력 블록 주기 기준)
    in_packets_per_sec = args.in_rate / args.block_in
    max_packets = max(3, int(np.ceil((args.queue_ms / 1000.0) * in_packets_per_sec)))
    q: "queue.Queue[Tuple[bytes, float]]" = queue.Queue(maxsize=max_packets)

    # MQTT 설정 (Callback API v2로 경고 제거)
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=args.client_id, clean_session=True)

    if args.username:
        client.username_pw_set(args.username, args.password or None)
    if args.tls:
        client.tls_set()

    # LWT (오프라인 표시)
    lwt = {"type": "lwt", "status": "offline", "ts": time.time(), "client_id": args.client_id}
    client.will_set(topic_status, json.dumps(lwt), qos=1, retain=True)

    reconnecting = {"flag": False, "delay": 1.0}

    def on_connect(c, u, flags, reason_code, properties=None):
        print(f"[MQTT] connected rc={reason_code}")
        reconnecting["flag"] = False
        reconnecting["delay"] = 1.0

        c.publish(topic_status, json.dumps({"type": "status", "status": "online", "ts": time.time()}), qos=1, retain=True)

        meta = {
            "ver": 2,
            "codec": "g711_mulaw",
            "in_rate": args.in_rate,
            "out_rate": args.out_rate,
            "channels": args.channels,
            "block_in": args.block_in,
            "sample_width_in": 2,  # int16
            "sample_width_out": 1, # ulaw 8bit
            "payload": "binary_ulaw_bytes",
            "start_ts": time.time(),
        }
        c.publish(topic_meta, json.dumps(meta), qos=1, retain=True)

    def on_disconnect(c, u, reason_code, properties=None):
        print(f"[MQTT] disconnected rc={reason_code}")
        if reason_code != 0:
            reconnecting["flag"] = True

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect

    client.connect(args.host, args.port, keepalive=25)
    client.loop_start()

    # ratecv 상태(연속 스트림 품질을 위해 state 유지)
    ratecv_state: Optional[tuple] = None

    # sounddevice 콜백: RawInputStream(int16) -> bytes
    bytes_per_frame_in = args.channels * 2  # int16
    expected_bytes = args.block_in * bytes_per_frame_in

    def audio_callback(indata, frames, time_info, status):
        if status:
            pass

        produced_ts = time.time()
        pcm16 = bytes(indata)  # interleaved int16 bytes

        if len(pcm16) != expected_bytes:
            return

        try:
            q.put_nowait((pcm16, produced_ts))
        except queue.Full:
            # 지연 누적 방지: 오래된 프레임 drop
            try:
                _ = q.get_nowait()
                q.put_nowait((pcm16, produced_ts))
            except queue.Empty:
                pass

    seq = 0
    last_stat = time.time()
    tx_bytes = 0
    tx_pkts = 0

    print(f"[PUB] host={args.host}:{args.port} topic={args.topic}")
    print(f"[PUB] capture: {args.in_rate}Hz, ch={args.channels}, block_in={args.block_in} frames")
    print(f"[PUB] send: {args.out_rate}Hz + μ-law(8bit), qos={args.qos_audio}, queue={max_packets} (~{args.queue_ms}ms)")

    try:
        with sd.RawInputStream(
            channels=args.channels,
            samplerate=args.in_rate,
            blocksize=args.block_in,
            dtype="int16",
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

                pcm16, produced_ts = q.get()
                now = time.time()
                q_delay_ms = (now - produced_ts) * 1000.0

                # 1) 16kHz PCM16 -> 8kHz PCM16 (ratecv)
                # audioop.ratecv는 (converted_bytes, new_state) 반환
                # width=2 (int16), nchannels=args.channels
                converted, ratecv_state = audioop.ratecv(
                    pcm16, 2, args.channels, args.in_rate, args.out_rate, ratecv_state
                )

                # 2) 8kHz PCM16 -> μ-law 8bit
                ulaw = audioop.lin2ulaw(converted, 2)

                # 3) MQTT publish (바이너리 payload)
                client.publish(topic_audio, ulaw, qos=args.qos_audio, retain=False)

                seq += 1
                tx_pkts += 1
                tx_bytes += len(ulaw)

                if now - last_stat > 2.0:
                    kbps = (tx_bytes * 8) / (now - last_stat) / 1000.0
                    print(f"[STAT] pkts={tx_pkts}  ulaw_bytes={tx_bytes}  ~{kbps:.1f} kbps  q_delay≈{q_delay_ms:.0f} ms")
                    last_stat = now
                    tx_pkts = 0
                    tx_bytes = 0

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
