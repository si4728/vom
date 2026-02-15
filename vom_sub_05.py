import time
import json
import queue
import argparse
import audioop
from typing import Optional

import sounddevice as sd
import paho.mqtt.client as mqtt


def main():
    ap = argparse.ArgumentParser(description="8kHz + μ-law(G.711) subscriber over MQTT (binary payload)")
    ap.add_argument("--host", default="218.146.225.166")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--topic", default="stream/mulaw8k1", help="base topic (meta,audio,status 하위 토픽 사용)")
    ap.add_argument("--qos-audio", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--client-id", default=f"mulaw8k-sub-{int(time.time())}")
    ap.add_argument("--prebuffer-ms", type=int, default=300, help="초기 선버퍼(ms) 200~800 권장")
    ap.add_argument("--max-queue-ms", type=int, default=2000, help="최대 큐 지연(ms), 초과시 오래된 오디오 drop")
    ap.add_argument("--block", type=int, default=160, help="출력 block frames (8kHz에서 20ms=160)")
    ap.add_argument("--username")
    ap.add_argument("--password")
    ap.add_argument("--tls", action="store_true")
    args = ap.parse_args()

    topic_audio = f"{args.topic}/audio"
    topic_meta = f"{args.topic}/meta"
    topic_status = f"{args.topic}/status"

    # 기본값(메타 받으면 갱신)
    rate = 8000
    channels = 1

    # 출력 블록: frames
    block_frames = args.block

    # 큐는 "PCM16 bytes" 단위로 저장
    q: "queue.Queue[bytes]" = queue.Queue()

    # 큐 상한(대략 ms 기준) - 너무 쌓이면 지연이 늘어나므로 오래된 것 drop
    # PCM16: 2 bytes/sample * channels
    def queue_bytes_limit(r: int, ch: int) -> int:
        return int((args.max_queue_ms / 1000.0) * r * ch * 2)

    q_limit_bytes = queue_bytes_limit(rate, channels)
    q_bytes = 0
    q_lock = False  # 가볍게(단일 스레드 콜백 + 오디오 콜백이므로 엄밀히는 lock 권장, 하지만 단순화)

    # 오디오 출력 스트림은 meta에 따라 시작
    stream: Optional[sd.RawOutputStream] = None

    def ensure_stream(r: int, ch: int):
        nonlocal stream, q_limit_bytes, rate, channels
        if stream is not None and rate == r and channels == ch:
            return
        # 기존 스트림 정리
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
            stream = None

        rate = r
        channels = ch
        q_limit_bytes = queue_bytes_limit(rate, channels)

        def audio_out_callback(outdata, frames, time_info, status):
            nonlocal q_bytes
            need_bytes = frames * channels * 2
            out = bytearray()

            # 큐에서 필요한 만큼 채우기
            while len(out) < need_bytes:
                try:
                    chunk = q.get_nowait()
                except queue.Empty:
                    break
                out.extend(chunk)
                q_bytes -= len(chunk)

            # 부족하면 무음(0) 채움
            if len(out) < need_bytes:
                out.extend(b"\x00" * (need_bytes - len(out)))

            outdata[:] = bytes(out[:need_bytes])

        stream = sd.RawOutputStream(
            samplerate=rate,
            channels=channels,
            dtype="int16",
            blocksize=block_frames,
            callback=audio_out_callback,
        )
        stream.start()
        print(f"[AUDIO] started: {rate}Hz ch={channels} block={block_frames} frames")

    # MQTT
    cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=args.client_id, clean_session=True)
    if args.username:
        cli.username_pw_set(args.username, args.password or None)
    if args.tls:
        cli.tls_set()

    meta_ok = {"ok": False}

    def on_connect(c, u, flags, reason_code, properties=None):
        print(f"[MQTT] connected rc={reason_code}")
        c.subscribe(topic_meta, qos=1)
        c.subscribe(topic_status, qos=1)
        c.subscribe(topic_audio, qos=args.qos_audio)

    def on_message(c, u, msg):
        nonlocal q_bytes

        if msg.topic == topic_meta:
            try:
                meta = json.loads(msg.payload.decode("utf-8"))
            except Exception:
                return
            if meta.get("codec") == "g711_mulaw":
                r = int(meta.get("out_rate", 8000))
                ch = int(meta.get("channels", 1))
                ensure_stream(r, ch)
                meta_ok["ok"] = True
                print(f"[META] {meta}")
            return

        if msg.topic == topic_status:
            # 필요하면 상태 로그
            return

        if msg.topic == topic_audio:
            if stream is None:
                # meta를 아직 못 받았으면 기본값으로라도 시작
                ensure_stream(rate, channels)

            ulaw = msg.payload  # bytes (8bit)
            # μ-law -> PCM16
            try:
                pcm16 = audioop.ulaw2lin(ulaw, 2)  # width=2 (int16)
            except Exception:
                return

            # 큐에 쌓기 (너무 쌓이면 오래된 것 drop)
            q.put(pcm16)
            q_bytes += len(pcm16)

            while q_bytes > q_limit_bytes:
                try:
                    old = q.get_nowait()
                    q_bytes -= len(old)
                except queue.Empty:
                    q_bytes = 0
                    break

    cli.on_connect = on_connect
    cli.on_message = on_message

    cli.connect(args.host, args.port, keepalive=25)
    cli.loop_start()

    print(f"[SUB] host={args.host}:{args.port} topic={args.topic} qos={args.qos_audio}")
    print(f"[SUB] prebuffer={args.prebuffer_ms}ms (초기 버퍼 후 재생 안정화)")

    # 초기 선버퍼: 약간 모았다가 재생하면 끊김 줄어듦
    time.sleep(max(0.0, args.prebuffer_ms / 1000.0))
    ensure_stream(rate, channels)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        cli.loop_stop()
        try:
            cli.disconnect()
        except Exception:
            pass
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        print("[SUB] bye")


if __name__ == "__main__":
    main()
