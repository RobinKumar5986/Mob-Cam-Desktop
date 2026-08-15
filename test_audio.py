"""Audio path test. No phone required.

Pretends to be the Android AudioStreamer: opens port 4344, sends the format
handshake, then streams a 440 Hz tone in 20 ms chunks. Run the receiver GUI (or
just this script's built-in client) and you should hear a steady A note on the
virtual microphone.

    python3 test_audio.py            server + local client, plays the tone
    python3 test_audio.py --server   server only, connect the real GUI to it
"""

import json
import math
import socket
import struct
import sys
import threading
import time

import protocol
from audio_output import AudioOutput, AudioOutputError, guess_loopback_device
from audio_receiver import AudioFormat, AudioReceiver

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_FRAMES = SAMPLE_RATE // 50
TONE_HZ = 440.0
DURATION = 6.0


def tone_chunk(phase):
    """One 20 ms chunk of a sine wave, plus the phase to continue from."""
    samples = []
    step = 2 * math.pi * TONE_HZ / SAMPLE_RATE
    for _ in range(CHUNK_FRAMES):
        samples.append(int(math.sin(phase) * 12000))
        phase += step
    return struct.pack(f"<{CHUNK_FRAMES}h", *samples), phase % (2 * math.pi)


def fake_phone(stop_event):
    """Serve one client on the audio port the way the phone does."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", protocol.AUDIO_PORT))
    server.listen(1)
    server.settimeout(1.0)
    print(f"[phone] listening on {protocol.AUDIO_PORT}")

    while not stop_event.is_set():
        try:
            client, address = server.accept()
        except socket.timeout:
            continue
        print(f"[phone] client {address}")
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        hello = {
            "protocolVersion": 1, "sampleRate": SAMPLE_RATE, "channels": CHANNELS,
            "encoding": "pcm_s16le", "chunkBytes": CHUNK_FRAMES * 2,
            "formatName": "VOICE_16K",
        }
        client.sendall(protocol.encode_message(
            protocol.TYPE_HELLO, json.dumps(hello).encode()))

        phase = 0.0
        sent = 0
        started = time.perf_counter()
        try:
            while not stop_event.is_set() and time.perf_counter() - started < DURATION:
                payload, phase = tone_chunk(phase)
                client.sendall(protocol.encode_message(protocol.TYPE_AUDIO, payload))
                sent += 1
                # Pace it like a real microphone rather than blasting the socket.
                target = started + sent * 0.02
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            client.sendall(protocol.encode_message(protocol.TYPE_BYE))
        except OSError as exc:
            print(f"[phone] client gone: {exc}")
        finally:
            client.close()
            print(f"[phone] sent {sent} chunks")
        break

    server.close()


def main():
    server_only = "--server" in sys.argv
    stop_event = threading.Event()

    phone = threading.Thread(target=fake_phone, args=(stop_event,), daemon=True)
    phone.start()
    time.sleep(0.3)

    if server_only:
        print("Fake phone running. Connect the receiver GUI now.")
        try:
            phone.join()
        except KeyboardInterrupt:
            stop_event.set()
        return

    device = guess_loopback_device()
    print(f"[pc] output device: {device or 'system default (monitor only)'}")
    output = AudioOutput(device=device, monitor=not device,
                         on_status=lambda m: print(f"[pc] {m}"),
                         on_error=lambda m: print(f"[pc] ERROR {m}"))

    def on_format(fmt: AudioFormat):
        print(f"[pc] format {fmt}")
        try:
            output.open(fmt)
        except AudioOutputError as exc:
            print(f"[pc] could not open output: {exc}")

    receiver = AudioReceiver(
        on_format=on_format,
        on_chunk=output.feed,
        on_connected=lambda: print("[pc] connected"),
        on_disconnected=lambda: print("[pc] disconnected"),
    )
    receiver.start()

    try:
        time.sleep(DURATION + 1.5)
    except KeyboardInterrupt:
        pass

    stop_event.set()
    receiver.stop()
    output.close()

    print(f"[pc] chunks {receiver.chunks_received}, "
          f"bytes {receiver.bytes_received}, underruns {output.underruns}")
    expected = int(DURATION * 50)
    if receiver.chunks_received < expected * 0.8:
        print(f"FAIL: expected about {expected} chunks")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
