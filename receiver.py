import socket
import struct
import numpy as np
import cv2

HOST = "localhost"
PORT = 5000

def recv_exact(sock, size):
    """Read exactly `size` bytes from the socket, or return None if connection closes."""
    buf = b""
    while len(buf) < size:
        chunk = sock.recv(size - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to {HOST}:{PORT} ...")
    sock.connect((HOST, PORT))
    print("Connected. Waiting for frames...")

    try:
        while True:
            header = recv_exact(sock, 4)
            if header is None:
                print("Connection closed by device.")
                break

            frame_len = struct.unpack(">I", header)[0]

            jpeg_bytes = recv_exact(sock, frame_len)
            if jpeg_bytes is None:
                print("Connection closed while reading frame.")
                break

            frame_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            cv2.imshow("MobCam", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        sock.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()