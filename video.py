import socket
import struct
import cv2
import numpy as np

HOST = ''  # Listen on all interfaces
PORT = 5000

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print("Waiting for Pepper to connect...")

conn, addr = server_socket.accept()
print("Connected by", addr)

try:
    while True:
        # Read message length (4 bytes)
        raw_len = b''
        while len(raw_len) < 4:
            more = conn.recv(4 - len(raw_len))
            if not more:
                raise Exception("Connection closed")
            raw_len += more
        frame_len = struct.unpack('>I', raw_len)[0]

        # Read frame data
        frame_data = b''
        while len(frame_data) < frame_len:
            more = conn.recv(frame_len - len(frame_data))
            if not more:
                raise Exception("Connection closed")
            frame_data += more

        # Convert bytes to numpy array
        nparr = np.frombuffer(frame_data, np.uint8)
        # Decode image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is not None:
            cv2.imshow("Pepper Camera Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()
