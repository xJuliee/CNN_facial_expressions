import socket

PEPPER_IP = '192.168.0.101'
PORT = 6001

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    s.connect((PEPPER_IP, PORT))
    print("Connection successful!")
    s.close()
except Exception as e:
    print(f"Connection failed: {e}")
