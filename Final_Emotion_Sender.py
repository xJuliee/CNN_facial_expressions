import socket
import struct
import cv2
import numpy as np
from keras.models import load_model
from collections import deque, Counter

# === CNN Setup ===
model = load_model("CNNs/Final_CNN.keras")
class_mapping = {
    0: "angry",
    1: "confused",
    2: "disgust",
    3: "fear",
    4: "happy",
    5: "neutral",
    6: "sad",
    7: "surprise"
}
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
IMG_SIZE = (75, 75)
prediction_buffer = deque(maxlen=10)

def preprocess_face(gray_face):
    face_resized = cv2.resize(gray_face, IMG_SIZE)
    face_equalized = cv2.equalizeHist(face_resized)
    face_normalized = face_equalized / 255.0
    face_normalized = np.expand_dims(face_normalized, axis=-1)  # (75,75,1)
    face_input = np.expand_dims(face_normalized, axis=0)        # (1,75,75,1)
    return face_input

# === Emotion Sender Setup ===
EMOTION_HOST = 'localhost'  # IP address of receiver project
EMOTION_PORT = 6000         # Port receiver project is listening on

def send_emotion_label(label):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as emotion_socket:
            emotion_socket.connect((EMOTION_HOST, EMOTION_PORT))
            emotion_socket.sendall(label.encode('utf-8'))
    except ConnectionRefusedError:
        print("[!] Emotion receiver is not running or not reachable.")
    except Exception as e:
        print(f"[!] Error sending emotion label: {e}")

# === Socket Setup for receiving video from Pepper ===
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
        # Receive 4-byte header indicating JPEG size
        raw_len = b''
        while len(raw_len) < 4:
            more = conn.recv(4 - len(raw_len))
            if not more:
                raise Exception("Connection closed")
            raw_len += more
        frame_len = struct.unpack('>I', raw_len)[0]

        # Receive JPEG data
        frame_data = b''
        while len(frame_data) < frame_len:
            more = conn.recv(frame_len - len(frame_data))
            if not more:
                raise Exception("Connection closed")
            frame_data += more

        # Decode the frame
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            face_input = preprocess_face(face_gray)

            predictions = model.predict(face_input, verbose=0)[0]
            predictions[0] *= 0.1  # reduce angry
            predictions[1] *= 5    # boost confused
            predictions[2] *= 2    # boost disgust
            predictions[6] *= 0.5  # reduce sad
            predictions /= np.sum(predictions)

            class_id = np.argmax(predictions)
            prediction_buffer.append(class_id)
            most_common_id = Counter(prediction_buffer).most_common(1)[0][0]
            emotion_label = class_mapping[most_common_id]

            # === Send emotion label to other project ===
            send_emotion_label(emotion_label)

            print("\nEmotion probabilities:")
            for idx, prob in enumerate(predictions):
                print(f"{class_mapping[idx]}: {prob * 100:.2f}%")

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion_label} ({predictions[most_common_id]*100:.1f}%)",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            top_indices = np.argsort(predictions)[::-1][:3]
            for i, idx in enumerate(top_indices):
                label = f"{class_mapping[idx]}: {predictions[idx]*100:.1f}%"
                cv2.putText(frame, label, (x, y + h + 20 + i*25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.imshow("Pepper Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()