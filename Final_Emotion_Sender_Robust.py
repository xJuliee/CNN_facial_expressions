import socket
import struct
import cv2
import numpy as np
from keras.models import load_model
from collections import deque
import time

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
REPEAT_THRESHOLD = 2  # Number of repeats needed to resend same emotion

def preprocess_face(gray_face):
    face_resized = cv2.resize(gray_face, IMG_SIZE)
    face_equalized = cv2.equalizeHist(face_resized)
    face_normalized = face_equalized / 255.0
    face_normalized = np.expand_dims(face_normalized, axis=-1)
    face_input = np.expand_dims(face_normalized, axis=0)
    return face_input

# === Emotion Sender Setup ===
EMOTION_HOST = '192.168.0.102'  # Use Pepper's IP
EMOTION_PORT = 6000

class EmotionSender:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = None
        self.connect()

    def connect(self):
        while True:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.host, self.port))
                print("[EmotionSender] Connected to emotion receiver.")
                break
            except Exception as e:
                print("[EmotionSender] Connection failed, retrying in 1s...")
                time.sleep(1)

    def send(self, label):
        try:
            self.socket.sendall(label.encode('utf-8'))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
            print("[EmotionSender] Connection lost. Reconnecting...")
            self.connect()
            try:
                self.socket.sendall(label.encode('utf-8'))
            except Exception as e:
                print("[EmotionSender] Failed to send after reconnect:", e)

    def close(self):
        if self.socket:
            try:
                self.socket.close()
            except:
                pass

# === Video Receiver Server Setup ===
HOST = ''
PORT = 5000

def start_server():
    while True:
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((HOST, PORT))
            server_socket.listen(1)
            print("\n[Classifier] Waiting for Pepper to connect on port", PORT)
            conn, addr = server_socket.accept()
            print("[Classifier] Connected by", addr)

            emotion_sender = EmotionSender(EMOTION_HOST, EMOTION_PORT)
            handle_connection(conn, emotion_sender)

        except Exception as e:
            print("[Classifier] Server error:", e)
            time.sleep(2)
        finally:
            try:
                server_socket.close()
            except:
                pass

def handle_connection(conn, emotion_sender):
    prediction_buffer = deque(maxlen=REPEAT_THRESHOLD)
    last_sent_emotion = None

    try:
        while True:
            # === Receive 4-byte JPEG size header ===
            raw_len = b''
            while len(raw_len) < 4:
                more = conn.recv(4 - len(raw_len))
                if not more:
                    raise Exception("Connection closed by Pepper")
                raw_len += more
            frame_len = struct.unpack('>I', raw_len)[0]

            # === Receive JPEG image bytes ===
            frame_data = b''
            while len(frame_data) < frame_len:
                more = conn.recv(frame_len - len(frame_data))
                if not more:
                    raise Exception("Connection closed during frame")
                frame_data += more

            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # === Emotion Detection ===
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_gray = gray[y:y+h, x:x+w]
                face_input = preprocess_face(face_gray)

                predictions = model.predict(face_input, verbose=0)[0]
                predictions[0] *= 1  # angry
                predictions[1] *= 8    # confused
                predictions[2] *= 2    # disgust
                predictions[3] *= 0.7  # fear
                predictions[6] *= 0.5  # sad
                predictions /= np.sum(predictions)

                class_id = np.argmax(predictions)
                emotion_label = class_mapping[class_id]

                # Emotion logic
                prediction_buffer.append(emotion_label)

                if emotion_label != last_sent_emotion:
                    emotion_sender.send(emotion_label)
                    last_sent_emotion = emotion_label
                    prediction_buffer.clear()
                elif prediction_buffer.count(emotion_label) >= REPEAT_THRESHOLD:
                    emotion_sender.send(emotion_label)
                    prediction_buffer.clear()

                print("\nEmotion probabilities:")
                for idx, prob in enumerate(predictions):
                    print(f"{class_mapping[idx]}: {prob * 100:.2f}%")

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{emotion_label} ({predictions[class_id]*100:.1f}%)",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

                top_indices = np.argsort(predictions)[::-1][:3]
                for i, idx in enumerate(top_indices):
                    label = f"{class_mapping[idx]}: {predictions[idx]*100:.1f}%"
                    cv2.putText(frame, label, (x, y + h + 20 + i*25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            cv2.imshow("Pepper Emotion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt()

    except KeyboardInterrupt:
        print("[Classifier] Interrupted by user.")
        conn.close()
        cv2.destroyAllWindows()
        exit()

    except Exception as e:
        print("[Classifier] Connection lost:", e)

    finally:
        try:
            conn.close()
        except:
            pass
        emotion_sender.close()

if __name__ == "__main__":
    start_server()