import socket
import cv2
import pickle
import struct
from datetime import datetime
import threading

IMG_SIZE = 200

model = cv2.face.LBPHFaceRecognizer_create()
model.read("model.yml")


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

names = {
    0: ("Mujtabaa Hussain", "100523733071"),
    1: ("Friend", "000000")
}


lock = threading.Lock()

# Attendance function
def mark_attendance(name, roll):
    with open("attendance.csv", "a+", newline="") as f:
        f.seek(0)
        data = f.readlines()

        recorded_rolls = [line.split(",")[1] for line in data if len(line.split(",")) > 1]

        if roll not in recorded_rolls:
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

            f.write(f"{name},{roll},{timestamp}\n")
            print(f"✅ Attendance marked: {name} ({roll})")


# 🔥 Handle each client
def handle_client(client, addr):
    print(f"✅ Connected: {addr}")

    data = b""
    payload_size = struct.calcsize("Q")

    while True:
        try:
            while len(data) < payload_size:
                packet = client.recv(4096)
                if not packet:
                    return
                data += packet

            packed_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_size)[0]

            while len(data) < msg_size:
                data += client.recv(4096)

            frame_data = data[:msg_size]
            data = data[msg_size:]

            frame = pickle.loads(frame_data)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

                label, confidence = model.predict(face)

                if confidence < 90:
                    name, roll = names.get(label, ("Unknown", "0"))

                    # 🔒 Thread-safe write
                    with lock:
                        mark_attendance(name, roll)

                    display_text = f"{name} ({roll})"
                    color = (0, 255, 0)
                else:
                    display_text = "Unknown"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, display_text, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow(f"Client {addr}", frame)

            if cv2.waitKey(1) == 27:
                break

        except Exception as e:
            print(f"❌ Error with {addr}: {e}")
            break

    client.close()
    print(f"❌ Disconnected: {addr}")


# 🔥 Server setup
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("0.0.0.0", 9999))
server.listen(5)

print("🚀 Server started. Waiting for clients...")

# 🔥 MULTI-CLIENT LOOP
while True:
    client, addr = server.accept()

    client_thread = threading.Thread(
        target=handle_client,
        args=(client, addr)
    )
    client_thread.start()