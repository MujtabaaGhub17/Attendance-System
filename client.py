import socket
import cv2
import pickle
import struct

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("127.0.0.1", 9999))   # change IP if needed

# ✅ External camera (change index if needed)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)     

if not cap.isOpened():
    print("❌ Camera not found. Try index 0,1,2")
    exit()

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    _, buffer = cv2.imencode('.jpg', frame)
    data = pickle.dumps(buffer)

    message = struct.pack("Q", len(data)) + data

    try:
        client.sendall(message)
    except:
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
client.close()