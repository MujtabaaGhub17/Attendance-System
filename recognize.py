import cv2
import time
import csv
from datetime import datetime
marked = set()

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

model = cv2.face.LBPHFaceRecognizer_create()
model.read("trainer.yml")

labels = {}

# ✅ Safe label loading
with open("labels.txt", "r") as f:
    for line in f:
        parts = line.strip().split(",", 1)

        if len(parts) != 2:
            print(f"Skipping bad label line: {line}")
            continue

        user_id, name = parts
        labels[int(user_id)] = name

def mark_attendance(name):
    with open("attendance.csv", "r") as f:
        existing = f.read()

    if name not in existing:
        now = datetime.now()
        time = now.strftime("%H:%M:%S")
        date = now.strftime("%Y-%m-%d")
        if "(" in name:
            name_only = name.split("(")[0]
            roll = name.split("(")[1].replace(")", "")
        else:
            name_only = name
            roll = ""
        with open("attendance.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name_only, roll, date, time])

# ✅ External camera (NO DSHOW)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera failed to open")
    exit()

# ⏳ Small warm-up (important for external cam)
time.sleep(2)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame not captured, retrying...")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.2, 4)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        user_id, confidence = model.predict(face)

        print("Predicted ID:", user_id, "Confidence:", confidence)

        if confidence < 100:
            name = labels.get(user_id, "Unknown")
            text = name
            color = (0, 255, 0)
            mark_attendance(name) 
        else:
            text = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
