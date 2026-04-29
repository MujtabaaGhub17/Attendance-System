import cv2
import os

user_id = input("Enter Roll Number: ")
name = input("Enter Name: ")

with open("labels.txt", "a") as f:
    f.write(f"{user_id},{name}\n")

os.makedirs(f"dataset/{user_id}", exist_ok=True)

count = 0

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Program started")

cap = cv2.VideoCapture(0)  # keep 1 if external cam works

cap.set(3, 640)
cap.set(4, 480)

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

        count += 1
        file_path = f"dataset/{user_id}/{count}.jpg"
        cv2.imwrite(file_path, face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Img {count}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == 27 or count >= 40:
        break

cap.release()
cv2.destroyAllWindows()