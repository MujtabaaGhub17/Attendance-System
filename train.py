import cv2
import os
import numpy as np

dataset_path = "dataset"
IMG_SIZE = 200

faces = []
labels = []

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("🔄 Training started...")

for label in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, label)

    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces_detected:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

            faces.append(face)
            labels.append(int(label))

print("Faces trained:", len(faces))

if len(faces) == 0:
    print("❌ No faces found. Check dataset.")
    exit()

labels = np.array(labels)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, labels)
model.save("model.yml")

print("✅ Training complete!")