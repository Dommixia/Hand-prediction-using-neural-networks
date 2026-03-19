import cv2
import os
import time

labels = {
    ord('t'): "thumb",
    ord('i'): "index",
    ord('m'): "middle",
    ord('r'): "ring",
    ord('p'): "pinky"
}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

base_path = "dataset"
for label in labels.values():
    os.makedirs(os.path.join(base_path, label), exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

current_label = None
last_capture_time = 0
interval = 0.1  # seconds

print("Press keys:")
print("t=thumb, i=index, m=middle, r=ring, p=pinky")
print("ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    original_frame = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3,5)
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        blurred = cv2.GaussianBlur(face, (99, 99), 30)
        frame[y:y+h, x:x+w] = blurred

    text = f"Label: {current_label}" if current_label else "Label: None"
    cv2.putText(frame, text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

    if key in labels:
        current_label = labels[key]
        print(f"Switched to: {current_label}")

    if current_label:
        if time.time() - last_capture_time > interval:
            filename = os.path.join(
                base_path,
                current_label,
                f"{int(time.time())}.jpg"
            )
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")

            last_capture_time = time.time()

cap.release()
cv2.destroyAllWindows()