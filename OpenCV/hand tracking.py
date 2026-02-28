import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui

face_cascade = cv2.CascadeClassifier('haar_face.xml')
cap = cv2.VideoCapture(0)
model_path = "hand_landmarker.task"
screen_w, screen_h = pyautogui.size()

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

peace = cv2.imread(r"C:\Users\dommi\PycharmProjects\Laptop Control\OpenCV\peace.jpg")

options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path = model_path),
    running_mode = VisionRunningMode.VIDEO,
    num_hands = 2
)

landmarker = HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    result = landmarker.detect_for_video(mp_image, timestamp)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            h, w, _ = frame.shape

            index_up = hand_landmarks[8].y < hand_landmarks[5].y
            middle_up = hand_landmarks[12].y < hand_landmarks[9].y
            ring_down = hand_landmarks[16].y > hand_landmarks[13].y
            pinky_down = hand_landmarks[20].y > hand_landmarks[17].y

            if index_up and ring_down and pinky_down:
                mouse_x = int(hand_landmarks[8].x * screen_w)
                mouse_y = int(hand_landmarks[8].y * screen_h)
                pyautogui.moveTo(mouse_x, mouse_y)

            for landmark in hand_landmarks:
                cx = int(landmark.x * w)
                cy = int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            for start_idx, end_idx in HAND_CONNECTIONS:
                x1 = int(hand_landmarks[start_idx].x * w)
                y1 = int(hand_landmarks[start_idx].y * h)
                x2 = int(hand_landmarks[end_idx].x * w)
                y2 = int(hand_landmarks[end_idx].y * h)

                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.waitKey(1)

    cv2.imshow("Hand Tracking", frame)

    key = cv2.waitKey(1) & 0xFF

    if key==27:
        break

print(index_up, middle_up, ring_down, pinky_down)
cap.release()
cv2.destroyAllWindows()