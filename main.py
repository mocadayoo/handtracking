import cv2 as cv
import numpy as np
import time

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HAND_CONNECTS = [
    (0,1), (1,2), (2,3), (3,4),       # 親指
    (0,5), (5,6), (6,7), (7,8),       # 人差し指
    (9,10), (10,11), (11,12),         # 中指
    (13,14), (14,15), (15,16),        # 薬指
    (17,18), (18,19), (19,20),        # 小指
    (5,9), (9,13), (13,17), (0,17)    # 指の付け根
]

# mediapipe settings
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)
detector = vision.HandLandmarker.create_from_options(options)

# draw settings
HAND_CIRCLE_COLOR = (0, 255, 0)
HAND_CIRCLE_RADIUS = 5
HAND_CIRCLE_THICKNESS = -1
HAND_LINE_COLOR = (0, 0, 255)
HAND_LINE_TICKNESS = 2

# mpの0~1の座標をカメラの画質まで拡大、変換
def mp_convert_to_camera_wh(point, w, h):
    return (int(point.x * w), int(point.y * h))

def count_up_fingers(hand_landmarks, handedness):
    counter = 0
    # 親指を除いた指先
    tips = [8, 12, 16, 20]

    # 比較対象 第一関節
    mcp_base = [6, 10, 14, 18]

    # まず親指以外が立っているか判断
    for tip, base in zip(tips, mcp_base):
        if hand_landmarks[tip].y < hand_landmarks[base].y:
            counter += 1

    is_right = handedness.category_name == "Right"
    # 親指の付け根 < 小指の付け根で面裏を判断
    is_palm = (hand_landmarks[5].x < hand_landmarks[17].x) if is_right else (hand_landmarks[5].x > hand_landmarks[17].x)

    # 親指は 2と4の点のxをさっきのpalmなどを使って条件を反転させながら判断
    t4 = hand_landmarks[4]
    t2 = hand_landmarks[2]

    thumb_up = (t4.x < t2.x) if (is_right == is_palm) else (t4.x > t2.x)
    if thumb_up: counter += 1

    return counter

def draw_landmarks(img, landmarks):
    for line_point in HAND_CONNECTS:
        cv.line(img, landmarks[line_point[0]], landmarks[line_point[1]], HAND_LINE_COLOR, HAND_LINE_TICKNESS)
    for point in landmarks:
        cv.circle(img, point, HAND_CIRCLE_RADIUS, HAND_CIRCLE_COLOR, HAND_CIRCLE_THICKNESS)

camera = cv.VideoCapture(0)

ret, first_frame = camera.read()
H, W, _ = first_frame.shape

SCALE_DOWN = 0.25
resize = (int(W * SCALE_DOWN), int(H * SCALE_DOWN))

prev_time = 0

while camera.isOpened():
    ret, frame_bgr = camera.read()
    if not ret: break

    frame_bgr = cv.flip(frame_bgr, 1)
    for_mp_frame = cv.resize(frame_bgr, resize, interpolation=cv.INTER_NEAREST)
    rgb_mp_frame = cv.cvtColor(for_mp_frame, cv.COLOR_BGR2RGB) # mediapipeのRBBに合わせる
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_mp_frame)

    current_time = time.time()
    timestamp_ms = int(current_time * 1000)
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)

    if detection_result.hand_landmarks:
        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            cv.putText(frame_bgr, f"{count_up_fingers(hand_landmarks, detection_result.handedness[idx][0])}", ((idx * 30) + 150, 100), cv.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 2)

            scaleup_landmarks =  [(int(lm.x * W), int(lm.y * H)) for lm in hand_landmarks]
            draw_landmarks(frame_bgr, scaleup_landmarks)

    fps = 1 / (current_time - prev_time)
    cv.putText(frame_bgr, f"FPS: {int(fps)}", (20, 50), cv.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)

    prev_time = current_time
    cv.imshow('hand tracking test', frame_bgr)
    ky = cv.waitKey(1)
    if ky == ord("l"):
        break

camera.release()
cv.destroyAllWindows()