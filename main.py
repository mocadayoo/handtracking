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

def draw_landmarks(image, detection_result):
    if not detection_result.hand_landmarks:
        return image # 手がない場合はただの画像のみ返す

    h, w, _ = image.shape # カメラの画像の縦横比
    for hand_landmarks in detection_result.hand_landmarks:
        # 線を描画してから点を描画
        for connect in HAND_CONNECTS:
            p1 = hand_landmarks[connect[0]]
            p2 = hand_landmarks[connect[1]]
            cv.line(image,
                    (int(p1.x * w), int(p1.y * h)),
                    (int(p2.x * w), int(p2.y * h)),
                    HAND_LINE_COLOR, HAND_LINE_TICKNESS)

        for landmark in hand_landmarks:
            cx, cy = int(landmark.x * w), int(landmark.y * h) # さっき取得した比率をもとに0~1を変換
            cv.circle(image, (cx, cy), HAND_CIRCLE_RADIUS, HAND_CIRCLE_COLOR, HAND_CIRCLE_THICKNESS) # 緑の点で描画

    return image

"""
自分の環境では
1. iphone camera
2. obs camera
3. macbook camera
なので 0~2で 2
"""
camera = cv.VideoCapture(2)

while camera.isOpened():
    ret, frame_bgr = camera.read()
    if not ret: break

    frame_bgr = cv.flip(frame_bgr, 1)
    rgb_frame = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB) # mediapipeのRBBに合わせる
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms = int(time.time() * 1000)
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)

    annotated_image = draw_landmarks(frame_bgr, detection_result)

    cv.imshow('hand tracking test', annotated_image)
    if cv.waitKey(1) & 0xFF == ord('l'):
        break

camera.release()
cv.destroyAllWindows()