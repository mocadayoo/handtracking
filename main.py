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

    is_right_hand = handedness.category_name == "Right"
    # 親指の付け根 < 小指の付け根で面裏を判断
    is_palm = (hand_landmarks[5].x < hand_landmarks[17].x) if is_right_hand else (hand_landmarks[5].x > hand_landmarks[17].x)

    # 親指は 2と4の点のxをさっきのpalmなどを使って条件を反転させながら判断
    thumb_4 = hand_landmarks[4]
    thumb_2 = hand_landmarks[2]

    if is_right_hand:
        if is_palm:
            if thumb_4.x < thumb_2.x: counter += 1
        else:
            if thumb_4.x > thumb_2.x: counter += 1
    else:
        if is_palm:
            if thumb_4.x > thumb_2.x: counter += 1
        else:
            if thumb_4.x < thumb_2.x: counter += 1

    return counter

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
                    mp_convert_to_camera_wh(p1, w, h),
                    mp_convert_to_camera_wh(p2, w, h),
                    HAND_LINE_COLOR, HAND_LINE_TICKNESS)

        for landmark in hand_landmarks:
            cx, cy = mp_convert_to_camera_wh(landmark, w, h) # さっき取得した比率をもとに0~1を変換
            cv.circle(image, (cx, cy), HAND_CIRCLE_RADIUS, HAND_CIRCLE_COLOR, HAND_CIRCLE_THICKNESS) # 緑の点で描画

    return image

camera = cv.VideoCapture(0)

while camera.isOpened():
    ret, frame_bgr = camera.read()
    if not ret: break

    frame_bgr = cv.flip(frame_bgr, 1)
    rgb_frame = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB) # mediapipeのRBBに合わせる
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms = int(time.time() * 1000)
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)

    for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
        print(count_up_fingers(hand_landmarks, detection_result.handedness[idx][0]))
    annotated_image = draw_landmarks(frame_bgr, detection_result)

    cv.imshow('hand tracking test', annotated_image)
    if cv.waitKey(1) & 0xFF == ord('l'):
        break

camera.release()
cv.destroyAllWindows()