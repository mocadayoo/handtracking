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

# draw settings
HAND_CIRCLE_COLOR = (0, 255, 0)
HAND_CIRCLE_RADIUS = 5
HAND_CIRCLE_THICKNESS = -1
HAND_LINE_COLOR = (0, 0, 255)
HAND_LINE_TICKNESS = 2

last_detect_result = None

def update_result(result, _output_image, _timestamp_ms):
    global last_detect_result
    last_detect_result = result

# mediapipe settings
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=update_result
)
detector = vision.HandLandmarker.create_from_options(options)

# mpの0~1の座標をカメラの画質まで拡大、変換
def mp_convert_to_camera_wh(point, w, h):
    return (int(point.x * w), int(point.y * h))

# normalize関数
def normalize(v):
    L = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(L > 0, v / L, v)

def finger_up_count(hand_landmarks):
    counter = 0

    p = {i: np.array([hand_landmarks[i].x, hand_landmarks[i].y, hand_landmarks[i].z]) for i in range(21)}

    # 手首から中指の付け根へのベクトルを取る <- これを手の向きとする。
    hand_vec = p[9] - p[0]

    # 親指を除く　人差し指から小指までの 指先と2個目の関節 <- これのベクトルを手のベクトルと計算し、指がたっているかカウントする
    tips = [8,12,16,20]
    mcp_bases = [6,10,14,18]

    for tip_idx, base_idx in zip(tips, mcp_bases):
        finger_vec = p[tip_idx] - p[base_idx]

        if np.dot(normalize(finger_vec), normalize(hand_vec)) > 0: counter += 1

    # 人差し指の付け根から小指の付け根へのベクトル 親指がその向きと同じ方向にしまわれるはず。
    palm_vec = p[17] - p[5]
    # 親指の第二関節から指先までのベクトル palm_vecと比較
    thumb_vec = p[4] - p[2]

    if np.dot(normalize(thumb_vec), normalize(palm_vec)) < 0: counter += 1

    return counter

def draw_landmarks(img, landmarks):
    for line_point in HAND_CONNECTS:
        cv.line(img, landmarks[line_point[0]], landmarks[line_point[1]], HAND_LINE_COLOR, HAND_LINE_TICKNESS)
    for point in landmarks:
        cv.circle(img, point, HAND_CIRCLE_RADIUS, HAND_CIRCLE_COLOR, HAND_CIRCLE_THICKNESS)

camera = cv.VideoCapture(0)
ret, first_frame = camera.read()
H, W, _ = first_frame.shape

SCALE_DOWN = 0.3
resize = (int(W * SCALE_DOWN), int(H * SCALE_DOWN))
prev_time = 0

while camera.isOpened():
    ret, frame_bgr = camera.read()
    if not ret: break

    frame_bgr = cv.flip(frame_bgr, 1)
    for_mp_frame = cv.resize(frame_bgr, resize, interpolation=cv.INTER_LINEAR)
    rgb_mp_frame = cv.cvtColor(for_mp_frame, cv.COLOR_BGR2RGB) # mediapipeのRBBに合わせる
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_mp_frame)

    current_time = time.time()
    timestamp_ms = int(current_time * 1000)
    detector.detect_async(mp_image, timestamp_ms)

    if last_detect_result is not None and last_detect_result.hand_landmarks:
        for idx, (hand_landmarks, _handedness) in enumerate(zip(last_detect_result.hand_landmarks, last_detect_result.handedness)):
            cv.putText(frame_bgr, f"{finger_up_count(hand_landmarks)}", ((idx * 30) + 150, 100), cv.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 2)

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