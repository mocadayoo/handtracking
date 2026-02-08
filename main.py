import cv2 as cv
import numpy as np
import time
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from Utils.finger_pose import get_pose

# 手の点(21個)をつなぐ線を設定
HAND_CONNECTS = [
    (0,1), (1,2), (2,3), (3,4),       # 親指
    (0,5), (5,6), (6,7), (7,8),       # 人差し指
    (9,10), (10,11), (11,12),         # 中指
    (13,14), (14,15), (15,16),        # 薬指
    (17,18), (18,19), (19,20),        # 小指
    (5,9), (9,13), (13,17), (0,17)    # 指の付け根
]
# 検知の閾値
GET_FINGER_UP_THRESHOLD = -0.1
GET_FINGER_UP_THUMB_THRESHOLD = -0.3
# 検出に渡すカメラの画質を調整
SCALE_DOWN = 0.3

MAX_FPS = 30
# 誤差があるため少しだけ増やす
MAX_FPS += 5

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

# 正規化する関数
def normalize(v):
    L = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(L > 0, v / L, v)

def get_finger_up(hand_landmarks):
    p = {i: np.array([hand_landmarks[i].x, hand_landmarks[i].y, hand_landmarks[i].z]) for i in range(21)}

    # 基準となるベクトル
    hand_vec = p[9] - p[0] # 手首から中指付け根へのベクトル 親指以外の指の判定用
    palm_vec = p[5] - p[17] # 小指の付け根から人差し指の付け根へのベクトル 親指の判定用

    # それぞれの指の (指先, 第二関節)
    fingers = {
        "index": (8,6),
        "middle": (12,10),
        "ring": (16,14),
        "little": (20,18)
    }

    fingers_json = {}

    # 親指のベクトル (親指の指先 - 親指の第一関節)
    thumb_vec = p[4] - p[3]
    fingers_json["thumb"] = bool(np.dot(normalize(thumb_vec), normalize(palm_vec)) > GET_FINGER_UP_THRESHOLD) # palm_vecの方向に対してthumb_vecが180以内に納まっているか判断


    # 名前と中の使用する関節の番号を取り出して計算
    for name, (tip, mcp) in fingers.items():
        finger_vec = p[tip] - p[mcp] # 指の向き (ベクトル) を計算
        fingers_json[name] = bool(np.dot(normalize(finger_vec), normalize(hand_vec)) > GET_FINGER_UP_THUMB_THRESHOLD) # hand_vecの方向から180度の範囲内をfinger_vecが向いているか判断

    # jsonの値を全部取り出し、Trueかどうか、Trueなら1を返しそれをsumが計算する。
    fingers_json["up_count"] = sum(1 for status in fingers_json.values() if status is True)
    return fingers_json

def draw_landmarks(img, landmarks):
    for line_point in HAND_CONNECTS:
        cv.line(img, landmarks[line_point[0]], landmarks[line_point[1]], HAND_LINE_COLOR, HAND_LINE_TICKNESS)
    for point in landmarks:
        cv.circle(img, point, HAND_CIRCLE_RADIUS, HAND_CIRCLE_COLOR, HAND_CIRCLE_THICKNESS)

camera = cv.VideoCapture(0)
ret, first_frame = camera.read()
H, W, _ = first_frame.shape

past_frame_time = deque(maxlen=30)
target_frame_time = 1 / MAX_FPS
resize = (int(W * SCALE_DOWN), int(H * SCALE_DOWN))
prev_time = 0

while camera.isOpened():
    current_time = time.perf_counter()
    ret, frame_bgr = camera.read()
    if not ret: break

    frame_bgr = cv.flip(frame_bgr, 1)
    for_mp_frame = cv.resize(frame_bgr, resize, interpolation=cv.INTER_LINEAR)
    rgb_mp_frame = cv.cvtColor(for_mp_frame, cv.COLOR_BGR2RGB) # mediapipeのRBBに合わせる
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_mp_frame)

    timestamp_ms = int(current_time * 1000)
    detector.detect_async(mp_image, timestamp_ms)

    if last_detect_result is not None and last_detect_result.hand_landmarks: # LIVE_STREAMの場合検知の処理が非同期なためデータがあるか確認
        for idx, (hand_landmarks, _handedness) in enumerate(zip(last_detect_result.hand_landmarks, last_detect_result.handedness)):
            finger_status = get_finger_up(hand_landmarks)
            cv.putText(frame_bgr, f"{finger_status['up_count']}", ((idx * 30) + 150, 100), cv.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
            print(finger_status)
            print(get_pose(finger_status))
            # 画面の縦横幅まで検知の処理結果の座標を拡大
            scaleup_landmarks =  [(int(lm.x * W), int(lm.y * H)) for lm in hand_landmarks]
            draw_landmarks(frame_bgr, scaleup_landmarks)

    past_frame_time.append(current_time - prev_time)
    avg_frame_time = sum(past_frame_time) / len(past_frame_time)
    # fps表示の計算 時間を使用
    avg_fps = 1 / avg_frame_time
    cv.putText(frame_bgr, f"FPS: {int(avg_fps)}", (20, 50), cv.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)

    prev_time = current_time
    cv.imshow('hand tracking test', frame_bgr)

    # (ms) fps制限 この計算の時間から処理がスタートしたときの時間を引くことでこのframeにかかった時間を算出
    #      算出した時間を1frameにかけてほしい時間から引き、その時間が1msとどっちがでかいか計算
    wait_time = max(1, int((target_frame_time - (time.perf_counter() - current_time)) * 1000))
    ky = cv.waitKey(wait_time)
    if ky == ord("l"):
        break

camera.release()
cv.destroyAllWindows()