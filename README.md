## 説明
このレポジトリはあくまで何かをリリースするためではなく、ただ勉強の一環で作成されています。

そのためissueなどを立てていただいても対応することができません。

またコードが汚いなどあるかもしれませんが、努力しますのでご容赦ください

## セットアップ

### 1. 仮想環境の作成そして有効 (推奨)
ライブラリの衝突やその他の問題を回避するために、この手順を実施することを推奨します。
```bash
# 仮想環境の作成
# 一番最後のvenvは好きな名前に変えてもいい。
# ただ変更した場合はこの後のコマンドのvenvもその名前に変更すること。
python -m venv venv

# windows
./venv/Scripts/activate

# linux & macos
source ./venv/bin/activate
```

### 2. 依存ライブラリのインストール
```bash
pip install mediapipe opencv-python numpy
```

### 3. hand_landmarkのモデルをダウンロード
1. [hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)
をダウンロードします。
2. ダウンロードしたモデルを`main.py`と同じ階層に移動します。

### 4. 実行
```bash
python main.py
```