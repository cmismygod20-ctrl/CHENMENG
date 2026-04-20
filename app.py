import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
from PIL import Image

# ===== 标题（中韩双语）=====
st.title("跌倒检测系统 / 낙상 감지 시스템")

# ===== 上传选择 =====
file = st.file_uploader(
    "上传图片或视频 / 이미지 또는 비디오 업로드",
    type=["jpg", "png", "mp4", "avi", "mov"]
)

# ===== 加载模型（只加载一次）=====
@st.cache_resource
def load_model():
    return YOLO("yolov8n-pose.pt")

model = load_model()

# ===== 判断函数（核心分析逻辑）=====
def detect_fall_from_keypoints(kpts):
    for person in kpts:
        x = person[:, 0]
        y = person[:, 1]

        width = max(x) - min(x)
        height = max(y) - min(y)

        if width > height:
            return True
    return False

# ===== 处理图片 =====
def process_image(image):
    results = model(image)

    if results[0].keypoints is not None:
        kpts = results[0].keypoints.xy.cpu().numpy()
        return detect_fall_from_keypoints(kpts)
    return False

# ===== 处理视频 =====
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fall_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        if results[0].keypoints is not None:
            kpts = results[0].keypoints.xy.cpu().numpy()

            if detect_fall_from_keypoints(kpts):
                fall_detected = True
                break

    cap.release()
    return fall_detected

# ===== 主逻辑 =====
if file is not None:

    file_type = file.type

    st.info("分析中... / 분석 중...")

    # 👉 图片
    if "image" in file_type:
        image = Image.open(file)
        st.image(image, caption="上传图片 / 업로드된 이미지")

        fall = process_image(image)

    # 👉 视频
    elif "video" in file_type:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())

        st.video(file)

        fall = process_video(tfile.name)

    else:
        st.warning("不支持的文件类型 / 지원하지 않는 형식")
        fall = None

    # ===== 输出结果 =====
    if fall is not None:
        if fall:
            st.error("⚠️ 检测到跌倒！ / 낙상이 감지되었습니다!")
        else:
            st.success("✅ 未检测到跌倒 / 정상 상태입니다")
