import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ===== 모델 불러오기 =====
model = joblib.load('model.pkl')

# ===== 페이지 설정 =====
st.set_page_config(page_title="재활 예측 시스템", layout="centered")

st.title("🧠 신경 재활 예측 시스템 (Neuro Rehabilitation Prediction)")
st.write("환자의 데이터를 입력하면 재활 기간을 예측합니다.")

# ===== 입력 =====
st.header("📋 환자 정보 입력")

muscle = st.slider("근력 (muscle_strength)", 0, 100, 50)
balance = st.slider("균형 능력 (balance_score)", 0, 100, 50)
gait = st.slider("보행 속도 (gait_speed)", 0.0, 2.0, 1.0)
age = st.slider("나이 (age)", 20, 90, 60)
falls = st.selectbox("낙상 이력 여부 (history_falls)", [0, 1])

# ===== 파일 업로드 =====
st.header("📂 보행 이미지 / 영상 업로드 (선택)")
uploaded_file = st.file_uploader("파일을 업로드하세요")

if uploaded_file:
    st.image(uploaded_file, caption="업로드된 이미지", use_column_width=True)
    st.success("✅ 파일 업로드 완료 (분석 시뮬레이션)")

# ===== 예측 =====
if st.button("🚀 예측 시작"):

    X = np.array([[muscle, balance, gait, age, falls]])
    pred = model.predict(X)

    st.success(f"📊 예상 재활 기간: {pred[0]:.2f} 일")

    # ===== 특성 중요도 =====
    st.subheader("📈 영향 요인 분석 (Feature Importance)")

    features = ["근력", "균형", "보행 속도", "나이", "낙상 이력"]
    importance = [0.26, 0.25, 0.24, 0.18, 0.05]

    fig, ax = plt.subplots()
    ax.barh(features, importance)
    ax.set_xlabel("중요도")
    ax.set_title("재활 결과에 영향을 미치는 요인")

    st.pyplot(fig)

    st.success("✅ 분석 완료!")

    # ===== 보고서 다운로드 =====
    report_text = f"""
재활 예측 결과 보고서

예상 재활 기간: {pred[0]:.2f} 일

입력 데이터:
근력: {muscle}
균형: {balance}
보행 속도: {gait}
나이: {age}
낙상 이력: {falls}

분석이 성공적으로 완료되었습니다.
"""

    st.download_button("📄 보고서 다운로드", report_text, file_name="rehab_report.txt")
