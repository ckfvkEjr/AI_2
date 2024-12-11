import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import Orange
import numpy as np

# 스펙트로그램 생성 함수
def audio_to_spectrogram(audio_path, save_path="spectrogram.png"):
    y, sr = librosa.load(audio_path)
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    plt.figure(figsize=(12, 6))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

# Orange 워크플로 로드 함수
@st.cache(allow_output_mutation=True)
def load_workflow_from_ows(ows_file):
    try:
        workflow = Orange.workflow.read_workflow(ows_file)
        st.success("Orange 워크플로가 성공적으로 로드되었습니다!")
        return workflow
    except Exception as e:
        st.error(f"워크플로 로드 중 오류 발생: {e}")
        return None

# Orange 워크플로 실행 함수
def run_workflow(workflow, image_path):
    try:
        # 이미지 데이터를 Orange의 Table 형식으로 변환
        table = Orange.data.Table.from_file(image_path)
        output = workflow.run(table)
        return output
    except Exception as e:
        st.error(f"워크플로 실행 중 오류 발생: {e}")
        return None

# Streamlit UI
st.title("음악 데이터 분류 애플리케이션 (Orange 기반)")

# OWS 파일 업로드
uploaded_ows = st.file_uploader("Orange 워크플로 파일(.ows)을 업로드하세요", type=["ows"])

if uploaded_ows:
    workflow = load_workflow_from_ows(uploaded_ows)

    # 오디오 파일 업로드
    uploaded_audio = st.file_uploader("음악 파일을 업로드하세요", type=["mp3", "wav"])

    if uploaded_audio:
        # 스펙트로그램 생성
        st.write("스펙트로그램 생성 중...")
        spectrogram_path = audio_to_spectrogram(uploaded_audio)

        # 생성된 스펙트로그램 표시
        st.image(spectrogram_path, caption="생성된 스펙트로그램", use_column_width=True)

        # Orange 워크플로 실행
        if workflow:
            output = run_workflow(workflow, spectrogram_path)
            if output:
                # 결과 출력
                st.write("분류 결과:")
                st.write(output)
