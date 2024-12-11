import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import Orange
import gdown
import io

# Google Drive 파일 ID
ows_file_id = '1PoZpN8KsuwQhFpv7cRQX1BZYJ64kIkAS'  # .ows 파일 ID

# Google Drive에서 .ows 파일 다운로드 함수
@st.cache(allow_output_mutation=True)
def load_ows_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'workflow.ows'
    gdown.download(url, output, quiet=False)

    # Orange 워크플로 로드
    workflow = Orange.workflow.read_workflow(output)
    return workflow

# 스펙트로그램 생성 함수
def generate_spectrogram(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)

    # 스펙트로그램을 이미지로 저장
    fig, ax = plt.subplots()
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='viridis')
    ax.set(title='Mel Spectrogram')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    return Image.open(buf), S_DB

# Streamlit 애플리케이션
st.title("오디오 파일 분류 - Orange 워크플로 통합")
st.write("오디오 파일을 업로드하고 분류 결과를 확인하세요.")

# Orange 워크플로 로드
st.write("워크플로를 로드 중입니다. 잠시만 기다려주세요...")
workflow = load_ows_from_drive(ows_file_id)
if workflow:
    st.success("워크플로가 성공적으로 로드되었습니다!")
else:
    st.error("워크플로 로드에 실패했습니다.")

# 파일 업로드 컴포넌트
uploaded_file = st.file_uploader("오디오 파일을 업로드하세요", type=["wav", "mp3", "ogg", "flac"])

if uploaded_file is not None:
    # 스펙트로그램 생성
    st.write("### 업로드된 오디오 파일")
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes, format=f"audio/{uploaded_file.name.split('.')[-1]}")

    st.write("### 생성된 스펙트로그램")
    spectrogram_image, spectrogram_data = generate_spectrogram(uploaded_file)
    st.image(spectrogram_image, caption="스펙트로그램", use_column_width=True)

    # Orange 워크플로에 입력 데이터 전달
    input_data = Orange.data.Table.from_numpy(
        domain=Orange.data.Domain([Orange.data.ContinuousVariable(f"feature_{i}") for i in range(spectrogram_data.shape[0])]),
        X=spectrogram_data.T
    )

    workflow.set_input("Data", input_data)
    workflow.run()

    # 출력 결과 가져오기
    results = workflow.get_output("Predictions")

    st.write("### 분류 결과")
    if results is not None:
        for i, instance in enumerate(results):
            st.write(f"샘플 {i + 1}: {instance[0]} (확률: {instance[1]:.2f})")
    else:
        st.warning("분류 결과를 가져오지 못했습니다.")
