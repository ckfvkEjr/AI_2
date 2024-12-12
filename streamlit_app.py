import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
from fastai.vision.all import *
from PIL import Image
import gdown

# Google Drive 파일 ID
file_id = '14y7xPjVyBg_oFasSuODSHQP3b2BSm6vt'

# 모델 로드 함수
@st.cache_data
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)
    learner = load_learner(output)
    return learner

# 멜 스펙트로그램 생성 함수 (이미지 저장)
def create_mel_spectrogram(audio_file, output_path="mel_spectrogram.png"):
    # 오디오 로드 (10초)
    y, sr = librosa.load(audio_file)

    # 멜 스펙트로그램 생성
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 멜 스펙트로그램 시각화 및 저장
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return output_path

# 메인 앱
def main():
    st.title('음악 장르 분류기')

    # 모델 로드
    st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
    learner = load_model_from_drive(file_id)
    st.success("모델이 성공적으로 로드되었습니다!")

    # 음악 파일 업로드
    uploaded_music = st.file_uploader("음악 파일을 업로드하세요 (MP3, WAV)", type=['mp3', 'wav'])

    if uploaded_music is not None:
        # 멜 스펙트로그램 생성 및 저장
        mel_spec_path = create_mel_spectrogram(uploaded_music)

        # 이미지 열기
        mel_spec_image = PILImage.create(mel_spec_path)

        # 모델 예측
        try:
            pred, pred_idx, probs = learner.predict(mel_spec_image)

            # 결과 표시
            st.image(mel_spec_path, caption="멜 스펙트로그램", use_column_width=True)
            st.write(f"**예측된 장르:** {pred}")

            st.markdown("<h4>장르별 확률:</h4>", unsafe_allow_html=True)
            labels = learner.dls.vocab
            for label, prob in zip(labels, probs):
                st.write(f"- {label}: {prob:.4f}")

        except Exception as e:
            st.error(f"모델 예측 중 오류 발생: {e}")

# 앱 실행
if __name__ == "__main__":
    main()
