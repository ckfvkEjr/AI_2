import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from fastai.vision.all import *
from PIL import Image
import gdown
import io

# Google Drive 파일 ID
file_id = '14y7xPjVyBg_oFasSuODSHQP3b2BSm6vt'

# 멜 스펙트로그램 생성 함수
def create_mel_spectrogram(audio_file):
    # 오디오 로드 (10초)
    y, sr = librosa.load(audio_file, duration=10)  
    
    # 멜 스펙트로그램 생성
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    
    # 로그 스케일 변환
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 멜 스펙트로그램 시각화
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    
    # 이미지를 바이트로 저장
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    
    return buf

# 모델 로드 함수
@st.cache_data
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)
    learner = load_learner(output)
    return learner

def main():
    st.title('음악 장르 분류기')
    
    # 모델 로드
    st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
    learner = load_model_from_drive(file_id)
    st.success("모델이 성공적으로 로드되었습니다!")

    # 음악 파일 업로드
    uploaded_music = st.file_uploader("음악 파일을 업로드하세요 (MP3, WAV)", type=['mp3', 'wav'])
    
    if uploaded_music is not None:
        # 멜 스펙트로그램 생성
        mel_spec_buf = create_mel_spectrogram(uploaded_music)
        
        # 이미지 열기
        mel_spec_image = Image.open(mel_spec_buf)
        
        # 예측 수행
        try:
            # 모델에 멜 스펙트로그램 이미지 입력
            pred, pred_idx, probs = learner.predict(mel_spec_image)
            
            # 결과 표시
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 멜 스펙트로그램")
                st.image(mel_spec_image, caption="음악 파일의 멜 스펙트로그램", use_column_width=True)
            
            with col2:
                st.write("### 분류 결과")
                st.write(f"**예측된 장르:** {pred}")
                
                st.markdown("<h4>장르별 확률:</h4>", unsafe_allow_html=True)
                labels = learner.dls.vocab
                for label, prob in zip(labels, probs):
                    st.markdown(f"""
                        <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 5px 0;">
                            <strong style="color: #333;">{label}:</strong>
                            <div style="background-color: #d3d3d3; border-radius: 5px; width: 100%; padding: 2px;">
                                <div style="background-color: #4CAF50; width: {prob.item()*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                                    {prob.item():.4f}
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"모델 예측 중 오류 발생: {e}")

# 앱 실행
if __name__ == "__main__":
    main()
