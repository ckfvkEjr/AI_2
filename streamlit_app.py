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
    # 오디오 로드
    y, sr = librosa.load(audio_file, duration=10)  # 10초 로드
    
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

# Google Drive에서 파일 다운로드 함수
@st.cache(allow_output_mutation=True)
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)

    # Fastai 모델 로드
    learner = load_learner(output)
    return learner

def display_left_content(image, prediction, probs, labels):
    st.write("### 왼쪽: 기존 출력 결과")
    if image is not None:
        st.image(image, caption="업로드된 이미지", use_column_width=True)
    st.write(f"예측된 클래스: {prediction}")
    st.markdown("<h4>클래스별 확률:</h4>", unsafe_allow_html=True)
    for label, prob in zip(labels, probs):
        st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 5px 0;">
                <strong style="color: #333;">{label}:</strong>
                <div style="background-color: #d3d3d3; border-radius: 5px; width: 100%; padding: 2px;">
                    <div style="background-color: #4CAF50; width: {prob*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                        {prob:.4f}
                    </div>
                </div>
        """, unsafe_allow_html=True)

def display_right_content(prediction, data):
    st.write("### 오른쪽: 동적 분류 결과")
    cols = st.columns(3)

    # 1st Row - Images
    for i in range(3):
        with cols[i]:
            st.image(data['images'][i], caption=f"이미지: {prediction}", use_column_width=True)
    # 2nd Row - YouTube Videos
    for i in range(3):
        with cols[i]:
            st.video(data['videos'][i])
            st.caption(f"유튜브: {prediction}")
    # 3rd Row - Text
    for i in range(3):
        with cols[i]:
            st.write(data['texts'][i])

# 메인 앱 함수
def main():
    # 스타일링을 통해 페이지 마진 줄이기
    st.markdown("""
        <style>
        .reportview-container .main .block-container {
            max-width: 90%;
            padding-top: 1rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)

    # 모델 로드
    st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
    learner = load_model_from_drive(file_id)
    st.success("모델이 성공적으로 로드되었습니다!")

    labels = learner.dls.vocab

    # 분류에 따라 다른 콘텐츠 관리
    content_data = {
        labels[0]: {
            'images': [
                "https://via.placeholder.com/300?text=Label1_Image1",
                "https://via.placeholder.com/300?text=Label1_Image2",
                "https://via.placeholder.com/300?text=Label1_Image3"
            ],
            'videos': [
                "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
                "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
                "https://www.youtube.com/watch?v=3JZ_D3ELwOQ"
            ],
            'texts': [
                "Label 1 관련 첫 번째 텍스트 내용입니다.",
                "Label 1 관련 두 번째 텍스트 내용입니다.",
                "Label 1 관련 세 번째 텍스트 내용입니다."
            ]
        },
        labels[1]: {
            'images': [
                "https://via.placeholder.com/300?text=Label2_Image1",
                "https://via.placeholder.com/300?text=Label2_Image2",
                "https://via.placeholder.com/300?text=Label2_Image3"
            ],
            'videos': [
                "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
                "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
                "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
            ],
            'texts': [
                "Label 2 관련 첫 번째 텍스트 내용입니다.",
                "Label 2 관련 두 번째 텍스트 내용입니다.",
                "Label 2 관련 세 번째 텍스트 내용입니다."
            ]
        },
        labels[2]: {
            'images': [
                "https://i.ibb.co/Gp5KgvV/memed-io-output.jpg",
                "https://i.ibb.co/Gp5KgvV/memed-io-output.jpg",
                "https://i.ibb.co/Gp5KgvV/memed-io-output.jpg"
            ],
            'videos': [
                "https://www.youtube.com/watch?v=5tafCyiYGpU",
                "https://www.youtube.com/watch?v=unrcrAUdqH8",
                "https://www.youtube.com/watch?v=3JZ_D3ELwOQ"
            ],
            'texts': [
                "핑동",
                "커스텀마이징 쉬운건 좋은데",
                "똥겜인듯"
            ]
        }
    }

    # 탭 생성
    tab1, tab2 = st.tabs(["이미지 분류", "음악 분류"])

    with tab1:
        st.header("이미지 분류")
        # 파일 업로드 컴포넌트 (jpg, png, jpeg, webp, tiff 지원)
        uploaded_image = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg", "webp", "tiff"])

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            img = PILImage.create(uploaded_image)
            prediction, _, probs = learner.predict(img)

            left_column, right_column = st.columns([1, 2])  # 왼쪽과 오른쪽의 비율 조정

            with left_column:
                display_left_content(image, prediction, probs, labels)

            with right_column:
                # 분류 결과에 따른 콘텐츠 선택
                data = content_data.get(prediction, {
                    'images': ["https://via.placeholder.com/300"] * 3,
                    'videos': ["https://www.youtube.com/watch?v=3JZ_D3ELwOQ"] * 3,
                    'texts': ["기본 텍스트"] * 3
                })
                display_right_content(prediction, data)

    with tab2:
        st.header("음악 분류")
        # 음악 파일 업로드
        uploaded_music = st.file_uploader("음악 파일을 업로드하세요 (MP3, WAV)", 
                                          type=['mp3', 'wav'])
        
        if uploaded_music is not None:
            # 멜 스펙트로그램 생성
            mel_spec_buf = create_mel_spectrogram(uploaded_music)
            
            # 이미지 열기
            mel_spec_image = Image.open(mel_spec_buf)
            
            # 예측 수행
            try:
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
                    for label, prob in zip(labels, probs):
                        st.markdown(f"""
                            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 5px 0;">
                                <strong style="color: #333;">{label}:</strong>
                                <div style="background-color: #d3d3d3; border-radius: 5px; width: 100%; padding: 2px;">
                                    <div style="background-color: #4CAF50; width: {prob.item()*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                                        {prob.item():.4f}
                                    </div>
                                </div>
                        """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"모델 예측 중 오류 발생: {e}")

# 앱 실행
if __name__ == "__main__":
    main()
