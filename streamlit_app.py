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
    # 오디오 로드
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

# 왼쪽 콘텐츠 표시 함수
def display_left_content(image, prediction, probs, labels):
    st.write("### 왼쪽: 기존 출력 결과")
    if image is not None:
        st.image(image, caption="멜 스펙트로그램", use_column_width=True)
    st.write(f"**예측된 장르:** {prediction}")
    st.markdown("<h4>장르별 확률:</h4>", unsafe_allow_html=True)
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

# 오른쪽 콘텐츠 표시 함수
def display_right_content(prediction, data):
    st.write("### 오른쪽: 동적 분류 결과")
    
    # 여러 개의 열을 나누기 위해 3개 열 생성
    cols = st.columns(3)  # 3개의 열을 생성
    
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
    
    

# 메인 앱
def main():
    st.title('음악 장르 분류기')

    # 스타일링: 페이지 마진 줄이기
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

    # 분류에 따라 다른 콘텐츠 관리
    content_data = {
        'Label[0]': {
            'images': [
                "https://i.ibb.co/Gp5KgvV/memed-io-output.jpg",
                "https://i.ibb.co/Gp5KgvV/memed-io-output.jpg",
                "https://i.ibb.co/Gp5KgvV/memed-io-output.jpg"
            ],
            'videos': [
                "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
                "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
                "https://www.youtube.com/watch?v=3JZ_D3ELwOQ"
            ],
            'texts': [
                "Label 1",
                "Label 1",
                "Label 1 "
            ]
        },
        'Label[1]': {
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
        'Label[2]': {
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
        'Label[3]': {
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
        'Label[4]': {
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
        'Label[5]': {
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
        }
    }

    # 레이아웃 설정
    left_column, right_column = st.columns([1, 2])  # 왼쪽과 오른쪽의 비율 조정

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
            print(pred)

            with left_column:
                display_left_content(mel_spec_path, pred, probs, learner.dls.vocab)

            with right_column:
                data = content_data.get(pred, {
                    'images': ["https://via.placeholder.com/300"] * 3,
                    'videos': ["https://www.youtube.com/watch?v=3JZ_D3ELwOQ"] * 3,
                    'texts': ["기본 텍스트"] * 3
                })
                display_right_content(pred, data)




        except Exception as e:
            st.error(f"모델 예측 중 오류 발생: {e}")

# 앱 실행
if __name__ == "__main__":
    main()
