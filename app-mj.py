import streamlit as st
from PIL import Image
import os
import tempfile
from predict-mj import load_model, predict_image

# 페이지 설정
st.set_page_config(page_title="FindEye", page_icon="👁️", layout="wide")

st.title("👁️ FindEye - 실종자 탐색 시스템")
st.markdown("보호자가 입력한 특징과 유사한 인물을 찾아드립니다.")

# 사이드바 - 모델 로드
st.sidebar.header("⚙️ 설정")
weights_path = st.sidebar.text_input(
    "모델 경로 (best.pt)",
    value="best.pt"
)

# 사이드바 - 조건 입력
st.sidebar.header("🔍 찾는 사람 특징")
top_color = st.sidebar.selectbox(
    "상의 밝기",
    ["상관없음", "bright_top (밝은 상의)", "dark_top (어두운 상의)"]
)
bottom_color = st.sidebar.selectbox(
    "하의 밝기",
    ["상관없음", "bright_bottom (밝은 하의)", "dark_bottom (어두운 하의)"]
)

# 필터 조건 생성
def get_filter(top, bottom):
    if top == "상관없음" and bottom == "상관없음":
        return None
    
    top_key = "bright_top" if "bright_top" in top else "dark_top" if "dark_top" in top else None
    bottom_key = "bright_bottom" if "bright_bottom" in bottom else "dark_bottom" if "dark_bottom" in bottom else None
    
    return top_key, bottom_key

# 유사도 계산
def calc_similarity(pred_class, top_key, bottom_key):
    score = 0
    if top_key and top_key in pred_class:
        score += 1
    if bottom_key and bottom_key in pred_class:
        score += 1
    return score

# 메인 - 이미지 업로드
st.header("📁 사진 업로드")
uploaded_files = st.file_uploader(
    "CCTV 사진을 업로드하세요 (여러 장 가능)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("🔍 분석 시작"):
        try:
            model = load_model(weights_path)
            filter_condition = get_filter(top_color, bottom_color)
            
            results_list = []
            
            with st.spinner("분석 중..."):
                for uploaded_file in uploaded_files:
                    # 임시 파일로 저장
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    
                    preds = predict_image(model, tmp_path)
                    
                    for pred in preds:
                        similarity = 0
                        if filter_condition:
                            top_key, bottom_key = filter_condition
                            similarity = calc_similarity(pred["class"], top_key, bottom_key)
                        else:
                            similarity = 1
                        
                        results_list.append({
                            "file": uploaded_file.name,
                            "class": pred["class"],
                            "confidence": pred["confidence"],
                            "similarity": similarity,
                            "image_path": tmp_path
                        })
                    
                    os.unlink(tmp_path)
            
            # 유사도 순으로 정렬
            results_list.sort(key=lambda x: (x["similarity"], x["confidence"]), reverse=True)
            
            # 결과 출력
            st.header("📊 분석 결과")
            
            if not results_list:
                st.warning("감지된 인물이 없어요.")
            else:
                for i, res in enumerate(results_list):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(res["image_path"], width=200)
                    with col2:
                        st.markdown(f"**파일명**: {res['file']}")
                        st.markdown(f"**분류**: {res['class']}")
                        st.markdown(f"**신뢰도**: {res['confidence']:.2%}")
                        st.markdown(f"**유사도**: {'⭐' * res['similarity']}")
                    st.divider()
        
        except Exception as e:
            st.error(f"오류 발생: {e}")
