import streamlit as st
from PIL import Image
import os
from predict_mj import load_model, predict_image

# 페이지 설정
st.set_page_config(page_title="FindEye", page_icon="👁️", layout="wide")

st.title("👁️ FindEye - 실종자 탐색 시스템")
st.markdown("보호자가 입력한 특징과 유사한 인물을 찾아드립니다.")

# 사이드바 - 모델 로드
st.sidebar.header("⚙️ 설정")
weights_path = st.sidebar.text_input("모델 경로 (best.pt)", value="best.pt")

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

# 데이터셋 폴더에서 이미지 전부 가져오기
def get_all_images(dataset_path="dataset"):
    image_list = []
    for split in ["train", "valid", "test"]:
        img_folder = os.path.join(dataset_path, split, "images")
        if os.path.exists(img_folder):
            for filename in os.listdir(img_folder):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_list.append(os.path.join(img_folder, filename))
    return image_list

# 유사도 계산
def calc_similarity(pred_class, top_key, bottom_key):
    score = 0
    if top_key and top_key in pred_class:
        score += 1
    if bottom_key and bottom_key in pred_class:
        score += 1
    return score

# 분석 시작 버튼
if st.button("🔍 분석 시작"):
    try:
        model = load_model(weights_path)
        
        # 조건 파싱
        top_key = "bright_top" if "bright_top" in top_color else "dark_top" if "dark_top" in top_color else None
        bottom_key = "bright_bottom" if "bright_bottom" in bottom_color else "dark_bottom" if "dark_bottom" in bottom_color else None

        # 데이터셋 이미지 전부 가져오기
        all_images = get_all_images("dataset")
        st.info(f"총 {len(all_images)}장 분석 중...")

        results_list = []

        with st.spinner("분석 중..."):
            for image_path in all_images:
                preds = predict_image(model, image_path)
                for pred in preds:
                    similarity = calc_similarity(pred["class"], top_key, bottom_key)
                    results_list.append({
                        "image_path": image_path,
                        "class": pred["class"],
                        "confidence": pred["confidence"],
                        "similarity": similarity,
                        "image_data": Image.open(image_path).copy()
                    })

        # 유사도 순으로 정렬
        results_list.sort(key=lambda x: (x["similarity"], x["confidence"]), reverse=True)

        # 상위 20개만 출력
        st.header("📊 분석 결과 (유사도 높은 순)")
        top_results = results_list[:20]

        if not top_results:
            st.warning("감지된 인물이 없어요.")
        else:
            cols = st.columns(4)
            for i, res in enumerate(top_results):
                with cols[i % 4]:
                    st.image(res["image_data"], use_column_width=True)
                    st.markdown(f"**{res['class']}**")
                    st.markdown(f"신뢰도: {res['confidence']:.2%}")
                    st.markdown(f"유사도: {'⭐' * res['similarity']}")

    except Exception as e:
        st.error(f"오류 발생: {e}")
