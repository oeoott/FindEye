"""
app/streamlit_app.py

FindEye 메인 웹 앱.

흐름:
    1. 사용자가 CCTV 영상을 업로드
    2. 파일명 기준으로 video_metadata.csv에서 장소/시작시간 자동 매칭
       (매칭 실패 시 사용자가 직접 입력)
    3. 상의/하의 밝기 조건 선택
    4. "검색" 버튼 클릭 -> YOLO로 프레임별 탐지 실행
    5. 조건에 맞는 후보만 crop + 장소 + 탐지시각 + confidence로 출력
    6. 전체 탐지 인원 수 / 후보 수 요약 표시

수정사항:
    - 후보 카드: 가로 2열로 출력
    - 후보 사진 세로 길이 통일 (CARD_HEIGHT = 200px)
    - 사진 옆에 장소/시각/신뢰도 정보 표시
    - Confidence 기본값 0.5로 변경

실행:
    streamlit run app/streamlit_app.py
"""

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from model.inference import FindEyeDetector, DEFAULT_WEIGHTS_PATH  # noqa: E402
from app.filtering import (  # noqa: E402
    load_metadata_table,
    match_metadata,
    search_video,
    candidates_to_dataframe,
    VideoMetadata,
)
from app.visualize import bgr_to_rgb  # noqa: E402

st.set_page_config(page_title="FindEye", page_icon="🔍", layout="wide")

CARD_HEIGHT = 200  # 후보 사진 세로 길이 통일 (px)


@st.cache_resource
def load_detector():
    if not DEFAULT_WEIGHTS_PATH.exists():
        return None
    try:
        return FindEyeDetector()
    except Exception as e:
        st.error(f"모델 로딩 실패: {e}")
        return None


@st.cache_data
def load_metadata():
    return load_metadata_table()


def resize_to_height(img_bgr: np.ndarray, height: int) -> np.ndarray:
    """세로 길이를 height로 통일, 가로는 비율 유지."""
    h, w = img_bgr.shape[:2]
    if h == 0:
        return img_bgr
    ratio = height / h
    new_w = max(1, int(w * ratio))
    return cv2.resize(img_bgr, (new_w, height))


def main():
    st.title("🔍 FindEye")
    st.caption("CCTV 영상에서 인상착의(상의/하의 밝기) 조건에 맞는 사람을 찾아드립니다.")

    detector = load_detector()
    metadata_df = load_metadata()

    if detector is None:
        st.warning(
            f"⚠️ 학습된 가중치 파일을 찾을 수 없습니다: `{DEFAULT_WEIGHTS_PATH}`\n\n"
            "`weights/best.pt` 파일을 프로젝트에 추가한 뒤 다시 실행해주세요."
        )

    # -----------------------------------------------------------------------
    # 1. 영상 업로드
    # -----------------------------------------------------------------------
    st.subheader("1. CCTV 영상 업로드")
    uploaded_file = st.file_uploader("영상 파일을 업로드하세요", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is None:
        st.info("영상을 업로드하면 다음 단계가 활성화됩니다.")
        return

    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        video_path = tmp_file.name

    st.video(uploaded_file)

    # -----------------------------------------------------------------------
    # 2. 메타데이터(장소/시간) 자동 매칭
    # -----------------------------------------------------------------------
    st.subheader("2. 장소 / 촬영 시작 시간")
    matched_meta = match_metadata(uploaded_file.name, metadata_df)

    if matched_meta.matched:
        st.success(f"메타데이터 자동 매칭 완료: **{matched_meta.location}** / {matched_meta.start_time}")
        location = matched_meta.location
        start_time = matched_meta.start_time
    else:
        st.warning("등록된 메타데이터가 없습니다. 직접 입력해주세요. (파일명: " + uploaded_file.name + ")")
        col1, col2 = st.columns(2)
        with col1:
            location = st.text_input("촬영 장소", value="")
        with col2:
            start_date = st.date_input("촬영 시작 날짜")
            start_time_input = st.time_input("촬영 시작 시각")
        import datetime as _dt
        start_time = _dt.datetime.combine(start_date, start_time_input) if location else None

    # -----------------------------------------------------------------------
    # 3. 검색 조건 선택
    # -----------------------------------------------------------------------
    st.subheader("3. 인상착의 조건 선택")
    col1, col2 = st.columns(2)
    with col1:
        top_choice = st.selectbox("상의 밝기", ["밝음", "어두움"])
    with col2:
        bottom_choice = st.selectbox("하의 밝기", ["밝음", "어두움"])

    top_bright = (top_choice == "밝음")
    bottom_bright = (bottom_choice == "밝음")

    with st.expander("⚙️ 고급 설정"):
        conf_threshold = st.slider(
            "Confidence 임계값", min_value=0.1, max_value=0.9, value=0.5, step=0.05
        )
        frame_stride = st.slider(
            "프레임 샘플링 간격 (작을수록 정확하지만 느림)",
            min_value=1, max_value=30, value=5, step=1
        )
        max_candidates = st.number_input(
            "최대 후보 수 (0 = 제한 없음)",
            min_value=0, value=0, step=1
        )

    # -----------------------------------------------------------------------
    # 4. 검색 실행
    # -----------------------------------------------------------------------
    st.subheader("4. 검색")
    search_disabled = (detector is None) or (not location)

    if search_disabled and detector is not None:
        st.caption("촬영 장소를 입력해야 검색할 수 있습니다.")

    if st.button("🔎 검색 시작", type="primary", disabled=search_disabled):
        progress_bar = st.progress(0, text="탐지 진행 중...")

        def update_progress(current_frame, total_frames):
            if total_frames > 0:
                pct = min(current_frame / total_frames, 1.0)
                progress_bar.progress(pct, text=f"탐지 진행 중... ({current_frame}/{total_frames} 프레임)")

        meta_for_search = VideoMetadata(
            video_file=uploaded_file.name,
            location=location,
            start_time=start_time,
            matched=True,
        )

        try:
            candidates, total_detected = search_video(
                video_path=video_path,
                detector=detector,
                top_bright=top_bright,
                bottom_bright=bottom_bright,
                metadata=meta_for_search,
                conf_threshold=conf_threshold,
                frame_stride=frame_stride,
                max_candidates=max_candidates if max_candidates > 0 else None,
                progress_callback=update_progress,
            )
        except Exception as e:
            st.error(f"검색 중 오류가 발생했습니다: {e}")
            return
        finally:
            progress_bar.empty()

        # -------------------------------------------------------------------
        # 5. 결과 출력
        # -------------------------------------------------------------------
        st.subheader("5. 검색 결과")

        col1, col2 = st.columns(2)
        col1.metric("전체 탐지 인원 수", total_detected)
        col2.metric("조건에 맞는 후보 수", len(candidates))

        if not candidates:
            st.info("조건에 맞는 후보를 찾지 못했습니다. Confidence 임계값을 낮추거나 다른 조건을 시도해보세요.")
            return

        st.markdown(f"**검색 조건**: 상의 {top_choice} / 하의 {bottom_choice}")
        st.markdown("---")

        for i in range(0, len(candidates), 2):
            row_candidates = candidates[i:i + 2]
            cols = st.columns(2)
            for col, cand in zip(cols, row_candidates):
                with col:
                    img_resized = resize_to_height(cand.crop_image, CARD_HEIGHT)
                    img_rgb = bgr_to_rgb(img_resized)

                    img_col, info_col = st.columns([1, 1])
                    with img_col:
                        st.image(img_rgb)
                    with info_col:
                        st.markdown(f"**📍 장소**  \n{cand.location}")
                        st.markdown(
                            f"**🕒 시각**  \n"
                            f"{cand.detected_time.strftime('%Y-%m-%d %H:%M:%S') if cand.detected_time else 'N/A'}"
                        )
                        st.markdown(f"**🎯 신뢰도**  \n{cand.detection.confidence:.2f}")
                        st.markdown(f"**🏷️ 클래스**  \n{cand.detection.class_name}")
            st.divider()

        st.markdown("**전체 결과 테이블**")
        result_df = candidates_to_dataframe(candidates)
        st.dataframe(result_df, use_container_width=True)

        csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 결과 CSV 다운로드",
            data=csv_bytes,
            file_name=f"findeye_results_{uploaded_file.name}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
