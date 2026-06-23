"""
app/visualize.py

탐지 결과를 화면에 보여주기 위한 시각화 유틸리티.
- 프레임 위에 탐지 박스 + 라벨 그리기
- 후보 crop 이미지에 정보 오버레이
- BGR(OpenCV) <-> RGB(Streamlit/PIL) 변환 헬퍼

streamlit_app.py에서 이 모듈을 가져다 화면에 표시한다.
"""

import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from model.inference import Detection  # noqa: E402

# 클래스별 박스 색상 (BGR 순서, OpenCV 기준)
CLASS_COLORS = {
    0: (60, 200, 60),    # bright_top_bright_bottom - 초록
    1: (60, 150, 230),   # bright_top_dark_bottom   - 주황
    2: (230, 160, 60),   # dark_top_bright_bottom   - 파랑계열
    3: (60, 60, 220),    # dark_top_dark_bottom     - 빨강
}
DEFAULT_COLOR = (200, 200, 200)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """OpenCV(BGR) 이미지를 Streamlit/PIL에서 쓰는 RGB로 변환."""
    if image is None or image.size == 0:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    highlight_class_id: int = None,
) -> np.ndarray:
    """
    프레임 위에 탐지된 모든 박스와 라벨(클래스명 + confidence)을 그린다.

    Args:
        frame: 원본 프레임 (BGR)
        detections: Detection 리스트
        highlight_class_id: 지정하면 해당 클래스만 굵은 테두리로 강조,
                             나머지는 흐리게(반투명) 표시

    Returns:
        박스가 그려진 새 이미지 (원본은 수정하지 않음)
    """
    output = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.box_xyxy]
        color = CLASS_COLORS.get(det.class_id, DEFAULT_COLOR)

        is_target = (highlight_class_id is None) or (det.class_id == highlight_class_id)
        thickness = 3 if is_target else 1

        if not is_target:
            # 조건에 맞지 않는 탐지는 옅게 표시
            overlay = output.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
            output = cv2.addWeighted(overlay, 0.35, output, 0.65, 0)
            continue

        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

        label = f"{det.class_name} {det.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        label_y1 = max(0, y1 - text_h - baseline - 4)
        cv2.rectangle(output, (x1, label_y1), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(
            output, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )

    return output


def annotate_crop(crop: np.ndarray, confidence: float, label: str = None) -> np.ndarray:
    """
    crop된 후보 이미지 하단에 confidence(및 선택적 라벨)를 텍스트로 오버레이.
    카드형 UI에서 썸네일로 보여줄 때 사용.
    """
    if crop is None or crop.size == 0:
        return crop

    output = crop.copy()
    h, w = output.shape[:2]

    bar_height = 22
    bar = np.zeros((bar_height, w, 3), dtype=np.uint8)
    text = f"{confidence:.0%}" if label is None else f"{label} {confidence:.0%}"

    cv2.putText(
        bar, text, (4, bar_height - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
    )

    return np.vstack([output, bar])


def draw_summary_overlay(frame: np.ndarray, total_detected: int, num_candidates: int) -> np.ndarray:
    """프레임 좌상단에 '전체 탐지 인원 / 후보 수' 요약 텍스트를 그린다."""
    output = frame.copy()
    text = f"Total: {total_detected}  |  Candidates: {num_candidates}"

    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(output, (5, 5), (15 + text_w, 15 + text_h + baseline), (0, 0, 0), -1)
    cv2.putText(
        output, text, (10, 10 + text_h),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
    )
    return output