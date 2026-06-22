"""
model/inference.py

학습된 YOLO 모델(weights/best.pt)을 불러와 이미지/프레임 단위로
사람(인상착의 클래스) 탐지를 수행하는 래퍼 모듈.

app/filtering.py, app/streamlit_app.py에서 이 모듈을 가져다 쓴다.

클래스 정의 (data.yaml과 동일해야 함):
    0: bright_top_bright_bottom
    1: bright_top_dark_bottom
    2: dark_top_bright_bottom
    3: dark_top_dark_bottom
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # ultralytics 미설치 환경에서도 모듈 임포트는 가능하도록


# data.yaml과 반드시 동일한 순서로 유지
CLASS_NAMES = {
    0: "bright_top_bright_bottom",
    1: "bright_top_dark_bottom",
    2: "dark_top_bright_bottom",
    3: "dark_top_dark_bottom",
}

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "weights" / "best.pt"


@dataclass
class Detection:
    """탐지 결과 1건을 담는 데이터 클래스."""
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: tuple  # (x1, y1, x2, y2) - 원본 프레임 좌표 기준 픽셀 단위
    frame_idx: Optional[int] = None  # 영상에서 몇 번째 프레임인지 (이미지 단일 추론 시 None)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """이 탐지 박스 영역만 잘라낸 이미지(BGR ndarray)를 반환."""
        x1, y1, x2, y2 = [int(v) for v in self.box_xyxy]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return frame[y1:y2, x1:x2].copy()


class FindEyeDetector:
    """
    YOLO 가중치를 로드하고 프레임/이미지 단위 탐지를 수행하는 클래스.

    사용 예:
        detector = FindEyeDetector()  # weights/best.pt 자동 로드
        detections = detector.detect(frame, conf_threshold=0.4)
        for d in detections:
            print(d.class_name, d.confidence)
    """

    def __init__(self, weights_path: Optional[str] = None):
        if YOLO is None:
            raise ImportError(
                "ultralytics 패키지가 설치되어 있지 않습니다. "
                "`pip install ultralytics` 로 설치 후 다시 시도해주세요."
            )

        self.weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"가중치 파일을 찾을 수 없습니다: {self.weights_path}\n"
                f"학습된 best.pt 파일을 weights/ 폴더에 넣어주세요."
            )

        self.model = YOLO(str(self.weights_path))

        # 모델 자체에 클래스 이름이 들어있으면 그것을 신뢰 (학습 시점 data.yaml 기준)
        # 없으면 위에서 정의한 CLASS_NAMES를 fallback으로 사용
        model_names = getattr(self.model, "names", None)
        if model_names:
            self.class_names = {int(k): v for k, v in model_names.items()}
        else:
            self.class_names = CLASS_NAMES

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        frame_idx: Optional[int] = None,
    ) -> List[Detection]:
        """
        단일 프레임(이미지)에 대해 탐지를 수행한다.

        Args:
            frame: BGR ndarray (cv2로 읽은 이미지/프레임)
            conf_threshold: confidence 임계값 (이 값 미만은 버림)
            iou_threshold: NMS IoU 임계값
            frame_idx: 영상 프레임 번호 (시간 계산용, 선택)

        Returns:
            Detection 객체 리스트
        """
        results = self.model.predict(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )

        detections = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            cls_id = int(cls_id)
            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=self.class_names.get(cls_id, f"unknown_{cls_id}"),
                    confidence=float(conf),
                    box_xyxy=tuple(box.tolist()),
                    frame_idx=frame_idx,
                )
            )

        return detections

    def detect_filtered(
        self,
        frame: np.ndarray,
        target_class_id: int,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        frame_idx: Optional[int] = None,
    ) -> List[Detection]:
        """detect()를 수행하되, 지정한 target_class_id에 해당하는 탐지만 반환."""
        all_detections = self.detect(
            frame,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            frame_idx=frame_idx,
        )
        return [d for d in all_detections if d.class_id == target_class_id]

    def total_person_count(self, detections: List[Detection]) -> int:
        """전체 탐지 인원 수 (클래스 무관)."""
        return len(detections)


def class_id_from_filters(top_bright: bool, bottom_bright: bool) -> int:
    """
    상의/하의 밝기 선택(bool)을 받아 해당하는 class_id를 반환.

    Args:
        top_bright: True면 상의 밝음, False면 어두움
        bottom_bright: True면 하의 밝음, False면 어두움
    """
    mapping = {
        (True, True): 0,    # bright_top_bright_bottom
        (True, False): 1,   # bright_top_dark_bottom
        (False, True): 2,   # dark_top_bright_bottom
        (False, False): 3,  # dark_top_dark_bottom
    }
    return mapping[(top_bright, bottom_bright)]
