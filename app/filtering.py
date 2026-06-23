"""
app/filtering.py

영상 업로드 -> 메타데이터(장소/시작시간) 자동 매칭 -> 프레임별 YOLO 탐지 ->
사용자가 선택한 조건(상의/하의 밝기)에 맞는 후보만 필터링 -> 탐지 시각 계산

까지의 파이프라인을 담당하는 모듈. streamlit_app.py에서 이 모듈을 가져다 쓴다.

메타데이터 매칭 방식:
    영상 파일명을 키로 하여 video_metadata.csv에서 location/start_time을 조회한다.
    매칭되는 항목이 없으면 None을 반환하며, 호출 측(streamlit_app.py)에서
    사용자가 직접 장소/시간을 입력하도록 안내한다.
"""

import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import cv2
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from model.inference import FindEyeDetector, Detection, class_id_from_filters  # noqa: E402

DEFAULT_METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "video_metadata.csv"

# 영상 전체를 다 돌면 느리므로, 기본값으로 N프레임마다 한 번씩만 탐지 수행
DEFAULT_FRAME_STRIDE = 5


@dataclass
class VideoMetadata:
    """영상 한 건에 대한 메타데이터 (장소/시작시간)."""
    video_file: str
    location: str
    start_time: datetime
    matched: bool = True  # False면 메타데이터 매칭 실패 -> 사용자 직접 입력 필요


@dataclass
class Candidate:
    """필터링을 통과한 최종 후보 한 명."""
    detection: Detection
    location: str
    detected_time: Optional[datetime]
    video_file: str
    crop_image: "object" = field(default=None, repr=False)  # np.ndarray, 순환 typing 방지용


def load_metadata_table(metadata_path: Optional[str] = None) -> pd.DataFrame:
    """video_metadata.csv를 읽어온다. 없으면 빈 DataFrame 반환."""
    path = Path(metadata_path) if metadata_path else DEFAULT_METADATA_PATH
    if not path.exists():
        return pd.DataFrame(columns=["video_file", "location", "start_time"])

    df = pd.read_csv(path)
    df["start_time"] = pd.to_datetime(df["start_time"])
    return df


def match_metadata(video_filename: str, metadata_df: pd.DataFrame) -> VideoMetadata:
    """
    업로드된 영상 파일명을 기준으로 메타데이터 테이블에서 장소/시작시간을 찾는다.

    매칭 실패 시 matched=False와 함께 현재 시각을 기본값으로 채워 반환한다.
    (실제 사용 시 streamlit_app.py에서 matched=False면 사용자 입력 폼을 보여준다.)
    """
    row = metadata_df[metadata_df["video_file"] == video_filename]

    if row.empty:
        return VideoMetadata(
            video_file=video_filename,
            location="(알 수 없음 - 직접 입력 필요)",
            start_time=datetime.now(),
            matched=False,
        )

    record = row.iloc[0]
    return VideoMetadata(
        video_file=video_filename,
        location=str(record["location"]),
        start_time=pd.to_datetime(record["start_time"]).to_pydatetime(),
        matched=True,
    )


def compute_detected_time(start_time: datetime, frame_idx: int, fps: float) -> datetime:
    """영상 시작시간 + 프레임 위치(초) 로 실제 탐지 시각을 계산."""
    if fps <= 0:
        fps = 30.0  # 비정상적인 fps 값 방어
    elapsed_seconds = frame_idx / fps
    return start_time + timedelta(seconds=float(elapsed_seconds))


def search_video(
    video_path: str,
    detector: FindEyeDetector,
    top_bright: bool,
    bottom_bright: bool,
    metadata: VideoMetadata,
    conf_threshold: float = 0.4,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    max_candidates: Optional[int] = None,
    progress_callback=None,
) -> tuple:
    """
    영상 전체를 프레임 단위로 순회하며 조건에 맞는 후보를 찾는다.

    Args:
        video_path: 로컬에 저장된 영상 파일 경로
        detector: FindEyeDetector 인스턴스 (미리 로드해서 재사용 권장)
        top_bright / bottom_bright: 상의/하의 밝기 조건 (True=밝음, False=어두움)
        metadata: match_metadata()로 얻은 VideoMetadata
        conf_threshold: confidence 임계값
        frame_stride: N프레임마다 한 번씩 탐지 수행 (속도/정확도 트레이드오프)
        max_candidates: 후보를 이 개수만큼 찾으면 조기 종료 (None이면 끝까지 탐색)
        progress_callback: callable(current_frame, total_frames) - 진행률 표시용 (Streamlit progress bar 등)

    Returns:
        (candidates: List[Candidate], total_detected_persons: int)
    """
    target_class_id = class_id_from_filters(top_bright, bottom_bright)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"영상을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    candidates: List[Candidate] = []
    total_detected_persons = 0
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride == 0:
                all_detections = detector.detect(frame, conf_threshold=conf_threshold, frame_idx=frame_idx)
                total_detected_persons += len(all_detections)

                matching = [d for d in all_detections if d.class_id == target_class_id]

                for det in matching:
                    detected_time = (
                        compute_detected_time(metadata.start_time, frame_idx, fps)
                        if metadata.matched or metadata.start_time
                        else None
                    )
                    candidates.append(
                        Candidate(
                            detection=det,
                            location=metadata.location,
                            detected_time=detected_time,
                            video_file=metadata.video_file,
                            crop_image=det.crop(frame),
                        )
                    )

                if progress_callback:
                    progress_callback(frame_idx, total_frames)

                if max_candidates and len(candidates) >= max_candidates:
                    break

            frame_idx += 1
    finally:
        cap.release()

    return candidates, total_detected_persons


def candidates_to_dataframe(candidates: List[Candidate]) -> pd.DataFrame:
    """후보 리스트를 결과 테이블(DataFrame)로 변환 (보고서/다운로드용)."""
    rows = []
    for c in candidates:
        rows.append({
            "video_file": c.video_file,
            "location": c.location,
            "detected_time": c.detected_time.strftime("%Y-%m-%d %H:%M:%S") if c.detected_time else "N/A",
            "class_name": c.detection.class_name,
            "confidence": round(c.detection.confidence, 4),
            "frame_idx": c.detection.frame_idx,
        })
    return pd.DataFrame(rows)