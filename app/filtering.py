"""
app/filtering.py

영상 업로드 -> 메타데이터(장소/시작시간) 자동 매칭 -> 프레임별 YOLO 탐지 ->
사용자가 선택한 조건(상의/하의 밝기)에 맞는 후보만 필터링 -> 탐지 시각 계산

수정사항:
- 신뢰도 0.5 미만 후보 자동 제외 (MIN_CONFIDENCE)
- IoU 기반 동일 인물 중복 제거 (is_duplicate)
"""

import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from model.inference import FindEyeDetector, Detection, class_id_from_filters  # noqa: E402

DEFAULT_METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "video_metadata.csv"
DEFAULT_FRAME_STRIDE = 8
MIN_CONFIDENCE = 0.5        # ★ 신뢰도 0.5 미만 후보 제외
IOU_DEDUP_THRESHOLD = 0.5   # ★ 동일 인물 중복 제거 IoU 임계값


@dataclass
class VideoMetadata:
    """영상 한 건에 대한 메타데이터 (장소/시작시간)."""
    video_file: str
    location: str
    start_time: datetime
    matched: bool = True


@dataclass
class Candidate:
    """필터링을 통과한 최종 후보 한 명."""
    detection: Detection
    location: str
    detected_time: Optional[datetime]
    video_file: str
    crop_image: "object" = field(default=None, repr=False)


def load_metadata_table(metadata_path: Optional[str] = None) -> pd.DataFrame:
    """video_metadata.csv를 읽어온다. 없으면 빈 DataFrame 반환."""
    path = Path(metadata_path) if metadata_path else DEFAULT_METADATA_PATH
    if not path.exists():
        return pd.DataFrame(columns=["video_file", "location", "start_time"])
    df = pd.read_csv(path)
    if "video_file" in df.columns:
        df["video_file"] = df["video_file"].fillna("").astype(str).str.strip()
    if "location" in df.columns:
        df["location"] = df["location"].fillna("").astype(str)
    if "start_time" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    return df


def match_metadata(video_filename: str, metadata_df: pd.DataFrame) -> VideoMetadata:
    """파일명 기준으로 메타데이터 매칭. 없으면 matched=False 반환."""
    if metadata_df is None or metadata_df.empty:
        return VideoMetadata(
            video_file=str(video_filename or ""),
            location="(알 수 없음 - 직접 입력 필요)",
            start_time=datetime.now(),
            matched=False,
        )

    if "video_file" not in metadata_df.columns:
        return VideoMetadata(
            video_file=str(video_filename or ""),
            location="(알 수 없음 - 직접 입력 필요)",
            start_time=datetime.now(),
            matched=False,
        )

    filename = str(video_filename or "").strip()
    normalized_values = [
        str(value).strip() if pd.notna(value) else ""
        for value in metadata_df["video_file"].tolist()
    ]

    record = None
    for idx, value in enumerate(normalized_values):
        if value == filename:
            record = metadata_df.iloc[idx]
            break

    if record is None:
        return VideoMetadata(
            video_file=filename,
            location="(알 수 없음 - 직접 입력 필요)",
            start_time=datetime.now(),
            matched=False,
        )
    start_time_value = record.get("start_time")
    parsed_timestamp = pd.to_datetime(start_time_value, errors="coerce")
    if pd.isna(parsed_timestamp):
        parsed_start_time = datetime.now()
    else:
        parsed_start_time = parsed_timestamp.to_pydatetime()

    return VideoMetadata(
        video_file=filename,
        location=str(record.get("location", "")),
        start_time=parsed_start_time,
        matched=True,
    )


def compute_detected_time(start_time: datetime, frame_idx: int, fps: float) -> datetime:
    """영상 시작시간 + 프레임 위치로 탐지 시각 계산."""
    if fps <= 0:
        fps = 30.0
    elapsed_seconds = frame_idx / fps
    return start_time + timedelta(seconds=float(elapsed_seconds))


def compute_iou(box1: tuple, box2: tuple) -> float:
    """★ 두 박스의 IoU 계산 (동일 인물 판단용)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def is_duplicate(new_box: tuple, existing_candidates: List["Candidate"]) -> bool:
    """★ 기존 후보들과 IoU가 높으면 동일 인물로 판단하여 제외."""
    for c in existing_candidates:
        if compute_iou(new_box, c.detection.box_xyxy) > IOU_DEDUP_THRESHOLD:
            return True
    return False


def search_video(
    video_path: str,
    detector: FindEyeDetector,
    top_bright: bool,
    bottom_bright: bool,
    metadata: VideoMetadata,
    conf_threshold: float = MIN_CONFIDENCE,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    max_candidates: Optional[int] = None,
    progress_callback=None,
) -> tuple:
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
                all_detections = detector.detect(
                    frame, conf_threshold=conf_threshold, frame_idx=frame_idx
                )
                total_detected_persons += len(all_detections)

                matching = [
                    d for d in all_detections
                    if d.class_id == target_class_id
                    and d.confidence >= MIN_CONFIDENCE  # ★ 신뢰도 필터
                ]

                for det in matching:
                    # ★ 동일 인물 중복 제거
                    if is_duplicate(det.box_xyxy, candidates):
                        continue

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
        
    candidates.sort(key=lambda c: c.detection.confidence, reverse=True)

    return candidates, total_detected_persons


def candidates_to_dataframe(candidates: List[Candidate]) -> pd.DataFrame:
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