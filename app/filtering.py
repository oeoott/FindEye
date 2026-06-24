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
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image

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
    crop_image: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class Track:
    """단일 사람 트랙을 저장하는 데이터 클래스."""
    track_id: int
    last_box: tuple
    last_seen_frame: int
    age: int
    feature: Optional[np.ndarray]
    first_detection: Detection
    first_detected_time: Optional[datetime]
    first_crop_image: Optional[np.ndarray]
    best_target_detection: Optional[Detection]
    best_target_detected_time: Optional[datetime]
    best_target_crop_image: Optional[np.ndarray]
    best_target_confidence: float
    location: str
    video_file: str


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


def _box_iou(box_a: tuple, box_b: tuple) -> float:
    """두 박스의 IoU를 계산한다."""
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b

    inter_x1 = max(x1_a, x1_b)
    inter_y1 = max(y1_a, y1_b)
    inter_x2 = min(x2_a, x2_b)
    inter_y2 = min(y2_a, y2_b)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, x2_a - x1_a) * max(0.0, y2_a - y1_a)
    area_b = max(0.0, x2_b - x1_b) * max(0.0, y2_b - y1_b)

    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class AppearanceEncoder:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if hasattr(torchvision.models, 'ResNet50_Weights') else None
        if weights is not None:
            base_model = torchvision.models.resnet50(weights=weights)
        else:
            base_model = torchvision.models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(base_model.children())[:-1]).to(self.device)
        self.model.eval()
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def encode(self, crop: np.ndarray) -> Optional[np.ndarray]:
        if crop is None or crop.size == 0:
            return None

        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feature = self.model(tensor)
            feature = feature.squeeze().cpu().numpy()
            norm = np.linalg.norm(feature)
            if norm > 0:
                feature = feature / norm
            return feature
        except Exception:
            return None


_GLOBAL_APPEARANCE_ENCODER: Optional[AppearanceEncoder] = None


def get_appearance_encoder() -> AppearanceEncoder:
    global _GLOBAL_APPEARANCE_ENCODER
    if _GLOBAL_APPEARANCE_ENCODER is None:
        _GLOBAL_APPEARANCE_ENCODER = AppearanceEncoder()
    return _GLOBAL_APPEARANCE_ENCODER


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    dot = float(np.dot(a, b))
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


def _match_tracks(
    tracks: List[Track],
    detections: List[Detection],
    detection_features: List[Optional[np.ndarray]],
    iou_threshold: float = 0.3,
    appearance_threshold: float = 0.45,
):
    """트랙과 현재 프레임 검출을 IoU + appearance 유사도로 매칭한다."""
    matches = []
    unmatched_tracks = set(range(len(tracks)))
    unmatched_detections = set(range(len(detections)))

    if not tracks or not detections:
        return matches, unmatched_tracks, unmatched_detections

    match_scores = []
    for t_idx, track in enumerate(tracks):
        for d_idx, det in enumerate(detections):
            iou_value = _box_iou(track.last_box, det.box_xyxy)
            appearance_score = 0.0
            if track.feature is not None and detection_features[d_idx] is not None:
                appearance_score = _cosine_similarity(track.feature, detection_features[d_idx])

            if iou_value >= iou_threshold or appearance_score >= appearance_threshold:
                score = max(iou_value, appearance_score)
                match_scores.append((score, t_idx, d_idx, iou_value, appearance_score))

    match_scores.sort(key=lambda item: item[0], reverse=True)
    for score, t_idx, d_idx, iou_value, appearance_score in match_scores:
        if t_idx not in unmatched_tracks or d_idx not in unmatched_detections:
            continue
        matches.append((t_idx, d_idx))
        unmatched_tracks.remove(t_idx)
        unmatched_detections.remove(d_idx)

    return matches, unmatched_tracks, unmatched_detections


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

    appearance_encoder = get_appearance_encoder()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"영상을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracks: List[Track] = []
    finished_tracks: List[Track] = []
    next_track_id = 1
    total_detected_persons = 0
    frame_idx = 0
    max_track_age = 3

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride == 0:
                all_detections = detector.detect(frame, conf_threshold=conf_threshold, frame_idx=frame_idx)
                total_detected_persons += len(all_detections)

                all_features = [appearance_encoder.encode(det.crop(frame)) for det in all_detections]
                matches, unmatched_tracks, unmatched_detections = _match_tracks(
                    tracks,
                    all_detections,
                    all_features,
                    iou_threshold=0.3,
                    appearance_threshold=0.45,
                )

                for t_idx in unmatched_tracks:
                    tracks[t_idx].age += 1

                for t_idx, d_idx in matches:
                    track = tracks[t_idx]
                    det = all_detections[d_idx]
                    feature = all_features[d_idx]
                    track.last_box = det.box_xyxy
                    track.last_seen_frame = frame_idx
                    track.age = 0
                    if feature is not None:
                        if track.feature is None:
                            track.feature = feature
                        else:
                            updated = track.feature * 0.8 + feature * 0.2
                            norm = np.linalg.norm(updated)
                            track.feature = updated / norm if norm > 0 else updated

                    if det.class_id == target_class_id and det.confidence >= track.best_target_confidence:
                        track.best_target_detection = det
                        track.best_target_confidence = det.confidence
                        track.best_target_detected_time = compute_detected_time(metadata.start_time, frame_idx, fps) if metadata.matched or metadata.start_time else None
                        track.best_target_crop_image = det.crop(frame)

                for d_idx in unmatched_detections:
                    det = all_detections[d_idx]
                    feature = all_features[d_idx]
                    detected_time = (
                        compute_detected_time(metadata.start_time, frame_idx, fps)
                        if metadata.matched or metadata.start_time
                        else None
                    )
                    best_target_detection = det if det.class_id == target_class_id else None
                    best_target_detected_time = detected_time if det.class_id == target_class_id else None
                    best_target_crop_image = det.crop(frame) if det.class_id == target_class_id else None
                    best_target_confidence = det.confidence if det.class_id == target_class_id else 0.0
                    tracks.append(
                        Track(
                            track_id=next_track_id,
                            last_box=det.box_xyxy,
                            last_seen_frame=frame_idx,
                            age=0,
                            feature=feature,
                            first_detection=det,
                            first_detected_time=detected_time,
                            first_crop_image=det.crop(frame),
                            best_target_detection=best_target_detection,
                            best_target_detected_time=best_target_detected_time,
                            best_target_crop_image=best_target_crop_image,
                            best_target_confidence=best_target_confidence,
                            location=metadata.location,
                            video_file=metadata.video_file,
                        )
                    )
                    next_track_id += 1

                active_tracks: List[Track] = []
                for track in tracks:
                    if track.age <= max_track_age:
                        active_tracks.append(track)
                    else:
                        finished_tracks.append(track)
                tracks = active_tracks

                if progress_callback:
                    progress_callback(frame_idx, total_frames)

                if max_candidates:
                    target_tracks = sum(1 for track in tracks if track.best_target_detection is not None)
                    if target_tracks >= max_candidates:
                        break

            frame_idx += 1
    finally:
        cap.release()

    candidates = []
    for track in tracks + finished_tracks:
        if track.first_detection.class_id != target_class_id:
            continue
        candidates.append(
            Candidate(
                detection=track.first_detection,
                location=track.location,
                detected_time=track.first_detected_time,
                video_file=track.video_file,
                crop_image=track.first_crop_image,
            )
        )

    # 후보 병합: 동일 인물로 보이는 항목은 최초 등장 후보만 남긴다.
    deduped = _dedupe_candidates(candidates, iou_threshold=0.3, hist_threshold=0.7)

    return deduped, total_detected_persons


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


def _dedupe_candidates(candidates: List[Candidate], iou_threshold: float = 0.3, hist_threshold: float = 0.7) -> List[Candidate]:
    """후보 리스트에서 같은 사람으로 판단되는 항목들을 병합하여 최초 등장 후보만 남긴다.

    병합 기준: 박스 IoU >= iou_threshold 또는 크롭 HSV 히스토그램 상관도 >= hist_threshold
    """
    if not candidates:
        return []

    # 등장 시간이 빠른 순으로 정렬해서 먼저 등장한 후보를 우선 유지
    order = list(range(len(candidates)))
    order.sort(key=lambda i: candidates[i].detected_time or datetime.min)

    used = [False] * len(candidates)
    unique = []

    # 사전 계산: 히스토그램 (있을 때만)
    hists = [None] * len(candidates)
    for idx, c in enumerate(candidates):
        img = c.crop_image
        if img is None or getattr(img, "size", 0) == 0:
            continue
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            hists[idx] = hist.flatten()
        except Exception:
            hists[idx] = None

    for i in order:
        if used[i]:
            continue
        base = candidates[i]
        used[i] = True
        for j in order:
            if used[j] or j == i:
                continue
            other = candidates[j]

            # IoU
            try:
                iou_val = _box_iou(base.detection.box_xyxy, other.detection.box_xyxy)
            except Exception:
                iou_val = 0.0

            hist_sim = 0.0
            if hists[i] is not None and hists[j] is not None:
                try:
                    hist_sim = cv2.compareHist(hists[i].astype('float32'), hists[j].astype('float32'), cv2.HISTCMP_CORREL)
                except Exception:
                    hist_sim = 0.0

            if iou_val >= iou_threshold or hist_sim >= hist_threshold:
                used[j] = True

        unique.append(base)

    return unique