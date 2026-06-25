# FindEye

AI 기반 인물 속성 필터링을 통한 실종자 골든타임 확보 서비스

CCTV 영상에서 인상착의(상의/하의 밝기) 조건에 맞는 사람을 자동으로 찾아, 탐지된 장소와 시각을 함께 제시한다. 수백 시간 분량의 영상을 사람이 직접 돌려보는 대신, 신고된 옷차림 정보만으로 후보를 추려 초기 수색 시간을 줄이는 것이 목표다.

## 동작 방식

1. CCTV 영상 업로드
2. 파일명 기준으로 `data/video_metadata.csv`에서 촬영 장소·시작 시간 자동 매칭 (없으면 직접 입력)
3. 상의/하의 밝기 조건 선택 (밝음/어두움)
4. YOLO로 프레임 단위 탐지 후 조건에 맞는 후보만 필터링
5. 동일 인물 중복 제거 — 트래킹 + IoU·색상 히스토그램 유사도로 같은 사람의 반복 검출을 하나로 병합
6. 후보별 crop 이미지 + 장소 + 탐지 시각 + 신뢰도 출력, 전체 탐지 인원 수 요약

탐지 시각은 `영상 시작 시간 + (프레임 번호 / fps)`로 계산한다.

## 클래스

상의·하의 밝기 조합 4개 클래스로 분류한다.

| ID | 클래스명 | 의미 |
|----|----------|------|
| 0 | bright_top_bright_bottom | 상의 밝음 / 하의 밝음 |
| 1 | bright_top_dark_bottom | 상의 밝음 / 하의 어두움 |
| 2 | dark_top_bright_bottom | 상의 어두움 / 하의 밝음 |
| 3 | dark_top_dark_bottom | 상의 어두움 / 하의 어두움 |

## 모델

- 베이스: YOLOv11 (Ultralytics 8.3.0)
- 데이터셋: Roboflow `findeye_osp` v1 (train 903장 / valid 77장)
- 실험 과정: 모델 크기 비교(n/s/m), learning rate, 밝기 증강(hsv_v) 순으로 조정
- 최종 모델: `yolo11s`, 100 epochs, `lr0=0.005`, `hsv_v=0.2`, `imgsz=640`, `batch=16`
  - 명도가 분류 기준이므로 밝기 증강은 낮게 유지, Recall을 우선해 `s` 크기 선택

### 최종 성능 (valid 기준)

| Metric | Score |
|--------|-------|
| mAP50 | 0.987 |
| mAP50-95 | 0.777 |
| Recall | 0.976 |

실험별 상세 비교는 `results/experiments_summary.csv`, 클래스 분포는 `results/dataset_summary.csv` 참고.

## 프로젝트 구조

```
FindEye/
├── notebooks/
│   ├── 01_baseline.ipynb          # 베이스라인 학습 (yolo11n, 50e)
│   ├── 02_model_size.ipynb        # 모델 크기 비교 (n/s/m)
│   ├── 03_lr_aug_experiment.ipynb # lr / 밝기 증강 실험
│   └── 05_final_model.ipynb       # 최종 모델 학습
├── utils/
│   ├── check_dataset.py           # 데이터셋 구조·라벨 무결성 검증
│   ├── count_labels.py            # 클래스별 라벨 개수 집계
│   └── collect_results.py         # 실험 results.csv 비교 요약
├── model/
│   └── inference.py               # 학습 가중치 로드 + 탐지 래퍼
├── app/
│   ├── filtering.py               # 영상 순회 + 조건 필터링 + 중복 제거 + 시각 계산
│   ├── visualize.py               # 박스/crop 시각화 유틸
│   └── streamlit_app.py           # 메인 웹 앱
├── data/
│   └── video_metadata.csv         # 영상 파일명 ↔ 장소·시작시간 매핑
├── tests/
│   └── test_filtering.py          # 메타데이터 매칭·중복 제거 단위 테스트
├── weights/
│   └── best.pt                    # 최종 학습 가중치
├── results/                       # 학습 결과·통계
├── data.yaml
└── requirements.txt
```

## 설치 및 실행

Python 3.10 기준으로 동작을 확인했다.

```bash
pip install -r requirements.txt
```

`weights/best.pt`가 있는 상태에서 웹 앱 실행:

```bash
streamlit run app/streamlit_app.py
```

## 영상 메타데이터 등록

업로드한 영상의 파일명이 `data/video_metadata.csv`에 있으면 장소·시작 시간이 자동으로 채워진다. 없으면 앱에서 직접 입력하면 된다.

CSV 형식은 다음과 같다.

```csv
video_file,location,start_time
park-cctv.mp4,공원,2026-06-21 14:25:00
market-cctv.mov,시장,2026-06-17 11:37:00
```

- `video_file`: 업로드할 영상 파일명 (확장자 포함, 정확히 일치해야 매칭됨)
- `location`: 촬영 장소
- `start_time`: 촬영 시작 시각 (`YYYY-MM-DD HH:MM:SS`) — 탐지 시각 계산의 기준점

## 테스트

메타데이터 매칭과 중복 제거 로직에 대한 단위 테스트를 제공한다. 프로젝트 루트에서 실행한다.

```bash
python -m unittest discover -s tests
```

## 데이터셋 검증·통계

```bash
python utils/check_dataset.py --data_dir /path/to/dataset
python utils/count_labels.py --data_dir /path/to/dataset
```

## 학습 실험 결과 비교

```bash
python utils/collect_results.py --runs_dir runs/detect
```

## 라이선스

이 프로젝트는 MIT License를 따른다. Copyright (c) 2026 oeoott.

저작권 고지와 라이선스 전문을 포함하는 조건 아래, 사용·복제·수정·배포·재라이선스 및 판매가 자유롭게 허용된다. 단, 소프트웨어는 어떠한 보증도 없이 "있는 그대로(as is)" 제공되며, 사용으로 인해 발생하는 문제에 대해 저작권자는 책임지지 않는다.

전체 내용은 [LICENSE](LICENSE) 파일을 참고.
