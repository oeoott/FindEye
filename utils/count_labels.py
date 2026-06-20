"""
count_labels.py

YOLO 데이터셋의 train/valid/test 각 split에서 클래스별 라벨(객체) 개수를 집계한다.
클래스 불균형 확인 및 보고서 작성용 통계 자료로 사용.

결과물:
- 콘솔에 split별 / 클래스별 카운트 테이블 출력
- results/dataset_summary.csv 로 저장 (보고서에 바로 첨부 가능한 형태)

사용법:
    python utils/count_labels.py --data_dir /path/to/dataset --yaml data.yaml
    python utils/count_labels.py --data_dir /path/to/dataset --yaml data.yaml --out results/dataset_summary.csv
"""

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

SPLITS = ["train", "valid", "test"]


def count_split_labels(data_dir: Path, split: str, class_names: dict):
    """split 하나의 라벨 폴더를 돌면서 클래스별 객체 수 + 이미지 수를 센다."""
    lbl_dir = data_dir / split / "labels"
    counts = defaultdict(int)
    num_images_with_labels = 0
    num_empty_labels = 0

    if not lbl_dir.exists():
        return counts, 0, 0

    label_files = sorted(lbl_dir.glob("*.txt"))

    for lbl_file in label_files:
        lines = lbl_file.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            num_empty_labels += 1
            continue

        num_images_with_labels += 1
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cls_id = int(parts[0])
            except ValueError:
                continue
            cls_name = class_names.get(cls_id, f"unknown_class_{cls_id}")
            counts[cls_name] += 1

    return counts, num_images_with_labels, num_empty_labels


def main():
    parser = argparse.ArgumentParser(description="FindEye 클래스별 라벨 개수 집계")
    parser.add_argument("--data_dir", type=str, required=True, help="데이터셋 루트 경로")
    parser.add_argument("--yaml", type=str, default=None, help="data.yaml 경로 (기본: data_dir/data.yaml)")
    parser.add_argument("--out", type=str, default="results/dataset_summary.csv", help="결과 csv 저장 경로")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    yaml_path = Path(args.yaml) if args.yaml else data_dir / "data.yaml"

    with open(yaml_path, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    raw_names = data_config.get("names")
    # data.yaml의 names가 list 형태로 올 수도, dict 형태로 올 수도 있어 둘 다 처리
    if isinstance(raw_names, list):
        class_names = {i: name for i, name in enumerate(raw_names)}
    else:
        class_names = {int(k): v for k, v in raw_names.items()}

    print(f"📋 클래스 목록: {class_names}\n")

    rows = []
    grand_total = defaultdict(int)

    for split in SPLITS:
        counts, num_labeled_images, num_empty = count_split_labels(data_dir, split, class_names)

        if not counts and num_labeled_images == 0 and num_empty == 0:
            print(f"[{split}] 폴더 없음 또는 라벨 없음, 스킵\n")
            continue

        print(f"{'=' * 50}")
        print(f"[{split}]")
        print("=" * 50)
        split_total = 0
        for cls_id in sorted(class_names.keys()):
            cls_name = class_names[cls_id]
            cnt = counts.get(cls_name, 0)
            split_total += cnt
            grand_total[cls_name] += cnt
            print(f"  {cls_name:30s}: {cnt:5d}")
            rows.append({"split": split, "class": cls_name, "count": cnt})

        print(f"  {'-' * 40}")
        print(f"  {'총 객체 수':30s}: {split_total:5d}")
        print(f"  {'라벨 있는 이미지 수':30s}: {num_labeled_images:5d}")
        print(f"  {'빈 라벨(배경) 이미지 수':30s}: {num_empty:5d}\n")

        rows.append({"split": split, "class": "TOTAL_OBJECTS", "count": split_total})
        rows.append({"split": split, "class": "IMAGES_WITH_LABELS", "count": num_labeled_images})
        rows.append({"split": split, "class": "EMPTY_LABEL_IMAGES", "count": num_empty})

    # 전체 합산
    print(f"{'=' * 50}")
    print("[전체 합산]")
    print("=" * 50)
    for cls_name, cnt in grand_total.items():
        print(f"  {cls_name:30s}: {cnt:5d}")
        rows.append({"split": "ALL", "class": cls_name, "count": cnt})

    total_all = sum(grand_total.values())
    print(f"  {'-' * 40}")
    print(f"  {'전체 총 객체 수':30s}: {total_all:5d}")
    rows.append({"split": "ALL", "class": "TOTAL_OBJECTS", "count": total_all})

    # 클래스 불균형 경고
    if grand_total:
        max_cls = max(grand_total, key=grand_total.get)
        min_cls = min(grand_total, key=grand_total.get)
        if grand_total[min_cls] > 0:
            ratio = grand_total[max_cls] / grand_total[min_cls]
            if ratio >= 2.0:
                print(
                    f"\n⚠️  클래스 불균형 주의: '{max_cls}'({grand_total[max_cls]}개) vs "
                    f"'{min_cls}'({grand_total[min_cls]}개), 비율 {ratio:.1f}배"
                )
        elif grand_total[min_cls] == 0:
            print(f"\n⚠️  '{min_cls}' 클래스는 라벨이 0개입니다. 데이터셋을 확인해주세요.")

    # CSV 저장
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 결과 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
