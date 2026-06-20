"""
check_dataset.py

YOLO 데이터셋(Roboflow 다운로드 결과물)의 구조와 무결성을 검증하는 스크립트.

검증 항목:
1. train/valid/test 폴더와 images/labels 하위 폴더 존재 여부
2. 이미지 파일 수와 라벨 파일 수 일치 여부
3. 라벨이 없는 이미지(누락) / 이미지가 없는 라벨(고아 파일) 탐지
4. 라벨 파일 형식 검증 (class_id x_center y_center width height, 0~1 범위)
5. data.yaml의 클래스 수(nc)와 실제 라벨에 등장하는 class_id 범위 일치 여부

사용법:
    python utils/check_dataset.py --data_dir /path/to/dataset
    python utils/check_dataset.py --data_dir /path/to/dataset --yaml data.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ["train", "valid", "test"]


def find_split_dirs(data_dir: Path, split: str):
    """Roboflow 표준 구조(split/images, split/labels)를 찾는다."""
    img_dir = data_dir / split / "images"
    lbl_dir = data_dir / split / "labels"
    return img_dir, lbl_dir


def list_files(folder: Path, extensions=None):
    if not folder.exists():
        return []
    if extensions:
        return sorted(
            [f for f in folder.iterdir() if f.suffix.lower() in extensions]
        )
    return sorted(folder.iterdir())


def validate_label_file(label_path: Path, num_classes: int):
    """라벨 파일 한 줄씩 형식 검증. 문제 목록을 반환."""
    issues = []
    try:
        lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    except Exception as e:
        return [f"읽기 실패: {e}"]

    if not lines:
        issues.append("빈 라벨 파일 (배경 이미지일 수 있음, 의도된 것인지 확인 필요)")
        return issues

    for i, line in enumerate(lines, start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            issues.append(f"  L{i}: 필드 개수가 5가 아님 -> '{line}'")
            continue

        cls_id_str, x, y, w, h = parts
        if not cls_id_str.lstrip("-").isdigit():
            issues.append(f"  L{i}: class_id가 정수가 아님 -> '{cls_id_str}'")
            continue

        cls_id = int(cls_id_str)
        if not (0 <= cls_id < num_classes):
            issues.append(
                f"  L{i}: class_id({cls_id})가 클래스 범위(0~{num_classes - 1})를 벗어남"
            )

        for name, val in [("x_center", x), ("y_center", y), ("width", w), ("height", h)]:
            try:
                v = float(val)
                if not (0.0 <= v <= 1.0):
                    issues.append(f"  L{i}: {name} 값({v})이 0~1 범위를 벗어남")
            except ValueError:
                issues.append(f"  L{i}: {name} 값이 숫자가 아님 -> '{val}'")

    return issues


def check_split(data_dir: Path, split: str, num_classes: int):
    print(f"\n{'=' * 60}")
    print(f"[{split}] 검사 중...")
    print("=" * 60)

    img_dir, lbl_dir = find_split_dirs(data_dir, split)

    if not img_dir.exists() and not lbl_dir.exists():
        print(f"  ⚠️  {split} 폴더 자체가 없음 (스킵)")
        return {"split": split, "skipped": True}

    images = list_files(img_dir, IMG_EXTENSIONS)
    labels = list_files(lbl_dir, {".txt"})

    image_stems = {f.stem for f in images}
    label_stems = {f.stem for f in labels}

    missing_labels = image_stems - label_stems   # 이미지는 있는데 라벨이 없음
    orphan_labels = label_stems - image_stems     # 라벨은 있는데 이미지가 없음

    print(f"  이미지 수: {len(images)}")
    print(f"  라벨 수:   {len(labels)}")

    if missing_labels:
        print(f"  ⚠️  라벨 누락된 이미지: {len(missing_labels)}개")
        for stem in sorted(missing_labels)[:5]:
            print(f"      - {stem}")
        if len(missing_labels) > 5:
            print(f"      ... 외 {len(missing_labels) - 5}개")
    else:
        print("  ✅ 모든 이미지에 라벨 존재")

    if orphan_labels:
        print(f"  ⚠️  대응 이미지 없는 라벨(고아 파일): {len(orphan_labels)}개")
        for stem in sorted(orphan_labels)[:5]:
            print(f"      - {stem}")
    else:
        print("  ✅ 고아 라벨 파일 없음")

    # 라벨 형식 검증
    format_issues = {}
    for lbl in labels:
        issues = validate_label_file(lbl, num_classes)
        if issues:
            format_issues[lbl.name] = issues

    if format_issues:
        print(f"  ⚠️  형식 문제가 있는 라벨 파일: {len(format_issues)}개")
        shown = 0
        for fname, issues in format_issues.items():
            if shown >= 3:
                print(f"      ... 외 {len(format_issues) - 3}개 파일에 문제 있음")
                break
            print(f"    [{fname}]")
            for issue in issues:
                print(f"      {issue}")
            shown += 1
    else:
        print("  ✅ 모든 라벨 형식 정상")

    return {
        "split": split,
        "skipped": False,
        "num_images": len(images),
        "num_labels": len(labels),
        "missing_labels": len(missing_labels),
        "orphan_labels": len(orphan_labels),
        "format_issues": len(format_issues),
    }


def main():
    parser = argparse.ArgumentParser(description="FindEye 데이터셋 검증")
    parser.add_argument("--data_dir", type=str, required=True, help="데이터셋 루트 경로 (train/valid/test 상위 폴더)")
    parser.add_argument("--yaml", type=str, default=None, help="data.yaml 경로 (기본: data_dir/data.yaml)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ 데이터셋 경로가 존재하지 않습니다: {data_dir}")
        sys.exit(1)

    yaml_path = Path(args.yaml) if args.yaml else data_dir / "data.yaml"
    if not yaml_path.exists():
        print(f"❌ data.yaml을 찾을 수 없습니다: {yaml_path}")
        sys.exit(1)

    with open(yaml_path, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    num_classes = data_config.get("nc")
    class_names = data_config.get("names")
    print(f"📋 data.yaml 로드 완료: nc={num_classes}, names={class_names}")

    results = []
    for split in SPLITS:
        result = check_split(data_dir, split, num_classes)
        results.append(result)

    # 최종 요약
    print(f"\n{'=' * 60}")
    print("📊 최종 요약")
    print("=" * 60)
    total_problems = 0
    for r in results:
        if r["skipped"]:
            continue
        problems = r["missing_labels"] + r["orphan_labels"] + r["format_issues"]
        total_problems += problems
        status = "✅ 이상 없음" if problems == 0 else f"⚠️  문제 {problems}건"
        print(f"  {r['split']:6s}: 이미지 {r['num_images']:4d} / 라벨 {r['num_labels']:4d}  -> {status}")

    if total_problems == 0:
        print("\n✅ 데이터셋 검증 통과: 학습 진행 가능합니다.")
    else:
        print(f"\n⚠️  총 {total_problems}건의 문제가 발견되었습니다. 위 내용을 확인 후 수정해주세요.")


if __name__ == "__main__":
    main()
