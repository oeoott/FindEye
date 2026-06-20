"""
collect_results.py

여러 YOLO 학습 실험(runs/detect/<exp_name>/results.csv)을 모아
최종 epoch 기준 성능 지표(Precision, Recall, mAP50, mAP50-95)를 비교 요약한다.

03_lr_aug_experiment.ipynb 에서 진행한 실험들
(baseline, lr=0.005, brightness augmentation, 최종 100epoch 모델)을
노트북 밖에서도 재사용/재집계할 수 있도록 스크립트로 분리한 버전.

사용법:
    # 기본: 아래 EXPERIMENT_PATHS에 정의된 실험들을 자동으로 찾아 비교
    python utils/collect_results.py

    # runs 디렉토리를 직접 지정 (Colab 외 환경에서 결과물을 옮겨온 경우)
    python utils/collect_results.py --runs_dir /path/to/runs/detect --out results/experiments_summary.csv

    # 실험 폴더를 glob 패턴으로 전체 자동 탐색
    python utils/collect_results.py --auto_discover --runs_dir /path/to/runs/detect
"""

import argparse
import glob
from pathlib import Path

import pandas as pd

# 03_lr_aug_experiment.ipynb 기준 실험 이름 -> 폴더 glob 패턴 매핑
# 실제 경로는 Colab 환경(드라이브 마운트 위치 등)에 따라 달라질 수 있으므로
# --runs_dir 인자로 루트를 바꿔가며 사용
DEFAULT_EXPERIMENTS = {
    "0. Baseline (50e)": "weights",  # /content/drive/MyDrive/FindEye/weights
    "1. LR 0.005 (50e)": "lr_0005_exp*",
    "2. Aug 0.2 (50e)": "aug_brightness_exp*",
    "3. Baseline (100e)": "baseline_100e*",
    "4. Optimized LR (100e)": "optimized_100e*",
    "5. 최종 (LR0.005 + Aug0.2, 100e)": "final_100epoch_exp*",
}

METRIC_COLUMNS = {
    "metrics/precision(B)": "Precision",
    "metrics/recall(B)": "Recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
}


def find_results_csv(runs_dir: Path, pattern: str):
    """패턴에 맞는 실험 폴더들 중 results.csv가 있는 가장 최근 것을 찾는다."""
    candidates = sorted(glob.glob(str(runs_dir / pattern)))
    for folder in reversed(candidates):  # 최신 실험(번호가 큰 것)부터 확인
        csv_path = Path(folder) / "results.csv"
        if csv_path.exists():
            return csv_path
    return None


def summarize_experiment(exp_name: str, csv_path: Path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    final_row = df.iloc[-1]

    summary = {"실험명": exp_name, "Total Epochs": int(final_row.name) + 1}
    for raw_col, label in METRIC_COLUMNS.items():
        if raw_col in df.columns:
            summary[label] = round(float(final_row[raw_col]), 4)
        else:
            summary[label] = None

    return summary, df


def auto_discover_experiments(runs_dir: Path):
    """runs_dir 바로 아래 모든 하위 폴더 중 results.csv가 있는 것을 전부 실험으로 취급."""
    experiments = {}
    for folder in sorted(runs_dir.iterdir()):
        if folder.is_dir() and (folder / "results.csv").exists():
            experiments[folder.name] = folder.name
    return experiments


def main():
    parser = argparse.ArgumentParser(description="FindEye 실험 결과 수집/비교")
    parser.add_argument(
        "--runs_dir", type=str, default="runs/detect",
        help="YOLO 실험 결과들이 모여 있는 상위 디렉토리 (예: runs/detect)"
    )
    parser.add_argument(
        "--out", type=str, default="results/experiments_summary.csv",
        help="요약 결과 csv 저장 경로"
    )
    parser.add_argument(
        "--auto_discover", action="store_true",
        help="DEFAULT_EXPERIMENTS 대신 runs_dir 하위 모든 실험 폴더를 자동 탐색"
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"❌ runs_dir이 존재하지 않습니다: {runs_dir}")
        print("   --runs_dir 옵션으로 실제 실험 결과 경로를 지정해주세요.")
        return

    experiments = (
        auto_discover_experiments(runs_dir) if args.auto_discover else DEFAULT_EXPERIMENTS
    )

    summaries = []
    not_found = []

    for exp_name, pattern in experiments.items():
        csv_path = find_results_csv(runs_dir, pattern)
        if csv_path is None:
            not_found.append(exp_name)
            continue
        summary, _ = summarize_experiment(exp_name, csv_path)
        summaries.append(summary)
        print(f"✅ [{exp_name}] -> {csv_path}")

    if not_found:
        print(f"\n⚠️  결과를 찾지 못한 실험 ({len(not_found)}개):")
        for name in not_found:
            print(f"   - {name}")

    if not summaries:
        print("\n❌ 수집된 실험 결과가 없습니다. --runs_dir 경로를 확인해주세요.")
        return

    result_df = pd.DataFrame(summaries).set_index("실험명")

    print(f"\n{'=' * 70}")
    print("📊 실험 결과 요약 (최종 epoch 기준)")
    print("=" * 70)
    print(result_df.to_string())

    # 최고 mAP50 모델 하이라이트
    if "mAP50" in result_df.columns and result_df["mAP50"].notna().any():
        best_exp = result_df["mAP50"].idxmax()
        best_score = result_df["mAP50"].max()
        print(f"\n🏆 최고 mAP50 모델: {best_exp} (mAP50={best_score})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, encoding="utf-8-sig")
    print(f"\n✅ 결과 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
