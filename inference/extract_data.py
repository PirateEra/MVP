#!/usr/bin/env python
# extract_data.py
import re, argparse
from pathlib import Path
from functools import reduce
import pandas as pd

# ---------------------- 1) 인수 파싱 ------------------------------- #
parser = argparse.ArgumentParser(
    description="Parse ranking logs (ndcg@10) and build an Excel grid."
)
parser.add_argument(
    "--datasets", "-d", nargs="+", required=True,
    help="데이터셋 prefix 목록 (예: dl19 dl20 trec-covid)"
)
parser.add_argument(
    "--log_dir", "-l", type=Path, default=Path.cwd(),
    help="txt 로그들이 모여있는 폴더 (default: 현재 디렉터리)"
)
parser.add_argument(
    "--output", "-o", type=Path, default=Path("hyperparameter_results.xlsx"),
    help="Excel 저장 경로 (default: ./hyperparameter_results.xlsx)"
)
args = parser.parse_args()

# ---------------------- 2) 보조 함수 ------------------------------- #
def parse_model_path(path: str) -> dict:
    rex = lambda key: re.search(fr"{key}_([\d.]+)", path)
    return {k.upper(): float(rex(k).group(1)) for k in ("ortho", "warmup", "temp")}

def parse_log_file(txt_path: Path, tag: str) -> pd.DataFrame:
    rows, current_path = [], None
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MODEL"):
                current_path = line.split(":", 1)[1].strip()
            elif "ndcg@10" in line.lower():
                metric = float(line.split(":")[1])
                row = parse_model_path(current_path)
                row[tag.upper()] = metric
                rows.append(row)
    return pd.DataFrame(rows)

# ---------------------- 3) txt → DataFrame ------------------------- #
dfs = []
for name in args.datasets:
    file_path = args.log_dir / f"result_{name}.txt"
    if not file_path.exists():
        print(f"⚠️  {file_path} not found (skip)")
        continue
    dfs.append(parse_log_file(file_path, name))

if not dfs:
    raise SystemExit("❌  No valid txt files — 종료합니다.")

# ---------------------- 4) merge & 저장 --------------------------- #
df = reduce(lambda l, r: pd.merge(l, r, on=["ORTHO", "WARMUP", "TEMP"], how="outer"), dfs)
df = df.sort_values(["ORTHO", "WARMUP", "TEMP"]).reset_index(drop=True)

# Excel
with pd.ExcelWriter(args.output, engine="xlsxwriter") as w:
    df.to_excel(w, index=False, sheet_name="Results")
print(f"✅  saved → {args.output.resolve()}")

# 노트북에서 보면 바로 미리보기
try:
    from IPython.display import display
    display(df)
except ImportError:
    pass
