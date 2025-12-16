#!/bin/bash

# ./compute_and_save_metrics.sh <input directory> <output directory> <file_prefix>

set -e
set -u

TEST_DATA=(dl19 dl20 trec-covid nfcorpus signal news robust04 scifact touche dbpedia-entity)

file_prefix="${3:-}"

for data in ${TEST_DATA[@]}; do
  echo "Running: data=${data}"

  mkdir -p "${2%/}"
  LOG_DIR="${2%/}"

  python ./beir_eval.py --path "${1%/}/${file_prefix}${data}.jsonl" 2>&1 | tee "${LOG_DIR}/${data}.txt"
done
