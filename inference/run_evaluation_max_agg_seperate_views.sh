TEST_DATA=(dl19 dl20 trec-covid nfcorpus signal news robust04 scifact touche dbpedia-entity)

VIEW_INDICES=(0 1 2 3)

for single_view_idx in ${VIEW_INDICES[@]}; do
  # evaluate data on 4-view model with max aggregation
  for data in ${TEST_DATA[@]}; do
    CUDA_VISIBLE_DEVICES=0 python3 ./evaluation.py \
      --input_path ./eval_data/${data}.jsonl \
      --output_path ./outputs/max_aggregation/view_${single_view_idx}/mvp-${data}.jsonl \
      --topk 100 \
      --n_special_tokens 4 \
      --aggregation_strategy single_view \
      --single_view_index $single_view_idx \
      --model_path ../checkpoints/aggregation/max/MVP/tfmr_0_step30000
  done
done