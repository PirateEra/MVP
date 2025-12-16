TEST_DATA=(dl19 dl20 trec-covid nfcorpus signal news robust04 scifact touche dbpedia-entity)

# evaluate data on 4-view model with max aggregation
for data in ${TEST_DATA[@]}; do
  CUDA_VISIBLE_DEVICES=0 python3 ./evaluation.py \
    --input_path ./eval_data/${data}.jsonl \
    --output_path ./outputs/smooth_max_aggregation/mvp-${data}.jsonl \
    --topk 100 \
    --n_special_tokens 4 \
    --aggregation_strategy smooth_max \
    --model_path ../checkpoints/aggregation/smooth_max/MVP/tfmr_0_step27000
done
