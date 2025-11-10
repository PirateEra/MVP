TEST_DATA=(dl19 dl20 trec-covid nfcorpus signal news robust04 scifact touche dbpedia-entity)
# TEST_DATA=(dl19 dl20)
# additional flags
# --model_path Jun421/MVP-3b
# --measure_flops 
for data in ${TEST_DATA[@]}; do
    CUDA_VISIBLE_DEVICES=0 python3 ./evaluation.py --input_path ./eval_data/${data}.jsonl \
        --output_path ./outputs/mvp-${data}.jsonl --topk 100 \
        --n_special_tokens 4 \
        --model_path Jun421/MVP-base
done
