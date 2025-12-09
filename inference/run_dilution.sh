TEST_DATA=(dl19 dl20)
#TEST_DATA=(dl19)
for data in ${TEST_DATA[@]}; do
    CUDA_VISIBLE_DEVICES=0 python3 ./anchor_dilution_test.py --input_path ./eval_data/${data}.jsonl \
        --topk 100 \
        --n_special_tokens 4 \
        --model_path Jun421/MVP-base
done
