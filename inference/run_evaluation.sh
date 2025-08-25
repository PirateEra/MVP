TEST_DATA=(dl19 dl20)
for data in ${TEST_DATA[@]}; do
    CUDA_VISIBLE_DEVICES=0 python3 evaluation.py --input_path /home/tako/kjun/MVT5_inference/eval_data/baseline/${data}.jsonl \
        --output_path ./outputs/mvp-${data}.jsonl --topk 100 \
        --n_special_tokens 4 \
        --model_path /data/kjun/checkpoints/EMNLP_FINAL_CHECKPOINT/base \
        --measure_flops
done
