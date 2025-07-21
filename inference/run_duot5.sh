#  duot5-base-msmarco 

# python duto_eval.

BEIR_DATA=(trec-covid signal robust04 dbpedia-entity)
for data in ${BEIR_DATA[@]}; do
    CUDA_VISIBLE_DEVICES=0 python duot5_eval.py \
        --test_path eval_data/baseline/${data}.jsonl \
        --output_path ./outputs/duot5-${data}.jsonl \
        --dataname ${data} \
        --outname duot5-${data} \
        --sub_mode nobsize \
        --bsize 32 \

done


