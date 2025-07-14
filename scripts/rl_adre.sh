export NCCL_DEBUG=INFO
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1

BASE_PATH="./"
MODEL_PATH="$BASE_PATH/model/r1_15b_adre_sft/checkpoint-250"
TRAIN_DATA_PATH="$BASE_PATH/data/rl_math/rl_train_data.parquet"
EVAL_DATA_PATH="$BASE_PATH/data/rl_math/rl_eval_data.parquet"
OUTPUT_DIR="$BASE_PATH/model/adre_adre_rk16"


export PYTHONPATH="$PWD:$PYTHONPATH"

TOTAL_BATCH_SIZE=128
TOTAL_MINI_BATCH_SIZE=128
NUM_GPU=4
MICRO_BATCH_SIZE_PER_GPU=1

REMAINDER_1=$(($TOTAL_BATCH_SIZE % $NUM_GPU))
REMAINDER_2=$(($TOTAL_MINI_BATCH_SIZE % $NUM_GPU))
MINI_BATCH_SIZE_PER_GPU=$(($TOTAL_MINI_BATCH_SIZE/$NUM_GPU))
REMAINDER_3=$(($MINI_BATCH_SIZE_PER_GPU % $MICRO_BATCH_SIZE_PER_GPU))

if [ $REMAINDER_1 -eq 0 ] && [ $REMAINDER_2 -eq 0 ] && [ $REMAINDER_3 -eq 0 ]; then
    BATCH_SIZE_PER_GPU=$(($TOTAL_BATCH_SIZE / $NUM_GPU))
    echo "BATCH_SIZE_PER_GPU $BATCH_SIZE_PER_GPU"
    echo "MINI_BATCH_SIZE_PER_GPU $MINI_BATCH_SIZE_PER_GPU"
    ACCUMULATION_STEPS=$(($MINI_BATCH_SIZE_PER_GPU / $MICRO_BATCH_SIZE_PER_GPU))
    echo "ACCUMULATION_STEPS $ACCUMULATION_STEPS"
    echo "MICRO_BATCH_SIZE_PER_GPU $MICRO_BATCH_SIZE_PER_GPU"
    python -m torch.distributed.run --nproc_per_node=$NUM_GPU --nnodes=1 adre/rl_main.py --model $MODEL_PATH \
        --dataset $TRAIN_DATA_PATH \
        --eval_dataset $EVAL_DATA_PATH \
        --save_dir $OUTPUT_DIR \
        --eval_step 5 \
        --save_step 10 \
        --system_prompt "Please reason step by step, and put your final answer within \\boxed{}." \
        --type_key data_source \
        --logging_step 1 \
        --lr 1e-6 \
        --episodes 3 \
        --batch_size $BATCH_SIZE_PER_GPU \
        --micro_batch_size_per_gpu $MICRO_BATCH_SIZE_PER_GPU \
        --gradient_accumulation_steps $ACCUMULATION_STEPS \
        --use_kl_loss \
        --use_kl_estimator_k3 \
        --init_kl_coef 1e-4 \
        --reward_std \
        --entropy_coeff 1e-3 \
        --token_level_loss \
        --bf16 \
        --rollout_batch_size 128 \
        --log_prob_batch_size 4 \
        --rollout_n 8 \
        --rollout_n_max 16 \
        --rollout_n_min 2 \
        --max_new_tokens 4096 \
        --temperature 0.6 \
        --clip_eps_high=0.28 \
        --use_adapter \
        --length_penalty_alpha 1e-3


else
    echo "ERROR"
fi

