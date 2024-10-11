# Define the usage function to print the help message

usage() {
  echo "Usage: $0 [options] [positional arguments]"
  echo "Options:"
  echo "  --dataset_name <value>           Name of the dataset"
  echo "  --model_org <value>              Organization of the model"
  echo "  --model_name <value>             Name of the model"
  echo "  --qr_model_org <value>           Organization of the QR model"
  echo "  --qr_model_name <value>          Name of the QR model"
  echo "  --lm_target_epoch <value>        Epoch for LM target"
  echo "  --qr_validation_percent <value>  Validation Percent for QR model"
  echo "  --qr_score <value>               Score Function for QR model"
  echo "  --train_target                   Set TRAIN_TARGET to true"
  echo "  --pred_target                    Set PRED_TARGET to true"
  echo "  --run_mse_pinball                Set RUN_MSE_PINBALL to true"
  echo "  --run_gaussian                   Set RUN_GAUSSIAN to true"
  echo "  --run_iqr                        Set RUN_IQR to true"
  echo "  --run_gaussian_pinball           set RUN_GAUSSIAN_PINBALL to true"
  echo "  --train_shadow                   Set TRAIN_SHADOW to true"
  echo "  --pred_shadow                    Set PRED_SHADOW to true"
  echo "  --extend_task_pred               Set EXTEND_TASK_PRED to true"
  echo "  --shadow_start <value>           Index of Shadow start"
  echo "  --shadow_end <value>             Index of Shadow end"
  echo "  --use_deepspeed                  Set USE_DEEPSPEED to true"
  echo "  --deepspeed_level <value>        Level of deepspeed to use"
  echo "  --qr_lr <value>                  Learning rate of QR model"
  echo "  --seeds <value>                  Seeds for QR model training"
  echo "  --qr_train_batch_size <value>    Batch size for QR model training"
  echo "  --qr_full_batch_size <value>     Full batch size for qr model training"
  echo "  --prefix_dir <value>             Root directory for storing results"  
  echo "  -h, --help                       Display this help message"
}

# Define the long options
longopts="dataset_name:,model_org:,model_name:,qr_model_org:,qr_model_name:,lm_target_epoch:,qr_validation_percent:,qr_score:,train_target,pred_target,run_mse_pinball,run_gaussian,run_iqr,run_gaussian_pinball,train_shadow,pred_shadow,extend_task_pred,shadow_start:,shadow_end:,use_deepspeed,deepspeed_level:,qr_lr:,seeds:,qr_train_batch_size:,qr_full_batch_size:,prefix_dir:,help"


# Parse the command line arguments using getopt
args=$(getopt -a -o h --long $longopts -- "$@")

# Check if getopt command was successful
if [ "$?" -ne 0 ]; then
  usage
  exit 1
fi

# Evaluate the set of parsed arguments
eval set -- "$args"

# Initialize variables for options
# dataset name should be from wikitext, wikitext_sample, xsum, ag_news
DATASET_NAME=wikitext_sample
LM_TARGET_EPOCH=1

MODEL_ORG=facebook
MODEL_NAME=opt-6.7b

QR_MODEL_ORG=facebook
QR_MODEL_NAME=opt-125m

TRAIN_TARGET=false
PRED_TARGET=false
RUN_MSE_PINBALL=false
RUN_GAUSSIAN=false
RUN_IQR=false
RUN_GAUSSIAN_PINBALL=false
TRAIN_SHADOW=false
PRED_SHADOW=false
EXTEND_TASK_PRED=false

SHADOW_START=1
SHADOW_END=4

USE_DEEPSPEED=false
PRED_USE_DEEPSPEED=false
DEEPSPEED_LEVEL=3
USE_FP16=True
TORCH_DTYPE="float16"
QR_VALIDATION_PERCENT=5
QR_SCORE="nce"

QR_LR="2e-5"
SEEDS="42 1024 512 2048 256"
QR_TRAIN_BATCH_SIZE=8
QR_FULL_BATCH_SIZE=128
PREFIX_DIR="."

# Loop through the arguments and set the variables accordingly
while true; do
  case "$1" in
    --dataset_name)
      DATASET_NAME="$2"
      shift 2 ;;
    --model_org)
      MODEL_ORG="$2"
      shift 2 ;;
    --model_name)
      MODEL_NAME="$2"
      shift 2 ;;
    --qr_model_org)
      QR_MODEL_ORG="$2"
      shift 2 ;;
    --qr_model_name)
      QR_MODEL_NAME="$2"
      shift 2 ;;
    --lm_target_epoch)
      LM_TARGET_EPOCH="$2"
      shift 2 ;;
    --qr_validation_percent)
      QR_VALIDATION_PERCENT="$2"
      shift 2 ;;
    --qr_score)
      QR_SCORE="$2"
      shift 2 ;;
    --train_target)
      TRAIN_TARGET=true
      shift ;;
    --pred_target)
      PRED_TARGET=true
      shift ;;
    --run_mse_pinball)
      RUN_MSE_PINBALL=true
      shift ;;
    --run_gaussian)
      RUN_GAUSSIAN=true
      shift ;;
    --run_iqr)
      RUN_IQR=true
      shift ;;
    --run_gaussian_pinball)
      RUN_GAUSSIAN_PINBALL=true
      shift ;;
    --train_shadow)
      TRAIN_SHADOW=true
      shift ;;
    --pred_shadow)
      PRED_SHADOW=true
      shift ;;
    --extend_task_pred)
      EXTEND_TASK_PRED=true
      shift ;;
    --shadow_start)
      SHADOW_START="$2"
      shift 2 ;;
    --shadow_end)
      SHADOW_END="$2"
      shift 2 ;;
    --use_deepspeed)
      USE_DEEPSPEED=true
      shift ;;
    --deepspeed_level)
      DEEPSPEED_LEVEL="$2"
      shift 2 ;;
    --qr_lr)
      QR_LR="$2"
      shift 2 ;;
    --seeds)
      SEEDS="$2"
      shift 2 ;;
    --qr_train_batch_size)
      QR_TRAIN_BATCH_SIZE="$2"
      shift 2 ;;
    --qr_full_batch_size)
      QR_FULL_BATCH_SIZE="$2"
      shift 2 ;;
    --prefix_dir)
      PREFIX_DIR="$2"
      shift 2 ;;
    -h | --help)
      usage
      exit 0 ;;
    --)
      shift
      break ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1 ;;
  esac
done

NUM_CORES=8
CUDA_DEVICES="0,1,2,3,4,5,6,7"

GRADIENT_ACCUMULATION_STEPS=$(($(($QR_FULL_BATCH_SIZE / $QR_TRAIN_BATCH_SIZE)) / $NUM_CORES))


# Print the values of the variables
echo "DATASET_NAME: $DATASET_NAME"
echo "MODEL_ORG: $MODEL_ORG"
echo "MODEL_NAME: $MODEL_NAME"
echo "QR_MODEL_ORG: $QR_MODEL_ORG"
echo "QR_MODEL_NAME: $QR_MODEL_NAME"
echo "LM_TARGET_EPOCH: $LM_TARGET_EPOCH"
echo "QR_VALIDATION_PERCENT: $QR_VALIDATION_PERCENT"
echo "QR_SCORE: $QR_SCORE"
echo "TRAIN_TARGET: $TRAIN_TARGET"
echo "PRED_TARGET: $PRED_TARGET"
echo "RUN_MSE_PINBALL: $RUN_MSE_PINBALL"
echo "RUN_GAUSSIAN: $RUN_GAUSSIAN"
echo "RUN_IQR: $RUN_IQR"
echo "RUN_GAUSSIAN_PINBALL: $RUN_GAUSSIAN_PINBALL"
echo "TRAIN_SHADOW: $TRAIN_SHADOW"
echo "PRED_SHADOW: $PRED_SHADOW"
echo "EXTEND_TASK_PRED: $EXTEND_TASK_PRED"
echo "SHADOW_START: $SHADOW_START"
echo "SHADOW_END: $SHADOW_END"
echo "USE_DEEPSPEED: $USE_DEEPSPEED"
echo "DEEPSPEED_LEVEL: $DEEPSPEED_LEVEL"
echo "QR_LR: $QR_LR"
echo "SEEDS: $SEEDS"
echo "QR_TRAIN_BATCH_SIZE: $QR_TRAIN_BATCH_SIZE"
echo "QR_FULL_BATCH_SIZE: $QR_FULL_BATCH_SIZE"
echo "GRADIENT_ACCUMULATION_STEPS: $GRADIENT_ACCUMULATION_STEPS"
echo "PREFIX_DIR: $PREFIX_DIR"

case "$DEEPSPEED_LEVEL" in
    "1")
        DEEPSPEED_CONFIG=ds_config_stage1.json
        ;;
    "2")
        DEEPSPEED_CONFIG=ds_config_stage2_mod.json
        ;;
    "3")
        DEEPSPEED_CONFIG=ds_config_stage3_mod.json
        ;;
esac


# default parameters
MAX_SEQ_LENGTH=768
PUBLIC_PRIVATE_VAL_RATIO="0.45,0.45,0.1"

case "$DATASET_NAME" in
    "wikitext_sample")
        DATASET=wikitext
        DATASET_CONFIG=wikitext-103-raw-v1
        TEXT_COLUMN_NAME=text
        DATASET_NAME=wikitext_sample
        DATASET_FILE_NAME=wikitext
        SAMPLE_MIN_NUM_CHARS=25
        CHECK_POINT_NUM=
        case "$LM_TARGET_EPOCH" in 
        "1")
            CHECK_POINT_NUM=1585
            SHADOW_CHECK_POINT_NUMS="1415 1416 1417 1418 1419 1420 1421 1422"
            ;;
        "2")
            CHECK_POINT_NUM=3171
            SHADOW_CHECK_POINT_NUMS="2836 2839"
            ;;
        "3")
            CHECK_POINT_NUM=4755
            SHADOW_CHECK_POINT_NUMS="4254 4257"
            ;;
        esac
        PUBLIC_PRIVATE_VAL_RATIO="0.1,0.1,0.025"
        ;;
    "xsum")
        DATASET=EdinburghNLP/xsum
        DATASET_CONFIG=default
        TEXT_COLUMN_NAME=document
        DATASET_NAME=xsum
        DATASET_FILE_NAME=EdinburghNLP_xsum
        SAMPLE_MIN_NUM_CHARS=0
        case "$LM_TARGET_EPOCH" in 
        "1")
            CHECK_POINT_NUM=1431
            SHADOW_CHECK_POINT_NUMS="1296"
            ;;
        "2")
            CHECK_POINT_NUM=2863
            SHADOW_CHECK_POINT_NUMS="2592"
            ;;
        "3")
            CHECK_POINT_NUM=4293
            SHADOW_CHECK_POINT_NUMS="3888"
            ;;
        esac
        ;;
    "ag_news")
        DATASET=ag_news
        DATASET_CONFIG=default
        TEXT_COLUMN_NAME=text
        DATASET_NAME=ag_news
        DATASET_FILE_NAME=ag_news
        SAMPLE_MIN_NUM_CHARS=0
        case "$LM_TARGET_EPOCH" in 
        "1")
            CHECK_POINT_NUM=800
            SHADOW_CHECK_POINT_NUMS="725"
            ;;
        "2")
            CHECK_POINT_NUM=1600
            SHADOW_CHECK_POINT_NUMS="1450"
            ;;
        "3")
            CHECK_POINT_NUM=2400
            SHADOW_CHECK_POINT_NUMS="2175"
            ;;
        esac
        ;;
esac


MASTER_PORT=29800
LM_TRAIN_BATCH_SIZE=1
LM_PRED_BATCH_SIZE=1
LM_GRADIENT_ACCUMULATION_STEPS=8
LM_NUM_TRAIN_EPOCHS=3


PREDICT_SPLITS="public_train,public_test,private,validation"

NUM_EXPERIMENTS=None
EXPERIMENT_IDX=None
REGRESSION_NUM_EPOCHS=4
QR_TRAIN_SPLIT="public_train"
TEST_PRIVATE_SPLIT="private"
TEST_PUBLIC_SPLIT="public_test"
SENTENCE_KEYS=${TEXT_COLUMN_NAME}

# # # train target model
if ${TRAIN_TARGET}; then
    echo "training target model"
    for SEED in 42
    do        
        if ${USE_DEEPSPEED}; then
            WORLD_SIZE=${NUM_CORES} deepspeed -i "localhost:${CUDA_DEVICES}" --master_port ${MASTER_PORT} run_clm_lira.py \
            --deepspeed ${DEEPSPEED_CONFIG} \
            --model_name_or_path ${MODEL_ORG}/${MODEL_NAME} \
            --dataset_name ${DATASET} \
            --dataset_config_name ${DATASET_CONFIG} \
            --text_column_name ${TEXT_COLUMN_NAME} \
            --sample_min_num_chars ${SAMPLE_MIN_NUM_CHARS} \
            --torch_dtype ${TORCH_DTYPE} \
            --fp16 ${USE_FP16} \
            --do_train \
            --do_eval \
            --block_size ${MAX_SEQ_LENGTH} \
            --per_device_train_batch_size ${LM_TRAIN_BATCH_SIZE} \
            --per_device_eval_batch_size ${LM_TRAIN_BATCH_SIZE} \
            --gradient_accumulation_steps ${LM_GRADIENT_ACCUMULATION_STEPS} \
            --learning_rate 5e-5 \
            --num_train_epochs ${LM_NUM_TRAIN_EPOCHS} \
            --do_public_private_split True \
            --public_private_val_ratio ${PUBLIC_PRIVATE_VAL_RATIO} \
            --add_bos_eos \
            --output_dir "${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/task_model/expid_xx/$SEED" \
            --overwrite_output_dir \
            --save_only_model \
            --logging_strategy epoch \
            --evaluation_strategy epoch \
            --save_strategy epoch \
            --logging_steps 100 \
            --eval_steps 100 \
            --save_steps 100 \
            --seed $SEED \
            --overwrite_cache
        else
            WORLD_SIZE=${NUM_CORES} CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} torchrun --nproc_per_node=${NUM_CORES} --master_port=${MASTER_PORT} run_clm_lira.py \
            --model_name_or_path ${MODEL_ORG}/${MODEL_NAME} \
            --dataset_name ${DATASET} \
            --dataset_config_name ${DATASET_CONFIG} \
            --text_column_name ${TEXT_COLUMN_NAME} \
            --sample_min_num_chars ${SAMPLE_MIN_NUM_CHARS} \
            --do_train \
            --do_eval \
            --block_size ${MAX_SEQ_LENGTH} \
            --per_device_train_batch_size ${LM_TRAIN_BATCH_SIZE} \
            --per_device_eval_batch_size ${LM_TRAIN_BATCH_SIZE} \
            --gradient_accumulation_steps ${LM_GRADIENT_ACCUMULATION_STEPS} \
            --learning_rate 5e-5 \
            --num_train_epochs ${LM_NUM_TRAIN_EPOCHS} \
            --do_public_private_split True \
            --public_private_val_ratio ${PUBLIC_PRIVATE_VAL_RATIO} \
            --add_bos_eos \
            --output_dir "${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/task_model/expid_xx/$SEED" \
            --overwrite_output_dir \
            --save_only_model \
            --logging_strategy epoch \
            --evaluation_strategy epoch \
            --save_strategy epoch \
            --logging_steps 100 \
            --eval_steps 100 \
            --save_steps 100 \
            --seed $SEED \
            --overwrite_cache
        fi
    done
fi

# # # # make prediction and create quantile regression datasets
if ${PRED_TARGET}; then
    echo "predicting using target model"
    for SEED in 42
    do
        if [ "$CHECK_POINT_NUM" = "0" ]; then
            LM_INPUT_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/task_model/expid_xx/$SEED/"
            LM_PRED_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/task_pred/expid_xx/$SEED/"
        else
            LM_INPUT_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/task_model/expid_xx/$SEED/checkpoint-${CHECK_POINT_NUM}"
            LM_PRED_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/task_pred/expid_xx/$SEED/checkpoint-${CHECK_POINT_NUM}"
        fi

        echo "lm_input_dir=${LM_INPUT_DIR}"
        echo "lm_prd_dir=${LM_PRED_DIR}"

        if ${PRED_USE_DEEPSPEED}; then
            WORLD_SIZE=${NUM_CORES} deepspeed -i "localhost:${CUDA_DEVICES}" --master_port ${MASTER_PORT} run_clm_lira.py \
            --deepspeed ${DEEPSPEED_CONFIG} \
            --model_name_or_path ${LM_INPUT_DIR} \
            --dataset_name ${DATASET} \
            --dataset_config_name ${DATASET_CONFIG} \
            --text_column_name ${TEXT_COLUMN_NAME} \
            --sample_min_num_chars ${SAMPLE_MIN_NUM_CHARS} \
            --torch_dtype ${TORCH_DTYPE} \
            --do_predict \
            --block_size ${MAX_SEQ_LENGTH} \
            --per_device_train_batch_size ${LM_PRED_BATCH_SIZE} \
            --per_device_eval_batch_size ${LM_PRED_BATCH_SIZE} \
            --do_public_private_split True \
            --public_private_val_ratio ${PUBLIC_PRIVATE_VAL_RATIO} \
            --add_bos_eos \
            --output_dir ${LM_PRED_DIR} \
            --seed $SEED \
            --predict_split ${PREDICT_SPLITS} \
            --predict_normalize \
            --overwrite_cache \
            --predict_chunk_size 16384
        else
            WORLD_SIZE=${NUM_CORES} CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} torchrun --nproc_per_node=${NUM_CORES} --master_port=${MASTER_PORT} run_clm_lira.py \
            --model_name_or_path ${LM_INPUT_DIR} \
            --dataset_name ${DATASET} \
            --dataset_config_name ${DATASET_CONFIG} \
            --text_column_name ${TEXT_COLUMN_NAME} \
            --sample_min_num_chars ${SAMPLE_MIN_NUM_CHARS} \
            --do_predict \
            --block_size ${MAX_SEQ_LENGTH} \
            --per_device_train_batch_size ${LM_PRED_BATCH_SIZE} \
            --per_device_eval_batch_size ${LM_PRED_BATCH_SIZE} \
            --do_public_private_split True \
            --public_private_val_ratio ${PUBLIC_PRIVATE_VAL_RATIO} \
            --add_bos_eos \
            --output_dir ${LM_PRED_DIR} \
            --seed $SEED \
            --predict_split ${PREDICT_SPLITS} \
            --predict_normalize \
            --overwrite_cache \
            --predict_chunk_size 16384
        fi
    done
fi


# # # train shadow models
if ${TRAIN_SHADOW}; then
    echo "training shadow model"
    for SEED in 42
    do
        for IDX in $(seq $SHADOW_START $SHADOW_END)
        do        
            if ${USE_DEEPSPEED}; then
                WORLD_SIZE=${NUM_CORES} deepspeed -i "localhost:${CUDA_DEVICES}" --master_port ${MASTER_PORT} run_clm_lira.py \
                --deepspeed ${DEEPSPEED_CONFIG} \
                --model_name_or_path ${MODEL_ORG}/${MODEL_NAME} \
                --dataset_name ${DATASET} \
                --dataset_config_name ${DATASET_CONFIG} \
                --text_column_name ${TEXT_COLUMN_NAME} \
                --sample_min_num_chars ${SAMPLE_MIN_NUM_CHARS} \
                --torch_dtype ${TORCH_DTYPE} \
                --fp16 ${USE_FP16} \
                --do_train \
                --do_eval \
                --block_size ${MAX_SEQ_LENGTH} \
                --per_device_train_batch_size ${LM_TRAIN_BATCH_SIZE} \
                --per_device_eval_batch_size ${LM_TRAIN_BATCH_SIZE} \
                --gradient_accumulation_steps ${LM_GRADIENT_ACCUMULATION_STEPS} \
                --learning_rate 5e-5 \
                --num_train_epochs ${LM_NUM_TRAIN_EPOCHS} \
                --do_public_private_split True \
                --public_private_val_ratio ${PUBLIC_PRIVATE_VAL_RATIO} \
                --experiment_idx ${IDX}\
                --add_bos_eos \
                --output_dir "${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/shadow_model/expid_${IDX}/$SEED" \
                --overwrite_output_dir \
                --save_only_model \
                --logging_strategy epoch \
                --evaluation_strategy epoch \
                --save_strategy epoch \
                --logging_steps 100 \
                --eval_steps 100 \
                --save_steps 100 \
                --seed $SEED \
                --overwrite_cache \
                --use_public_for_training            
            else
                WORLD_SIZE=${NUM_CORES} CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} torchrun --nproc_per_node=${NUM_CORES} --master_port=${MASTER_PORT} run_clm_lira.py \
                --model_name_or_path ${MODEL_ORG}/${MODEL_NAME} \
                --dataset_name ${DATASET} \
                --dataset_config_name ${DATASET_CONFIG} \
                --text_column_name ${TEXT_COLUMN_NAME} \
                --sample_min_num_chars ${SAMPLE_MIN_NUM_CHARS} \
                --do_train \
                --do_eval \
                --block_size ${MAX_SEQ_LENGTH} \
                --per_device_train_batch_size ${LM_TRAIN_BATCH_SIZE} \
                --per_device_eval_batch_size ${LM_TRAIN_BATCH_SIZE} \
                --gradient_accumulation_steps ${LM_GRADIENT_ACCUMULATION_STEPS} \
                --learning_rate 5e-5 \
                --num_train_epochs ${LM_NUM_TRAIN_EPOCHS} \
                --do_public_private_split True \
                --public_private_val_ratio ${PUBLIC_PRIVATE_VAL_RATIO} \
                --experiment_idx ${IDX}\
                --add_bos_eos \
                --output_dir "${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/shadow_model/expid_${IDX}/$SEED" \
                --overwrite_output_dir \
                --save_only_model \
                --logging_strategy epoch \
                --evaluation_strategy epoch \
                --save_strategy epoch \
                --logging_steps 100 \
                --eval_steps 100 \
                --save_steps 100 \
                --seed $SEED \
                --overwrite_cache \
                --use_public_for_training
            fi
        done
    done
fi


# # # # make prediction and create quantile regression datasets
if ${PRED_SHADOW}; then
    echo "predicting using shadow model"
    for SEED in 42
    do
        for IDX in $(seq $SHADOW_START $SHADOW_END)
        do
            if [ "$SHADOW_CHECK_POINT_NUMS" = "0" ]; then
                if ${PRED_USE_DEEPSPEED}; then
                    WORLD_SIZE=${NUM_CORES} deepspeed -i "localhost:${CUDA_DEVICES}" --master_port ${MASTER_PORT} run_clm_lira.py \
                    --deepspeed ${DEEPSPEED_CONFIG} \
                    --model_name_or_path "${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/shadow_model/expid_${IDX}/$SEED" \
                    --dataset_name ${DATASET} \
                    --dataset_config_name ${DATASET_CONFIG} \
                    --text_column_name ${TEXT_COLUMN_NAME} \
                    --sample_min_num_chars ${SAMPLE_MIN_NUM_CHARS} \
                    --torch_dtype ${TORCH_DTYPE} \
                    --do_predict \
                    --block_size ${MAX_SEQ_LENGTH} \
                    --per_device_train_batch_size ${LM_PRED_BATCH_SIZE} \
                    --per_device_eval_batch_size ${LM_PRED_BATCH_SIZE} \
                    --do_public_private_split True \
                    --public_private_val_ratio ${PUBLIC_PRIVATE_VAL_RATIO} \
                    --add_bos_eos \
                    --output_dir "${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/shadow_pred/expid_${IDX}/$SEED" \
                    --seed $SEED \
                    --predict_split ${PREDICT_SPLITS} \
                    --predict_normalize \
                    --overwrite_cache \
                    --predict_chunk_size 16384
                else
                    WORLD_SIZE=${NUM_CORES} CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} torchrun --nproc_per_node=${NUM_CORES} --master_port=${MASTER_PORT} run_clm_lira.py \
                    --model_name_or_path "${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/shadow_model/expid_${IDX}/$SEED" \
                    --dataset_name ${DATASET} \
                    --dataset_config_name ${DATASET_CONFIG} \
                    --text_column_name ${TEXT_COLUMN_NAME} \
                    --sample_min_num_chars ${SAMPLE_MIN_NUM_CHARS} \
                    --do_predict \
                    --block_size ${MAX_SEQ_LENGTH} \
                    --per_device_train_batch_size ${LM_PRED_BATCH_SIZE} \
                    --per_device_eval_batch_size ${LM_PRED_BATCH_SIZE} \
                    --do_public_private_split True \
                    --public_private_val_ratio ${PUBLIC_PRIVATE_VAL_RATIO} \
                    --add_bos_eos \
                    --output_dir "${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/shadow_pred/expid_${IDX}/$SEED" \
                    --seed $SEED \
                    --predict_split ${PREDICT_SPLITS} \
                    --predict_normalize \
                    --overwrite_cache \
                    --predict_chunk_size 16384
                fi
            else
                for SHADOW_CHECK_POINT_NUM in ${SHADOW_CHECK_POINT_NUMS}
                do
                    if [ -d "${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/shadow_model/expid_${IDX}/$SEED/checkpoint-${SHADOW_CHECK_POINT_NUM}" ]; then
                        echo "shadow prediction from checkpoint ${SHADOW_CHECK_POINT_NUM}"
    
                        if ${PRED_USE_DEEPSPEED}; then
                            WORLD_SIZE=${NUM_CORES} deepspeed -i "localhost:${CUDA_DEVICES}" --master_port ${MASTER_PORT} run_clm_lira.py \
                            --deepspeed ${DEEPSPEED_CONFIG} \
                            --model_name_or_path "${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/shadow_model/expid_${IDX}/$SEED/checkpoint-${SHADOW_CHECK_POINT_NUM}" \
                            --dataset_name ${DATASET} \
                            --dataset_config_name ${DATASET_CONFIG} \
                            --text_column_name ${TEXT_COLUMN_NAME} \
                            --sample_min_num_chars ${SAMPLE_MIN_NUM_CHARS} \
                            --torch_dtype ${TORCH_DTYPE} \
                            --do_predict \
                            --block_size ${MAX_SEQ_LENGTH} \
                            --per_device_train_batch_size ${LM_PRED_BATCH_SIZE} \
                            --per_device_eval_batch_size ${LM_PRED_BATCH_SIZE} \
                            --do_public_private_split True \
                            --public_private_val_ratio ${PUBLIC_PRIVATE_VAL_RATIO} \
                            --add_bos_eos \
                            --output_dir "${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/shadow_pred/expid_${IDX}/$SEED/checkpoint-${SHADOW_CHECK_POINT_NUM}" \
                            --seed $SEED \
                            --predict_split ${PREDICT_SPLITS} \
                            --predict_normalize \
                            --overwrite_cache \
                            --predict_chunk_size 16384
                        else
                            WORLD_SIZE=${NUM_CORES} CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} torchrun --nproc_per_node=${NUM_CORES} --master_port=${MASTER_PORT} run_clm_lira.py \
                            --model_name_or_path "${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/shadow_model/expid_${IDX}/$SEED/checkpoint-${SHADOW_CHECK_POINT_NUM}" \
                            --dataset_name ${DATASET} \
                            --dataset_config_name ${DATASET_CONFIG} \
                            --text_column_name ${TEXT_COLUMN_NAME} \
                            --sample_min_num_chars ${SAMPLE_MIN_NUM_CHARS} \
                            --do_predict \
                            --block_size ${MAX_SEQ_LENGTH} \
                            --per_device_train_batch_size ${LM_PRED_BATCH_SIZE} \
                            --per_device_eval_batch_size ${LM_PRED_BATCH_SIZE} \
                            --do_public_private_split True \
                            --public_private_val_ratio ${PUBLIC_PRIVATE_VAL_RATIO} \
                            --add_bos_eos \
                            --output_dir "${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/shadow_pred/expid_${IDX}/$SEED/checkpoint-${SHADOW_CHECK_POINT_NUM}" \
                            --seed $SEED \
                            --predict_split ${PREDICT_SPLITS} \
                            --predict_normalize \
                            --overwrite_cache \
                            --predict_chunk_size 16384
                        fi
                    fi
                done
            fi
        done
    done
fi


if ${EXTEND_TASK_PRED}; then
    echo "extending task pred file"
    PREFIX_DIR="/home/rongtz/test/mia/"
    for SEED in 42
    do
        if [ "$CHECK_POINT_NUM" = "0" ]; then
            LM_PRED_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/task_pred/expid_xx/$SEED/"
        else
            LM_PRED_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/task_pred/expid_xx/$SEED/checkpoint-${CHECK_POINT_NUM}"
        fi

        echo "task_pred_dir=${LM_PRED_DIR}"

        python compute_extend_clm_scores.py \
        --task_name ${DATASET} \
        --text_column_name ${TEXT_COLUMN_NAME} \
        --task_pred_dir ${LM_PRED_DIR} \
        --task_seed $SEED \
        --num_experiments ${NUM_EXPERIMENTS} \
        --experiment_idx ${EXPERIMENT_IDX} \
        --predict_split ${PREDICT_SPLITS}
    done
fi


# # run gaussain_regression
if ${RUN_GAUSSIAN}; then
    echo "running gaussian regression"
    echo "SEEDS=${SEEDS}"
    case "$QR_SCORE" in
        "nce")
            REGRESSION_MODEL_DIR="sample_level_gaussian_regression_model"
            REGRESSION_PRED_DIR="sample_level_gaussian_regression_pred"
            LABEL_COLUMN="label"
            ;;
        "mink")
            REGRESSION_MODEL_DIR="sample_level_gaussian_regression_mink_model"
            REGRESSION_PRED_DIR="sample_level_gaussian_regression_mink_pred"
            LABEL_COLUMN="normalized_mink_nce"
            ;;
        "zlib")
            REGRESSION_MODEL_DIR="sample_level_gaussian_regression_zlib_model"
            REGRESSION_PRED_DIR="sample_level_gaussian_regression_zlib_pred"
            LABEL_COLUMN="normalized_zlib_score"
            ;;
    esac
    for SEED in ${SEEDS}
    do

    
        if [ "$CHECK_POINT_NUM" = "0" ]; then
            INPUT_TRAIN_FILE="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/task_pred/expid_xx/42/predict_results_${DATASET_FILE_NAME}_${QR_TRAIN_SPLIT}_${NUM_EXPERIMENTS}_${EXPERIMENT_IDX}.parquet"
            MODEL_OUTPUT_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/${REGRESSION_MODEL_DIR}/expid_xx/42/${QR_MODEL_NAME}/$SEED"
        else
            INPUT_TRAIN_FILE="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/task_pred/expid_xx/42/checkpoint-${CHECK_POINT_NUM}/predict_results_${DATASET_FILE_NAME}_${QR_TRAIN_SPLIT}_${NUM_EXPERIMENTS}_${EXPERIMENT_IDX}.parquet"
            MODEL_OUTPUT_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/${REGRESSION_MODEL_DIR}/expid_xx/42/checkpoint-${CHECK_POINT_NUM}/${QR_MODEL_NAME}/$SEED"
        fi
        
        echo "input_train_file=${INPUT_TRAIN_FILE}"
        echo "model_output_dir=${MODEL_OUTPUT_DIR}"

        WORLD_SIZE=${NUM_CORES} CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} torchrun --nproc_per_node=${NUM_CORES} --master_port=${MASTER_PORT} run_glue_quantile_regression_lira.py \
        --model_name_or_path ${QR_MODEL_ORG}/${QR_MODEL_NAME} \
        --do_train \
        --do_eval \
        --train_file ${INPUT_TRAIN_FILE} \
        --validation_split_percentage ${QR_VALIDATION_PERCENT} \
        --sample_min_num_chars ${SAMPLE_MIN_NUM_CHARS} \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --per_device_train_batch_size ${QR_TRAIN_BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        --learning_rate ${QR_LR} \
        --num_train_epochs ${REGRESSION_NUM_EPOCHS} \
        --output_dir ${MODEL_OUTPUT_DIR} \
        --overwrite_output_dir \
        --regression_type gaussian_regression \
        --optim "adamw_hf" \
        --lr_scheduler_type cosine \
        --seed $SEED \
        --sentence_keys ${SENTENCE_KEYS} \
        --label_column ${LABEL_COLUMN} \
        --save_only_model \
        --logging_strategy epoch \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --logging_steps 100 \
        --eval_steps 100 \
        --save_steps 100 \
        --save_total_limit 1 \
        --load_best_model_at_end \
        --metric_for_best_model eval_loss \
        --overwrite_cache
    done


    for SEED in ${SEEDS}
    do
        if [ "$CHECK_POINT_NUM" = "0" ]; then
            MODEL_INPUT_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/${REGRESSION_MODEL_DIR}/expid_xx/42/${QR_MODEL_NAME}/$SEED"
            PRED_OUTPUT_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/${REGRESSION_PRED_DIR}/expid_xx/42/${QR_MODEL_NAME}/$SEED"
        else
            MODEL_INPUT_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/${REGRESSION_MODEL_DIR}/expid_xx/42/checkpoint-${CHECK_POINT_NUM}/${QR_MODEL_NAME}/$SEED"
            PRED_OUTPUT_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/${REGRESSION_PRED_DIR}/expid_xx/42/checkpoint-${CHECK_POINT_NUM}/${QR_MODEL_NAME}/$SEED"
        fi

        echo "model_input_dir=${MODEL_INPUT_DIR}"
        echo "pred_output_dir=${PRED_OUTPUT_DIR}"

        WORLD_SIZE=${NUM_CORES} CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} torchrun --nproc_per_node=${NUM_CORES} --master_port=${MASTER_PORT} run_glue_quantile_regression_lira.py \
        --model_name_or_path ${MODEL_INPUT_DIR} \
        --dataset_name ${DATASET} \
        --dataset_config_name ${DATASET_CONFIG} \
        --sample_min_num_chars ${SAMPLE_MIN_NUM_CHARS} \
        --do_predict \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --per_device_train_batch_size 32 \
        --regression_type gaussian_regression \
        --do_public_private_split True \
        --public_private_val_ratio ${PUBLIC_PRIVATE_VAL_RATIO} \
        --output_dir ${PRED_OUTPUT_DIR} \
        --save_strategy epoch \
        --predict_split ${PREDICT_SPLITS} \
        --sentence_keys ${SENTENCE_KEYS} \
        --overwrite_cache
    done
fi

# # run iqr_regression
if ${RUN_IQR}; then
    echo "running iqr regression"
    case "$QR_SCORE" in
        "nce")
            REGRESSION_MODEL_DIR="sample_level_iqr_regression_model"
            REGRESSION_PRED_DIR="sample_level_iqr_regression_pred"
            LABEL_COLUMN="label"
            ;;
        "mink")
            REGRESSION_MODEL_DIR="sample_level_iqr_regression_mink_model"
            REGRESSION_PRED_DIR="sample_level_iqr_regression_mink_pred"
            LABEL_COLUMN="normalized_mink_nce"
            ;;
        "zlib")
            REGRESSION_MODEL_DIR="sample_level_iqr_regression_zlib_model"
            REGRESSION_PRED_DIR="sample_level_iqr_regression_zlib_pred"
            LABEL_COLUMN="normalized_zlib_score"
            ;;
    esac
    for SEED in ${SEEDS}
    do
        if [ "$CHECK_POINT_NUM" = "0" ]; then
            INPUT_TRAIN_FILE="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/task_pred/expid_xx/42/predict_results_${DATASET_FILE_NAME}_${QR_TRAIN_SPLIT}_${NUM_EXPERIMENTS}_${EXPERIMENT_IDX}.parquet"
            MODEL_OUTPUT_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/${REGRESSION_MODEL_DIR}/expid_xx/42/${QR_MODEL_NAME}/$SEED"
        else
            INPUT_TRAIN_FILE="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/task_pred/expid_xx/42/checkpoint-${CHECK_POINT_NUM}/predict_results_${DATASET_FILE_NAME}_${QR_TRAIN_SPLIT}_${NUM_EXPERIMENTS}_${EXPERIMENT_IDX}.parquet"
            MODEL_OUTPUT_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/${REGRESSION_MODEL_DIR}/expid_xx/42/checkpoint-${CHECK_POINT_NUM}/${QR_MODEL_NAME}/$SEED"
        fi

        echo "input_train_file=${INPUT_TRAIN_FILE}"
        echo "model_output_dir=${MODEL_OUTPUT_DIR}"

        WORLD_SIZE=${NUM_CORES} CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} torchrun --nproc_per_node=${NUM_CORES} --master_port=${MASTER_PORT} run_glue_quantile_regression_lira.py \
        --model_name_or_path ${QR_MODEL_ORG}/${QR_MODEL_NAME} \
        --do_train \
        --do_eval \
        --train_file ${INPUT_TRAIN_FILE} \
        --validation_split_percentage ${QR_VALIDATION_PERCENT} \
        --sample_min_num_chars ${SAMPLE_MIN_NUM_CHARS} \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --per_device_train_batch_size ${QR_TRAIN_BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        --learning_rate ${QR_LR} \
        --num_train_epochs ${REGRESSION_NUM_EPOCHS} \
        --output_dir ${MODEL_OUTPUT_DIR} \
        --overwrite_output_dir \
        --regression_type iqr_regression \
        --optim "adamw_hf" \
        --lr_scheduler_type cosine \
        --seed $SEED \
        --sentence_keys ${SENTENCE_KEYS} \
        --label_column ${LABEL_COLUMN} \
        --save_only_model \
        --logging_strategy epoch \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --logging_steps 100 \
        --eval_steps 100 \
        --save_steps 100 \
        --save_total_limit 1 \
        --load_best_model_at_end \
        --metric_for_best_model eval_loss \
        --overwrite_cache
    done

    for SEED in ${SEEDS}
    do
        if [ "$CHECK_POINT_NUM" = "0" ]; then
            MODEL_INPUT_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/${REGRESSION_MODEL_DIR}/expid_xx/42/${QR_MODEL_NAME}/$SEED"
            PRED_OUTPUT_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/${REGRESSION_PRED_DIR}/expid_xx/42/${QR_MODEL_NAME}/$SEED"
        else
            MODEL_INPUT_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/${REGRESSION_MODEL_DIR}/expid_xx/42/checkpoint-${CHECK_POINT_NUM}/${QR_MODEL_NAME}/$SEED"
            PRED_OUTPUT_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/${REGRESSION_PRED_DIR}/expid_xx/42/checkpoint-${CHECK_POINT_NUM}/${QR_MODEL_NAME}/$SEED"
        fi

        echo "model_input_dir=${MODEL_INPUT_DIR}"
        echo "pred_output_dir=${PRED_OUTPUT_DIR}"

        WORLD_SIZE=${NUM_CORES} CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} torchrun --nproc_per_node=${NUM_CORES} --master_port=${MASTER_PORT} run_glue_quantile_regression_lira.py \
        --model_name_or_path ${MODEL_INPUT_DIR} \
        --dataset_name ${DATASET} \
        --dataset_config_name ${DATASET_CONFIG} \
        --sample_min_num_chars ${SAMPLE_MIN_NUM_CHARS} \
        --do_predict \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --per_device_train_batch_size 32 \
        --regression_type iqr_regression \
        --do_public_private_split True \
        --public_private_val_ratio ${PUBLIC_PRIVATE_VAL_RATIO} \
        --output_dir ${PRED_OUTPUT_DIR} \
        --save_strategy epoch \
        --predict_split ${PREDICT_SPLITS} \
        --sentence_keys ${SENTENCE_KEYS} \
        --overwrite_cache
    done
fi


# # run mse_pinball_regression
if ${RUN_MSE_PINBALL}; then
    echo "running mse pinball regression"
    case "$QR_SCORE" in
        "nce")
            REGRESSION_MODEL_DIR="sample_level_mse_pinball_regression_model"
            REGRESSION_PRED_DIR="sample_level_mse_pinball_regression_pred"
            LABEL_COLUMN="label"
            ;;
        "mink")
            REGRESSION_MODEL_DIR="sample_level_mse_pinball_regression_mink_model"
            REGRESSION_PRED_DIR="sample_level_mse_pinball_regression_mink_pred"
            LABEL_COLUMN="normalized_mink_nce"
            ;;
        "zlib")
            REGRESSION_MODEL_DIR="sample_level_mse_pinball_regression_zlib_model"
            REGRESSION_PRED_DIR="sample_level_mse_pinball_regression_zlib_pred"
            LABEL_COLUMN="normalized_zlib_score"
            ;;
    esac
    for SEED in ${SEEDS}
    do
        if [ "$CHECK_POINT_NUM" = "0" ]; then
            INPUT_TRAIN_FILE="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/task_pred/expid_xx/42/predict_results_${DATASET_FILE_NAME}_${QR_TRAIN_SPLIT}_${NUM_EXPERIMENTS}_${EXPERIMENT_IDX}.parquet"
            MODEL_OUTPUT_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/${REGRESSION_MODEL_DIR}/expid_xx/42/${QR_MODEL_NAME}/$SEED"
        else
            INPUT_TRAIN_FILE="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/task_pred/expid_xx/42/checkpoint-${CHECK_POINT_NUM}/predict_results_${DATASET_FILE_NAME}_${QR_TRAIN_SPLIT}_${NUM_EXPERIMENTS}_${EXPERIMENT_IDX}.parquet"
            MODEL_OUTPUT_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/${REGRESSION_MODEL_DIR}/expid_xx/42/checkpoint-${CHECK_POINT_NUM}/${QR_MODEL_NAME}/$SEED"
        fi
        
        echo "input_train_file=${INPUT_TRAIN_FILE}"
        echo "model_output_dir=${MODEL_OUTPUT_DIR}"

        WORLD_SIZE=${NUM_CORES} CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} torchrun --nproc_per_node=${NUM_CORES} --master_port=${MASTER_PORT} run_glue_quantile_regression_lira.py \
        --model_name_or_path ${QR_MODEL_ORG}/${QR_MODEL_NAME} \
        --do_train \
        --do_eval \
        --train_file ${INPUT_TRAIN_FILE} \
        --validation_split_percentage ${QR_VALIDATION_PERCENT} \
        --sample_min_num_chars ${SAMPLE_MIN_NUM_CHARS} \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --per_device_train_batch_size ${QR_TRAIN_BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        --learning_rate ${QR_LR} \
        --num_train_epochs ${REGRESSION_NUM_EPOCHS} \
        --output_dir ${MODEL_OUTPUT_DIR} \
        --overwrite_output_dir \
        --regression_type mse_pinball_regression \
        --optim "adamw_hf" \
        --lr_scheduler_type cosine \
        --seed $SEED \
        --sentence_keys ${SENTENCE_KEYS} \
        --label_column ${LABEL_COLUMN} \
        --save_only_model \
        --logging_strategy epoch \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --logging_steps 100 \
        --eval_steps 100 \
        --save_steps 100 \
        --save_total_limit 1 \
        --load_best_model_at_end \
        --metric_for_best_model eval_loss \
        --overwrite_cache
    done

    for SEED in ${SEEDS}
    do
        if [ "$CHECK_POINT_NUM" = "0" ]; then
            MODEL_INPUT_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/${REGRESSION_MODEL_DIR}/expid_xx/42/${QR_MODEL_NAME}/$SEED"
            PRED_OUTPUT_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/${REGRESSION_PRED_DIR}/expid_xx/42/${QR_MODEL_NAME}/$SEED"
        else
            MODEL_INPUT_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/${REGRESSION_MODEL_DIR}/expid_xx/42/checkpoint-${CHECK_POINT_NUM}/${QR_MODEL_NAME}/$SEED"
            PRED_OUTPUT_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/${REGRESSION_PRED_DIR}/expid_xx/42/checkpoint-${CHECK_POINT_NUM}/${QR_MODEL_NAME}/$SEED"
        fi

        echo "model_input_dir=${MODEL_INPUT_DIR}"
        echo "pred_output_dir=${PRED_OUTPUT_DIR}"

        WORLD_SIZE=${NUM_CORES} CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} torchrun --nproc_per_node=${NUM_CORES} --master_port=${MASTER_PORT} run_glue_quantile_regression_lira.py \
        --model_name_or_path ${MODEL_INPUT_DIR} \
        --dataset_name ${DATASET} \
        --dataset_config_name ${DATASET_CONFIG} \
        --sample_min_num_chars ${SAMPLE_MIN_NUM_CHARS} \
        --do_predict \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --per_device_train_batch_size 32 \
        --regression_type mse_pinball_regression \
        --do_public_private_split True \
        --public_private_val_ratio ${PUBLIC_PRIVATE_VAL_RATIO} \
        --output_dir ${PRED_OUTPUT_DIR} \
        --save_strategy epoch \
        --predict_split ${PREDICT_SPLITS} \
        --sentence_keys ${SENTENCE_KEYS} \
        --overwrite_cache
    done
fi

# # run gaussian_pinball_regression
if ${RUN_GAUSSIAN_PINBALL}; then
    echo "running gaussian pinball regression"
    case "$QR_SCORE" in
        "nce")
            REGRESSION_MODEL_DIR="sample_level_gaussian_pinball_regression_model"
            REGRESSION_PRED_DIR="sample_level_gaussian_pinball_regression_pred"
            LABEL_COLUMN="label"
            ;;
        "mink")
            REGRESSION_MODEL_DIR="sample_level_gaussian_pinball_regression_mink_model"
            REGRESSION_PRED_DIR="sample_level_gaussian_pinball_regression_mink_pred"
            LABEL_COLUMN="normalized_mink_nce"
            ;;
        "zlib")
            REGRESSION_MODEL_DIR="sample_level_gaussian_pinball_regression_zlib_model"
            REGRESSION_PRED_DIR="sample_level_gaussian_pinball_regression_zlib_pred"
            LABEL_COLUMN="normalized_zlib_score"
            ;;
    esac
    for SEED in ${SEEDS}
    do
        if [ "$CHECK_POINT_NUM" = "0" ]; then
            INPUT_TRAIN_FILE="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/task_pred/expid_xx/42/predict_results_${DATASET_FILE_NAME}_${QR_TRAIN_SPLIT}_${NUM_EXPERIMENTS}_${EXPERIMENT_IDX}.parquet"
            MODEL_OUTPUT_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/${REGRESSION_MODEL_DIR}/expid_xx/42/${QR_MODEL_NAME}/$SEED"
        else
            INPUT_TRAIN_FILE="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/task_pred/expid_xx/42/checkpoint-${CHECK_POINT_NUM}/predict_results_${DATASET_FILE_NAME}_${QR_TRAIN_SPLIT}_${NUM_EXPERIMENTS}_${EXPERIMENT_IDX}.parquet"
            MODEL_OUTPUT_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/${REGRESSION_MODEL_DIR}/expid_xx/42/checkpoint-${CHECK_POINT_NUM}/${QR_MODEL_NAME}/$SEED"
        fi
        
        echo "input_train_file=${INPUT_TRAIN_FILE}"
        echo "model_output_dir=${MODEL_OUTPUT_DIR}"

        WORLD_SIZE=${NUM_CORES} CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} torchrun --nproc_per_node=${NUM_CORES} --master_port=${MASTER_PORT} run_glue_quantile_regression_lira.py \
        --model_name_or_path ${QR_MODEL_ORG}/${QR_MODEL_NAME} \
        --do_train \
        --do_eval \
        --train_file ${INPUT_TRAIN_FILE} \
        --validation_split_percentage ${QR_VALIDATION_PERCENT} \
        --sample_min_num_chars ${SAMPLE_MIN_NUM_CHARS} \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --per_device_train_batch_size ${QR_TRAIN_BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        --learning_rate ${QR_LR} \
        --num_train_epochs ${REGRESSION_NUM_EPOCHS} \
        --output_dir ${MODEL_OUTPUT_DIR} \
        --overwrite_output_dir \
        --regression_type gaussian_pinball_regression \
        --optim "adamw_hf" \
        --lr_scheduler_type cosine \
        --seed $SEED \
        --sentence_keys ${SENTENCE_KEYS} \
        --label_column ${LABEL_COLUMN} \
        --save_only_model \
        --logging_strategy epoch \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --logging_steps 100 \
        --eval_steps 100 \
        --save_steps 100 \
        --save_total_limit 1 \
        --load_best_model_at_end \
        --metric_for_best_model eval_loss \
        --overwrite_cache
    done

    for SEED in ${SEEDS}
    do
        if [ "$CHECK_POINT_NUM" = "0" ]; then
            MODEL_INPUT_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/${REGRESSION_MODEL_DIR}/expid_xx/42/${QR_MODEL_NAME}/$SEED"
            PRED_OUTPUT_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/${REGRESSION_PRED_DIR}/expid_xx/42/${QR_MODEL_NAME}/$SEED"
        else
            MODEL_INPUT_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/${REGRESSION_MODEL_DIR}/expid_xx/42/checkpoint-${CHECK_POINT_NUM}/${QR_MODEL_NAME}/$SEED"
            PRED_OUTPUT_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/${REGRESSION_PRED_DIR}/expid_xx/42/checkpoint-${CHECK_POINT_NUM}/${QR_MODEL_NAME}/$SEED"
        fi

        echo "model_input_dir=${MODEL_INPUT_DIR}"
        echo "pred_output_dir=${PRED_OUTPUT_DIR}"

        WORLD_SIZE=${NUM_CORES} CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} torchrun --nproc_per_node=${NUM_CORES} --master_port=${MASTER_PORT} run_glue_quantile_regression_lira.py \
        --model_name_or_path ${MODEL_INPUT_DIR} \
        --dataset_name ${DATASET} \
        --dataset_config_name ${DATASET_CONFIG} \
        --sample_min_num_chars ${SAMPLE_MIN_NUM_CHARS} \
        --do_predict \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --per_device_train_batch_size 32 \
        --regression_type gaussian_pinball_regression \
        --do_public_private_split True \
        --public_private_val_ratio ${PUBLIC_PRIVATE_VAL_RATIO} \
        --output_dir ${PRED_OUTPUT_DIR} \
        --save_strategy epoch \
        --predict_split ${PREDICT_SPLITS} \
        --sentence_keys ${SENTENCE_KEYS} \
        --overwrite_cache
    done
fi