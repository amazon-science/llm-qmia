# Define the usage function to print the help message
usage() {
  echo "Usage: $0 [options] [positional arguments]"
  echo "Options:"
  echo "  --dataset_name <value>              Name of the dataset"
  echo "  --model_org <value>                 Organization of the model"
  echo "  --model_name <value>                Name of the model"
  echo "  --qr_model_org <value>              Organization of the QR model"
  echo "  --qr_model_name <value>             Name of the QR model"
  echo "  --lm_target_epoch <value>           Epoch for LM target"
  echo "  --lira_model_names <value>          Names of the Lira models"
  echo "  --plot_lira_model_names <value>     Names of the Lira models to be included in plotting"
  echo "  --max_num_lira_experiments <value>  Maximum of Lira models"
  echo "  --mismatch_plot_all                 set MISMATCH_PLOT_ALL to True"
  echo "  --seeds <value>                     SEEDS for plotting"
  echo "  --prefix_dir <value>                Root directory for storing results"  
  echo "  -h, --help                          Display this help message"
}

# Define the long options
longopts="dataset_name:,model_org:,model_name:,qr_model_org:,qr_model_name:,lm_target_epoch:,lira_model_names:,plot_lira_model_names:,max_num_lira_experiments:,mismatch_plot_all,seeds:,prefix_dir:,help"

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

LIRA_MODEL_NAMES="pythia-160m,pythia-410m,pythia-1.4b,pythia-2.8b,pythia-6.9b"
PLOT_LIRA_MODEL_NAMES=""
MAX_NUM_LIRA_EXPERIMENTS=4

MISMATCH_PLOT_ALL=0
SEEDS="42,1024,512,2048,256"
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
    --lira_model_names)
      LIRA_MODEL_NAMES="$2"
      shift 2 ;;
    --plot_lira_model_names)
      PLOT_LIRA_MODEL_NAMES="$2"
      shift 2 ;;
    --max_num_lira_experiments)
      MAX_NUM_LIRA_EXPERIMENTS="$2"
      shift 2 ;;
    --mismatch_plot_all)
      MISMATCH_PLOT_ALL=1
      shift ;;
    --seeds)
      SEEDS="$2"
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

# Print the values of the variables
echo "DATASET_NAME: $DATASET_NAME"
echo "MODEL_ORG: $MODEL_ORG"
echo "MODEL_NAME: $MODEL_NAME"
echo "QR_MODEL_ORG: $QR_MODEL_ORG"
echo "QR_MODEL_NAME: $QR_MODEL_NAME"
echo "LM_TARGET_EPOCH: $LM_TARGET_EPOCH"
echo "LIRA_MODEL_NAMES: $LIRA_MODEL_NAMES"
echo "PLOT_LIRA_MODEL_NAMES: $PLOT_LIRA_MODEL_NAMES"
echo "MAX_NUM_LIRA_EXPERIMENTS: $MAX_NUM_LIRA_EXPERIMENTS"
echo "MISMATCH_PLOT_ALL: $MISMATCH_PLOT_ALL"
echo "SEEDS: $SEEDS"
echo "PREFIX_DIR: $PREFIX_DIR"


# default parameters
case "$DATASET_NAME" in
    "wikitext_sample")
        DATASET=wikitext
        DATASET_CONFIG=wikitext-103-raw-v1
        TEXT_COLUMN_NAME=text
        DATASET_NAME=wikitext_sample
        DATASET_FILE_NAME=wikitext
        SAMPLE_MIN_NUM_CHARS=25
        case "$LM_TARGET_EPOCH" in 
        "1")
            CHECK_POINT_NUM=1585
            SHADOW_CHECK_POINT_NUMS="1416,1417,1418,1419,1420,1421"
            ;;
        "2")
            CHECK_POINT_NUM=3171
            SHADOW_CHECK_POINT_NUMS="2836,2839"
            ;;
        "3")
            CHECK_POINT_NUM=4755
            SHADOW_CHECK_POINT_NUMS="4254,4257"
            ;;
        esac
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


TASK_SEED="42"
QR_VALIDATION_PERCENT=5
REGRESSION_NUM_EPOCHS=4
QR_TRAIN_SPLIT="public_train"
SENTENCE_KEYS=${TEXT_COLUMN_NAME}

PRIVATE_SPLIT="private"
PUBLIC_SPLIT="public_test"

TASK_PRED_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/task_pred/expid_xx/${TASK_SEED}/checkpoint-${CHECK_POINT_NUM}"
REGRESSION_PRED_DIR_TEMPLATE="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/sample_level_{}_pred/expid_xx/${TASK_SEED}/checkpoint-${CHECK_POINT_NUM}/${QR_MODEL_NAME}"
SHADOW_PRED_DIR_TEMPLATE="${PREFIX_DIR}/runs/{model_name}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/shadow_pred/expid_*/${TASK_SEED}/checkpoint-{checkpoint}"
PLOT_OUTPUT_DIR="${PREFIX_DIR}/runs/${MODEL_NAME}/${DATASET_NAME}_${TEXT_COLUMN_NAME}/sample_level_plots/checkpoint-${CHECK_POINT_NUM}/${QR_MODEL_NAME}/"

echo "plotting ${DATASET_NAME}"
echo ${TASK_PRED_DIR}


python compute_qmia_llm_full_roc_ensemble.py \
--task_name ${DATASET} \
--task_pred_dir ${TASK_PRED_DIR} \
--regression_pred_dir_template ${REGRESSION_PRED_DIR_TEMPLATE} \
--task_seed ${TASK_SEED} \
--seeds ${SEEDS} \
--private_split ${PRIVATE_SPLIT} \
--public_split ${PUBLIC_SPLIT} \
--output_dir ${PLOT_OUTPUT_DIR} \
--lira_pred_dir_template ${SHADOW_PRED_DIR_TEMPLATE} \
--target_model ${MODEL_NAME} \
--plot_lira_models ${PLOT_LIRA_MODEL_NAMES} \
--lira_models ${LIRA_MODEL_NAMES} \
--shadow_checkpoint_nums ${SHADOW_CHECK_POINT_NUMS} \
--text_column_name ${TEXT_COLUMN_NAME} \
--max_num_lira_experiments ${MAX_NUM_LIRA_EXPERIMENTS} \
--mismatch_plot_all ${MISMATCH_PLOT_ALL}