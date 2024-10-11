#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import json

import datasets
import evaluate
import numpy as np
from datasets import load_dataset
import pandas as pd

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import scipy.stats

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.39.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


#TODO! make dataset private/public/test dataset selection a la LIRA. Save sentence-score pairs
#based on https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py
import contextlib
@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    num_experiments: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "total number of private/public partitions for MIA purposes"
            )
        },
    )
    experiment_idx: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "experiment number for MIA purposes, needs to be an integer in {0, num_experiments-1}"
            )
        },
    )
    do_public_private_split: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to split public and private data."
            )
        },
    )
    use_public_for_training: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to use public or private split for training."
            )
        },
    )
    public_private_val_ratio: Optional[str] = field(
        default='0.45,0.45,0.1',
        metadata={
            "help": (
                "public private and private split ratios."
            )
        },
    )
    pkeep: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "percentage split in public/private data for MIA"
            )
        },
    )
    val_pkeep: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "percentage split in public train/test data for MIA"
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    predict_split: Optional[str] = field(default=None, metadata={"help": "split for prediction"})
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    sentence_keys: Optional[str] = field(
        default=None,
        metadata={"help": "sentence keys"}
    )
    text_label: Optional[str] = field(
        default=None,
        metadata={"help": "column to be used for creating the text"}
    )
    sample_min_num_chars: Optional[int] = field(
        default=0,
        metadata={"help": "filter sample less than given number of chars."}
    )
    label_column: Optional[str] = field(
        default=None,
        metadata={"help": "column to be used as label"}        
    )
    format_input: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to format the input."
            )
        },
    )
    bootstrap_seed: Optional[int] = field(
        default=None,
        metadata={
            "help": "seed for bootstrapped sampling of training set"
        }
    )

    
    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json", "jsonl", "parquet"], "`train_file` should be a csv or a json file."
            if self.validation_file:
                validation_extension = self.validation_file.split(".")[-1]
                assert (
                    validation_extension == train_extension
                ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    regression_type: str = field(
        default="gaussian_regression",
        metadata={
            "help": (
                "The regression type for quantile regression."
            )
        }
    )
    kl_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "kl divergence weight for gaussian regression."
            )
        }
    )
    quantiles: str = field(
        default=None,
        metadata={
            "help": "target quantiles for quantile regression"
        }
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )     
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
#                 streaming=data_args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
#                 streaming=data_args.streaming,
            )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file}
        if data_args.validation_file:
            data_files["validation"] = data_args.validation_file

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        dataset_args = {}
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            if data_args.bootstrap_seed is None:
                logger.info("split training data to create validation split")
                raw_datasets["validation"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    **dataset_args,
                )
                raw_datasets["train"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    **dataset_args,
                )
            else:
                logger.info(f"split training data to create validation split by random permutation with seed={data_args.bootstrap_seed}.")
                validation_split_size = round(len(raw_datasets["train"]) * (data_args.validation_split_percentage * 0.01))
                with temp_seed(data_args.bootstrap_seed):
                    random_permute = np.random.permutation(len(raw_datasets["train"]))
                raw_datasets["train_validation"] = raw_datasets["train"]
                raw_datasets["validation"] = raw_datasets["train_validation"].select(random_permute[:validation_split_size])
                raw_datasets["train"] = raw_datasets["train_validation"].select(random_permute[validation_split_size:])
                raw_datasets.pop("train_validation")
        
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.

    train_split_name ='train'
    if data_args.num_experiments is not None:
        assert data_args.experiment_idx is not None, 'experiment_idx needs to be set if using num_experiments'
        assert data_args.experiment_idx< data_args.num_experiments and data_args.experiment_idx>=0, 'experiment_idx needs to be in {0,..., num_experiments-1}'
        with temp_seed(42):
            num_train_samples = len(raw_datasets["train"])
            master_keep = np.random.uniform(
                size=(data_args.num_experiments, num_train_samples), low=0, high=1
            )
        order = master_keep.argsort(0)
        master_keep = order < int(data_args.pkeep * data_args.num_experiments)
        keep = np.array(master_keep[data_args.experiment_idx], dtype=bool)
        private_indices = list(np.where(keep)[0])
        public_indices = list(np.where(~keep)[0])
        raw_datasets['public_private'] = raw_datasets["train"]
        raw_datasets['public_train'] = raw_datasets["train"].select(public_indices)
        raw_datasets['private'] = raw_datasets["train"].select(private_indices)
        train_split_name = 'private'
    
    elif data_args.do_public_private_split:
        if data_args.use_public_for_training:
            train_split_name ='public_train'
        else:
            train_split_name ='private'

        public_private_val_ratio = [0.]+[float(p) for p in data_args.public_private_val_ratio.split(',')]
        cumulative_ratios = np.cumsum(public_private_val_ratio)
        with temp_seed(42):
            num_train_samples = len(raw_datasets["train"])
            master_keep = np.random.uniform(
                size=(num_train_samples), low=0, high=1
            )
        private_keep = np.logical_and(~np.array(master_keep<cumulative_ratios[0], dtype=bool), np.array(master_keep<=cumulative_ratios[1], dtype=bool), )
        public_train_keep = np.logical_and(~np.array(master_keep<cumulative_ratios[1], dtype=bool), np.array(master_keep<=cumulative_ratios[2], dtype=bool), )
        public_val_keep = np.logical_and(~np.array(master_keep<cumulative_ratios[2], dtype=bool), np.array(master_keep<=cumulative_ratios[3], dtype=bool), )

        private_indices = list(np.where(private_keep)[0])
        public_train_indices = list(np.where(public_train_keep)[0])
        public_val_indices = list(np.where(public_val_keep)[0])
        
        
        logger.info("private & public_train = {}".format(len(set(private_indices) & set(public_train_indices))))
        logger.info("private & public_test = {}".format(len(set(private_indices) & set(public_val_indices))))
        logger.info("public_train & public_test = {}".format(len(set(public_train_indices) & set(public_val_indices))))
        
        if data_args.experiment_idx and data_args.lira_train_ratio:
            #further subsample public train to add data variation
            with temp_seed(42+data_args.experiment_idx):
                np.random.shuffle(public_train_indices)
                public_train_indices = public_train_indices[:int(data_args.lira_train_ratio(len(public_train_indices)))]
                public_train_indices = list(sorted(public_train_indices))

        raw_datasets['public_train'] = raw_datasets["train"].select(public_train_indices)
        raw_datasets['public_test'] = raw_datasets["train"].select(public_val_indices)
        raw_datasets['private'] = raw_datasets["train"].select(private_indices)
        raw_datasets.pop("train")
    
    logger.info(f"{raw_datasets}")
    logger.info(f"Using {train_split_name} for training")

    if data_args.sample_min_num_chars > 0:
        logger.info(f"Filter samples with less than {data_args.sample_min_num_chars} chars.")
        text_column_name = data_args.sentence_keys.split(",")[0]
        for split, dataset in raw_datasets.items():
            raw_datasets[split] = dataset.filter(lambda example: len(example[text_column_name]) >= data_args.sample_min_num_chars)
        
    # Labels
    # We don't have labels, this is a quantile regression task
    is_regression = True
    num_labels = 1

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        problem_type="regression",
        regression_type=model_args.regression_type
    )
    if not hasattr(config, "regression_type"):
        config.regression_type = model_args.regression_type
    if not hasattr(config, "kl_weight"):
        config.kl_weight = model_args.kl_weight
    
    model_quantiles = None
    if model_args.quantiles is not None:
        model_quantiles = [float(y) for y in model_args.quantiles.split(",")]
    elif model_args.regression_type == "iqr_regression" and model_args.quantiles is None:
        model_quantiles = [1-scipy.stats.norm.sf(0), 1-scipy.stats.norm.sf(1)]
    elif model_args.regression_type == "mse_pinball_regression" and model_args.quantiles is None:
        model_quantiles = [1-scipy.stats.norm.sf(1)]
    if not hasattr(config, "quantiles") and model_quantiles is not None:
        config.quantiles = model_quantiles
        logger.info(f"quantiles={model_quantiles}")
        
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    from transformers import GPTNeoXTokenizerFast
    from transformers import LlamaTokenizerFast
    if isinstance(tokenizer, GPTNeoXTokenizerFast):
        tokenizer.pad_token = '<|padding|>'
        tokenizer.pad_token_id = 1
        config.pad_token_id = 1
    elif isinstance(tokenizer, LlamaTokenizerFast):
        tokenizer.pad_token = '<unk>'
        tokenizer.pad_token_id = 0
        config.pad_token_id = 0

    logger.info(f"pad_token={tokenizer.pad_token}")

    if not tokenizer.padding_side == "right":
        tokenizer.padding_side = "right"

    logger.info(f"padding_side={tokenizer.padding_side}")

    from modeling_quantile_regression import BertForQuantileRegression, OPTForQuantileRegression, GPTNeoXForQuantileRegression
    model_class_dict = {
        "BertForMaskedLM": BertForQuantileRegression,
        "BertForQuantileRegression": BertForQuantileRegression,
        "OPTForCausalLM": OPTForQuantileRegression,
        "OPTForQuantileRegression": OPTForQuantileRegression,
        "GPTNeoXForCausalLM": GPTNeoXForQuantileRegression,
        "GPTNeoXForQuantileRegression": GPTNeoXForQuantileRegression,
    }
    model_class = model_class_dict[config.architectures[0]]
    logger.info(f"base_model_class={config.architectures[0]}")

    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    logger.info(f"config={model.config}")
    # Preprocessing the raw_datasets
    if data_args.sentence_keys is not None:
        sentence_keys = data_args.sentence_keys.split(",")
        if len(sentence_keys) >= 2:
            sentence1_key, sentence2_key = sentence_keys[:2]
        else:
            sentence1_key, sentence2_key = sentence_keys[0], None
    elif data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        elif "sentence" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence", None
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None
    
    logger.info(f"sentence1_key={sentence1_key},sentence1_key={sentence2_key}")
    
    # change the text for prediction if using label
    if data_args.text_label is not None:
        if not data_args.format_input:
            # for bert: [CLS] label [SEP] sentence1 [SEP] sentence2 [SEP]
            def create_text_with_label(examples):
                # we are using "orig_label" from the classification task not the "label" for regression
                if sentence2_key is None:
                    text_with_label_list = [
                        str(label) + f" {tokenizer.sep_token} " + sentence1
                        for label, sentence1 in zip(examples[data_args.text_label], examples[sentence1_key])
                    ]
                else:
                    text_with_label_list = [
                        str(label) + f" {tokenizer.sep_token} " + sentence1 + f" {tokenizer.sep_token} " + sentence2
                        for label, sentence1, sentence2 in zip(examples[data_args.text_label], examples[sentence1_key], examples[sentence2_key])
                    ]
                examples["text_with_label"] =   text_with_label_list  

                return examples
        else:
            # label: <label>\n<sentence1_key>: <sentence1>\n<sentence2_key>: <sentence2>
            def create_text_with_label(examples):
                # we are using "orig_label" from the classification task not the "label" for regression
                if sentence2_key is None:
                    text_with_label_list = [
                        f"label: {label}\n{sentence1_key}: {sentence1}"
                        for label, sentence1 in zip(examples[data_args.text_label], examples[sentence1_key])
                    ]
                else:
                    text_with_label_list = [
                        f"label: {label}\n{sentence1_key}: {sentence1}\n{sentence2_key}: {sentence2}"
                        for label, sentence1, sentence2 in zip(examples[data_args.text_label], examples[sentence1_key], examples[sentence2_key])
                    ]
                examples["text_with_label"] =   text_with_label_list  

                return examples

        raw_datasets = raw_datasets.map(
            create_text_with_label,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Creating text with label on dataset",
        )
        sentence1_key, sentence2_key = "text_with_label", None    
    elif data_args.format_input:
        # change the text for prediction
        # sentence1_key: sentence1\nsentence2_key: sentence2
        def create_formatted_text(examples):
            # we are using "orig_label" from the classification task not the "label" for regression
            if sentence2_key is None:
                formatted_text_list = [
                    f"{sentence1_key}: {sentence1}"
                    for sentence1 in examples[sentence1_key]
                ]
            else:
                formatted_text_list = [
                    f"{sentence1_key}: {sentence1}\n{sentence2_key}: {sentence2}"
                    for sentence1, sentence2 in zip(examples[sentence1_key], examples[sentence2_key])
                ]
            examples["formatted_text"] =   formatted_text_list  

            return examples

        raw_datasets = raw_datasets.map(
            create_formatted_text,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Creating formatted text on dataset",
        )
        sentence1_key, sentence2_key = "formatted_text", None        
    
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(raw_datasets["train"])), 3):
            logger.info(f"Sample {index} of the training set: {raw_datasets['train'][index]}.")

    
    # add option to select label for regression
    label_column = "label"
    if data_args.label_column is not None:
        label_column = data_args.label_column
    
    logger.info(f"using {label_column} as label")

    def preprocess_function(examples):
        examples.pop("input_ids", None)
        examples.pop("attention_mask", None)
        examples.pop("labels", None)

        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        
        if label_column in examples:
            examples["label"] = examples[label_column]
            result["label"] = examples[label_column]

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and label_column in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        predict_split = "test_matched" if data_args.task_name == "mnli" else "test"
        if data_args.predict_split is not None:
            predict_dataset = [
                raw_datasets[p_split] for p_split in data_args.predict_split.split(",")
            ]
            logger.info(f"predict on {data_args.predict_split}")
            predict_split = data_args.predict_split
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = evaluate.load("glue", data_args.task_name, cache_dir=model_args.cache_dir)
    elif is_regression:
        metric = evaluate.load("mse", cache_dir=model_args.cache_dir)
    else:
        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.    
    from modeling_quantile_regression import gaussian_loss_fn, pinball_loss_fn
    
    def compute_metrics(p: EvalPrediction):
        if config.regression_type == "regression":
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            mse = torch.nn.functional.mse_loss(torch.tensor(preds), torch.tensor(p.label_ids)).detach().cpu()
            mae = torch.nn.functional.l1_loss(torch.tensor(preds), torch.tensor(p.label_ids)).detach().cpu()
            result = {
                "mse": mse,
                "mae": mae,
            }
        elif config.regression_type == "gaussian_regression":
            preds = p.predictions
            pred_mu = preds[:, 0]
            mse_uncertainty, kl = gaussian_loss_fn(torch.tensor(preds), torch.tensor(p.label_ids), return_kl=True)
            mse_uncertainty = mse_uncertainty.mean().detach().cpu()
            kl = kl.mean().detach().cpu()
            mse = torch.nn.functional.mse_loss(torch.tensor(pred_mu), torch.tensor(p.label_ids)).detach().cpu()
            mae = torch.nn.functional.l1_loss(torch.tensor(pred_mu), torch.tensor(p.label_ids)).detach().cpu()
            result = {
                "mse_uncertainty": mse_uncertainty,
                "mse": mse,
                "mae": mae,
                "kl": kl
            }
        elif config.regression_type == "mse_pinball_regression":
            preds = p.predictions
            pred_mu = preds[:, 0]
            mse = torch.nn.functional.mse_loss(torch.tensor(pred_mu), torch.tensor(p.label_ids)).detach().cpu()
            mae = torch.nn.functional.l1_loss(torch.tensor(pred_mu), torch.tensor(p.label_ids)).detach().cpu()
            pinball_losses = [
                pinball_loss_fn(preds[:, [i+1]], torch.tensor(p.label_ids), quantiles=torch.tensor([quantile])).mean().detach().cpu()
                for i, quantile in enumerate(model_quantiles)
            ]
            result = {
                f"pinball_{quantile}": pinball_losses[i]
                for i, quantile in enumerate(model_quantiles)
            }
            result["mse"] = mse
            result["mae"] = mae
        elif config.regression_type == "iqr_regression":
            preds = p.predictions
            pinball_losses = [
                pinball_loss_fn(preds[:, [i]], torch.tensor(p.label_ids), quantiles=torch.tensor([quantile])).mean().detach().cpu()
                for i, quantile in enumerate(model_quantiles)
            ]
            result = {
                f"pinball_{quantile}": pinball_losses[i]
                for i, quantile in enumerate(model_quantiles)
            }
            for idx, quantile in enumerate(model_quantiles):
                if abs(quantile - 0.5) < 1e-8:
                    mse = torch.nn.functional.mse_loss(torch.tensor(preds[:, idx]), torch.tensor(p.label_ids)).detach().cpu()
                    mae = torch.nn.functional.l1_loss(torch.tensor(preds[:, idx]), torch.tensor(p.label_ids)).detach().cpu()
                    result["mse"] = mse
                    result["mae"] = mae
        elif config.regression_type == "gaussian_pinball_regression":
            preds = p.predictions
            pred_mu = preds[:, 0]
            mse_uncertainty, kl = gaussian_loss_fn(torch.tensor(preds), torch.tensor(p.label_ids), return_kl=True)
            mse_uncertainty = mse_uncertainty.mean().detach().cpu()
            kl = kl.mean().detach().cpu()
            pinball_loss_mean = pinball_loss_fn(torch.tensor(preds[:, [0]]), torch.tensor(p.label_ids), quantiles=torch.tensor([0.5])).mean().detach().cpu()
            pinball_loss_meanpstd = pinball_loss_fn(torch.tensor(preds[:, [0]]) + torch.sqrt(torch.tensor(preds[:, [1]])), torch.tensor(p.label_ids), quantiles=torch.tensor([1-scipy.stats.norm.sf(1)])).mean().detach().cpu()
            mse = torch.nn.functional.mse_loss(torch.tensor(pred_mu), torch.tensor(p.label_ids)).detach().cpu()
            mae = torch.nn.functional.l1_loss(torch.tensor(pred_mu), torch.tensor(p.label_ids)).detach().cpu()
            result = {
                "mse_uncertainty": mse_uncertainty,
                "mse": mse,
                "mae": mae,
                "kl": kl,
                "pinball_0.5": pinball_loss_mean,
                f"pinball_loss_{1-scipy.stats.norm.sf(1)}": pinball_loss_meanpstd,
            }
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result    

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]

        return logits.detach().clone()

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            valid_mm_dataset = raw_datasets["validation_mismatched"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(valid_mm_dataset), data_args.max_eval_samples)
                valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
            eval_datasets.append(valid_mm_dataset)
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        task_name = data_args.dataset_name  if data_args.task_name is None else data_args.task_name
        tasks = [task_name]
        
        if not isinstance(predict_dataset, list):
            predict_datasets = [predict_dataset]
        else:
            predict_datasets = predict_dataset
        
        predict_split_list = predict_split.split(",")
        logger.info(f"{predict_split_list}")
        for task in tasks:
            for predict_dataset, cur_predict_split in zip(predict_datasets, predict_split_list):
                if trainer.is_world_process_zero():
                    logger.info(cur_predict_split)
                    df_pred = pd.DataFrame(predict_dataset)
                # Removing the `label` columns because it contains -1 and Trainer won't like that.
                if "label" in predict_dataset.column_names:
                    predict_dataset = predict_dataset.remove_columns("label")
                full_predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
                predictions = np.squeeze(full_predictions) if is_regression else np.argmax(full_predictions, axis=1)

                output_predict_file = f"predict_results_{task.replace('/', '_')}_{cur_predict_split}_{data_args.num_experiments}_{data_args.experiment_idx}.parquet"
                if trainer.is_world_process_zero():
                    df_pred["pred_label"] = predictions.tolist()
                    df_pred["pred_score"] = full_predictions.tolist()

                    if config.regression_type == "regression":
                        df_pred["predicted_mu"] = full_predictions[:,0].tolist()
                        df_pred["predicted_std"] = np.ones_like(full_predictions[:,0]).tolist()
                    elif config.regression_type == 'gaussian_regression':
                        df_pred["predicted_mu"] = full_predictions[:,0].tolist()
                        df_pred["predicted_std"] = np.sqrt(full_predictions[:,1]).tolist()
                    elif config.regression_type == 'iqr_regression':
                        sorted_predictions = np.sort(full_predictions, axis=1)
                        df_pred["predicted_mu"] = sorted_predictions[:,0].tolist()
                        df_pred["predicted_std"] = (sorted_predictions[:,1] - sorted_predictions[:, 0]).tolist()
                    elif config.regression_type =='mse_pinball_regression':
                        assert len(model_quantiles)==1, 'if more than one quantile is used for mse_pinball_regression I dont know what to save here '
                        sorted_predictions = np.sort(full_predictions, axis=1)
                        df_pred["predicted_mu"] = sorted_predictions[:,0].tolist()
                        df_pred["predicted_std"] = (sorted_predictions[:,1] - sorted_predictions[:, 0]).tolist()
                    elif config.regression_type == 'gaussian_pinball_regression':
                        df_pred["predicted_mu"] = full_predictions[:,0].tolist()
                        df_pred["predicted_std"] = np.sqrt(full_predictions[:,1]).tolist()

                    df_pred.to_parquet(
                        os.path.join(training_args.output_dir, output_predict_file)
                    )

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
