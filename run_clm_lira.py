#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
# Copyright 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
#
# This code is based on work by HuggingFace Inc. and has been modified by Amazon.com, Inc. or its affiliates.
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
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import random
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import json
import datasets
import evaluate
import torch
from datasets import load_dataset
import numpy as np
import scipy.special
import scipy.stats
import pandas as pd
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    DataCollatorForTokenClassification,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from tokenizers.processors import TemplateProcessing

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.39.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

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
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
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
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
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
    #TODO! new data splitting options :(
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
    lira_train_ratio: Optional[float] = field(
        default=0.9,
        metadata={"help": "Lira specific setting, how much of the public data to use for training (inject data variability)"}
    )
    pkeep: Optional[float] = field(
        default=0.5,
        metadata={
            "help": (
                "percentage split of private data for MIA"
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
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    group_text: bool = field(default=False, metadata={"help": "Whether to group text into blocks"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    predict_split: Optional[str] = field(default='private,public_train,public_validation', metadata={"help": "split for prediction"})
    predict_normalize: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to normalize the prediction."
            )
        },
    )
    text_column_name: Optional[str] = field(
        default=None, 
        metadata={"help": "text column name"}
    )
    predict_chunk_size: Optional[int] = field(
        default=1024,
        metadata={"help": "chunk size for using trainer for prediction"}
    )
    save_by_chunk: Optional[bool] = field(
        default=False, metadata={"help": "whether to save in chunks"}
    )
    add_bos_eos: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to add bos and eos tokens to the text"}
    )
    sample_min_num_chars: Optional[int] = field(
        default=0,
        metadata={"help": "filter sample less than given number of chars."}
    )
    shift_text_pos: Optional[int] = field(
        default=0,
        metadata={"help": "shift samples by given number of tokens"}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt", "parquet"], "`train_file` should be a csv, a json or a txt or a parquet file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt", "parquet"], "`validation_file` should be a csv, a json or a txt or a parquet file."


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
    send_example_telemetry("run_clm", model_args, data_args)

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

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None and data_args.train_file is None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
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
    logger.info(f"dataset_name={data_args.dataset_name}")

    # Dataset splitting. Train - Train/Public
    # MB
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

        if data_args.experiment_idx is not None and data_args.lira_train_ratio:
            #further subsample public train to add data variation
            with temp_seed(42+data_args.experiment_idx):
                np.random.shuffle(public_train_indices)
                public_train_indices = public_train_indices[:int(data_args.lira_train_ratio * len(public_train_indices))]
                public_train_indices = list(sorted(public_train_indices))

#         raw_datasets['public_private'] = raw_datasets["train"]
        raw_datasets['public_train'] = raw_datasets["train"].select(public_train_indices)
        raw_datasets['public_test'] = raw_datasets["train"].select(public_val_indices)
        raw_datasets['private'] = raw_datasets["train"].select(private_indices)
        raw_datasets.pop("train")

    logger.info(f"Using {train_split_name} for training")

    if data_args.sample_min_num_chars > 0:
        logger.info(f"Filter samples with less than {data_args.sample_min_num_chars} chars.")
        for split, dataset in raw_datasets.items():
            raw_datasets[split] = dataset.filter(lambda example: len(example[data_args.text_column_name]) >= data_args.sample_min_num_chars)


    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    logger.info(f"tokenizer_kwargs={tokenizer_kwargs}")
    
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    logger.info(f"tokenizer={type(tokenizer)}")

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

    # add [BOS] and [EOS] to the text
    if data_args.add_bos_eos:
        logger.info(f"Add BOS={tokenizer.bos_token}, EOS={tokenizer.eos_token}")
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{tokenizer.bos_token} $A {tokenizer.eos_token}",
            special_tokens=[(tokenizer.bos_token, tokenizer.bos_token_id), (tokenizer.eos_token, tokenizer.eos_token_id)],
        )
    logger.info(f"Encoded emtpy string:{tokenizer('')}")

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    
    if training_args.do_train:
        column_names = list(raw_datasets[train_split_name].features)
    else:
        column_names = list(raw_datasets["validation"].features)

    if data_args.text_column_name is None:
        text_column_name = "text" if "text" in column_names else column_names[0]
    else:
        text_column_name = data_args.text_column_name
    
    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    

    # For preprocessing, we will adopt similar style as token-classification
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    if hasattr(config, "max_position_embeddings"):
        max_pos_embeddings = config.max_position_embeddings
    else:
        # Define a default value if the attribute is missing in the config.
        max_pos_embeddings = 1024

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > max_pos_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, max_pos_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            if max_pos_embeddings > 0:
                block_size = min(1024, max_pos_embeddings)
            else:
                block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    
    if data_args.group_text:
        def tokenize_function(examples):
            with CaptureLogger(tok_logger) as cl:
                output = tokenizer(examples[text_column_name])
            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                    " before being passed to the model."
                )
            return output
    else:
        def tokenize_function(examples):
            tokenized_inputs = tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=block_size
            )

            return tokenized_inputs

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names if data_args.group_text else None,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names if data_args.group_text else None,
            )

    
            
    if data_args.group_text:
        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            # rename original label
            if "label" in result:
                result["orig_label"] = result.pop("label")
            if "labels" in result:
                result["orig_labels"] = result.pop("labels")
            
            result["labels"] = result["input_ids"].copy()
            return result
        group_text_desc=f"Grouping texts in chunks of {block_size}"
    else:
        def group_texts(examples):
            # rename original label
            if "label" in examples:
                examples["orig_label"] = examples.pop("label")
            if "labels" in examples:
                examples["orig_labels"] = examples.pop("labels")
                
            examples["labels"] = examples["input_ids"].copy()
            return examples
        group_text_desc=f"Single texts in chunks of {block_size}"        
    
    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=group_text_desc,
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )

    if not data_args.group_text and data_args.shift_text_pos:
        def shift_texts(examples):
            # shift text
            examples["input_ids"] = [[tokenizer.pad_token_id] * data_args.shift_text_pos + input_ids for input_ids in examples["input_ids"]]
            examples["attention_mask"] = [[0] * data_args.shift_text_pos + attention_mask for attention_mask in examples["attention_mask"]]
            examples["labels"] = examples["input_ids"].copy()
            
            return examples
        shift_text_desc=f"Shift text by {data_args.shift_text_pos}"
        
        with training_args.main_process_first(desc="grouping texts together"):
            if not data_args.streaming:
                lm_datasets = lm_datasets.map(
                    shift_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=shift_text_desc,
                )
            else:
                lm_datasets = lm_datasets.map(
                    shift_texts,
                    batched=True,
                )        
            
    if training_args.do_train:
        if train_split_name not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets[train_split_name]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    if training_args.do_predict or data_args.test_file is not None:
        if data_args.predict_split is None:
            if "test" not in raw_datasets:
                raise ValueError("--do_predict requires a test dataset")
            predict_split = "test"
            predict_dataset = [lm_datasets["test"]]
        elif data_args.predict_split is not None:
            predict_dataset = [
                lm_datasets[p_split] for p_split in data_args.predict_split.split(",")
            ]
            logger.info(f"predict on {data_args.predict_split}")
            predict_split = data_args.predict_split
        if data_args.max_predict_samples is not None:
            predict_dataset = [d.select(range(min(len(d), data_args.max_predict_samples))) for d in predict_dataset]


    if data_args.group_text:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForTokenClassification(
            tokenizer, 
            padding="longest",
            pad_to_multiple_of=8
        )


    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the {train_split_name} set: {train_dataset[index]}.")

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    if training_args.do_predict:
        logger.info("*** Predict ***")
        
        # redefine preprocess function and compute_metrics for prediction        
        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]

            logits = logits[:, :-1, :]
            labels = labels[:, 1:]

            excluded_logits = logits.detach().clone()
            sample_indices = torch.arange(labels.shape[0]).repeat_interleave(labels.shape[1])
            token_indices = torch.arange(labels.shape[1]).repeat(labels.shape[0])
            
            # label for masked tokens would have value -100 by default
            masked_sample_indices, masked_token_indices = torch.where(labels < 0)

            # set excluded indices to -inf, if masked indices, just set last index to -inf
            excluded_logits[
                sample_indices,
                token_indices,
                labels.clamp(min=-1).reshape(-1),
            ] = -np.inf

            logsum_scores = logits[sample_indices, token_indices, labels.clamp(min=-1).reshape(-1)].reshape(labels.shape) - torch.logsumexp(excluded_logits, axis=-1)
            hinge_scores = logits[sample_indices, token_indices, labels.clamp(min=-1).reshape(-1)].reshape(labels.shape) - torch.amax(excluded_logits, axis=-1)
            pred_labels = logits.argmax(dim=-1)
            cross_entropies = torch.nn.functional.cross_entropy(
                logits.transpose(1, 2),
                labels,
                reduction="none"
            )

            logsum_scores[masked_sample_indices, masked_token_indices] = -100.
            hinge_scores[masked_sample_indices, masked_token_indices] = -100.
            cross_entropies[masked_sample_indices, masked_token_indices] = -100.
            pred_labels[masked_sample_indices, masked_token_indices] = -100

            return (pred_labels, logsum_scores, hinge_scores, cross_entropies)

        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            pred_labels, logsum_scores, hinge_scores, cross_entropies = preds
            
            labels = labels[:, 1:].reshape(-1)
            pred_labels = pred_labels.reshape(-1)
#             pred_labels = pred_labels[:, :-1].reshape(-1)
            return metric.compute(predictions=pred_labels, references=labels)
        
        # set trainer to store logsum and hinge score
        trainer.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        trainer.compute_metrics = compute_metrics
        
        def predict_in_chunks(dataset, chunk_size, task, cur_predict_split):
            # we need to predict in chunk_size as vocab_size is large
            pred_length_list = []
            pred_labels_list = []
            logsum_scores_list = []
            hinge_scores_list = []
            cross_entropies_list = []
            num_chunks = (len(dataset) + chunk_size - 1) // chunk_size  # Calculate number of batches
            chunk_metrics_list = []
            chunk_size_list = []
            for i in tqdm(range(num_chunks)):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(dataset))
                chunk_size_list.append(end_idx - start_idx)
                dataset_chunk = dataset.select(range(start_idx,end_idx))
                chunk_predictions, chunk_labels, chunk_metrics = trainer.predict(dataset_chunk, metric_key_prefix="predict")
                chunk_pred_labels, chunk_logsum_scores, chunk_hinge_scores, chunk_cross_entropies = chunk_predictions
                
                chunk_pred_labels_list = chunk_pred_labels.tolist()
                chunk_logsum_scores_list = chunk_logsum_scores.tolist()
                chunk_hinge_scores_list = chunk_hinge_scores.tolist()
                chunk_cross_entropies_list = chunk_cross_entropies.tolist()
                chunk_pred_length_list = [(x[data_args.shift_text_pos:].index(-100) + data_args.shift_text_pos)  if -100 in x[data_args.shift_text_pos:] else len(x) for x in chunk_pred_labels_list]

                for i_item, length in enumerate(chunk_pred_length_list):
                    if length == 0:
                        logger.info(chunk_pred_labels_list[i_item])
                        logger.info(dataset_chunk[i_item])
                        raise ValueError

                filtered_chunk_pred_labels_list = [lst[data_args.shift_text_pos:k] for lst, k in  zip(chunk_pred_labels_list, chunk_pred_length_list)]
                filtered_chunk_logsum_scores_list = [lst[data_args.shift_text_pos:k] for lst, k in  zip(chunk_logsum_scores_list, chunk_pred_length_list)]
                filtered_chunk_hinge_scores_list = [lst[data_args.shift_text_pos:k] for lst, k in  zip(chunk_hinge_scores_list, chunk_pred_length_list)]
                filtered_chunk_cross_entropies_list = [lst[data_args.shift_text_pos:k] for lst, k in  zip(chunk_cross_entropies_list, chunk_pred_length_list)]

                if not data_args.save_by_chunk:
                    pred_length_list += chunk_pred_length_list
                    pred_labels_list += filtered_chunk_pred_labels_list
                    logsum_scores_list += filtered_chunk_logsum_scores_list
                    hinge_scores_list += filtered_chunk_hinge_scores_list
                    cross_entropies_list += filtered_chunk_cross_entropies_list

                    chunk_metrics_list.append(chunk_metrics)
                else:
                    # if save by chunk
                    pred_length_list = chunk_pred_length_list
                    pred_labels_list = filtered_chunk_pred_labels_list
                    logsum_scores_list = filtered_chunk_logsum_scores_list
                    hinge_scores_list = filtered_chunk_hinge_scores_list
                    cross_entropies_list = filtered_chunk_cross_entropies_list
                    chunk_metrics_list = [chunk_metrics]
                    chunk_size_list = [1]
                    
                    if trainer.is_world_process_zero():
                        df_pred = pd.DataFrame(dataset_chunk)
                        df_pred["pred_length"] = pred_length_list
                        df_pred["pred_labels"] = pred_labels_list
                        df_pred["logsum_scores"] = logsum_scores_list
                        df_pred["hinge_scores"] = hinge_scores_list
                        df_pred["cross_entropies"] = cross_entropies_list
                        df_pred["avg_cross_entropy"] = df_pred["cross_entropies"].apply(np.mean)
                        df_pred["avg_hinge_score"] = df_pred["hinge_scores"].apply(np.mean)
                        df_pred["labels"] = df_pred["hinge_scores"]
                        # use average cross entropy as sample level label
                        df_pred["label"] = - df_pred["avg_cross_entropy"]

                        output_predict_file = f"predict_results_{task.replace('/', '_')}_{cur_predict_split}_{data_args.num_experiments}_{data_args.experiment_idx}_part_{i:03d}_of_{num_chunks:03d}.parquet"
                        df_pred.to_parquet(
                            os.path.join(training_args.output_dir, output_predict_file)
                        )
                        
            all_metrics = dict()
            for key in chunk_metrics_list[0]:
                if key.endswith("runtime"):
                    all_metrics[key] = float(np.sum([m[key] for m in chunk_metrics_list]))
                else:
                    all_metrics[key] = float(np.average([m[key] for m in chunk_metrics_list], weights=chunk_size_list))                    
                
            return pred_length_list, pred_labels_list, logsum_scores_list, hinge_scores_list, cross_entropies_list, all_metrics
        
        logger.info(f"use_legacy_prediction_loop={trainer.args.use_legacy_prediction_loop}")

        tasks = [data_args.dataset_name]
        if not isinstance(predict_dataset, list):
            predict_datasets = [predict_dataset]
        else:
            predict_datasets = predict_dataset
        predict_split_list = predict_split.split(",")
        logger.info(f"Prediction for {predict_split_list}")
        norm_constant_dict = None
        for task in tasks:
            for predict_dataset, cur_predict_split in zip(predict_datasets, predict_split_list):
                if trainer.is_world_process_zero():
                    print(cur_predict_split)
                    if not data_args.save_by_chunk:
                        df_pred = pd.DataFrame(predict_dataset)

                pred_length_list, pred_labels_list, logsum_scores_list, hinge_scores_list, cross_entropies_list, metrics = predict_in_chunks(predict_dataset, data_args.predict_chunk_size, task, cur_predict_split)
                
                output_predict_file = f"predict_results_{task.replace('/', '_')}_{cur_predict_split}_{data_args.num_experiments}_{data_args.experiment_idx}.parquet"
                if trainer.is_world_process_zero():
                    if not data_args.save_by_chunk:
                        # remove masked
                        df_pred["pred_length"] = pred_length_list
                        df_pred["pred_labels"] = pred_labels_list
                        df_pred["logsum_scores"] = logsum_scores_list
                        df_pred["hinge_scores"] = hinge_scores_list
                        df_pred["cross_entropies"] = cross_entropies_list
                        df_pred["avg_cross_entropy"] = df_pred["cross_entropies"].apply(np.mean)
                        df_pred["avg_hinge_score"] = df_pred["hinge_scores"].apply(np.mean)

                        if data_args.predict_normalize:
                            if norm_constant_dict is None:
                                kept_hinge_scores = np.concatenate(df_pred["hinge_scores"])

                                token_hinge_score_mean = float(np.mean(kept_hinge_scores))
                                token_hinge_score_std = float(np.std(kept_hinge_scores))
                                sample_cross_entropy_mean = float(df_pred["avg_cross_entropy"].mean())
                                sample_cross_entropy_std = float(df_pred["avg_cross_entropy"].std())
                                sample_hinge_score_mean = float(df_pred["avg_hinge_score"].mean())
                                sample_hinge_score_std = float(df_pred["avg_hinge_score"].std())

                                norm_constant_dict = {
                                    "token_hinge_score_mean": token_hinge_score_mean,
                                    "token_hinge_score_std": token_hinge_score_std,
                                    "sample_cross_entropy_mean": sample_cross_entropy_mean,
                                    "sample_cross_entropy_std": sample_cross_entropy_std,
                                    "sample_hinge_score_mean": sample_hinge_score_mean,
                                    "sample_hinge_score_std": sample_hinge_score_std,
                                }

                                with open(os.path.join(training_args.output_dir, "normalizer.json"), "w") as f:
                                    json.dump(norm_constant_dict, f)

                            # we just directly overwrite labels as it is the same as input_ids
                            df_pred["labels"] = df_pred["hinge_scores"].apply(lambda y: [(x - norm_constant_dict["token_hinge_score_mean"]) / norm_constant_dict["token_hinge_score_std"] for x in y])
                            # use average cross entropy as sample level label
                            df_pred["normalized_avg_negative_cross_entropy"] = - (df_pred["avg_cross_entropy"] - norm_constant_dict["sample_cross_entropy_mean"]) / norm_constant_dict["sample_cross_entropy_std"]
                            df_pred["normalized_avg_hinge_score"] = (df_pred["avg_hinge_score"] - norm_constant_dict["sample_hinge_score_mean"]) / norm_constant_dict["sample_hinge_score_std"]
                            df_pred["label"] = df_pred["normalized_avg_negative_cross_entropy"]
                        else:
                            df_pred["labels"] = df_pred["hinge_scores"]
                            # use average cross entropy as sample level label
                            df_pred["label"] = - df_pred["avg_cross_entropy"]


                        df_pred.to_parquet(
                            os.path.join(training_args.output_dir, output_predict_file)
                        )

                    trainer.log_metrics(cur_predict_split, metrics)
                    trainer.save_metrics(cur_predict_split, metrics)
        
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
