import pandas as pd
import numpy as np
from glob import glob
import os
import math
import scipy
import scipy.stats
import zlib
import matplotlib.pyplot as plt
import argparse
import sklearn.metrics
import itertools
import logging
import json


def main():
    # first gather file of task pred
    # we only need to read data for private split and public_test split
    if args.task_pred_dir is None:
        task_pred_dir = os.path.join(args.data_dir, args.task_name, "task_pred", args.task_seed)
    else:
        task_pred_dir = args.task_pred_dir

    logger.info(f"task_pred_dir={task_pred_dir}")

    # we first read the prediction score of the task model
    logger.info(f"Prediction for {args.predict_split_list}")
    
    norm_constant_dict = None
    for cur_predict_split in args.predict_split_list:
        cur_split_fname = os.path.join(task_pred_dir, f"predict_results_{args.task_name.replace('/', '_')}_{cur_predict_split}_{args.num_experiments}_{args.experiment_idx}.parquet")

        task_split_dataframe = pd.read_parquet(cur_split_fname)

        # min-k attack https://github.com/iamgroot42/mimir/blob/main/mimir/attacks/min_k.py
        k_min = 0.2
        task_mink_nce = -task_split_dataframe["cross_entropies"].apply(lambda x: np.mean(sorted(x)[-math.ceil(len(x) * k_min):])).values
        task_split_dataframe["mink_nce"] = task_mink_nce
        
        # z-lib attack https://github.com/iamgroot42/mimir/blob/main/mimir/attacks/zlib.py
        zlib_entropy = task_split_dataframe[args.text_column_name].apply(lambda x: len(zlib.compress(bytes(x, "utf-8")))-8).values
        zlib_score = -task_split_dataframe["cross_entropies"].apply(np.sum).values / np.clip(zlib_entropy, 1, None)
        task_split_dataframe["zlib_score"] = zlib_score

        task_nce = - task_split_dataframe["avg_cross_entropy"].values

        if norm_constant_dict is None:
            sample_mink_nce_mean = float(task_split_dataframe["mink_nce"].mean())
            sample_mink_nce_std = float(task_split_dataframe["mink_nce"].std())
            sample_zlib_score_mean = float(task_split_dataframe["zlib_score"].mean())
            sample_zlib_score_std = float(task_split_dataframe["zlib_score"].std())
           

            norm_constant_dict = {
                "sample_mink_nce_mean": sample_mink_nce_mean,
                "sample_mink_nce_std": sample_mink_nce_std,
                "sample_zlib_score_mean": sample_zlib_score_mean,
                "sample_zlib_score_std": sample_zlib_score_std,

            }

            with open(os.path.join(task_pred_dir, "extended_normalizer.json"), "w") as f:
                json.dump(norm_constant_dict, f)

        # use average cross entropy as sample level label
        task_split_dataframe["normalized_mink_nce"] = (task_split_dataframe["mink_nce"] - norm_constant_dict["sample_mink_nce_mean"]) / norm_constant_dict["sample_mink_nce_std"]
        task_split_dataframe["normalized_zlib_score"] = (task_split_dataframe["zlib_score"] - norm_constant_dict["sample_zlib_score_mean"]) / norm_constant_dict["sample_zlib_score_std"]

        task_split_dataframe.to_parquet(
            cur_split_fname
        )


def parse_args():
    parser = argparse.ArgumentParser(description='Basic args')
    parser.add_argument('--task_name', type=str, default='qnli')
    parser.add_argument('--text_column_name', type=str, default="text", help='text column name')
    parser.add_argument('--data_dir', type=str, default='/data/qmia_llm_results/', help='output dir for intermediate results')
    parser.add_argument('--task_pred_dir', type=str, default=None, help='task pred dir for intermediate results')
    parser.add_argument('--task_seed', type=str, default='42', help='task model seed') 
    parser.add_argument('--num_experiments', type=str, default=None, help='number of experiments for task')
    parser.add_argument('--experiment_idx', type=str, default=None, help='experiment idx for task')
    parser.add_argument('--predict_split', type=str, default="public_train,public_test,private,validation", help='prediction splits')

    args, _ = parser.parse_known_args()

    return args

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s %(asctime)-15s pid-%(process)d: %(message)s'
    )
    logger = logging.getLogger(__name__)

    args = parse_args()

    args.predict_split_list = args.predict_split.split(",")

    main()
