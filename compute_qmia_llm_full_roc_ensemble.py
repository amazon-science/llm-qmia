import pandas as pd
import numpy as np
from glob import glob
import os
import math
import scipy
import scipy.stats
import scipy.special
import zlib
import matplotlib.pyplot as plt
import argparse
import sklearn.metrics
import itertools
import logging


def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = sklearn.metrics.roc_curve(x, score)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, sklearn.metrics.auc(fpr, tpr), acc


def do_plot(metric, legend, scores, labels):
    if args.flip_score == 1:
        scores = -scores

    fpr, tpr, auc, acc = sweep(
        scores,
        labels, 
    )

    low = tpr[np.where(fpr<.001)[0][-1]]
    low_2 = tpr[np.where(fpr<.01)[0][-1]]

    full_metric_test = 'Attack %s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f TPR@1%%FPR of %.4f'%(legend, auc,acc, low, low_2)
    full_metric_dict = {
        "legend": legend,
        "auc": auc,
        "acc": acc,
        "tpr@0.1%": low,
        "tpr@1%": low_2,
    }
    
    print(full_metric_test)

    metric_text = ''
    if metric == 'auc':
        metric_text = 'auc=%.3f'%auc
    elif metric == 'acc':
        metric_text = 'acc=%.3f'%acc
    
    if legend.startswith("lira"):
        plt.plot(fpr, tpr, label=legend+metric_text, linestyle='dotted')
    elif legend.startswith("mismatched-lira"):
        plt.plot(fpr, tpr, label=legend+metric_text, linestyle='dashed')
    else:
        plt.plot(fpr, tpr, label=legend+metric_text, linestyle='-')
    
    return full_metric_test, full_metric_dict


def plot_figure(legend_list, scores_list, labels, figure_fname, metric_fname, metric='auc', min_xlim=1e-4, min_ylim=1e-4):
    full_metric_text_list = list()
    full_metric_dict_list = list()
    plt.figure(figsize=[12, 10])
    for legend, scores in zip(legend_list, scores_list):
        metric_text, metric_dict = do_plot(metric, legend, scores, labels)
        full_metric_text_list.append(metric_text)
        full_metric_dict_list.append(metric_dict)
            
    plt.semilogx()
    plt.semilogy()
    plt.xlim(min_xlim,1)
    plt.ylim(min_ylim,1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls='--', color='gray')
    plt.legend(fontsize=10)
    plt.savefig(figure_fname)
    
    with open(metric_fname, "w") as f:
        f.write("\n".join(full_metric_text_list))
        
    return full_metric_dict_list

def main():
    # gather files of predicted results
    regression_types = ['gaussian_regression', 'iqr_regression', 'mse_pinball_regression', 'gaussian_pinball_regression']
    mink_regression_types = [f'{x}_mink' for x in regression_types]
    zlib_regression_types = [f'{x}_zlib' for x in regression_types]
    regression_types = regression_types + mink_regression_types + zlib_regression_types

    # first gather file of task pred
    if args.task_pred_dir is None:
        task_pred_dir = os.path.join(args.data_dir, args.task_name, "task_pred", args.task_seed)
    else:
        task_pred_dir = args.task_pred_dir

    logger.info(f"task_pred_dir={task_pred_dir}")

    regression_pred_dirs = [
        (r_type, [(seed, pred_dir) for seed in args.seed_list for pred_dir in glob(os.path.join(args.regression_pred_dir_template.format(r_type), seed), recursive=True)])
        for r_type in regression_types
    ]

    regression_pred_dirs = [
        (r_type, pred_dirs)
        for r_type, pred_dirs in regression_pred_dirs if pred_dirs
    ]

    logger.info(f"regression_pred_dirs={regression_pred_dirs}")

    lira_pred_dirs = None
    num_lira_experiments = 0
    if args.lira_pred_dir_template is not None:
        lira_pred_dirs = [glob(args.lira_pred_dir_template.format(model_name=args.target_model, checkpoint=checkpoint_num)) for checkpoint_num in args.shadow_checkpoint_num_list]
        lira_pred_dirs = [_dir for _dirs in lira_pred_dirs for _dir in _dirs]
        lira_pred_dirs.sort(key=lambda x: int(x[x.find("expid_")+len("expid_"): x.find("expid_")+x[x.find("expid_"):].find("/")]))

        num_lira_experiments = len(lira_pred_dirs)

    logger.info(f"lira_pred_dirs={lira_pred_dirs}")

    mismatched_lira_pred_dirs_dict = dict()
    for mismatched_model in args.mismatched_lira_model_list:
        print(args.lira_pred_dir_template.format(model_name=mismatched_model, checkpoint="({})".format("|".join(args.shadow_checkpoint_num_list))))
        mismatched_lira_pred_dirs = [glob(args.lira_pred_dir_template.format(model_name=mismatched_model, checkpoint=checkpoint_num)) for checkpoint_num in args.shadow_checkpoint_num_list]
        mismatched_lira_pred_dirs = [_dir for _dirs in mismatched_lira_pred_dirs for _dir in _dirs]
        mismatched_lira_pred_dirs.sort(key=lambda x: int(x[x.find("expid_")+len("expid_"): x.find("expid_")+x[x.find("expid_"):].find("/")]))
        mismatched_lira_pred_dirs_dict[mismatched_model] = mismatched_lira_pred_dirs

    logger.info(f"mismatched_lira_pred_dirs_dict={mismatched_lira_pred_dirs_dict}")

    # we first read the prediction score of the task model
    task_private_fname = glob(os.path.join(task_pred_dir, f"predict_results_{args.task_name.replace('/', '_')}_{args.private_split}_*.parquet"))[0]
    task_public_fname = glob(os.path.join(task_pred_dir, f"predict_results_{args.task_name.replace('/', '_')}_{args.public_split}_*.parquet"))[0]
    
    task_private_dataframe = pd.read_parquet(task_private_fname)
    task_public_dataframe = pd.read_parquet(task_public_fname)

    task_dataframe = pd.concat([task_private_dataframe, task_public_dataframe])
    
    is_private = np.concatenate((np.ones(task_private_dataframe.shape[0]), np.zeros(task_public_dataframe.shape[0])))
    task_scores = task_dataframe["label"].values
    task_nce = - task_dataframe["avg_cross_entropy"].values
    task_hinge = task_dataframe["hinge_scores"].apply(np.mean).values
    task_norm_mink = task_dataframe["normalized_mink_nce"].values
    task_norm_zlib = task_dataframe["normalized_zlib_score"].values
    pred_lengths = task_dataframe["pred_length"].values
    pred_length_low = int(np.quantile(pred_lengths, 0.25))
    pred_length_upp = int(np.quantile(pred_lengths, 0.75))
    pred_length_min = np.amin(pred_lengths)
    pred_length_max = np.amax(pred_lengths)

    low_length_indices = np.where(pred_lengths <= pred_length_low)[0]
    mid_length_indices = np.where((pred_lengths > pred_length_low) & (pred_lengths <= pred_length_upp))[0]
    upp_length_indices = np.where(pred_lengths > pred_length_upp)[0]

    # min-k attack https://github.com/iamgroot42/mimir/blob/main/mimir/attacks/min_k.py
    def min_k_prob(x, k_min=0.2):
        n_k_min = math.ceil(len(x) * k_min)
        min_k_logits = -np.array(sorted(x)[-n_k_min:])
        logsum_min_k_probs = scipy.special.logsumexp(min_k_logits)
        logmean_min_k_probs = logsum_min_k_probs - math.log(n_k_min)

        return np.exp(logmean_min_k_probs)

    # min-k
    k_min = 0.2
    task_mink_nce = -task_dataframe["cross_entropies"].apply(lambda x: np.mean(sorted(x)[-math.ceil(len(x) * k_min):])).values
    task_mink_prob = task_dataframe["cross_entropies"].apply(lambda x: min_k_prob(x, k_min=k_min)).values

    # z-lib attack https://github.com/iamgroot42/mimir/blob/main/mimir/attacks/zlib.py
    task_zlib_entropy = task_dataframe[args.text_column_name].apply(lambda x: len(zlib.compress(bytes(x, "utf-8")))-8).values
    task_zlib_score = -task_dataframe["cross_entropies"].apply(np.sum).values / np.clip(task_zlib_entropy, 1, None)

    # task hinge prob
    baseline_legend_list = ["yeom:", "min_k:", "min_k_prob:", "zlib:"]
    baseline_score_list = [task_scores, task_mink_nce, task_mink_prob, task_zlib_score]


    # get lira preds
    lira_legend_list = list()
    lira_fixed_legend_list = list()
    lira_zscore_list = list()
    lira_pred_diff_list = list()
    if lira_pred_dirs:
        shadow_nce_list = list()
        shadow_hinge_list = list()
        shadow_mink_nce_list = list()
        shadow_mink_prob_list = list()
        shadow_zlib_score_list = list()
        for shadow_pred_dir in lira_pred_dirs:
            shadow_private_fanme = glob(os.path.join(shadow_pred_dir, f"predict_results_{args.task_name.replace('/', '_')}_{args.private_split}_*.parquet"))
            shadow_public_fname = glob(os.path.join(shadow_pred_dir, f"predict_results_{args.task_name.replace('/', '_')}_{args.public_split}_*.parquet"))
            if not shadow_private_fanme or not shadow_public_fname:
                continue
            shadow_private_fanme = shadow_private_fanme[0]
            shadow_public_fname = shadow_public_fname[0]

            shadow_private_dataframe = pd.read_parquet(shadow_private_fanme)
            shadow_public_dataframe = pd.read_parquet(shadow_public_fname)

            shadow_dataframe = pd.concat([shadow_private_dataframe, shadow_public_dataframe])
            shadow_nce = - shadow_dataframe["avg_cross_entropy"].values
            shadow_hinge = shadow_dataframe["hinge_scores"].apply(np.mean).values
            shadow_mink_nce = -shadow_dataframe["cross_entropies"].apply(lambda x: np.mean(sorted(x)[-math.ceil(len(x) * k_min):])).values
            shadow_mink_prob = shadow_dataframe["cross_entropies"].apply(lambda x: min_k_prob(x, k_min=k_min)).values
            shadow_zlib_entropy = shadow_dataframe[args.text_column_name].apply(lambda x: len(zlib.compress(bytes(x, "utf-8")))-8).values
            shadow_zlib_score = -shadow_dataframe["cross_entropies"].apply(np.sum).values / np.clip(shadow_zlib_entropy, 1, None)

            shadow_nce_list.append(shadow_nce)
            shadow_hinge_list.append(shadow_hinge)
            shadow_mink_nce_list.append(shadow_mink_nce)
            shadow_mink_prob_list.append(shadow_mink_prob)
            shadow_zlib_score_list.append(shadow_zlib_score)

        all_shadow_nce = np.stack(shadow_nce_list).T
        all_shadow_hinge = np.stack(shadow_hinge_list).T
        all_shadow_mink_nce = np.stack(shadow_mink_nce_list).T
        all_shadow_mink_prob = np.stack(shadow_mink_prob_list).T
        all_shadow_zlib_score = np.stack(shadow_zlib_score_list).T

        num_lira_experiments = len(shadow_hinge_list)
        logger.info(f"lira {num_lira_experiments}")
        for lira_target, all_shadow_target_score, task_target_score in zip(["nce", "hinge", "mink_nce", "mink_prob", "zlib"], [all_shadow_nce, all_shadow_hinge, all_shadow_mink_nce, all_shadow_mink_prob, all_shadow_zlib_score], [task_nce, task_hinge, task_mink_nce, task_mink_prob, task_zlib_score]):
            for k in [2, 4, 6, 8, 16]:
                if k > min(num_lira_experiments, args.max_num_lira_experiments):
                    break
                lira_mean = np.mean(all_shadow_target_score[:, :k], axis=1)
                lira_std = np.std(all_shadow_target_score[:, :k], axis=1)

                logger.info(f"{lira_target} n={k}:min_lira_std={np.amin(lira_std)} max_lira_std={np.amax(lira_std)}")
                
                lira_pred_diff = task_target_score - lira_mean
                if lira_target == "mink_prob":
                    lira_z_score = lira_pred_diff / np.maximum(lira_std, 1e-8)
                else:
                    lira_z_score = lira_pred_diff / np.maximum(lira_std, 1e-4)

                
                lira_legend_list.append(f"lira-{lira_target}-{k}:")
                lira_fixed_legend_list.append(f"lira-{lira_target}-{k}-fixed:")
                lira_zscore_list.append(lira_z_score)
                lira_pred_diff_list.append(lira_pred_diff)


    # get lira preds of mismatched models
    mismatched_lira_legend_list = list()
    mismatched_lira_fixed_legend_list = list()
    mismatched_lira_zscore_list = list()
    mismatched_lira_pred_diff_list = list()
    for model_name in args.mismatched_lira_model_list:
        lira_pred_dirs = mismatched_lira_pred_dirs_dict[model_name]
        if lira_pred_dirs:            
            shadow_nce_list = list()
            shadow_hinge_list = list()
            shadow_mink_nce_list = list()
            shadow_mink_prob_list = list()
            shadow_zlib_score_list = list()
            for shadow_pred_dir in lira_pred_dirs:
                shadow_private_fanme = glob(os.path.join(shadow_pred_dir, f"predict_results_{args.task_name.replace('/', '_')}_{args.private_split}_*.parquet"))
                shadow_public_fname = glob(os.path.join(shadow_pred_dir, f"predict_results_{args.task_name.replace('/', '_')}_{args.public_split}_*.parquet"))
                if not shadow_private_fanme or not shadow_public_fname:
                    continue
                shadow_private_fanme = shadow_private_fanme[0]
                shadow_public_fname = shadow_public_fname[0]

                shadow_private_dataframe = pd.read_parquet(shadow_private_fanme)
                shadow_public_dataframe = pd.read_parquet(shadow_public_fname)

                shadow_dataframe = pd.concat([shadow_private_dataframe, shadow_public_dataframe])
                shadow_nce = - shadow_dataframe["avg_cross_entropy"].values
                shadow_hinge = shadow_dataframe["hinge_scores"].apply(np.mean).values
                shadow_mink_nce = -shadow_dataframe["cross_entropies"].apply(lambda x: np.mean(sorted(x)[-math.ceil(len(x) * k_min):])).values
                shadow_mink_prob = shadow_dataframe["cross_entropies"].apply(lambda x: min_k_prob(x, k_min=k_min)).values
                shadow_zlib_entropy = shadow_dataframe[args.text_column_name].apply(lambda x: len(zlib.compress(bytes(x, "utf-8")))-8).values
                shadow_zlib_score = -shadow_dataframe["cross_entropies"].apply(np.sum).values / np.clip(shadow_zlib_entropy, 1, None)

                shadow_nce_list.append(shadow_nce)
                shadow_hinge_list.append(shadow_hinge)
                shadow_mink_nce_list.append(shadow_mink_nce)
                shadow_mink_prob_list.append(shadow_mink_prob)
                shadow_zlib_score_list.append(shadow_zlib_score)

            all_shadow_nce = np.stack(shadow_nce_list).T
            all_shadow_hinge = np.stack(shadow_hinge_list).T
            all_shadow_mink_nce = np.stack(shadow_mink_nce_list).T
            all_shadow_mink_prob = np.stack(shadow_mink_prob_list).T
            all_shadow_zlib_score = np.stack(shadow_zlib_score_list).T

            num_lira_experiments = len(shadow_hinge_list)
            logger.info(f"mismatched_model {model_name} lira {num_lira_experiments}")
            for lira_target, all_shadow_target_score, task_target_score in zip(["nce", "hinge", "mink_nce", "mink_prob", "zlib"], [all_shadow_nce, all_shadow_hinge, all_shadow_mink_nce, all_shadow_mink_prob, all_shadow_zlib_score], [task_nce, task_hinge, task_mink_nce, task_mink_prob, task_zlib_score]):
                largest_k = 0
                for k in [2, 4, 6, 8, 16]:
                    if k > min(num_lira_experiments, args.max_num_lira_experiments):
                        break
                    largest_k = k

                for k in [2, 4, 6, 8, 16]:
                    if k > min(num_lira_experiments, args.max_num_lira_experiments):
                        break
                    if not args.mismatch_plot_all and k < largest_k:
                        continue
                    lira_mean = np.mean(all_shadow_target_score[:, :k], axis=1)
                    lira_std = np.std(all_shadow_target_score[:, :k], axis=1)
    
                    logger.info(f"{lira_target} n={k}:min_lira_std={np.amin(lira_std)} max_lira_std={np.amax(lira_std)}")
                    
                    lira_pred_diff = task_target_score - lira_mean
                    if lira_target == "mink_prob":
                        lira_z_score = lira_pred_diff / np.maximum(lira_std, 1e-8)
                    else:
                        lira_z_score = lira_pred_diff / np.maximum(lira_std, 1e-4)

                    mismatched_lira_legend_list.append(f"mismatched-lira-{model_name}-{lira_target}-{k}:")
                    mismatched_lira_fixed_legend_list.append(f"mismatched-lira-{model_name}-{lira_target}-{k}-fixed:")

                    mismatched_lira_zscore_list.append(lira_z_score)
                    mismatched_lira_pred_diff_list.append(lira_pred_diff)


    regression_suffix = "_regression"

    os.makedirs(args.output_dir, exist_ok=True)
    # for each regression_type, we compute the z-score for all the seeds
    # we additionally create the z-score for the ensemble (if using fixed std, just the mean without assuming std)
    ensemble_pred_diff_dict = dict()
    ensemble_z_score_dict = dict()
    for regression_type, pred_dirs in regression_pred_dirs:
        logger.info(regression_type)
        cur_seed_pred_diff_dict = dict()
        cur_seed_z_score_dict = dict()
        pred_mean_list = list()
        pred_std_list = list()
        for seed, pred_dir in pred_dirs:          
            private_fname = glob(os.path.join(pred_dir, f"predict_results_{args.task_name.replace('/', '_')}_{args.private_split}_*.parquet"))
            public_fname = glob(os.path.join(pred_dir, f"predict_results_{args.task_name.replace('/', '_')}_{args.public_split}_*.parquet"))

            if not private_fname or not public_fname:
                continue
            private_fname = private_fname[0]
            public_fname = public_fname[0]

            regression_private_dataframe = pd.read_parquet(private_fname)
            regression_public_dataframe = pd.read_parquet(public_fname)

            if "predicted_mu" not in regression_private_dataframe.columns:
                logger.info("extracting mean and std")
                dataframe_compute_mean_std(regression_private_dataframe, regression_type)
                dataframe_compute_mean_std(regression_public_dataframe, regression_type)
            regression_dataframe = pd.concat([regression_private_dataframe, regression_public_dataframe]).copy()

            pred_mean = regression_dataframe["predicted_mu"].values
            pred_std = regression_dataframe["predicted_std"].values
            pred_std = np.maximum(pred_std, 1e-4)

            pred_mean_list.append(pred_mean)
            pred_std_list.append(pred_std)

            if regression_type.endswith("mink"):
                pred_diff = task_norm_mink - pred_mean
            elif regression_type.endswith("zlib"):
                pred_diff = task_norm_zlib - pred_mean
            else:
                pred_diff = task_scores - pred_mean
            z_score = pred_diff / pred_std

            cur_seed_pred_diff_dict[seed] = pred_diff
            cur_seed_z_score_dict[seed] = z_score

        cur_pred_mean = np.stack(pred_mean_list).T
        cur_pred_std = np.stack(pred_std_list).T

        ensemble_pred_mean = np.mean(cur_pred_mean, axis=1)
        ensemble_pred_std = np.sqrt(np.mean(cur_pred_std ** 2 + cur_pred_mean ** 2, axis=1) - ensemble_pred_mean ** 2)

        if regression_type.endswith("mink"):
            task_target = task_norm_mink
        elif regression_type.endswith("zlib"):
            task_target = task_norm_zlib
        else:
            task_target = task_scores

        ensemble_pred_dff = task_target - ensemble_pred_mean
        ensemble_z_score = ensemble_pred_dff / ensemble_pred_std

        ensemble_pred_diff_dict[regression_type] = ensemble_pred_dff
        ensemble_z_score_dict[regression_type] = ensemble_z_score


        # plot figure with estimated std
        z_score_legend_list = ["yeom:", f"{regression_type.replace(regression_suffix, '')}_ensemble:"] + [f"{regression_type.replace(regression_suffix, '')}_seed={seed}:" for seed in cur_seed_z_score_dict]
        z_score_list = [task_scores, ensemble_z_score] + [cur_seed_z_score_dict[seed] for seed in cur_seed_z_score_dict]
        z_score_figure_fname = os.path.join(args.output_dir, f"{regression_type}.png")
        z_score_metric_fname = os.path.join(args.output_dir, f"{regression_type}_metrics.txt")
        plot_figure(z_score_legend_list, z_score_list, is_private, z_score_figure_fname, z_score_metric_fname)
        # plot figure with fixed std
        pred_diff_legend_list = ["yeom:", f"{regression_type.replace(regression_suffix, '')}_fixed_ensemble:"] + [f"{regression_type.replace(regression_suffix, '')}_fixed_seed={seed}:" for seed in cur_seed_z_score_dict]
        pred_diff_list = [task_scores, ensemble_pred_dff] + [cur_seed_pred_diff_dict[seed] for seed in cur_seed_pred_diff_dict]
        pred_diff_figure_fname = os.path.join(args.output_dir, f"{regression_type}_fixed.png")
        pred_diff_metric_fname = os.path.join(args.output_dir, f"{regression_type}_fixed_metrics.txt")
        plot_figure(pred_diff_legend_list, pred_diff_list, is_private, pred_diff_figure_fname, pred_diff_metric_fname)

    all_metric_dict_list = list()

    # plot figure with estimated std for different uq methods
    z_score_legend_list = baseline_legend_list + [f"{regression_type.replace(regression_suffix, '')}_ensemble:" for regression_type in ensemble_z_score_dict] + lira_legend_list + mismatched_lira_legend_list
    z_scores_list = baseline_score_list + [ensemble_z_score_dict[seed] for seed in ensemble_z_score_dict] + lira_zscore_list + mismatched_lira_zscore_list
    z_score_figure_fname = os.path.join(args.output_dir, f"all_regression_types.png")
    z_score_metric_fname = os.path.join(args.output_dir, f"all_regression_types_metrics.txt")
    all_metric_dict_list.extend(
        plot_figure(z_score_legend_list, z_scores_list, is_private, z_score_figure_fname, z_score_metric_fname)
    )
    # plot figure with fixed std
    pred_diff_legend_list = baseline_legend_list + [f"{regression_type.replace(regression_suffix, '')}_fixed_ensemble:" for regression_type in ensemble_pred_diff_dict] + lira_fixed_legend_list + mismatched_lira_fixed_legend_list
    pred_diff_list = baseline_score_list + [ensemble_pred_diff_dict[seed] for seed in ensemble_pred_diff_dict] + lira_pred_diff_list+ mismatched_lira_pred_diff_list
    pred_diff_figure_fname = os.path.join(args.output_dir, f"all_regression_types_fixed.png")
    pred_diff_metric_fname = os.path.join(args.output_dir, f"all_regression_types_fixed_metrics.txt")
    all_metric_dict_list.extend(
        plot_figure(pred_diff_legend_list, pred_diff_list, is_private, pred_diff_figure_fname, pred_diff_metric_fname)
    )    

    df_metrics = pd.DataFrame(all_metric_dict_list)
    df_metrics["method"] = df_metrics["legend"].apply(lambda x: x.rstrip(":"))
    df_metrics.drop(["legend"], axis=1).drop_duplicates().to_json(
        os.path.join(args.output_dir, f"all_method_metrics.json"),
        orient="records",
        lines=True
    )

def parse_args():
    parser = argparse.ArgumentParser(description='Basic args')
    parser.add_argument('--task_name', type=str, default='qnli')
    parser.add_argument('--text_column_name', type=str, default="text", help='text column name')
    parser.add_argument('--data_dir', type=str, default='/data/qmia_llm_results/', help='output dir for intermediate results')
    parser.add_argument('--task_pred_dir', type=str, default=None, help='task pred dir for intermediate results')
    parser.add_argument('--regression_pred_dir_template', type=str, default=None, help='regression pred dir template for intermediate results')
    parser.add_argument('--lira_pred_dir_template', type=str, default=None, help='lira pred dir template for intermediate results')
    parser.add_argument('--target_model', type=str, help='target model name')
    parser.add_argument('--plot_lira_models', type=str, help='lira models to be plotted')
    parser.add_argument('--shadow_checkpoint_nums', type=str, help='shadow checkpoint nums')
    parser.add_argument('--lira_models', type=str, default="", help='lira model names')
    parser.add_argument('--task_seed', type=str, default='42', help='task model seed') 
    parser.add_argument('--seeds', type=str, default='42', help='seeds to be plotted') 
    parser.add_argument('--private_split', type=str, default='train', help='name of private spit')
    parser.add_argument('--public_split', type=str, default='public_test', help='name of public split for testing')
    parser.add_argument('--output_dir', type=str, help='name of output directrory')
    parser.add_argument('--flip_score', type=int, default=0, help='whether to flip_score')
    parser.add_argument('--max_num_lira_experiments', type=int, default=16, help="max number of lira experiments to consider")
    parser.add_argument('--mismatch_plot_all', type=int, default=0, help='whether to plot all lira-k for mismatched lira')

    args, _ = parser.parse_known_args()
    
    return args

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s %(asctime)-15s pid-%(process)d: %(message)s'
    )
    logger = logging.getLogger(__name__)

    args = parse_args()
    
    args.seed_list = args.seeds.split(",")
    args.lira_model_list = args.lira_models.split(",")
    args.plot_lira_model_list = args.plot_lira_models.split(",") if args.plot_lira_models else []
    args.mismatched_lira_model_list = [x for x in args.lira_model_list if x != args.target_model]
    args.shadow_checkpoint_num_list = args.shadow_checkpoint_nums.split(",")
    logger.info(f"{args.shadow_checkpoint_nums}:shadow_checkpoint_num_list={args.shadow_checkpoint_num_list}")
    
    main()
