sh run_pred_clm.sh --dataset_name wikitext_sample --model_org EleutherAI --model_name pythia-410m --qr_model_org EleutherAI --qr_model_name pythia-160m --lm_target_epoch 1 --train_target --pred_target

sh run_pred_clm.sh --dataset_name wikitext_sample --model_org EleutherAI --model_name pythia-410m --qr_model_org EleutherAI --qr_model_name pythia-160m --lm_target_epoch 1 --train_shadow --pred_shadow

sh run_pred_clm.sh --dataset_name wikitext_sample --model_org EleutherAI --model_name pythia-410m --qr_model_org EleutherAI --qr_model_name pythia-160m --lm_target_epoch 1 --extend_task_pred

sh run_pred_clm.sh --dataset_name wikitext_sample --model_org EleutherAI --model_name pythia-410m --qr_model_org EleutherAI --qr_model_name pythia-160m --lm_target_epoch 1 --run_gaussian

sh plot_figure.sh --dataset_name wikitext_sample --model_org EleutherAI --model_name pythia-410m --qr_model_org EleutherAI --qr_model_name pythia-160m --lm_target_epoch 1  --lira_model_names "pythia-410m" --plot_lira_model_names "pythia-410m"
