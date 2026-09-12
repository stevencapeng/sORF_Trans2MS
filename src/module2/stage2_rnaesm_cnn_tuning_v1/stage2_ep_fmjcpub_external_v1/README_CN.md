# Stage2 enhanced_position：FMJCpub 独立测试

该脚本使用 `enhanced_position` 的10个最佳 checkpoint，在同一个
FMJCpub 阳性＋ribotricer 阴性独立数据集上预测。

模型结构和超参数不需要手工重复填写：脚本从每个 checkpoint 的
`args` 字段恢复，并从训练脚本导入完全相同的模型类、位置特征、
padding 和 pooling 实现。

## 运行命令

```python
!python /disk3/hejp/00_thesis/4_predict_translation/src/src_v3/src_rnafm_self_cross_attention_v3/rnafm_10run_strict_v4/stage2_v2_MSpp_ribotricer/stage2_rnaesm_cnn_tuning_v1/stage2_enhanced_position_fmjcpub_external_v1/predict_fmjcpub_enhanced_position_10runs.py \
  --input_table /disk4hu/hejp/00_thesis/4_predict_translation/results/training_result/training_v3_2/FMJCpub_independent/FMJCpub_MS_pos_ribotricer_neg_with_rnafm_esm2_paths.pkl \
  --train_script /disk3/hejp/00_thesis/4_predict_translation/src/src_v3/src_rnafm_self_cross_attention_v3/rnafm_10run_strict_v4/stage2_v2_MSpp_ribotricer/stage2_rnaesm_cnn_tuning_v1/train_stage2_rnaesm_cnn_tuned_v1.py \
  --model_root /disk4hu/hejp/00_thesis/4_predict_translation/results/training_result/training_v3_2/stage2_v2_MSpp_ribotricer/stage2_rnaesm_cnn_enhanced_position_10runs \
  --out_dir /disk4hu/hejp/00_thesis/4_predict_translation/results/training_result/training_v3_2/FMJCpub_independent/stage2_enhanced_position_10runs \
  --config_name enhanced_position \
  --label_col binary_label \
  --positive_value 1 \
  --expected_models 10 \
  --threshold 0.5 \
  --batch_size 512 \
  --num_workers 2 \
  --skip_existing
```

如果将 FMJCpub 表另存为 CSV 或 CSV.GZ，只需替换 `--input_table`；
脚本同时支持 PKL、CSV/CSV.GZ 和 TSV/TSV.GZ。

## checkpoint 位置

脚本默认在 `--model_root` 下寻找：

```text
enhanced_position/
  cnn_enhanced_position_run00/
    cnn_enhanced_position_run00_best_model.pt
  ...
  cnn_enhanced_position_run09/
    cnn_enhanced_position_run09_best_model.pt
```

## 主要输出

```text
run00/cnn_enhanced_position_run00_fmjcpub_predictions.csv
...
run09/cnn_enhanced_position_run09_fmjcpub_predictions.csv
fmjcpub_all_runs_metrics.csv
fmjcpub_10runs_summary_mean_std.csv
fmjcpub_enhanced_position_ensemble_predictions.csv
fmjcpub_enhanced_position_ensemble_metrics.csv
fmjcpub_enhanced_position_ensemble_metrics.json
fmjcpub_prediction_manifest.json
```

最终统一作图推荐使用 ensemble 文件中的：

- `y_true`：真实标签；
- `y_prob`：10个 checkpoint 概率的算术平均；
- `y_pred`：阈值0.5下的集成预测；
- `y_prob_model_00 ... y_prob_model_09`：各 run 概率；
- `y_prob_std`：同一样本在10个 checkpoint 间的概率标准差。

对应统一作图 manifest：

```python
{
    "dataset": "fmjcpub_external",
    "model": "Enhanced position",
    "pred_file": "/disk4hu/hejp/00_thesis/4_predict_translation/results/training_result/training_v3_2/FMJCpub_independent/stage2_enhanced_position_10runs/fmjcpub_enhanced_position_ensemble_predictions.csv",
    "true_col": "y_true",
    "score_col": "y_prob",
}
```
