# Stage2 RNA-FM + ESM2 CNN 单独调参与结构优化 v1

## 1. 目的

该脚本包从原 Stage2 baseline 套件中单独取出 CNN，并保持以下内容不变：

- 数据集不重新匹配、不重新抽样、不删样本；
- 复用原来的 `run00...run09` 固定 train/valid/test split；
- RNA 输入仍为 upstream、ORF、downstream 三段 RNA-FM token；
- 蛋白输入仍为 ESM2 token；
- 不加载 Stage1，不进行预训练；
- 每个 run 仍保存模型、训练曲线、测试预测和完整指标。

调参配置只按 **mean best validation ROC-AUC** 排名。测试集 ROC 会保存，但不用于选择结构，避免用测试集调参。

## 2. 两种模型结构

### baseline

严格复现原 v4 CNN：

```text
每段 token → Conv1d(k=3/5/7) → BatchNorm → GELU → Dropout
          → masked global max pooling

upstream + ORF + downstream → RNA head → 256
protein                     → protein head → 256
RNA 256 + protein 256       → classifier
```

配置名：`baseline_repro`。

### enhanced

```text
token + 相对位置通道
→ LayerNorm + 1×1 projection
→ 多尺度 residual CNN
→ masked mean + max pooling

protein另加前25 aa局部池化（可选）
RNA与protein可加入乘积和绝对差交互（可选）
→ classifier
```

主要改进理由：

1. 原 global max pooling 只能判断“是否出现特征”，难以判断“出现在哪里”；位置通道让模型区分 AUG 邻近上游信号和蛋白 N 端信号。
2. `mean + max` 同时保留整体趋势和最强局部响应。
3. residual multi-scale CNN 增加感受野，同时比 Transformer 更轻。
4. LayerNorm 不依赖 batch 统计，更适合当前较小的 CNN batch size。
5. 前25 aa池化与已有蛋白消融中 N 端信号较强的结果相呼应。

## 3. 8个预设配置

| 配置 | 作用 |
|---|---|
| `baseline_repro` | 严格复现原CNN |
| `baseline_lowdrop` | 只降低dropout，判断原模型是否欠拟合 |
| `enhanced_meanmax` | residual CNN + LayerNorm + mean/max |
| `enhanced_position` | 加入相对位置通道 |
| `enhanced_deep_position` | 位置通道 + 两层残差块 |
| `enhanced_position_nterm25` | 位置通道 + 蛋白前25 aa池化 |
| `enhanced_position_nterm25_interaction` | 再加入RNA–蛋白交互融合 |
| `enhanced_wide_position_nterm25` | 更宽通道和3/7/11卷积核 |

## 4. 推荐流程

### 第一步：3-run pilot筛选

先运行8个配置 × 3个run，共24次训练。下面命令直接复用原 baseline v4 已生成的 prepared dataset 和 fixed split。

```python
!python /disk3/hejp/00_thesis/4_predict_translation/src/src_v3/src_rnafm_self_cross_attention_v3/rnafm_10run_strict_v4/stage2_v2_MSpp_ribotricer/stage2_rnaesm_cnn_tuning_v1/run_stage2_rnaesm_cnn_tuning_v1.py \
  --train_script /disk3/hejp/00_thesis/4_predict_translation/src/src_v3/src_rnafm_self_cross_attention_v3/rnafm_10run_strict_v4/stage2_v2_MSpp_ribotricer/stage2_rnaesm_cnn_tuning_v1/train_stage2_rnaesm_cnn_tuned_v1.py \
  --search_space /disk3/hejp/00_thesis/4_predict_translation/src/src_v3/src_rnafm_self_cross_attention_v3/rnafm_10run_strict_v4/stage2_v2_MSpp_ribotricer/stage2_rnaesm_cnn_tuning_v1/cnn_search_space_v1.json \
  --dataset_dir /disk4hu/hejp/00_thesis/4_predict_translation/results/training_result/training_v3_2/stage2_v2_MSpp_ribotricer_strict_10runs \
  --dataset_prefix translation_strict_lenmatched \
  --esm2_token_dir /disk4hu/hejp/00_thesis/4_predict_translation/results/features/features_v3_esm2_token_stage2_MSpp_neg \
  --prepared_dataset_dir /disk4hu/hejp/00_thesis/4_predict_translation/results/training_result/training_v3_2/stage2_v2_MSpp_ribotricer/stage2_v2_rnaesm_baselines_10runs/prepared_datasets \
  --split_root /disk4hu/hejp/00_thesis/4_predict_translation/results/training_result/training_v3_2/stage2_v2_MSpp_ribotricer/stage2_v2_rnaesm_baselines_10runs/generated_splits \
  --out_root /disk4hu/hejp/00_thesis/4_predict_translation/results/training_result/training_v3_2/stage2_v2_MSpp_ribotricer/stage2_rnaesm_cnn_tuning_v1_pilot3 \
  --start_run 0 \
  --n_runs 3 \
  --seed 42 \
  --epochs 50 \
  --batch_size 16 \
  --patience 10 \
  --grad_clip 1.0 \
  --num_workers 2 \
  --skip_existing
```

查看验证集排名：

```python
import pandas as pd

ranking_file = (
    "/disk4hu/hejp/00_thesis/4_predict_translation/results/training_result/"
    "training_v3_2/stage2_v2_MSpp_ribotricer/"
    "stage2_rnaesm_cnn_tuning_v1_pilot3/"
    "cnn_config_ranking_by_validation_roc.csv"
)

ranking = pd.read_csv(ranking_file)
display(ranking)
BEST = ranking.iloc[0]["config_name"]
print("Best validation config:", BEST)
```

先检查 `baseline_repro` 的 test ROC 是否接近原结果 `0.8890 ± 0.0085`。3-run标准差可能较大，但不应出现系统性大幅偏离。

### 第二步：最佳配置正式跑10次

在同一个 ipynb 中，`BEST` 会被替换成第一步验证集排名第一的配置名：

```python
!python /disk3/hejp/00_thesis/4_predict_translation/src/src_v3/src_rnafm_self_cross_attention_v3/rnafm_10run_strict_v4/stage2_v2_MSpp_ribotricer/stage2_rnaesm_cnn_tuning_v1/run_stage2_rnaesm_cnn_tuning_v1.py \
  --train_script /disk3/hejp/00_thesis/4_predict_translation/src/src_v3/src_rnafm_self_cross_attention_v3/rnafm_10run_strict_v4/stage2_v2_MSpp_ribotricer/stage2_rnaesm_cnn_tuning_v1/train_stage2_rnaesm_cnn_tuned_v1.py \
  --search_space /disk3/hejp/00_thesis/4_predict_translation/src/src_v3/src_rnafm_self_cross_attention_v3/rnafm_10run_strict_v4/stage2_v2_MSpp_ribotricer/stage2_rnaesm_cnn_tuning_v1/cnn_search_space_v1.json \
  --dataset_dir /disk4hu/hejp/00_thesis/4_predict_translation/results/training_result/training_v3_2/stage2_v2_MSpp_ribotricer_strict_10runs \
  --dataset_prefix translation_strict_lenmatched \
  --esm2_token_dir /disk4hu/hejp/00_thesis/4_predict_translation/results/features/features_v3_esm2_token_stage2_MSpp_neg \
  --prepared_dataset_dir /disk4hu/hejp/00_thesis/4_predict_translation/results/training_result/training_v3_2/stage2_v2_MSpp_ribotricer/stage2_v2_rnaesm_baselines_10runs/prepared_datasets \
  --split_root /disk4hu/hejp/00_thesis/4_predict_translation/results/training_result/training_v3_2/stage2_v2_MSpp_ribotricer/stage2_v2_rnaesm_baselines_10runs/generated_splits \
  --out_root /disk4hu/hejp/00_thesis/4_predict_translation/results/training_result/training_v3_2/stage2_v2_MSpp_ribotricer/stage2_rnaesm_cnn_tuning_v1_final10 \
  --config_names {BEST} \
  --start_run 0 \
  --n_runs 10 \
  --seed 42 \
  --epochs 50 \
  --batch_size 16 \
  --patience 10 \
  --grad_clip 1.0 \
  --num_workers 2 \
  --skip_existing
```

## 5. 主要输出

```text
out_root/
├── cnn_dataset_split_manifest.csv
├── selected_cnn_configs.json
├── all_configs_all_runs_metrics.csv
├── all_configs_summary_mean_std.csv
├── cnn_config_ranking_by_validation_roc.csv
└── 配置名/
    ├── all_runs_test_metrics.csv
    ├── summary_mean_std.csv
    └── cnn_配置名_runXX/
        ├── *_best_model.pt
        ├── *_training_history.csv
        ├── *_test_predictions.csv
        ├── *_test_metrics.json
        └── *_split_info.json
```

## 6. 显存建议

- 24 GB显存优先保持 `--batch_size 16`；
- `enhanced_wide_position_nterm25` 若显存不足，改为 `--batch_size 8`；
- 不建议先扩大到完整笛卡尔网格。先找出有效的结构组件，再围绕最佳配置微调 `dropout/lr/conv_hidden`。
