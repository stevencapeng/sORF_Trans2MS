# Stage 1 RNA-FM 全局 Transformer + enhanced position（ROC选模）

## 模型改动

本版本以原 Stage 1 `RNAFMSelfAttentionModel` 为基线，只加入显式位置通道：

```text
RNA-FM token（640维）
+ p, p², sin(πp), cos(πp)（4维）
= 644维
→ 区域独立投影到256维
→ segment embedding
→ 三段拼接后的全局 Transformer
→ masked mean + max pooling
→ 256 → 128 → 1
```

位置定义：

- Upstream：`-1 → 0`，远端到AUG；左侧padding。
- ORF：`0 → 1`，AUG到stop；右侧padding。
- Downstream：`0 → 1`，stop到远端；右侧padding。

原始训练划分、损失函数、优化器和测试指标均保持不变。本版本改为根据
**验证集 AUROC** 选择最佳 checkpoint，ReduceLROnPlateau 和 early stopping 也监控
同一个验证集 AUROC。测试集只用于评估最佳 checkpoint，不参与选模。

## 文件

- `build_model_rnafm_self_attention_enhanced_position.py`：模型。
- `train_rnafm_self_attention_enhanced_position.py`：单run训练。
- `run_10runs_enhanced_position.py`：多run调度与汇总。

## 正式运行10次

```python
!python /disk3/hejp/00_thesis/4_predict_translation/src/src_v3/src_rnafm_self_cross_attention_v3/rnafm_10run_strict_v4/stage1_rnafm_self_attention_enhanced_position_v2_roc_select/run_10runs_enhanced_position.py \
  --train_script /disk3/hejp/00_thesis/4_predict_translation/src/src_v3/src_rnafm_self_cross_attention_v3/rnafm_10run_strict_v4/stage1_rnafm_self_attention_enhanced_position_v2_roc_select/train_rnafm_self_attention_enhanced_position.py \
  --dataset_dir /disk4hu/hejp/00_thesis/4_predict_translation/results/training_result/training_v3_2/rnafm_only \
  --dataset_prefix translation_strict_lenmatched \
  --out_dir /disk4hu/hejp/00_thesis/4_predict_translation/results/training_result/training_v3_2/rnafm_self_attention_strict_10runs/afteroptimize/enhanced_position_rocselect_ffmult2_wd5e4_10runs \
  --run_name_prefix rnafm_self_attention_enhanced_position_rocselect_strict \
  --position_mode enhanced \
  --monitor_metric roc_auc \
  --start_run 0 \
  --n_runs 10 \
  --seed 42 \
  --batch_size 64 \
  --epochs 50 \
  --num_heads 8 \
  --num_layers 2 \
  --dropout 0.2 \
  --lr 1e-4 \
  --weight_decay 5e-4 \
  --ff_mult 2 \
  --branch_hidden_dim 256 \
  --fusion_hidden_dim 256 \
  --patience 10 \
  --grad_clip 1.0 \
  --num_workers 2 \
  --skip_existing
```

## 无位置通道对照（可选）

同一命令把 `--position_mode enhanced` 改成 `--position_mode none`，并使用不同的
`--out_dir` 和 `--run_name_prefix`。该对照仍采用 upstream 左padding，因此只比较
位置通道本身；正确mask后，padding方向不会成为主要差异。

## 输出

- 每个run：根据验证集 AUROC 保存的最佳模型、history、测试预测、测试指标、配置文件和日志。
- 总目录：`all_runs_test_metrics.csv` 与 `summary_mean_std.csv`。
- `all_runs_test_metrics.csv` 额外记录 `monitor_metric`、`best_valid_metric` 和 `best_epoch`。

`--monitor_metric` 默认为 `roc_auc`。仅为了回溯旧版时，才需要改成
`average_precision`。

## 说明

位置通道按每个区域的有效token长度归一化，不是固定核苷酸编号。因此它表达的是
相对于AUG或stop的区域内位置。若上游长度固定为450 nt，则可近似映射到具体的
上游核苷酸距离。
