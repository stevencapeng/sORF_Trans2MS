# ESM2 + RNA-FM cross-attention scripts

这一版用于同时读取：

- ESM2 蛋白 ORF token：`<esm2_token_dir>/<sample_id>.pt`
- RNA-FM RNA token：
  - `<rnafm_token_dir>/upstream/<sample_id>_upstream.pt`
  - `<rnafm_token_dir>/orf/<sample_id>_orf.pt`
  - `<rnafm_token_dir>/downstream/<sample_id>_downstream.pt`

标签定义：

- 阳性：`label_raw` 含有 `MS++`
- 阴性：`label_raw == ribotricer`
- 默认会将阴性随机下采样到与阳性一样多；如果不想平衡，使用 `--no_balance`

## 1. 准备数据集

```bash
python /disk3/hejp/00_thesis/4_predict_translation/src/src_v3/src_esm2_rnafm_cross_attention_v3/prepare_dataset_esm2_rnafm_cross_attention_MSpp.py \
  --metadata_csv /disk3/hejp/00_thesis/4_predict_translation/data/processed/dataset_nt450_v3/train_pos_neg_merged.csv.gz \
  --esm2_token_dir /disk4hu/hejp/00_thesis/4_predict_translation/results/features/features_v3_esm2_token_pos_neg \
  --rnafm_token_dir /disk4hu/hejp/00_thesis/4_predict_translation/results/features/features_rnafm_token_ribotricer_openprot_nt450 \
  --out_pkl /disk4hu/hejp/00_thesis/4_predict_translation/results/training_result/esm2_rnafm_cross_attention_MSpp/dataset_esm2_rnafm_cross_MSpp.pkl
```

如果你的 RNA-FM token 目录不是 `features_rnafm_token_ribotricer_openprot_nt450`，把 `--rnafm_token_dir` 换成你真实路径即可。

## 2. 训练 cross-attention 模型

```bash
python /disk3/hejp/00_thesis/4_predict_translation/src/src_v3/src_esm2_rnafm_cross_attention_v3/train_esm2_rnafm_cross_attention.py \
  --dataset_pkl /disk4hu/hejp/00_thesis/4_predict_translation/results/training_result/esm2_rnafm_cross_attention_MSpp/dataset_esm2_rnafm_cross_MSpp.pkl \
  --out_dir /disk4hu/hejp/00_thesis/4_predict_translation/results/training_result/esm2_rnafm_cross_attention_MSpp \
  --run_name esm2_rnafm_cross_attention_MSpp \
  --batch_size 16 \
  --epochs 50 \
  --lr 1e-4 \
  --branch_hidden_dim 256 \
  --fusion_hidden_dim 256 \
  --num_heads 8 \
  --num_layers 2
```

## 3. 显存不够时建议

Cross-attention 同时使用 ESM2 ORF token 和 RNA-FM upstream/ORF/downstream token，比单独 ESM2 self-attention 更吃显存。若 OOM，建议依次尝试：

```bash
--batch_size 8
```

或者：

```bash
--branch_hidden_dim 128 --num_heads 4 --batch_size 16
```

如果内存而不是显存吃紧，可以加：

```bash
--no_preload
```

## 4. 模型结构

简化表示：

```text
ESM2 ORF protein token
    -> Linear projection
    -> optional 1-layer self-attention

RNA-FM upstream + ORF + downstream RNA token
    -> each segment Linear projection
    -> add RNA segment embedding
    -> concatenate tokens
    -> optional 1-layer self-attention

ESM2 tokens <-> RNA-FM tokens
    -> bidirectional cross-attention blocks
    -> mean/max pooling for each modality
    -> concatenate ESM2 pooled + RNA-FM pooled
    -> MLP classifier
```

输出文件包括：

- `<run_name>.log`
- `<run_name>_history.csv`
- `<run_name>_best_model.pt`
- `<run_name>_test_predictions.csv`
- `<run_name>_test_metrics.json`
