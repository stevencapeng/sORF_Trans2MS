import os
import json
import torch
import pandas as pd
import argparse
from tqdm import tqdm
from Bio.Seq import Seq
from transformers import AutoTokenizer, AutoModel

# =========================
# 路径和参数
# =========================
# MODEL_DIR = "/disk3/hejp/00_thesis/4_predict_translation/model/esm2_t33_650M_UR50D"
# DATA_PATH = "/disk3/hejp/00_thesis/4_predict_translation/data/processed/ribotricer_and_MSpp_and_double_minus_nt450/train_table_4class_with_ribotricer_rtorfs.pkl"
# SAVE_DIR = "/disk3/hejp/00_thesis/4_predict_translation/results/features/features_esm2_token_ribotricer_openprot_nt450"

MODEL_DIR = "/disk3/hejp/00_thesis/4_predict_translation/model/esm2_t33_650M_UR50D"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHUNK_SIZE = 1000  # 大蛋白可以分块
REPR_LAYER = 33    # 取 ESM2 第33层 token embedding

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")


# =========================
# 序列处理
# =========================
def safe_nt_seq(seq):
    if pd.isna(seq):
        return ""
    seq = str(seq).upper()
    return "".join([x if x in {"A","T","C","G","N"} else "N" for x in seq])


def translate_orf(nt_seq):
    nt_seq = safe_nt_seq(nt_seq)
    if len(nt_seq) >= 3 and nt_seq[-3:] in {"TAA","TAG","TGA"}:
        nt_seq = nt_seq[:-3]
    usable_len = len(nt_seq) - len(nt_seq) % 3
    nt_seq = nt_seq[:usable_len]
    if len(nt_seq) == 0:
        return ""
    protein = str(Seq(nt_seq).translate(to_stop=False))
    protein = protein.replace("*","")
    protein = "".join([aa for aa in protein if aa in STD_AA])
    return protein


def split_sequence(seq, chunk_size):
    return [seq[i:i+chunk_size] for i in range(0, len(seq), chunk_size)]


# =========================
# 模型加载（本地 HF ESM2）
# =========================
def load_esm2_model(model_dir):
    print("Loading ESM2 tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}")
    return tokenizer, model


# =========================
# token embedding 提取
# =========================
@torch.no_grad()
def get_token_embeddings(seqs, tokenizer, model, batch_size=16, chunk_size=1000):
    """
    seqs: list of protein sequences
    输出: list，每个元素 tensor(L, dim)
    """
    all_emb = []
    for i, seq in enumerate(tqdm(seqs, desc="Processing proteins")):
        chunks = split_sequence(seq, chunk_size)
        protein_emb = []
        for j, chunk in enumerate(chunks):
            data = [(f"chunk_{i}_{j}", chunk)]
            batch = tokenizer(data, return_tensors="pt", padding=True, truncation=True)
            batch = {k:v.to(DEVICE) for k,v in batch.items()}
            outputs = model(**batch)
            hidden = outputs.last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            seq_len = mask.sum(1).item()
            token_emb = hidden[0,1:seq_len-1,:].cpu()
            protein_emb.append(token_emb)
        protein_emb = torch.cat(protein_emb, dim=0)
        all_emb.append(protein_emb)
    return all_emb


# =========================
# 主程序
# =========================
# DATA_PATH = ""
#SAVE_DIR = ""
def main():
    parser = argparse.ArgumentParser(
        description="Extract ESM2 token-level embeddings for microprotein regions."
    )
    parser.add_argument(
        "--DATA_PATH",
        default="",
    )

    parser.add_argument(
        "--SAVE_DIR",
        default="",
    )
    args = parser.parse_args()
    os.makedirs(args.SAVE_DIR, exist_ok=True)
    print("Loading input table...")
    
    #original code#
    #df = pd.read_pickle(args.DATA_PATH)
    
    #correct 20260630#
    if args.DATA_PATH.endswith((".pkl", ".pickle")):
        df=pd.read_pickle(args.DATA_PATH)
    elif args.DATA_PATH.endswith(".json"):
        df=pd.read_json(args.DATA_PATH)
    else:
        df=pd.read_csv(args.DATA_PATH, compression="infer")
    #correct 20260630#
    
    print("Input shape:", df.shape)

    df["protein_seq_for_esm2"] = df["orf_seq"].apply(translate_orf)
    df = df[df["protein_seq_for_esm2"].str.len()>0].copy()
    print("Valid protein count:", df.shape[0])

    tokenizer, model = load_esm2_model(MODEL_DIR)

    # 提取 token embeddings
    emb_list = get_token_embeddings(df["protein_seq_for_esm2"].tolist(), tokenizer, model, batch_size=8, chunk_size=CHUNK_SIZE)

    summary = []
    for idx, row in enumerate(df.itertuples()):
        save_path = os.path.join(args.SAVE_DIR, f"{row.sample_id}.pt")
        torch.save(emb_list[idx], save_path)
        summary.append({
            "sample_id": row.sample_id,
            "label_raw": row.label_raw,
            "label_id": int(row.label_id),
            "protein_len": len(row.protein_seq_for_esm2),
            "embedding_shape": list(emb_list[idx].shape),
            "save_path": save_path,
            "status": "ok"
        })

    summary_df = pd.DataFrame(summary)
    summary_csv = os.path.join(args.SAVE_DIR, "esm2_token_embedding_summary.csv")
    summary_json = os.path.join(args.SAVE_DIR, "esm2_token_embedding_summary.json")

    summary_df.to_csv(summary_csv, index=False)
    with open(summary_json,"w",encoding="utf-8") as f:
        json.dump({
            "data_path": args.DATA_PATH,
            "save_dir": args.SAVE_DIR,
            "model": "esm2_t33_650M_UR50D",
            "repr_layer": REPR_LAYER,
            "chunk_size": CHUNK_SIZE,
            "device": str(DEVICE),
            "n_input": int(df.shape[0]),
            "n_success": int((summary_df["status"]=="ok").sum()),
            "n_error": int((summary_df["status"]!="ok").sum())
        }, f, ensure_ascii=False, indent=2)

    print("Done. Summary CSV:", summary_csv)
    print("JSON summary:", summary_json)


if __name__ == "__main__":
    main()