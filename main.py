# main.py
import os
import glob
import json
import sys
import pickle
import hashlib
from typing import Tuple, List, Dict, Any, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.tokenizer import tokenizer, build_tree_from_output, build_context_from_tokens
from src.dataset import TxRandomWindowDataset, TxStrideWindowDataset
from src.model import GPTModel
from src.engine import train_network, valid_network, score_transactions

# necessario per evitare errori di valore quando si tokenizzano interi enormi
sys.set_int_max_str_digits(1_000_000)


# cache helpers
def _files_fingerprint(paths: list[str], extra_paths: Optional[list[str]] = None) -> str:
    h = hashlib.sha256()
    all_paths = list(paths)
    if extra_paths:
        all_paths.extend(extra_paths)

    for p in all_paths:
        st = os.stat(p)
        h.update(p.encode("utf-8"))
        h.update(str(st.st_size).encode("utf-8"))
        h.update(str(int(st.st_mtime)).encode("utf-8"))
    return h.hexdigest()[:16]


def load_token_cache(cache_path: str, expected_fingerprint: str):
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)
    if payload.get("fingerprint") != expected_fingerprint:
        return None
    return payload


def save_token_cache(
    cache_path: str,
    fingerprint: str,
    out_token_seqs: list[list[str]],
    tree_token_seqs: list[list[str]],
    ctx_token_seqs: list[list[str]],
):
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    payload = {
        "fingerprint": fingerprint,
        "out_token_seqs": out_token_seqs,
        "tree_token_seqs": tree_token_seqs,
        "ctx_token_seqs": ctx_token_seqs,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_vocabs_if_missing(
    path: str,
    out_vocab: dict,
    tree_vocab: dict,
    ctx_vocab: dict,
    meta: dict | None = None,
) -> bool:
    if os.path.exists(path):
        return False

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "version": 1,
        "meta": meta or {},
        "out_vocab": out_vocab,
        "tree_vocab": tree_vocab,
        "ctx_vocab": ctx_vocab,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return True


def load_vocabs(path: str):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    out_vocab = payload["out_vocab"]
    tree_vocab = payload["tree_vocab"]
    ctx_vocab = payload["ctx_vocab"]

    for name, v in [("out_vocab", out_vocab), ("tree_vocab", tree_vocab), ("ctx_vocab", ctx_vocab)]:
        if "[PAD]" not in v or "[UNK]" not in v:
            raise ValueError(f"{name} missing [PAD] or [UNK].")
        if v["[PAD]"] != 0 or v["[UNK]"] != 1:
            raise ValueError(f"{name} expected [PAD]=0 and [UNK]=1.")

    return out_vocab, tree_vocab, ctx_vocab, payload.get("meta", {})


# vocab helpers
def build_vocab_and_ids(token_list, pad_token="[PAD]", unk_token="[UNK]"):
    vocab = {pad_token: 0, unk_token: 1}
    for t in token_list:
        if t not in vocab:
            vocab[t] = len(vocab)

    ids = torch.tensor([vocab.get(t, vocab[unk_token]) for t in token_list], dtype=torch.long)
    return vocab, ids


def build_vocab(token_list, pad_token="[PAD]", unk_token="[UNK]"):
    vocab = {pad_token: 0, unk_token: 1}
    for t in token_list:
        if t not in vocab:
            vocab[t] = len(vocab)
    return vocab


# other utils
def load_and_merge_logs(file_paths: list[str], verbose: bool = False) -> pd.DataFrame:
    dfs = []
    for path in file_paths:
        if verbose:
            print(f"Reading {path}...")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tmp = pd.DataFrame.from_records(data)
        if "_id" in tmp.columns:
            tmp = tmp.drop(columns=["_id"])
        dfs.append(tmp)

        if verbose:
            print(f" -> Loaded {len(tmp)} rows from {path}")

    df = pd.concat(dfs, ignore_index=True)
    return df


def tokenize_one_row(row: pd.Series) -> list[str]:
    one = pd.DataFrame([row.to_dict()])
    toks = tokenizer(one)
    return [str(t) for t in toks]


def build_token_seqs_from_df(df: pd.DataFrame) -> Tuple[list[list[str]], list[list[str]], list[list[str]]]:
    seqs = [tokenize_one_row(r) for _, r in df.iterrows()]

    out_token_seqs, tree_token_seqs, ctx_token_seqs = [], [], []
    for seq in seqs:
        toks, tree = build_tree_from_output(seq)
        ctx = build_context_from_tokens(toks)
        out_token_seqs.append([str(t) for t in toks])
        tree_token_seqs.append([str(x) for x in tree])
        ctx_token_seqs.append([str(c) for c in ctx])

    return out_token_seqs, tree_token_seqs, ctx_token_seqs


def get_or_build_token_seqs(
    df: pd.DataFrame,
    file_paths: list[str],
    cache_path: str = "cache/tokenized_seqs.pkl",
    extra_fingerprint_paths: Optional[list[str]] = None,
) -> Tuple[list[list[str]], list[list[str]], list[list[str]]]:
    fp = _files_fingerprint(file_paths, extra_paths=extra_fingerprint_paths)
    cached = load_token_cache(cache_path, fp)
    if cached is not None:
        print(f"Loaded tokenization cache: {cache_path}")
        return cached["out_token_seqs"], cached["tree_token_seqs"], cached["ctx_token_seqs"]

    print("Tokenizing data (cache miss)...")
    out_token_seqs, tree_token_seqs, ctx_token_seqs = build_token_seqs_from_df(df)
    save_token_cache(cache_path, fp, out_token_seqs, tree_token_seqs, ctx_token_seqs)
    print(f"Saved tokenization cache: {cache_path}")
    return out_token_seqs, tree_token_seqs, ctx_token_seqs


def flatten_token_seqs(token_seqs: list[list[str]]) -> list[str]:
    return [t for s in token_seqs for t in s]


def get_or_build_vocabs(
    out_token_seqs: list[list[str]],
    tree_token_seqs: list[list[str]],
    ctx_token_seqs: list[list[str]],
    file_paths: list[str],
    vocab_path: str = "models/vocabs.json",
):
    if os.path.exists(vocab_path):
        out_vocab, tree_vocab, ctx_vocab, _meta = load_vocabs(vocab_path)
        print(f"Loaded vocabs from {vocab_path}")
        return out_vocab, tree_vocab, ctx_vocab

    print(f"No vocab file found at {vocab_path}. Building and saving...")
    all_out = flatten_token_seqs(out_token_seqs)
    all_tree = flatten_token_seqs(tree_token_seqs)
    all_ctx = flatten_token_seqs(ctx_token_seqs)

    out_vocab, _ = build_vocab_and_ids(all_out)
    tree_vocab, _ = build_vocab_and_ids(all_tree)
    ctx_vocab, _ = build_vocab_and_ids(all_ctx)

    save_vocabs_if_missing(
        vocab_path,
        out_vocab=out_vocab,
        tree_vocab=tree_vocab,
        ctx_vocab=ctx_vocab,
        meta={"source_files": file_paths},
    )
    print(f"Saved vocabs to {vocab_path}")
    return out_vocab, tree_vocab, ctx_vocab


def encode_token_seqs(token_seqs: list[list[str]], vocab: dict) -> list[torch.Tensor]:
    unk_id = vocab["[UNK]"]
    return [torch.tensor([vocab.get(t, unk_id) for t in seq], dtype=torch.long) for seq in token_seqs]


def make_split_indices(n: int, seed: int = 42, train_frac: float = 0.6, val_frac: float = 0.2):
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return train_idx, val_idx, test_idx


def subset(lst, idxs):
    return [lst[i] for i in idxs]



def train_main():

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    EMBEDDING_DIM = 64
    BLOCK_SIZE = 128 # context window, i.e. quanti token il modello vede alla volta
    BATCH_SIZE = 16 # numero di finestre per batch
    EPOCHS = 10

    file_paths = [
        "data/LogNuovoPiccolo.json",
        "data/LogNuovoGrande.json",
    ]

    print("Loading datasets...")
    df = load_and_merge_logs(file_paths, verbose=True)
    print("Merging datasets...")
    print(f"TOTALE: Il dataset completo ha {len(df)} righe.")

    tokenizer_src = None

    # ricordate di cancellare i file di cache di vocabolario e tokenizzazione se cambiate la logica del tokenizer, per evitare di riutilizzare cache obsolete
    out_token_seqs, tree_token_seqs, ctx_token_seqs = get_or_build_token_seqs(
        df=df,
        file_paths=file_paths,
        cache_path="cache/tokenized_seqs.pkl",
        extra_fingerprint_paths=tokenizer_src,
    )

    out_vocab, tree_vocab, ctx_vocab = get_or_build_vocabs(
        out_token_seqs, tree_token_seqs, ctx_token_seqs,
        file_paths=file_paths,
        vocab_path="models/vocabs.json",
    )

    out_seqs = encode_token_seqs(out_token_seqs, out_vocab)
    tree_seqs = encode_token_seqs(tree_token_seqs, tree_vocab)
    ctx_seqs = encode_token_seqs(ctx_token_seqs, ctx_vocab)

    for i in range(len(out_seqs)):
        assert out_seqs[i].numel() == tree_seqs[i].numel() == ctx_seqs[i].numel()

    print("Creating Datasets...")
    train_idx, val_idx, test_idx = make_split_indices(len(out_seqs), seed=42)

    train_out = subset(out_seqs, train_idx)
    train_tree = subset(tree_seqs, train_idx)
    train_ctx = subset(ctx_seqs, train_idx)

    val_out = subset(out_seqs, val_idx)
    val_tree = subset(tree_seqs, val_idx)
    val_ctx = subset(ctx_seqs, val_idx)

    test_out = subset(out_seqs, test_idx)
    test_tree = subset(tree_seqs, test_idx)
    test_ctx = subset(ctx_seqs, test_idx)

    steps_per_epoch = 2000
    num_samples_per_epoch = steps_per_epoch * BATCH_SIZE

    train_ds = TxRandomWindowDataset(
        train_out, train_tree, train_ctx,
        block_size=BLOCK_SIZE,
        num_samples_per_epoch=num_samples_per_epoch,
    )
    val_ds = TxStrideWindowDataset(
        val_out, val_tree, val_ctx,
        block_size=BLOCK_SIZE,
        stride=BLOCK_SIZE,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)

    print("Train windows per epoch:", len(train_ds))
    print("Val windows:", len(val_ds))
    print("Train batches per epoch:", len(train_loader))
    print("Val batches:", len(val_loader))

    print("Initializing Model...")
    model = GPTModel(
        vocab_size=len(out_vocab),
        tree_vocab_size=len(tree_vocab),
        ctx_vocab_size=len(ctx_vocab),
        embed_dim=EMBEDDING_DIM,
        feed_forward_dim=4 * EMBEDDING_DIM,
        num_heads=8,
        key_dim=EMBEDDING_DIM,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    train_loss_fn = nn.CrossEntropyLoss()
    score_loss_fn = nn.CrossEntropyLoss(reduction="none")  #reduction none per il calcolo del NLL per token, altrimenti otteniamo una media delle per-token losses e perdiamo l'informazione per token necessaria per il calcolo dei punteggi

    os.makedirs("models", exist_ok=True)

    print("Starting Training...")
    last_val_window_nlls = None
    last_ckpt_path = None

    for epoch in range(EPOCHS):
        train_network(model, optimizer, train_loss_fn, train_loader, DEVICE, epoch, EPOCHS)

        window_nlls, _, _ = valid_network(model, score_loss_fn, val_loader, DEVICE, epoch, EPOCHS)
        last_val_window_nlls = window_nlls

        last_ckpt_path = f"models/anomaly_detection_model_{epoch}.pt"
        torch.save({"model_state": model.state_dict()}, last_ckpt_path)
        print("Model saved!")

    if not last_val_window_nlls:
        raise ValueError("No validation NLLs collected, cannot set threshold.")

    thr = torch.quantile(torch.tensor(last_val_window_nlls, dtype=torch.float64), 0.99).item()
    print(f"Chosen threshold (99th percentile) on Val window NLL(sum): {thr:.2f}")

    return last_ckpt_path



def validation_main(MODEL_PATH="models_encoded_integers/anomaly_detection_model_9.pt", VOCAB_PATH="models/vocabs.json"):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    EMBEDDING_DIM = 64
    BLOCK_SIZE = 128

    SCORE_KEY = "mean_token_nll"

    file_paths = [
        "data/LogNuovoPiccolo.json",
        "data/LogNuovoGrande.json",
    ]

    df = load_and_merge_logs(file_paths, verbose=False)

    tokenizer_src = None

    out_token_seqs, tree_token_seqs, ctx_token_seqs = get_or_build_token_seqs(
        df=df,
        file_paths=file_paths,
        cache_path="cache/tokenized_seqs.pkl",
        extra_fingerprint_paths=tokenizer_src,
    )

    out_vocab, tree_vocab, ctx_vocab = get_or_build_vocabs(
        out_token_seqs, tree_token_seqs, ctx_token_seqs,
        file_paths=file_paths,
        vocab_path=VOCAB_PATH,
    )

    model = GPTModel(
        vocab_size=len(out_vocab),
        tree_vocab_size=len(tree_vocab),
        ctx_vocab_size=len(ctx_vocab),
        embed_dim=EMBEDDING_DIM,
        feed_forward_dim=4 * EMBEDDING_DIM,
        num_heads=8,
        key_dim=EMBEDDING_DIM,
    ).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if "model_state" not in checkpoint:
        raise KeyError("Checkpoint must contain key 'model_state'.")

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print("Model loaded from", MODEL_PATH)

    score_loss_fn = nn.CrossEntropyLoss(reduction="none")


    _train_idx, val_idx, _test_idx = make_split_indices(len(out_token_seqs), seed=42)

    val_out = [torch.tensor([out_vocab.get(t, out_vocab["[UNK]"]) for t in out_token_seqs[i]], dtype=torch.long) for i in val_idx]
    val_tree = [torch.tensor([tree_vocab.get(t, tree_vocab["[UNK]"]) for t in tree_token_seqs[i]], dtype=torch.long) for i in val_idx]
    val_ctx = [torch.tensor([ctx_vocab.get(t, ctx_vocab["[UNK]"]) for t in ctx_token_seqs[i]], dtype=torch.long) for i in val_idx]

    print("Computing validation score distribution...")
    
    # calcoliamo un punteggio per ogni transazione aggregando il negative log likelihood su tutte le finestre della transazione
    val_scores = score_transactions(
        model=model,
        loss_function_none=score_loss_fn,
        out_seqs=val_out,
        tree_seqs=val_tree,
        ctx_seqs=val_ctx,
        block_size=BLOCK_SIZE,
        device=DEVICE,
    )

    if not val_scores:
        raise ValueError("No validation scores computed (empty val_scores).")

    val_tx_scores = [s[SCORE_KEY] for s in val_scores]
    thr = torch.quantile(torch.tensor(val_tx_scores, dtype=torch.float64), 0.99).item()
    print(f"Threshold (99th percentile val {SCORE_KEY}): {thr:.6f}")

    # flagghiamo come anomalie le transazioni di attacco che superano la soglia
    ATTACK_DIR = "data/AttacksLogs"
    attack_paths = sorted(glob.glob(os.path.join(ATTACK_DIR, "*.json")))
    print(f"Found {len(attack_paths)} files in {ATTACK_DIR}")

    def df_to_encoded_seqs(df_local: pd.DataFrame):
        out_tok, tree_tok, ctx_tok = build_token_seqs_from_df(df_local)
        out_s = encode_token_seqs(out_tok, out_vocab)
        tree_s = encode_token_seqs(tree_tok, tree_vocab)
        ctx_s = encode_token_seqs(ctx_tok, ctx_vocab)
        return out_s, tree_s, ctx_s

    attack_out, attack_tree, attack_ctx = [], [], []
    attack_meta = []

    for path in attack_paths:
        with open(path, "r", encoding="utf-8") as f:
            attack_data = json.load(f)

        attack_df = pd.DataFrame.from_records(attack_data)
        if "_id" in attack_df.columns:
            attack_df = attack_df.drop(columns=["_id"])

        a_out, a_tree, a_ctx = df_to_encoded_seqs(attack_df)

        attack_out.extend(a_out)
        attack_tree.extend(a_tree)
        attack_ctx.extend(a_ctx)

        for local_i in range(len(a_out)):
            attack_meta.append((os.path.basename(path), local_i))

    # filtriamo le transazioni di attacco troppo corte per essere scorse (i.e. più corte di block_size + 1) 
    # dato che la nostra funzione di scoring si basa sull'avere almeno una finestra completa di token per calcolare il punteggio
    attack_out_f, attack_tree_f, attack_ctx_f, attack_meta_f = [], [], [], []
    for i in range(len(attack_out)):
        if attack_out[i].numel() > BLOCK_SIZE + 1:
            attack_out_f.append(attack_out[i])
            attack_tree_f.append(attack_tree[i])
            attack_ctx_f.append(attack_ctx[i])
            attack_meta_f.append(attack_meta[i])

    attack_out, attack_tree, attack_ctx, attack_meta = attack_out_f, attack_tree_f, attack_ctx_f, attack_meta_f

    n_attacks = len(attack_out)
    print("Total attack transactions collected:", len(attack_meta))
    print("Scorable attack transactions:", n_attacks)

    if n_attacks == 0:
        print("No attack transactions found. Exiting.")
        return

    # usiamo tutte le transazioni di attacco scorable e tutte le transazioni normali scorable per l'eval
    scorable_val = [i for i in val_idx if len(out_token_seqs[i]) > BLOCK_SIZE + 1]
    print("Scorable validation transactions:", len(scorable_val))

    normal_out = [torch.tensor([out_vocab.get(t, out_vocab["[UNK]"]) for t in out_token_seqs[i]], dtype=torch.long) for i in scorable_val]
    normal_tree = [torch.tensor([tree_vocab.get(t, tree_vocab["[UNK]"]) for t in tree_token_seqs[i]], dtype=torch.long) for i in scorable_val]
    normal_ctx = [torch.tensor([ctx_vocab.get(t, ctx_vocab["[UNK]"]) for t in ctx_token_seqs[i]], dtype=torch.long) for i in scorable_val]

    print("Scoring attack transactions...")
    attack_scores = score_transactions(
        model=model,
        loss_function_none=score_loss_fn,
        out_seqs=attack_out,
        tree_seqs=attack_tree,
        ctx_seqs=attack_ctx,
        block_size=BLOCK_SIZE,
        device=DEVICE,
    )

    print("Scoring normal transactions...")
    normal_scores = score_transactions(
        model=model,
        loss_function_none=score_loss_fn,
        out_seqs=normal_out,
        tree_seqs=normal_tree,
        ctx_seqs=normal_ctx,
        block_size=BLOCK_SIZE,
        device=DEVICE,
    )

    atk = [s[SCORE_KEY] for s in attack_scores]
    nor = [s[SCORE_KEY] for s in normal_scores]

    def q(x, p):
        return float(torch.quantile(torch.tensor(x, dtype=torch.float64), p).item())

    print("\nScore stats (mean_token_nll):")
    print(
        f" Attacks: n={len(atk)}  min={min(atk):.4f}  p50={q(atk,0.5):.4f}  "
        f"p90={q(atk,0.9):.4f}  max={max(atk):.4f}"
    )
    print(
        f" Normals: n={len(nor)}  min={min(nor):.4f}  p50={q(nor,0.5):.4f}  "
        f"p90={q(nor,0.9):.4f}  max={max(nor):.4f}"
    )
    print(f" Threshold: {thr:.4f}")

    y_true = [1] * len(attack_scores) + [0] * len(normal_scores)
    y_score = [s[SCORE_KEY] for s in attack_scores] + [s[SCORE_KEY] for s in normal_scores]
    y_pred = [1 if sc > thr else 0 for sc in y_score]

    TP = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    TN = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    FP = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    FN = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    precision = TP / max(1, TP + FP)
    recall = TP / max(1, TP + FN)
    f1 = (2 * precision * recall) / max(1e-12, (precision + recall))
    acc = (TP + TN) / max(1, (TP + TN + FP + FN))

    print("\n================== BALANCED EVAL ==================")
    print(f"Score metric: {SCORE_KEY}")
    print(f"Threshold: {thr:.6f}")
    print(f"N attacks: {len(attack_scores)}  N normals: {len(normal_scores)}")
    print(f"TP={TP}  FP={FP}  TN={TN}  FN={FN}")
    print(f"Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}  Acc={acc:.4f}")

    # false negatives 
    missed = []
    for idx, (sc, yp) in enumerate(zip([s[SCORE_KEY] for s in attack_scores], y_pred[: len(attack_scores)])):
        if yp == 0:
            fn_file, fn_local = attack_meta[idx]
            missed.append((idx, sc, fn_file, fn_local))
    missed.sort(key=lambda x: x[1])

    print("\nMissed attacks (FN), lowest score first:")
    for idx, sc, fn_file, fn_local in missed[:50]:
        print(f"  global_attack_idx={idx}  score={sc:.6f}  file={fn_file}  local_tx={fn_local}")

    # top scored attacks
    top_attacks = sorted(
        [(i, s[SCORE_KEY], attack_meta[i][0], attack_meta[i][1]) for i, s in enumerate(attack_scores)],
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    print("\nTop-10 attacks by score:")
    for i, sc, fn_file, fn_local in top_attacks:
        print(f"  global_attack_idx={i}  score={sc:.6f}  file={fn_file}  local_tx={fn_local}")


def main(run_train=True, run_validate=True):
    ckpt = None
    if run_train:
        ckpt = train_main()

    if run_validate:
        if ckpt is not None:
            validation_main(MODEL_PATH=ckpt)
        else:
            validation_main()


if __name__ == "__main__":
    main(run_train=False)