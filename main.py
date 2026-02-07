import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.tokenizer import tokenizer, build_tree_from_output, build_context_from_tokens
from src.dataset import NextTokenTripleDataset
from src.model import GPTModel
from src.engine import train_network, valid_network


def build_vocab_and_ids(token_list, pad_token="[PAD]", unk_token="[UNK]"):
    """Funzione helper per creare vocab e tensori ID"""
    vocab = {pad_token: 0, unk_token: 1}
    for t in token_list:
        if t not in vocab:
            vocab[t] = len(vocab)

    ids = torch.tensor(
        [vocab.get(t, vocab[unk_token]) for t in token_list],
        dtype=torch.long
    )
    return vocab, ids


def main():
    # 1. Configurazione
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    JSON_PATH = "data/LogNuovoPiccolo.json"  # Assicurati che il percorso sia giusto
    EMBEDDING_DIM = 64
    BLOCK_SIZE = 128
    BATCH_SIZE = 8
    EPOCHS = 10

    # ... (Codice precedente: Configurazione DEVICE, etc.) ...

    # 2. Caricamento e Fusione dei Dati (MERGE)
    print("Loading datasets...")

    # Lista dei file da caricare
    file_paths = [
        "data/LogNuovoPiccolo.json",
        "data/LogNuovoGrande.json"
    ]

    dataframes_list = []

    for path in file_paths:
        print(f"Reading {path}...")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            temp_df = pd.DataFrame.from_records(data)

            # Pulizia preliminare (rimuove _id se esiste)
            if '_id' in temp_df.columns:
                temp_df = temp_df.drop(columns=['_id'])

            dataframes_list.append(temp_df)
            print(f" -> Caricate {len(temp_df)} righe da {path}")

        except FileNotFoundError:
            print(f"ERRORE: Non ho trovato il file {path}. Controlla la cartella 'data'!")
            return  # Ferma tutto se manca un file

    # Unione dei DataFrame (Concatenazione verticale)
    print("Merging datasets...")
    df = pd.concat(dataframes_list, ignore_index=True)

    print(f"TOTALE: Il dataset completo ha {len(df)} righe.")

    # ... (Il resto del codice dal punto 3. Preprocessing in poi rimane IDENTICO) ...
    # 3. Preprocessing (Il Tokenizer Gigante)
    print("Tokenizing data (this might take a while)...")
    raw_output = tokenizer(df)
    tokens_flat, tree_list = build_tree_from_output(raw_output)
    context_list = build_context_from_tokens(tokens_flat)

    print(f"Total tokens: {len(tokens_flat)}")

    # 4. Costruzione Vocabolari e ID
    print("Building vocabularies...")
    out_vocab, out_ids = build_vocab_and_ids(raw_output)
    tree_vocab, tree_ids = build_vocab_and_ids([str(x) for x in tree_list])  # tree elements converted to str
    ctx_vocab, ctx_ids = build_vocab_and_ids(context_list)

    # 5. Dataset e DataLoader
    print("Creating Datasets...")
    dataset = NextTokenTripleDataset(out_ids, tree_ids, ctx_ids, block_size=BLOCK_SIZE)

    N = len(dataset)
    n_train = int(0.6 * N)
    n_val = int(0.2 * N)
    n_test = N - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 6. Inizializzazione Modello
    print("Initializing Model...")
    model = GPTModel(
        vocab_size=len(out_vocab),
        tree_vocab_size=len(tree_vocab),
        ctx_vocab_size=len(ctx_vocab),
        embed_dim=EMBEDDING_DIM,
        feed_forward_dim=4 * EMBEDDING_DIM,
        num_heads=8,
        key_dim=EMBEDDING_DIM
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    # 7. Training Loop
    print("Starting Training...")
    for epoch in range(EPOCHS):
        train_network(model, optimizer, loss_fn, train_loader, DEVICE, epoch, EPOCHS)
        valid_network(model, loss_fn, val_loader, DEVICE, epoch, EPOCHS)

    # 8. Salvataggio
    torch.save({"model_state": model.state_dict()}, "models/anomaly_detection_model.pt")
    print("Model saved!")


if __name__ == "__main__":
    main()