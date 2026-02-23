import sys
import time
import torch
from torch.amp import autocast, GradScaler


def train_network(model, optimizer, loss_function, trainloader, device, epoch, num_epochs):
    print(f"Epoch {epoch + 1}: Training Started")
    sys.stdout.flush()

    scaler = GradScaler()

    model.train()
    running_loss = 0.0
    num_batches = 0

    tokens_seen = 0
    start_time = time.perf_counter()

    for out_ids, tree_ids, ctx_ids, y in trainloader:
        out_ids = out_ids.to(device, non_blocking=True)
        tree_ids = tree_ids.to(device, non_blocking=True)
        ctx_ids = ctx_ids.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda"):
            outputs = model(out_ids, tree_ids, ctx_ids)
            loss = loss_function(outputs.transpose(2, 1), y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        num_batches += 1
        tokens_seen += y.numel()

        if num_batches % 50 == 0:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            tps = tokens_seen / max(1e-9, elapsed)
            print(
                f"Batch {num_batches}/{len(trainloader)} | "
                f"loss {running_loss / num_batches:.4f} | "
                f"tokens/s {tps:,.0f}"
            )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    avg_loss = running_loss / max(1, num_batches)
    tps = tokens_seen / max(1e-9, elapsed)
    print(f"Epoch {epoch + 1}/{num_epochs} | Avg train loss: {avg_loss:.4f} | tokens/s: {tps:,.0f}")


def valid_network(model, loss_function, valloader, device, epoch, num_epochs):
    print(f"Epoch {epoch + 1}: Validation Started")
    sys.stdout.flush()
    model.eval()

    total_tokens = 0
    total_nll = 0.0
    num_windows = 0
    window_nlls = []

    with torch.no_grad():
        for out_ids, tree_ids, ctx_ids, y in valloader:
            out_ids = out_ids.to(device, non_blocking=True)
            tree_ids = tree_ids.to(device, non_blocking=True)
            ctx_ids = ctx_ids.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            outputs = model(out_ids, tree_ids, ctx_ids)
            B, T, V = outputs.shape

            logits = outputs.reshape(B * T, V)
            targets = y.reshape(B * T)

            per_token_loss = loss_function(logits, targets)

            total_nll += per_token_loss.sum().item()
            total_tokens += per_token_loss.numel()

            per_window_nll = per_token_loss.reshape(B, T).sum(dim=1)
            window_nlls.extend(per_window_nll.detach().cpu().tolist())
            num_windows += B

    avg_nll_per_token = total_nll / max(1, total_tokens)
    avg_nll_per_window = sum(window_nlls) / max(1, len(window_nlls))

    print(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"Val NLL/token: {avg_nll_per_token:.4f} | "
        f"Val NLL/window(sum): {avg_nll_per_window:.2f} | "
        f"Val windows: {num_windows}"
    )

    return window_nlls, avg_nll_per_token, avg_nll_per_window


def score_transactions(model, loss_function_none, out_seqs, tree_seqs, ctx_seqs, block_size, device, stride=None):
    model.eval()
    results = []
    stride = block_size if stride is None else stride

    with torch.no_grad():
        for j in range(len(out_seqs)):
            out = out_seqs[j].to(device)
            tree = tree_seqs[j].to(device)
            ctx = ctx_seqs[j].to(device)

            L = out.numel()
            if L <= block_size + 1:
                continue

            total_nll = 0.0
            total_tokens = 0
            max_window_nll = -1e30
            num_windows = 0

            for start in range(0, L - block_size - 1, stride):
                x_out = out[start:start + block_size].unsqueeze(0)
                x_tree = tree[start:start + block_size].unsqueeze(0)
                x_ctx = ctx[start:start + block_size].unsqueeze(0)
                y = out[start + 1:start + 1 + block_size].unsqueeze(0)

                outputs = model(x_out, x_tree, x_ctx)
                B, T, V = outputs.shape

                logits = outputs.reshape(B * T, V)
                targets = y.reshape(B * T)

                per_token = loss_function_none(logits, targets) 
                w_nll = float(per_token.sum().item())

                total_nll += w_nll
                total_tokens += T
                max_window_nll = max(max_window_nll, w_nll)
                num_windows += 1

            results.append({
                "tx_index": j,
                "num_tokens": int(L),
                "total_nll": float(total_nll),
                "mean_token_nll": float(total_nll / max(1, total_tokens)),
                "num_windows": int(num_windows),
                "max_window_nll": float(max_window_nll),
            })

    return results
