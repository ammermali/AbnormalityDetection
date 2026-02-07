import torch
import sys


def train_network(model, optimizer, loss_function, trainloader, device, epoch, num_epochs):
    print(f"Epoch {epoch + 1}: Training Started")
    sys.stdout.flush()
    model.train()
    running_loss = 0.0
    num_batches = 0

    for out_ids, tree_ids, ctx_ids, y in trainloader:
        out_ids, tree_ids, ctx_ids, y = out_ids.to(device), tree_ids.to(device), ctx_ids.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(out_ids, tree_ids, ctx_ids)

        # Transpose per CrossEntropyLoss: (Batch, Classes, Seq) vs (Batch, Seq)
        loss = loss_function(outputs.transpose(2, 1), y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

        # Facoltativo: stampa progresso ogni tot batch se lento
        # if num_batches % 100 == 0: print(f".", end="")

    avg_loss = running_loss / num_batches
    print(f"Epoch {epoch + 1}/{num_epochs} | Avg train loss: {avg_loss:.4f}")


def valid_network(model, loss_function, valloader, device, epoch, num_epochs):
    print(f"Epoch {epoch + 1}: Validation Started")
    sys.stdout.flush()
    model.eval()
    running_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for out_ids, tree_ids, ctx_ids, y in valloader:
            out_ids, tree_ids, ctx_ids, y = out_ids.to(device), tree_ids.to(device), ctx_ids.to(device), y.to(device)
            outputs = model(out_ids, tree_ids, ctx_ids)
            loss = loss_function(outputs.transpose(2, 1), y)
            running_loss += loss.item()
            num_batches += 1

    avg_loss = running_loss / num_batches
    print(f"Epoch {epoch + 1}/{num_epochs} | Avg val loss: {avg_loss:.4f}")