import torch
from tqdm.auto import tqdm
from save import save_checkpoint


def train_step(model, loader, loss_fn, optimizer, device):
    model.train()
    loss_total, acc_total = 0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_total += loss.item()
        acc_total += (y_pred.argmax(dim=1) == y).float().mean().item()

    return loss_total / len(loader), acc_total / len(loader)


def test_step(model, loader, loss_fn, device):
    model.eval()
    loss_total, acc_total = 0, 0

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            y_pred = model(x)

            loss_total += loss_fn(y_pred, y).item()
            acc_total += (y_pred.argmax(dim=1) == y).float().mean().item()

    return loss_total / len(loader), acc_total / len(loader)


def train(model, train_loader, test_loader, loss_fn, optimizer, epochs, device, scheduler=None):
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "best_acc": [],
    }

    best_acc = 0

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device
        )

        test_loss, test_acc = test_step(
            model,
            test_loader,
            loss_fn,
            device
        )

        save_checkpoint(model, optimizer, epoch + 1, results, "models/latest.pth")

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(model, optimizer, epoch + 1, results, "models/best.pth")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["best_acc"].append(best_acc)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, "
            f"best_acc={best_acc:.4f}"
        )

    return results                                                                                         