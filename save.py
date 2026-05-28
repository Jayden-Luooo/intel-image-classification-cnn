from pathlib import Path
import torch


def save_checkpoint(model, optimizer, epoch, results, path):
    """Save model, optimizer, epoch, and training results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "results": results
    }, path)