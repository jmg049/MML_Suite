import torch
from torch.utils.data import DataLoader
from data.avmnist import AVMNIST
from modalities import Modality

# instantiate your dataset exactly as you would for training
train_ds = AVMNIST(
    data_fp="/home/jmg/code/python/MML_Suite/DATA/avmnist/validation_subset.csv",
    split="train",
    missing_patterns=None,      # full modality
    selected_patterns=None,
    target_modality=Modality.MULTIMODAL,
)

loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=4)

# accumulators: sums and sums of squares
stats = {
    Modality.IMAGE: {"sum": 0.0, "sum2": 0.0, "n": 0},
    Modality.AUDIO: {"sum": 0.0, "sum2": 0.0, "n": 0},
}

with torch.no_grad():
    for batch in loader:
        for mod in [Modality.IMAGE, Modality.AUDIO]:
            if mod in batch:
                x = batch[mod]                   # e.g. [B, ...]
                stats[mod]["sum"]  += x.sum().item()
                stats[mod]["sum2"] += (x**2).sum().item()
                stats[mod]["n"]    += x.numel()

# compute mean & std
for mod, st in stats.items():
    μ = st["sum"] / st["n"]
    var = st["sum2"] / st["n"] - μ**2
    σ = var**0.5
    print(f"{mod.name:>6} → mean = {μ:.4f},  std = {σ:.4f}")
