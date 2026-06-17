from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader


def get_zinc_loaders(batch_size: int = 1, root: str = "data/"):
    train_dataset = ZINC(root=root, subset=True, split="train")
    val_dataset = ZINC(root=root, subset=True, split="val")
    test_dataset = ZINC(root=root, subset=True, split="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
