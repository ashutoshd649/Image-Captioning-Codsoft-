"""
src/dataset.py
Flickr8k dataset loader compatible with PyTorch DataLoader.
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import config


def load_captions_df(captions_file: str) -> pd.DataFrame:
    """
    Load captions.txt (Flickr8k format).
    Expected columns: image, caption
    """
    df = pd.read_csv(captions_file)
    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]
    if "image" not in df.columns:
        df.rename(columns={df.columns[0]: "image", df.columns[1]: "caption"}, inplace=True)
    df["caption"] = df["caption"].str.strip()
    df["image"]   = df["image"].str.strip()
    return df


def get_image_transforms(split: str = "train") -> transforms.Compose:
    """Return torchvision transforms for train / val-test splits."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32)),
            transforms.RandomCrop(config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std= [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std= [0.229, 0.224, 0.225]),
        ])


class Flickr8kDataset(Dataset):
    """
    PyTorch Dataset for Flickr8k image-caption pairs.

    Args:
        df        : DataFrame with columns [image, caption]
        vocab     : Vocabulary object
        images_dir: Path to folder containing JPEG images
        split     : 'train' | 'val' | 'test'
    """

    def __init__(self, df: pd.DataFrame, vocab, images_dir: str, split: str = "train"):
        self.df         = df.reset_index(drop=True)
        self.vocab      = vocab
        self.images_dir = images_dir
        self.transform  = get_image_transforms(split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["image"])

        # Load & transform image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Tokenise caption
        caption_tokens = (
            [self.vocab.stoi[config.START_TOKEN]]
            + self.vocab.encode(row["caption"])
            + [self.vocab.stoi[config.END_TOKEN]]
        )
        caption_tensor = torch.tensor(caption_tokens, dtype=torch.long)
        return image, caption_tensor


def collate_fn(batch):
    """Pad captions to the same length within a batch."""
    images, captions = zip(*batch)
    images   = torch.stack(images, dim=0)
    lengths  = [len(c) for c in captions]
    max_len  = max(lengths)
    padded   = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded[i, :len(cap)] = cap
    return images, padded, torch.tensor(lengths, dtype=torch.long)


def build_dataloaders(df: pd.DataFrame, vocab, images_dir: str):
    """
    Split df into train / val / test and return three DataLoaders.
    """
    n       = len(df)
    n_train = int(n * config.TRAIN_SPLIT)
    n_val   = int(n * config.VAL_SPLIT)
    n_test  = n - n_train - n_val

    df_train = df.iloc[:n_train]
    df_val   = df.iloc[n_train : n_train + n_val]
    df_test  = df.iloc[n_train + n_val :]

    train_ds = Flickr8kDataset(df_train, vocab, images_dir, split="train")
    val_ds   = Flickr8kDataset(df_val,   vocab, images_dir, split="val")
    test_ds  = Flickr8kDataset(df_test,  vocab, images_dir, split="test")

    loader_kwargs = dict(
        batch_size  = config.BATCH_SIZE,
        collate_fn  = collate_fn,
        pin_memory  = (config.DEVICE == "cuda"),
        num_workers = 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
