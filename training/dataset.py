import os
import pandas as pd
from torch.utils.data import Dataset

from preprocessing.image_preprocess import preprocess_image
from preprocessing.text_preprocess import preprocess_text


class MultimodalDataset(Dataset):
    def __init__(self, csv_path, image_root):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root

        # match symptoms.csv exactly
        required_cols = {"image_id", "text", "label", "label_name"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(
                f" CSV must contain columns: {required_cols}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # correct image path
        image_path = os.path.join(
            self.image_root,
            row["label_name"],
            f"{row['image_id']}.jpg"
        )

        image = preprocess_image(image_path)
        text = preprocess_text(row["text"])

        return {
            "image": image,
            "input_ids": text["input_ids"].squeeze(0),
            "attention_mask": text["attention_mask"].squeeze(0),
            "label": int(row["label"])
        }
