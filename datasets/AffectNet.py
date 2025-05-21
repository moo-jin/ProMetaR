from dassl.data.datasets import Datum, DatasetBase
from dassl.data.data_manager import build_data_loader
import os
import pandas as pd
from PIL import Image
from torchvision import transforms

class AffectNet(DatasetBase):
    dataset_dir = "affectnet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        data_dir = os.path.join(root, self.dataset_dir)

        train_file = os.path.join(data_dir, "training.csv")
        val_file = os.path.join(data_dir, "validation.csv")

        def read_csv(csv_path):
            df = pd.read_csv(csv_path)
            items = []
            for _, row in df.iterrows():
                img_path = os.path.join(data_dir, "images", row["file_path"])
                label = int(row["expression"])
                items.append(Datum(impath=img_path, label=label))
            return items

        train = read_csv(train_file)
        val = read_csv(val_file)
        test = val  # AffectNet은 validation set이 test로도 쓰임

        super().__init__(train_x=train, val=val, test=test)

        self.classnames = [
            "neutral", "happy", "sad", "surprise", "fear", "disgust", "anger", "contempt"
        ]