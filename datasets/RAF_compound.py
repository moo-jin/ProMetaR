import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD  # fallback 분리기 사용 가능 (없어도 무방)

CLASSNAMES = [
    "Happily Surprised", "Happily Disgusted", "Sadly Fearfult", "Sadly Angry", "Sadly Surprised", "Sadly Disgusted", "Fearfully Angry", "Fearfully Surprised", "Angrily Surprised", "Angrily Disgusted", "Disgustedly Surprised"
]


@DATASET_REGISTRY.register()
class RAF(DatasetBase):

    dataset_dir = "RAF_coumpound"  # 대소문자 정확히 반영

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)

        self.split_path = os.path.join(self.dataset_dir, "split_zhou_raf_compound.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # Split 존재 여부 확인
        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.dataset_dir)
        else:
            raise FileNotFoundError(f"split_zhou_affectnet.json not found at {self.split_path}")

        # Few-shot 구성
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            if os.path.exists(preprocessed):
                print(f"📦 Loading few-shot split from {preprocessed}")
                with open(preprocessed, "rb") as f:
                    data = pickle.load(f)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"💾 Saving few-shot split to {preprocessed}")
                with open(preprocessed, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Subsample 클래스 (base / new / all)
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        # 최종 데이터셋 초기화
        super().__init__(train_x=train, val=val, test=test)