from pathlib import Path
import sys
sys.path.append(".")
from models.rlnc_models import RlncModelV2

class Configuration(object):
    def __init__(self) -> None:
        self.WORKDIR = Path("./")
        
        # Path to the nodule blocks folder provided for the LUNA25 training data. 
        self.DATADIR_LUNA25 = Path("./data/LUNA25/luna25_nodule_blocks")
        self.DATADIR_LIDC = Path("./data/LIDC-LUNA25")
        
        # Path to the folder containing the CSVs for training and validation.
        self.LUNA25_CSV_FP = Path("./data/LUNA25/luna25_dataset_csv/LUNA25_Public_Training_Development_Data.csv")
        self.LIDC_CSV_FP = Path("./data/LIDC-LUNA25/Meta/meta_info_annotated.csv")

        # Results will be saved in the /results/ directory, inside a subfolder named according to the specified EXPERIMENT_NAME and MODE.
        self.EXPERIMENT_DIR = self.WORKDIR / "results"
        if not self.EXPERIMENT_DIR.exists():
            self.EXPERIMENT_DIR.mkdir(parents=True)
        
        self.MODE = "3D" # 2D or 3D
        self.MODE_CLASS = RlncModelV2(classes=1, pretrained=True)

        # Training parameters
        self.SEED = 2025
        self.NUM_WORKERS = 32
        self.SIZE_MM = 50
        self.SIZE_PX = 64
        self.BATCH_SIZE = 256
        self.ROTATION = ((-20, 20), (-20, 20), (-20, 20))
        self.TRANSLATION = True
        self.EPOCHS = 50
        self.WARM_EPOCHS = 5
        self.PATIENCE = 15
        self.PATCH_SIZE = [64, 128, 128]
        self.LEARNING_RATE = 3e-4
        self.WEIGHT_DECAY = 1e-4
        self.EXPERIMENT_NAME = f"LUNA25-RLNC-epoch{self.EPOCHS}"
        self.K_FOLDS = 5
        self.EMA_DECAY = 0.95
    
    def convert_paths_to_str(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_paths_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_paths_to_str(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_paths_to_str(item) for item in obj)
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
    
    def to_dict(self):
        res = self.convert_paths_to_str(self.__dict__)
        
        return res

config = Configuration()