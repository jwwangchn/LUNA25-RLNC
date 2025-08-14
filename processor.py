import io
import os
import logging
import numpy as np
import dataloader
import torch

from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2

from models.rlnc_model import RlncModelV1, RlncModelV2, MultiModelEnsembler, EmaModelWrapper, MultiAugmentedInferencer

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)

class MalignancyProcessor:
    """
    Loads a chest CT scan and predicts the malignancy around a nodule.
    """
    def __init__(self,
                 mode="2D",
                 suppress_logs=False,
                 model_root="",
                 model_folder="LUNA25-baseline-2D"):
        self.patch_size_px = 64
        self.patch_size_mm = 50
        self.model_folder = model_folder
        self.operation_mode = mode
        self.quiet_mode = suppress_logs
        self.fold_num = 5

        if not self.quiet_mode:
            logging.info("Initializing the deep learning system")
        
        if model_root == "":
            self.model_directory = "/opt/app/resources/"
        else:
            self.model_directory = model_root
            
        self._initialize_model()
            
    def _initialize_model(self):
        """
        Initializes the model based on the provided model name.
        """
        ensemble_models = []
        for fold_idx in range(self.fold_num):
            checkpoint_path = os.path.join(self.model_directory, self.model_folder, f"model{fold_idx}.bin")
            print("load model: ", checkpoint_path)
            decrypted_buffer = self.decrypt(checkpoint_path, "luna25")
            checkpoint = torch.load(decrypted_buffer)
        
            model = RlncModelV2(classes=1, pretrained=False).cuda()
            model = EmaModelWrapper(model)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            ensemble_models.append(model.module)
        
        checkpoint_path = os.path.join(
            self.model_directory,
            self.model_folder,
            "fullmodel.bin"
        )
        print("load model: ", checkpoint_path)
        decrypted_buffer = self.decrypt(checkpoint_path, "luna25")
        checkpoint = torch.load(decrypted_buffer)
        model = RlncModelV1(classes=1, pretrained=False).cuda()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        ensemble_models.append(model)
        
        self.ensemble_model = MultiModelEnsembler(ensemble_models)

    def define_inputs(self, ct_volume, header_info, nodule_coordinates):
        """
        Sets the CT scan volume, header information, and nodule coordinates.
        """
        self.ct_data = ct_volume
        self.header = header_info
        self.nodule_positions = nodule_coordinates

    def extract_patch(self, coordinates, target_shape, interpolation_mode):
        """
        Extracts a patch centered at the given coordinates from the CT scan.
        """
        target_spacing = (self.patch_size_mm / self.patch_size_px,) * 3
        resampled_patch = dataloader.extract_patch(
            CTData=self.ct_data,
            coord=coordinates,
            srcVoxelOrigin=self.header["origin"],
            srcWorldMatrix=self.header["transform"],
            srcVoxelSpacing=self.header["spacing"],
            output_shape=target_shape,
            voxel_spacing=target_spacing,
            coord_space_world=True,
            mode=interpolation_mode,
        )

        resampled_patch = resampled_patch.astype(np.float32)
        resampled_patch = dataloader.clip_and_scale(resampled_patch)
        return resampled_patch

    def decrypt(self, encrypted_path: str, password: str):
        with open(encrypted_path, 'rb') as f:
            salt = f.read(16)
            nonce = f.read(16)
            tag = f.read(16)
            ciphertext = f.read()

        key = PBKDF2(password, salt, dkLen=32, count=1000000)

        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        try:
            plaintext = cipher.decrypt_and_verify(ciphertext, tag)
        except (ValueError, KeyError):
            return None
        
        buffer = io.BytesIO(plaintext)
        buffer.seek(0)
        return buffer

    def _process_model(self, execution_mode):
        """
        Processes the model based on the execution mode (2D or 3D).
        """
        if not self.quiet_mode:
            logging.info(f"Running inference in {execution_mode} mode")

        patch_dim = self.patch_size_px
        if execution_mode == "2D":
            target_shape = (1, patch_dim, patch_dim)
        else:
            target_shape = (patch_dim, patch_dim, patch_dim)

        patch_collection = [
            self.extract_patch(position, target_shape, execution_mode)
            for position in self.nodule_positions
        ]
        patch_tensor = torch.from_numpy(np.array(patch_collection)).cuda()
        infer = MultiAugmentedInferencer(model=self.ensemble_model, device=patch_tensor.device)

        return infer(patch_tensor).cpu().numpy()

    def predict(self):
        """
        Predicts malignancy scores for all nodule positions.
        Returns a tuple of (probabilities, raw_logits).
        """
        raw_logits = self._process_model(self.operation_mode)
        probabilities = torch.sigmoid(torch.from_numpy(raw_logits)).numpy()
        
        return probabilities, raw_logits