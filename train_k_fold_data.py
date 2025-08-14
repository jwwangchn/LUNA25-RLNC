import os
import sys
import logging
import json
import argparse
import importlib
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime

import copy
import pprint
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from sklearn.model_selection import StratifiedGroupKFold
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, roc_curve

from dataloader import CTCaseDataset, worker_init_fn

# global variables
config = None
WRITER = None
DEVICE = None

# set seed
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# init logger
def init_logger(log_path: str = "logs/train.log"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(levelname)s][%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        filename=log_path,
        filemode="w"
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s][%(asctime)s] %(message)s", "%H:%M:%S")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


def load_config(config_name: str):
    """
    Load the configuration file and return its contents.

    Args:
        config_name (str): The name of the configuration file.

    Returns:
        dict: The contents of the configuration file.
    """
    sys.path.append(os.path.abspath('.'))
    module_path = f"configs.{config_name}"
    config_module = importlib.import_module(module_path)
    return getattr(config_module, 'config')

class ModelEma(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

class Evaluator:
    def __init__(self, y_true, y_pred, threshold=0.5):
        self.y_true = y_true
        self.y_pred = y_pred
        
        self.y_pred_class = (self.y_pred >= threshold).astype(int)

    @staticmethod
    def calculate_luna25_auc(y_true, y_pred):
        # calc AUC
        auc = roc_auc_score(y_true, y_pred)

        # Bootstrapping for 95% confidence intervals
        n_bootstraps = 1000
        rng = np.random.RandomState(seed=42)
        bootstrapped_aucs = []

        for _ in range(n_bootstraps):
            # Resample the data
            indices = rng.choice(len(y_true), len(y_true), replace=True)
            if len(np.unique(y_true[indices])) < 2:
                # Skip this resample if only one class is present
                continue
            score = roc_auc_score(y_true[indices], y_pred[indices])
            bootstrapped_aucs.append(score)

        # calculate 95% confidence interval
        ci_lower = np.percentile(bootstrapped_aucs, 2.5)
        ci_upper = np.percentile(bootstrapped_aucs, 97.5)

        return {"auc": auc, "ci_lower": ci_lower, "ci_upper": ci_upper}

    @staticmethod
    def calculate_luna25_sensitivity(y_true, y_pred):
        """
        Computes the sensitivity (recall) at 95% specificity for a classifier.
        
        Parameters:
            y_true (np.ndarray): Ground truth binary labels (0 = benign, 1 = malignant).
            y_pred (np.ndarray): Predicted probability scores from the classifier.

        Returns:
            dict: Sensitivity (recall) at 95% specificity, and the threshold used.
        """
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        
        # Find the threshold corresponding to 95% specificity (FPR = 1 - specificity)
        target_fpr = 1 - 0.95  # 5% false positive rate
        idx = np.where(fpr <= target_fpr)[-1]  # Get the last index where FPR <= 5%
        
        if idx.size == 0:
            raise ValueError("No threshold meets the target specificity requirement.")
        
        # Extract sensitivity (TPR) and threshold
        sensitivity = tpr[idx[-1]]
        threshold = thresholds[idx[-1]]

        return {"sensitivity": sensitivity, "threshold": threshold}
    
    @staticmethod
    def calculate_luna25_specificity(y_true, y_pred):
        """
        Computes the specificity at 95% sensitivity for a classifier.
        
        Parameters:
            y_true (np.ndarray): Ground truth binary labels (0 = benign, 1 = malignant).
            y_pred (np.ndarray): Predicted probability scores from the classifier.

        Returns:
            dict: Specificity at 95% sensitivity, and the threshold used.
        """
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        
        # Find the threshold corresponding to 95% sensitivity (TPR = 0.95)
        target_tpr = 0.95  # Sensitivity (TPR) threshold
        idx = np.where(tpr >= target_tpr)[0]  # Get indices where TPR >= target_tpr
        
        if idx.size == 0:
            raise ValueError("No threshold meets the target sensitivity requirement.")
        
        # Extract specificity (1 - FPR) and threshold
        specificity = 1 - fpr[idx[0]]
        threshold = thresholds[idx[0]]
        
        return {"specificity": specificity, "threshold": threshold}

    def run(self):
        # a. AUC (Area Under ROC Curve)
        fpr, tpr, _ = metrics.roc_curve(self.y_true, self.y_pred)
        auc_metric = metrics.auc(fpr, tpr)

        # b. Average Precision / PR-AUC
        ap_score = metrics.average_precision_score(self.y_true, self.y_pred)

        # c. Accuracy
        accuracy = metrics.accuracy_score(self.y_true, self.y_pred_class)

        # d. Precision
        precision = metrics.precision_score(self.y_true, self.y_pred_class, zero_division=0)

        # e. Recall / Sensitivity
        recall = metrics.recall_score(self.y_true, self.y_pred_class, zero_division=0)

        # f. F1-Score
        f1 = metrics.f1_score(self.y_true, self.y_pred_class, zero_division=0)

        # g. Specificity
        specificity = metrics.recall_score(self.y_true, self.y_pred_class, pos_label=0)

        # h. Confusion Matrix
        confusion_matrix = metrics.confusion_matrix(self.y_true, self.y_pred_class)
        
        # i. Classification Report
        class_report = metrics.classification_report(self.y_true,
                                                     self.y_pred_class,
                                                     target_names=['Class 0', 'Class 1'],
                                                     zero_division=0)
        
        # j. calc offical metrics
        official_auc = Evaluator.calculate_luna25_auc(self.y_true, self.y_pred)
        official_sensitivity = Evaluator.calculate_luna25_sensitivity(self.y_true, self.y_pred)
        official_specificity = Evaluator.calculate_luna25_specificity(self.y_true, self.y_pred)
            
        return {
            "auc": auc_metric,
            "ap_score": ap_score,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'confusion_matrix': confusion_matrix,
            'classification_report': class_report,
            "official_auc": official_auc,
            "official_sensitivity": official_sensitivity,
            "official_specificity": official_specificity
        }

class KFoldManager:
    def __init__(self, config):
        self.df_luna25 = pd.read_csv(config.LUNA25_CSV_FP)
        self.df_lidc = pd.read_csv(config.LIDC_CSV_FP)
        self.config = config
    
        self.dataset = self._create_dataset_luna25()
        self.dataset_lidc = self._create_dataset_lidc()
        self.fold_indices = self._get_fold_indices()
        
        self.fold_idx = 1
        self.pos_weight = None

    def _create_dataset_luna25(self):
        dataset = CTCaseDataset(
            data_dir=self.config.DATADIR_LUNA25,
            dataset=self.df_luna25,
            mode=self.config.MODE,
            rotations=self.config.ROTATION,
            translations=self.config.TRANSLATION,
            size_mm=self.config.SIZE_MM,
            size_px=self.config.SIZE_PX,
            patch_size=self.config.PATCH_SIZE
        )
        
        return dataset
    
    def _create_dataset_lidc(self):
        dataset = CTCaseDataset(
            data_dir=self.config.DATADIR_LIDC,
            dataset=self.df_lidc,
            mode=self.config.MODE,
            rotations=self.config.ROTATION,
            translations=self.config.TRANSLATION,
            size_mm=self.config.SIZE_MM,
            size_px=self.config.SIZE_PX,
            patch_size=self.config.PATCH_SIZE,
        )
        
        return dataset

    def _get_fold_indices(self):
        skf = StratifiedGroupKFold(n_splits=self.config.K_FOLDS, shuffle=True, random_state=42)
        _ = np.zeros(len(self.df_luna25.label.values))
        indices = skf.split(_, self.df_luna25.label.values, self.df_luna25.PatientID.values)
        
        return indices
    
    def _get_balanced_sampler(self, subset_indices, factor):
        labels = self.df_luna25.label.values[subset_indices]
        labels_lidc = self.df_lidc.label.values[:]
        n_samples = len(labels) + len(labels_lidc)
        
        full_labels = np.concatenate((labels, labels_lidc), axis=0)
        
        unique, cnts = np.unique(full_labels, return_counts=True)
        cnt_dict = dict(zip(unique, cnts))
        
        self.pos_weight = torch.tensor([cnt_dict[0] / factor / cnt_dict[1]]).to(DEVICE)
        
        logging.info(f"numbers: label=0->{cnt_dict[0]}, label=1->{cnt_dict[1]}")
        weights = [
            (n_samples / factor) / float(cnt_dict[label]) if label == 1 
            else n_samples / float(cnt_dict[label])
            for label in full_labels
        ]
        logging.info(f"weights: label=0->{n_samples / float(cnt_dict[0])}, label=1->{(n_samples / factor) / float(cnt_dict[1])}")

        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(labels) + len(labels_lidc))

        return sampler

    def get_fold_dataloaders(self, train_indices, valid_indices):
        train_subset = Subset(self.dataset,
                              train_indices)
        train_full = ConcatDataset([train_subset, self.dataset_lidc])
        valid_subset = Subset(self.dataset,
                              valid_indices)
        
        logging.info(f"Fold {self.fold_idx}: full luna25 number: {len(self.dataset)}, train number: {len(train_full)}, luna25 number: {len(train_subset)}, lidc number: {len(self.dataset_lidc)}, validation subset number: {len(valid_subset)}")
        
        self.fold_idx += 1
        
        train_sampler = self._get_balanced_sampler(train_indices, factor=2)
        train_loader = DataLoader(train_full,
                                  batch_size=self.config.BATCH_SIZE,
                                  shuffle=False,
                                  num_workers=self.config.NUM_WORKERS,
                                  pin_memory=True,
                                  sampler=train_sampler,
                                  worker_init_fn=worker_init_fn)
        valid_loader = DataLoader(valid_subset,
                                  batch_size=self.config.BATCH_SIZE,
                                  shuffle=False,
                                  num_workers=self.config.NUM_WORKERS,
                                  pin_memory=True,
                                  sampler=None,
                                  worker_init_fn=worker_init_fn)
        
        return train_loader, valid_loader

# single folder trainer
class KFoldTrainer:
    def __init__(self, config, train_loader, valid_loader, exp_save_root, fold_idx, loss_pos_weight):
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.exp_save_root = exp_save_root
        self.fold_idx = fold_idx
        self.device = DEVICE
        self.model = copy.deepcopy(self.config.MODE_CLASS).to(self.device)
        self.ema_model = ModelEma(self.model, decay=self.config.EMA_DECAY, device=self.device)
        
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler(self.optimizer, self.config.EPOCHS)
        self.loss_function = self._init_loss_fn(loss_pos_weight)
        
        self.best_metric = -1
        self.best_metric_epoch = -1
        self.patience = config.PATIENCE
        self.counter = 0
        self.early_stop_flag = False
        self.current_epoch = 0

        self.y_true, self.y_pred = None, None

    def _init_optimizer(self):        
        return optim.Adam(self.model.parameters(),
                          lr=self.config.LEARNING_RATE,
                          weight_decay=self.config.WEIGHT_DECAY)
        
    def _init_scheduler(self, optimizer, num_epochs):
        return SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=self.config.WARM_EPOCHS),
                CosineAnnealingLR(
                    optimizer, T_max=num_epochs - self.config.WARM_EPOCHS, eta_min=1e-6)
            ],
            milestones=[self.config.WARM_EPOCHS]
        )
    
    def _init_loss_fn(self, pos_weight):
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def _save_model(self, epoch, curr_best_metric):
        model_path = self.exp_save_root / f"fold_{self.fold_idx}_epoch{epoch}.pth"
        ema_model_path = self.exp_save_root / f"fold_{self.fold_idx}_epoch{epoch}_ema.pth"
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.ema_model.state_dict(), ema_model_path)
        
        if curr_best_metric > self.best_metric:
            self.best_metric = curr_best_metric
            self.best_metric_epoch = epoch
            model_path = self.exp_save_root / f"fold_{self.fold_idx}_best.pth"
            ema_model_path = self.exp_save_root / f"fold_{self.fold_idx}_best_ema.pth"
            torch.save(self.model.state_dict(), model_path)
            torch.save(self.ema_model.state_dict(), ema_model_path)

            logging.info(f"Saved new best metric model for Fold {self.fold_idx + 1}, curr best metric: {curr_best_metric:.4f}")

            config_dict = self.config.to_dict()
            if "MODE_CLASS" in config_dict:
                config_dict.pop("MODE_CLASS")

            metadata = {
                "curr_best_metric": self.best_metric,
                "epoch": self.best_metric_epoch,
                "config": config_dict,
                "fold_idx": self.fold_idx,
                "model_path": str(model_path)
            }
            with open(self.exp_save_root / f"fold_{self.fold_idx}_results.json", "w") as f:
                json.dump(metadata, f, indent=4)

    def _update_early_stopping(self, curr_best_metric):
        if curr_best_metric > self.best_metric:
            self.counter = 0
        else:
            self.counter += 1
            if self.counter > self.config.PATIENCE:
                logging.info(f"Early stopping at epoch {self.current_epoch} for Fold {self.fold_idx + 1}")
                self.early_stop_flag = True

    def train_one_epoch(self):
        self.model.train()
        
        epoch_loss = 0.0
        step = 0

        for batch_data in tqdm(self.train_loader, desc=f"Training Epoch {self.current_epoch} Fold {self.fold_idx + 1}"):
            inputs, labels = batch_data["image"].to(self.device), batch_data["label"].float().to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            # Update the EMA model
            self.ema_model.update(self.model)
            
            epoch_loss += loss.item()
            step += 1
            
        self.scheduler.step()
        
        epoch_loss /= step
        logging.info(f"Epoch {self.current_epoch} Fold {self.fold_idx + 1} average train loss: {epoch_loss:.4f}")
        
        return epoch_loss

    def validate_one_epoch(self):
        self.model.eval()
        self.ema_model.module.eval()
        y_true, y_pred, ema_y_pred = [], [], []
        epoch_loss, ema_epoch_loss = 0.0, 0.0
        step = 0
        with torch.no_grad():
            for batch_data in self.valid_loader:
                inputs, labels = batch_data["image"].to(self.device), batch_data["label"].float().to(self.device)
                    
                outputs = self.model(inputs)
                ema_outputs = self.ema_model.module(inputs)
                
                loss = self.loss_function(outputs.squeeze(), labels.squeeze())
                epoch_loss += loss.item()
                ema_loss = self.loss_function(ema_outputs.squeeze(), labels.squeeze())
                ema_epoch_loss += ema_loss.item()
                step += 1

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(F.sigmoid(outputs).cpu().numpy())
                ema_y_pred.extend(F.sigmoid(ema_outputs).cpu().numpy())
                
            epoch_loss /= step
            logging.info(f"Epoch {self.current_epoch} Fold {self.fold_idx + 1} average validate loss: {epoch_loss:.4f}")
            ema_epoch_loss /= step
            logging.info(f"Epoch {self.current_epoch} Fold {self.fold_idx + 1} average validate loss (EMA): {ema_epoch_loss:.4f}")
        
        # calc metrics
        self.y_true, self.y_pred, self.ema_y_pred = np.array(y_true), np.array(y_pred), np.array(ema_y_pred)
        evaluator = Evaluator(self.y_true, self.y_pred)
        metrics = evaluator.run()
        metrics["validate_loss"] = epoch_loss
        
        ema_evaluator = Evaluator(self.y_true, self.ema_y_pred)
        ema_metrics = ema_evaluator.run()
        ema_metrics["validate_loss"] = ema_epoch_loss
        
        return metrics, ema_metrics
    
    def post_training_validate(self):
        model_path = self.exp_save_root / f"fold_{self.fold_idx}_best.pth"
        ckpt = torch.load(model_path)
        self.model.load_state_dict(ckpt)
        
        metrics_dict, ema_metrics_dict = self.validate_one_epoch()
        
        return metrics_dict, ema_metrics_dict

    def step(self):
        if self.early_stop_flag:
            return None
        
        # train single epoch
        epoch_loss = self.train_one_epoch()
        metrics_dict, ema_metrics_dict = self.validate_one_epoch()
        
        self._update_early_stopping(ema_metrics_dict['auc'])
        self._save_model(self.current_epoch, ema_metrics_dict['auc'])
        
        self.current_epoch += 1
        
        metrics_dict["train_loss"] = epoch_loss
        ema_metrics_dict["train_loss"] = epoch_loss
        
        return metrics_dict, ema_metrics_dict

class Pipeline:
    def __init__(self, config, exp_save_root):
        self.config = config
        self.exp_save_root = exp_save_root
        
        # init trainer
        self.fold_trainers = self._initialize_fold_trainers()

    def _initialize_fold_trainers(self):
        kf_manager = KFoldManager(self.config)
        trainers = []
        for fold_idx, (train_indices, valid_indices) in enumerate(kf_manager.fold_indices):
            train_loader, valid_loader = kf_manager.get_fold_dataloaders(train_indices, valid_indices)

            trainer = KFoldTrainer(self.config,
                                   train_loader,
                                   valid_loader,
                                   self.exp_save_root,
                                   fold_idx,
                                   loss_pos_weight=kf_manager.pos_weight)
            trainers.append(trainer)
            
        return trainers

    def _validate_one_epoch_for_all_folds(self):
        y_true, y_pred, ema_y_pred = None, None, None

        for trainer in self.fold_trainers:
            if y_true is None:
                y_true = np.zeros((0,) + trainer.y_true.shape[1:])
                y_pred = np.zeros((0,) + trainer.y_pred.shape[1:])
                ema_y_pred = np.zeros((0,) + trainer.ema_y_pred.shape[1:])

            y_true = np.concatenate([y_true, trainer.y_true], axis=0)
            y_pred = np.concatenate([y_pred, trainer.y_pred], axis=0)
            ema_y_pred = np.concatenate([ema_y_pred, trainer.ema_y_pred], axis=0)
        
            
        return Evaluator(y_true, y_pred).run(), Evaluator(y_true, ema_y_pred).run()

    def _summarize_final_results(self):
        results = [trainer.best_metric for trainer in self.fold_trainers]
        mean_acc = np.mean(results)
        std_acc = np.std(results)
        logging.info(f"K-Fold Validation Completed.")
        logging.info(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        logging.info(f"Individual Fold Accuracies: {[round(a, 4) for a in results]}")

    def run(self):
        # 1. outer epoch
        for global_epoch in range(self.config.EPOCHS):
            print(f"=" * 50 + f"training epoch {global_epoch=}" + f"=" * 50)
            epoch_metrics, ema_epoch_metrics = [], []
            for idx, trainer in enumerate(self.fold_trainers):
                if trainer.early_stop_flag:
                    continue
                
                metrics, ema_metrics = trainer.step()
                
                if metrics is not None:
                    epoch_metrics.append(metrics)
                    
                if ema_metrics is not None:
                    ema_epoch_metrics.append(ema_metrics)
            
            logging.info(f"=" * 50 + f"validation epoch {global_epoch=}" + f"=" * 50)
            metrics, ema_metrics = self._validate_one_epoch_for_all_folds()
            
            avg_train_loss, avg_validate_loss = 0.0, 0.0
            ema_avg_train_loss, ema_avg_validate_loss = 0.0, 0.0
            for epoch_metric in epoch_metrics:
                avg_train_loss += epoch_metric["train_loss"]
                avg_validate_loss += epoch_metric["validate_loss"]
            
            for ema_epoch_metric in ema_epoch_metrics:
                ema_avg_train_loss += ema_epoch_metric["train_loss"]
                ema_avg_validate_loss += ema_epoch_metric["validate_loss"]
                
            # tensorboard
            for idx, epoch_metric in enumerate(epoch_metrics):
                WRITER.add_scalar(f"fold_auc/fold{idx + 1}", epoch_metric["auc"], global_epoch)
                WRITER.add_scalar(f"fold_accuracy/fold{idx + 1}", epoch_metric["accuracy"], global_epoch)
                WRITER.add_scalar(f"fold_train_loss/fold{idx + 1}", epoch_metric["train_loss"], global_epoch)
                WRITER.add_scalar(f"fold_validate_loss/fold{idx + 1}", epoch_metric["validate_loss"], global_epoch)
                
            for idx, ema_epoch_metric in enumerate(ema_epoch_metrics):
                WRITER.add_scalar(f"ema_fold_auc/fold{idx + 1}", ema_epoch_metric["auc"], global_epoch)
                WRITER.add_scalar(f"ema_fold_accuracy/fold{idx + 1}", ema_epoch_metric["accuracy"], global_epoch)
                WRITER.add_scalar(f"ema_fold_train_loss/fold{idx + 1}", ema_epoch_metric["train_loss"], global_epoch)
                WRITER.add_scalar(f"ema_fold_validate_loss/fold{idx + 1}", ema_epoch_metric["validate_loss"], global_epoch)
            
            metrics["train_loss"] = avg_train_loss / len(epoch_metrics)
            metrics["validate_loss"] = avg_validate_loss / len(epoch_metrics)
            
            ema_metrics["train_loss"] = ema_avg_train_loss / len(ema_epoch_metrics)
            ema_metrics["validate_loss"] = ema_avg_validate_loss / len(ema_epoch_metrics)
            
            formatted_dict_str = pprint.pformat(metrics, indent=4, width=120)
            logging.info(f"Epoch {global_epoch}, Original Model: \n{formatted_dict_str}")
            
            ema_formatted_dict_str = pprint.pformat(ema_metrics, indent=4, width=120)
            logging.info(f"Epoch {global_epoch}, EMA Model: \n{ema_formatted_dict_str}")
            
            for key, value in metrics.items():
                if key in ["confusion_matrix", "official_auc", "classification_report", "official_auc", "official_sensitivity", "official_specificity"]:
                    formatted_dict_str = pprint.pformat(metrics[key], indent=2)
                    WRITER.add_text(f"metric/{key}", formatted_dict_str, global_epoch)
                else:
                    WRITER.add_scalar(f"metric/{key}", value, global_epoch)
                    
            for key, value in ema_metrics.items():
                if key in ["confusion_matrix", "official_auc", "classification_report", "official_auc", "official_sensitivity", "official_specificity"]:
                    formatted_dict_str = pprint.pformat(ema_metrics[key], indent=2)
                    WRITER.add_text(f"ema_metric/{key}", formatted_dict_str, global_epoch)
                else:
                    WRITER.add_scalar(f"ema_metric/{key}", value, global_epoch)
            
            if all(trainer.early_stop_flag for trainer in self.fold_trainers):
                logging.info("All folds have early stopped, stop training.")
                break
        
        self._summarize_final_results()
        
        # 2. best model metric
        for idx, trainer in enumerate(self.fold_trainers):
            if trainer.early_stop_flag:
                continue
            
            metrics, ema_metrics = trainer.post_training_validate()
            
        print(f"=" * 50 + f"validation final best model" + f"=" * 50)
        metrics, ema_metrics = self._validate_one_epoch_for_all_folds()
        formatted_dict_str = pprint.pformat(metrics, indent=4, width=120)
        logging.info(f"Epoch {self.config.EPOCHS + 5}, Original Model: \n{formatted_dict_str}")
        
        ema_formatted_dict_str = pprint.pformat(ema_metrics, indent=4, width=120)
        logging.info(f"Epoch {self.config.EPOCHS + 5}, EMA Model: \n{ema_formatted_dict_str}")
        
        for key, value in metrics.items():
            if key in ["confusion_matrix", "official_auc", "classification_report", "official_auc", "official_sensitivity", "official_specificity"]:
                formatted_dict_str = pprint.pformat(metrics[key], indent=2)
                WRITER.add_text(f"metric/{key}", formatted_dict_str, self.config.EPOCHS + 5)
            else:
                WRITER.add_scalar(f"metric/{key}", value, self.config.EPOCHS + 5)
                
        for key, value in ema_metrics.items():
            if key in ["confusion_matrix", "official_auc", "classification_report", "official_auc", "official_sensitivity", "official_specificity"]:
                formatted_dict_str = pprint.pformat(ema_metrics[key], indent=2)
                WRITER.add_text(f"ema_metric/{key}", formatted_dict_str, self.config.EPOCHS + 5)
            else:
                WRITER.add_scalar(f"ema_metric/{key}", value, self.config.EPOCHS + 5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load configuration file")
    parser.add_argument('--config', type=str, required=True, help='config file name, no .py')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--k_fold', type=int, default=-1)
    args = parser.parse_args()

    # set global device
    DEVICE = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    config = load_config(args.config)
    config.EPOCHS = args.epoch
    config.WARM_EPOCHS = args.epoch // 10
    
    # if k_fold is set, use k_fold in args
    if args.k_fold != -1:
        config.K_FOLDS = args.k_fold
    
    set_seed(config.SEED)

    experiment_name = f"{config.EXPERIMENT_NAME}-{config.MODE}-epoch{config.EPOCHS}-k_fold{config.K_FOLDS}-{datetime.today().strftime('%Y%m%d_%H%M%S')}"
    exp_save_root = config.EXPERIMENT_DIR / experiment_name
    exp_save_root.mkdir(parents=True, exist_ok=True)
    
    config_file_path = os.path.join("./configs", f"{args.config}.py")
    shutil.copy(config_file_path, exp_save_root / f"{args.config}.py")

    init_logger(log_path=exp_save_root / "train.log")
    WRITER = SummaryWriter(exp_save_root / "tensorboard")
    pipe = Pipeline(config, exp_save_root)
    pipe.run()
