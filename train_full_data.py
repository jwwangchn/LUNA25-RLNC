import os
import argparse
import importlib
import shutil
import sys
import logging
import json
import numpy as np
import torch
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import warnings
import random
import pandas
from datetime import datetime
from dataloader import get_data_loader

config = None

torch.backends.cudnn.benchmark = True

def init_logger(log_path: str = "logs/train.log"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(levelname)s][%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        filename=log_path,
        filemode="w",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s][%(asctime)s] %(message)s", "%H:%M:%S")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

def load_config(config_name: str):
    sys.path.append(os.path.abspath('.'))
    module_path = f"configs.{config_name}"
    config_module = importlib.import_module(module_path)

    config_ = getattr(config_module, 'config')

    return config_

def make_weights_for_balanced_classes(labels):
    """Making sampling weights for the data samples
    :returns: sampling weights for dealing with class imbalance problem

    """
    n_samples = len(labels)
    unique, cnts = np.unique(labels, return_counts=True)
    cnt_dict = dict(zip(unique, cnts))

    weights = []
    for label in labels:
        if label == 1:
            n_samples_ = n_samples / 1.35
        else:
            n_samples_ = n_samples
        
        weights.append(n_samples_ / float(cnt_dict[label]))
    return weights

def calculate_auc(y_true, y_pred):
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

    ci_lower = np.percentile(bootstrapped_aucs, 2.5)
    ci_upper = np.percentile(bootstrapped_aucs, 97.5)

    return {"auc": auc, "ci_lower": ci_lower, "ci_upper": ci_upper}

def calculate_sensitivity(y_true, y_pred):
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

def calculate_specificity(y_true, y_pred):
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

def train(
    train_csv_path,
    valid_csv_path,
    exp_save_root,
):
    """
    Train a ResNet18 or an I3D model
    """
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    logging.info(f"Training with {train_csv_path}")
    logging.info(f"Validating with {valid_csv_path}")

    train_df = pandas.read_csv(train_csv_path)
    valid_df = pandas.read_csv(valid_csv_path)

    logging.info(
        f"Number of malignant training samples: {train_df.label.sum()}"
    )
    logging.info(
        f"Number of benign training samples: {len(train_df) - train_df.label.sum()}"
    )

    logging.info(valid_df.columns)
    logging.info(
        f"Number of malignant validation samples: {valid_df.label.sum()}"
    )
    
    logging.info(
        f"Number of benign validation samples: {len(valid_df) - valid_df.label.sum()}"
    )

    # create a training data loader
    weights = make_weights_for_balanced_classes(train_df.label.values)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_df))

    train_loader = get_data_loader(
        config.DATADIR_LUNA25,
        train_df,
        mode=config.MODE,
        sampler=sampler,
        workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        rotations=config.ROTATION,
        translations=config.TRANSLATION,
        size_mm=config.SIZE_MM,
        size_px=config.SIZE_PX,
        patch_size=config.PATCH_SIZE
    )

    valid_loader = get_data_loader(
        config.DATADIR_LUNA25,
        valid_df,
        mode=config.MODE,
        workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        rotations=None,
        translations=None,
        size_mm=config.SIZE_MM,
        size_px=config.SIZE_PX,
    )

    device = torch.device("cuda:0")
    model = config.MODE_CLASS.to(device)

    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Initialize lists for storing losses
    train_losses = []
    valid_losses = []

    best_metric = -1
    best_metric_epoch = -1
    epochs = config.EPOCHS
    patience = config.PATIENCE

    config_dict = config.to_dict()

    # Remove unwanted key from dictionary, e.g., 'param_to_remove'
    if "MODE_CLASS" in config_dict:
        config_dict.pop("MODE_CLASS")

    counter = 0
    for epoch in range(epochs):
        if counter > patience:
            logging.info(f"Model not improving for {patience} epochs")
            break

        logging.info("-" * 10)
        logging.info("epoch {}/{}".format(epoch + 1, epochs))

        # train
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in tqdm(train_loader):
            step += 1
            inputs, labels = batch_data["image"], batch_data["label"]
            labels = labels.float().to(device)
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_df) // train_loader.batch_size
            if step % 100 == 0:
                logging.info(
                    "{}/{}, train_loss: {:.4f}".format(step, epoch_len, loss.item())
                )
        epoch_loss /= step
        train_losses.append(epoch_loss)
        logging.info(
            "epoch {} average train loss: {:.4f}".format(epoch + 1, epoch_loss)
        )
        # validate
        model.eval()
        epoch_loss = 0
        step = 0

        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.float32, device=device)
            for val_data in valid_loader:
                step += 1
                val_images, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                val_images = val_images.to(device)
                val_labels = val_labels.float().to(device)
                outputs = model(val_images)
                loss = loss_function(outputs.squeeze(), val_labels.squeeze())
                epoch_loss += loss.item()
                y_pred = torch.cat([y_pred, outputs], dim=0)
                y = torch.cat([y, val_labels], dim=0)

            epoch_len = len(valid_df) // valid_loader.batch_size
            epoch_loss /= step
            valid_losses.append(epoch_loss)
            logging.info(
                "epoch {} average valid loss: {:.4f}".format(epoch + 1, epoch_loss)
            )

            y_pred = torch.sigmoid(y_pred.reshape(-1)).data.cpu().numpy().reshape(-1)
            y = y.data.cpu().numpy().reshape(-1)

            # a. AUC (Area Under ROC Curve)
            fpr, tpr, _ = metrics.roc_curve(y, y_pred)
            auc_metric = metrics.auc(fpr, tpr)
            logging.info(f"ROC-AUC Score: {auc_metric:.4f}")

            # b. Average Precision / PR-AUC
            ap_score = metrics.average_precision_score(y, y_pred)
            logging.info(f"Average Precision (AP) Score: {ap_score:.4f}")

            logging.info("\n--- 2. core metrics ---")
            threshold = 0.5
            y_pred_class = (y_pred >= threshold).astype(int)

            # c. Accuracy
            accuracy = metrics.accuracy_score(y, y_pred_class)
            logging.info(f"Accuracy: {accuracy:.4f}")

            # d. Precision
            precision = metrics.precision_score(y, y_pred_class, zero_division=0)
            logging.info(f"Precision: {precision:.4f}")

            # e. Recall / Sensitivity
            recall = metrics.recall_score(y, y_pred_class, zero_division=0)
            logging.info(f"Recall (Sensitivity): {recall:.4f}")

            # f. F1-Score
            f1 = metrics.f1_score(y, y_pred_class, zero_division=0)
            logging.info(f"F1-Score: {f1:.4f}")

            # g. Specificity
            specificity = metrics.recall_score(y, y_pred_class, pos_label=0)
            logging.info(f"Specificity: {specificity:.4f}")

            logging.info("\n--- 3. reports ---")

            # h. Confusion Matrix
            logging.info("Confusion Matrix:")
            conf_matrix = metrics.confusion_matrix(y, y_pred_class)
            # tn, fp, fn, tp
            logging.info(conf_matrix)
            logging.info(f"(TN: {conf_matrix[0,0]}, FP: {conf_matrix[0,1]})")
            logging.info(f"(FN: {conf_matrix[1,0]}, TP: {conf_matrix[1,1]})")

            # i. Classification Report
            logging.info("\nClassification Report:")
            class_report = metrics.classification_report(y, y_pred_class, target_names=['Class 0', 'Class 1'], zero_division=0)
            logging.info(class_report)
            
            logging.info("\n--- 4. official metrics ---")
            official_auc = calculate_auc(y, y_pred)
            official_sensitivity = calculate_sensitivity(y, y_pred)
            official_specificity = calculate_specificity(y, y_pred)
            
            logging.info(f"AUC: {float(official_auc['auc'])}")
            logging.info(f"AUC 95% CI lower bound: {float(official_auc['ci_lower'])}")
            logging.info(f"AUC 95% CI upper bound: {float(official_auc['ci_upper'])}")
            logging.info(f"Sensitivity: {float(official_sensitivity['sensitivity'])}")
            logging.info(f"Specificity: {float(official_specificity['specificity'])}")
            
            torch.save(
                model.state_dict(),
                exp_save_root / f"epoch_{epoch}.pth",
            )

            if accuracy > best_metric:
                counter = 0
                best_metric = accuracy
                best_metric_epoch = epoch + 1

                torch.save(
                    model.state_dict(),
                    exp_save_root / "best_metric_model.pth",
                )

                metadata = {
                    "train_csv": str(train_csv_path),
                    "valid_csv": str(valid_csv_path),
                    "best_auc": best_metric,
                    "epoch": best_metric_epoch,
                    "config": config_dict,
                }
                np.save(
                    exp_save_root / "results.npy",
                    metadata,
                )
                
                with open(exp_save_root / "results.json", "w") as f:
                    json.dump(metadata, f, indent=4, ensure_ascii=False)

                logging.info("saved new best metric model")

            logging.info(f"current epoch: {epoch + 1} current Accuracy: {accuracy}, AUC: {auc_metric}, best Accuracy (metric): {best_metric} at epoch {best_metric_epoch}")
        counter += 1

    logging.info(
        "train completed, best_metric: {:.4f} at epoch: {}".format(
            best_metric, best_metric_epoch
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load configuration file")
    parser.add_argument('--config', type=str, required=True, help='config file name, no .py')
    args = parser.parse_args()

    config = load_config(config_name=args.config)
    
    experiment_name = f"{config.EXPERIMENT_NAME}-{config.MODE}-epoch{config.EPOCHS}-{datetime.today().strftime('%Y%m%d_%H%M%S')}"
    exp_save_root = config.EXPERIMENT_DIR / experiment_name
    exp_save_root.mkdir(parents=True, exist_ok=True)
    
    init_logger(log_path=exp_save_root / "train.log")
    
    config_file_path = os.path.join("./configs", f"{args.config}.py")
    shutil.copy(config_file_path, exp_save_root / f"{args.config}.py")
    
    logging.info(f"{experiment_name=}, {exp_save_root=}")
    
    train(
        train_csv_path=config.LUNA25_CSV_FP,
        valid_csv_path=config.LUNA25_CSV_FP,
        exp_save_root=exp_save_root)