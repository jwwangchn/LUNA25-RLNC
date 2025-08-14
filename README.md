# 📦 RLNC: A Robust Framework for Lung Nodule Malignancy Classification
Welcome to the official repository for the RLNC (A Robust Framework for Lung Nodule Malignancy Classification) algorithm. This repository contains the complete source code, training guides, and usage instructions.

## ⚙️ Environment Setup
Before you begin, please ensure your environment is set up correctly.

1. **Create a Conda Environment**:

We recommend using Conda to manage project dependencies. Create and activate a new environment with Python 3.10 using the following commands:
```shell
conda create -n rlnc python==3.10
conda activate rlnc
```

2. **Install Dependencies**:

All required dependencies are listed in the requirements.txt file. Run the following command to install them:

```shell
pip install -r requirements.txt
```

3. **Verify Installation**:

Run the following commands to ensure the Python version is correct and all packages have been installed successfully:
```shell
python --version
pip list
```

## 📂 Data Preparation
To train and evaluate the model correctly, please organize your datasets according to the following directory structure. You will need to specify the appropriate data paths in the configuration files.

### Data Directory Structure

1. **LUNA25 Dataset**

```plaintext
data/
└── LUNA25/
    ├── luna25_dataset_csv/
    │   └── LUNA25_Public_Training_Development_Data.csv  <-- Core CSV file
    └── luna25_nodule_blocks/
        ├── image/                                       <-- Nodule image data
        └── metadata/                                    <-- Nodule metadata
```

2. **LIDC Dataset (Converted to LUNA25 format)**

```plaintext
data/
└── LIDC-LUNA25/
    ├── Meta/
    │   └── meta_info_annotated.csv                      <-- Core CSV file
    ├── image/                                           <-- Nodule image data
    └── metadata/                                        <-- Nodule metadata
```

### Configuration File Setup

Before starting the training, you must set the correct data paths and key parameters in the configuration files located in the `configs/` folder (e.g., `rlnc_luna25_full_data.py` and `rlnc_luna25_k_fold_data.py`).

- `self.MODE`: Must be set to `"3D"`.
- `self.MODE_CLASS`: The class used to instantiate the model.
- `self.EXPERIMENT_NAME`: Specify a name for your experiment.
- `self.K_FOLDS`: The number of folds for k-fold cross-validation.
- Data Paths:
    - `self.DATADIR_LUNA25`: The storage path for LUNA25 images (e.g., `data/LUNA25/luna25_nodule_blocks/`).
    - `self.LUNA25_CSV_FP`: The full path to the LUNA25 training CSV file.
    - `self.DATADIR_LIDC`: The storage path for LIDC images (e.g., `data/LIDC-LUNA25/`).
    - `self.LIDC_CSV_FP`: The full path to the LIDC training CSV file.

## 🚀 Model Training
We provide two training modes: training a single model on the full LUNA25 dataset and training 5 separate models using 5-fold cross-validation.

### Mode 1: Train a Single Model (Full Data)

This mode uses all the data defined in `LUNA25_Public_Training_Development_Data.csv` to train one final model.

To run:
```shell
python train_full_data.py --config=rlnc_luna25_full_data
```
Note: The `--config` argument specifies the configuration file name from the `configs/` directory (without the `.py` extension).

### Mode 2: Train K-Fold Cross-Validation Models

This mode splits the dataset into 5 folds (k_fold=5) and trains 5 models sequentially. For each model, 4 folds are used for training and 1 is used for validation.

To run:
```shell
python train_k_fold_data.py --config=rlnc_luna25_k_fold_data
```

## 🤖 Model Inference
You can run inference on new data using the provided `inference.py` script.

To run:
Before executing the script, set the following environment variables to point to your data and resource paths:
```shell
# Set the root directory for your input data
export INPUT_PATH=/path/to/your/demo/input

# Set the input JSON filename containing nodule locations
export INPUT_FILENAME=nodule-locations.json

# Set the output directory for inference results
export OUTPUT_PATH=/path/to/your/demo/output

# Set the path to resources, including model files
export RESOURCE_PATH=/path/to/your/results

# Set the input directory containing chest CT scans
export INPUT_CHEST_CT=/path/to/your/demo/input/images/chest-ct

# Run the inference script
python inference.py
```

## 🐳 Building the Docker Image
To build the Docker container required for submission to Grand-Challenge, run:
```shell
./do_save.sh
```
This will generate a `*.tar.gz` file, which can be uploaded to Grand-Challenge.

More information on testing and deploying your container can be found in the Grand-Challenge Documentation.

## 🛠️ Extending the Baseline

This baseline provides a starting point. Participants are encouraged to:

- Implement more advanced AI models.
- Explore alternative data preprocessing and augmentation techniques.
- Perform ensemble learning.
- Train models using entire or larger CT scan inputs.

For questions, please refer to the LUNA25 Challenge Page. Good luck!

## 🔐 Notes on Model Encryption

- **Why Encryption?**: The decision to open-source this project was made late in the competition process. To protect intellectual property during development, all submitted and utilized model files were encrypted. The processor.py script in this repository is configured by default to load these encrypted models.

- **How to Encrypt Your Own Models**: If you train your own models using this codebase and wish to use them with the provided inference framework, you must encrypt them first.

```shell
python tools/model_encrypt.py
```

Important: Before running this script, you must open `tools/model_encrypt.py` and modify the parameters (e.g., model paths, output paths) to match your specific setup.

## 🙏 Acknowledgements and Citation
We would like to extend our sincere thanks to the LUNA25 Challenge organizing committee for providing the valuable dataset and competition platform.

If you use this codebase in your research, please consider citing our work:

```
@misc{rlnc2024,
  author       = {Your Name/Team Name},
  title        = {RLNC: A Robust Framework for Lung Nodule Malignancy Classification},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/your-repo-link}}
}
```
