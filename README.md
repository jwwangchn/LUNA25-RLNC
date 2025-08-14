# 📦 RLNC: A Robust Framework for Lung Nodule Malignancy Classification

The codes for the RLNC algorithm are available in this repository.

## ⚙️ Setting up the Environment
To set up the required environment for the RLNC algorithm:
1. **Create an environment and esure Python is Installed**: Install Python 3.10 or higher:
    ```bash
    conda create -n rlnc python==3.10
    ```
2. **Install Dependencies**:
    - Run the following command to install the dependencies listed in `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
3. **Verify Installation**:
    - Test the installation by running:
    ```bash
    python --version
    pip list
    ```
    Ensure all required packages are listed and no errors are reported.

## 🚀 Performing a Training Run
1. **Set up training configurations**

Open `rlnc_luna25.py` to edit your training configurations. Key parameters include:

- `self.MODE`: Must be "3D".
- `self.MODE_CLASS`: 实例化模型
- `self.EXPERIMENT_NAME`: Specify the name of your experiment.
- `self.K_FOLDS`: Number of folds for k-fold cross-validation.
- Data paths:
    - `self.DATADIR_LUNA25`: the path where the luna25 images are stored.
    - `self.LUNA25_CSV_FP`: the path to the luna25 training csv file.
    - `self.DATADIR_LIDC`: the path where the lidc images are stored.
    - `self.LIDC_CSV_FP`: the path to the lidc training csv file.



2. **Training the Model**

To train the model using the `train.py` script:
```bash
python train.py --config rlnc_luna25 --device 0 --epoch 50
```
This script uses the settings from experiment_config.py to initialize and train the model.

## 🧪 Testing the Trained Algorithm
1. **Configure the inference script**

Open the `inference.py` script and configure:
- `INPUT_PATH`: Path to the input data (CT, nodule locations and clinical information). Keep as `Path("/input")` for Grand-Challenge.
- `RESOUCE_PATH`: Path to resources (e.g., pretrained models weights) in the container. Defaults to `/results` directory (see Dockerfile)
- `OUTPUT_PATH`: Path to store the output in your local directory. Keep as `Path("/output")` for Grand-Challenge.
- **Inputs for the `run()` function**:
    - `mode`: Match this to the mode used during training (2D or 3D).
    - `model_name`: Specify the experiment_name matching the training configuration (corresponding to experiment_name directory that contains the model weights in `/results`).

2. **Updating the Docker Image Tag**

In `do_test_run.sh`, update the Docker image tag as needed:
```bash
DOCKER_IMAGE_TAG="luna25-baseline-3d-algorithm-open-development-phase"
```


3. **Running the Test Script**

To test the trained model for running inference run: 
```bash
./do_test_run.sh
``` 

This script performs the following:
- Uses Docker to execute the `inference.py` script.
- Mounts necessary input and output directories.
- Adjusts the Docker image tag (if updated) before running.

## 🐳 Building the Docker Image
To build the Docker container required for submission to Grand-Challenge run:
```bash
./do_save.sh
```
This will output a *.tar.gz file, which can be uploaded to Grand-Challenge.
More information on testing and deploying your container can be found [here](https://grand-challenge.org/documentation/test-and-deploy-your-container/).

## 🛠️ Extending the Baseline
While this baseline provides a starting point, participants are encouraged to:

- Implement advanced AI models.
- Explore alternative data preprocessing and augmentation techniques.
- perform Ensemble Learning
- train models using entire or larger CT scan inputs

For questions, refer to the [LUNA25 Challenge Page](https://luna25.grand-challenge.org/).

Good luck!