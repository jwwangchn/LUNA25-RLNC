# LIDC Preprocessing with Pylidc library
[Medium Link](https://medium.com/@jaeho3690/how-to-start-your-very-first-lung-cancer-detection-project-using-python-part-1-3ab490964aae)

This repository would preprocess the LIDC-IDRI dataset. We use pylidc library to save nodule images into an .npy file format.
The code file structure is as below

```
+-- LIDC-IDRI
|    # This file should contain the original LIDC dataset
+-- LIDC-IDRI_v1
|    # This file should contain the LIDC dataset saved in nii.gz format to process the header information.
+-- data
|    # This file contains the preprocessed data
|   |-- Image
|       +-- LIDC-IDRI-0001_0.npy
|       +-- LIDC-IDRI-0002_0.npy
|       +-- ...
|   |-- Mask
|       +-- LIDC-IDRI-0001_0.npy
|       +-- LIDC-IDRI-0002_0.npy
|       +-- ...
|   |-- Meta
|       +-- meta_info.csv
|   |-- Metadata
|       +-- LIDC-IDRI-0001_0.npy
+-- config_file_create.py
|    # Creates configuration file. You can edit the hyperparameters of the Pylidc library here
+-- prepare_dataset.py
|    # Run this file to preprocess the LIDC-IDRI dicom files. Results would be saved in the data folder
+-- utils.py
     # Utility script

```
## 1.Download LIDC-IDRI dataset
First you would have to download the whole LIDC-IDRI [dataset](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI).
On the website, you will see the Data Acess section. You would need to click Search button to specify the images modality.
I clicked on CT only and downloaded total of 1010 patients.

## 2. Set up pylidc library
You would need to set up the pylidc library for preprocessing. There is an instruction in the [documentation](https://pylidc.github.io/install.html).
Make sure to create the configuration file as stated in the instruction. Right now I am using library version 0.2.1

## 3. Explanation for each python file
```bash
python config_file_create.py
```
This python script contains the configuration setting for the directories. Change the directories settings to where you want to save your output files. Without modification, it will automatically save the preprocessed file in the data folder.
Running this script will create a configuration file 'lung.conf'

This utils.py script contains function to segment the lung. Segmenting the lung and nodule are two different things. Segmenting the lung leaves the lung region only, while segmenting the nodule is finding prosepctive lung nodule regions in the lung. Don't get confused. 

```bash
python prepare_dataset.py
```
This python script will create the image, mask ,Metadata files and save them to the data folder. The script will also create a meta_info.csv file containing information about whether the nodule is
cancerous. In the LIDC Dataset, each nodule is annotated at a maximum of 4 doctors. Each doctors have annotated the malignancy of each nodule in the scale of 1 to 5. 
I have chosen the average label for each nodule as the final malignanc. If the label is 3, we ignore this nodule.  The meta_csv data contains all the information and will be used later in the classification stage.
This prepare_dataset.py looks for the lung.conf file. The configuration file should be in the same directory. Running this script will output .npy files for each patch with a size of 128*128*64



## 4. Data folder
the data folder stores all the output images,masks.
inside the data folder there are 4 subfolders. 

### 1. Image

The Image folder contains the patched nodule .npy folders

### 2. Mask

The Mask folder contains the mask files for the patch.

### 3. Meta

The Meta folder contains the meta_info.csv file.

### 4. Metadata

The Metadata folder contains the header information for the nodule



