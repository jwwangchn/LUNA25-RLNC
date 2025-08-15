import sys
import os
from pathlib import Path
import glob
from configparser import ConfigParser
import pandas as pd
import numpy as np
import warnings
import pylidc as pl
from tqdm import tqdm
from statistics import median_high
import utils
from utils import is_dir_path,segment_lung
from pylidc.utils import consensus
from PIL import Image
import SimpleITK as sitk
import multiprocessing
from multiprocessing import Pool, Process, Manager
warnings.filterwarnings(action='ignore')

# Read the configuration file generated from config_file_create.py
parser = ConfigParser()
parser.read('lung.conf')

#Get Directory setting
DICOM_DIR = is_dir_path(parser.get('prepare_dataset','LIDC_DICOM_PATH'))
MASK_DIR = is_dir_path(parser.get('prepare_dataset','MASK_PATH'))
IMAGE_DIR = is_dir_path(parser.get('prepare_dataset','IMAGE_PATH'))
CLEAN_DIR_IMAGE = is_dir_path(parser.get('prepare_dataset','CLEAN_PATH_IMAGE'))
CLEAN_DIR_MASK = is_dir_path(parser.get('prepare_dataset','CLEAN_PATH_MASK'))
META_DIR = is_dir_path(parser.get('prepare_dataset','META_PATH'))
METADATA_DIR = is_dir_path(parser.get('prepare_dataset','METADATA_PATH'))

#Hyper Parameter setting for prepare dataset function
mask_threshold = parser.getint('prepare_dataset','Mask_Threshold')

#Hyper Parameter setting for pylidc
confidence_level = parser.getfloat('pylidc','confidence_level')
padding = parser.getint('pylidc','padding_size')

class MakeDataSet:
    def __init__(self, LIDC_Patients_list, IMAGE_DIR, MASK_DIR,CLEAN_DIR_IMAGE,CLEAN_DIR_MASK,METADATA_DIR,META_DIR, mask_threshold, padding, confidence_level=0.5,patch_size: np.array = np.array([128, 128, 64]),):
        self.IDRI_list = LIDC_Patients_list
        self.img_path = IMAGE_DIR
        self.metadata_path=METADATA_DIR
        self.mask_path = MASK_DIR
        self.clean_path_img = CLEAN_DIR_IMAGE
        self.clean_path_mask = CLEAN_DIR_MASK
        self.meta_path = META_DIR
        self.patch_size = np.array(patch_size)
        self.mask_threshold = mask_threshold
        self.c_level = confidence_level
        self.padding = [(padding,padding),(padding,padding),(0,0)]
        self.meta = pd.DataFrame(index=[],columns=['PatientID','SeriesInstanceUID','CoordX','CoordY','CoordZ','NoduleID','label'])
        self.meta_list=Manager().list()

    def calculate_malignancy(self,nodule):
        # Calculate the malignancy of a nodule with the annotations made by 4 doctors. Return median high of the annotated cancer, True or False label for cancer
        # if median high is above 3, we return a label True for cancer
        # if it is below 3, we return a label False for non-cancer
        # if it is 3, we return ambiguous
        list_of_malignancy =[]
        centroids=[]
        for annotation in nodule:
            list_of_malignancy.append(annotation.malignancy)
            centroids.append(annotation.centroid)
        stacked_arrays = np.vstack(centroids)

        # 计算每一列的均值
        centroid = np.mean(stacked_arrays, axis=0)
        malignancy = median_high(list_of_malignancy)
        if  malignancy > 3:
            return centroid,malignancy,True
        elif malignancy < 3:
            return centroid,malignancy, False
        else:
            return centroid,malignancy, 'Ambiguous'
    def save_meta(self):
        """Saves the information of nodule to csv file"""
        meta_lists=list(self.meta_list)
        for meta_list in meta_lists:
            tmp = pd.Series(meta_list,index=['PatientID','SeriesInstanceUID','CoordX','CoordY','CoordZ','NoduleID','label'])
            self.meta = self.meta.append(tmp,ignore_index=True)
        print("Saved Meta data")
        self.meta.to_csv(self.meta_path+'meta_info.csv',index=False)

    def prepare_dataset(self,IDRI_list):
        # This is to name each image and mask

        # Make directory
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        if not os.path.exists(self.mask_path):
            os.makedirs(self.mask_path)
        if not os.path.exists(self.clean_path_img):
            os.makedirs(self.clean_path_img)
        if not os.path.exists(self.clean_path_mask):
            os.makedirs(self.clean_path_mask)
        if not os.path.exists(self.meta_path):
            os.makedirs(self.meta_path)




        for patient in tqdm(IDRI_list):
            pid = patient #LIDC-IDRI-0001~
            nii_dir = Path('/mnt/workspace/workgroup/weining.wjw/02-Codes/LUNA25-AI4Lung/data/LIDC-IDRI-Full/LIDC-IDRI-V1/LIDC-IDRI') / pid
            extent = self.patch_size // 2
            pad_width = [(e, e) for e in extent] 
            # glob .nii.gz files for retain header information
            image_files = list(nii_dir.rglob("*.nii.gz"))
            if not image_files:
                continue
            image=sitk.ReadImage(image_files[0])
            _, header = utils.itk_image_to_numpy_image(image)
            # for retain image information
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            nodules_annotation = scan.cluster_annotations()
            vol = scan.to_volume()
            print("Patient ID: {} Dicom Shape: {} Number of Annotated Nodules: {}".format(pid,vol.shape,len(nodules_annotation)))
            image2=sitk.GetImageFromArray(np.transpose(vol,(2,0,1)))
            if len(nodules_annotation) > 0:
                # Patients with nodules
                for nodule_idx, nodule in enumerate(nodules_annotation):
                    # We calculate the malignancy information
                    patient_image_dir = self.img_path + "/" + pid +f"_{nodule_idx}" +".npy"
                    patient_metadata_dir = self.metadata_path + "/" + pid + f"_{nodule_idx}"+".npy"
                    patient_mask_dir=self.mask_path+ "/" + pid + f"_{nodule_idx}"+".npy"
                    centroid,malignancy, cancer_label = self.calculate_malignancy(nodule)
                    # Only handle cases other than 3
                    if malignancy==3:
                        continue
                    mask_ori=np.zeros_like(vol)
                    mask, cbbox, masks = consensus(nodule,self.c_level,self.padding)
                    mask_ori[cbbox]=mask
                    centroid_phy=image.TransformContinuousIndexToPhysicalPoint(centroid.transpose())
                    pad = False
                    
                    coord = np.array(image.TransformPhysicalPointToIndex(centroid_phy))
                    # extract nodule_patch according the extent
                    upper_limit_breach = np.any(coord - extent < 0)
                    lower_limit_breach = np.any(coord + extent > np.array(image2.GetSize()))
                    if upper_limit_breach or lower_limit_breach:
                        pad = True
                    if pad:
                        image2 = sitk.ConstantPad(
                                        image2,
                                        [int(e) for e in extent],
                                        [int(e) for e in extent],
                                        constant=-1024,
                                    )

                        mask_ori = np.pad(mask_ori,
                                    pad_width=pad_width,
                                            mode='constant',
                                            constant_values=0)
                    #for z,
                    image_patch = image2[
                            int(coord[1] - extent[1]) : int(coord[1] + extent[1]),
                            int(coord[0] - extent[0]) : int(coord[0] + extent[0]),
                            int(coord[2] - extent[2]) : int(coord[2] + extent[2]),
                        ]
                    mask_patch=mask_ori[
                            int(coord[0] - extent[0]) : int(coord[0] + extent[0]),
                            int(coord[1] - extent[1]) : int(coord[1] + extent[1]),
                            int(coord[2] - extent[2]) : int(coord[2] + extent[2]),
                        ]
                    # standardization
                    image_patch=sitk.GetArrayFromImage(image_patch)-1024
                    image_patch[image_patch<-1024]=-1024
                    self.meta_list.append([pid,scan.series_instance_uid,centroid_phy[0],centroid_phy[1],centroid_phy[2],nodule_idx,np.int32(cancer_label)])
                    np.save(patient_image_dir,image_patch)
                    np.save(patient_metadata_dir,header)
                    np.save(patient_mask_dir,np.transpose(mask_patch, (2, 0, 1)))
                    self.save_meta(meta_list)

            else:
                continue
if __name__ == '__main__':
    LIDC_IDRI_list= [f for f in os.listdir(DICOM_DIR) if not f.startswith('.')]
    LIDC_IDRI_list.sort()

    test= MakeDataSet(LIDC_IDRI_list,IMAGE_DIR,MASK_DIR,CLEAN_DIR_IMAGE,CLEAN_DIR_MASK,METADATA_DIR,META_DIR,mask_threshold,padding,confidence_level)
    num_workers=1
    processes = []
    split_paths = [[] for _ in range(num_workers)]
    for i, path in enumerate(LIDC_IDRI_list[:]):
        split_paths[i % (num_workers )].append(path)
    for j in range(num_workers):
        process_id =  j
        process = Process(target=test.prepare_dataset,args=(split_paths[process_id],))
        process.start()
        processes.append(process) 
    for process in processes:
        process.join()   
    test.save_meta()
    
