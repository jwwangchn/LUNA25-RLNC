import os
from pathlib import Path
from glob import glob
import json
import random
import torch
import SimpleITK
import numpy as np
from processor import MalignancyProcessor


def set_global_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_global_seed(1024)

INPUT_PATH = Path(os.getenv("INPUT_PATH", "/input"))
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "/output"))
RESOURCE_PATH = Path(os.getenv("RESOURCE_PATH", "/opt/app/resources"))

def transform(input_image, point):
    """
    Convert a point from NumPy (ZYX) to SimpleITK (XYZ) index space,
    apply the transformation, and return the result back in NumPy (ZYX) format.
    """
    sitk_point = list(reversed(point))
    physical_point = input_image.TransformContinuousIndexToPhysicalPoint(sitk_point)
    return np.array(list(reversed(physical_point)))

def itk_image_to_numpy_image(input_image):
    """
    Convert a SimpleITK image to a NumPy array with metadata header.
    """
    numpy_image = SimpleITK.GetArrayFromImage(input_image)
    origin = np.array(list(reversed(input_image.GetOrigin())))
    spacing = np.array(list(reversed(input_image.GetSpacing())))

    # Compute transform matrix
    dims = numpy_image.ndim
    zero_point = np.zeros(dims)
    t_origin = transform(input_image, zero_point)

    basis_vectors = []
    for i in range(dims):
        unit_vector = np.zeros(dims)
        unit_vector[i] = 1
        transformed = transform(input_image, unit_vector)
        basis_vectors.append(transformed - t_origin)

    transform_matrix = np.vstack(basis_vectors) @ np.diag(1 / spacing)

    header = {
        "origin": origin,
        "spacing": spacing,
        "transform": transform_matrix,
    }

    return numpy_image, header

class NoduleProcessor:
    def __init__(self,
                 ct_image_file,
                 nodule_locations,
                 clinical_information,
                 mode="2D",
                 model_folder="LUNA25-baseline-2D"):
        """
        Initialize the nodule processor with CT image, nodule locations, and clinical data.
        """
        self._image_file = ct_image_file
        self.nodule_locations = nodule_locations
        self.clinical_information = clinical_information
        self.mode = mode
        self.model_folder = model_folder

        self.processor = MalignancyProcessor(
            mode=mode,
            suppress_logs=True,
            model_root=RESOURCE_PATH,
            model_folder=model_folder,
        )

    def predict(self, input_image: SimpleITK.Image, coords: np.array) -> list:
        """
        Predict malignancy risk for each nodule in `coords`.
        """
        numpy_image, header = itk_image_to_numpy_image(input_image)
        risks = []

        for coord in coords:
            self.processor.define_inputs(numpy_image, header, [coord])
            malignancy_risk, _ = self.processor.predict()
            risks.append(float(np.array(malignancy_risk).reshape(-1)[0]))

        return risks

    def load_inputs(self):
        """
        Load CT image and nodule coordinates from input files.
        """
        print(f"Reading {self._image_file}")
        image = SimpleITK.ReadImage(str(self._image_file))

        points = self.nodule_locations["points"]
        self.annotationIDs = [p["name"] for p in points]
        coords = np.array([p["point"] for p in points])  # shape: (n, 3), format: XYZ
        self.coords = np.flip(coords, axis=1)  # convert to ZYX

        return image, self.coords, self.annotationIDs

    def process(self):
        """
        Main processing pipeline: load inputs, run predictions, and format output.
        """
        image, coords, annotation_ids = self.load_inputs()
        output = self.predict(image, coords)

        assert len(output) == len(annotation_ids), "Number of outputs must match number of inputs"

        coords = np.flip(coords, axis=1)  # restore to XYZ for output

        results = {
            "name": "Points of interest",
            "type": "Multiple points",
            "points": [],
            "version": {"major": 1, "minor": 0}
        }

        for name, coord, prob in zip(annotation_ids, coords, output):
            results["points"].append({
                "name": name,
                "point": coord.tolist(),
                "probability": prob
            })

        return results


def run(mode="2D",
        model_folder="LUNA25-baseline-2D"):
    """
    Main entry point: load inputs, process, and write output.
    """
    input_nodule_locations = load_json_file(location=INPUT_PATH / os.getenv("INPUT_FILENAME", "nodule-locations.json"))
    input_clinical_information = load_json_file(location=INPUT_PATH / "clinical-information-lung-ct.json")
    input_chest_ct = load_image_path(location=INPUT_PATH / os.getenv("INPUT_CHEST_CT", "images/chest-ct"))

    _show_torch_cuda_info()

    processor = NoduleProcessor(
        ct_image_file=input_chest_ct,
        nodule_locations=input_nodule_locations,
        clinical_information=input_clinical_information,
        mode=mode,
        model_folder=model_folder,
    )

    malignancy_risks = processor.process()

    write_json_file(location=OUTPUT_PATH / "lung-nodule-malginancy-likelihoods.json", content=malignancy_risks)
    print(f"Completed writing output to {OUTPUT_PATH}")
    print(f"Output: {malignancy_risks}")
    return 0


def load_json_file(*, location):
    with open(location, "r") as f:
        return json.load(f)


def write_json_file(*, location, content):
    with open(location, "w") as f:
        json.dump(content, f, indent=4)


def load_image_path(*, location):
    input_files = (
        glob(str(location / "*.tif")) +
        glob(str(location / "*.tiff")) +
        glob(str(location / "*.mha"))
    )
    assert len(input_files) == 1, "Please upload only one .mha file per job"
    return input_files[0]

def _show_torch_cuda_info():
    import torch
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch version: {torch.version.cuda}")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: {torch.cuda.current_device()}")
        print(f"\tproperties: {torch.cuda.get_device_properties(torch.cuda.current_device())}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run(mode="3D",
                         model_folder="weights"))
    