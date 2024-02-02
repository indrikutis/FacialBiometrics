# FacialBiometrics #

## Installation ##

### Clone the Repository ###
```bash
git clone https://github.com/indrikutis/FacialBiometrics.git
cd FacialBiometrics
```

### Setup Virtual Environment (Optional but Recommended) ###

```bash
python -m venv venv
venv\Scripts\activate
source venv/bin/activate
```

### Install Dependencies ###

```bash
pip install -r requirements.txt
```

## Project tool  frameworks ##

### DeepFace: ###

GitHub link to the repository: 
```bash
https://github.com/serengil/deepface
```

#### Input: ####
1. img_file_paths: "File_paths/file_paths.csv"
2. output_filename = "DeepFace_analysis_results.xlsx"
3. dataset_name = 'subjects_0-1999_72_imgs' - used for the results sheet to indicate the dataset
4. image_sampling_rate = 0.7

#### Attributes: ####
1. Age
2. Gender: Male, Female
3. Race: asian, white, middle eastern, indian, latino and black
4. Emotion : angry, fear, neutral, sad, disgust, happy and surprise

#### Backend models: ####
1. opencv
2. ssd
3. dlib
4. mtcnn
5. retinaface
6. mediapipe
7. yolov8
8. yunet
9. fastmtcnn

### FairFace ###

GitHub link to the repository: 
```bash
https://github.com/dchen236/FairFace
```

The model extracts aligned faces from the image and saves to detected_faces folder. 

#### Input ####
1. img_file_paths: "File_paths/file_paths.csv"
2. output_filename = "DeepFace_analysis_results.xlsx"
3. dataset_name = 'subjects_0-1999_72_imgs' - used for the results sheet to indicate the dataset
4. image_sampling_rate = 0.7

#### Attributes: ####
1. Age group: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+
2. Gender: Male, Female
3. Race7: White, Black, Latino_Hispanic, East, Southeast Asian, Indian, Middle Eastern
4. Race4: White, Black, Asian and Indian

#### Notes ####
The models and scripts were tested on a device with 8Gb GPU, it takes under 2 seconds to predict the 5 images in the test folder.


## Project datasets ##


### UTKFace ###

Face attribute extration dataset. 

Attributes are stored in the file name: [age]_ [gender]_ [race]_[date&time].jpg

Attributes:
1. Age: 0 to 116
2. Gender: 0 (male) or 1 (female)
3. Race: integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
4. Date and time:m in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace
5. Single image per person

### FGNET ###

Age estimation, face recognition and verification dataset.

Attributes:
1. Age (with the age gap up to 45 years)
2. Several images per person


### LFW ###

Face recognition and verification dataset.

Attributes:
1. Several images per person


## Running the tool ##

### Generating file paths ###

The script takes a directory path and generates a .csv file with one column 'Image Paths', containing the full paths of all images found within the directory to be analyzed.

- **Results Folder**: The folder where the prepared dataset and associated files will be stored. Example: `"File_paths"`
- **Root Directory Path**: The root directory path containing the raw images of the dataset. Example: `"C:/path/to/dataset"`
- **CSV Filename**: The name of the CSV file where the dataset information will be saved. Example: `"File_paths/lfw_all_subset_photos.csv"`

```bash
python "Tool scripts\Generate_file_paths.py"
```

### Deepface framework ###

Deepface framework to extract the facial attributes.

- **Image File Paths**: The file containing the paths to the images in the dataset. Example: `"File_paths/lfw_all_subset_photos.csv"`
- **Output Filename**: The name of the Excel file where the analysis results will be saved. Example: `"DeepFace_analysis_results_lfw_all_subset_photos_1_sampling.xlsx"`
- **Dataset Name**: A unique identifier for the dataset being analyzed. Example: `"lfw_all_subset_photos"`
- **Image Sampling Rate**: The rate at which images are sampled during the analysis. Example: `0.3`

```bash
python "Tool scripts\Deepface.py"
```

### Fairface framework ###

Fairface framework to extract the facial attributes.

- **Input CSV**: The CSV file containing information about the dataset. Example: `"File_paths/lfw_all_subset_photos.csv"`
- **Dataset Name**: A unique identifier for the dataset being analyzed. Example: `"lfw_all_subset_photos"`
- **Output Filename**: The name of the Excel file where the analysis results will be saved. Example: `"FairFace_analysis_results_1_lfw_all_subset_photos.xlsx"`
- **Image Sampling Rate**: The rate at which images are sampled during the analysis. Example: `1`


```bash
python "Tool scripts\Fairface.py"
```

### Generate dataset info for tool analysis ###

#### UTKFace ####

Generates UTKFace dataset info file with image_name, age, gender, race.

- **Dataset Path**: The path to the UTKFace dataset. Example: `"C:/path/to/UTKFace_dataset"`
- **Output Excel Path**: The path and filename for the Excel file where the dataset information will be saved. Example: `"Dataset_info/UTKFace_dataset_info.xlsx"`

```bash
python "Tool scripts\UTKFace.py"
```

#### FGNET ####

Generates UTKFace dataset info file with image_name, age.

- **Dataset Path**: The path to the FGNET dataset. Example: `"C:/path/to/FGNET_dataset"`
- **Output Excel Path (Dataset Info)**: The path and filename for the Excel file where the dataset information will be saved. Example: `"Dataset_info/FGNET_dataset_info.xlsx"`
- **Output CSV Path (Image Pairs)**: The path and filename for the CSV file containing image pairs. Example: `"FGNET_image_pairs.csv"`


```bash
python "Tool scripts\FGNET.py"
```


### Tool analysis ###

Merges the results of the FairFace analysis and the UTKFace dataset information in order to output the age, gender, race accuracies and demographic distributions. NOTE: FGNET dataset does not have model predictions by race since it only encodes the age

- **FairFace Analysis Results File Path**: The path to the FairFace analysis results Excel file. Example: `"Tool_results/FairFace_analysis_results_UTKFace_all.xlsx"`
- **UTKFace Dataset Info File Path**: The path to the UTKFace dataset information Excel file. Example: `"Dataset_info/UTKFace_dataset_info.xlsx"`
- **Merged Results File Name**: The name of the Excel file where the merged results will be saved. Example: `"UTKFace_FairFace_merged_results_1.xlsx"`
- **Age Range for Analysis**: The specified age range for the analysis. Example: `5`
- **Analysis Tool Used**: The tool used for the analysis (e.g., 'FairFace', 'DeepFace'). Example: `'FairFace'`



## Face verification ##

### Generating file pairs ###

Face verification and recognition requires generating image pairs. All the generates pairs are of the true correct pairs by matching every image with every other image from the same subsets.

- **Root Folder**: The root folder containing the LFW Subset dataset. Example: `"C:/path/to/lfw_dataset"`
- **Folder Name for Output**: The folder where the generated CSV file will be stored. Example: `"File_paths"`
- **CSV File Path**: The path and filename for the CSV file containing image pairs. Example: `"image_pairs_lfw_subset.csv"`


Deepface framework performs face recognition verification on pairs of images from the LFW and FGNET datasets and saves the results in an Excel file.


- **Root Folder for Results**: The root folder where the results will be stored. Example: `"Face_recognition_results/"`
- **CSV File Path for Image Pairs**: The path to the CSV file containing pairs of images. Example: `"File_paths/image_pairs_lfw_subset.csv"`
- **Sampling Rate**: The rate at which image pairs are sampled during the verification. Example: `1`
- **Excel File Path for Results**: The path and filename for the Excel file where the verification results will be saved. Example: `"Face_recognition_results/verification_results_lfw_subset_1_sampling.csv"`

```bash
python Face_recognition_scripts/Generate_image_pairs.py
```

## Face recognition ##

Face recognition is performed on the set of image pairs generated in the step before. 

- **Source Path**: The path to the original LFW dataset. Example: `C:/path/to/lfw_dataset`
- **Destination Path**: The path where the image subset will be stored. Example: `C:/path/to/lfw_dataset_subset/`
- **Maximum Total Images to Copy**: The maximum total number of images to copy to the subset. Example: `500`
- **Database Path for Face Recognition**: The path to the database for face recognition. Example: `C:/path/to/lfw_all_subset_photos`
- **Model Name for Face Recognition**: The name of the face recognition model to be used. Example: `VGG-Face`
- **Output CSV Path for Face Recognition Results**: The path and filename for the CSV file where face recognition results will be saved. Example: `Face_recognition_results/face_recognition_results_FGNET.csv`
- **Accuracy CSV Path**: The path and filename for the CSV file where face recognition accuracy will be saved. Example: `Face_recognition_results/face_recognition_accuracy_lfw_th_0.7_subset.csv`


```bash
python Face_recognition_scripts/Face_recognition.py
```

## Bias analysis ##


- **Attribute File Path**: The path to the attribute analysis data file. Example: `"Tool_results/FairFace_analysis_results_1_lfw_all_subset_photos.xlsx"`
- **Verification Results Path**: The path to the CSV file containing face recognition verification results. Example: `"Face_recognition_results/verification_results_lfw_subset_1_sampling.csv"`
- **Merged Dataframe Path**: The path and filename for the CSV file where the merged results will be saved. Example: `"Bias_analysis_results/Fairface_bias_analysis_results_1_lfw_all_subset_photos.csv"`

```bash
python Bias_assessment.py
```



