# FacialBiometrics #

## Installation ##


## Usage ##


## Generating file paths: ##

The script takes a directory path and generates a .csv file with one column 'Image Paths', containing the full paths of all images found within the directory to be analyzed.

Input: directory path


## DeepFace: ##

GitHub link to the repository: https://github.com/serengil/deepface

### Input: ### 
1. img_file_paths: "File_paths/file_paths.csv"
2. output_filename = "DeepFace_analysis_results.xlsx"
3. dataset_name = 'subjects_0-1999_72_imgs' - used for the results sheet to indicate the dataset
4. image_sampling_rate = 0.7

### Attributes: ###
1. Age
2. Gender: Male, Female
3. Race: asian, white, middle eastern, indian, latino and black
4. Emotion : angry, fear, neutral, sad, disgust, happy and surprise

### Backend models: ###
1. opencv
2. ssd
3. dlib
4. mtcnn
5. retinaface
6. mediapipe
7. yolov8
8. yunet
9. fastmtcnn

## FairFace:

GitHub link to the repository: https://github.com/dchen236/FairFace

The model extracts aligned faces from the image and saves to detected_faces folder. 

### Input ###
1. img_file_paths: "File_paths/file_paths.csv"
2. output_filename = "DeepFace_analysis_results.xlsx"
3. dataset_name = 'subjects_0-1999_72_imgs' - used for the results sheet to indicate the dataset
4. image_sampling_rate = 0.7

### Attributes: ###
1. Age group: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+
2. Gender: Male, Female
3. Race7: White, Black, Latino_Hispanic, East, Southeast Asian, Indian, Middle Eastern
4. Race4: White, Black, Asian and Indian

### Notes ###
The models and scripts were tested on a device with 8Gb GPU, it takes under 2 seconds to predict the 5 images in the test folder.

## Facer:

Attributes provided by the Facer model have been found to not be useful for the further research. 

## Datasets ##

### FER-2013: test emotion classification. ###

Attributes: 
1. Angry
2. Disgust
3. Fear
4. Happy
5. Sad
6. Surprise
7. Neutral attribute annotations

### UTKFace ###

Attributes are stored in the file name: [age]_ [gender]_ [race]_[date&time].jpg

Attributes:
1. Age: 0 to 116
2. Gender: 0 (male) or 1 (female)
3. Race: integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
4. Date and time:m in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace