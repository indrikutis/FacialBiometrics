from __future__ import print_function, division
import warnings
warnings.filterwarnings("ignore")
import os.path
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import  transforms
import dlib
import os


def extract_relative_paths(imgs, dataset_name):
    """Extracts the relative paths of images from the dataset folder

    Args:
        imgs (list): List of absolute paths to images
        dataset_name (str): The name of the dataset

    Returns:
        list: List of relative paths of images from the dataset folder
    """

    relative_paths = []

    for img_path in imgs:
        rel_path = os.path.relpath(img_path, dataset_name)
        
        # Replace backslashes with forward slashes (for compatibility on Windows)
        rel_path = rel_path.replace("\\", "/")

        relative_paths.append(rel_path)

    return relative_paths

# NOTE: Functions taken from Fairface github

def detect_face(image_paths,  SAVE_DETECTED_AT, default_max_size=800,size = 300, padding = 0.25):
    cnn_face_detector = dlib.cnn_face_detection_model_v1('dlib_models/mmod_human_face_detector.dat')
    sp = dlib.shape_predictor('dlib_models/shape_predictor_5_face_landmarks.dat')
    base = 2000  # largest width and height
    for index, image_path in enumerate(image_paths):
        if index % 10 == 0:
            print('---%d/%d---' %(index, len(image_paths)))
        img = dlib.load_rgb_image(image_path)

        print("loading: ", image_path)

        old_height, old_width, _ = img.shape

        if old_width > old_height:
            new_width, new_height = default_max_size, int(default_max_size * old_height / old_width)
        else:
            new_width, new_height =  int(default_max_size * old_width / old_height), default_max_size
        img = dlib.resize_image(img, rows=new_height, cols=new_width)

        dets = cnn_face_detector(img, 1)
        num_faces = len(dets)
        if num_faces == 0:
            print("Sorry, there were no faces found in '{}'".format(image_path))
            continue

        # Find the 5 face landmarks we need to do the alignment.
        faces = dlib.full_object_detections()
        for detection in dets:
            rect = detection.rect
            faces.append(sp(img, rect))
        images = dlib.get_face_chips(img, faces, size=size, padding = padding)
        for idx, image in enumerate(images):
            img_name = os.path.basename(image_path)  # Extracting filename from the path
            base_name, ext = os.path.splitext(img_name)
            face_name = os.path.join(SAVE_DETECTED_AT, base_name + "_face" + str(idx) + ext)

            dlib.save_image(image, face_name)

def predidct_age_gender_race(save_prediction_at, imgs, image_sampling_rate, dataset_name, imgs_path = 'cropped_faces/'):

    sampled_images = imgs.sample(frac=image_sampling_rate, random_state=42)
    imgs = sampled_images

    img_names = sampled_images.values
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_fair_7 = torchvision.models.resnet34(pretrained=True)
    model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
    model_fair_7.load_state_dict(torch.load("res34_models/res34_fair_align_multi_7_20190809.pt", map_location=device))   #('fair_face_models/fairface_alldata_20191111.pt'))
    model_fair_7 = model_fair_7.to(device)
    model_fair_7.eval()

    model_fair_4 = torchvision.models.resnet34(pretrained=True)
    model_fair_4.fc = nn.Linear(model_fair_4.fc.in_features, 18)
    model_fair_4.load_state_dict(torch.load("res34_models/res34_fair_align_multi_4_20190809.pt", map_location=device))   #('fair_face_models/fairface_alldata_4race_20191111.pt'))
    model_fair_4 = model_fair_4.to(device)
    model_fair_4.eval()

    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # img pth of face images
    face_names = []
    original_img_names = []
    original_img_full_paths = []
    # list within a list. Each sublist contains scores for all races. Take max for predicted race
    race_scores_fair = []
    gender_scores_fair = []
    age_scores_fair = []
    race_preds_fair_7 = []
    gender_preds_fair = []
    age_preds_fair = []
    race_scores_fair_4 = []
    race_preds_fair_4 = []

    for index, img_name in enumerate(img_names):
        if index % 1000 == 0:
            print("Predicting... {}/{}".format(index, len(img_names)))

        face_names.append(img_name)
        img_base_name = os.path.basename(img_name)
        original_img_names.append(img_base_name)

        matching_paths = [path for path in imgs.values if dataset_name in path and img_base_name in path]

        cropped_paths = []
        for path in matching_paths:
            # Find the starting position of the dataset name in the path
            dataset_index = path.index(dataset_name)
            cropped_path = path[dataset_index:]
            cropped_paths.append(cropped_path)

        cropped_paths_str = ",".join(cropped_paths)
        cropped_paths_str = cropped_paths_str.replace("\\\\", "\\")
        original_img_full_paths.append(cropped_paths_str)
        
        image = dlib.load_rgb_image(img_name)
        image = trans(image)
        image = image.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
        image = image.to(device)

        # fair
        outputs = model_fair_7(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        race_outputs = outputs[:7]
        gender_outputs = outputs[7:9]
        age_outputs = outputs[9:18]

        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
        age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

        race_pred = np.argmax(race_score)
        gender_pred = np.argmax(gender_score)
        age_pred = np.argmax(age_score)

        race_scores_fair.append(race_score)
        gender_scores_fair.append(gender_score)
        age_scores_fair.append(age_score)

        race_preds_fair_7.append(race_pred)
        gender_preds_fair.append(gender_pred)
        age_preds_fair.append(age_pred)

        # fair 4 class
        outputs = model_fair_4(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        race_outputs = outputs[:4]
        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        race_pred = np.argmax(race_score)

        race_scores_fair_4.append(race_score)
        race_preds_fair_4.append(race_pred)

    # Extract relative paths of images from dataset folder
    extracted_path = ''
    for img_path in imgs:
        if dataset_name in img_path and img_name in img_path:
            extracted_path = img_path.split(dataset_name)[1].split(img_name)[0] + img_name
            print(extracted_path)

    result = pd.DataFrame({
        'path': original_img_full_paths,
        'dataset_name': dataset_name,
        'image_name': original_img_names,
        'face_name_align': face_names,
        'race_preds_fair_7': race_preds_fair_7,
        'race_preds_fair_4': race_preds_fair_4,
        'gender_preds_fair': gender_preds_fair,
        'age_preds_fair': age_preds_fair,
        'race_scores_fair_7': race_scores_fair,
        'race_scores_fair_4': race_scores_fair_4,
        'gender_scores_fair': gender_scores_fair,
        'age_scores_fair': age_scores_fair
    })
    
    result.columns = ['path',
                      'dataset_name',
                      'image_name',
                      'face_name_align',
                      'race_preds_fair_7',
                      'race_preds_fair_4',
                      'gender_preds_fair',
                      'age_preds_fair',
                      'race_scores_fair_7',
                      'race_scores_fair_4',
                      'gender_scores_fair',
                      'age_scores_fair']
    
    # Check if 'race' column exists, create if not
    if 'race7' not in result.columns:
        result['race7'] = ''

    result.loc[result['race_preds_fair_7'] == 0, 'race7'] = 'White'
    result.loc[result['race_preds_fair_7'] == 1, 'race7'] = 'Black'
    result.loc[result['race_preds_fair_7'] == 2, 'race7'] = 'Latino_Hispanic'
    result.loc[result['race_preds_fair_7'] == 3, 'race7'] = 'East Asian'
    result.loc[result['race_preds_fair_7'] == 4, 'race7'] = 'Southeast Asian'
    result.loc[result['race_preds_fair_7'] == 5, 'race7'] = 'Indian'
    result.loc[result['race_preds_fair_7'] == 6, 'race7'] = 'Middle Eastern'

    # race fair 4
    if 'race4' not in result.columns:
        result['race4'] = ''

    result.loc[result['race_preds_fair_4'] == 0, 'race4'] = 'White'
    result.loc[result['race_preds_fair_4'] == 1, 'race4'] = 'Black'
    result.loc[result['race_preds_fair_4'] == 2, 'race4'] = 'Asian'
    result.loc[result['race_preds_fair_4'] == 3, 'race4'] = 'Indian'

    # gender
    if 'gender' not in result.columns:
        result['gender'] = ''

    result.loc[result['gender_preds_fair'] == 0, 'gender'] = 'Male'
    result.loc[result['gender_preds_fair'] == 1, 'gender'] = 'Female'

    # age
    if 'age_group' not in result.columns:
        result['age_group'] = ''

    result.loc[result['age_preds_fair'] == 0, 'age_group'] = '0-2'
    result.loc[result['age_preds_fair'] == 1, 'age_group'] = '3-9'
    result.loc[result['age_preds_fair'] == 2, 'age_group'] = '10-19'
    result.loc[result['age_preds_fair'] == 3, 'age_group'] = '20-29'
    result.loc[result['age_preds_fair'] == 4, 'age_group'] = '30-39'
    result.loc[result['age_preds_fair'] == 5, 'age_group'] = '40-49'
    result.loc[result['age_preds_fair'] == 6, 'age_group'] = '50-59'
    result.loc[result['age_preds_fair'] == 7, 'age_group'] = '60-69'
    result.loc[result['age_preds_fair'] == 8, 'age_group'] = '70+'

    #print(result.columns)

    result[['path','dataset_name','image_name','face_name_align',
            'race7', 'race4',
            'gender', 'age_group',
            'race_scores_fair_7', 'race_scores_fair_4',
            'gender_scores_fair', 'age_scores_fair']].to_excel(save_prediction_at, index=False)

    print("saved results at ", save_prediction_at)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":

    #Please create a csv with one column 'Image Paths', contains the full paths of all images to be analyzed.
    #Also please change working directory to this file.
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--csv', dest='input_csv', action='store',
    #                     help='csv file of image path where col name for image path is "Image Paths')
    dlib.DLIB_USE_CUDA = True
    print("using CUDA?: %s" % dlib.DLIB_USE_CUDA)
    # args = parser.parse_args()
    SAVE_DETECTED_AT = "detected_faces"
    ensure_dir(SAVE_DETECTED_AT)

    input_csv = "File_paths/lfw_all_subset_photos.csv"
    DATASET_NAME = "lfw_all_subset_photos"
    output_filename = "FairFace_analysis_results_1_lfw_all_subset_photos.xlsx"
    image_sampling_rate = 1

    imgs = pd.read_csv(input_csv)['Image Paths']
    detect_face(imgs, SAVE_DETECTED_AT)
    print("detected faces are saved at ", SAVE_DETECTED_AT)

    results_folder = "Tool_results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    predidct_age_gender_race(results_folder + "/" + output_filename, imgs, image_sampling_rate, DATASET_NAME, SAVE_DETECTED_AT)