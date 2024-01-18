# import sys
# import torch
# import facer

# device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"

# # image: 1 x 3 x h x w
# image = facer.hwc2bchw(facer.read_hwc("C:/INDRES/DTU/Semester 3/Special course/Datasets/subjects_0-1999_72_imgs/0/0.png")).to(device=device)

# face_detector = facer.face_detector("retinaface/mobilenet", device=device)
# with torch.inference_mode():
#     faces = face_detector(image)

# face_attr = facer.face_attr("farl/celeba/224", device=device)
# with torch.inference_mode():
#     faces = face_attr(image, faces)

# labels = face_attr.labels
# face1_attrs = faces["attrs"][0] # get the first face's attributes

# print(labels)

# for prob, label in zip(face1_attrs, labels):
#     if prob > 0.5:
#         print(label, prob.item())

import os
import torch
import facer
import pandas as pd

def analyze_and_save_to_excel(dataset_root_folder):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_list = []

    # Iterate through all datasets in the root folder
    for dataset_folder in os.listdir(dataset_root_folder):
        dataset_path = os.path.join(dataset_root_folder, dataset_folder)

        if not os.path.isdir(dataset_path):
            continue

        # Iterate through all images in the dataset
        for image_name in os.listdir(dataset_path):
            image_path = os.path.join(dataset_path, image_name)

            if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            try:
                image = facer.hwc2bchw(facer.read_hwc(image_path)).to(device=device)

                # Face detection
                face_detector = facer.face_detector("retinaface/mobilenet", device=device)
                with torch.inference_mode():
                    faces = face_detector(image)

                # Face attributes analysis
                face_attr = facer.face_attr("farl/celeba/224", device=device)
                with torch.inference_mode():
                    faces = face_attr(image, faces)

                # Extract relevant information
                labels = face_attr.labels
                face1_attrs = faces["attrs"][0]  # get the first face's attributes

                results_list.append({
                    'Dataset': dataset_folder,
                    'Image': image_name,
                    'FaceAttributes': {label: prob.item() for prob, label in zip(face1_attrs, labels)},
                    'Notes': ""
                })

            except Exception as e:
                results_list.append({
                    'Dataset': dataset_folder,
                    'Image': image_name,
                    'FaceAttributes': {},
                    'Notes': str(e)
                })

                print(f"Error analyzing {image_path}: {e}")
                continue

    results_df = pd.DataFrame(results_list)

    excel_filename = "Facer_analysis_results.xlsx"
    results_df.to_excel(excel_filename, index=False)
    print(f"Results saved to {excel_filename}")

dataset_root_folder_facer = "C:/INDRES/DTU/Semester 3/Special course/Datasets/Test/"

analyze_and_save_to_excel(dataset_root_folder_facer)
