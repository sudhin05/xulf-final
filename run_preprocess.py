"""
Docstring for run_preprocess:

Workflow:

first step: load the dataset from the specified path (Unzipping)
second step: preprocess the dataset (Resizing, cropping)
third step: save the preprocessed dataset as img/1_ohwx_man/img0001.png and so on

"""
import zipfile
import urllib.request
import os
import logging
import argparse

from setupCodes.app_no_gradio import crop_images, resize_images

logging.basicConfig(filename="preprocess.log",filemode='w',level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def download_zip(url, zip_path):
    if not os.path.exists(os.path.dirname(zip_path)):
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)

    urllib.request.urlretrieve(url, zip_path)
    logging.info("Downloaded zip file")

def unzip_dataset(zip_path, extract_path):
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    logging.info("Extracted zip file")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str)
    parser.add_argument("--zip_path", type=str, default="Dataset/dataset.zip")
    parser.add_argument("--extract_path", type=str, default="Dataset/Imgs")
    parser.add_argument("--model_dir", type=str, default="models")
    args = parser.parse_args()

    download_zip(args.url, args.zip_path)
    unzip_dataset(args.zip_path,args.extract_path)

    #Cropping
    crop_images(
        input_folder=args.extract_path,
        output_folder="Dataset/1024x1024",
        aspect_ratios=["1024x1024"],
        yolo_folder=None,
        save_yolo=False,
        batch_size=4,
        gpu_ids="0",
        overwrite=True,
        selected_class="person",
        save_as_png=False,
        sam2_prompt=False,
        debug_mode=False,
        skip_no_detection=False,
        padding_value=0,
        padding_unit="percent",
        model_dir="model"
    )


if __name__ == "__main__":
    main()



# python app_no_gradio.py crop \
#   --input  /path/to/input_images \   # HERE THE UNZIPPED DATASET WOULD BNE GOING
#   --output /path/to/crops \           # WE COULD EITHER JUST OVERWRITE ON THE UNZIPPED PATH, OR CREATE A NEW FOLDER WHICHEVER WORKS
#   --aspect-ratios "1024x1024" \
#   --batch-size 4 \
#   --gpu-ids "0" \
#   --class person \
#   --overwrite

# python app_no_gradio.py resize \
#   --input  /path/to/crops_root \  #CROPPED FOLDER
#   --output /path/to/resized_root \  #FOLDER WHICH WOULD BE THEN GOING AS INPUT TO OUR MODEL, LOOK AT NOTE 1
#   --resolutions "1024x1024" \
#   --threads 8 \
#   --overwrite