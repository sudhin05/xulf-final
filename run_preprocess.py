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
    args = parser.parse_args()

    download_zip(args.url_path, args.zip_path)
    unzip_dataset(args.zip_path,args.extract_path)


if __name__ == "__main__":
    main()
