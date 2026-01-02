"""
run_preprocess.py

To Dos:
0) Clone Repo, cd into it, run this script with necessary args
1) Download + unzip dataset
2) Find the folder that actually contains images (handles "Akanksha/" nested inside the zip)
3) Crop to the requested aspect ratio(s) using newCodes.app_no_gradio.crop_images
   IMPORTANT: crop_images is a GENERATOR 
4) Resize the cropped Images using newCodes.app_no_gradio.resize_images (also a GENERATOR)

5) Resolve toml file
6) Launch kohya training using the resolved toml file
"""

import argparse
import logging
import os
import zipfile
import urllib.request
import subprocess
from pathlib import Path
import sys

from newCodes.app_final import crop_images, resize_images

logging.basicConfig(
    filename="preprocess.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def download_zip(url: str, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(zip_path))
    logging.info("Downloaded zip file to %s", zip_path)


def unzip_dataset(zip_path: Path, extract_path: Path) -> None:
    extract_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), "r") as zip_ref:
        zip_ref.extractall(str(extract_path))
    logging.info("Extracted zip file to %s", extract_path)


def pick_image_dir(root: Path) -> Path:
    """
    If root directly contains images: return root, If root contains exactly one subdir: recurse into it (common zip layout)
    Otherwise return root (for caller to handle)
    """
    if not root.exists():
        return root

    files = [p for p in root.iterdir() if p.is_file()]
    if any(p.suffix.lower() in IMG_EXTS for p in files):
        return root

    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        return pick_image_dir(subdirs[0])

    return root


def consume_generator(gen) -> None:
    for msg in gen:
        logging.info("%s", msg)

def run(cmd, check=True):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=check)

def main() -> None:
    parser = argparse.ArgumentParser()

    #Important arguments that will be given either in command or in azure input
        #Img Proc args
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--dest_root", type=str, default="Dataset/final-imgs", help='Dest Folder with concept)')
    parser.add_argument("--concept", default="1_ohwx_woman", help="Kohya concept folder name")

        #Kohya args
    parser.add_argument("--flux_dir", required=True, help="Azure input: flux weights directory")
    parser.add_argument("--saves_dir", required=True, help="Azure output: model save directory")
    parser.add_argument("--config", default="setupCodes/runConfig.toml", help="Base TOML config")


    #These arguments can be left as default
    parser.add_argument("--zip_path", type=str, default="Dataset/dataset.zip")
    parser.add_argument("--extract_path", type=str, default="Dataset/temp-imgs")
    parser.add_argument("--img_output_root", type=str, default="Dataset", help='Root output folder (e.g. "Dataset")')
    

    parser.add_argument("--aspect_ratios", type=str, default="1024x1024", help='e.g. "1024x1024" or "3x4,4x5"')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--selected_class", type=str, default="person")
    parser.add_argument("--overwrite", action="store_true", default=True)

    args = parser.parse_args()

    base = Path(__file__).resolve().parent

    zip_path = Path(args.zip_path)
    if not zip_path.is_absolute():
        zip_path = base / zip_path

    extract_path = Path(args.extract_path)
    if not extract_path.is_absolute():
        extract_path = base / extract_path

    img_output_root = Path(args.img_output_root)
    if not img_output_root.is_absolute():
        img_output_root = base / img_output_root

    dest_root = Path(args.dest_root)
    if not dest_root.is_absolute():
        dest_root = base / dest_root

    dest_path = dest_root / args.concept
    dest_path.mkdir(parents=True, exist_ok=True)

    #Below maybe needs checking for azure
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = base / model_dir

    logging.info("zip_path     =", zip_path)
    logging.info("extract_path =", extract_path)
    logging.info("img_output_root  =", img_output_root)
    logging.info("model_dir    =", model_dir)

    download_zip(args.url, zip_path)
    unzip_dataset(zip_path, extract_path)

    input_folder = pick_image_dir(extract_path)
    logging.info("input_folder =", input_folder)

    gen = crop_images(
        input_folder=str(input_folder),
        output_folder=str(img_output_root),
        aspect_ratios=str(args.aspect_ratios),
        yolo_folder=None,
        save_yolo=False,
        batch_size=int(args.batch_size),
        gpu_ids=str(args.gpu_ids),
        overwrite=bool(args.overwrite),
        selected_class=str(args.selected_class),
        save_as_png=False,
        sam2_prompt=False,
        debug_mode=False,
        skip_no_detection=False,
        padding_value=0,
        padding_unit="percent",
        model_dir=str(model_dir),
    )

    consume_generator(gen)
    logging.info("Cropped at:", img_output_root / args.aspect_ratios.split(",")[0].strip())

    gen2 = resize_images(
            Model_Dir=str(model_dir),
            input_folder=str(img_output_root),
            output_folder=str(dest_path),
            resolutions="1024x1024",
            save_as_png=False,
            num_threads=4,
            overwrite=bool(args.overwrite),
    )
    consume_generator(gen2)
    logging.info("Resized at:", dest_root)


    ############KOHYA TRAINING ##############
    flux_dir = Path(args.flux_dir).resolve()
    saves_dir = Path(args.saves_dir).resolve()
    logging.info(f"FLUX_DIR={flux_dir}")
    logging.info(f"SAVES_DIR={saves_dir}")

    config_src = Path(args.config).read_text()

    resolved = (
        config_src
        .replace("${{inputs.flux_weights}}", str(flux_dir))
        .replace("${{inputs.train_data}}", str(dest_root))
        .replace("${{outputs.ModelSaves}}", str(saves_dir))
    )

    resolved_path = Path("runConfig.resolved.toml")
    resolved_path.write_text(resolved)

    print("Wrote", resolved_path)
    print("Resolved train_data_dir:", dest_root)
    print("Resolved pretrained model:", flux_dir / "flux1-dev.safetensors")

    script_dir = Path(__file__).parent
    # flux_train_script = script_dir / ".." / "kohya_ss" / "sd-scripts" / "flux_train_network.py"

    flux_train_script = "kohya_ss/sd-scripts/flux_train_network.py"
    run([
        "accelerate", "launch",
        "--num_machines", "1",
        "--num_processes", "1",
        "--mixed_precision", "bf16",
        "--dynamo_backend", "no",
        str(flux_train_script),
        "--config_file", str(resolved_path),
    ])    


if __name__ == "__main__":
    main()
