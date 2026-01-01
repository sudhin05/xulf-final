import argparse
from huggingface_hub import snapshot_download

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", default="Sudhin05/Lt-Yolo")
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--revision", default=None)  
    ap.add_argument("--allow_patterns", nargs="*", default=None)  
    args = ap.parse_args()

    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=args.out_dir,
        local_dir_use_symlinks=False,  
        resume_download=True,
        allow_patterns=args.allow_patterns,
    )

if __name__ == "__main__":
    main()
