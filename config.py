from pathlib import Path
import shutil

if __name__ == '__main__':

    src_dir = Path("train/train")
    dest_dir = Path("data/cat")

    dest_dir.mkdir(parents=True, exist_ok=True)

    for file in src_dir.iterdir():
        if "cat" in file.name:
            shutil.copy(src_dir/Path(file.name), dest_dir/Path(file.name))
    



