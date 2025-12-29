import os
from PIL import Image, UnidentifiedImageError

root = "/mnt/data/wonjung/datasets/ImageNet/train"
bad_list_path = "./bad_imagenet_files.txt"

bad = []
for dirpath, _, filenames in os.walk(root):
    for fn in filenames:
        if not fn.lower().endswith((".jpeg", ".jpg", ".png")):
            continue
        path = os.path.join(dirpath, fn)
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                im.convert("RGB").load()
        except (UnidentifiedImageError, OSError, ValueError):
            bad.append(path)

with open(bad_list_path, "w") as f:
    for p in bad:
        f.write(p + "\n")

print(f"bad images: {len(bad)}개, 목록 저장: {bad_list_path}")