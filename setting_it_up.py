import gdown
import zipfile
import os
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd

# Download
file_id = "1vpG9NBHI_9V5zJsY6Cpe1uys2HTnC2vh"
url = f"https://drive.google.com/uc?id={file_id}"
zip_file = "data.zip"
gdown.download(url, output=zip_file, quiet=False)

# Extract to temp dir
temp_dir = "temp_extract"
os.makedirs(temp_dir, exist_ok=True)
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(temp_dir)

# Move inner data/ to target data/
shutil.move(os.path.join(temp_dir, "data"), "data")

# Cleanup
shutil.rmtree(temp_dir)


######################################## CONVERT TO YOLO FORMAT ########################################
# Paths
CSV_PATH = "data/train.csv"
IMAGE_DIR = "data/train_images"
OUTPUT_DIR = "yolo_dataset"

# Classes
classes = ['Trophozoite', 'WBC', 'NEG']
class_to_id = {cls: idx for idx, cls in enumerate(classes)}

# Load CSV
df = pd.read_csv(CSV_PATH)

# Ensure output dirs exist
for split in ['train', 'val']:
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

# Split images into train/val
unique_images = df['Image_ID'].unique()
train_imgs, val_imgs = train_test_split(unique_images, test_size=0.1, random_state=42)

split_map = {img_id: 'train' for img_id in train_imgs}
split_map.update({img_id: 'val' for img_id in val_imgs})

# Group annotations by image
grouped = df.groupby('Image_ID')

for image_id, group in grouped:
    split = split_map[image_id]

    # Copy image to correct folder
    src_img = os.path.join(IMAGE_DIR, image_id)
    dst_img = os.path.join(OUTPUT_DIR, 'images', split, image_id)
    shutil.copyfile(src_img, dst_img)

    # Read image size
    with Image.open(src_img) as img:
        width, height = img.size

    # Create YOLO label file
    label_path = os.path.join(OUTPUT_DIR, 'labels', split, image_id.replace('.jpg', '.txt'))
    with open(label_path, 'w') as f:
        for _, row in group.iterrows():
            cls = class_to_id[row['class']]
            # Get box
            x_min, y_min, x_max, y_max = row[['xmin', 'ymin', 'xmax', 'ymax']]
            x_center = ((x_min + x_max) / 2) / width
            y_center = ((y_min + y_max) / 2) / height
            box_width = (x_max - x_min) / width
            box_height = (y_max - y_min) / height

            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

yaml_path = os.path.join(OUTPUT_DIR, 'data.yaml')
with open(yaml_path, 'w') as f:
    f.write(f"""path: {OUTPUT_DIR}
train: images/train
val: images/val

names:
""")
    for idx, name in enumerate(classes):
        f.write(f"  {idx}: {name}\n")

print("✅ YOLO dataset created in:", OUTPUT_DIR)
