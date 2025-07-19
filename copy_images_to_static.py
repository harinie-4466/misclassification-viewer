import os
import pandas as pd
import shutil

# Load the prediction results
df = pd.read_csv("point_predictions.csv")

# Get list of unique image filenames
image_list = df['filename'].unique()

# Create the static/images directory if it doesn't exist
os.makedirs("static/images", exist_ok=True)

# Copy each image from images/ to static/images/
for img in image_list:
    src = os.path.join("images", img)
    dst = os.path.join("static/images", img)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Copied: {img}")
    else:
        print(f"Image not found: {img}")
