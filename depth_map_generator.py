
"""
Generate model-predicted depth maps from the RGB dataset.

Run from the directory containing RGB_Dataset:
    python depth_map_generator.py

Expected input: RGB_Dataset/<split>/<class_name>/<image_filename>
Saved output: Depth_Anything_V2_Dataset/<split>/<class_name>/<image_filename>
Split names, class folders, filenames, and image extensions are preserved.
Each class folder should contain only images supported by the pipeline.

The script uses the Hugging Face Depth Anything V2 Small checkpoint. Model
files are downloaded on first use unless already cached. CUDA is selected
when available; otherwise inference runs on the CPU.

The pipeline's depth image is normalized independently for each image to
0-255 and saved as an 8-bit grayscale image. Constant-valued images become
all zeros. These are predicted relative-depth visualizations, not measured
depth values in physical units. The original file extension determines the
OpenCV output encoding (for example, JPEG uses lossy compression).

Existing output paths are skipped without checking their contents. To
regenerate an output after changing the model or processing, remove that
output first.

For reproducibility, record the checkpoint revision and package versions
used for the experiments; this script currently does not pin them.
"""
import os
import cv2
import torch 
import numpy as np 
from transformers import pipeline


# Select an available CUDA device, falling back to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load the pretrained Depth Anything V2 Small depth-estimation pipeline.
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)

print("Successfully loaded depth_anything_v2-small depth_anything_v2 model. ")

# Paths are relative to the current working directory.
original_destination = 'RGB_Dataset'
depth_destination = f"Depth_Anything_V2_Dataset" 

os.makedirs(depth_destination, exist_ok=True)

# Mirror the input split/class structure and generate missing depth maps.
for folder in os.listdir(original_destination):  # Iterate over dataset splits (for example, train, validation, test).
    print(f"Processing the {folder} folder")
    new_folder = os.path.join(depth_destination, folder)
    old_folder = os.path.join(original_destination, folder)
    if (not os.path.isdir(new_folder)):
        os.mkdir(new_folder)  
    for class_name in os.listdir(old_folder):   # Iterate over class folders.
        new_folder2 = os.path.join(new_folder, class_name)
        old_folder2 = os.path.join(old_folder, class_name)
        if (not os.path.isdir(new_folder2)):
            os.mkdir(new_folder2)
        for im in os.listdir(old_folder2):   # Iterate over images in this class.
            desti_name = os.path.join(new_folder2, im)
            # Resume previous runs without overwriting existing outputs.
            if os.path.exists(desti_name):
                continue

            image_directory = os.path.join(old_folder2,im)
            print("Processing image name : ", image_directory)
        
            result = pipe(image_directory)
            depth_map = result["depth"] # Use the pipeline-produced PIL depth image.
            # Convert to a NumPy array for per-image normalization.
            depth_map = np.array(depth_map)

            # Scale this image to 0-255; this does not preserve a shared depth scale across images.
            depth_min = depth_map.min()
            depth_max = depth_map.max()

            if depth_max > depth_min:
                 depth_norm = (depth_map - depth_min)/(depth_max-depth_min)
            else: 
                 depth_norm = depth_map * 0     # Avoid division by zero for constant-valued images.

            depth_8bit = (depth_norm*255).astype('uint8') 
            

            # Save with the original filename and extension.
            cv2.imwrite(desti_name,depth_8bit )
