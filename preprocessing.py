import os.path as op
import subprocess

import nibabel as nib
import numpy as np
from nilearn.maskers import NiftiMasker
from sklearn.ensemble import RandomForestClassifier

# Directory containing the data
data_dir = "Sequences"

# Load the 3 images and the segmentation
print("Loading images...")
img1 = nib.load(op.join(data_dir, "Image_01.nii.gz"))
img2 = nib.load(op.join(data_dir, "Image_02.nii.gz"))
img3 = nib.load(op.join(data_dir, "Image_03.nii.gz"))
seg = nib.load(op.join(data_dir, "Image_01_SEG.nii.gz"))

# Create a mask using image AFNI's 3dAutomask and Image_03
# Only create the mask if it does not exist yet
mask_fn = "mask.nii.gz"
if not op.exists(op.join(data_dir, mask_fn)):
    print("Creating mask to reduce computation time...")
    subprocess.run(
        f"3dAutomask -prefix {mask_fn} Image_03.nii.gz",
        shell=True,
        cwd=data_dir,
    )

# Coregister and resample the images and segmentation using Image_03 as reference
# Useing AFNI's 3dAllineate

# List with image names that need to be coregistered
images = ["Image_01.nii.gz", "Image_02.nii.gz", "Image_01_SEG.nii.gz"]
images_out = []

print("Coregistering images...")
# Loop over images
for image in images:

    out_image = image.split(".")[0] + "_coreg.nii.gz"
    images_out.append(out_image)

    # Check if the output image already exists
    if op.exists(op.join(data_dir, out_image)):
        print(f"{out_image} already exists. Skipping...")
        continue

    print(f"Coregistering {image}...")

    # Create the command
    allineate_command = f"3dAllineate -base Image_03.nii.gz -source {image} -prefix {out_image} -overwrite"

    # Run the command
    subprocess.run(
        allineate_command,
        shell=True,
        cwd=data_dir,
    )

# Load the coregistered images and mask them using nilearn's NiftiMasker
# Create a masker object
masker = NiftiMasker(mask_img=op.join(data_dir, mask_fn))

# Load the images
print("Loading coregistered images...")
img1_coreg = nib.load(op.join(data_dir, images_out[0]))
img1_coreg_data = masker.fit_transform(img1_coreg)
img2_coreg = nib.load(op.join(data_dir, images_out[1]))
img2_coreg_data = masker.fit_transform(img2_coreg)
img3_data = masker.fit_transform(img3)
seg_coreg = nib.load(op.join(data_dir, images_out[2]))
seg_coreg_data = masker.fit_transform(seg_coreg)

# Train a random forest classifier to predict the segmentation from the images
# Get the data ready
X = img1_coreg_data.flatten().reshape(-1, 1)
y = seg_coreg_data.flatten()

# Make sure y is an integer array
y = y.astype(int)

# Create the Random Forest Classifier model
model = RandomForestClassifier(n_jobs=-1, verbose=1)

# Train the model
print("Training the model...")
model.fit(X, y)

# Evaluate the model
print("Evaluating the model...")
accuracy = model.score(X, y)

# Evaluate the model on the training data
print(f"Accuracy: {accuracy}")

# Predict the segmentation of Image_03
print("Predicting the segmentation of Image_03...")
X_pred = img3_data.flatten().reshape(-1, 1)
y_pred = model.predict(X_pred)

# Reshape the prediction to the original image shape and save it
pred_img = masker.inverse_transform(y_pred[np.newaxis, :])
nib.save(pred_img, op.join(data_dir, "Image_03_SEG_pred.nii.gz"))

print("Done!")
