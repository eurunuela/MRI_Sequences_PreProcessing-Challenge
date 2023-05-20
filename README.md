# MRI Sequences PreProcessing Challenge

## Requirements

- Python 3
- [AFNI](https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/background_install/install_instructs/index.html)

Required 3rd party libraries:

- [Numpy](https://numpy.org/)
- [Nibabel](https://nipy.org/nibabel/)
- [Nilearn](https://nilearn.github.io/)

## Usage

Clone the repository:

```bash
git clone https://github.com/eurunuela/MRI_Sequences_PreProcessing-Challenge.git
```

Move into the repository folder:

```bash
cd MRI_Sequences_PreProcessing-Challenge
```

Create a virtual environment:

```bash
conda create -n challenge python=3.8
```

Activate the virtual environment:

```bash
conda activate challenge
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the script:

```bash
python preprocessing.py
```

## Summary

The script `preprocessing.py` performs the following steps:

1. Reads the MRI images from the `Sequences` folder.
2. Runs AFNI's `3dAutomask` to create a brain mask for the reference image.
3. Runs AFNI's `3dAllineate` to align the images to the reference image.
4. Loads the aligned images into a `niimg` object.
5. Trains a `RandomForestClassifier` to classify the voxels of `Image_01` based on the labels of the segmentation.
6. Predicts the labels of the voxels of `Image_03` using the trained classifier.
7. Saves the predicted labels as a Nifti file named `Image_03_SEG_pred.nii.gz`.

## Report

During the completion of the challenge, I encountered several noteworthy aspects. The initial task of coding the challenge proved to be straightforward, taking approximately one hour to write the complete script and verify the functionality of each step. In particular, I paid careful attention to ensuring the accuracy of the alignment computed by AFNI. Furthermore, I dedicated an additional hour to researching the implementation of coregistration and resampling techniques using Python. However, due to my familiarity with AFNI, I ultimately decided to utilize it for these processes. One notable limitation I encountered was the scarce amout of the training data provided for effectively training an accurate classifier. With only one image and its segmentation available, the results obtained with the random forest classifier were far from satisfactory. In addition, I explored the use of a logistic regression classifier, which yielded equally poor results when compared to the random forest classifier used to compute the final outcomes. I also explored the use of a support vector machine (SVM) classifier. Unfortunately, due to the computational intensity of training the SVM model, my laptop was unable to complete the training process. Finally, I made sure the code was well commented and followed the black and isort style guides. Overall, I enjoyed completing the challenge but would have liked to have more training data available to train a more accurate classifier.
