# Front Facial Reconstruction for Deep Fake Applications

Our project aims to design a model that reconstructs a person’s face in frontal position from side-view images. 
We believe that leveraging multiple image inputs will enhance the synthesis quality when the source image isn’t of a frontal face. 
This can be done by extracting spatial features of a side view face and estimating the 3D transformation matrix that would transform it into a frontal position. 
Applying the transformation and adding the corresponding color to the features would yield a reconstructed front-facing view of the individual.

---
# Downloading the Dataset
To retrieve the dataset execute the following command then unzip the data.

```
python libs/download.py
```
Along with dataset you also have to download the image bank required. Download the image bank from the google drive link:
https://drive.google.com/drive/folders/1RJ7RPkz-Ug1oUzqGDCK2P7YFAObxPr2X?usp=sharing

And place it in the `dataset/image_bank/samples` folder created when the above command was executed

---
# How to Execute

Run the following command:
```
python main.py --query_image <path to input image>
```
---
