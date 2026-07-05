# Intel Image Classification with a Custom CNN

Hi, this is Jayden. This is my first deep learning project.

I am still learning and improving my coding and deep learning skills, so this project mainly focuses on understanding the full workflow of a CNN image classification project rather than building the most advanced model.

More computer vision and machine learning projects will be added in the future.

---

## Project Overview

This project trains a custom convolutional neural network (CNN) to classify natural scene images into six categories:

- buildings
- forest
- glacier
- mountain
- sea
- street

---

## Dataset

This project uses the Intel Image Classification dataset from Kaggle:

```text
https://www.kaggle.com/datasets/puneet6060/intel-image-classification
```

---

## Pipeline

```text
1. Load image dataset
2. Apply image transforms and data augmentation
3. Build a custom CNN model
4. Train the model on the training set
5. Evaluate the model on the test set
6. Save the best model checkpoint
7. Load the best model for prediction
8. Visualize prediction results
```

---

## Model Architecture

The model is a custom CNN written in PyTorch. It contains three convolutional blocks and a classifier head.

Each convolutional block follows this basic structure:

```text
Conv2d → BatchNorm2d → ReLU → Conv2d → BatchNorm2d → ReLU → MaxPool2d
```

The classifier head follows this structure:

```text
AdaptiveAvgPool2d → Flatten → Dropout → Linear
```

Main components:

- `Conv2d`: extracts image features
- `BatchNorm2d`: stabilizes training
- `ReLU`: adds non-linearity
- `MaxPool2d`: reduces spatial size
- `AdaptiveAvgPool2d`: compresses feature maps to a fixed size
- `Dropout`: reduces overfitting
- `Linear`: outputs class logits

---

## Training

The model is trained using:

```text
Loss function: CrossEntropyLoss
Optimizer: Adam
Device: CUDA GPU if available
Checkpoint: best model saved as models/best.pth
```

During training, the best model is saved based on test accuracy.

---

## Results

The model reached approximately **89% best test accuracy** during training.

Training curves such as loss and accuracy are visualized in `main.ipynb`.

Prediction visualization for images from the `seg_pred` folder is saved to:

```text
results/predictions.png
```

## Project Structure

```text
├── main.ipynb
├── model.py
├── train.py
├── save.py
├── predict.py
├── README.md
├── requirements.txt
├── .gitignore
└── results/
    └── predictions.png
```

---

## Notes

This project is mainly for learning and practice. The model is a manually built CNN, not a pretrained model.

Future improvements may include:

- Transfer learning with pretrained models
- More advanced CNN architectures

Thanks for watching !!!
