# Rice Leaf Disease Detection

A deep-learning pipeline that classifies rice leaf images into 6 disease categories using a custom CNN built with TensorFlow/Keras.

## Disease Classes

| Class | Folder name |
|---|---|
| Bacterial Leaf Blight | `bacterial_leaf_blight` |
| Brown Spot | `brown_spot` |
| Healthy Rice Leaf | `healthy_rice_leaf` |
| Leaf Blast | `leaf_blast` |
| Leaf Scald | `leaf_scald` |
| Sheath Blight | `sheath_blight` |

## Pipeline Overview

1. **Download dataset** – uses `kagglehub` to fetch the [Rice Disease Dataset](https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset).
2. **Rename folders** – normalises class folder names to snake_case.
3. **Train / Val / Test split** – 70 % / 15 % / 15 % stratified random split (seed 42).
4. **Data augmentation** – rotation, zoom, shear, horizontal flip applied to the training set via `ImageDataGenerator`.
5. **CNN model** – four `Conv2D + MaxPooling2D` blocks followed by `GlobalAveragePooling2D`, a 256-unit Dense layer (dropout 0.4), and a softmax output layer.
6. **Training** – 20 epochs with the Adam optimiser and categorical cross-entropy loss.
7. **Evaluation** – final accuracy and loss reported on the held-out test set.
8. **Plots** – training vs validation accuracy and loss curves.

## Quickstart (Google Colab)

```bash
pip install kagglehub tensorflow matplotlib
```

Open `rice_leaf_disease_detection.ipynb` in Google Colab and run all cells in order.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- kagglehub
- matplotlib
