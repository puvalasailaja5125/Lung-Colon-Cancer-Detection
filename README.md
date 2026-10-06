# A Deep Learning Based Lung and Colon Cancer Detection Using DenseNet

A deep learning-based image classification system for detecting and classifying **lung and colon cancer from histopathological images** using **DenseNet121**, **Convolutional Block Attention Module (CBAM)**, **Focal Loss**, and **Grad-CAM** for explainability.

## 📌 Overview

Histopathological image analysis is an important part of cancer diagnosis, but manually examining large numbers of tissue images can be time-consuming. This project explores a deep learning approach for automatically classifying histopathological images into five lung and colon tissue categories.

The proposed system combines the feature extraction capability of **DenseNet121** with **CBAM attention** to focus on informative image regions. **Focal Loss** is used during training to give greater importance to difficult examples, while **Grad-CAM** provides visual explanations of model predictions.

> **Note:** This project is intended for academic and research purposes and is **not a medical diagnostic system**.

---

## 🎯 Objectives

* Develop a deep learning model for lung and colon histopathological image classification.
* Use **DenseNet121** for extracting deep image features.
* Integrate **CBAM** to improve attention to informative features.
* Use **Focal Loss** for hard-example learning.
* Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix.
* Use **Grad-CAM** to visualize regions contributing to model predictions.
* Provide a single-image prediction pipeline with class probabilities.

---

## 🧠 Classes

The model classifies images into five categories:

| Class       | Description                  |
| ----------- | ---------------------------- |
| `colon_aca` | Colon Adenocarcinoma         |
| `colon_n`   | Normal Colon Tissue          |
| `lung_aca`  | Lung Adenocarcinoma          |
| `lung_n`    | Normal Lung Tissue           |
| `lung_scc`  | Lung Squamous Cell Carcinoma |

---

## 🏗️ Model Architecture

The proposed model follows the pipeline:

```text
Input Histopathological Image
            ↓
       Preprocessing
            ↓
       DenseNet121
            ↓
          CBAM
            ↓
Global Average Pooling
            ↓
      Dropout (0.3)
            ↓
   Fully Connected Layer
            ↓
         Softmax
            ↓
     Five-Class Output
```

### Architecture Components

#### 1. DenseNet121

DenseNet121 is used as the main feature extraction backbone. The model is initialized with ImageNet-pretrained weights to leverage learned visual representations.

#### 2. CBAM

The **Convolutional Block Attention Module (CBAM)** is added after the DenseNet121 backbone.

CBAM contains:

* **Channel Attention** – identifies important feature channels.
* **Spatial Attention** – identifies important spatial regions.

This allows the network to emphasize informative features in histopathological images.

#### 3. Global Average Pooling

Global Average Pooling converts the spatial feature maps into a compact feature representation before classification.

#### 4. Dropout

A dropout rate of **0.3** is used before the final classification layer to help reduce overfitting.

#### 5. Softmax Classification

The final fully connected layer contains **5 output neurons** with a softmax activation function.

---

## 🔥 Focal Loss

Focal Loss is used as the training loss function for the proposed model.

It reduces the relative contribution of easily classified examples and gives greater importance to difficult examples.

The configuration used in this project is:

```text
Alpha = 0.25
Gamma = 2
```

The dataset is approximately balanced; therefore, Focal Loss is used primarily for **hard-example learning rather than correcting severe class imbalance**.

---

## 📊 Dataset

The project uses the **LC25000 lung and colon histopathological image dataset**.

The dataset contains five classes with approximately 5,000 images per class.

### Dataset Distribution

| Class                        | Approx. Images |
| ---------------------------- | -------------: |
| Colon Adenocarcinoma         |          5,000 |
| Normal Colon                 |          5,000 |
| Lung Adenocarcinoma          |          5,000 |
| Normal Lung                  |          5,000 |
| Lung Squamous Cell Carcinoma |          5,000 |
| **Total**                    |     **25,000** |

The project uses:

* **22,501 images** for training/validation.
* **2,499 images** for testing.
* Image size: **224 × 224**
* Batch size: **16**

---

## 🔄 Data Preprocessing

The images are processed before being provided to the model.

The preprocessing pipeline includes:

1. Loading histopathological images.
2. Converting images to RGB.
3. Resizing images to **224 × 224**.
4. Applying DenseNet-compatible preprocessing.
5. Creating batches using TensorFlow `tf.data`.
6. Applying data augmentation during training where configured.

---

## 🧪 Model Experiments

Three model configurations were evaluated.

### Model 1 — Baseline DenseNet121

```text
DenseNet121
     ↓
Global Average Pooling
     ↓
Dropout
     ↓
Softmax
```

### Model 2 — DenseNet121 + CBAM

```text
DenseNet121
     ↓
CBAM
     ↓
Global Average Pooling
     ↓
Dropout
     ↓
Softmax
```

### Model 3 — Proposed Model

```text
DenseNet121
     ↓
CBAM
     ↓
Global Average Pooling
     ↓
Dropout
     ↓
Softmax
```

The third configuration additionally uses **Focal Loss during training**.

> Focal Loss is a training objective and is therefore not an additional architecture layer.

---

## 📈 Results

The evaluated test accuracies were:

| Model                           | Test Accuracy |
| ------------------------------- | ------------: |
| DenseNet121 Baseline            |    **95.64%** |
| DenseNet121 + CBAM              |    **95.84%** |
| DenseNet121 + CBAM + Focal Loss |    **97.12%** |

The proposed configuration achieved an improvement of **1.48 percentage points** over the baseline.

### Final Model Class-wise Performance

| Class                        | Precision | Recall | F1-Score |
| ---------------------------- | --------: | -----: | -------: |
| Colon Adenocarcinoma         |      0.98 |   1.00 |     0.99 |
| Normal Colon                 |      1.00 |   0.99 |     0.99 |
| Lung Adenocarcinoma          |      0.94 |   0.94 |     0.94 |
| Normal Lung                  |      1.00 |   1.00 |     1.00 |
| Lung Squamous Cell Carcinoma |      0.94 |   0.93 |     0.93 |

The main classification confusion was observed between **lung adenocarcinoma and lung squamous cell carcinoma**.

---

## 🔍 Explainability with Grad-CAM

To make the model's predictions more interpretable, **Gradient-weighted Class Activation Mapping (Grad-CAM)** is used.

Grad-CAM generates a heatmap showing image regions that contribute to the predicted class.

The workflow is:

```text
Input Image
     ↓
DenseNet121 + CBAM
     ↓
Prediction
     ↓
Target Class
     ↓
Gradients
     ↓
Grad-CAM Heatmap
     ↓
Overlay on Original Image
```

This provides a visual indication of which regions the model considered important when making its prediction.

---

## 🖼️ Example Prediction

The project also includes a single-image inference pipeline.

Example output:

```text
Predicted Class: Lung Adenocarcinoma
Confidence: 88.12%
```

Example class probabilities:

```text
Colon Adenocarcinoma       : 0.03%
Normal Colon               : 0.01%
Lung Adenocarcinoma        : 88.12%
Normal Lung                : 5.23%
Lung Squamous Cell Carcinoma: 6.61%
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras
* DenseNet121
* CBAM
* Focal Loss

### Explainable AI

* Grad-CAM

### Data Processing

* NumPy
* TensorFlow `tf.data`

### Visualization

* Matplotlib
* Confusion Matrix
* Classification Metrics

---

## 💻 Environment

The project was developed and tested in the following environment:

```text
Operating System : Windows
Python           : 3.13.16
TensorFlow       : 2.21.0
GPU              : NVIDIA RTX 3050
Image Size       : 224 × 224
Batch Size       : 16
```

> TensorFlow 2.21 on native Windows was used in this project. GPU availability depends on the TensorFlow/CUDA environment and configuration.

---

## 📂 Project Structure

A recommended repository structure is:

```text
lung-colon-cancer-detection/
│
├── dataset/
│   └── LUNG_COLON_IMAGE_SET/
│
├── models/
│   └── densenet121_cbam_focal.keras
│
├── notebooks/
│   └── training.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── cbam.py
│   ├── focal_loss.py
│   ├── train.py
│   ├── evaluate.py
│   └── gradcam.py
│
├── results/
│   ├── model_accuracy_comparison.png
│   ├── per_class_metrics.png
│   ├── confusion_matrix_final.png
│   └── gradcam_all_classes.png
│
├── requirements.txt
├── README.md
└── LICENSE
```

The exact structure can be modified depending on how the final GitHub repository is organized.

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd lung-colon-cancer-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

A typical `requirements.txt` can contain:

```text
tensorflow==2.21.0
numpy
matplotlib
scikit-learn
pillow
```

Use the exact package versions from the project's working environment if additional dependencies are required by the implementation.

---

## ▶️ Running the Project

### Training

After placing the dataset in the expected directory, run the training script:

```bash
python src/train.py
```

### Evaluation

```bash
python src/evaluate.py
```

### Grad-CAM

```bash
python src/gradcam.py
```

### Single Image Prediction

```bash
python src/predict.py --image path/to/image.jpeg
```

The exact commands should be adjusted to match the scripts included in the repository.

---

## 📌 Limitations

* The model was trained and evaluated on the LC25000 dataset.
* Performance on external datasets may differ from the reported results.
* The model predicts one of the five predefined classes and may produce a confident prediction for images outside the training distribution.
* The reported experiments used a limited training setup and should not be interpreted as evidence of clinical effectiveness.
* The model has not been clinically validated.
* Grad-CAM provides an interpretability visualization but does not establish clinical reasoning or causality.

---

## 🔮 Future Scope

Potential improvements include:

* Training for more epochs with systematic hyperparameter tuning.
* Evaluation on independent external histopathology datasets.
* Cross-validation for more robust performance estimation.
* Comparison with additional CNN and transformer architectures.
* Improved handling of lung adenocarcinoma and lung SCC confusion.
* More extensive explainability evaluation.
* Development of a web-based inference interface.
* Deployment as a research-oriented image analysis application.

---

## 👩‍💻 Authors

**PUVALA SAILAJA**
**DEBBADWAR BASWARAJ**
**DACHEPALLY ROHIT**
**PANEM MOUNIKA**
**VALLAPU SAI BHAGATH**

Department of Computer Science and Engineering
Anurag University, Hyderabad, India

---

## ⚠️ Disclaimer

This project is developed for **academic and research purposes only**.

It is not intended to replace professional medical diagnosis, clinical judgment, pathology examination, or medical advice. The predictions generated by this model should not be used for making clinical decisions.
