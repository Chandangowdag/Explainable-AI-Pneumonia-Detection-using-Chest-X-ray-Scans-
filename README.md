# Explainable-AI-Pneumonia-Detection-using-Chest-X-ray-Scans-


## 📌 Project Overview

This project implements an automated **Pneumonia Detection System** using a pre-trained **VGG19 Convolutional Neural Network (CNN)** and Explainable AI (XAI) techniques. The model classifies chest X-ray images as either:

- NORMAL
- PNEUMONIA

To improve transparency and interpretability, the project integrates:

- Grad-CAM (Gradient-weighted Class Activation Mapping)
- LIME (Local Interpretable Model-Agnostic Explanations)

These methods help visualize the regions of chest X-rays that influence the model's predictions.

---

## 🚀 Features

- Transfer Learning using VGG19
- Binary Classification of Chest X-ray Images
- Data Augmentation and Preprocessing
- Performance Evaluation Metrics
- Explainable AI Visualizations
- Quantitative Evaluation of Explanations
- Grad-CAM Heatmap Generation
- LIME-Based Local Explanations

---

## 📂 Dataset Structure

```text
chest_xray/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Dataset Source:

- Chest X-Ray Images (Pneumonia) Dataset
- Available on Kaggle and medical imaging repositories.

---

## 🛠️ Tech Stack

| Category | Tools |
|-----------|---------|
| Deep Learning | TensorFlow, Keras |
| Computer Vision | OpenCV |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Machine Learning | Scikit-learn |
| Explainability | LIME, Grad-CAM |

---

## 🏗️ Model Architecture

### Base Network

- VGG19 (Pre-trained on ImageNet)
- Top layers removed (`include_top=False`)
- Transfer Learning approach

### Classification Head

```text
Input Image (256x256x3)
        │
        ▼
      VGG19
        │
        ▼
Global Average Pooling
        │
        ▼
Dense Layer
        │
        ▼
Softmax Layer
        │
        ▼
Prediction
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/pneumonia-detection-vgg19-xai.git
cd pneumonia-detection-vgg19-xai
```

Install dependencies:

```bash
pip install tensorflow
pip install numpy pandas matplotlib
pip install opencv-python
pip install scikit-learn
pip install lime
pip install scikit-image
```

Or create a requirements file and install:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Step 1: Prepare Dataset

Place the dataset inside:

```text
chest_xray/
```

following the structure shown above.

### Step 2: Launch Notebook

```bash
jupyter notebook Vgg19_xai.ipynb
```

Run all cells sequentially.

---

## 💾 Model Saving

The trained model can be saved as:

```python
model.save("vgg19_pneumonia_model.h5")
```

Saved model location:

```text
saved_model/
└── vgg19_pneumonia_model.h5
```

---

## 📊 Evaluation Metrics

The project evaluates model performance using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Example Output:

```text
Accuracy : 95.2%
Precision: 94.8%
Recall   : 95.1%
F1 Score : 94.9%
```

---

# 🔍 Explainable AI (XAI)

## Grad-CAM

Grad-CAM generates heatmaps showing which regions of the X-ray contributed most to the prediction.

### Workflow

```text
Input X-ray
     │
     ▼
VGG19 Prediction
     │
     ▼
Gradient Computation
     │
     ▼
Heatmap Generation
     │
     ▼
Overlay on Original Image
```

### Sample Output

- Original X-ray
- Heatmap Visualization
- Overlay Result

---

## LIME

LIME explains predictions by perturbing image segments and measuring their impact on the output.

### Generated Explanations

- Positive Regions
- Negative Regions
- Feature Importance Maps
- Segmented Interpretations

---

# 📈 XAI Evaluation Metrics

The notebook evaluates explanation quality using:

| Metric | Description |
|----------|-------------|
| Fidelity | Agreement between explanation and model |
| Sufficiency | Information retained by highlighted regions |
| Comprehensiveness | Performance drop after removing important regions |
| Confidence Drop | Reduction in prediction confidence |
| Sparsity | Concentration of explanation regions |
| Insertion Score | Confidence gain as important pixels are added |

---

## 🔄 Project Workflow

```text
Chest X-ray Image
        │
        ▼
Image Preprocessing
        │
        ▼
VGG19 Classification
        │
        ▼
Prediction
        │
 ┌──────┴──────┐
 ▼             ▼
Grad-CAM      LIME
 ▼             ▼
Visual Explanations
        │
        ▼
XAI Evaluation Metrics
```

---

## 📁 Project Structure

```text
Pneumonia-Detection-VGG19-XAI/
│
├── dataset/
│
├── saved_model/
│   └── vgg19_pneumonia_model.h5
│
├── outputs/
│   ├── gradcam_results/
│   ├── lime_results/
│   └── evaluation_metrics/
│
├── Vgg19_xai.ipynb
├── requirements.txt
└── README.md
```

---

## 🎯 Results

The model successfully:

✅ Detects pneumonia from chest X-rays

✅ Generates interpretable Grad-CAM heatmaps

✅ Produces LIME explanations

✅ Evaluates explanation quality using quantitative XAI metrics

This enhances trust, transparency, and reliability in AI-assisted medical diagnosis.

---

## 🔮 Future Enhancements

- Fine-tuning VGG19 layers
- Comparison with ResNet50, DenseNet121, EfficientNet
- SHAP-based explainability
- Streamlit/Flask deployment
- Multi-class lung disease classification
- Clinical report generation

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Open a Pull Request

---

## 📜 License

This project is intended for educational and research purposes only.

The predictions generated by this model should not be used as a substitute for professional medical diagnosis.

---

## 👨‍💻 Author

**Chandan Gowda G**

