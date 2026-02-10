# Breast Histopathology Image Classification using DINOv2 (400×)

This repository implements a **breast cancer histopathology image classification pipeline** using a **frozen DINOv2 Vision Transformer (ViT-L/14)** as a feature extractor and a lightweight **MLP linear probe** trained on top of learned representations.  
The project focuses on **robust evaluation using stratified 5-fold cross-validation**, class-imbalance handling, and medical-grade training practices.

---

## 📌 Project Overview

- **Task**: Binary classification of breast histopathology images (Benign vs Malignant)
- **Magnification**: 400×
- **Backbone**: DINOv2 ViT-L/14 (frozen)
- **Classifier**: Multi-layer perceptron (MLP)
- **Evaluation**: Stratified 5-fold cross-validation + independent test set
- **Framework**: PyTorch

---

- Only the **MLP head is trained**
- DINOv2 backbone is used strictly as a **feature extractor**
- CLS token is used for classification
- Attention maps and patch tokens are optionally exposed for analysis

## Cross-Validation

- **5-Fold Stratified Cross-Validation**
- Each fold:
- Trains a separate MLP head
- Saves best checkpoint by validation AUROC
- Aggregated metrics and plots across folds:
- Train vs Validation loss
- Accuracy curves
- AUROC trends

---

## Cross-Validation Results

**5-Fold Stratified Cross-Validation Performance**

| Fold | Best Epoch | Best Val AUROC | Final Train AUROC |
|-----:|-----------:|---------------:|------------------:|
| 1 | 15 | 0.9479 | 0.9738 |
| 2 | 45 | 0.9441 | 0.9916 |
| 3 | 73 | 0.9561 | 0.9950 |
| 4 | 24 | 0.9641 | 0.9782 |
| 5 | 42 | 0.9649 | 0.9865 |
| **Mean** | – | **0.9554** | **0.9850** |
| **Std** | – | **0.0084** | **0.0080** |

- Low standard deviation across folds indicates **stable and consistent learning**
- Minimal overfitting observed between training and validation AUROC

---

## Final Test Evaluation (Ensemble)

Predictions from all 5 trained folds were **ensembled** and evaluated on a **held-out test set**.

**Test Set Performance**
- **Accuracy**: **82.69%**
- **AUROC**: **0.9605**

This demonstrates strong generalization despite class imbalance and domain complexity.

---

## 🛠 Tech Stack

- PyTorch
- DINOv2 (Vision Transformer)
- TorchVision
- Scikit-learn
- NumPy
- Matplotlib

---
## 🚀 Future Extensions

- Fine-tuning DINOv2 instead of frozen features
- Multi-magnification learning (40× / 100× / 400×)
- Multi-instance learning (MIL) for whole-slide images


