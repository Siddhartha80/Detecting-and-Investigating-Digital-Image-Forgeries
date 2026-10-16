<div align="center">

# 🕵️ Detecting and Investigating Digital Image Forgeries

### Deep learning for passive image forensics: authentic vs. tampered image classification

*A comparative study of DenseNet201, EfficientNetB0, ResNet50 and Xception (plus attention and cross-validation variants) on the CASIA 2.0 forgery dataset.*

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)
![Domain](https://img.shields.io/badge/Domain-Image%20Forensics-blueviolet)
![Dataset](https://img.shields.io/badge/Dataset-CASIA%202.0-informational)

</div>

---

## 📖 Overview

Image-editing tools have made it trivial to alter photographs, which raises serious questions about the trustworthiness of visual content used in journalism, social media, and legal evidence. Manipulations such as **copy-move** and **splicing** are often invisible to the human eye.

This project takes a **passive (blind) forensics** approach: no watermarks or embedded signatures are needed. Instead, convolutional neural networks learn to separate **authentic** images from **tampered** ones directly from pixel content. Several pretrained architectures are fine-tuned with transfer learning and compared under the same data pipeline.

**Highlights**

- Binary classifier (Real vs. Fake) built with transfer learning from ImageNet
- Side-by-side comparison of **DenseNet201, EfficientNetB0, ResNet50 and Xception**
- Experiments with **fine-tuning**, an **attention layer**, and **5-fold stratified cross-validation** on DenseNet201
- Per-image inference cells that classify a single image as *Real* or *Fake*

---

## 🧠 Approach

```
 CASIA 2.0 images ──▶ Resize (224×224) & normalize ──▶ Train/validation split (80/20)
                                                              │
                                                              ▼
   Real / Fake ◀── Dense(2, softmax) ◀── Dropout ◀── Dense(256) ◀── GAP ◀── Pretrained CNN backbone
```

| Step | Details |
|------|---------|
| **Data** | CASIA 2.0 authentic (`Au`) and tampered (`Tp`) images; labels: `1 = Real`, `0 = Fake` |
| **Preprocessing** | Resize to 224×224, scale pixels to `[0, 1]`, custom batch generator (batch size 32) |
| **Backbone** | ImageNet-pretrained, base layers frozen at first |
| **Head** | Global Average Pooling → Dense(256, ReLU) → Dropout(0.5) → Dense(2, softmax) |
| **Optimization** | Adam, categorical cross-entropy, `EarlyStopping` + `ReduceLROnPlateau` |
| **Fine-tuning** | Last 20 layers of DenseNet201 unfrozen, learning rate 1e-5 |

---

## 🔬 Models Explored

| Model | Notebook | Notes |
|-------|----------|-------|
| **DenseNet201** | `Densenet_fin.ipynb` | Frozen-backbone baseline (up to 20 epochs) |
| **DenseNet201 + fine-tuning / attention / 5-fold CV** | `Semi_final.ipynb`, `densenet_partial.ipynb` | Full experiment flow: baseline → fine-tune → attention layer → stratified K-fold, plus single-image inference demos |
| **EfficientNetB0** | `EfficientNet.ipynb` | Uses the `efficientnet` package, 10 epochs |
| **ResNet50** | `ResNet.ipynb` | Held-out test split (942 test images) |
| **Xception** | `Xception.ipynb` | Depthwise-separable convolution backbone |

---

## 📊 Dataset

[CASIA Image Tampering Detection Dataset (v2.0)](https://www.kaggle.com/datasets/sophatvathana/casia-dataset) with authentic (`Au`) and tampered (`Tp`) images covering copy-move, splicing, and other manipulations.

- The notebooks load the `.jpg`/`.png` files, giving **9,418 images** (7,534 train / 1,884 validation)
- The project report describes the full dataset as 12,616 images (7,492 authentic, 5,124 tampered); the notebooks use the subset above because `.tif` files are skipped by the loader
- Download the dataset separately; it is **not** included in this repository

---

## 📈 Results

Accuracy and loss comparison, as reported in the project report:

| Model | Train Acc. | Validation Acc. | Test Acc. | Validation Loss |
|-------|-----------:|----------------:|----------:|----------------:|
| DenseNet201 | 0.863 | 0.819 | 0.809 | 0.429 |
| DenseNet201 (fine-tuned) | 0.887 | 0.822 | **0.819** | 0.524 |
| EfficientNetB0 | 0.863 | 0.816 | 0.815 | 0.436 |
| Xception | 0.836 | 0.798 | 0.797 | 0.465 |
| ResNet50 | 0.783 | 0.774 | 0.768 | 0.531 |
| DenseNet201 + 5-fold CV ("cross model") | 0.784 | 0.777 | 0.771 | 0.578 |
| DenseNet201 + attention layer | 0.783 | 0.769 | 0.229 ¹ | 0.541 |

¹ The attention variant's test run diverged (test loss ≈ 12.6), even though its training and validation accuracy were in line with the other models. It is kept for transparency and is a good candidate for future debugging.

**Takeaways**

- Fine-tuned **DenseNet201** and **EfficientNetB0** performed best, at roughly 82% test accuracy.
- EfficientNetB0 reached the strongest recall for forged images in the report (≈ 0.97) at a much lower compute cost per epoch than Xception.
- Simply adding an attention layer or cross-validation did not improve over the plain fine-tuned DenseNet in this setup.

---

## 🗂️ Repository Structure

```
.
├── Densenet_fin.ipynb         # DenseNet201 baseline
├── densenet_partial.ipynb     # DenseNet201 experiments (partial run)
├── Semi_final.ipynb           # DenseNet201: fine-tuning, attention, K-fold, inference demos
├── EfficientNet.ipynb         # EfficientNetB0
├── ResNet.ipynb               # ResNet50
├── Xception.ipynb             # Xception
├── Detecting and Investigating Digital Image Forgeries.docx   # Project paper
├── E4_sip_report.docx         # Project report (supervised by faculty guides)
├── requirements.txt           # Python dependencies
└── README.md
```

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.9–3.11
- A GPU is strongly recommended. Training logs show roughly 10–35 minutes per epoch on CPU.

### Installation

```bash
git clone https://github.com/Siddhartha80/Detecting-and-Investigating-Digital-Image-Forgeries.git
cd Detecting-and-Investigating-Digital-Image-Forgeries

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Run

1. Download **CASIA 2.0** from [Kaggle](https://www.kaggle.com/datasets/sophatvathana/casia-dataset).
2. In the notebook you want to run, update the dataset paths. They are currently Windows paths such as `D:\archive (1)\CASIA2\Au\` and `...\CASIA2\Tp\`.
3. Launch Jupyter and run the cells in order:

```bash
jupyter notebook
```

> The notebooks were written against an older Keras API (`Adam(lr=...)`, `keras.preprocessing`). If you hit deprecation errors on a recent TensorFlow, use `learning_rate=` instead of `lr=`, or install a TensorFlow 2.x version contemporary with the notebooks.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Deep learning:** TensorFlow / Keras (DenseNet201, ResNet50, Xception), `efficientnet` (EfficientNetB0)
- **ML utilities:** scikit-learn (splits, metrics, stratified K-fold)
- **Visualization:** Matplotlib, Seaborn (training curves, confusion matrices)
- **Environment:** Jupyter Notebook

---

## 🚀 Future Work

- Stronger CNN and transformer-based backbones for better feature extraction
- **Localization** of tampered regions, not just image-level classification
- Real-time forgery detection for social media pipelines
- **Multi-modal** detection that adds metadata, sensor noise, and contextual information
- Evaluation on additional datasets (e.g., deepfake and CelebA-based sets) to test generalization

---

## 👥 Authors

| Name | Role |
|------|------|
| M Ramya Sree | Author |
| [Mettu Siddhartha](https://github.com/Siddhartha80) | Author |
| Poli Vamsi Vardhan Reddy | Author |
| Dr. Nidhin Prabhakar T V | Faculty guide |
| Dr. Rimjhim Padam Singh | Faculty guide |

Department of Computer Science and Engineering, Amrita School of Computing, Bengaluru, **Amrita Vishwa Vidyapeetham**.

---

## 📬 Contact

**Mettu Siddhartha** · [LinkedIn](https://linkedin.com/in/siddhartha-mettu/) · [GitHub](https://github.com/Siddhartha80)

---

## 📚 Selected References

1. Zanardelli et al., *Image forgery detection: a survey of recent deep-learning approaches*, Multimedia Tools and Applications, 2022.
2. Barad & Goswami, *Image Forgery Detection using Deep Learning: A Survey*, ICACCS 2020.
3. Zhong & Pun, *An End-to-End Dense-InceptionNet for Image Copy-Move Forgery Detection*, IEEE TIFS, 2020.
4. Zhang & Ni, *A Dense U-Net with Cross-Layer Intersection for Detection and Localization of Image Forgery*, ICASSP 2020.
5. Kadam et al., *Detection and Localization of Multiple Image Splicing Using MobileNet V1*, IEEE Access, 2021.
6. Sharma & Singh, *Deep CNN with ResNet-50 Learning Algorithm for Copy-Move Forgery Detection*, ICSC 2021.

The full reference list is in the project report.

---

## ⚠️ Disclaimer

This project is for research and educational purposes. Classifier output is probabilistic and was evaluated on a single benchmark dataset, so it should not be used as standalone evidence in forensic or legal settings.

---

<div align="center">

⭐ If you find this project useful, consider giving it a star!

</div>
