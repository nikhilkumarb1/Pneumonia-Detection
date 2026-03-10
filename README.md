# 🫁 PneumoScan - AI Pneumonia Detection from Chest X-rays

An AI-powered web application that detects **Pneumonia from Chest X-ray images** using a **Vision Transformer (ViT)** deep learning model.  
The system uses a **FastAPI backend** for model inference and a **Streamlit frontend** for interactive visualization.

This project demonstrates how **Deep Learning + Medical Imaging + APIs** can be combined to build an intelligent healthcare diagnostic tool.

---

## 🚀 Features

- Upload Chest X-ray images
- AI-powered pneumonia detection
- Confidence score for prediction
- Risk level classification (LOW / MEDIUM / HIGH)
- Human-readable medical verdict
- Real-time inference using FastAPI
- Interactive visualization using Streamlit
- Probability chart for model predictions

---

## 🧠 Model

This project uses a pretrained Vision Transformer model from Hugging Face:

Model:  
`nickmuchi/vit-finetuned-chest-xray-pneumonia`

Architecture:
- Vision Transformer (ViT)
- Fine-tuned on Chest X-ray dataset
- Binary classification:
  - NORMAL
  - PNEUMONIA

---

## 🏗️ Tech Stack

### Backend
- FastAPI
- PyTorch
- HuggingFace Transformers
- PIL

### Frontend
- Streamlit
- Plotly
- Requests

### AI / ML
- Vision Transformer (ViT)
- Image Classification
- Medical Image Analysis

---


---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/pneumoscan-ai.git
cd pneumoscan-ai
```
## Create virtual environment
```bash
python -m venv venv
```
## Activate Environment
```bash
venv\Scripts\activate
```
## Install Dependencies
```bash
pip install -r requirements.txt
```

