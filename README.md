# 🎭 Multimodal Video Sentiment & Emotion Analysis

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/-HuggingFace-FDEE21?style=for-the-badge&logo=HuggingFace&logoColor=black)
![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

An advanced deep learning model that predicts both **Sentiment** (Positive, Neutral, Negative) and **Emotion** (Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise) from video inputs. It leverages a powerful tri-modal architecture combining visual, audio, and textual features for rich, contextual understanding.

## 🧠 Architecture Overview

[![Architecture diagram of tearcake/video-sentiment-model](https://gitdiagram.com/tearcake/video-sentiment-model/diagram.png)](https://gitdiagram.com/tearcake/video-sentiment-model?utm_source=readme&utm_medium=picture)

The model fuses three distinct modalities to achieve state-of-the-art context awareness:

- **📝 Text Modality (`TextEncoder`)**: Utilizes a frozen **BERT** (`bert-base-uncased`) model from Hugging Face to extract deep semantic meaning from spoken utterances.
- **🎬 Video Modality (`VideoEncoder`)**: Employs a pre-trained **3D ResNet-18** to capture spatial and temporal features directly from the video frames.
- **🔊 Audio Modality (`AudioEncoder`)**: Uses a custom **1D Convolutional Neural Network (CNN)** that processes Mel Spectrograms generated using `torchaudio`.

These features are combined through a dense fusion layer and passed into two distinct classification heads: one for Sentiment (3 classes) and one for Emotion (7 classes).

## 📊 Dataset

This project is trained and evaluated on the **[MELD (Multimodal Emotion Lines Dataset)](https://affective-meld.github.io/)**, a large-scale dataset of multi-party conversations containing textual dialogues, audio, and visual information.

## ☁️ Cloud Integration & Training

The training pipeline is fully orchestrated with **AWS SageMaker**.

- Distributed training support on `ml.g5.xlarge` GPU instances.
- S3 integration for seamless dataset loading and artifact storage.
- Real-time monitoring with **TensorBoard**.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch & Torchaudio
- Transformers (Hugging Face)
- OpenCV
- AWS CLI & SageMaker Python SDK (if running on AWS)

### Installation

1. Clone the repository.
2. Install the necessary requirements:
   ```bash
   pip install -r training/requirement.txt
   ```
3. Ensure `ffmpeg` is installed on your system for audio extraction. (You can use the provided `training/install_ffmpeg.py`).

### Training Locally or on AWS

To initiate a SageMaker training job, run:

```bash
python train_sagemaker.py
```

To run training locally, use:

```bash
python training/train.py
```

_(Make sure the MELD dataset is correctly structured in the `dataset` folder.)_

## 📁 Project Structure

- `training/`: Contains all model architectures (`models.py`), dataset loaders (`meld_dataset.py`), and the training loop script.
- `deployment/`: Contains scripts for deploying the trained model as an AWS SageMaker endpoint.
- `train_sagemaker.py`: Entry point for launching cloud-based training jobs.

---
