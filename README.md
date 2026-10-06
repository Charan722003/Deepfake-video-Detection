# Deepfake Video Detection using Transformer-CNN Hybrid Model

## 📌 Project Overview

Deepfake videos are manipulated or synthetically generated videos that can realistically replace or alter a person's identity, facial expressions, or other visual characteristics. Detecting such manipulated content is an important challenge in computer vision, digital forensics, and media security.

This project presents a deep learning-based **Deepfake Video Detection System** using a combination of **Convolutional Neural Networks (CNN), Vision Transformer (ViT), and Swin Transformer** architectures. The system extracts visual features from video frames and classifies them as **Real** or **Fake**.

The project also incorporates **attention-based visualization / Grad-CAM** techniques to improve model interpretability by highlighting the regions of a frame that influence the prediction. A graphical user interface (GUI) is also included for testing deepfake videos.

---

## 🎯 Objectives

The main objectives of this project are:

- To develop an automated system for detecting deepfake videos.
- To extract meaningful visual features from video frames.
- To implement CNN-based deepfake detection.
- To implement Vision Transformer (ViT) for image-based deepfake detection.
- To implement Swin Transformer for hierarchical visual feature extraction.
- To combine Transformer and CNN-based approaches for improved detection.
- To visualize important regions contributing to model predictions.
- To provide a simple GUI for deepfake video prediction.
- To evaluate the models using real and manipulated video datasets.

---

## 🧠 Models Used

### 1. Custom CNN

A Convolutional Neural Network is used to extract spatial features from video frames. CNN layers learn important visual patterns that can help distinguish between real and manipulated facial regions.

### 2. Vision Transformer (ViT)

Vision Transformer divides an image into patches and uses self-attention mechanisms to learn relationships between different regions of the image. ViT is used to capture global visual information from video frames.

### 3. Swin Transformer

Swin Transformer uses a hierarchical architecture and shifted-window attention mechanism to efficiently learn both local and global visual features.

### 4. Transformer-CNN Hybrid / Ensemble Approach

The project combines CNN and Transformer-based approaches to utilize their complementary strengths. CNN focuses on local spatial features, while Transformer architectures capture broader relationships between image regions.

---

## 📂 Datasets

The project was trained and experimented with two deepfake datasets.

### UADFV Dataset

The **UADFV (Uncompressed Video-based DeepFake)** dataset contains real and manipulated videos and is used for developing and evaluating deepfake detection models.

The dataset was used for frame extraction, model training, validation, and testing.

### Celeb-DF v2 Dataset

The **Celeb-DF v2** dataset contains high-quality real and deepfake videos. It provides more challenging manipulated videos and was used to evaluate the performance and generalization of the deepfake detection models.

Using both UADFV and Celeb-DF v2 provides experimentation across datasets with different characteristics and manipulation qualities.

---

## 🔄 Project Workflow

```text
Input Video
     ↓
Video Frame Extraction
     ↓
Image Preprocessing
     ↓
Feature Extraction
     ↓
 ┌───────────────┬────────────────┬─────────────────┐
 │     CNN       │      ViT       │  Swin Transformer│
 └───────────────┴────────────────┴─────────────────┘
     ↓
Feature Learning / Ensemble
     ↓
Real / Fake Classification
     ↓
Prediction
     ↓
Grad-CAM / Attention Visualization
     ↓
GUI Output
```
🛠️ Technologies Used
- Python
- PyTorch
- Torchvision
- Hugging Face Transformers
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Pillow
- TensorBoard
- Jupyter Notebook
- Tkinter
- Grad-CAM / Attention Visualization

📁 Project Structure

Deepfake-Video-Detection/
│

├── data/
│   └── .gitkeep
│

├── frames/
│   └── .gitkeep
│

├── models/
│   └── .gitkeep
│

├── notebook/
│   └── Hybrid_model_UADFV.ipynb
│

├── Documentation/
│   ├── Final_Documentation.pdf
│   └── Datasets.pdf
│

├── requirements.txt

├── .gitignore

└── README.md

The data/ and frames/ folders are kept as placeholders. The original datasets and extracted video frames are not included in this repository because of their large size.

📊 Model Files

During experimentation, trained model weights were generated for the different architectures:
- cnn_model.pth
- vit_model.pth
- swin_model.pth

The trained model files are large and are therefore not included directly in this GitHub repository.

Approximate model sizes:
- CNN Model          ~196 MB
- Swin Transformer   ~105 MB
- ViT Model          ~327 MB

🔍 Explainability

To improve the interpretability of the deepfake detection system, the project includes Grad-CAM / attention-based visualization.
The visualization highlights important regions of the input frame that contribute to the model's prediction. This helps understand which facial or visual regions the model considers important when classifying a frame as real or fake.

🖥️ Graphical User Interface

A Tkinter-based GUI is included in the project for testing the trained deepfake detection models.
The GUI allows a user to provide a video and obtain a prediction from the trained model.
The system can display the classification result and visualization generated from the model.

⚙️ Installation

Clone the repository:
git clone https://github.com/yourusername/Deepfake-Video-Detection.git

Navigate into the project:
cd Deepfake-Video-Detection

Create a virtual environment:
python -m venv venv

Activate the environment on Windows:
venv\Scripts\activate

Install the required libraries:
pip install -r requirements.txt

📦 Requirements

The main Python libraries used in this project include:
torch
torchvision
transformers
scikit-learn
numpy
matplotlib
tqdm
opencv-python
Pillow
tensorboard

📓 Running the Project

The main experimentation and model implementation are available in the Jupyter Notebook:
notebook/Hybrid_model.ipynb

Open the notebook using Jupyter Notebook or JupyterLab:
jupyter notebook

Then navigate to:notebook/Hybrid_model.ipynb

Before running the notebook, place the required dataset files in the appropriate local data/ directory.

The notebook performs the major stages of the project, including:

1. Dataset loading
2. Video frame extraction
3. Image preprocessing
4. CNN model training
5. Vision Transformer training
6. Swin Transformer experimentation
7. Model evaluation
8. Prediction
9. Attention / Grad-CAM visualization

📈 Results

The project was experimented with using both UADFV and Celeb-DF v2 datasets.
For the UADFV dataset, the CNN and Transformer-based models achieved strong classification performance during experimentation.
The project also achieved approximately 85% training accuracy and 95% validation accuracy during experimentation with the Celeb-DF v2 dataset.
An overall project performance of approximately 86% accuracy was used as the reported result for the developed deepfake detection system.

Note: Model performance can vary depending on dataset split, preprocessing, number of frames, training configuration, and hardware.

📌 Key Features

- Deepfake video detection
- CNN-based feature extraction
- Vision Transformer (ViT)
- Swin Transformer
- Transformer-CNN hybrid approach
- UADFV dataset experimentation
- Celeb-DF v2 dataset experimentation
- Video frame extraction
- Real/Fake classification
- Grad-CAM visualization
- Attention-based visualization
- TensorBoard training monitoring
- Tkinter GUI
- PyTorch implementation

🔮 Future Work

Future improvements can include:

- Temporal modeling of consecutive video frames.
- Integration of advanced video Transformers.
- Improved cross-dataset generalization.
- Real-time deepfake detection.
- Deployment as a web application.
- ONNX-based optimized inference.
- Integration of larger and more diverse deepfake datasets.
- Improved video-level classification using temporal features.

👨‍💻 Author

Vasanthala Devi Naga Charan

B.Tech – Artificial Intelligence and Data Science

📄 Documentation

Project documentation and dataset-related information are available in the Documentation/ directory.





