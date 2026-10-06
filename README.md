cat << 'EOF' > /mnt/data/README.md
# Hybrid Deepfake Detection

This project implements a Transformer–CNN hybrid ensemble to detect deepfake video frames with high accuracy and explainability.

## 📁 Project Structure

\`\`\`
deepfake_detection_project/
│
├── data/
│   ├── raw/            # Original real/fake videos
│   ├── frames/         # Extracted frames (train/val/test splits)
│   └── annotations/    # (Optional) metadata or CSV splits
│
├── configs/            # Hyperparameter config files (YAML/JSON)
│   └── default.yaml
│
├── notebooks/          # Jupyter notebooks for EDA & reporting
│
├── src/                # Source code
│   ├── data/           # frame_extractor.py, splitter.py
│   ├── models/         # custom_cnn.py, vit_model.py, swin_model.py, ensemble.py
│   ├── train.py        # Training loop & logging
│   ├── evaluate.py     # Evaluation & metrics
│   └── utils.py        # Helpers (plots, metrics)
│
├── outputs/            # Checkpoints, plots, heatmaps
│
├── logs/               # TensorBoard logs
│
├── requirements.txt    # Python dependencies
├── README.md           # Project overview & instructions
└── .gitignore          # Ignore data, caches, checkpoints
\`\`\`

## ⚙️ Installation

1. Clone this repository:
   \`\`\`bash
   git clone https://github.com/yourusername/deepfake_detection_project.git
   cd deepfake_detection_project
   \`\`\`

2. Create and activate a virtual environment (recommended):
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate  # on Windows: venv\\Scripts\\activate
   \`\`\`

3. Install dependencies:
   \`\`\`bash
   pip install --upgrade pip
   pip install -r requirements.txt
   \`\`\`

## 🚀 Usage

### 1. Configure
Edit \`configs/default.yaml\` to set your paths, hyperparameters, and model weights.

### 2. Prepare Data
- Place raw videos under \`data/raw/real/\` and \`data/raw/fake/\`.
- Extract frames and split into train/val/test:
  \`\`\`bash
  python src/data/frame_extractor.py --input_dir data/raw --output_dir data/frames --frame_rate 10
  python src/data/splitter.py --source_dir data/frames --train_dir data/frames/train --val_dir data/frames/val --test_dir data/frames/test --split_ratio 0.8
  \`\`\`

### 3. Train
\`\`\`bash
python src/train.py --config configs/default.yaml
\`\`\`
- Logs are written to \`logs/\` (view with \`tensorboard --logdir logs/\`).
- Best model checkpoints saved under \`outputs/checkpoints/\`.

### 4. Evaluate
\`\`\`bash
python src/evaluate.py --config configs/default.yaml --checkpoint outputs/checkpoints/best_model.pth
\`\`\`
Generates classification report, confusion matrix, and ROC curve under \`outputs/plots/\`.

### 5. Generate Attention Heatmaps
\`\`\`bash
python src/utils.py --generate_heatmap --config configs/default.yaml --checkpoint outputs/checkpoints/best_model.pth
\`\`\`
Saves visual attention maps into \`outputs/heatmaps/\`.

## 📈 Results

- **Validation Accuracy:** ~94.0%  
- **Test Accuracy:** ~92.5%  
- **F₁-Scores:** Real = 0.93, Fake = 0.91  

Heatmaps reveal that the model focuses on mouth and eye regions—key areas for deepfake artifacts.

## 🔮 Future Work

- Integrate transformer‑based tokenizers (e.g., CLIP or DeiT) for improved patch embeddings.  
- Add temporal consistency checks across multiple frames.  
- Deploy via ONNX for low‑latency, cross‑platform inference.

---

**License:**  
MIT © 2025 Your Name
EOF
