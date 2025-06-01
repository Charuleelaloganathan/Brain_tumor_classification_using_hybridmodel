# Brain_tumor_classification_using_hybridmodel

🧠 Brain Tumor Classification using Hybrid Deep Learning (ResNet50 + ViT + GCN)
This project aims to classify brain tumors from MRI images using a hybrid deep learning model combining ResNet50, Vision Transformer (ViT), and Graph Convolutional Networks (GCN). The system demonstrates enhanced performance by integrating feature extraction, attention mechanisms, and structural relationships from graph data.

🔬 Project Highlights
🔍 Input: MRI images of brain tumors

🧠 Output: Tumor type classification (e.g., Meningioma, Glioma, Pituitary)

🧩 Model Architecture:

ResNet50: For low- and mid-level feature extraction

ViT: For attention-based representation of image features

GCN: For leveraging inter-feature dependencies using graph structure

🚀 How It Works
Preprocessing:

Images are resized and normalized.

Augmentation applied to increase robustness.

Phase 1 - ResNet50:

Extracts spatial features.

Trained independently to establish baseline.

Phase 2 - ResNet50 + ViT:

Combines spatial features with global attention using Vision Transformer.

Phase 3 - ResNet50 + ViT + GCN:

Constructs a graph of features and uses GCN to model complex interdependencies.

Final classification is based on combined feature embeddings.

📊 Performance
Model	Accuracy	Precision	Recall	F1 Score
ResNet50	~88%	87%	88%	87%
ResNet50 + ViT	~91%	90%	91%	91%
ResNet50 + ViT + GCN	~93%	92%	93%	92%

Demo
Upload an MRI image

View step-by-step visualization:

Original ➜ 2. Preprocessed ➜ 3. ResNet Features ➜ 4. ViT Features ➜ 5. GCN Prediction

🛠️ Tech Stack
Python, TensorFlow / PyTorch

NumPy, OpenCV, Matplotlib

Colab / Jupyter Notebook

Scikit-learn for evaluation

📚 Dataset
Source: Kaggle or Medical open datasets (Sartaj Bhuvaji's dataset)

Contains labeled MRI scans for multiple tumor types

🧠 Future Work
Integrate real-time detection with webcam input

Deploy as a web app using Flask or Streamlit

Apply to multi-modal datasets (CT, PET)

🤝 Acknowledgements
Inspired by research on hybrid deep learning and medical imaging.

Thanks to open-source contributors and dataset providers.

Displays final prediction and 2D scatter plot of embeddings
