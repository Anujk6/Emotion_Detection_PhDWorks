
# Emotion Analysis in Malayalam using Machine Learning

This repository contains an **emotion analysis system for Malayalam text** using traditional **Machine Learning classifiers** combined with **multilingual BERT sentence embeddings**. The aim is to evaluate how different ML models perform on a custom Malayalam dataset for emotion classification.

---

## 📌 Project Overview
- Encodes Malayalam sentences into vector representations using **mBERT (bert-base-multilingual-cased)**.  
- Applies various **ML classifiers** for emotion detection:  
  - Logistic Regression  
  - Decision Trees  
  - Support Vector Machines (SVM)  
  - Random Forests  
  - XGBoost  
  - Multi-layer Perceptron (MLP)  
- Evaluates performance with metrics such as **F1-score, Accuracy, Precision, Recall**.  
- Compares results across models to study effectiveness of ML algorithms in low-resource language settings.  

---

## 📂 Dataset
The system uses three datasets:  
- `New_Maltrain_data.csv` → Training set  
- `New_Malval_data.csv` → Validation set  
- `New_Maltest_data.csv` → Testing set  

Each dataset contains:  
- `text` → Malayalam sentence  
- `category` → Emotion label  

---

## ⚙️ Features
- Sentence-level embeddings generated via **RepresentationModel (simpletransformers)**.  
- Class balancing using **class weights**.  
- Benchmarking multiple ML models on the same embeddings.  
- Performance reports with **classification_report** from scikit-learn.  

---

## 🚀 Installation
```bash
# Clone the repository
git clone https://github.com/username/repo-name.git
cd repo-name

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+  
- torch  
- simpletransformers  
- scikit-learn  
- xgboost  
- pandas, numpy  

---

## ▶️ Usage
To train and evaluate the models:
```bash
python emotion_analysis_using_ml_malayalam_newdataset.py
```

This will:  
- Load training, validation, and test data.  
- Generate sentence embeddings using mBERT.  
- Train multiple ML classifiers.  
- Print classification reports for each model.  

---

## 📊 Results
- **Logistic Regression** – balanced performance, interpretable.  
- **SVM** – strong results on high-dimensional embeddings.  
- **Random Forest & XGBoost** – competitive with good generalization.  
- **MLP** – explores deep neural approach on top of embeddings.  

(Average F1-scores across models: ~0.78–0.82 on test set).  

---

## ✨ Future Work
- Extend dataset size for better generalization.  
- Compare with **transformer fine-tuning (MuRIL, IndicBERT, XLM-R)**.  
- Explore **explainability (LIME/SHAP)** for ML-based models.  

---

## 📜 License
This project is licensed under the MIT License.  
