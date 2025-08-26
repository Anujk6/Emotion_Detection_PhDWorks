
# Emotion Detection using MuRIL with Cross-Validation

This repository contains an **emotion detection system for Malayalam text** using the **MuRIL (Multilingual Representations for Indian Languages)** transformer model. The system leverages **cross-validation techniques** to ensure robust evaluation and generalization across multiple folds.

---

## 📌 Project Overview
- Implements emotion classification for Malayalam sentences.  
- Uses the **MuRIL-base-cased model** fine-tuned on a custom dataset.  
- Handles **imbalanced datasets** with oversampling and under-sampling strategies.  
- Evaluates performance using **F1-score, accuracy, and classification reports**.  
- Employs **5-fold Stratified Cross-Validation** for reliable performance estimation.  

---

## 📂 Dataset
The project uses three datasets:
- `New_Maltrain_data.csv` → Training set  
- `New_Malval_data.csv` → Validation set  
- `New_Maltest_data.csv` → Testing set  

Each dataset contains:
- `text` → Malayalam sentence  
- `category` → Emotion label  

---

## ⚙️ Features
- Preprocessing with label encoding for categorical emotions.  
- Balancing imbalanced data using **oversampling** and **over-under sampling**.  
- Fine-tuning MuRIL with **simpletransformers**.  
- Evaluation metrics:  
  - **Macro F1 Score**  
  - **Accuracy**  
  - **Misclassification analysis with bar charts**  

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
- transformers==4.31.0  
- simpletransformers  
- scikit-learn  
- matplotlib  
- pandas  

---

## ▶️ Usage
To train and evaluate the model:
```bash
python murilbased_emotion_crossvalidation.py
```

The script will:
- Load training, validation, and test datasets  
- Perform dataset balancing  
- Train MuRIL model  
- Run cross-validation  
- Save trained model and tokenizer to Google Drive  

---

## 📊 Results
- Achieved **average cross-validation accuracy ≈ 85%**  
- Best folds reached accuracy above **86%**  
- Error analysis shows class-wise misclassification trends  

---

## 📈 Example Output
- Macro F1 Score: ~0.84  
- Accuracy: ~0.85  
- Misclassification distribution chart  

---

## 📦 Model Saving
The trained model and tokenizer are saved to:
```
/content/drive/My Drive/Emotion_Model_Directory
```
