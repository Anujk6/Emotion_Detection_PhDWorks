
# Emotion Detection in Malayalam using FastText Skip-gram + LIME

This repository contains an **emotion detection system for Malayalam text**. The project combines **custom FastText skip-gram embeddings** with an **LSTM model** and integrates **LIME (Local Interpretable Model-agnostic Explanations)** to provide interpretability at the word level.

---

## 📌 Project Overview
- Builds **custom embeddings** using FastText (Skip-gram) trained on Malayalam data.  
- Applies **IndicNLP library** for tokenization.  
- Trains an **LSTM classifier** for emotion detection.  
- Evaluates using accuracy, precision, recall, and F1-score.  
- Uses **LIME** for explainability: highlights which words influence predictions most.  

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
- Word-level tokenization using **IndicNLP tokenizer**.  
- FastText Skip-gram embeddings trained with embedding dimension = 100.  
- LSTM-based classification network with dropout regularization.  
- Performance metrics:  
  - **Accuracy**  
  - **Precision (weighted)**  
  - **Recall (weighted)**  
  - **F1-score (weighted)**  
- Explainability with **LIME word-level explanations**.  

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
- fasttext  
- indic-nlp-library  
- tensorflow / keras  
- scikit-learn  
- lime  
- pandas, numpy, tqdm  

---

## ▶️ Usage
To train and evaluate the model:
```bash
python ft_skipgram_lime_custom_embeddings.py
```

This will:
- Train FastText Skip-gram embeddings.  
- Train LSTM model with embeddings.  
- Evaluate performance on test set.  
- Generate **LIME explanations** for interpretability.  

---

## 📊 Results
- **Test Accuracy:** ~84%  
- **Precision, Recall, F1-score** evaluated per class.  
- LIME provides word-level interpretability for test samples.  

---

## 📈 Example LIME Output
```
Text: "കുറ്റബോധത്തിൽ നിന്ന് രക്ഷപ്പെടാൻ"
Predicted Emotion: Fear
LIME Explanation:
[('കുറ്റബോധത്തിൽ', -0.08), ('രക്ഷപ്പെടാൻ', -0.03), ('നിന്ന്', -0.02)]
```

---

## 📦 Model Saving
- FastText embeddings: `/content/drive/MyDrive/fasttext_malayalam_skip.bin`  
- Trained LSTM model: `/content/drive/MyDrive/LSTM_ED_MALAYALAM_FT_SKIP.h5`  
- LIME results: `/content/drive/MyDrive/lime_explanations_word_level_updated.csv`  

---

## ✨ Future Work
- Expand training dataset for better generalization.  
- Compare performance with **Transformer-based models** (MuRIL, mBERT, XLM-R).  
- Enhance interpretability with **SHAP** in addition to LIME.  

---

## 📜 License
This project is licensed under the MIT License.
