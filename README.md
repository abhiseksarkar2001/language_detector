# 🌐 Language Detector

A robust and scalable language identification system that supports up to **200 languages** using FastText and **n-gram-based classification models**. Designed for accurate language prediction from short and long text samples.

---

## 📁 Repository Structure

```
language_detector/
├── fast_text_30Lang.ipynb        # FastText model for 30 languages
├── fast_text_for_200Lang.ipynb   # FastText model for 200 languages
├── n-gram.ipynb                  # n-gram + Logistic Regression model
├── Links and File Structure.pdf  # Overview with Google Drive links
└── README.md                     # Project documentation
```

---

## 🔍 Features

* 🔠 **30 & 200 language support** using FastText.
* 🧠 **n-gram based Logistic Regression** classifier for a compact and effective language detector.
* 🧪 Easy-to-use inference for testing on new text inputs.
* 📈 High accuracy achieved across both models.
* 🛠️ Pre-trained models and vectorizers provided via [Google Drive](#🔗-links).

---

## 📦 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/abhiseksarkar2001/language_detector.git
   cd language_detector
   ```

2. Install dependencies:

   ```bash
   pip install fasttext scikit-learn joblib tqdm
   ```

---

## 🚀 How to Use

### 🔤 1. FastText Model (30 or 200 Languages)

#### Load the Pre-trained Model

```python
import fasttext

# For 30-language model
model = fasttext.load_model("lang_detect_model_30lang.bin")

# For 200-language model
model = fasttext.load_model("lang_detect_model_200lang.bin")
```

#### Predict Language

```python
text = "Bonjour tout le monde"
pred = model.predict(text)
language = pred[0][0].replace('__label__', '')
confidence = pred[1][0]

print(f"Detected Language: {language}, Confidence: {confidence:.2f}")
```

---

### 🧩 2. n-gram Model (Logistic Regression)

#### Load Model & Vectorizer

```python
import joblib

clf = joblib.load("clf.joblib")
vectorizer = joblib.load("vectorizer.joblib")
```

#### Predict Language

```python
text = "Ceci est un test"
features = vectorizer.transform([text])
pred = clf.predict(features)

print(f"Predicted Language: {pred[0]}")
```

---

## 📊 Accuracy

| Model Type        | Language Coverage | Accuracy |
| ----------------- | ----------------- | -------- |
| FastText          | 30 Languages      | \~99.83%  |
| FastText          | 200 Languages     | \~98.45%  |
| n-gram            | 30 Languages      | \~99.45%  |

Accuracy may vary based on dataset and text length.

---

## 📂 Datasets & Resources

All models, training data, and pre-trained files are available in the shared Google Drive folders:

* 🔗 [FastText Models & Data](https://drive.google.com/drive/folders/1S3Y9tya-zzIGq18h1KiOVYVZuOw_lRV1?usp=sharing)
* 🔗 [n-gram Model & Data](https://drive.google.com/drive/folders/1s__vNvziy-RGzzSSn-3dRPTJ2vWJ3mvh?usp=sharing)

Each folder contains:

* `.bin` (FastText models)
* `.joblib` (scikit-learn model & vectorizer)
* `train.txt` (formatted FastText data)
* `.ipynb` (training & evaluation notebooks)

---

## 📌 Highlights

* Supports text classification in **low-resource languages**.
* Suitable for **language-aware NLP applications** (translation, sentiment analysis, search).
* Efficient training and prediction times.

---

## 🧪 Evaluation Tips

For benchmarking:

* Use text of at least 20 characters for higher accuracy.
* Include diverse samples per language.
* Evaluate confidence scores for uncertain predictions.

---

## 📃 License

This repository currently has **no license** specified. For collaboration or reuse, please contact the author.

---

## 🙌 Author

**Abhisek Sarkar**
🔗 [GitHub Profile](https://github.com/abhiseksarkar2001)

---
