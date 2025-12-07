# Marathi Dialect Classification: TF-IDF + Logistic Regression

This project classifies Marathi sentences into dialect regions using a simple machine learning model. It is optimized for low-resource environments (no GPU required).

---

## 📁 Project Structure

```
marathi-dialect-asr/
├── train_data.csv
├── test_data.csv
├── script.py          # Main script for training + evaluation
└── README.md          # (this file)
```

---

## ⚙️ Requirements

Install the required libraries with:

```bash
pip install pandas scikit-learn
```

Tested with:

* Python 3.10+
* scikit-learn 1.3+
* pandas 1.4+

---

## ▶️ How to Run

From the folder containing `script.py`, run:

```bash
python script.py
```

Make sure `train_data.csv` and `test_data.csv` are in the same directory.

---

## 🧠 What the Script Does

1. **Loads data** from `train_data.csv` and `test_data.csv`, expecting columns:

   * `sentence` (Marathi in Devanagari)
   * `label` (one of: D1, D2, D3, D4)

2. **Vectorizes** text using TF-IDF (unigrams only)

3. **Trains** a `LogisticRegression` model on the TF-IDF features

4. **Evaluates** on test data:

   * Accuracy
   * Precision, Recall, F1 for each dialect
   * Confusion Matrix

5. **Predicts** on example sentences provided at the end of the script

---

## 📊 Output Example

After running, you'll see output like:

```
Test Accuracy: 0.61
Classification Report:
  D1: Precision 0.76, Recall 0.70
  D2: ...
Confusion Matrix:
 [[394  53  36  76]...]
```

---

## 🧾 Labels Mapping

| Label | Dialect Region                  |
| ----- | ------------------------------- |
| D1    | Southern Konkan (Sindhudurg)    |
| D2    | Northern Konkan (Nashik, Dhule) |
| D3    | Standard Marathi (Pune Rural)   |
| D4    | Varhadi (Nagpur Rural)          |

---

## 📌 Notes

* You can modify the TF-IDF `ngram_range` to (1,2) to include bigrams
* Model and vectorizer are **not saved** yet; you can add `joblib.dump()` to store them
* Designed to be lightweight and runnable on laptops

---

