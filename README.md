
<img src = "assets/cover page.png" width = 700>

# 🎬 Movie reviews Sentiment Analysis with DistilBERT

This project demonstrates **fine-tuning a pretrained Transformer model** (`distilbert-base-uncased`) on the **IMDB movie reviews dataset** for **binary sentiment classification (positive / negative)**.

The goal of this repository is **learning-oriented but industry-aligned**:

* Use a real-world dataset (IMDB)
* Use Hugging Face Transformers + Datasets
* Follow the same fine-tuning workflow used in production ML teams

---

## 📌 What this project covers

* Loading a large NLP dataset using **Hugging Face Datasets**
* Tokenizing text with a pretrained **BERT-style tokenizer**
* Fine-tuning a pretrained **DistilBERT** model
* Training with PyTorch `DataLoader`
* Evaluating accuracy on a held-out test set
* Running inference on custom sentences

---

## 🧠 Model

* **Base model:** `distilbert-base-uncased`
* **Task:** Sequence Classification
* **Labels:**

  * `0` → Negative review
  * `1` → Positive review

The classification head is **randomly initialized** and then **fine-tuned on IMDB**.

---

## 📊 Dataset

* **Dataset:** IMDB Movie Reviews
* **Source:** Hugging Face Datasets
* **Size:**

  * Train: 25,000 reviews
  * Test: 25,000 reviews

For faster experimentation, a **subset of the dataset** is used during training.

Each sample contains:

```json
{
  "text": "movie review text",
  "label": 0 or 1
}
```

---

## 🏗️ Project Structure

```text
.
├── imdb_finetuning.ipynb   # Main training notebook
├── README.md               # Project documentation
```

> The notebook-first approach allows easy debugging and experimentation. The logic can later be migrated to a `.py` training script.

---

## ⚙️ Training Pipeline

1. Load IMDB dataset
2. Tokenize text (padding + truncation)
3. Convert dataset to PyTorch tensors
4. Create `DataLoader`s
5. Fine-tune DistilBERT using AdamW
6. Evaluate accuracy on test data

Loss is computed automatically by Hugging Face when labels are provided.

---

## 🚀 How to Run

### 1️⃣ Install dependencies

```bash
pip install torch transformers datasets tqdm
```

### 2️⃣ Open the notebook

```bash
jupyter notebook imdb_finetuning.ipynb
```

### 3️⃣ (Optional) Use GPU

If CUDA is available, the notebook automatically runs on GPU.

---

## 🧪 Example Inference

After training, the model can be tested on custom sentences:

```python
texts = [
    "This movie was absolutely amazing",
    "I regret watching this film"
]
```

The model outputs predicted sentiment labels for each sentence.

---

## 📈 Results

* Accuracy improves significantly after fine-tuning
* The model learns sentiment even though the base model was not sentiment-trained

(Exact accuracy depends on dataset subset size and number of epochs.)

---

## 🔮 Next Steps

Possible extensions:

* Convert notebook to a production-ready `.py` script
* Add learning rate scheduler
* Freeze base model layers
* Save and reload the fine-tuned model
* Dockerize the training environment

---

## 📚 References

* Hugging Face Transformers
* Hugging Face Datasets
* DistilBERT: Smaller, Faster, Cheaper BERT

---

## ✨ Motivation

This project is meant to bridge the gap between **theory (Transformers)** and **real-world ML workflows**, using clean, minimal, and reproducible code.
