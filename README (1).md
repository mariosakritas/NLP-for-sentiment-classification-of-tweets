# NLP for Sentiment Classification of Tweets

This project explores the use of Recurrent Neural Networks (RNNs) for sentiment classification on Twitter data. It combines natural language processing (NLP) techniques with a custom-built deep learning architecture to categorize tweets based on sentiment.

---

## 🧠 Project Overview

Tweets are short, noisy texts rich in emotional content. The aim of this project is to build an RNN-based model capable of classifying the sentiment of tweets into predefined categories (e.g., positive, neutral, negative). 

The architecture used is a character-level RNN trained from scratch — without relying on pretrained embeddings — to better capture informal and slang-heavy language typical of social media.

---

## 🔧 Model Architecture

The core of the model is an RNN that takes character sequences as input and processes them through hidden layers before outputting sentiment predictions via a softmax layer.

![RNN Architecture](./NMA_DEEP/RNN_Archit.png)

---

## 📊 Model Performance

The model was evaluated on a held-out validation set. Below is a visualization of its performance over training epochs.

![Model Performance](./NMA_DEEP/model_performance.png)

We also explored the final softmax outputs for positive sentiment examples:

![Softmax Positive Associations](./NMA_DEEP/positive_associations_softmax.png)

---

## 📁 Project Structure

```
NMA_DEEP/
│
├── NMA_NOTEBOOKS/                 # Jupyter notebooks and Python training script
│   └── RNN_sentiment_classification.py
│
├── nltk_data/                     # NLTK data for preprocessing
│
├── model_performance.png         # Model training performance figure
├── RNN_Archit.png                # RNN architecture visualization
├── positive_associations_softmax.png  # Softmax output for positive predictions
└── merged_training.pkl           # Serialized training data
```

---

## 📚 Dependencies

- Python ≥ 3.8  
- NumPy  
- PyTorch  
- NLTK  
- Matplotlib  
- scikit-learn  

To install requirements:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

From the `NMA_NOTEBOOKS` folder, you can run:

```bash
python RNN_sentiment_classification.py
```

Or explore training interactively in the `RNN_attempt.ipynb` notebook.

---

## 🧠 Notes

- Preprocessing includes tokenization, lowercasing, and character-level representation.
- This project emphasizes learning from raw text, without feature engineering.
- The training data was preprocessed and serialized for speed in `merged_training.pkl`.

---

## 📬 Contact

For questions, open an issue or reach out to [Marios Akritas](https://github.com/mariosakritas).

