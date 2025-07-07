# RNN-LastMile
# 📦 Last-Mile Delivery ETA Prediction using LSTM

This repository contains PyTorch-based implementations of deep-learning models for **predicting Estimated Time of Arrival (ETA)** in last-mile delivery from sequential stop data. Two notebook variants are included:

1. **`LSTM_LastMile.ipynb`** – baseline LSTM model on synthetic delivery-sequence data  
2. **`LSTM_MLP.ipynb`** – enhanced model that combines LSTM sequence encoding with a feed-forward (MLP) head for the final stop’s features

---

## 🚀 Overview of the Notebooks

### `LSTM_LastMile.ipynb`
* Uses a single LSTM to encode the entire sequence of prior stops  
* Trains on simulated multivariate time-series inputs (each stop has several numeric features)  
* Outputs a continuous ETA prediction (in hours)  

### `LSTM_MLP.ipynb`
* Splits each record into  
  • a sequence of previous stops (LSTM branch)  
  • the final stop’s feature vector (MLP branch)  
* Concatenates both representations for a richer context before the final regression layer  
* Shows how combining sequential and point-in-time information can improve accuracy  

---

## 🧠 Model Architectures

Baseline LSTM (`LSTM_LastMile`)

    [Delivery Sequence] → [LSTM Layer] → [Dense Layers] → ETA (regression)

LSTM + Last-Stop MLP (`LSTM_MLP`)

    [Previous Stops] ──► [LSTM Layer]
                         │
    [Last Stop Features] ─► [MLP Head]
                         │
                      Concatenate
                         │
                    [Dense Layers] → ETA (regression)

---

## 📊 Dataset

Both notebooks generate **synthetic** data with PyTorch:
* Each sequence contains multiple delivery stops with numeric features (e.g., time since dispatch, distance traveled, traffic index).  
* The target variable is the actual delivery time at the last stop.

---

## 🛠 Requirements

* Python 3.8+  
* PyTorch  
* pandas  
* numpy  

Install dependencies:

    pip install torch pandas numpy

---

## 🏃‍♂️ Quick Start (Google Colab)

Run the notebooks in one click:

* Baseline LSTM – <https://colab.research.google.com/github/aminKMT/RNN-LastMile/blob/main/LSTM_LastMile.ipynb>  
* LSTM + MLP – <https://colab.research.google.com/github/aminKMT/RNN-LastMile/blob/main/LSTM_MLP.ipynb>

---

## 📈 Outputs

* Trains for 100 epochs with MSE loss and the Adam optimizer  
* Prints a predicted ETA for an unseen delivery example  
* Demonstrates how LSTM-based models capture temporal dependencies in logistics data

---

## 📚 Citation / Attribution

If you use or adapt this work in research, projects, or teaching, please cite or reference this repository.

---

## ✍️ Author

**Amin Keramati**  
Assistant Professor of Supply Chain & Data Science, Widener University  
GitHub: <https://github.com/aminKMT>
