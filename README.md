# W-Conv-Informer

#  W-Conv-Informer: A Wavelet-Augmented Informer for Time Series Forecasting

This repository contains a modified version of the [Informer](https://github.com/zhouhaoyi/Informer2020) model for **long sequence time-series forecasting**. The key innovations of this architecture focus on addressing the **temporal loss problem** of transformers when applied to time series data.

---

##  Key Modifications

### 1.  Wavelet Transform Preprocessing
- Before entering the encoder, raw input sequences are passed through a **Wavelet Transform** based on the **first derivative of the Gaussian (Mexican Hat / Ricker wavelet)**.
- This transformation emphasizes local temporal variations, enabling the model to better capture short-term trends and abrupt changes in the signal.

### 2.  Conv1D Stack Before Attention
- A series of **N Conv1D layers** is applied **before the attention mechanism** in each encoder layer.
- These convolutional layers extract local dependencies and enhance hierarchical temporal features.

### 3. Residual Connection with Attention Input
- A **residual skip connection** is applied between the **output of the last Conv1D layer** and the **attention sub-layer**.
- This ensures that **local temporal information** is preserved and propagated, addressing the **temporal information loss** that typically occurs due to global attention pooling.

---

##  Model Architecture Overview

Input Series
│
[ Wavelet Transform ]
│
→ Transformed Series
│
→ Positional Encoding
│
→ Encoder Layer:
├─ Conv1D → Conv1D → ... → Conv1D
└─ Residual Connection to ↓
└─ ProbSparse Attention
↓
FeedForward + Norm
↓
(standard Informer decoder)


Transformers, while powerful in NLP and vision, often suffer from **temporal information loss** when used in time-series forecasting — especially due to:
- Fixed positional encodings,
- Lack of explicit local context modeling,
- Aggressive sequence compression (e.g., Informer's distillation).

This architecture enhances the **temporal locality awareness** of the Informer while preserving its long-range modeling capabilities.

---

##  Use Cases

- Cryptocurrency forecasting (e.g., Bitcoin price)
- Financial time-series analysis
- Energy consumption prediction
- Traffic and weather modeling

---

##  Experiments

This model was benchmarked against the original Informer and LSTM baselines on hourly and daily datasets. The W-Conv-Informer showed:
- Improved generalization
- Lower MSE
- Better training stability
- Reduced overfitting

---

##  Installation

```bash
git clone https://github.com/CharisMichailidis/W-Conv-Informer.git
cd W-Conv-Informer
pip install -r requirements.txt

