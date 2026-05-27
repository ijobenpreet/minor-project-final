# 🚀 Sequence-Based IoT Botnet Detection with Cross-Dataset Generalization

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge\&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange?style=for-the-badge\&logo=tensorflow)
![Research](https://img.shields.io/badge/Research-IEEE%20Ready-success?style=for-the-badge)
![IoT](https://img.shields.io/badge/Domain-IoT%20Security-red?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-N--BaIoT%20%2B%20CICIoT2023-purple?style=for-the-badge)

### 🔐 Deep Learning Framework for Obfuscation-Resilient IoT Malware Detection

*A Research-Oriented CNN-LSTM Framework for Temporal Behavioral Analysis of IoT Botnet Traffic*

</div>

---

# 🌟 Project Overview

This project proposes a **sequence-based deep learning framework** for detecting IoT botnet attacks using temporal behavioral analysis.

The system combines:

* 🧠 CNN + LSTM Hybrid Deep Learning
* 📊 Cross-Dataset Generalization
* 🔐 Obfuscation-Resilient Detection
* 🌐 Real IoT Traffic Analysis
* 📈 Research-Oriented Evaluation

The framework is trained on the benchmark **N-BaIoT** dataset and evaluated using the modern **CICIoT2023** dataset to analyze robustness against evolving and unseen IoT malware attacks.

---

# 📚 Table of Contents

1. Introduction
2. Problem Statement
3. Research Motivation
4. Objectives
5. Datasets Used
6. Project Architecture
7. Technologies Used
8. Installation Guide
9. Dataset Setup
10. Data Preprocessing
11. Sequence Generation
12. CNN-LSTM Architecture
13. Training Workflow
14. Cross-Dataset Evaluation
15. Obfuscation Detection Strategy
16. Evaluation Metrics
17. Results Expected
18. Future Prevention System
19. Folder Structure
20. Research Contributions
21. Future Scope
22. Conclusion

---

# 🌐 1. Introduction

The rapid expansion of Internet of Things (IoT) devices has introduced severe cybersecurity challenges in modern smart environments.

IoT devices such as:

* Smart cameras
* Smart thermostats
* Doorbells
* Sensors
* Smart routers
* Medical IoT devices

are highly vulnerable to malware infections due to:

* Weak security mechanisms
* Default credentials
* Limited computational power
* Lack of firmware updates

Botnet malware such as:

* Mirai
* Bashlite (Gafgyt)
* DDoS malware
* Command & Control (C2) attacks

can compromise thousands of IoT devices simultaneously.

Traditional intrusion detection systems often fail against evolving and obfuscated malware behaviors.

This project addresses this issue using sequence-based deep learning models capable of learning temporal attack patterns from network traffic.

---

# ⚠️ 2. Problem Statement

Existing IoT intrusion detection systems suffer from:

❌ Signature dependency
❌ Poor generalization
❌ Weak temporal analysis
❌ Inability to detect evolving attacks
❌ High false positives

There is a need for an intelligent detection system capable of:

✅ Learning behavioral patterns
✅ Detecting unseen attacks
✅ Handling modified/obfuscated malware
✅ Performing cross-dataset evaluation

---

# 💡 3. Research Motivation

Modern IoT malware continuously evolves to evade detection mechanisms.

Sequence models such as:

* LSTM
* CNN-LSTM
* Transformer-based models

can capture:

* Temporal dependencies
* Sequential attack behavior
* Long-term traffic relationships

This motivates the use of sequence deep learning for intelligent IoT malware detection.

---

# 🎯 4. Objectives

## Primary Objectives

* Detect IoT botnet attacks using deep learning.
* Analyze temporal network traffic behavior.
* Build a CNN-LSTM hybrid architecture.
* Train using benchmark IoT datasets.
* Evaluate generalization using modern datasets.
* Simulate robustness against obfuscated attacks.

## Secondary Objectives

* Reduce false positives.
* Improve attack detection accuracy.
* Create research-oriented framework.
* Support future prevention systems.

---

# 📦 5. Datasets Used

## 🥇 N-BaIoT Dataset (Training Dataset)

### Dataset Description

The N-BaIoT dataset contains real IoT network traffic collected from devices infected with Mirai and Bashlite botnets.

### Features

* 7+ million records
* 115 statistical features
* Sequential traffic behavior
* Real IoT device traffic
* Benign and malicious samples

### Devices Included

* Danmini Doorbell
* Ecobee Thermostat
* Provision Camera
* Samsung Webcam
* Baby Monitor

### Why Selected

✅ Benchmark dataset
✅ Sequence-friendly
✅ Widely used in research
✅ Ideal for LSTM models

---

## 🥈 CICIoT2023 Dataset (Testing Dataset)

### Dataset Description

The CICIoT2023 dataset contains modern IoT attack traffic and realistic attack scenarios.

### Files Used

* Merged01.csv
* Merged02.csv
* Optional: Merged03.csv

### Why Selected

✅ Modern attack behaviors
✅ Cross-dataset validation
✅ Better generalization testing
✅ Simulates evolving malware

---

# 🧠 6. Project Architecture

```text
                DATA COLLECTION
                ├── N-BaIoT Dataset
                └── CICIoT2023 Dataset

                        ↓

                DATA PREPROCESSING
                ├── Cleaning
                ├── Normalization
                ├── Feature Selection
                └── Label Encoding

                        ↓

                SEQUENCE GENERATION
                ├── Sliding Window
                └── Temporal Sequences

                        ↓

                CNN-LSTM MODEL
                ├── Conv1D Layers
                ├── MaxPooling
                ├── LSTM Layers
                └── Dense Layers

                        ↓

                MODEL TRAINING
                └── N-BaIoT Dataset

                        ↓

                MODEL TESTING
                └── CICIoT2023 Dataset

                        ↓

                PERFORMANCE EVALUATION
                ├── Accuracy
                ├── Precision
                ├── Recall
                ├── F1-Score
                └── Confusion Matrix
```

---

# ⚙️ 7. Technologies Used

| Technology       | Purpose                   |
| ---------------- | ------------------------- |
| Python           | Core programming language |
| TensorFlow       | Deep learning framework   |
| Keras            | Neural network APIs       |
| Pandas           | Data processing           |
| NumPy            | Numerical operations      |
| Scikit-learn     | ML preprocessing          |
| Matplotlib       | Visualization             |
| Jupyter Notebook | Experimentation           |

---

# 🛠️ 8. Installation Guide

## Clone Repository

```bash
git clone https://github.com/yourusername/iot-botnet-detection.git
cd iot-botnet-detection
```

## Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

## Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
```

---

# ⬇️ 9. Dataset Setup

## N-BaIoT Dataset

Official Source:

[https://archive-beta.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot](https://archive-beta.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot)

Recommended Files:

* benign_traffic.csv
* mirai_attacks.csv
* gafgyt_attacks.csv

---

## CICIoT2023 Dataset

Official Source:

[https://www.unb.ca/cic/datasets/iotdataset-2023.html](https://www.unb.ca/cic/datasets/iotdataset-2023.html)

Recommended Downloads:

* MERGED_CSV.zip
* Merged01.csv
* Merged02.csv

Avoid:

❌ PCAP files
❌ Raw packet captures

---

# 🧹 10. Data Preprocessing

The preprocessing pipeline includes:

## Steps

1. Remove missing values
2. Drop duplicate rows
3. Encode labels
4. Normalize features
5. Select useful features
6. Convert to sequences

## Label Encoding

```python
Benign = 0
Attack = 1
```

## Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

# 🔄 11. Sequence Generation

Temporal sequences are generated using a sliding window mechanism.

## Example

```python
def create_sequences(data, labels, window_size=10):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(labels[i+window_size])
    return np.array(X), np.array(y)
```

---

# 🤖 12. CNN-LSTM Architecture

## Model Workflow

```text
Input Sequences
        ↓
Conv1D Layer
        ↓
MaxPooling Layer
        ↓
LSTM Layer
        ↓
Dense Layer
        ↓
Output Prediction
```

## Example Architecture

```python
model = Sequential([
    Conv1D(64, 3, activation='relu'),
    MaxPooling1D(2),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

---

# 🧪 13. Training Workflow

## Training Dataset

✅ N-BaIoT

## Workflow

1. Load dataset
2. Preprocess features
3. Generate sequences
4. Train CNN-LSTM
5. Save trained model

---

# 🌍 14. Cross-Dataset Evaluation

## Testing Dataset

✅ CICIoT2023

## Purpose

The trained model is tested on unseen modern attack traffic to evaluate:

* Generalization capability
* Robustness
* Detection of evolving attacks

---

# 🔐 15. Obfuscation Detection Strategy

This project simulates obfuscation-resistant detection through:

* Cross-dataset validation
* Behavioral learning
* Temporal sequence analysis

Instead of relying on static signatures, the model learns behavioral attack patterns.

---

# 📊 16. Evaluation Metrics

The following metrics are used:

| Metric           | Purpose                     |
| ---------------- | --------------------------- |
| Accuracy         | Overall performance         |
| Precision        | False positive control      |
| Recall           | Attack detection capability |
| F1-Score         | Balanced evaluation         |
| Confusion Matrix | Error analysis              |

---

# 📈 17. Results Expected

Expected outcomes include:

✅ High detection accuracy
✅ Better generalization
✅ Reduced false positives
✅ Temporal attack understanding
✅ Robustness against unseen attacks

---

# 🛡️ 18. Future Prevention System

Future extensions may include:

* Real-time intrusion prevention
* Automated IP blocking
* Traffic rate limiting
* Alert systems
* Dashboard visualization

---

# 📂 19. Folder Structure

```text
project/
│
├── data/
│   ├── nbaiot/
│   └── cic/
│
├── notebooks/
├── models/
├── src/
├── results/
└── README.md
```

---

# 🏆 20. Research Contributions

## Major Contributions

✅ Sequence-based IoT malware detection
✅ CNN-LSTM hybrid architecture
✅ Cross-dataset evaluation
✅ Obfuscation-resilient framework
✅ Temporal behavioral analysis

---

# 🔮 21. Future Scope

Potential future work:

* Transformer-based architectures
* Federated learning
* Edge AI deployment
* Real-time IDS systems
* Zero-day malware detection

---

# ✅ 22. Conclusion

This project presents a research-oriented deep learning framework for IoT botnet detection using sequence modeling techniques.

The proposed CNN-LSTM architecture learns temporal attack behavior from benchmark datasets and evaluates generalization using modern IoT attack traffic.

The framework demonstrates strong potential for:

* Behavioral malware detection
* Cross-dataset robustness
* Obfuscation-resilient security
* Future intelligent IoT intrusion detection systems

---

# 📖 References

1. N-BaIoT Dataset - UCI Machine Learning Repository
2. CICIoT2023 Dataset - Canadian Institute for Cybersecurity
3. TensorFlow Documentation
4. Keras Deep Learning Documentation
5. Recent IEEE IoT Security Research Papers

---

<div align="center">

## ⭐ If you found this project useful, consider giving it a star ⭐

### 🔐 Researching Intelligent IoT Security using Deep Learning

</div>
