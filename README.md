# Heart Disease Prediction using Federated Learning (Flower)

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![ML Framework](https://img.shields.io/badge/scikit--learn-1.8.0-orange.svg)](https://scikit-learn.org/)
[![FL Framework](https://img.shields.io/badge/Flower-1.13.0-orange.svg)](https://flower.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This repository implements a **Privacy-Preserving Federated Learning (FL)** pipeline designed to predict the **10-year risk of coronary heart disease (CHD)**. Unlike traditional machine learning where all data is gathered in one place, this project simulates a decentralized collaboration between three independent hospital silos.

By leveraging the **Flower (flwr)** framework and **Ray**, this project demonstrates how medical institutions can build a high-performance **Global Model** without ever exchanging raw patient records, fully adhering to **HIPAA/GDPR** data privacy standards.

## Repository Structure

```text
heart-disease-prediction-federated-learning/
├── data/                    # Local hospital silos
├── src/                     # Core Federated Logic
│   ├── client.py            # Local Hospital Logic (Fit/Eval)
│   ├── server.py            # Global Coordinator (Aggregation)
│   ├── partition_data.py    # Data Silo Simulator
│   └── utils.py             # Serialization helpers
├── requirements.txt         # Project dependencies
└── README.md                # Documentation
```

## How Federated Learning Works (The Workflow)

The core idea of this project is that **data stays at the hospital, while the model travels.**

1. **Global Handshake:** The central server creates a "blank" model and sends it to all participating hospitals.

2. **Local Training:** Each hospital trains that model using only its own local patients. The hospital's data never leaves its secure server.

3. **Sending Updates:** Instead of sending patient files, the hospitals send back a small file containing the "learned weights" (the patterns the model found).

4. **Averaging Knowledge:** The server takes the patterns from all hospitals and averages them together to create a smarter "Global Model."

5. **Iteration:** This cycle repeats for 5 rounds. Each time, the hospitals start with a smarter model than the round before, leading to a highly accurate final result.

## Data Governance & Security

1. **Privacy by Design:** Raw patient records are processed only in the hospital's local memory.

2. **Encapsulation:** Only mathematical coefficients are transmitted over the network.

3. **Silo Standardization:** The system automatically ensures that all hospitals use the same clinical features (e.g., Blood Pressure, Cholesterol, Glucose) so the Global Model remains aligned.

## Installation & Usage

### 1. Environment Setup

It is recommended to use a virtual environment to isolate project dependencies and prevent version conflicts.

**Windows:**

```bash
# Create the virtual environment
python -m venv .venv

# Activate the environment
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate the environment
source .venv/bin/activate
```

### 2. Install Dependecies

```bash
pip install -r requirements.txt
```

### 3. Run the Simulation

Ensure framingham.csv is located in the data/ directory, then execute:

```bash
# 1. Partition the data into 3 hospital silos
python src/partition_data.py

# 2. Start the Federated Simulation
python src/server.py
```

### Performance Results

The simulation proves that collaborative learning achieves Global Convergence, reaching a performance level of 85.4%

This result is significantly more robust than what a single hospital could achieve on its own.

| Training Phase     | Round     | Global Accuracy | Global Loss | Status               |
| :----------------- | :-------- | :-------------- | :---------- | :------------------- |
| **Initialization** | `Round 1` | 84.90%          | 0.4587      | Initial Handshake    |
| **Collaboration**  | `Round 2` | 85.12%          | 0.4342      | Early Convergence    |
|                    | `Round 3` | 85.25%          | 0.3817      | Improving Confidence |
|                    | `Round 4` | 85.36%          | 0.3784      | Model Refinement     |
| **Finalization**   | `Round 5` | **85.42%**      | **0.3777**  | Optimal Convergence  |

---

[!IMPORTANT]

Analytical Note: You will notice the "Loss" continues to drop even when "Accuracy" stabilizes. This means the model is becoming more **confident** in its predictions. In a medical context, a lower loss is vital because it means the model is providing more reliable probability scores for patient risk.

### References

Dataset: [Framingham Heart Study Dataset](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset)

Framework: [Flowe: A Friendly Federated Learning Framework](https://flower.ai/)
