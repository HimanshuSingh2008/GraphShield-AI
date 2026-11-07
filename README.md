<div align="center">

# 🛡️ GraphShield AI

### Intelligent Graph-Based Fraud Detection System

**Securing Digital Transactions Through Artificial Intelligence and Graph Analytics**

<br>

<p>
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Machine%20Learning-Random%20Forest-orange?style=for-the-badge" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Graph%20Analytics-NetworkX-green?style=for-the-badge" alt="Graph Analytics">
  <img src="https://img.shields.io/badge/Framework-PyTorch-red?style=for-the-badge&logo=pytorch" alt="PyTorch">
</p>

<p>
  <img src="https://img.shields.io/badge/Status-Under%20Development-yellow?style=flat-square" alt="Project Status">
  <img src="https://img.shields.io/badge/Domain-Cybersecurity-purple?style=flat-square" alt="Domain">
  <img src="https://img.shields.io/badge/Focus-Fraud%20Detection-blue?style=flat-square" alt="Focus">
</p>

</div>

---

## 📌 About The Project

**GraphShield AI** is an intelligent, graph-based fraud detection system designed to identify suspicious financial transactions using Machine Learning and Graph Analytics.

Traditional fraud detection systems often analyze transactions individually. GraphShield AI takes a different approach by representing financial transactions as interconnected networks, allowing the system to analyze relationships between accounts and identify suspicious transaction patterns.

The system models financial accounts as **nodes** and transactions as **directed edges**, enabling graph-based analysis and machine learning to detect potentially fraudulent activities.

The project uses the PaySim synthetic financial transaction dataset to develop and evaluate fraud detection models.

### 🎯 Project Objectives

- Detect fraudulent and legitimate financial transactions.
- Represent financial transactions using graph structures.
- Analyze relationships between sender and receiver accounts.
- Extract meaningful transaction and behavioral features.
- Train machine learning models for fraud classification.
- Explore Graph Neural Networks for relationship-aware fraud detection.
- Compare traditional machine learning with graph-based learning approaches.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| Transaction Analysis | Analyze financial transaction records. |
| Graph Construction | Represent accounts as nodes and transactions as edges. |
| Feature Engineering | Extract transaction-based and behavioral features. |
| Fraud Classification | Classify transactions as fraudulent or legitimate. |
| Random Forest | Establish a traditional machine learning baseline. |
| GraphSAGE | Explore graph-based representation learning. |
| Model Evaluation | Evaluate models using classification metrics. |
| Model Comparison | Compare machine learning model performance. |

---

## 🧠 How GraphShield AI Works

<div align="center">

### System Workflow

</div>

```text
       PAYSim TRANSACTION DATASET
                  |
                  ▼
          DATA PREPROCESSING
                  |
                  ▼
          FEATURE ENGINEERING
                  |
                  ▼
           GRAPH CONSTRUCTION
                  |
          ┌───────┴────────┐
          ▼                ▼
    RANDOM FOREST       GRAPHSAGE
      BASELINE          MODEL
          |                |
          ▼                ▼
    FRAUD PREDICTION  FRAUD PREDICTION
          |                |
          └───────┬────────┘
                  ▼
           MODEL EVALUATION
                  |
                  ▼
          MODEL COMPARISON
                  |
                  ▼
       FRAUD / LEGITIMATE
          CLASSIFICATION
```

---

## 🔍 Graph-Based Approach

GraphShield AI uses graph theory to represent financial transaction networks.

**Graph Components:**

- **Nodes:** Represent individual financial accounts.
- **Edges:** Represent transactions between accounts.
- **Node Features:** Represent account-related behavioral information.
- **Edge Features:** Represent transaction amount, type, time, and other transaction attributes.

### Example

```text
       Account A
           |
           | Transaction
           ▼
       Account B
           |
           | Transaction
           ▼
       Account C
           |
           ▼
       Account D
```

This representation helps analyze transaction relationships instead of examining each transaction in isolation.

---

## 🛠️ Technology Stack

### Programming Language
- Python

### Data Processing
- Pandas
- NumPy

### Machine Learning
- Scikit-learn
- Random Forest

### Graph Analytics
- NetworkX

### Graph Neural Networks
- PyTorch
- PyTorch Geometric
- GraphSAGE

### Database / Backend Support
- Supabase
- Python-dotenv

---

## 📂 Dataset

**Dataset:** PaySim Synthetic Financial Dataset

GraphShield AI uses the PaySim dataset, which simulates mobile money transactions.

The dataset contains financial transaction information, including:

- Transaction Type
- Transaction Amount
- Sender Account
- Receiver Account
- Sender Balance
- Receiver Balance
- Fraud Labels

**Development Dataset:** A 200,000-record sample is used during development from the larger PaySim dataset.

**Prediction Target:**

- `0` – Legitimate Transaction
- `1` – Fraudulent Transaction

The primary prediction task is transaction-level fraud detection (edge classification), rather than account-level classification.

---

## ⚙️ Machine Learning Models

### 1. Random Forest

Random Forest is used as the baseline machine learning model.

It learns patterns from engineered transaction features and predicts whether a transaction is fraudulent or legitimate.

### 2. GraphSAGE

GraphSAGE is the graph-based learning approach explored in GraphShield AI.

It learns node representations by aggregating information from neighboring nodes in a transaction graph.

These representations can be combined with transaction-level information for fraud prediction.

### Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

For fraud detection, precision and recall are particularly important because incorrectly classifying fraudulent transactions can have significant consequences.

---

## 📁 Project Structure

```text
GraphShield-AI/
│
├── backend/
│   ├── .env
│   ├── upload_dataset.py
│   └── update_dataset.py
│
├── ml/
│   ├── preprocessing.py
│   ├── graph_builder.py
│   ├── prepare_splits.py
│   ├── train_baseline.py
│   ├── train_graphsage.py
│   └── compare_models.py
│
├── data/
│   ├── processed_transactions.csv
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── transaction_graph.pt
│   ├── random_forest_model.pkl
│   ├── graphsage_model.pth
│   └── model_comparison.csv
│
├── .gitignore
├── README.md
└── requirements.txt
```

*Note: Generated datasets, model files and environment files may be excluded from the public repository through `.gitignore`, depending on their size and sensitivity.*

---

## 🚀 Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/HimanshuSingh2008/GraphShield-AI.git
```

### Step 2: Navigate to the Project

```bash
cd GraphShield-AI
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows PowerShell**

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Prepare the Dataset

Place the PaySim dataset in the appropriate data directory and configure the preprocessing scripts according to the dataset location.

### Step 6: Run the Machine Learning Pipeline

```bash
python ml/preprocessing.py
```

```bash
python ml/graph_builder.py
```

```bash
python ml/prepare_splits.py
```

### Step 7: Train the Baseline Model

```bash
python ml/train_baseline.py
```

### Step 8: Train GraphSAGE

```bash
python ml/train_graphsage.py
```

### Step 9: Compare Models

```bash
python ml/compare_models.py
```

*Run the pipeline in sequence and ensure each generated input file is available before executing the next script.*

---

## 📊 Model Performance

GraphShield AI evaluates its models using standard classification metrics.

| Metric | Purpose |
|---|---|
| Accuracy | Overall prediction correctness |
| Precision | Correctness of fraud predictions |
| Recall | Ability to identify fraudulent transactions |
| F1-Score | Balance between precision and recall |

Model performance results will be added after completing the training and evaluation experiments.

---

## 🔐 Security & Data Management

- Environment variables are used for sensitive configuration.
- API keys and database credentials should never be committed.
- Supabase is used for storing project metadata.
- Large graph tensors and model artifacts are managed separately from database metadata.

---

## 🔮 Future Enhancements

- Develop an interactive fraud monitoring dashboard.
- Integrate real-time transaction analysis.
- Improve fraud detection using Graph Neural Networks.
- Implement advanced anomaly detection techniques.
- Visualize suspicious transaction networks.
- Add transaction risk scoring.
- Integrate automated fraud alerts.
- Deploy the system as a web-based application.

---

## 🎓 Project Information

**Project Name:** GraphShield AI

**Domain:** Artificial Intelligence | Machine Learning | Cybersecurity

**Project Category:** Graph-Based Financial Fraud Detection

**Developer:** Himanshu Singh

---

## 🤝 Contributions

Contributions, suggestions, and improvements are welcome.

If you find this project interesting, consider giving the repository a ⭐ on GitHub.

---

<div align="center">

### 🛡️ GraphShield AI

**Intelligent Detection. Connected Intelligence. Safer Transactions.**

*Building smarter fraud detection through AI and Graph Analytics.*

</div>
