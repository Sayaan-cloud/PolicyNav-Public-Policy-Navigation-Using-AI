# 🧭 PolicyNav — Public Policy Navigation Using AI

> **A fusion of Classical & Quantum Machine Learning to make public policy exploration intelligent, interactive, and futuristic.**

---

## 🚀 Overview

**PolicyNav** is a cutting-edge AI web app built to analyze and compare **government policies** using **Classical Machine Learning** and **Quantum Machine Learning** techniques.

It bridges the gap between traditional ML and next-gen Quantum Computing — allowing users to **search**, **compare**, and **understand** policies with higher accuracy and interpretability.

---

## 🌌 Why PolicyNav?

Governments release countless policies every year — often scattered, repetitive, or hard to understand.  
**PolicyNav** changes that by letting users:
- 🔍 Instantly **search** through policy data.  
- 🤝 **Compare** policy similarity using both ML and Quantum kernels.  
- 📊 **Visualize** which policies align most closely — helping decision-makers, researchers, and citizens.

---

## 🧠 Classical vs Quantum — The Core Difference

| Feature | Classical ML Approach | Quantum ML Approach |
|----------|-----------------------|---------------------|
| **Model Type** | TF-IDF + Cosine Similarity | Quantum Support Vector Classifier (QSVC) |
| **Computation** | Deterministic — based on vector space math | Probabilistic — based on quantum state interference |
| **Speed on Large Data** | Moderate | Scalable with quantum processors |
| **Insight** | Measures surface-level text similarity | Captures deeper, non-linear relationships in data |
| **Use Case** | Fast, explainable predictions | Experimental, high-accuracy for complex policy structures |

👉 In short:  
**Classical ML** gives speed and reliability, while **Quantum ML** adds a futuristic layer of accuracy inspired by qubit-level computations.

---

## 🧩 Features
- 🔍 **AI-Powered Search:** Explore similar public policies.
- ⚛️ **Dual Engines:** Classical + Quantum comparison.
- 📊 **TF-IDF + Quantum Kernel Models:** Real-time policy similarity scoring.
- 🌐 **FastAPI Web App:** Lightweight backend with responsive HTML templates.
- 🎨 **Visual Interface:** Ready for graphs and analytics dashboards.

---

## ⚙️ Tech Stack

| Layer | Technologies |
|-------|---------------|
| **Frontend** | HTML5, CSS3 (Jinja Templates) |
| **Backend** | FastAPI, Python |
| **ML Frameworks** | Scikit-learn, Qiskit, NumPy, Pandas |
| **Deployment Ready** | Uvicorn, GitHub |

---

## 📁 Folder Structure

edu_policies/
│
├── app.py
│
├── classical/
│   ├── classical_model.ipynb
│   ├── policy_tfidf_matrix.pkl
│   └── policy_vectorizer.pkl
│
├── quantum/
│   ├── quantum_model.ipynb
│   ├── quantum_policy_kernel_matrix.pkl
│   ├── quantum_policy_qsvc_model.pkl
│   ├── policy_scaler.pkl
│   └── policy_pca.pkl
│
├── datasets/
│   ├── education_policies.csv
│   ├── train_policies.csv
│   └── test_policies.csv
│
├── templates/
│   ├── classical.html
│   └── quantum.html
│
├── static/
│   └── (your CSS + JS)
│
├── .gitignore
├── requirements.txt
└── README.md

---

## 🧩 How It Works

1. User searches for a **policy or keyword**.  
2. The system processes it using both **classical** and **quantum pipelines**.  
3. Results are scored for **semantic similarity**.  
4. The app visualizes and displays similar policies ranked by relevance.

---

## ⚡ Quickstart Guide

```bash
# 1️⃣ Clone the repository
git clone https://github.com/Sayaan-cloud/PolicyNav-Public-Policy-Navigation-Using-AI.git

# 2️⃣ Move into the project folder
cd PolicyNav-Public-Policy-Navigation-Using-AI

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the app locally
uvicorn app:app --reload

# 5️⃣ Open in your browser
http://127.0.0.1:8000
