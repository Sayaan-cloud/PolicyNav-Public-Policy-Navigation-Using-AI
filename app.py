import os
import textwrap
from typing import Optional, List, Any, Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from passlib.context import CryptContext
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime

# Qiskit (quantum kernel)
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# Local DB
from database import users_collection

# ---------------- Config ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSICAL_DIR = os.path.join(BASE_DIR, "classical")
QUANTUM_DIR = os.path.join(BASE_DIR, "quantum")

VECTORIZER_PATH = os.path.join(CLASSICAL_DIR, "policy_vectorizer.pkl")
MATRIX_PATH = os.path.join(CLASSICAL_DIR, "policy_tfidf_matrix.pkl")
QSVC_PATH = os.path.join(QUANTUM_DIR, "quantum_policy_qsvc_model.pkl")
SCALER_PATH = os.path.join(QUANTUM_DIR, "policy_scaler.pkl")
PCA_PATH = os.path.join(QUANTUM_DIR, "policy_pca.pkl")

REQUIRED_FILES = [VECTORIZER_PATH, MATRIX_PATH, QSVC_PATH, SCALER_PATH, PCA_PATH]
missing = [p for p in REQUIRED_FILES if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(f"Required files missing: {missing}")

# ---------------- Load classical models/data ----------------
vectorizer = joblib.load(VECTORIZER_PATH)
matrix_data = joblib.load(MATRIX_PATH)
tfidf_matrix = matrix_data.get("matrix")
df = matrix_data.get("df").copy()

# ---------------- Load quantum models ----------------
qsvc = joblib.load(QSVC_PATH)
scaler: StandardScaler = joblib.load(SCALER_PATH)
pca: PCA = joblib.load(PCA_PATH)

# ---------------- Embedding model ----------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- Quantum kernel setup ----------------
n_qubits = getattr(pca, "n_components_", None)
if n_qubits is None or n_qubits <= 0:
    raise RuntimeError("PCA not loaded properly or n_components_ invalid.")

feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=1)
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

# ---------------- Prepare NLP text ----------------
if "text_for_nlp" not in df.columns:
    stakeholders_col = df["stakeholders"] if "stakeholders" in df.columns else pd.Series([""] * len(df))
    df["text_for_nlp"] = (
        df["title"].astype(str) + ". " +
        df["full_text"].astype(str) + ". Stakeholders: " +
        stakeholders_col.astype(str)
    ).str.lower()

doc_texts = df["text_for_nlp"].tolist()
doc_embeddings = embed_model.encode(doc_texts, show_progress_bar=False, convert_to_numpy=True)
doc_scaled = scaler.transform(doc_embeddings)
doc_reduced = pca.transform(doc_scaled)

# ---------------- App setup ----------------
app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------- Helpers ----------------
def make_json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray, list, tuple)):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    return obj

async def require_login(request: Request) -> str:
    user = request.cookies.get("user")
    if not user:
        raise HTTPException(status_code=403, detail="Please log in first.")
    return user

def clamp_top_k(v: int) -> int:
    try:
        n = int(v)
    except Exception:
        n = 5
    return max(1, min(10, n))

# ---------------- Search functions ----------------
def search_policies_classical(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    q = (query or "").lower()
    query_vec = vectorizer.transform([q])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = sims.argsort()[::-1][:top_k]

    results = []
    for idx in top_idx:
        row = df.iloc[idx]
        results.append({
            "title": row.get("title", "Unknown Policy"),
            "policy_id": row.get("policy_id", "N/A"),
            "region": row.get("region", "Unknown"),
            "year": row.get("year", "N/A"),
            "status": row.get("status", "N/A"),
            "summary": textwrap.shorten(str(row.get("full_text", "No description available")), width=250, placeholder="..."),
            "score": float(round(float(sims[idx]), 4))
        })
    return make_json_safe(results)

def quantum_pipeline_query(query: str, top_k: int = 5) -> Dict[str, Any]:
    q = (query or "").lower()
    q_emb = embed_model.encode([q], convert_to_numpy=True)
    q_scaled = scaler.transform(q_emb)
    q_reduced = pca.transform(q_scaled)

    try:
        pred = int(qsvc.predict(q_reduced)[0])
    except Exception:
        pred = None

    decision_score = None
    if hasattr(qsvc, "decision_function"):
        try:
            decision_score = float(qsvc.decision_function(q_reduced)[0])
        except Exception:
            decision_score = None
    elif hasattr(qsvc, "predict_proba"):
        try:
            probs = qsvc.predict_proba(q_reduced)[0]
            decision_score = float(max(probs))
        except Exception:
            decision_score = None

    try:
        K_q = quantum_kernel.evaluate(x_vec=q_reduced, y_vec=doc_reduced)
        sims = np.array(K_q).flatten()
    except Exception as e:
        raise RuntimeError(f"Quantum kernel evaluation failed: {e}")

    top_idx = sims.argsort()[::-1][:top_k]
    results = []
    for idx in top_idx:
        row = df.iloc[idx]
        results.append({
            "title": row.get("title", "Unknown Policy"),
            "policy_id": row.get("policy_id", "N/A"),
            "region": row.get("region", "Unknown"),
            "year": row.get("year", "N/A"),
            "status": row.get("status", "N/A"),
            "summary": textwrap.shorten(str(row.get("full_text", "No description available")), width=250, placeholder="..."),
            "score": float(round(float(sims[idx]), 6))
        })

    return make_json_safe({
        "prediction": pred,
        "decision_score": decision_score,
        "results": results
    })

# ---------------- Routes ----------------
# ---------- Homepage ----------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    df_local = pd.read_csv("datasets/education_policies.csv")
    policies = df_local.head(4).to_dict(orient="records")
    return templates.TemplateResponse("home.html", {"request": request, 
                                                    "policies": policies, 
                                                    "year": datetime.now().year})

# ---------- Login ----------
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    user = users_collection.find_one({"email": email})
    if not user or not pwd_context.verify(password, user["password"]):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials!"})
    response = RedirectResponse(url="/classical", status_code=303)
    response.set_cookie(key="user", value=email, httponly=True)
    return response

# ---------- Signup ----------
@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request, "error": None})

@app.post("/signup", response_class=HTMLResponse)
async def signup(request: Request, email: str = Form(...), password: str = Form(...)):
    if users_collection.find_one({"email": email}):
        return templates.TemplateResponse("signup.html", {"request": request, "error": "Email already registered!"})
    hashed_pw = pwd_context.hash(password)
    users_collection.insert_one({"email": email, "password": hashed_pw})
    response = RedirectResponse(url="/classical", status_code=303)
    response.set_cookie(key="user", value=email, httponly=True)
    return response

# ---------- Logout ----------
@app.get("/logout")
def logout(request: Request):
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie("access_token")
    return response

# ---------- Classical ----------
@app.get("/classical", response_class=HTMLResponse)
async def classical_home(request: Request, user: str = Depends(require_login)):
    return templates.TemplateResponse("classical.html", {"request": request, "results": None, "query": "", "top_k": 5})

@app.post("/classical_search", response_class=HTMLResponse)
async def classical_search(request: Request, query: str = Form(...), top_k: int = Form(5), user: str = Depends(require_login)):
    k = clamp_top_k(top_k)
    results = search_policies_classical(query, top_k=k)
    return templates.TemplateResponse("classical.html", {"request": request, "results": results, "query": query, "top_k": k})

# ---------- Quantum ----------
@app.get("/quantum", response_class=HTMLResponse)
async def quantum_page(request: Request, user: str = Depends(require_login)):
    return templates.TemplateResponse("quantum.html", {
        "request": request,
        "results": [],
        "query": "",
        "prediction": None,
        "decision_score": None,
        "top_k": 5
    })

@app.post("/quantum_search", response_class=HTMLResponse)
async def quantum_search(request: Request, query: str = Form(...), top_k: int = Form(5), user: str = Depends(require_login)):
    k = clamp_top_k(top_k)
    q_out = quantum_pipeline_query(query, top_k=k)
    return templates.TemplateResponse("quantum.html", {
        "request": request,
        "results": q_out["results"],
        "query": query,
        "prediction": q_out.get("prediction"),
        "decision_score": q_out.get("decision_score"),
        "top_k": k
    })

# ---------------- Run with uvicorn ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
