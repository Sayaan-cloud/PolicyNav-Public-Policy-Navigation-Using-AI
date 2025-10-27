from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib, pandas as pd, numpy as np, os, textwrap

# Classical ML
from sklearn.metrics.pairwise import cosine_similarity

# Quantum + preprocessing
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# ----------------- Config -----------------
CLASSICAL_DIR = "classical"
QUANTUM_DIR = "quantum"

VECTORIZER_PATH = os.path.join(CLASSICAL_DIR, "policy_vectorizer.pkl")
MATRIX_PATH = os.path.join(CLASSICAL_DIR, "policy_tfidf_matrix.pkl")
QSVC_PATH = os.path.join(QUANTUM_DIR, "quantum_policy_qsvc_model.pkl")
SCALER_PATH = os.path.join(QUANTUM_DIR, "policy_scaler.pkl")
PCA_PATH = os.path.join(QUANTUM_DIR, "policy_pca.pkl")

# ----------------- Check Files -----------------
missing = [p for p in [VECTORIZER_PATH, MATRIX_PATH, QSVC_PATH, SCALER_PATH, PCA_PATH] if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(f"Required files missing: {missing}")

# ----------------- Load Classical Artifacts -----------------
vectorizer = joblib.load(VECTORIZER_PATH)
data = joblib.load(MATRIX_PATH)
tfidf_matrix = data["matrix"]
df = data["df"].copy()

# ----------------- Load Quantum Artifacts -----------------
qsvc = joblib.load(QSVC_PATH)
scaler: StandardScaler = joblib.load(SCALER_PATH)
pca: PCA = joblib.load(PCA_PATH)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------- Quantum Kernel -----------------
n_qubits = pca.n_components_
feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=1)
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

# ----------------- Prepare NLP Column -----------------
if "text_for_nlp" not in df.columns:
    df["text_for_nlp"] = (
        df["title"].astype(str) + ". " +
        df["full_text"].astype(str) + ". Stakeholders: " +
        df.get("stakeholders", "").astype(str)
    ).str.lower()

# ----------------- Compute Document Embeddings -----------------
doc_texts = df["text_for_nlp"].tolist()
doc_embeddings = embed_model.encode(doc_texts, show_progress_bar=False, convert_to_numpy=True)
doc_scaled = scaler.transform(doc_embeddings)
doc_reduced = pca.transform(doc_scaled)

# ----------------- FastAPI Setup -----------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ---------- Helper ----------
def make_json_safe(obj):
    """Convert numpy/pandas types to JSON-safe primitives."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, list, tuple)):
        return [make_json_safe(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    else:
        return obj


# ---------- Classical Search ----------
def search_policies_classical(query: str, top_k: int = 5):
    query_vec = vectorizer.transform([query.lower()])
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
            "score": float(round(sims[idx], 3))
        })
    return make_json_safe(results)


# ---------- Quantum Search ----------
def quantum_pipeline_query(query: str, top_k: int = 5):
    q_emb = embed_model.encode([query.lower()], convert_to_numpy=True)[0:1]
    q_scaled = scaler.transform(q_emb)
    q_reduced = pca.transform(q_scaled)

    try:
        pred = qsvc.predict(q_reduced)[0]
    except Exception:
        pred = None

    decision_score = None
    if hasattr(qsvc, "decision_function"):
        try:
            decision_score = float(qsvc.decision_function(q_reduced)[0])
        except Exception:
            pass
    elif hasattr(qsvc, "predict_proba"):
        try:
            probs = qsvc.predict_proba(q_reduced)[0]
            decision_score = float(max(probs))
        except Exception:
            pass

    try:
        K_q = quantum_kernel.evaluate(x_vec=q_reduced.tolist(), y_vec=doc_reduced.tolist())
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
            "score": float(round(sims[idx], 4))
        })

    return make_json_safe({
        "prediction": int(pred) if pred is not None else None,
        "decision_score": decision_score,
        "results": results
    })


# ----------------- Routes -----------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("classical.html", {"request": request, "results": None, "query": "", "top_k": 5})


@app.post("/classical_search", response_class=HTMLResponse)
async def classical_search(request: Request, query: str = Form(...), top_k: int = Form(5)):
    try:
        results = search_policies_classical(query, top_k=top_k)
        return templates.TemplateResponse("classical.html", {
            "request": request,
            "results": results,
            "query": query,
            "top_k": top_k
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classical search failed: {e}")


@app.get("/quantum_search", response_class=HTMLResponse)
async def quantum_search_home(request: Request):
    return templates.TemplateResponse("quantum.html", {
        "request": request, 
        "results": None, 
        "query": "",
        "decision_score": None,
        "top_k": 5
    })


@app.post("/quantum_search", response_class=HTMLResponse)
async def quantum_search(request: Request, query: str = Form(...), top_k: int = Form(5)):
    try:
        q_out = quantum_pipeline_query(query, top_k=top_k)
        return templates.TemplateResponse("quantum.html", {
            "request": request,
            "results": q_out["results"],
            "query": query,
            "prediction": q_out["prediction"],
            "decision_score": q_out["decision_score"],
            "top_k": top_k
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quantum search failed: {e}")
