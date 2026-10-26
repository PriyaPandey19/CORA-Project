import os
from functools import lru_cache
from typing import List, Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

CORA_CLASSES = {
    0: "Case_Based",
    1: "Genetic_Algorithms",
    2: "Neural_Network",
    3: "Probabilistic_Methods",
    4: "Reinforcement_Learning",
    5: "Rule_Learning",
    6: "Theory",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "simple_gcn_cora.onnx")
CORA_NPZ = os.path.join(BASE_DIR, "cora_data.npz")
STATIC_DIR = os.path.join(BASE_DIR, "static")

model_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

app = FastAPI()


class GraphPredictRequest(BaseModel):
    node_features: List[List[float]]
    edge_indices: Optional[List[List[int]]] = None


class CoraNodeRequest(BaseModel):
    node_indices: List[int]


def softmax(scores: np.ndarray):
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=-1, keepdims=True)


def run_model(node_features: np.ndarray, edge_indices: np.ndarray, node_indices_to_return):
    output = model_session.run(
        ["logits"],
        {
            "node_features": node_features.astype(np.float32),
            "edge_indices": edge_indices.astype(np.int64),
        },
    )
    logits = output[0]
    probabilites = softmax(logits)
    predicted_classes = logits.argmax(axis=-1)

    results = []
    for i in node_indices_to_return:
        results.append({
            "node_index": int(i),
            "predicted_class_id": int(predicted_classes[i]),
            "predicted_class_name": CORA_CLASSES[int(predicted_classes[i])],
            "probabilites": probabilites[i].tolist(),
            "logits": logits[i].tolist(),
        })

    # return is OUTSIDE the loop now (before: only the first node was returned)
    return {
        "num_nodes": int(node_features.shape[0]),
        "num_edges": int(edge_indices.shape[1]),
        "predictions": results,
    }


@lru_cache(maxsize=1)
def load_cora():
    data = np.load(CORA_NPZ)
    return data["x"], data["edge_index"]


@app.get("/")
def home_page():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"service": "Simple GCN Cora API", "status": "Running"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "providers": model_session.get_providers()}


@app.get("/info")
@app.get("/model-info")
def model_info():
    return {
        "Model_Name": "SimpleGCN",
        "feature_dimension": 1433,
        "num_classes": 7,
        "class_mapping": CORA_CLASSES,
        "inputs": [
            {"name": i.name, "shape": [str(s) for s in i.shape], "type": i.type}
            for i in model_session.get_inputs()
        ],
        "outputs": [
            {"name": o.name, "shape": [str(s) for s in o.shape], "type": o.type}
            for o in model_session.get_outputs()
        ],
    }


@app.post("/predict")
def predict_custom_graph(request: GraphPredictRequest):
    if not request.node_features:
        raise HTTPException(400, "Node features cannot be empty")

    for vec in request.node_features:
        if len(vec) != 1433:
            raise HTTPException(422, "Each node's feature vector must have exactly 1433 features")

    node_features = np.array(request.node_features, dtype=np.float32)
    num_nodes = node_features.shape[0]

    if request.edge_indices:
        edge_index = np.array(request.edge_indices, dtype=np.int64)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise HTTPException(422, "edge_indices must have shape [2, num_edges]")
        if edge_index.min() < 0 or edge_index.max() >= num_nodes:
            raise HTTPException(422, f"edge indices must be between 0 and {num_nodes - 1}")
    else:
        node_ids = np.arange(num_nodes, dtype=np.int64)
        edge_index = np.vstack([node_ids, node_ids])

    return run_model(node_features, edge_index, list(range(num_nodes)))


@app.post("/predict/cora_node")
def predict_real_cora_nodes(request: CoraNodeRequest):
    if not request.node_indices:
        raise HTTPException(400, "node_indices cannot be empty")
    try:
        x, edge_index = load_cora()
    except Exception as error:
        raise HTTPException(500, f"Failed to load Cora data: {error}")

    largest = x.shape[0] - 1
    if any(i < 0 or i > largest for i in request.node_indices):
        raise HTTPException(400, f"node_index out of bounds (must be 0 to {largest})")

    return run_model(x, edge_index, request.node_indices)


if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")