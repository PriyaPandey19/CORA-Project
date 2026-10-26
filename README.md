# Cora GCN Explorer

An interactive paper-topic classifier for the Cora citation network. It runs a Graph Convolutional Network (GCN) in ONNX format and provides both a browser interface and an optional Streamlit interface.

The Cora dataset represents research papers as graph nodes and citations as edges. Each paper has a 1,433-value bag-of-words feature vector, and the model predicts one of seven research topics.

## Features

- Classify selected Cora papers by node index, or enter several indices at once.
- Submit a custom graph with node features and optional edge connections.
- Explore predicted topics, class probabilities, and raw model logits in the interfaces.
- Check whether the API is running and inspect model information.

## Run locally

### FastAPI and browser interface

Requirements: Python 3.9 or newer. From the project root, create a virtual environment, install dependencies, and start the API:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

Then open:

- `http://127.0.0.1:8000/` — browser-based Cora GCN Explorer.
- `http://127.0.0.1:8000/docs` — interactive API documentation.
- `http://127.0.0.1:8000/health` — API health status.

The API loads `simple_gcn_cora.onnx` and `cora_data.npz` from the project root.

### Optional Streamlit interface

The Streamlit app calls the FastAPI service. Keep the API running, open another terminal, and run:

```powershell
python -m pip install streamlit requests pandas
streamlit run streamlit_app.py
```

The sidebar's API URL defaults to the hosted service. Change it to `http://127.0.0.1:8000` to use your local API.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Returns service status and active ONNX Runtime providers. |
| `GET /info` or `GET /model-info` | Returns model dimensions, class mapping, and tensor details. |
| `POST /predict/cora_node` | Predicts topics for Cora node indices from `0` to `2707`. |
| `POST /predict` | Predicts topics for a graph supplied in the request. |

Example request for existing Cora nodes:

```json
{
  "node_indices": [0, 1, 2]
}
```

For a custom graph, send `node_features` as a non-empty array of node vectors. Every vector must contain exactly 1,433 numeric values. You may also provide `edge_indices` as a `[2, number_of_edges]` array of source and destination node indices. If edges are omitted, the API adds a self-loop to each node.

## Dataset

The repository includes the prebuilt `cora_data.npz` archive used for predictions on real Cora nodes. To regenerate it, install PyTorch and PyTorch Geometric, then run the exporter:

```powershell
python -m pip install torch torch-geometric
python export_cora.py
```

The exporter loads the Cora dataset via PyTorch Geometric and saves node features and edge indices to `cora_data.npz`.

## Topic classes

The model predicts one of these seven categories:

1. Case Based
2. Genetic Algorithms
3. Neural Networks
4. Probabilistic Methods
5. Reinforcement Learning
6. Rule Learning
7. Theory

## Project structure

```text
.
├── main.py                 # FastAPI routes and ONNX inference
├── streamlit_app.py        # Optional Streamlit client
├── export_cora.py          # Cora dataset archive exporter
├── requirements.txt        # API dependencies
├── simple_gcn_cora.onnx    # Trained ONNX model
├── cora_data.npz           # Cora features and graph edges
├── main.ipynb              # Jupyter notebook
└── static/                 # Browser UI (HTML, CSS, JavaScript)
```
