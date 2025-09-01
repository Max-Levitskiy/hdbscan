from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import hdbscan

app = FastAPI()

class Req(BaseModel):
    vectors: list[list[float]]
    min_cluster_size: int = 25
    min_samples: int | None = None
    metric: str = "euclidean"

class Res(BaseModel):
    labels: list[int]
    probabilities: list[float]
    outlier_scores: list[float]

@app.post("/hdbscan", response_model=Res)
def run(req: Req):
    X = np.asarray(req.vectors, dtype=np.float32)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=req.min_cluster_size,
        min_samples=req.min_samples,
        metric=req.metric,
        prediction_data=True
    ).fit(X)
    return Res(
        labels=clusterer.labels_.tolist(),
        probabilities=clusterer.probabilities_.tolist(),
        outlier_scores=clusterer.outlier_scores_.tolist()
    )
