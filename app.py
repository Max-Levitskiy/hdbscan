import os
from typing import List, Optional, Any, Dict, Tuple
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import numpy as np
import hdbscan
from supabase import create_client, Client

# --- Env ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
DEFAULT_TABLE = os.getenv("TABLE_NAME", "content_embeddings")
DEFAULT_ID_COL = os.getenv("ID_COLUMN", "id")
DEFAULT_VECTOR_COL = os.getenv("VECTOR_COLUMN", "embedding")
DEFAULT_CREATED_AT_COL = os.getenv("CREATED_AT_COLUMN", "created_at")
DEFAULT_CLUSTER_ID_COL = os.getenv("CLUSTER_ID_COLUMN", "cluster_id")
DEFAULT_CLUSTER_PROB_COL = os.getenv("CLUSTER_PROB_COLUMN", "cluster_prob")
DEFAULT_OUTLIER_SCORE_COL = os.getenv("OUTLIER_SCORE_COLUMN", "outlier_score")

# pagination defaults
DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", "5000"))
BATCH_WRITE_SIZE = int(os.getenv("BATCH_WRITE_SIZE", "5000"))
READ_PAGE_SIZE = int(os.getenv("READ_PAGE_SIZE", "5000"))

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
app = FastAPI(title="HDBSCAN for Supabase via n8n")

# -------- models ----------
class ClusterRequest(BaseModel):
    table: str = DEFAULT_TABLE
    id_column: str = DEFAULT_ID_COL
    vector_column: str = DEFAULT_VECTOR_COL
    created_at_column: str = DEFAULT_CREATED_AT_COL

    # Filtering
    where: Optional[str] = None               # raw PostgREST filter, e.g. "source=eq.news"
    days: Optional[int] = None                # e.g. last N days by created_at
    limit: Optional[int] = None               # hard cap on total rows

    # HDBSCAN params
    min_cluster_size: int = 25
    min_samples: Optional[int] = None
    metric: str = "euclidean"

    # Write-back params
    cluster_id_column: str = DEFAULT_CLUSTER_ID_COL
    cluster_prob_column: str = DEFAULT_CLUSTER_PROB_COL
    outlier_score_column: str = DEFAULT_OUTLIER_SCORE_COL
    write_back: bool = True

    # Memory & batching
    read_page_size: int = READ_PAGE_SIZE
    batch_write_size: int = BATCH_WRITE_SIZE

class ClusterResponse(BaseModel):
    count: int
    labels: List[int]
    probabilities: List[float]
    outlier_scores: List[float]

# -------- helpers ----------
def _parse_vector(v: Any) -> List[float]:
    """
    Normalize pgvector into python list[float].
    Handles cases:
      - already list (ideal)
      - '{1.0,2.0,...}' string
      - '(1.0,2.0,...)' string (rare)
    """
    if isinstance(v, list):
        return [float(x) for x in v]
    if isinstance(v, str):
        s = v.strip()
        if s.startswith('{') and s.endswith('}'):
            s = s[1:-1]
        elif s.startswith('(') and s.endswith(')'):
            s = s[1:-1]
        if not s:
            return []
        return [float(x) for x in s.split(',')]
    raise ValueError(f"Unsupported vector type: {type(v)}")

def _select_columns(id_col: str, vec_col: str, created_at_col: str) -> str:
    """
    We ask PostgREST to cast vector to float8[] for safer JSON (when supported).
    If cast fails in your project, drop `::float8[]` and rely on _parse_vector.
    """
    return f"{id_col},{vec_col}::{'float8[]'} as {vec_col},{created_at_col}"

def _apply_filters(q, req: ClusterRequest):
    # time window
    if req.days is not None and req.days > 0:
        # created_at >= now() - interval 'N days'
        # PostgREST: created_at=gte.2024-01-01T00:00:00Z — but we don't have server time here.
        # Simpler: rely on SQL via rpc or use `gte` with ISO now-req.days in client.
        # We'll compute timestamp here:
        from datetime import datetime, timedelta, timezone
        since = datetime.now(timezone.utc) - timedelta(days=req.days)
        q = q.gte(req.created_at_column, since.isoformat())

    # raw filter operators (comma-separated supported by PostgREST `and=` are tricky);
    # we allow one `where` and let user encode multiple with `and=(...)` if needed.
    if req.where:
        # format: "field=op.value", e.g. "source=eq.news"
        # You can chain more filters in n8n by calling multiple times or using and=
        parts = req.where.split('=')
        if len(parts) >= 3:
            field = parts[0]
            op = parts[1]
            value = '='.join(parts[2:])
            q = q.filter(field, f"{op}.{value}")
        else:
            # try pass-through (if user provided e.g. "and=(source.eq.news,status.eq.published)")
            q = q.filter("and", f"({req.where})")
    return q

def fetch_rows(req: ClusterRequest) -> Tuple[List[Any], List[List[float]]]:
    table = supabase.table(req.table)
    select_str = _select_columns(req.id_column, req.vector_column, req.created_at_column)

    # page through results
    start = 0
    page = req.read_page_size
    max_total = req.limit or DEFAULT_LIMIT

    ids: List[Any] = []
    vecs: List[List[float]] = []

    while start < max_total:
        end = min(start + page - 1, max_total - 1)
        q = table.select(select_str).order(req.created_at_column, desc=False).range(start, end)
        q = _apply_filters(q, req)
        res = q.execute()
        chunk = res.data or []
        if not chunk:
            break
        for row in chunk:
            try:
                ids.append(row[req.id_column])
                vecs.append(_parse_vector(row[req.vector_column]))
            except Exception as e:
                # skip malformed row
                print(f"Skipping row due to parse error: {e}")
        if len(chunk) < page:
            break
        start += page

    return ids, vecs

def write_back(
    req: ClusterRequest,
    ids: List[Any],
    labels: np.ndarray,
    probs: np.ndarray,
    outliers: np.ndarray,
):
    table = supabase.table(req.table)
    payloads = []
    for i, _id in enumerate(ids):
        payloads.append({
            req.id_column: _id,
            req.cluster_id_column: int(labels[i]),
            req.cluster_prob_column: float(probs[i]),
            req.outlier_score_column: float(outliers[i]),
        })
        if len(payloads) >= req.batch_write_size:
            table.upsert(payloads, on_conflict=req.id_column, returning="minimal").execute()
            payloads = []

    if payloads:
        table.upsert(payloads, on_conflict=req.id_column, returning="minimal").execute()

# -------- endpoints ----------
@app.get("/healthz")
def healthz():
    # quick auth + table existence check
    try:
        supabase.table(DEFAULT_TABLE).select(DEFAULT_ID_COL).limit(1).execute()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/cluster", response_model=ClusterResponse)
def cluster(req: ClusterRequest):
    try:
        ids, vecs = fetch_rows(req)
        if not ids:
            raise HTTPException(400, "No rows selected with current filters.")

        X = np.asarray(vecs, dtype=np.float32)
        if X.ndim != 2 or X.shape[0] != len(ids):
            raise HTTPException(500, "Vector shape mismatch.")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=req.min_cluster_size,
            min_samples=req.min_samples,
            metric=req.metric,
            prediction_data=True
        ).fit(X)

        labels = clusterer.labels_
        probs = clusterer.probabilities_
        outliers = clusterer.outlier_scores_

        if req.write_back:
            write_back(req, ids, labels, probs, outliers)

        return ClusterResponse(
            count=len(ids),
            labels=labels.tolist(),
            probabilities=probs.tolist(),
            outlier_scores=outliers.tolist(),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Clustering failed: {e}")
