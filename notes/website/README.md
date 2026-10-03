# Optional pHLASP website

From the repository root, run `python -m notes.website.app`. This serves the existing workspace at http://127.0.0.1:7860 and its original Gradio interface at `/legacy`. API documentation is at `/api/docs`.

The website still exposes the original five predictors. The two new ESM-2 + MLP approaches are benchmark-only; the interface does not silently substitute models. Frontend assets remain in [web/](../../web/), and vectorized inference in [src/workspace.py](../../src/workspace.py).
