"""Same-origin HTTP API and static frontend for the research workspace."""
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .config import ROOT
from . import workspace


class PairRequest(BaseModel):
    peptide: str = Field(max_length=100)
    allele: str = Field(max_length=50)
    include_plm: bool = True


class ScanRequest(BaseModel):
    sequence: str = Field(max_length=10000)
    alleles: list[str] = Field(min_length=1, max_length=workspace.MAX_ALLELES)
    arm: str = 'blosum_nn'


class BatchRequest(BaseModel):
    text: str = Field(max_length=30000)
    arm: str = 'blosum_nn'


class CompareRequest(BaseModel):
    pairs: list[PairRequest] = Field(min_length=1, max_length=30)
    include_plm: bool = True


def create_app():
    app = FastAPI(title='Peptide–HLA Stability Workspace', docs_url='/api/docs')

    @app.exception_handler(ValueError)
    async def invalid_input(request, exc):
        return JSONResponse({'detail':str(exc)}, status_code=400)

    @app.exception_handler(RuntimeError)
    async def unavailable_model(request, exc):
        return JSONResponse({'detail':str(exc)}, status_code=503)

    @app.exception_handler(OSError)
    async def unavailable_resource(request, exc):
        return JSONResponse({'detail':'Model resources could not be loaded. For new peptides, ESM-2 needs its public checkpoint and network access on first use. Try the sequence models only.'}, status_code=503)

    @app.get('/')
    def index():
        return FileResponse(ROOT/'web/index.html')

    @app.get('/api/bootstrap')
    def bootstrap():
        return workspace.bootstrap()

    @app.post('/api/predict')
    def predict(body: PairRequest):
        return workspace.predict_pair(body.peptide, body.allele, body.include_plm)

    @app.post('/api/scan')
    def scan(body: ScanRequest):
        return workspace.scan(body.sequence, body.alleles, body.arm)

    @app.post('/api/batch')
    def batch(body: BatchRequest):
        if body.arm not in workspace.BASELINES:
            raise ValueError('Batch ranking uses a sequence baseline. Compare selected pairs with ESM-2 separately.')
        pairs = workspace.parse_batch(body.text)
        return {'arm':body.arm, 'rows':workspace.score_pairs(pairs, [body.arm])}

    @app.post('/api/compare')
    def compare(body: CompareRequest):
        arms = list(workspace.MODEL_NAMES) if body.include_plm else workspace.BASELINES
        return {'rows':workspace.score_pairs([(p.peptide,p.allele) for p in body.pairs], arms)}

    app.mount('/assets', StaticFiles(directory=ROOT/'web'), name='assets')
    return app
