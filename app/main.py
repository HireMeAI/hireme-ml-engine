"""API REST du moteur ML — invoquée par le MatchingService (Java).

Le moteur n'expose aucune route publique : il est accessible uniquement depuis le
réseau Docker interne (§5.2). La Gateway ne le route pas.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import __version__
from .anonymizer import anonymise
from .matcher import compute_score, match_resume_to_jobs

app = FastAPI(title="HireMe AI — ML Engine", version=__version__)


class JobIn(BaseModel):
    id: str
    text: str


class MatchRequest(BaseModel):
    resume_text: str = Field(..., min_length=1)
    jobs: list[JobIn]
    known_pii: list[str] = Field(default_factory=list)
    top_n: int = 10


class ScoreRequest(BaseModel):
    resume_text: str
    job_text: str
    known_pii: list[str] = Field(default_factory=list)


@app.get("/health")
def health() -> dict:
    return {"status": "UP", "version": __version__}


@app.post("/anonymise")
def anonymise_endpoint(req: ScoreRequest) -> dict:
    result = anonymise(req.resume_text, known_pii=req.known_pii)
    return {"text": result.text, "removed": result.removed}


@app.post("/score")
def score_endpoint(req: ScoreRequest) -> dict:
    anon = anonymise(req.resume_text, known_pii=req.known_pii)
    return {"score": round(compute_score(anon.text, req.job_text), 4)}


@app.post("/match")
def match_endpoint(req: MatchRequest) -> dict:
    results = match_resume_to_jobs(
        req.resume_text,
        [j.model_dump() for j in req.jobs],
        known_pii=req.known_pii,
        top_n=req.top_n,
    )
    return {
        "matches": [
            {"job_id": r.job_id, "score": r.score, "shared_terms": r.shared_terms}
            for r in results
        ]
    }
