
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .anonymizer import anonymise


_STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "en", "a", "au", "aux",
    "pour", "avec", "dans", "sur", "par", "ce", "cette", "ces", "son", "sa", "ses",
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are",
    # Mots de structure / Boilerplate à ignorer dans le matching et XAI
    "competences", "competence", "skills", "skill",
    "experiences", "experience", "formations", "formation",
    "langues", "langue", "languages", "language",
    "description", "details", "requises", "requis", "required",
}


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if len(t) > 2 and t not in _STOPWORDS]
    return " ".join(tokens)


@dataclass
class MatchExplanation:
    job_id: str
    score: float  # 0.0 – 1.0
    shared_terms: list[str]


def _build_vectorizer(corpus: Sequence[str]) -> TfidfVectorizer:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    vec.fit(corpus)
    return vec


def compute_score(resume_text: str, job_text: str) -> float:
    """Similarité cosinus TF-IDF entre un CV et une offre, dans [0, 1]."""
    a, b = _normalize(resume_text), _normalize(job_text)
    if not a or not b:
        return 0.0
    vec = _build_vectorizer([a, b])
    matrix = vec.transform([a, b])
    return float(cosine_similarity(matrix[0], matrix[1])[0][0])


def _shared_terms(norm_resume: str, norm_job: str, top_n: int = 5) -> list[str]:
    """Termes communs (sur textes déjà normalisés), pour expliquer un score."""
    a, b = set(norm_resume.split()), set(norm_job.split())
    shared = a & b
    # Tri stable par longueur décroissante (heuristique : termes spécifiques d'abord).
    return sorted(shared, key=lambda t: (-len(t), t))[:top_n]


def _corpus_scores(norm_resume: str, norm_jobs: Sequence[str]) -> list[float]:
 
    if not norm_jobs:
        return []
    if not norm_resume or not any(norm_jobs):
        return [0.0] * len(norm_jobs)

    corpus = [norm_resume, *norm_jobs]
    vec = _build_vectorizer(corpus)
    matrix = vec.transform(corpus)
    sims = cosine_similarity(matrix[0], matrix[1:])[0]
    return [float(s) for s in sims]


def match_resume_to_jobs(
    resume_text: str,
    jobs: Sequence[dict],
    known_pii: Sequence[str] | None = None,
    top_n: int = 10,
) -> list[MatchExplanation]:

    anon = anonymise(resume_text, known_pii=known_pii)
    norm_resume = _normalize(anon.text)
    jobs = list(jobs)
    norm_jobs = [_normalize(job.get("text", "")) for job in jobs]

    scores = _corpus_scores(norm_resume, norm_jobs)

    results = [
        MatchExplanation(
            job_id=str(job.get("id", "")),
            score=round(score, 4),
            shared_terms=_shared_terms(norm_resume, norm_job),
        )
        for job, norm_job, score in zip(jobs, norm_jobs, scores)
    ]
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_n]
