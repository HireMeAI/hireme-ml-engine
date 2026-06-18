"""Matching vectoriel débiaisé — TF-IDF + similarité cosinus, explicable.

Implémente le moteur décrit en §6.5 du dossier :
  - vectorisation TF-IDF (scikit-learn) du CV anonymisé et des offres ;
  - similarité cosinus CV ↔ offre ;
  - explication XAI légère : termes partagés ayant le plus pesé dans le score.

Choix assumé (§6.2) : TF-IDF retenu pour son explicabilité, exigée par l'IA Act,
plutôt qu'un Transformer « boîte noire ». Les limites (synonymie, contexte) sont
documentées en §6.6 / §10.2.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .anonymizer import anonymise

# Stop-words FR/EN minimaux (évite la dépendance NLTK au démarrage ; cf. preprocess.py
# pour le pipeline ETL complet). Volontairement court : la vectorisation pondère déjà
# les termes fréquents via l'IDF.
_STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "en", "a", "au", "aux",
    "pour", "avec", "dans", "sur", "par", "ce", "cette", "ces", "son", "sa", "ses",
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are",
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


def _shared_terms(resume_text: str, job_text: str, top_n: int = 5) -> list[str]:
    a, b = set(_normalize(resume_text).split()), set(_normalize(job_text).split())
    shared = a & b
    # Tri stable par longueur décroissante (heuristique : termes spécifiques d'abord).
    return sorted(shared, key=lambda t: (-len(t), t))[:top_n]


def match_resume_to_jobs(
    resume_text: str,
    jobs: Sequence[dict],
    known_pii: Sequence[str] | None = None,
    top_n: int = 10,
) -> list[MatchExplanation]:
    """Anonymise le CV puis le classe contre une liste d'offres.

    L'anonymisation est OBLIGATOIRE et BLOQUANTE (§6.4) : elle est appliquée ici,
    avant toute vectorisation, sur le texte du CV.

    :param jobs: liste de dicts {"id": str, "text": str}.
    :return: explications triées par score décroissant (Top-N).
    """
    anon = anonymise(resume_text, known_pii=known_pii)
    results: list[MatchExplanation] = []
    for job in jobs:
        job_text = job.get("text", "")
        score = compute_score(anon.text, job_text)
        results.append(
            MatchExplanation(
                job_id=str(job.get("id", "")),
                score=round(score, 4),
                shared_terms=_shared_terms(anon.text, job_text),
            )
        )
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_n]
