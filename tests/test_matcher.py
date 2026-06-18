"""Tests du moteur de matching TF-IDF + cosinus, dont le test anti-biais central."""

import pytest

from app.matcher import compute_score, match_resume_to_jobs


def test_identical_text_scores_one():
    profile = "java spring boot microservices docker kafka"
    assert compute_score(profile, profile) == pytest.approx(1.0, abs=1e-3)


def test_unrelated_text_scores_low():
    resume = "developpeur java spring backend microservices"
    job = "boulanger patisserie vente boutique horaires"
    assert compute_score(resume, job) < 0.15


def test_partial_overlap_scores_between():
    resume = "developpeur python data machine learning pandas"
    job = "data scientist python pandas scikit learn statistiques"
    score = compute_score(resume, job)
    assert 0.1 < score < 1.0


def test_anti_bias_identical_skills_different_names_same_score():
    """Cœur de la promesse (§6.5, scénario 2) : deux CV aux compétences identiques
    mais aux noms d'origines différentes obtiennent un écart de score NUL après
    anonymisation."""
    job = [{"id": "job-1", "text": "developpeur java spring docker microservices kubernetes"}]

    cv_a = "Jean Dupont, 28 ans, Paris, developpeur java spring docker microservices"
    cv_b = "Mohammed Al-Hassan, 31 ans, Marseille, developpeur java spring docker microservices"

    res_a = match_resume_to_jobs(cv_a, job, known_pii=["Jean", "Dupont"])
    res_b = match_resume_to_jobs(cv_b, job, known_pii=["Mohammed", "Al-Hassan"])

    score_a, score_b = res_a[0].score, res_b[0].score
    ecart = abs(score_a - score_b)
    # Objectif dossier : écart < 5 % ; idéalement nul grâce à l'anonymisation.
    assert ecart < 0.05
    assert score_a == pytest.approx(score_b, abs=1e-6)


def test_ranking_orders_by_relevance():
    resume = "developpeur java spring boot backend microservices"
    jobs = [
        {"id": "j1", "text": "developpeur java spring microservices backend"},
        {"id": "j2", "text": "graphiste illustrator photoshop creation visuelle"},
        {"id": "j3", "text": "ingenieur backend java api rest spring"},
    ]
    results = match_resume_to_jobs(resume, jobs)
    assert results[0].job_id in {"j1", "j3"}
    assert results[-1].job_id == "j2"


def test_explanation_contains_shared_terms():
    resume = "developpeur java spring docker"
    jobs = [{"id": "j1", "text": "poste java spring docker kubernetes"}]
    results = match_resume_to_jobs(resume, jobs)
    assert "java" in results[0].shared_terms
    assert "spring" in results[0].shared_terms
