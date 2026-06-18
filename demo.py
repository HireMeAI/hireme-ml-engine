"""Démonstration reproductible du moteur de matching débiaisé.

Usage :
    python demo.py

Reproduit les scénarios 1 et 2 du dossier (§9.5) sur des données réelles, sans serveur.
"""

from app.anonymizer import anonymise
from app.matcher import match_resume_to_jobs

JOBS = [
    {"id": "lead-backend-java", "text": "Lead Developer Backend Java Spring Boot microservices Docker Kubernetes API REST"},
    {"id": "fullstack-react", "text": "Fullstack Developer React TypeScript Node Java Spring frontend backend"},
    {"id": "devops-junior", "text": "DevOps Junior CI CD Docker Kubernetes monitoring infrastructure cloud"},
    {"id": "graphiste", "text": "Graphiste UI Illustrator Photoshop creation visuelle identite de marque"},
]


def main():
    print("=== Scénario 1 — Parcours candidat (Emma) ===")
    cv = "Emma Martin, 23 ans, Paris 11e. Stage fullstack Java Spring, projet open source React, Docker."
    for m in match_resume_to_jobs(cv, JOBS, known_pii=["Emma", "Martin"]):
        print(f"  {m.job_id:20s}  {m.score*100:5.1f}%   termes: {', '.join(m.shared_terms)}")

    print("\n=== Scénario 2 — Anti-biais (mêmes compétences, noms différents) ===")
    job = [{"id": "lead-backend-java", "text": JOBS[0]["text"]}]
    cv_a = "Jean Dupont, 28 ans, Paris, developpeur java spring docker microservices"
    cv_b = "Mohammed Al-Hassan, 31 ans, Marseille, developpeur java spring docker microservices"
    sa = match_resume_to_jobs(cv_a, job, known_pii=["Jean", "Dupont"])[0].score
    sb = match_resume_to_jobs(cv_b, job, known_pii=["Mohammed", "Al-Hassan"])[0].score
    print(f"  Jean Dupont      → {sa*100:.2f}%")
    print(f"  Mohammed Al-Hassan → {sb*100:.2f}%")
    print(f"  Écart de score   → {abs(sa-sb)*100:.4f}%  (objectif < 5%)")

    print("\n=== Anonymisation appliquée (CV de Jean) ===")
    r = anonymise(cv_a, known_pii=["Jean", "Dupont"])
    print(f"  Avant : {cv_a}")
    print(f"  Après : {r.text}")
    print(f"  Retiré: {r.removed}")


if __name__ == "__main__":
    main()
