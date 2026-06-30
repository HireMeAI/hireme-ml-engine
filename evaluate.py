"""Évaluation quantitative du moteur de matching sur les données du seeder.

La vérité terrain provient du seeder (hireme/seeder/seed.mjs), qui peuple la plateforme
réelle (Gateway → services → DB) puis exporte `golden_set.csv`. Chaque ligne est un couple
(CV, offre) avec son étiquette :

    pertinent = 1  si  domaine(CV) == domaine(offre)   sinon 0

On compare cette vérité aux scores du modèle (`compute_score` / `match_resume_to_jobs`) :
  - métriques de classification (au seuil τ) : précision, rappel, F1, matrice de confusion ;
  - métriques indépendantes du seuil : ROC-AUC, PR-AUC ;
  - métriques de classement (usage réel Top-N) : Precision@k, Recall@k, MRR, NDCG@k ;
  - un contrôle d'équité : écart de score à compétences égales, noms différents.

Pré-requis : avoir lancé le seeder pour générer le golden_set.csv :
    cd ../hireme && docker compose up -d
    cd seeder && node seed.mjs        # écrit seeder/golden_set.csv

Usage :
    python evaluate.py                                   # chemin par défaut du seeder
    python evaluate.py --csv ..hireme/seeder/golden_set.csv --threshold 0.05 --k 5
    python evaluate.py --show --detail 0                 # voir CV/offres + scoring détaillé
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ndcg_score,
)

from app.matcher import compute_score, match_resume_to_jobs

# Emplacement par défaut du golden_set.csv produit par le seeder (repo voisin).
DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "..", "hireme", "seeder", "golden_set.csv")


def load_dataset_from_csv(path: str):
    """Charge le golden_set.csv produit par le seeder (données réelles de la pipeline).

    Colonnes attendues : cv_domain, job_domain, relevant, cv_text, job_text.
    On reconstruit les listes candidats/offres en dédupliquant par texte ; la pertinence
    est portée par l'égalité de domaine (cohérente avec la colonne `relevant`).
    """
    candidates, jobs = {}, {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cv_text, job_text = row["cv_text"], row["job_text"]
            if cv_text not in candidates:
                candidates[cv_text] = {"domain": row["cv_domain"], "text": cv_text}
            if job_text not in jobs:
                jobs[job_text] = {
                    "id": f"job-{len(jobs)}",
                    "domain": row["job_domain"],
                    "text": job_text,
                }
    return list(candidates.values()), list(jobs.values())


def score_pairs(candidates, jobs):
    """Calcule, pour chaque couple (CV, offre), l'étiquette réelle et le score du modèle."""
    y_true, y_score = [], []
    for c in candidates:
        for j in jobs:
            y_true.append(1 if c["domain"] == j["domain"] else 0)
            y_score.append(compute_score(c["text"], j["text"]))
    return y_true, y_score


def classification_metrics(y_true, y_score, threshold: float):
    """Évalue les couples (CV, offre) comme une classification binaire au seuil τ."""
    y_pred = [1 if s >= threshold else 0 for s in y_score]

    return {
        "n_pairs": len(y_true),
        "n_relevant": sum(y_true),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "confusion": confusion_matrix(y_true, y_pred),
    }


def ranking_metrics(candidates, jobs, k: int):
    """Pour chaque CV, classe toutes les offres et moyenne les métriques de ranking."""
    precisions, recalls, mrrs, ndcgs = [], [], [], []
    job_texts = [{"id": j["id"], "text": j["text"]} for j in jobs]

    for c in candidates:
        n_relevant = sum(1 for j in jobs if j["domain"] == c["domain"])
        if n_relevant == 0:
            continue
        ranked = match_resume_to_jobs(c["text"], job_texts, top_n=len(jobs))
        rel_by_id = {j["id"]: (1 if j["domain"] == c["domain"] else 0) for j in jobs}
        ranked_rel = [rel_by_id[r.job_id] for r in ranked]

        topk = ranked_rel[:k]
        precisions.append(sum(topk) / k)
        recalls.append(sum(topk) / n_relevant)
        first = next((i for i, r in enumerate(ranked_rel) if r == 1), None)
        mrrs.append(1.0 / (first + 1) if first is not None else 0.0)

        y_true = [rel_by_id[r.job_id] for r in ranked]
        y_score = [r.score for r in ranked]
        ndcgs.append(ndcg_score([y_true], [y_score], k=k))

    n = len(precisions)
    avg = lambda xs: sum(xs) / n if n else 0.0
    return {
        "queries": n,
        "k": k,
        "precision_at_k": avg(precisions),
        "recall_at_k": avg(recalls),
        "mrr": avg(mrrs),
        "ndcg_at_k": avg(ndcgs),
    }


def fairness_check():
    """Écart de score à compétences identiques, noms d'origines différentes"""
    job = [{"id": "job-1", "text": "developpeur java spring docker microservices kubernetes"}]
    cv_a = "Jean Dupont, 28 ans, Paris, developpeur java spring docker microservices"
    cv_b = "Mohammed Al-Hassan, 31 ans, Marseille, developpeur java spring docker microservices"
    sa = match_resume_to_jobs(cv_a, job, known_pii=["Jean", "Dupont"])[0].score
    sb = match_resume_to_jobs(cv_b, job, known_pii=["Mohammed", "Al-Hassan"])[0].score
    return sa, sb, abs(sa - sb)


def make_plots(y_true, y_score, threshold: float, out_path: str):
    """Génère une figure récapitulative (ROC, précision-rappel, balayage du seuil,
    matrice de confusion) et l'enregistre en PNG pour le dossier."""
    import matplotlib
    matplotlib.use("Agg")  # backend sans interface (sauvegarde fichier uniquement)
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import (
        roc_curve, auc, precision_recall_curve, average_precision_score,
        confusion_matrix,
    )

    yt = np.array(y_true)
    ys = np.array(y_score)
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Évaluation du moteur de matching HireMe", fontsize=14, fontweight="bold")

    # (1) Courbe ROC
    fpr, tpr, _ = roc_curve(yt, ys)
    ax[0, 0].plot(fpr, tpr, color="#1f3864", lw=2, label=f"AUC = {auc(fpr, tpr):.3f}")
    ax[0, 0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax[0, 0].set(title="Courbe ROC", xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs")
    ax[0, 0].legend(loc="lower right")

    # (2) Courbe précision-rappel
    prec, rec, _ = precision_recall_curve(yt, ys)
    ax[0, 1].plot(rec, prec, color="#c95336", lw=2, label=f"AP = {average_precision_score(yt, ys):.3f}")
    ax[0, 1].set(title="Courbe précision-rappel", xlabel="Rappel", ylabel="Précision")
    ax[0, 1].legend(loc="lower left")

    # (3) Balayage du seuil τ : précision / rappel / F1
    taus = np.linspace(0.01, 0.6, 60)
    P, R, F = [], [], []
    for t in taus:
        yp = (ys >= t).astype(int)
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        P.append(p); R.append(r); F.append(2 * p * r / (p + r) if p + r else 0.0)
    ax[1, 0].plot(taus, P, label="Précision", color="#1f3864")
    ax[1, 0].plot(taus, R, label="Rappel", color="#c95336")
    ax[1, 0].plot(taus, F, label="F1", color="#2e7d32")
    ax[1, 0].axvline(threshold, ls="--", color="grey", label=f"τ = {threshold}")
    ax[1, 0].set(title="Précision / rappel / F1 selon le seuil τ", xlabel="Seuil τ", ylabel="Score")
    ax[1, 0].legend(loc="best")

    # (4) Matrice de confusion au seuil choisi
    yp = (ys >= threshold).astype(int)
    cm = confusion_matrix(yt, yp)
    im = ax[1, 1].imshow(cm, cmap="Blues")
    ax[1, 1].set(title=f"Matrice de confusion (τ = {threshold})",
                 xticks=[0, 1], yticks=[0, 1],
                 xlabel="Prédit", ylabel="Réel")
    ax[1, 1].set_xticklabels(["non", "pertinent"])
    ax[1, 1].set_yticklabels(["non", "pertinent"])
    for (r_, c_), v in np.ndenumerate(cm):
        ax[1, 1].text(c_, r_, str(v), ha="center", va="center",
                      color="white" if v > cm.max() / 2 else "black", fontsize=12)
    fig.colorbar(im, ax=ax[1, 1], fraction=0.046, pad=0.04)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def show_dataset(candidates, jobs, detail_for: int = 0):
    """Affiche les données du seeder (CV + offres) et le détail de scoring d'un candidat."""
    print("\n" + "=" * 60)
    print("ÉTAPE 1 — OFFRES (seeder)")
    print("=" * 60)
    for j in jobs:
        print(f"  [{j['domain']:<8}] {j['id']:<12} | {j['text']}")

    print("\n" + "=" * 60)
    print("ÉTAPE 2 — CV (seeder)")
    print("=" * 60)
    for i, c in enumerate(candidates):
        print(f"  CV#{i:<3} [{c['domain']:<8}] | {c['text']}")

    # Détail du scoring pour UN candidat : son score face à chaque offre.
    c = candidates[detail_for]
    print("\n" + "=" * 60)
    print(f"ÉTAPE 3 — SCORING DÉTAILLÉ DU CV#{detail_for} (domaine {c['domain']})")
    print("=" * 60)
    print(f"  CV : {c['text']}")
    print(f"  {'offre':<14} {'domaine':<9} {'attendu':<9} {'score':>7}")
    print("  " + "-" * 42)
    rows = []
    for j in jobs:
        score = compute_score(c["text"], j["text"])
        attendu = "MATCH" if j["domain"] == c["domain"] else "non"
        rows.append((j["id"], j["domain"], attendu, score))
    # Tri par score décroissant : on voit ce que le modèle remonterait en premier.
    for jid, dom, attendu, score in sorted(rows, key=lambda r: -r[3]):
        flag = "✓" if attendu == "MATCH" else " "
        print(f"  {flag} {jid:<12} {dom:<9} {attendu:<9} {score:>6.3f}")


def main():
    parser = argparse.ArgumentParser(description="Évaluation du moteur de matching HireMe (données du seeder).")
    parser.add_argument("--csv", default=DEFAULT_CSV,
                        help="Chemin du golden_set.csv produit par le seeder (défaut : ../hireme/seeder/golden_set.csv).")
    parser.add_argument("--threshold", type=float, default=0.10, help="Seuil τ de décision pertinent/non.")
    parser.add_argument("--k", type=int, default=5, help="Profondeur des métriques de classement (Top-k).")
    parser.add_argument("--show", action="store_true",
                        help="Affiche les CV et offres du seeder + le détail de scoring d'un candidat.")
    parser.add_argument("--detail", type=int, default=0,
                        help="Index du candidat dont on détaille le scoring (avec --show).")
    parser.add_argument("--plot", nargs="?", const="reports/evaluation.png", default=None,
                        help="Génère un graphe PNG (ROC, précision-rappel, seuil, matrice). "
                             "Optionnel : chemin de sortie (défaut : reports/evaluation.png).")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"❌ golden_set introuvable : {args.csv}\n"
              f"   Lancez d'abord le seeder pour le générer :\n"
              f"     cd ../hireme && docker compose up -d\n"
              f"     cd seeder && node seed.mjs", file=sys.stderr)
        sys.exit(1)

    candidates, jobs = load_dataset_from_csv(args.csv)
    if not candidates or not jobs:
        print(f"❌ golden_set vide : {args.csv} (aucun CV/offre). Relancez le seeder sur une DB fraîche.",
              file=sys.stderr)
        sys.exit(1)

    if args.show:
        show_dataset(candidates, jobs, detail_for=args.detail)

    print("\n" + "=" * 60)
    print("ÉVALUATION DU MOTEUR DE MATCHING — HireMe")
    print("=" * 60)
    print(f"Source    : seeder → {os.path.relpath(args.csv)}")
    print(f"Candidats : {len(candidates)}   Offres : {len(jobs)}   Seuil τ : {args.threshold}")

    y_true, y_score = score_pairs(candidates, jobs)
    clf = classification_metrics(y_true, y_score, args.threshold)
    print("\n--- Classification (couples CV×offre, seuil τ) ---")
    print(f"  Couples évalués : {clf['n_pairs']}  (pertinents : {clf['n_relevant']})")
    print(f"  Précision : {clf['precision']:.3f}")
    print(f"  Rappel    : {clf['recall']:.3f}")
    print(f"  F1-score  : {clf['f1']:.3f}")
    print(f"  ROC-AUC   : {clf['roc_auc']:.3f}   (indépendant du seuil)")
    print(f"  PR-AUC    : {clf['pr_auc']:.3f}   (indépendant du seuil)")
    tn, fp, fn, tp = clf["confusion"].ravel()
    print(f"  Matrice   : TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    rk = ranking_metrics(candidates, jobs, args.k)
    print(f"\n--- Classement / recommandation Top-{rk['k']} ({rk['queries']} CV) ---")
    print(f"  Precision@{rk['k']} : {rk['precision_at_k']:.3f}")
    print(f"  Recall@{rk['k']}    : {rk['recall_at_k']:.3f}")
    print(f"  MRR          : {rk['mrr']:.3f}")
    print(f"  NDCG@{rk['k']}      : {rk['ndcg_at_k']:.3f}")

    sa, sb, gap = fairness_check()
    print("\n--- Équité (compétences égales, noms différents) ---")
    print(f"  Score A : {sa*100:.2f}%   Score B : {sb*100:.2f}%   Écart : {gap*100:.4f}%  (objectif < 5%)")

    if args.plot:
        path = make_plots(y_true, y_score, args.threshold, args.plot)
        print(f"\n📊 Graphe enregistré : {os.path.relpath(path)}")

    print("\n" + "=" * 60)
    print("Vérité terrain issue du seeder (domaines disjoints) — preuve de fonctionnement")
    print("et de non-régression ; un jeu annoté par des recruteurs réels resterait l'étape")
    print("=" * 60)


if __name__ == "__main__":
    main()
