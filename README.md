# HireMe AI — Moteur ML (anonymisation + matching débiaisé)

Cœur scientifique de la plateforme (BC04B). Invoqué par le `MatchingService` (Java) via
REST ; aucune route publique (réseau Docker interne uniquement).

## Contenu

| Fichier | Rôle |
| :--- | :--- |
| `app/anonymizer.py` | Anonymisation amont : retrait des identifiants directs/indirects **avant** tout NLP (§6.4). |
| `app/matcher.py` | Vectorisation TF-IDF + similarité cosinus, explicable (§6.5). |
| `app/main.py` | API FastAPI : `/health`, `/anonymise`, `/score`, `/match`. |
| `tests/` | Tests pytest, dont le test anti-biais (écart de score nul). |
| `demo.py` | Démonstration reproductible des scénarios §9.5. |

## Lancer en local

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest -q          # exécute la suite de tests
python demo.py     # rejoue les scénarios candidat + anti-biais
uvicorn app.main:app --reload   # démarre l'API sur :8000
```

## Limites assumées (cf. dossier §6.6 / §10.2-10.3)

- Anonymisation **par règles** (regex) : un NER spécialisé renforcerait la détection des
  proxys indirects (ex. ville isolée, loisir de niche). Non implémenté volontairement.
- Scores TF-IDF bruts sur textes courts : modestes par nature. La **calibration** (régression
  isotonique) annoncée nécessite un jeu de données labellisé réel — non encore disponible.
- Synonymie/contexte non capturés (« management » ≠ « leadership ») : évolution embeddings (V2).
