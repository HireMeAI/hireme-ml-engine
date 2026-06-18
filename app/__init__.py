"""HireMe AI — Moteur Python d'anonymisation et de matching débiaisé.

Ce paquet implémente le cœur scientifique décrit dans le dossier PFE (BC04B) :
  - `anonymizer` : retrait des identifiants directs et indirects AVANT tout NLP ;
  - `matcher`    : vectorisation TF-IDF + similarité cosinus, explicable (XAI léger).

Il est invoqué par le MatchingService (Java) via l'API REST exposée dans `main.py`.
"""

__version__ = "0.1.0"
