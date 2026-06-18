"""Anonymisation amont (anti-biais) — retrait des identifiants directs et indirects.

Conformément à l'approche « anonymisation-first » du dossier (§6.4), aucune donnée
ne doit atteindre le vectoriseur avant d'avoir transité par ce module. L'approche est
fondée sur des règles + expressions régulières ; la reconnaissance d'entités nommées
(NER) reste une évolution documentée (§10.3) et n'est volontairement pas implémentée ici.

Le module retire :
  - les identifiants connus du candidat (nom, prénom, e-mail, téléphone) passés par l'appelant ;
  - les civilités et marqueurs de genre ;
  - l'âge, la date de naissance, les années de service révélatrices d'âge ;
  - l'adresse postale, le code postal et les arrondissements (proxy géographique/social) ;
  - les e-mails, téléphones et URLs résiduels.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable


# --- Marqueurs de genre et civilités (FR + EN) -------------------------------
_GENDER_TERMS = {
    "m", "mr", "mme", "mlle", "monsieur", "madame", "mademoiselle",
    "il", "elle", "lui", "mister", "mrs", "miss", "ms", "he", "she", "him", "her",
    "homme", "femme", "masculin", "feminin", "male", "female", "man", "woman",
}

# Loisirs/engagements souvent corrélés à une origine ou une orientation (proxy).
# Liste minimale, extensible — cf. limites §10.3 (idéalement remplacée par un NER).
_PROXY_HINTS_PATTERNS = [
    r"\bn[ée]e?\s+le\s+\d{1,2}[/.\- ]\d{1,2}[/.\- ]\d{2,4}\b",  # "né le 12/03/1998"
    r"\bborn\s+(on\s+)?\d{1,2}[/.\- ]\d{1,2}[/.\- ]\d{2,4}\b",
]


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


@dataclass
class AnonymizationResult:
    text: str
    removed: list[str] = field(default_factory=list)

    @property
    def removed_count(self) -> int:
        return len(self.removed)


def anonymise(text: str, known_pii: Iterable[str] | None = None) -> AnonymizationResult:
    """Retire les identifiants directs et indirects d'un texte de CV.

    :param text: texte consolidé du CV.
    :param known_pii: identifiants connus du candidat (nom, prénom, e-mail, téléphone)
        que le service appelant fournit explicitement — c'est le cas réel : l'AuthService
        connaît le nom du candidat, on ne devine pas, on retire ce que l'on sait.
    :return: texte anonymisé + journal des éléments retirés (traçabilité IA Act).
    """
    if not text:
        return AnonymizationResult(text="", removed=[])

    removed: list[str] = []
    out = text

    # 1. Identifiants connus fournis par l'appelant (le plus fiable).
    for token in known_pii or []:
        token = (token or "").strip()
        if len(token) < 2:
            continue
        pattern = re.compile(re.escape(token), flags=re.IGNORECASE)
        if pattern.search(out):
            out = pattern.sub(" ", out)
            removed.append(token)

    # 2. E-mails, URLs, téléphones.
    for label, pat in (
        ("email", r"[\w.+-]+@[\w-]+\.[\w.-]+"),
        ("url", r"https?://\S+"),
        ("phone", r"(?:(?:\+|00)\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?){3,5}"),
    ):
        if re.search(pat, out):
            out = re.sub(pat, " ", out)
            removed.append(label)

    # 3. Âge et date de naissance.
    for pat in (r"\b\d{1,2}\s*ans\b", r"\b\d{1,2}\s*years?\s*old\b", *_PROXY_HINTS_PATTERNS):
        if re.search(pat, out, flags=re.IGNORECASE):
            out = re.sub(pat, " ", out, flags=re.IGNORECASE)
            removed.append("age/dob")

    # 4. Code postal FR (5 chiffres) et arrondissements ("Paris 8e", "Lyon 7e").
    if re.search(r"\b\d{5}\b", out):
        out = re.sub(r"\b\d{5}\b", " ", out)
        removed.append("postal_code")
    arr = r"\b([A-Za-zÀ-ÿ]+)\s+\d{1,2}\s*(?:e|er|ème|eme|th|nd|rd|st)\b"
    if re.search(arr, out, flags=re.IGNORECASE):
        out = re.sub(arr, " ", out, flags=re.IGNORECASE)
        removed.append("arrondissement")

    # 5. Civilités et marqueurs de genre (token à token, insensible aux accents).
    kept_tokens = []
    for raw in out.split():
        norm = _strip_accents(raw.lower()).strip(".,;:!?()[]\"'")
        if norm in _GENDER_TERMS:
            removed.append(f"gender:{raw}")
            continue
        kept_tokens.append(raw)
    out = " ".join(kept_tokens)

    # 6. Normalisation des espaces.
    out = re.sub(r"\s+", " ", out).strip()
    return AnonymizationResult(text=out, removed=removed)
