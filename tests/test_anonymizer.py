"""Tests du module d'anonymisation amont (anti-biais)."""

from app.anonymizer import anonymise


def test_removes_known_name():
    raw = "Jean Dupont, developpeur Java avec 5 ans d'experience Spring"
    result = anonymise(raw, known_pii=["Jean", "Dupont"])
    assert "Jean" not in result.text
    assert "Dupont" not in result.text
    # Les compétences pertinentes sont conservées.
    assert "java" in result.text.lower()
    assert "spring" in result.text.lower()


def test_removes_age_and_location():
    raw = "Marie, 25 ans, Paris 8e, 75008, ingenieure logiciel Python"
    result = anonymise(raw, known_pii=["Marie"])
    assert "25 ans" not in result.text
    assert "75008" not in result.text
    assert "Paris 8e" not in result.text
    assert "python" in result.text.lower()


def test_removes_email_and_phone():
    raw = "Contact : alice@example.com / +33 6 12 34 56 78 — data scientist"
    result = anonymise(raw)
    assert "@" not in result.text
    assert "data" in result.text.lower()
    assert "scientist" in result.text.lower()


def test_removes_gender_markers():
    raw = "Monsieur Paul, il est developpeur backend"
    result = anonymise(raw, known_pii=["Paul"])
    low = result.text.lower()
    assert "monsieur" not in low
    assert " il " not in f" {low} "
    assert "developpeur" in low


def test_tracks_removed_items_for_traceability():
    raw = "Jean Dupont, 30 ans, jean@mail.com"
    result = anonymise(raw, known_pii=["Jean", "Dupont"])
    assert result.removed_count >= 2  # nom + age + email


def test_empty_input_is_safe():
    result = anonymise("")
    assert result.text == ""
    assert result.removed_count == 0
