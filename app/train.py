import csv
import pickle
import os
import sys

# Add root directory to path to allow importing app module
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from app.matcher import _normalize
from sklearn.feature_extraction.text import TfidfVectorizer

def train():
    french_csv_path = "/Users/nebel/DEV/PROJECTS/PFE/data/NewJobspy/european_jobs_cleaned.csv"
    fallback_csv_paths = []
    
    corpus = []
    
    if os.path.exists(french_csv_path):
        print(f"Chargement du dataset principal français : {french_csv_path}...")
        with open(french_csv_path, mode="r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                header = []
            
            if header:
                try:
                    desc_idx = header.index("description")
                    title_idx = header.index("title")
                    
                    count = 0
                    for row in reader:
                        if len(row) > max(desc_idx, title_idx):
                            desc = row[desc_idx].strip()
                            title = row[title_idx].strip()
                            text = f"{title} {desc}".strip()
                            if text:
                                corpus.append(text)
                                count += 1
                                # Charger 25 000 offres pour un apprentissage IDF optimal et rapide
                                if count >= 25000:
                                    break
                except ValueError:
                    print("Colonnes introuvables dans le dataset principal, passage au fallback.")
    
    # Si le dataset français n'était pas disponible ou vide, utiliser le fallback
    if not corpus:
        print("Dataset principal français non disponible, chargement des fallbacks...")
        for csv_path in fallback_csv_paths:
            if not os.path.exists(csv_path):
                print(f"Fichier introuvable : {csv_path}, ignoré.")
                continue
            print(f"Lecture de {csv_path}...")
            with open(csv_path, mode="r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    continue
                try:
                    desc_idx = header.index("description")
                    title_idx = header.index("title")
                except ValueError:
                    continue
                
                for row in reader:
                    if len(row) > max(desc_idx, title_idx):
                        desc = row[desc_idx].strip()
                        title = row[title_idx].strip()
                        text = f"{title} {desc}".strip()
                        if text:
                            corpus.append(text)
    
    print(f"Total documents dans le corpus d'entraînement : {len(corpus)}")
    if not corpus:
        print("Erreur : Corpus vide. Impossible d'entraîner.")
        return
        
    print("Ajustement (Fitting) du TfidfVectorizer (Capped à 30 000 features)...")
    # Limiter à 30 000 mots-clés les plus importants pour un fichier pickle léger (1.2 Mo) et un matching ultra-rapide.
    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    vec.fit(corpus)
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectorizer.pkl")
    print(f"Sauvegarde du vectoriseur entraîné dans {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(vec, f)
    print("Entraînement du modèle français/technique terminé avec succès !")

if __name__ == "__main__":
    train()
