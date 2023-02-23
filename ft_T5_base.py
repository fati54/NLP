from transformers import pipeline
import math

'''
Le script lit le texte à résumer depuis un fichier, 
calcule le nombre de mots dans le texte, puis calcule les longueurs minimum et maximum du résumé en fonction de ce nombre de mots. 
Enfin, il utilise le modèle T5 pour générer le résumé en respectant les contraintes de longueur et écrit le résumé dans un autre fichier.

'''

# Chemin du fichier contenant le texte à résumer
input_file = "input.txt"

# Chemin du fichier où écrire le résumé
output_file = "outputT5.txt"

# Charger le modèle T5 pour le résumé de texte
model = pipeline("summarization", model="t5-base")

# Lire le texte à résumer depuis le fichier input_file
with open(input_file, "r") as f:
    text = f.read()

# Calculer le nombre de mots dans le texte
num_words = len(text.split())

# Calculer le nombre de mots minimum et maximum pour le résumé
min_length = math.ceil(num_words * 0.2)
max_length = math.ceil(num_words * 0.3)

# Générer le résumé en utilisant le modèle T5 avec les contraintes de longueur
summary = model(text, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]

# Écrire le résumé dans le fichier output_file
with open(output_file, "w") as f:
    f.write(summary)
