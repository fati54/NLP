import sys
from transformers import pipeline
import math
import re

# Récupérer le chemin du fichier d'entrée et de sortie depuis les arguments de la ligne de commande
input_file = sys.argv[1]
output_file = sys.argv[2]

# Charger le modèle T5 pour le résumé de texte
model = pipeline("summarization", model="t5-base")

# Lire le texte à résumer depuis le fichier input_file
with open(input_file, "r") as f:
    text = f.read()

# Calculer le nombre de mots dans le texte
text = re.sub(r'[^\w\s]','',text) # remove punctuation and special characters
num_words = len(text.split())

# Calculer le nombre de mots minimum et maximum pour le résumé
min_length = math.ceil(num_words * 0.2)
max_length = math.ceil(num_words * 0.3)

# Générer le résumé en utilisant le modèle T5
summary = model(text,max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]

# Écrire le résumé dans le fichier output_file
with open(output_file, "w") as f:
    f.write(summary)

# Print 
print(num_words)
print(min_length)
print(max_length)
