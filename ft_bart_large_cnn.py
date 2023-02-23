from transformers import pipeline

# Chemin du fichier contenant le texte à résumer
input_file = "fichier.txt"

# Chemin du fichier où écrire le résumé
output_file = "resume.txt"

# Charger le modèle BART pour le résumé de texte
model = pipeline("summarization", model="facebook/bart-large-cnn")

# Lire le texte à résumer depuis le fichier input_file
with open(input_file, "r") as f:
    text = f.read()

# Générer le résumé en utilisant le modèle BART
summary = model(text, max_length=400, min_length=20, do_sample=False)[0]["summary_text"]

# Écrire le résumé dans le fichier output_file
with open(output_file, "w") as f:
    f.write(summary)
