from transformers import pipeline

# charger le fichier texte d'entrée
input_file = "transcript.txt"
with open(input_file, "r") as f:
    input_text = f.read()

# créer une liste vide pour stocker les résumés générés
summaries = []

# boucle à travers les modèles et générer les résumés
models = {
    "google/pegasus-large": "Pegasus Large",
    "t5-base": "T5 Base",
    "facebook/bart-large-cnn": "BART Large CNN",
}
for model_name, model_desc in models.items():
    summarizer = pipeline("summarization", model=model_name)
    summary = summarizer(input_text, max_length=300, min_length=100, do_sample=False)
    summary_text = summary[0]["summary_text"]
    summaries.append(f"{model_desc}: {summary_text}")

# écrire la liste de résumés dans un fichier texte de sortie
output_file = "mon_fichier_resumes.txt"
with open(output_file, "w") as f:
    for summary in summaries:
        f.write(summary + "\n")
