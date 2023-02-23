import sys
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Récupérer le chemin du fichier d'entrée et de sortie depuis les arguments de la ligne de commande
input_file = sys.argv[1]
output_file = sys.argv[2]

# Charger le modèle T5 Conversational et le tokenizer correspondant
model = T5ForConditionalGeneration.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")
tokenizer = T5Tokenizer.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")

# Lire le dialogue à résumer depuis le fichier input_file
with open(input_file, "r") as f:
    dialogue = f.read()

# Prétraiter le dialogue en ajoutant le préfixe "summarize:" et en encodant avec le tokenizer
input_ids = tokenizer.encode("summarize: " + dialogue, return_tensors="pt")

# Générer le résumé en utilisant le modèle T5 Conversational
summary_ids = model.generate(input_ids=input_ids, num_beams=4, max_length=100, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Écrire le résumé dans le fichier output_file
with open(output_file, "w") as f:
    f.write(summary)

# Afficher le résumé
print(summary)
