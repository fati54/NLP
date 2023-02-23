import sys
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Récupérer le chemin du fichier d'entrée et de sortie depuis les arguments de la ligne de commande
input_file = sys.argv[1]
output_file = sys.argv[2]

# Charger le tokenizer et le modèle T5 pour le résumé de texte
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Lire le texte à résumer depuis le fichier input_file
with open(input_file, "r") as f:
    text = f.read()

# Split the input text into smaller segments of 512 tokens or less
segment_size = 512
segments = [text[i:i+segment_size] for i in range(0, len(text), segment_size)]

# Generate the summary for each segment and concatenate the results
summary = ""
for segment in segments:
    input_ids = tokenizer.encode(segment, max_length=segment_size, truncation=True, padding='longest', return_tensors='pt')
    output = model.generate(input_ids=input_ids, max_length=150, num_beams=4, early_stopping=True)
    summary += tokenizer.decode(output[0], skip_special_tokens=True)

# Écrire le résumé dans le fichier output_file
with open(output_file, "w") as f:
    f.write(summary)
