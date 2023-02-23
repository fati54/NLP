from transformers import pipeline

# Créer un pipeline pour la génération de texte
summarizer = pipeline("summarization", model="t5-base", max_length=1024, truncation=True)

# Texte du dialogue à résumer
dialogue = "Person A: Bonjour comment allez-vous? Person B: Je vais bien merci. Person A: Comment s'est passée votre journée? Person B: C'était une journée plutôt chargée, mais j'ai réussi à tout faire. Person A: C'est bien, vous devez être fatigué. Person B: Oui, un peu, mais ça va aller."

# Diviser le texte du dialogue en plusieurs phrases
sentences = dialogue.split(".")

# Résumer chaque phrase avec troncature de longueur maximale de 1024 caractères
summaries = []
for sentence in sentences:
    if len(sentence.strip()) > 0:
        summary = summarizer(sentence.strip())[0]["summary_text"]
        summaries.append(summary)

# Combinez les résumés en un seul texte
summary_text = ". ".join(summaries)

print(summary_text)
