from rouge import Rouge

def calculate_rouge_scores(hypothesis, reference):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    return scores[0]['rouge-1']['f'], scores[0]['rouge-2']['f'], scores[0]['rouge-l']['f']

# Charger le texte original, le résumé AI et le résumé humain
with open('texte_original_E.txt', 'r') as f:
    texte_original = f.read()

with open('resume_ai_E.txt', 'r') as f:
    resume_ai = f.read()

with open('resume_humain_E.txt', 'r') as f:
    resume_humain = f.read()

# Calculer les scores ROUGE pour les deux résumés
rouge_1_ai, rouge_2_ai, rouge_l_ai = calculate_rouge_scores(resume_ai, texte_original)
rouge_1_humain, rouge_2_humain, rouge_l_humain = calculate_rouge_scores(resume_humain, texte_original)

# Afficher les scores ROUGE
print("ROUGE-1 AI: ", rouge_1_ai)
print("ROUGE-2 AI: ", rouge_2_ai)
print("ROUGE-L AI: ", rouge_l_ai)

print("ROUGE-1 Humain: ", rouge_1_humain)
print("ROUGE-2 Humain: ", rouge_2_humain)
print("ROUGE-L Humain: ", rouge_l_humain)

# Trouver le gagnant pour chaque score
rouge_1_winner = 'AI' if rouge_1_ai > rouge_1_humain else 'Humain'
rouge_2_winner = 'AI' if rouge_2_ai > rouge_2_humain else 'Humain'
rouge_l_winner = 'AI' if rouge_l_ai > rouge_l_humain else 'Humain'

# Afficher les gagnants pour chaque score
print("ROUGE-1 Winner: ", rouge_1_winner)
print("ROUGE-2 Winner: ", rouge_2_winner)
print("ROUGE-L Winner: ", rouge_l_winner)
