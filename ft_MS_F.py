from transformers import pipeline
import re 

summarizer = pipeline("summarization", model="knkarthick/MEETING_SUMMARY")
text = '''
Gael: est-ce que LNRS va proposer du filtrage en mode SaaS? Si LNRS propose une formule SaaS, LNRS sera soumis au ongoing performance monitoring
Ronan: tout dépend si c'est LNRS qui gère complètement la configuration ou si LNRS ne fait qu'héberger l'instance. Si LNRS ne fait qu'héberger, mais que toute la configuration reste chez le client alors la responsabilité reste chez le client
Ronan: la conformité doit être owner des modèles. Ne souhaite pas que tout soit contrôlé par le vendeur
Daniela / Céline / David: Décide du paramétrage. C'est bien comme ça car en fonction de l'approche de risque de chacun, des activités, cela pourrait être risqué de ne pas pouvoir définir le paramétrage. Il faut qu'il y ait une partie qui reste à la main du client pour adapter au contexte de chacun
Ronan: notre problème aujourd'hui est qu'on a qu'une évaluation empirique du modèle et aurait besoin d'aide au niveau documentation et éléments de tests pour qu'on puisse valider
Celine: on refait les tests sur les algo car on n'a pas les mêmes comportements, parfois censé être applicable que dans un cadre limité mais en faisant les tests parfois s'applique plus largement. Aussi parfois la documentation est corrigée mais ne sont pas notifiés (il y a eu une erreur de paramétrage lié à ça )
Gael: dans le modèle on doit chaque année préciser les defects identifiés, l'évolution… Il faut que LNRS soit en capacité de partager les defects constatés pour que le client puisse faire son propre assessment du risque et ce que l'on met en place
Ronan: on est censés connaitre les 45 algo et les interdépendances des algos et sur des données représentatives de production
Preethi: utilise Swift pour model validation mais permet de tester que les messages Swift MT. Question comment gérer les MX qui seront mappés en FUF pour filtrage
Gael: le threshold apparait dans la documentation, du coup cela génère des questions alors que ce n'est pas modifiable. Il faut faire un assessment sur tous les paramètres
Ronan: comme pourquoi l'algo 35 est grisé?
Tuan: il faut modifier en profondeur la documentation. Le 72% est supprimé de la doc mais apparaît encore dans un écran de log et donc cela génère encore des questions
'''

# Calculer le nombre de mots dans le texte
text = re.sub(r'[^\w\s]','',text) # remove punctuation and special characters
num_words = len(text.split())
print(num_words)

summary = summarizer(text, max_length=300, min_length=50, do_sample=False)
print(summary)


output_file  = "resume_ai_F.txt"

with open(output_file , "w") as f:
    summary_text = '\n'.join([s['summary_text'] for s in summary])
    f.write(summary_text)


