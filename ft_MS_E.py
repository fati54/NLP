from transformers import pipeline
import re 

summarizer = pipeline("summarization", model="knkarthick/MEETING_SUMMARY")
text = '''
Gael: Will LNRS offer filtering in SaaS mode? If LNRS offers an SaaS formula, LNRS will be subject to ongoing performance monitoring.
Ronan: It all depends on whether LNRS manages the configuration completely or if LNRS only hosts the instance. If LNRS only hosts, but all configuration remains with the client, then responsibility remains with the client.
Ronan: Compliance should be the owner of the models. I don't want everything to be controlled by the vendor.
Daniela/Céline/David: Decide on the configuration. This is good because depending on each person's risk approach, activities, it could be risky not to be able to define the configuration. There needs to be a part that remains in the hands of the client to adapt to everyone's context.
Ronan: Our problem today is that we have only an empirical evaluation of the model and need help with documentation and test elements so that we can validate.
Celine: We are redoing tests on the algorithms because we don't have the same behavior, sometimes supposed to be applicable only in a limited framework, but by doing the tests, sometimes it applies more broadly. Also, sometimes the documentation is corrected but not notified (there was a configuration error related to this).
Gael: In the model, we must specify every year the identified defects, evolution... LNRS must be able to share the defects identified so that the client can make their own risk assessment and what we implement.
Ronan: We are supposed to know the 45 algorithms and the interdependencies of the algorithms and on representative production data.
Preethi: Uses Swift for model validation but allows testing only for Swift MT messages. Question is how to handle MX that will be mapped to FUF for filtering.
Gael: The threshold appears in the documentation, so this generates questions even though it is not modifiable. An assessment needs to be made on all parameters.
Ronan: Like, why is algorithm 35 grayed out?
Tuan: The documentation needs to be thoroughly modified. The 72% is removed from the doc but still appears in a log screen and so this still generates questions.
'''

# Calculer le nombre de mots dans le texte
text = re.sub(r'[^\w\s]','',text) # remove punctuation and special characters
num_words = len(text.split())

summary = summarizer(text, max_length=300, min_length=100, do_sample=False)
print(summary)
# Print 
print(num_words)

output_file  = "resume_ai_E.txt"

with open(output_file , "w") as f:
    summary_text = '\n'.join([s['summary_text'] for s in summary])
    f.write(summary_text)

print(num_words)
print(summary_text)


