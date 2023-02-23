from transformers import pipeline
summarizer = pipeline("summarization", model="knkarthick/bart-large-xsum-samsum")
conversation = '''Hannah: Hey, do you have Betty's number?
Amanda: Lemme check
Amanda: Sorry, can't find it.
Amanda: Ask Larry
Amanda: He called her last time we were at the park together
Hannah: I don't know him well
Amanda: Don't be shy, he's very nice
Hannah: If you say so..
Hannah: I'd rather you texted him
Amanda: Just text him 🙂
Hannah: Urgh.. Alright
Hannah: Bye
Amanda: Bye bye                                       
'''

output_file = "resume_samsum.txt"
result = summarizer(conversation)
with open(output_file, "w") as f:
    f.write(result)