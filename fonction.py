from transformers import pipeline

summarizer = pipeline("summarization", model="slauw87/bart-large-cnn-samsum")

conversation = '''Sugi: I am tired of everything in my life. 
Tommy: What? How happy you life is! I do envy you.
Sugi: You don't know that I have been over-protected by my mother these years. I am really about to leave the family and spread my wings.
Tommy: Maybe you are right.                                           
'''

summary = summarizer(conversation, max_length=50, min_length=10, do_sample=False)[0]['summary_text']

print(summary)
