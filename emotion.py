import speech_recognition as sr
import prediction


recogniser = sr.Recognizer()

audio = 'english.wav'

with sr.AudioFile(audio) as source:
    recogniser.adjust_for_ambient_noise(source,duration=1)
    recorded_audio = recogniser.record(source)
    text = recogniser.recognize_google(recorded_audio,language='en-US')


from transformers import AutoTokenizer,AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
   
text = prediction.preprocess(text)
sentiment = prediction.sentiment_analysis(text)
negatives = []
splitted = text.split(' ')
for i in text.split(' '):
    if prediction.sentiment_analysis(i) == 'negative':
        negatives.append(i)


print(f"In the audio provided, the sentiment is {sentiment}")





