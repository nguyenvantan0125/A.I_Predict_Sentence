from pyvi import ViTokenizer
import gtts
import playsound
import os
import speech_recognition as sr
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def readcsv(filename, header = None):
    a = pd.read_csv(filename, header = header,na_values="nan")
    a = a.dropna()
    return a

def vi_tokenizer(corpus):
    for idx, txt in enumerate(corpus):
        corpus[idx] = ViTokenizer.tokenize(txt)
    return corpus

def pipeline_svc():
    p_l = Pipeline([
    ("vect", CountVectorizer(token_pattern=u"(?u)\\b\\w+\\b")),
    ("clf", SVC(probability=True,)) #model
    ])
    return p_l
def pipeline_multinomialNB():
    p_l = Pipeline([
    ("vect", CountVectorizer(token_pattern=u"(?u)\\b\\w+\\b")),
    ("clf", MultinomialNB()) #model
    ])
    return p_l

class processing():
    def __init__(self):
        pass
    @staticmethod
    def speak(text):
        print("A.I predict: {}".format(text))
        tts = gtts.gTTS(text = text, lang='vi', slow=False)
        tts.save("sound.mp3")
        playsound.playsound("sound.mp3", False)
        os.remove("sound.mp3")
        
    def end(self):
        self.speak("cảm ơn, hẹn gặp lại")
    
    def reply(self,pipeline,user_speech):
        print("User: " + user_speech)
        data_test = [user_speech]
        data_test = vi_tokenizer(data_test) # tokenizer tách từ tiếng việt
        pred = pipeline.predict(data_test) # predict = pipile
        self.speak(*pred)
        
class interface(processing):
    def __init__(self,pipeline):
        self.AI_ear = sr.Recognizer()
        self.p_l = pipeline
    def excute(self):
        while True:   
            with sr.Microphone() as source:
                print("Listning...")
                audio_data = self.AI_ear.record(source, duration=3) # duartion số giây       
            try:
                user_speech = self.AI_ear.recognize_google(audio_data,language="vi-VN")
            except:
                user_speech = ""
                
            if  user_speech == "":
                super().speak("mời bạn nói lại")
            elif "kết" in user_speech:
                super().end()
                break
            else:
                super().reply(self.p_l, user_speech)
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
    

