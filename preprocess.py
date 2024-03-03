import pandas as pd
import numpy as np
import re
from spellchecker import SpellChecker
from tqdm import tqdm
tqdm.pandas()

class PreprocessDataFrame():
    def __init__(self,file_path):
        self.file_path = file_path

    def removeEmojis(self, 
                     text:str):
        emoj = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002500-\U00002BEF"  # chinese char
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642" 
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  # dingbats
            u"\u3030"
                            "]+", re.UNICODE)
        return re.sub(emoj, '', text)
    
    def spellWordChecking(self, 
                          text:str):
        spell = SpellChecker()
        misspelled = text.split(" ")
        forbidden = ["app", "spotify"]
        fin_sentence = []
        for word in misspelled:
          if (spell.correction(word)!= None) and (word not in forbidden):
              fin_sentence.append(spell.correction(word))
          else:
              fin_sentence.append(word)
        return " ".join(fin_sentence)
    
    def cleaningText(self, 
                     text:str,
                     is_spell:bool):
        text = self.removeEmojis(text)
        text = re.sub(r'@[A-Za-z0-9]+', '', text) # remove mentions
        text = re.sub(r'#[A-Za-z0-9]+', '', text) # remove hashtag
        text = re.sub(r'RT[\s]', '', text) # remove RT
        text = re.sub(r'http\S+', '', text) # remove link
        text = re.sub(r"[^\w\d'.\s]+",'',text) #remove all punctuations except ' and .
        text = re.compile(r"(.)\1{2,}").sub(r'\1\1',text) #remove duplicate alphabet/digit
        text = text.replace('\n', ' ') # replace new line into space
        if is_spell:
            text = self.spellWordChecking(text)
        return text 
    
    def cleaningDataFrame(self,
                          is_spell=False,
                          first_limit=None,
                          end_limit=None,
                          len_review=20):
        """
        len_review: define minimal boundary for length of review to be preprocessed
        """
        try:
            df = pd.read_csv(self.file_path)
            df = df.fillna(value="")
            df = df.drop_duplicates(subset="review_text", keep="last")
            df = df[df["review_text"].apply(lambda x: len(x) > len_review)]
            df = df.dropna()
            if first_limit!=None and end_limit!=None:
                df = df[first_limit:end_limit]
            df["clean_review_text"] = df["review_text"].progress_apply(lambda x: self.cleaningText(x.encode().decode('utf-8').lower(),is_spell))
            df = df.reset_index(drop=True)
            return df
        except Exception as error:
            print('Caught this error: ' + repr(error))