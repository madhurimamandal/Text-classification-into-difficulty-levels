#Packages
import nltk 
import pandas as pd
import glob
import re
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from collections import Counter
from nltk import sent_tokenize
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
    
class features:
    Dale_Chall_List = pd.read_csv("Dale Chall List.txt")
    columns = ['File Name','Total Number Of Sentences', 'Average Sentence Length', 'Average Word Length', \
               'Number of Uncommon Words', 'Number of Unique Words', 'Words with 1 to 3 syllables', \
               "Words with 4 syllables", "Words with 5 syllables", "Words with 6 syllables", \
               "Words with more than 7 syllables", "Average number of syllables"]
    ptcol = ['CC', 'CD', 'NNS', 'VBP', 'NN', 'RB', 'MD', 'VB', 'VBZ', 'VBD', 'VBG', 'IN', 'JJ',  'FW', 'WDT', \
     'RBR', 'PRP$', 'VBN', 'PRP', 'DT', 'JJS', 'RP', 'JJR', 'WRB',  'WP', 'NNP', 'WP$',\
     'PDT', 'RBS', "''", 'NNPS', 'SYM', 'EX','TO','UH']
    ptcol.sort()
    columns.extend(ptcol)

#Preprocessing
    def preprocessing(self,text1):    
        text1 = re.sub('[^a-zA-Z]', ' ', text1)
        return [word for word in text1.lower().split() if not word in set(stopwords.words('english'))]
 
#Feature extraction
    def avg_sentence_length(self, text, num_sents):    
        avg = float(len(text)/num_sents)
        return avg
 
    def avg_word_length(self, text):
        s=0
        for w in text:
            s+=len(w)
            a=s/len(text)
        return a
 
    def syllable_count_single_word(self, word):
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count += 1
        return count
 
    def avg_syllables(self, text):
        s=0
        for w in text:
            s+=self.syllable_count_single_word(w)
        a=s/len(text)
        return a
 
    def pos_count_in_list(self, list1):
        pt = pos_tag(list1)
        dictpt = dict(pt)
        dictpt = Counter(dictpt.values())
        vl = []
        for i in self.ptcol:
            if i in dictpt.keys():
                vl.append(dictpt[i])
            else:
                vl.append(0)
        return vl
 
    def dif_words(self, text):
        frequency = Counter(text)
        return len(frequency)
 
    def freq_syl(self, text):
        count = [0,0,0,0,0]
        uniq_words = Counter(text).keys()
        for word in uniq_words:
            x = self.syllable_count_single_word(word)
            if(x > 1 and x <=3):
                count[0]+=1
            elif(x == 4):
                count[1]+=1
            elif(x == 5):
                count[2]+=1
            elif(x == 6):
                count[3]+=1
            else:
                count[4]+=1
        return count
 
    def not_in_dale_chall(self,text):
        n = [w for w in text if w not in self.Dale_Chall_List]
        n1 = len(n)
        return n1

#CREATING THE DATAFRAMES
    def fextr(self, filename):
        c2 = [filename]
        file = open(filename, 'r', encoding = "utf8")
        text2 = file.read()
        txt = self.preprocessing(text2)
        avg1 = len(sent_tokenize(text2))
        c2.append(avg1) 
        c2.append(self.avg_sentence_length(txt, avg1))
        c2.extend([self.avg_word_length(txt),self.not_in_dale_chall(txt),self.dif_words(txt)])
        c2.extend(self.freq_syl(txt))
        c2.append(self.avg_syllables(txt))
        vallist = self.pos_count_in_list(txt)
        c2.extend(vallist)
        return c2
    
    def create_dataframe(self):
        data = []
        for filename in glob.glob("*.txt"):
            c2 = self.fextr(filename)
            data.append(c2)  
        df = pd.DataFrame(data,columns = self.columns)
        return(df)