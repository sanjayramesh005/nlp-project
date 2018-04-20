#For extracting common nouns using POS tagging
import nltk
def give_noun(text): #Gives nouns
    sent  =  nltk.sent_tokenize(text)
    for s in sent:
        d = nltk.word_tokenize(s)   
        pos_tags= (nltk.pos_tag(d))
    Nouns=[]
    for i in range(len(pos_tags)):
        if pos_tags[i][1]=='NNP':
            Nouns.append(pos_tags[i][0])
    return Nouns
        
def C_nouns(Nouns1, Nouns2): #Counts number of common nouns
    c = 0
    for noun in Nouns1:
        if noun in Nouns2:
            c += 1
            Nouns2.remove(noun)
    return c

def give_common_nouns(text1,text2):
    Nouns1 =give_noun(text1)
    Nouns2 =give_noun(text2)
    c=C_nouns(Nouns1, Nouns2)
    return c
