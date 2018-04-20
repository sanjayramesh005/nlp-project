import pandas as pd
import numpy as np
from tqdm import tqdm 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from random import shuffle
import nltk
from difflib import SequenceMatcher
from give_common_nouns import give_common_nouns
words = set(nltk.corpus.words.words())
df=pd.read_csv('train.csv', sep=',', delimiter=None)
df1 =df.as_matrix()
label=df1[:,5].astype('int')
sentences1=df1[:,3]
sentences2=df1[:,4]
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
character = [".",",","?","!",":","'",";",":","#"]
# character = np.load('character.npy')
def sentence_clean(sentence): #Sentence cleaning
    filtered_sentence=[]
    word_tokens = str(sentence).lower().split()
    word_tokens[-1]=word_tokens[-1].replace('?',' ')
    for w in word_tokens:
        for p in range(0,len(character)):
            w = w.replace(character[p],'') 
        w = ' '.join([i for i in w if not i.isdigit()])
        if w not in stop_words:
#            w=ps.stem(w)
            filtered_sentence.append(w)
    return filtered_sentence

def Vocabulary(sentences): #Vocanulary creation
    Vocabulary=[]
    for sentence in (sentences):
        sen=sentence_clean(sentence)        
        for word in sen:
            if len(word)>=3:
                if word not in Vocabulary:
                    Vocabulary.append(word)
    return Vocabulary


def vec(Vocab,sentence): #Vectorization using Bag of words model
#    train_samples=np.zeros([len(all_sentences),len(Vocab)])
    vector=np.zeros(len(Vocab))
    word_temp=sentence_clean(sentence)
    for q in range(0,len(word_temp)):
        for w in range(0,len(Vocab)):
           if (word_temp[q]==Vocab[w]):
              vector[w]=1 
    return [vector]
dist=np.zeros(len(sentences1))
angle=np.zeros(len(sentences1))
c=np.zeros(len(sentences1))
match_ratio=np.zeros(len(sentences1))

#Distance and angle calculation
for j in tqdm(range(0,len(sentences1)/10)):
    both_sentences=np.hstack((sentences1[j],sentences2[j]))
    Vocab=Vocabulary(both_sentences)
    vec1=vec(Vocab,sentences1[j])
    vec2=vec(Vocab,sentences2[j])
    dist[j] = np.linalg.norm(np.array(vec1)-np.array(vec2))
    if ((np.linalg.norm(vec1)*np.linalg.norm(vec2))==0):
        angle[j] = 0
    else:
        angle[j] = np.dot(vec1,np.transpose(vec2))/float((np.linalg.norm(vec1)*np.linalg.norm(vec2)))
    try:
        match_ratio[j]=SequenceMatcher(None, sentences1[j],sentences2[j]).ratio()
    except:
        match_ratio[j]=0
    try:    
        c[j]=give_common_nouns(sentences1[j],sentences2[j])
    except:
        c[j]=0
    
from sklearn import preprocessing    
x_train=np.transpose(np.vstack((dist[0:30000],angle[0:30000],c[0:30000],match_ratio[0:30000])))
normalizer = preprocessing.Normalizer().fit(x_train)
x_train = normalizer.transform(x_train)
x_valid=np.transpose(np.vstack((dist[30000:],angle[30000:],c[30000:],match_ratio[30000:])))
x_valid = normalizer.transform(x_valid)
from sklearn import metrics
# x_train=np.load('x_train.npy')
# x_train=x_train[:,3:4]
#x_train=np.delete(x_train,2,1)

# x_valid=np.load('x_valid.npy')
# x_valid=x_valid[:,3:4]
#x_valid=np.delete(x_valid,2,1)
y_train=label[0:30000]
y_valid=label[30000:]

#Using MLP classifier
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier()
clf.fit(x_train, y_train)
y_pred=clf.predict(x_valid)
acc_NN=(len(y_valid)-np.count_nonzero(y_valid-y_pred))/float(len(y_valid))
#proba_MLP=clf.predict_proba (x_valid)
#log_loss_MLP=metrics.log_loss(y_valid,proba_MLP[:,1])

#Using decison tree
from sklearn import tree
clf1 = tree.DecisionTreeClassifier()
clf1.fit(x_train, y_train)
y_pred=clf1.predict(x_valid)
acc_DT=(len(y_valid)-np.count_nonzero(y_valid-y_pred))/float(len(y_valid))
#proba_DT=clf1.predict_proba(x_valid) 
#log_loss_DT=metrics.log_loss(y_valid,proba_DT[:,1])
#print('LogLoss:','DT',log_loss_DT,',','MLP',log_loss_MLP)
print('Accuracy:','DT:',acc_DT,'MLP',acc_NN)

#Using SVM
#from sklearn.svm import LinearSVC
#from sklearn import preprocessing
#X_train_scaled = preprocessing.scale(x_train)
#X_valid_scaled = preprocessing.scale(x_valid)
#clf2= LinearSVC()
#clf2.fit(X_train_scaled , y_train)
#y_pred=clf2.predict(X_valid_scaled )
#acc_SVM=(len(y_valid)-np.count_nonzero(y_valid-y_pred))/len(y_valid) 
#proba_SVM=clf2.predict_proba(X_valid_scaled ) 
#log_loss_SVM=metrics.log_loss(y_valid,proba_SVM[:,1])
#print(acc_SVM)
#print(log_loss_SVM)
#Feature ranking
from feature_ranking import feature_ranking
imps=feature_ranking(x_train,y_train)
