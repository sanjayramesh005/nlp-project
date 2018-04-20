from os.path import join, dirname, abspath
from nltk.tokenize import word_tokenize
import pickle
from bs4 import BeautifulSoup as bs
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import json
import re
import clue_words
from nltk.parse.stanford import StanfordDependencyParser
from nltk.corpus import sentiwordnet as swn
from nltk.wsd import lesk
from pywsd.lesk import simple_lesk
import os

def get_all_reviews():
    data_path = join(dirname(dirname(abspath(__file__))), 'json')
    print data_path

    all_reviews = []
    all_ratings = []
    vocab = dict()
    stop_words = set(stopwords.words('english'))
    count = 0
    for filename in os.listdir(data_path):
        file_path = join(data_path, filename)
        fp = open(file_path, 'r')
        data = json.load(fp)['Reviews']
        for review in data:
            all_reviews.append(review['Content'])
            all_ratings.append(review['Ratings'])
            try:
                tokens = word_tokenize(review['Content'])

                # remove stop words
                tokens = [t for t in tokens if t not in stop_words]

                # Lemmatization
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(t, 'n') for t in tokens]
                tokens = [lemmatizer.lemmatize(t, 'v') for t in tokens]
                
                #  vocab.update(tokens)
                for t in tokens:
                    vocab[t] = vocab.get(t, 0) + 1

            except Exception as e:
                print e
                continue
        if count>=1:
            break
        count+=1

    #  print all_reviews
    vocab1 = [(v, i) for i, v in vocab.items()]

    vocab1.sort(reverse=True)
    return all_reviews, all_ratings, vocab1


all_reviews, all_ratings, vocab = get_all_reviews()

aspects = {
	'Service': 0,
	'Cleanliness': 1,
	'Value': 2,
	'Rooms': 3,
	'Location': 4
}


output = []
for ratings in all_ratings:
	b = [0]*5
	for aspect in aspects:
		try:
			r = int(ratings[aspect])
			b[aspects[aspect]] = r - 3
			if r == 3:
				b[aspects[aspect]] = 3
		except:
			b[aspects[aspect]] = 0
	output.append(b)

print output
# print vocab1[:100]

#  for i in range(100):
#      print vocab1[i]
#  print ''
#
#  for i in range(100):
#      print vocab1[i+100]

def sentence_break(reviews):
    sentences = []
    for text in reviews:
        p1 = re.compile("[\w)'][.?!]+(\s[A-Z0-9])*")
        p2 = re.compile("\d*\.\d+")
        while p2.search(text):
            a,b = p2.search(text).span()
            # print(text[a:b])
            text = text.replace(text[a:b],"0")
            # print(text[a:b])
        st = 1
        temp = 0
        end = 0
        for it in p1.finditer(text):
            end,temp = it.span() 
            if temp==len(text):
                sentences.append(text[st-1:end+1])
                break
            sentences.append(text[st-1:end+1])
            st = temp
    return sentences

def convert_pos_tags(tag):
    if tag.startswith('J'):
        return 'a'
    if tag.startswith('V'):
        return 'v'
    if tag.startswith('N') or tag.startswith('P'):
        return 'n'
    if tag.startswith('R'):
        return 'r'
    return 'x'

valid_relations = {
    'dobj': {('VBD', 'RBR'): 0},
    'nsubj': {('JJR', 'NN'): 2},
    'advmod': {('VBN', 'RB'): 0},
    'amod': {('NN', 'JJ'):0, ('NNP', 'JJ'):0, ('NN', 'VBP'):0},
    'nsubjpass': {('VBN', 'NNP'):2},
    'nsubj': {('JJ', 'NN'):2},
    'nmod': {('VBP', 'NN'):2, ('VBN', 'NN'):2},
}

def is_valid(rel, arg1, arg2):
    args = (arg1, arg2)
    if rel in valid_relations:
        if args in valid_relations[rel]:
            return valid_relations[rel][args]
        d = valid_relations[rel].keys()
        #  print d
        for dep in d:
            if args[0][0] == dep[0][0] and args[1][0] == dep[1][0]:
                return valid_relations[rel][dep]
    return -1

all_clue_words = []
for i, v in clue_words.clue_words.items():
    all_clue_words+=v
print all_clue_words
print "Getting reviews...."


print "all_reviews, all_ratings obtained."
sentiments = []


# Install Stanford Dependency Parser and add the path to jar files below
path_to_jar = '/home/sanjayr/Downloads/stanford-parser-full-2018-02-27/stanford-parser.jar'
path_to_models_jar = '/home/sanjayr/Downloads/stanford-parser-full-2018-02-27/stanford-parser-3.9.1-models.jar'
dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

lemmatizer = WordNetLemmatizer()


i = 0
num_reviews = 100
for review in all_reviews:
    #  if i<=6:
    #      i+=1
    #      continue
    sentences = sentence_break([review])
    print review
    #  print sentences
    sentiment = [0]*5
    
############################################################    
    try:
        parses = dependency_parser.raw_parse_sents(sentences)
    except Exception as e:
        print e
        sentiments.append(sentiment)
        continue
    idx = 0
    for parse in parses:
        dep = parse.next()
        dep = list(dep.triples())
        if not sentences[idx].strip():
            continue
		#  neg = False
        for word_pair in dep:
            t = is_valid(word_pair[1], word_pair[0][1], word_pair[2][1]) 
            if t==-1:
                #  print "INVALID TAG",word_pair
                continue
            sent = sentences[idx].split()
            #  print sent
            aspect_word = word_pair[t][0].lower()
            try:
                aspect_word = lemmatizer.lemmatize(aspect_word, convert_pos_tags(word_pair[t][1]))
            except Exception as e:
                print e
            senti_word = list(word_pair[2-t])
            senti_word[0] = senti_word[0].lower()
            #  print aspect_word, senti_word
            if len(senti_word[0])<=2:
                #  print "SENTI WORD", aspect_word, senti_word, word_pair
                continue
            tag_type = convert_pos_tags(senti_word[1])
            # word_sense = lesk(sent, senti_word[0], tag_type)
            #  print sentences[idx], senti_word[0], tag_type, word_pair
            try:
                #  word_sense = lesk(sent, senti_word[0], tag_type)
                word_sense = simple_lesk(sentences[idx], senti_word[0], tag_type)
                word_sense = str(word_sense)[8:-2]
                if not word_sense and tag_type is 'a':
                    #  word_sense = lesk(sent, senti_word[0], 's')
                    word_sense = simple_lesk(sentences[idx], senti_word[0], 's')
                    word_sense = str(word_sense)[8:-2]
            except:
                continue
            #  print tag_type, word_sense
            try:
                senti = swn.senti_synset(word_sense)
                score = senti.pos_score() - senti.neg_score()
            except Exception as e:
                continue
            #  print senti, score
            #  print sent, word_pair, aspect_word, senti_word, tag_type, word_sense, senti, score
            if aspect_word in clue_words.clue_words['cleanliness']:
                #  print aspect_word, senti_word, score, word_pair, word_sense
                sentiment[1]+=score
            if aspect_word in clue_words.clue_words['service']:
                #  print aspect_word, senti_word, score, word_pair, word_sense
                sentiment[0]+=score
            if aspect_word in clue_words.clue_words['rooms']:
                #  print aspect_word, senti_word, score, word_pair, word_sense
                sentiment[3]+=score
            if aspect_word in clue_words.clue_words['location']:
                #  print aspect_word, senti_word, score, word_pair, word_sense
                sentiment[4]+=score
            if aspect_word in clue_words.clue_words['value']:
                #  print aspect_word, senti_word, score, word_pair, word_sense
                sentiment[2]+=score
            if aspect_word not in all_clue_words:
                #  print 'CANNOT', aspect_word, senti_word, score, word_pair, word_sense
                pass
                

        idx+=1
	
    print '-------',sentiment, all_ratings[i]	
    sentiments.append(sentiment)

#############################################################

#############################################################################
    #  for sentence in sentences:
    #      result = dependency_parser.raw_parse(sentence)
    #      dep = result.next()
    #      dep = list(dep.triples())
    #      #  print 'dep', dep
    #      for word_pair in dep:
    #          aspect_word = word_pair[0][0]
    #          #  print aspect_word
    #          senti_word = word_pair[2][0]
    #          tag_type = convert_pos_tags(word_pair[2][1])
    #          if tag_type=='x':
    #              continue
    #          try:
    #              senti = swn.senti_synset(word_pair[2][0]+'.'+tag_type+'.01')
    #              score = senti.pos_score() - senti.neg_score()
    #          except Exception as e:
    #              print "CANNOT ", word_pair, e
    #              continue
    #          if aspect_word in clue_words.clue_words['cleanliness']:
    #              print aspect_word, senti_word, score, word_pair
    #              sentiment[1]+=score
    #          if aspect_word in clue_words.clue_words['service']:
    #              print aspect_word, senti_word, score, word_pair
    #              sentiment[0]+=score
    #          if aspect_word in clue_words.clue_words['rooms']:
    #              print aspect_word, senti_word, score, word_pair
    #              sentiment[3]+=score
    #          if aspect_word in clue_words.clue_words['location']:
    #              print aspect_word, senti_word, score, word_pair
    #              sentiment[4]+=score
    #          if aspect_word in clue_words.clue_words['value']:
    #              print aspect_word, senti_word, score, word_pair
    #              sentiment[2]+=score
    #
    #  print '-------',sentiment, all_ratings[i]
###############################################################################

    #  if i>=num_reviews:
    #      break
    i+=1

print sentiments
print output
def check_accuracy(actual_output, expected_output):
    accuracy = [0]*5
    count = [0]*5
    for i, a in enumerate(actual_output):
        e = expected_output[i]
        for j, x in enumerate(a):
            y = e[j]
            if y==3:
                accuracy[j]+=1
                # count[j]-=1
            elif x*y>0:
                accuracy[j]+=1
            elif x*y==0:
                count[j]-=1
            count[j]+=1

    accuracy = [float(v)/count[i] for i, v in enumerate(accuracy)]
    return accuracy

print check_accuracy(sentiments, output)
