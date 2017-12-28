from collections import defaultdict
import json
import langdetect
from multiprocessing import Pool, cpu_count
import numpy as np
import pickle
import re


def clean(sentence): 
    return sentence.lower().strip() 


def is_english(sentence): 
    try: 
        if langdetect.detect(sentence) == 'en': 
            return True 
        return False
    except: 
        return False


def english_key_values(enum_kv_pair): 
    i, (key, values) = enum_kv_pair
    print i 
    return key, filter(is_english, values) 


def filter_for_english(): 
    with open('inner_texts_all.json') as fp: 
        data = json.load(fp)
        print "len data:", len(data) 
    pool = Pool(cpu_count()-1)
    key_values = pool.map(english_key_values, enumerate(data.iteritems()))
    data_filtered = {k:v for k,v in key_values if v != []}
    with open('english_texts.json', 'w') as fp: 
        json.dump(data_filtered, fp)


def make_numerical_values(): 
    # get word map
    with open('google-10000-english.txt') as fp: 
        words = fp.read().split('\n')[:-1]
    assert(all(map(str.isalpha, words)))
    word_map = defaultdict(int)
    for w,i in zip(words,xrange(1,len(words)+1)): 
        word_map[w] = i
    reverse_word_map = {v:k for k,v in word_map.iteritems()}

    # load english sentences
    with open('english_texts.json') as fp: 
        english_texts = json.load(fp) 

    non_alpha_pattern = re.compile(r'[^A-Za-z ]')
    file_to_vectors = {}
    for file, sentences in english_texts.iteritems(): 
        vectors = []
        for sentence in sentences: 
            sentence = non_alpha_pattern.sub('', sentence.lower())
            sentence_split = sentence.split(' ')
            vector = [word_map[w] for w in sentence_split]
            vector = filter(lambda e: e != 0, vector)
            if len(vector) > 4: 
                print ' '.join([reverse_word_map[e] for e in vector])
                vectors.append(np.array(vector))
        file_to_vectors[file] = vectors
    
    with open('metadata.json') as fp: 
        metadata_list = json.load(fp) 
    metadata = {md['id']:md for md in metadata_list}
    classes = []
    vectors = []
    for file, file_vectors in file_to_vectors.iteritems():
        id_ = file.replace('.json', '') 
        class_ = 0 if metadata[id_]['class'] == 'safe' else 1
        for vec in file_vectors:
            vectors.append(vec) 
            classes.append(class_) 
    assert(len(classes) == len(vectors))

    classes = np.array(classes) 
    vectors = np.array(vectors) 
    np.save('xs', vectors) 
    np.save('ys', classes)
    

def main():
    filter_for_english()
    make_numerical_values()


if __name__ == "__main__":
    main()

