from utils import word_iterator, clean_word
from glob import glob
import os
import pickle as pkl
from collections import Counter
import numpy as np
from keras_preprocessing.sequence import pad_sequences
import torch
from sklearn.model_selection import train_test_split
import argparse
import logging

logging.basicConfig(level=logging.INFO)

all_letters = Counter()
all_diacritics= Counter()


def get_word_labels(word):
    tuples = []
    for t in word_iterator(word):
        tuples.append(t)

    if len(tuples) <= 0:
        return (), ()

    cln_word, labels = zip(*tuples)
    assert len(cln_word) == len(labels)
    
    return cln_word, labels


def get_sent_labels_from_file(file_name, max_sentence_length):

    logging.info ("Processing file %s..." %(file_name))
    cur_sent = []
    cur_labels = []

    total_sents= []
    total_labels = []

    with open(file_name, 'r', encoding='utf-8') as f:
        for word in f: # move word by word
            word = word.strip()
            if not word: # end of sentence    
                if len(cur_sent) <= max_sentence_length:      
                    total_sents.append(cur_sent)
                    total_labels.append(cur_labels)
                cur_sent=[]
                cur_labels=[]

                continue
    
            # get clean word and its diacritizations
            cln_word, diacrtics = get_word_labels(word)
            if not cln_word or not diacrtics: # empty word
                continue
            
            # add letters and diacritization suymbols to dicts
            for letter in cln_word:
                all_letters[letter]+=1
            
            for letter in diacrtics:
                all_diacritics[letter]+=1

            cur_sent.append(cln_word)
            cur_labels.append(diacrtics)
            
            assert len(cur_sent) == len(cur_labels)

        
        if len(cur_sent) <= max_sentence_length: # last sentence
            total_sents.append(cur_sent)
            total_labels.append(cur_labels)

    assert len(total_sents) == len(total_labels)
    return total_sents, total_labels



all_sents = []
all_labels = []

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus-dir', type=str, required=True)
    parser.add_argument('--max-sentence-length', type=int, default=256)
    parser.add_argument('--outdir', type=str, default='data/bin-dataset')

    args = parser.parse_args()
    
    #read files recursively
    files = glob(os.path.join(args.corpus_dir, '**', '*'), recursive=True)


    for filename in files:
        sent, lbl = get_sent_labels_from_file(filename, args.max_sentence_length)
        all_sents.extend(sent)
        all_labels.extend(lbl)
        
            
    common_letters, _ = zip(*all_letters.most_common(len(all_letters)))
    common_diacrtitics, _  = zip(*all_diacritics.most_common(len(all_diacritics)))

    letters_to_id = dict(zip(common_letters, range(len(common_letters))))
    diacritics_to_id = dict(zip(common_diacrtitics, range(len(common_diacrtitics))))

  
    letters_to_id['UNK'] = len(letters_to_id)
    letters_to_id['PAD'] = len(letters_to_id) # PAD symbols for letters
    diacritics_to_id['PAD'] = len(diacritics_to_id) # ignore symbol for labels

    logging.info('letters dict\n', letters_to_id)
    logging.info('diacritics dict\n', diacritics_to_id)

    logging.info("Writing dicts...")
    pkl.dump(letters_to_id, open(os.path.join(args.outdir, 'letter_dict.pkl'), 'wb'))
    pkl.dump(diacritics_to_id, open(os.path.join(args.outdir, 'labels_dict.pkl'), 'wb'))

    all_sents_ids = []
    all_labels_ids = []

    for sent, labels in zip(all_sents, all_labels):
        sent = sum(sent, ())
        labels = sum(labels, ())
        sent_ids = [letters_to_id[c] if c in letters_to_id else letters_to_id['UNK'] for c in sent]
        label_ids = [diacritics_to_id[d] if d in diacritics_to_id else diacritics_to_id['O'] for d in labels]
        all_sents_ids.append(sent_ids)
        all_labels_ids.append(label_ids)
    
    
    all_sents_ids = pad_sequences(all_sents_ids, maxlen=MAX_SENT_LENGTH, padding='post', value=letters_to_id['PAD'])
    all_labels_ids = pad_sequences(all_labels_ids, maxlen=MAX_SENT_LENGTH, padding='post', value=diacritics_to_id['PAD'])

    train_x_ids, val_x_ids, train_y_ids, val_y_ids = train_test_split(all_sents_ids, all_labels_ids, test_size=0.10)

    train_x_tensor = torch.LongTensor(train_x_ids)
    train_y_tensor = torch.LongTensor(train_y_ids)
    val_x_tensor = torch.LongTensor(val_x_ids)
    val_y_tensor = torch.LongTensor(val_y_ids)


    logging.info("Writing files...")

    torch.save(train_x_tensor, open(os.path.join(args.outdir, 'train_x_ids.pt', 'wb+')))
    torch.save(train_y_tensor, open(os.path.join(args.outdir, 'train_y_ids.pt', 'wb+')))

    torch.save(val_x_tensor, open('pretraining_dataset/val_x_ids.pt', 'wb+'))
    torch.save(val_y_tensor, open('pretraining_dataset/val_y_ids.pt', 'wb+'))