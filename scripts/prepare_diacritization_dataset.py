__author__='Muhammad Khalifa'

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


def get_word_labels(word):
    tuples = []
    for t in word_iterator(word):
        tuples.append(t)

    if len(tuples) <= 0:
        return (), ()

    cln_word, labels = zip(*tuples)
    assert len(cln_word) == len(labels)
    
    return cln_word, labels

def build_vocab(train_files):

    letters_dict = Counter()
    diacritics_dict = Counter()

    for file_name in train_files: 

        logging.info ("Reading file %s..." %(file_name))

        with open(file_name, 'r', encoding='utf-8') as f:
            for word in f: # move word by word
                word = word.strip()
                if not word: # end of sentence    
                    continue

                # get clean word and its diacritizations
                cln_word, diacrtics = get_word_labels(word)
                if not cln_word or not diacrtics: # empty word
                    continue
                # add letters and diacritization suymbols to dicts
                for letter in cln_word:
                    letters_dict[letter]+=1
                
                for letter in diacrtics:
                    diacritics_dict[letter]+=1

        return letters_dict, diacritics_dict

def get_ids(sent, labels, letters_dict, diacritics_dict):
    
    sent = sum(sent, ())
    labels = sum(labels, ())

    sent_ids = [letters_dict[c] if c in letters_dict else letters_dict['<unk>'] for c in sent]
    label_ids = [diacritics_dict[d] if d in diacritics_dict else diacritics_dict['O'] for d in labels] 

    return sent_ids, label_ids



def get_sent_labels_ids_from_file(file_name, max_sentence_length, letters_dict, diacritics_dict):

    logging.info ("Reading file %s..." %(file_name))
    
    cur_sent = []
    cur_labels = []

    total_sents= []
    total_labels = []

    with open(file_name, 'r', encoding='utf-8') as f:
        for word in f: # move word by word
            word = word.strip()
            if not word: # end of sentence  
                
                assert len(cur_sent) == len(cur_labels)
                cur_sent=cur_sent[:max_sentence_length]
                cur_labels=cur_labels[:max_sentence_length]  
                sent_ids, label_ids = get_ids(cur_sent, cur_labels, letters_dict, diacritics_dict)
                total_sents.append(sent_ids)
                total_labels.append(label_ids)

                cur_sent=[]
                cur_labels=[]

                continue
            # get clean word and its diacritizations
            cln_word, diacrtics = get_word_labels(word)

            if not cln_word or not diacrtics: # empty word
                continue
            
            cur_sent.append(cln_word)
            cur_labels.append(diacrtics)
            
            assert len(cur_sent) == len(cur_labels)
        
        # last sentence
        if cur_sent and cur_labels:
            cur_sent=cur_sent[:max_sentence_length]
            cur_labels=cur_labels[:max_sentence_length]  
            
            sent_ids, label_ids = get_ids(cur_sent, cur_labels, letters_dict, diacritics_dict)
            total_sents.append(sent_ids)
            total_labels.append(label_ids)

    assert len(total_sents) == len(total_labels)
    return total_sents, total_labels


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus-dir', type=str, required=True)
    parser.add_argument('--max-sentence-length', type=int, default=256)
    parser.add_argument('--vocab-size', type=int, default=36)
    parser.add_argument('--labels-size', type=int, default=9)

    parser.add_argument('--outdir', type=str, default='data/bin')

    args = parser.parse_args()
    
    #read files recursively
    files = glob(os.path.join(args.corpus_dir, '**', '*'), recursive=True)


    logging.info("building dicts..." )
    
    # iterate over training set and build dictionary
    train_files = map(lambda p:os.path.join(args.corpus_dir, 'train', p), 
    os.listdir(os.path.join(args.corpus_dir, 'train')))
    letters_dict, diacritics_dict = build_vocab(train_files)
      
            
    common_letters, _ = zip(*letters_dict.most_common(args.vocab_size))
    common_diacrtitics, _  = zip(*diacritics_dict.most_common(args.labels_size))

    letters_dict = dict(zip(common_letters, range(len(common_letters))))
    diacritics_dict = dict(zip(common_diacrtitics, range(len(common_diacrtitics))))


    # special symbols
    letters_dict['<unk>'] = len(letters_dict)
    letters_dict['<pad>'] = len(letters_dict) # <pad> symbols for letters
    diacritics_dict['<pad>'] = -1 # ignore symbol for labels

    logging.info('Letters dict:')
    logging.info(letters_dict)
    logging.info('Labels dict:')
    logging.info(diacritics_dict)

    #create output dir
    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)

    logging.info("Saving dicts...")
    pkl.dump(letters_dict, open(os.path.join(args.outdir, 'letter_dict.pkl'), 'wb+'))
    pkl.dump(diacritics_dict, open(os.path.join(args.outdir, 'labels_dict.pkl'), 'wb+'))


    #train split 

    for split in ['train','val', 'test']:
        logging.info("processing split %s..." %(split))
        
        all_sents_ids, all_labels_ids=[], []

        for fname in os.listdir(os.path.join(args.corpus_dir, split)):
            fname = os.path.join(args.corpus_dir, split,  fname)
            sent, lbl = get_sent_labels_ids_from_file(fname, args.max_sentence_length, letters_dict, diacritics_dict)
            all_sents_ids.extend(sent)
            all_labels_ids.extend(lbl)

        
        all_sents_ids = pad_sequences(all_sents_ids, maxlen=args.max_sentence_length, padding='post', value=letters_dict['<pad>'])
        all_labels_ids = pad_sequences(all_labels_ids, maxlen=args.max_sentence_length, padding='post', value=diacritics_dict['<pad>'])

        x_tensor = torch.LongTensor(all_sents_ids)
        y_tensor = torch.LongTensor(all_labels_ids)


        logging.info("Writing %s files..." %(split))

        torch.save(x_tensor, open(os.path.join(args.outdir, '%s_x_ids.pt' %(split)), 'wb+'))
        torch.save(y_tensor, open(os.path.join(args.outdir, '%s_y_ids.pt' %(split)), 'wb+'))

