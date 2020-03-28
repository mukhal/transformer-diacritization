# -*- coding: utf-8 -*-
import re
import os
import subprocess
from glob import glob
from xml.etree import ElementTree as ET
from utils import *
import argparse
import numpy as np 

class Preprocessor():
    def __init__(self):
        self.VOWEL_REGEX = re.compile('|'.join(['ُ', 'َ', 'ِ', 'ْ']))
        #for removing numbers and punctuations except [., !, ؟]
        self.NOISE_REGEX = r'([\d\\/\(\)\[\]\|\-’÷×*+_<>«»@#$%^&:]+)'
        self.out_dir = 'data/preprocessed'
        create_dir(self.out_dir) #create directory if it wasn't existed

    def preprocess(self, data_dir):
        """
        This method takes a path to the data as an input, then it does
        two things actually:
        -> clean this text from noise like English letters/numbers, some 
           punctuations symbols that are no good for us
        -> split the data into sentences and write those sentences in another
           directory (self.out_dir) where each word in the sentence is
           written in a seperate line. These sentences are seperated by
           a newline character (\n). Each file should contain roughly one
           million words in it.
        This function returns nothing.
        """
        # the first ** means every file and dir under 'diacritized_text'
        # the second * means every file in every directory
        files = glob(os.path.join(data_dir, '**', '*'), recursive=True)
        # We have around 397 files divided as:
        # -> 97 files from 'http://www.al-islam.com'
        # -> 2 files from 'aljazeera'
        # -> 170 files from 'al-kalema.org'
        # -> 39 files from 'enfal.de'
        # -> 1 file from 'manual'
        # -> 4 XML files from 'sulaity'
        # -> 56 files from 'diwanalarab'
        # -> 20 files from 'mohamed bn abdel-wahab'
        # -> 8 directories
        print("Preprocessing...")
        outFiles_count = 1
        word_count = 0
        outFile = open(os.path.join(self.out_dir, str(outFiles_count)), 'w', encoding='utf-8')
        for filename in files:
            #if filename is a directory
            if os.path.isdir(filename):
                continue
            #if filename is an xml file
            elif filename[:-4] == ".xml":
                tree = ET.parse(filename)
                content = "\n".join([node.text for node in tree.findall('.//text/body/p')])
            #otherwise
            else:
                with open(filename, 'r',encoding='utf-8') as fout:
                    #print("****", filename)
                    content = fout.read() #convert text from bytes into string
                    
            for sentence in re.split(r'؟|!|\.+', content):
                sentence = re.sub(self.NOISE_REGEX, '', sentence) #remove numbers and punctuations
                sentence = sentence.strip() #remove whitespaces at the end

                INSIDE = False #to make sure it found a diacritized word
                for word in re.split(r'\s+', sentence):
                    #make sure that the sentence is diacritized
                    if re.search(self.VOWEL_REGEX, word):
                        INSIDE = True
                        outFile.write(word +'\n')
                        word_count += 1
                if INSIDE:
                    outFile.write('\n')
                if word_count >= 10**6:
                    outFile.close()
                    word_count = 0
                    outFiles_count += 1
                    outFile = open(os.path.join(self.out_dir, str(outFiles_count)), 'w', encoding='utf-8')
            #print("DONE:", word_count)
        print("Done.")

    def split(self, test_ratio=0.2, train_ratio=0.7):
        """
        This method takes a train-test ratio as an input (20% default value)
        then, it splits the preprocessed data into two directories (train, test)
        using the given ratio. So, if we have 100 files and the given ratio is 30%,
        then we would have two directories:
        -> train with 70 files in it.. [1, 2, 3, ... 70]
        -> test with 30 files in it ..[71, 72, ... 100]
        """
        assert 0 <= test_ratio <= 1 and 0 <= train_ratio <= 1, 'Invalid Number for ratio'
        
        val_ratio = 1 - train_ratio - test_ratio
        num_files = len(os.listdir(self.out_dir))
        num_files = 10

        train, val, test = np.split(range(1, num_files+1), 
            [int(train_ratio* num_files), 
            int((train_ratio + val_ratio)* num_files)])
        
        print(train, val, test)
        train_dir = os.path.join(self.out_dir, 'train')
        create_dir(train_dir)
        val_dir = os.path.join(self.out_dir, 'val')
        create_dir(val_dir)
        test_dir = os.path.join(self.out_dir, 'test')
        create_dir(test_dir)
        #test_dir = os.path.join(self.out_dir, 'test', 'gold')
        #create_dir(test_dir)
        
        #move train, val and test files
        for f_i in map(str, train): 
            os.rename(os.path.join(self.out_dir, f_i), os.path.join(train_dir, f_i))
        
        for f_i in map(str, val): 
            os.rename(os.path.join(self.out_dir, f_i), os.path.join(val_dir, f_i))

        for f_i in map(str, test): 
            os.rename(os.path.join(self.out_dir, f_i), os.path.join(test_dir, f_i))
            

    def remove_diacritization(self):
        """This method aims at removing any diacritization from
        the test files, then write the cleaned version into
        another directory.
        We read from the 'gold' directory and write the cleaned
        version into 'test'
        """
        gold_dir = os.path.join(self.out_dir, 'test', 'gold')
        create_dir(gold_dir)
        test_dir = os.path.join(self.out_dir, 'test', 'test')
        create_dir(test_dir)
        for in_filename in os.listdir(gold_dir):
            print('FILE:', in_filename)
            out_filename = os.path.join(test_dir, in_filename)
            in_filename = os.path.join(gold_dir, in_filename)
            with open(in_filename, 'r', encoding='utf-8') as fin:
                with open(out_filename, 'w', encoding='utf-8') as fout:
                    for word in fin.readlines():
                        word = word
                        cleaned = clean_word(word)
                        fout.write(cleaned)

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True)
    parser.add_argument('--test-ratio', type=float, default=0.2)

    args= parser.parse_args()

    p = Preprocessor()
    #p.preprocess(args.corpus)
    p.split(args.test_ratio)
    p.remove_diacritization()