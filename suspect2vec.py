import numpy as np 
import tensorflow as tf
import warnings 
warnings.filterwarnings('ignore')

class Suspect2Vec(object):

    def __init__(self):
        '''
        '''
        pass
        
                
    def fit(self, data):
        '''
        '''        
        # preprocessing to convert suspects in data to integers from 0..n-1
        self.suspect_union = set([])
        for suspect_set in data:
            self.suspect_union = self.suspect_union.union(set(suspect_set))
        self.suspect_union = list(self.suspect_union)
        n = len(self.suspect_union)
        
        self.suspect2id = dict(zip(self.suspect_union, range(n)))

        train_data = []
        for S in data:
            train_data.append([self.suspect2id[s] for s in S])
        
        
    def predict(self, sample):
        '''       
        '''
        
        sample = [self.suspect2id[s] for s in sample if s in self.suspec2id]
            
