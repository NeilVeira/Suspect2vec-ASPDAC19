import subprocess 
import numpy as np 
#import tensorflow as tf
import warnings 
warnings.filterwarnings('ignore')

def sigmoid(x):
    return 1/(1+np.exp(-x))
    

class Suspect2Vec(object):

    def __init__(self, dim=100, epochs=1000, eta=0.01, lambd=0.1):
        '''
        '''
        self._dim = dim 
        self._epochs = epochs 
        self._eta = eta 
        self._lambda = lambd 
        
    @staticmethod 
    def mat_vec_mul(A, b):
        return tf.squeeze(tf.matmul(A,tf.expand_dims(b,1)))        

    def _generate_batch(self):
        '''
        Yield random subsetes of the training data sets along with 1-hot encodings of 
        the full sets. 
        '''
        for i in range(len(self.train_data)):
            # TODO: different include probabilities?
            include = np.random.randint(0,2,size=len(self.suspect_union))
            sample = self.one_hot_data[i] * include
            yield sample, self.one_hot_data[i]

                
    def fit(self, data):
        '''
        Learn the embed_in to predict suspect sets. 
        data: iterable of iterables of suspects
        '''        
        # preprocessing to convert suspects in data to integers from 0..n-1
        self.suspect_union = set([])
        for suspect_set in data:
            self.suspect_union = self.suspect_union.union(set(suspect_set))
        self.suspect_union = list(self.suspect_union)
        n = len(self.suspect_union)
        m = len(data)

        self.suspect2id = dict(zip(self.suspect_union, range(n)))

        self.train_data = []
        for S in data:
            self.train_data.append(np.array([self.suspect2id[s] for s in S]))
        
        self.one_hot_data = np.zeros((m,n), dtype=np.bool_)
        for i in range(m):
            self.one_hot_data[i][self.train_data[i]] = 1
            
        with open("in.txt","w") as f:
            f.write("%i %i\n" %(m,n))
            for row in self.one_hot_data:
                f.write(" ".join(map(str,map(int,row)))+"\n")
        
        cmd = "./suspect2vec -in in.txt -out out.txt -epochs %i -dim %i -eta %.6f -lambda %.6f" \
                %(self._epochs,self._dim,self._eta,self._lambda)
        #print(cmd)
        stdout,stderr = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()
        
        with open("out.txt") as f:
            self.embed_in = []
            self.embed_out = []
            for i in range(n):
                self.embed_in.append(list(map(float,f.readline().strip().split())))
            for i in range(n):
                self.embed_out.append(list(map(float,f.readline().strip().split())))
            self.embed_in = np.array(self.embed_in)            
            self.embed_out = np.array(self.embed_out)      

        return self.embed_in 

        '''# build graph
        train_inputs = tf.placeholder(tf.float32, shape=[n])
        train_labels = tf.placeholder(tf.float32, shape=[n])
        embed_in = tf.Variable(tf.random_uniform([n,self._dim], -1, 1))

        cnt = tf.reduce_sum(train_inputs)
        embed = Suspect2Vec.mat_vec_mul(tf.transpose(embed_in), train_inputs) / cnt  

        logits = Suspect2Vec.mat_vec_mul(embed_in, embed)
        self.pred = tf.sigmoid(logits)
        ce_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=train_labels, logits=logits))

        opt = tf.train.GradientDescentOptimizer(self._eta).minimize(ce_loss)

        # Training
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        for step in range(self._epochs):
            for sample,target in self._generate_batch():
                _,loss,pred = self.sess.run([opt,ce_loss,self.pred], feed_dict={train_inputs:sample, train_labels:target})

                if np.isnan(loss):
                    print(pred)
                    print(pred-target)

                    assert False
            #if step%100 == 0:
            print("Step %i loss: %.4f" %(step,loss))
            #print(pred)

        self.embed_in = self.sess.run(embed_in)'''
        
        
    def predict(self, sample):
        '''       
        Predict the remaiing suspects in the given suspect subset. 
        sample: iterable of suspects
        Returns: list of suspects
        '''
        n = len(self.suspect_union)
        ret = list(sample)
        sample = [self.suspect2id[s] for s in sample if s in self.suspect2id]
        sample_vec = np.mean(self.embed_in[sample], axis=0)
        pred_probs = sigmoid(np.matmul(self.embed_out,sample_vec))
        
        for i in range(n):
            if pred_probs[i] > 0.5 and self.suspect_union[i] not in ret:
                ret.append(self.suspect_union[i])
        return ret 
