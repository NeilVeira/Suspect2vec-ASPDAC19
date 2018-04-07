import numpy as np 
import tensorflow as tf
import warnings 
warnings.filterwarnings('ignore')


class Suspect2Vec(object):

    def __init__(self, dim=100, epochs=1000, eta=0.01):
        '''
        '''
        self._dim = dim 
        self._epochs = epochs 
        self._eta = eta 

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
        Learn the embeddings to predict suspect sets. 
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

        # build graph
        train_inputs = tf.placeholder(tf.float32, shape=[n])
        train_labels = tf.placeholder(tf.float32, shape=[n])
        embeddings = tf.Variable(tf.random_uniform([n,self._dim], -1, 1))

        cnt = tf.reduce_sum(train_inputs)
        embed = Suspect2Vec.mat_vec_mul(tf.transpose(embeddings), train_inputs) / cnt  

        logits = Suspect2Vec.mat_vec_mul(embeddings, embed)
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

        self.embeddings = self.sess.run(embeddings)

        
    def predict(self, sample):
        '''       
        Predict the remaiing suspects in the given suspect subset. 
        sample: iterable of suspects
        Returns: list of suspects
        '''
        ret = list(sample)
        sample = [self.suspect2id[s] for s in sample if s in self.suspect2id]
        pred_probs = self.sess.run([self.pred], feed_dict={train_inputs:sample})
        n = len(self.suspect_union)
        
        for i in range(n):
            if pred_probs[i] > 0.5 and self.suspectunion[i] not in ret:
                ret.append(self.suspectunion[i])
        return ret 
