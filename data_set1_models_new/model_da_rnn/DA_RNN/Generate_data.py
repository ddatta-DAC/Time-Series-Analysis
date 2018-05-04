import numpy as np
import random
import pandas as pd
from sklearn import preprocessing
import sklearn.model_selection

data_file_1 = './nasdaq100_padding.csv'
data_file_2 = './weather.csv'
data_set = 2

def normalize (df):

    cols = list(df.columns)
    for c in cols:
        x = df[c]
        scaler = preprocessing.StandardScaler().fit(x.values.reshape(-1, 1))
        x = scaler.transform(x.values.reshape(-1, 1))
        df[c] = x

    return df



class Input_data:
    def __init__(self, batch_size, n_step_encoder, n_step_decoder, n_hidden_encoder):                                   
        # read the data
        global data_file_1
        global data_file_2
        global data_set

        if data_set == 1:
            data = pd.read_csv(data_file_1, index_col=None)

        if data_set == 2:
            data = pd.read_csv(data_file_2, index_col=0)


        self.data = np.array(data)
        self.train_day = 90
        self.val_day = 7
        self.test_day = 7
        minutes = 390


        self.train = self.data[:self.train_day * minutes, :]
        self.val = self.data[self.train_day * minutes:(self.train_day + self.val_day) * minutes,:]
        self.test = self.data[(self.train_day + self.val_day) * minutes:(self.train_day + self.val_day + self.test_day) * minutes,:]

        print self.train.shape
        print self.val.shape
        print self.test.shape


        # split it into 85 %, 5%, 10%
        self.train, self.val, self.test = np.split(data.sample(frac=1), [int(.85 * len(data)), int(.9 * len(data))])
        self.train = np.asarray(self.train)
        self.val = np.asarray(self.val)
        self.test = np.asarray(self.test)


        # parameters for the network
        self.batch_size = batch_size
        self.n_hidden_state = n_hidden_encoder
        self.n_step_encoder = n_step_encoder
        self.n_step_decoder = n_step_decoder

        self.n_train = len(self.train)
        self.n_val = len(self.val)
        self.n_test = len(self.test)
        self.n_feature = self.data.shape[1]- 1
        self.n_label = 1

        # data normalization
        self.mean = np.mean(self.train,axis=0)
        self.stdev = np.std(self.train,axis=0)
        # in case the stdev=0,then we will get nan
        for i in range (len(self.stdev)):
            if self.stdev[i]<0.00000001:
                self.stdev[i] = 1
        self.train = (self.train-self.mean)/self.stdev
        self.test = (self.test-self.mean)/self.stdev
        self.val = (self.val - self.mean)/self.stdev


        print self.train.shape
        print self.val.shape
        print self.test.shape

        return


    def next_batch(self):
        # generate of a random index from the range [0, self.n_train -self.n_step_decoder +1]                 
        index = random.sample(np.arange(0,self.n_train-self.n_step_decoder),self.batch_size)  
        index = np.array(index)
        # the shape of batch_x, label, previous_y                 
        batch_x = np.zeros([self.batch_size,self.n_step_encoder, self.n_feature])
        label = np.zeros([self.batch_size, self.n_label])
        previous_y = np.zeros([self.batch_size,self.n_step_decoder, self.n_label])                      
        temp = 0

        for item in index:
            batch_x[temp,:,:] = self.train[item:item+self.n_step_encoder, :self.n_feature]             
            previous_y[temp,:,0] = self.train[item:item + self.n_step_decoder, -1]
            temp += 1
        label[:,0] = np.array(self.train[index + self.n_step_decoder, -1])                 
        encoder_states = np.swapaxes(batch_x, 1, 2)
        return batch_x, label, previous_y, encoder_states

        
    def validation(self):  
        index = np.arange(0, self.n_val-self.n_step_decoder)
        index_size = len(index)
        val_x = np.zeros([index_size, self.n_step_encoder, self.n_feature])
        val_label = np.zeros([index_size, self.n_label])
        val_prev_y = np.zeros([index_size, self.n_step_decoder, self.n_label])
        temp = 0
        for item in index:
            val_x[temp,:,:] = self.val[item:item + self.n_step_encoder, :self.n_feature]
            val_prev_y[temp,:,0] = self.val[item:item + self.n_step_decoder, -1]        
            temp += 1

        val_label[:, 0] = np.array(self.val[index + self.n_step_decoder, -1])
        encoder_states_val = np.swapaxes(val_x,1,2)
        return val_x, val_label, val_prev_y, encoder_states_val


        
    def testing(self):
        index = np.arange(0,self.n_test-self.n_step_decoder)
        index_size = len(index)
        test_x = np.zeros([index_size, self.n_step_encoder, self.n_feature])
        test_label = np.zeros([index_size, self.n_label])
        test_prev_y = np.zeros([index_size, self.n_step_decoder, self.n_label])
        temp = 0
        for item in index:
            test_x[temp,:,:] = self.test[item:item + self.n_step_encoder, :self.n_feature]
            test_prev_y[temp,:,0] = self.test[item:item + self.n_step_decoder, -1]        
            temp += 1

        test_label[:, 0] = np.array(self.test[index + self.n_step_decoder, -1])
        encoder_states_test = np.swapaxes(test_x,1,2)
        return test_x, test_label, test_prev_y, encoder_states_test
    
 