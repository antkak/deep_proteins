"""
Cascaded Convolution Model

- Pranav Shrestha (ps2958)
- Jeffrey Wan (jw3468)

"""

import pickle
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import Embedding, Dense, TimeDistributed, Concatenate, BatchNormalization
from keras.layers import Bidirectional, Activation, Dropout, CuDNNGRU, Conv1D
from keras import backend as K

from sklearn.model_selection import train_test_split, KFold
from keras.metrics import categorical_accuracy
from keras import backend as K
from keras.regularizers import l1, l2
# from keras.optimizers import Nadam
import tensorflow as tf
from matplotlib import pyplot


### Data Retrieval
# cb6133         = np.load("../data/cb6133.npy")
cb6133filtered = np.load("../data/cb6133filtered.npy")
cb513          = np.load("../data/cb513.npy")

print()
# print(cb6133.shape)
print(cb6133filtered.shape)
print(cb513.shape)

maxlen_seq = r = 700 # protein residues padded to 700
f = 57  # number of features for each residue

residue_list = list('ACEDGFIHKMLNQPSRTWVYX') + ['NoSeq']
q8_list      = list('LBEGIHST') + ['NoSeq']

columns = ["id", "len", "input", "profiles", "expected"]

def get_data(arr, bounds=None):
    
    if bounds is None: bounds = range(len(arr))
    
    data = [None for i in bounds]
    for i in bounds:
        seq, q8, profiles = '', '', []
        for j in range(r):
            jf = j*f
            
            # Residue convert from one-hot to decoded
            residue_onehot = arr[i,jf+0:jf+22]
            residue = residue_list[np.argmax(residue_onehot)]

            # Q8 one-hot encoded to decoded structure symbol
            residue_q8_onehot = arr[i,jf+22:jf+31]
            residue_q8 = q8_list[np.argmax(residue_q8_onehot)]

            if residue == 'NoSeq': break      # terminating sequence symbol

            nc_terminals = arr[i,jf+31:jf+33] # nc_terminals = [0. 0.]
            sa = arr[i,jf+33:jf+35]           # sa = [0. 0.]
            profile = arr[i,jf+35:jf+57]      # profile features
            
            seq += residue # concat residues into amino acid sequence
            q8  += residue_q8 # concat secondary structure into secondary structure sequence
            profiles.append(profile)
        
        data[i] = [str(i+1), len(seq), seq, np.array(profiles), q8]
    
    return pd.DataFrame(data, columns=columns)

### Train-test Specification
train_df = get_data(cb6133filtered)
test_df  = get_data(cb513)

# The custom accuracy metric used for this task
def accuracy(y_true, y_pred):
    y = tf.argmax(y_true, axis =- 1)
    y_ = tf.argmax(y_pred, axis =- 1)
    mask = tf.greater(y, 0)
    return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())

# Maps the sequence to a one-hot encoding
def onehot_to_seq(oh_seq, index):
    s = ''
    for o in oh_seq:
        i = np.argmax(o)
        if i != 0:
            s += index[i]
        else:
            break
    return s

def seq2onehot(seq, n):
    out = np.zeros((len(seq), maxlen_seq, n))
    for i in range(len(seq)):
        for j in range(maxlen_seq):
            out[i, j, seq[i, j]] = 1
    return out

# Computes and returns the n-grams of a particualr sequence, defaults to trigrams
def seq2ngrams(seqs, n = 1):
    return np.array([[seq[i : i + n] for i in range(len(seq))] for seq in seqs])

# Loading and converting the inputs to trigrams
train_input_seqs, train_target_seqs = \
    train_df[['input', 'expected']][(train_df.len.astype(int) <= maxlen_seq)].values.T
train_input_grams = seq2ngrams(train_input_seqs)

# Same for test
test_input_seqs = test_df['input'].values.T
test_input_grams = seq2ngrams(test_input_seqs)

# Initializing and defining the tokenizer encoders and decoders based on the train set
tokenizer_encoder = Tokenizer()
tokenizer_encoder.fit_on_texts(train_input_grams)
tokenizer_decoder = Tokenizer(char_level = True)
tokenizer_decoder.fit_on_texts(train_target_seqs)

# Using the tokenizer to encode and decode the sequences for use in training
# Inputs
train_input_data = tokenizer_encoder.texts_to_sequences(train_input_grams)
train_input_data = sequence.pad_sequences(train_input_data,
                                          maxlen = maxlen_seq, padding='post')

# Targets
train_target_data = tokenizer_decoder.texts_to_sequences(train_target_seqs)
train_target_data = sequence.pad_sequences(train_target_data,
                                           maxlen = maxlen_seq, padding='post')
train_target_data = to_categorical(train_target_data)

# Use the same tokenizer defined on train for tokenization of test
test_input_data = tokenizer_encoder.texts_to_sequences(test_input_grams)
test_input_data = sequence.pad_sequences(test_input_data,
                                         maxlen = maxlen_seq, padding='post')

# Computing the number of words and number of tags for the keras model
n_words = len(tokenizer_encoder.word_index) + 1
n_tags = len(tokenizer_decoder.word_index) + 1

train_input_data_alt = train_input_data
train_input_data = seq2onehot(train_input_data, n_words)
train_profiles = train_df.profiles.values

test_input_data_alt = test_input_data
test_input_data = seq2onehot(test_input_data, n_words)
test_profiles = test_df.profiles.values

train_profiles_np = np.zeros((len(train_profiles), maxlen_seq, 22))
for i, profile in enumerate(train_profiles):
    for j in range(profile.shape[0]):
        for k in range(profile.shape[1]):
            train_profiles_np[i, j, k] = profile[j, k]

test_profiles_np = np.zeros((len(test_profiles), maxlen_seq, 22))
for i, profile in enumerate(test_profiles):
    for j in range(profile.shape[0]):
        for k in range(profile.shape[1]):
            test_profiles_np[i, j, k] = profile[j, k]

def decode_results(y_, reverse_decoder_index):
    print("prediction: " + str(onehot_to_seq(y_, reverse_decoder_index).upper()))
    return str(onehot_to_seq(y_, reverse_decoder_index).upper())

def run_test(_model, data1, data2, data3, csv_name, npy_name):
    reverse_decoder_index = {value:key for key,value in tokenizer_decoder.word_index.items()}
    reverse_encoder_index = {value:key for key,value in tokenizer_encoder.word_index.items()}
    
    # Get predictions using our model
    y_test_pred = _model.predict([data1, data2, data3])

    decoded_y_pred = []
    for i in range(len(test_input_data)):
        res = decode_results(y_test_pred[i], reverse_decoder_index)
        decoded_y_pred.append(res)

    # Set Columns
    out_df = pd.DataFrame()
    out_df["id"] = test_df.id.values
    out_df["expected"] = decoded_y_pred

    # Save results
    with open(csv_name, "w") as f:
        out_df.to_csv(f, index=False)

    np.save(npy_name, y_test_pred)

def run_test_single_input(_model, data1, csv_name, npy_name):
    reverse_decoder_index = {value:key for key,value in tokenizer_decoder.word_index.items()}
    reverse_encoder_index = {value:key for key,value in tokenizer_encoder.word_index.items()}
    
    # Get predictions using our model
    y_test_pred = _model.predict(data1)

    decoded_y_pred = []
    for i in range(len(data1[:])):
        res = decode_results(y_test_pred[i], reverse_decoder_index)
        decoded_y_pred.append(res)

    # Set Columns
    out_df = pd.DataFrame()
    out_df["id"] = test_df.id.values
    out_df["expected"] = decoded_y_pred

    # Save results
    with open(csv_name, "w") as f:
        out_df.to_csv(f, index=False)

    np.save(npy_name, y_test_pred)


""" Run below for a single run """
def train(X_train, y_train, X_val=None, y_val=None):
    """
    Define model and use this function for training
    """
    model = create_CNN(n_super_blocks=2)
    assert(model is not None)
    model.compile(
        optimizer="Nadam",
        loss = "categorical_crossentropy",
        metrics = ["accuracy", accuracy])
    
    if X_val is not None and y_val is not None:
        history = model.fit( X_train, y_train,
            batch_size = 128, epochs = 100,
            validation_data = (X_val, y_val))
    else:
        history = model.fit( X_train, y_train,
            batch_size = 128, epochs = 100)

    return history, model

# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['acc'], color='blue', label='train')
    pyplot.plot(history.history['val_acc'], color='orange', label='test')
    # save plot to file
    filename = 'model_diagn'
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()

##Eddie
#n_super blocks : Number of pairs of convolutional blocks
def create_CNN(n_super_blocks=2):
    def conv_block(inp,ind):
        c1 = Conv1D(36,1,padding='same',activation='linear',name='c1_'+str(ind))(inp)
        c3=Conv1D(64,3,padding='same',activation='linear',name='c3_'+str(ind))(inp)
        c7=Conv1D(64,7,padding='same',activation='linear',name='c7_'+str(ind))(inp)
        c9=Conv1D(64,9,padding='same',activation='linear',name='c9_'+str(ind))(inp)
        conc = Concatenate(axis=-1,name='conc_'+str(ind))([c3,c7,c9])
        bn = BatchNormalization(name='bn_'+str(ind))(conc)
        drop = Dropout(0.4,name='drop_'+str(ind))(bn)
        act = Activation('relu',name='relu_'+str(ind))(drop)
        
        cc9 = Conv1D(27,9,padding='same',activation='linear',name='cc9_'+str(ind))(act)
        bn2 =  BatchNormalization(name='bn2_'+str(ind))(cc9)
        drop2 = Dropout(0.4,name='drop2_'+str(ind))(bn2)
        act2 = Activation('relu',name='relu2_'+str(ind))(drop2)
        
        concat2 = Concatenate(axis=-1,name='conc2_'+str(ind))([c1,act,act2])
        return concat2

    def dense_block(inp,ind):
        d1 = TimeDistributed(Dense(455,activation='linear',name='d_'+str(ind)))(inp)
        bn = BatchNormalization(name='bnd_'+str(ind))(d1)
        drop = Dropout(0.2,name='d_dropout_'+str(ind))(bn)
        act = Activation('relu',name='relu_d_'+str(ind))(drop)
        return act

    
    c_ind=0
    d_ind=0
    inp = Input((700,22))
    inpp = inp

    for i in range(n_super_blocks):
        c1 = conv_block(inpp,c_ind)
        c2 = conv_block(c1,c_ind+1)
        c_ind+=2
        inpp = dense_block(c2,d_ind)
        d_ind+=1

    o = TimeDistributed(Dense(9,activation='softmax'))(inpp)
    m = Model(inp,o)
    m.summary()
    return m

print(train_input_data.shape)
print(train_input_data_alt.shape)
print(train_profiles_np.shape)
print(train_target_data.shape)

randomize = np.arange(len(train_target_data))
np.random.shuffle(randomize)

train_input_data = train_input_data[randomize]
train_input_data_alt = train_input_data_alt[randomize]
train_profiles_np =  train_profiles_np[randomize]
train_target_data = train_target_data[randomize]

val_p = 0.2
vn = int(val_p*train_target_data.shape[0])

# # To use 3.3 Bidirectional GRU with convolutional blocks from paper (using a validation set) use:
# X_train = [train_input_data[vn:,:,:], train_input_data_alt[vn:,:], train_profiles_np[vn:,:,:]]
# y_train = train_target_data[vn:,:,:]
# X_val = [train_input_data[:vn,:,:], train_input_data_alt[:vn,:], train_profiles_np[:vn,:,:]]
# y_val = train_target_data[:vn,:,:]

# # To use 3.3 Bidirectional GRU with convolutional blocks from paper (without a validation set) use:
# X_train = [train_input_data, train_input_data_alt, train_profiles_np]
# y_train = train_target_data
# X_val = None
# y_val = None

# To use any other model with a simple one hot residue encoding (using a validation set) use:
X_train = train_input_data[vn:,:,:]
y_train = train_target_data[vn:,:,:]
X_val = train_input_data[:vn,:,:]
y_val = train_target_data[:vn,:,:]
print('X_train shape: ' + str(X_train.shape))
print('y_train shape: ' + str(y_train.shape))
print('X_val shape: ' + str(X_val.shape))
print('y_val shape: ' + str(y_val.shape))


history, model = train(X_train, y_train, X_val=X_val, y_val=y_val)

# Save the model as a JSON format
model.save_weights("cb513_weights_1.h5")
with open("model_tyt.json", "w") as json_file:
    json_file.write(model.to_json())

# Save training history for parsing
with open("history_tyt.pkl", "wb") as hist_file:
    pickle.dump(history.history, hist_file)


# Predict on test dataset and save the output (1 input model)
run_test_single_input(model,
    test_input_data[:],
    "cb513_test_1.csv", "cb513_test_prob_1.npy")

# # Predict on test dataset and save the output (3 input model)
# run_test(model,
#     test_input_data[:],
#     test_input_data_alt[:],
#     test_profiles_np[:],
#     "cb513_test_1.csv", "cb513_test_prob_1.npy")
""" End single run """

summarize_diagnostics(history)