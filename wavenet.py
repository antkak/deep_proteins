import keras
from keras import Model
from keras import backend as K
import keras_metrics as km
from keras.models import load_model
from keras.layers import TimeDistributed,Activation,Flatten,Conv1D, Dense, Input, Reshape, Concatenate,BatchNormalization, Multiply, RepeatVector,MaxPooling1D, Dropout, Add, Average
from keras.callbacks import TensorBoard
from keras.optimizers import Adam, SGD
from keras.losses import categorical_crossentropy








def wavenet_proteins():
    def wavenet_block(n_filters, filter_size, dilation_rate):
        def f(input_):
            residual = input_
            tanh_out = Conv1D(n_filters,filter_size,dilation_rate=dilation_rate, activation='tanh', padding='causal')(input_)
            sig_out = Conv1D(n_filters,filter_size,dilation_rate=dilation_rate, activation='sigmoid', padding='causal')(input_)
            #merged = Conv1D(n_filters,filter_size,dilation_rate=dilation_rate,activation='custom_activation',padding='same')(input_)
            merged = Multiply()([tanh_out,sig_out])
            skip_out = Conv1D(1,1,activation='relu',padding='same')(merged)
            _out = Add()([skip_out,residual])
            return _out,skip_out
        return f


    inp  = Input(shape=(maxlen_seq, n_words))
    inp_profiles = Input((maxlen_seq,22))
    inp_c = Concatenate(axis=-1)([inp,inp_profiles])
    a,b = wavenet_block(256,3,1)(inp)
    skip_connections=[b]
    for i in range(9):
        a,b = wavenet_block(612,3,2**((i+1)%9))(a)
        skip_connections.append(b)
    n= Add()(skip_connections)
    n = Activation('relu')(n)
    n = Conv1D(128,1,activation='relu',padding='same')(n)
    
    o = TimeDistributed(Dense(9,activation='softmax'))(n)
    m = Model([inp,inp_profiles],o)
    m.summary()
    return m