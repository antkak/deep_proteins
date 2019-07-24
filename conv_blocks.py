import keras
from keras.models import Model
from keras.layers import Conv1D, Dense, Input, Concatenate,Activation,BatchNormalization,Dropout,TimeDistributed

import numpy as np





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


