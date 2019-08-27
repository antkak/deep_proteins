# packages 
import os
from keras.layers import Conv2D, MaxPooling2D, Concatenate, UpSampling2D
from keras.models import Model, Input
from keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot
from hilbertcurve.hilbertcurve import HilbertCurve
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



# load training data
DATA_PATH = '../data/hilbert_images'
X = []
Y = []

# input and output as lists of tensors
for i in range(5534):
    X.append(np.load(DATA_PATH+'/input/train_in_{}.npy'.format(i)))
    Y.append(np.load(DATA_PATH+'/output/train_out_{}.npy'.format(i)))

# load test data
TEST_PATH = '../data/hilbert_test'
Z = []
# input and output as lists of tensors
for i in range(514):
    Z.append(np.load(TEST_PATH+'/input/test_in_{}.npy'.format(i)))
Z = np.array(Z)

# define training
def train(X_train, y_train, X_val=None, y_val=None):
    """
    Define model and use this function for training
    """
    model = create_CNN()
    assert(model is not None)
    initial_lrate = 0.001
    model.compile(
        optimizer=Adam(lr=initial_lrate),
        loss = "categorical_crossentropy",
        metrics = ["accuracy"])

    if X_val is not None and y_val is not None:
        history = model.fit( X_train, y_train,
            batch_size = 8, epochs = 10,
            validation_data = (X_val, y_val))
    else:
        history = model.fit( X_train, y_train,
            batch_size = 8, epochs = 10)

    return history, model

# define model
def create_CNN():
    def downblock(x,kernel_size=3,filters=64):
        x = Conv2D(filters,kernel_size,padding='same',activation='relu')(x)
        x = Conv2D(filters,kernel_size,padding='same',activation='relu')(x)
        return MaxPooling2D()(x), x

    def upblock(x,c,kernel_size=3,filters=64):
        x = UpSampling2D()(x)
        x = Concatenate(axis=-1)([x,c])
        x = Conv2D(filters,kernel_size,padding='same',activation='relu')(x)
        x = Conv2D(filters,kernel_size,padding='same',activation='relu')(x)
        return x

    inp = Input(shape=(32, 32, 44))
    x, c1 = downblock(inp)
    x, c2 = downblock(x)
    x = Conv2D(16,3,padding='same',activation='relu')(x)
    x = upblock(x,c2)
    x = upblock(x,c1)
    o = Conv2D(9,1,padding='same',activation='softmax')(x)
    
    m = Model(inp,o)
    m.summary()
    return m

# plot diagnostic learning curves
def summarize_diagnostics(history):
    try:
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
    except:
        # plot loss
        pyplot.subplot(211)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(history.history['loss'], color='blue', label='train')
        # plot accuracy
        pyplot.subplot(212)
        pyplot.title('Classification Accuracy')
        pyplot.plot(history.history['acc'], color='blue', label='train')
        # save plot to file
        filename = 'model_diagn'
        pyplot.savefig(filename + '_plot.png')
        pyplot.close()

# split validation and training
val_p = 0.02
vn = int(val_p*5534)
X_train = np.array(X[vn:])
Y_train = np.array(Y[vn:])
X_val   = np.array(X[:vn])
Y_val   = np.array(Y[:vn])

print(X_train[0].shape, Y_train[0].shape)

# train 
history, model = train(X_train, Y_train, X_val=X_val, y_val=Y_val)

# plot
summarize_diagnostics(history)

# save model

# predict from test set
def reverse_hilbert(tensor):
    # input 2^p x 2^p x C tensor
    p = int(np.log2(tensor.shape[0]))
    C = tensor.shape[-1]

    hilbert_curve = HilbertCurve(p, 2)

    sequence = np.zeros((4**p,C))
    for i in range(2**p):
        for j in range(2**p):
            dist = hilbert_curve.distance_from_coordinates([i,j])
            sequence[dist,:] = tensor[i,j,:]

    return sequence

def decode_sequence(sequence):
    code =['L','B','E','G','I','H','S','T','NoSeq']
    d = {i:c for i,c in enumerate(code)}
    l = sequence.shape[0]
    string = ''
    for i in range(l):
        index = np.argmax(sequence[i,:])
        if index == 8:
            break
        string += d[index]

    return string

def run_test(_model,X_test):
    # Get predictions using our model
    y_test_pred = _model.predict(X_test)

    decoded = []
    for i in range(y_test_pred.shape[0]):
        sequence = reverse_hilbert(y_test_pred[i,:,:,:])
        decoded.append(decode_sequence(sequence))

        # Set Columns
    print(decoded)
    out_df = pd.DataFrame()
    out_df["id"] = np.arange(y_test_pred.shape[0])
    out_df["expected"] = decoded
    # Save results
    with open('hilbert_solution.csv', "w") as f:
        out_df.to_csv(f, index=False)

run_test(model, Z)







