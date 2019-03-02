from PIL import Image
from utils import preprocess_input
import numpy as np
from sklearn import metrics
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPool2D,Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.models import load_model


def get_data(data_path_file):
    with open(data_path_file,'r',encoding='utf-8') as f:
        data = f.readlines()

    content=[]
    label=[]
    for d in data:
        tmp = d.split('\t')
        im = Image.open(tmp[0])
        x = preprocess_input(np.array(im,dtype='float32'))
        content.append(x)
        label.append(int(tmp[-1].strip().strip('\n'))-1)
    return np.array(content), label

def ToneNet(train_data,train_label,test_data,test_label, wigth,heigth,channels,lr,activation,epochs,batch_size):
    train_label = np_utils.to_categorical(train_label,num_classes = 4)
    test_label = np_utils.to_categorical(test_label,num_classes = 4)

    model = Sequential()

    model.add(Conv2D(
        filters=64,  
        kernel_size=(5,5),
        strides=(3,3),  
        padding='same',  
        input_shape=(wigth, heigth, channels),  
    ))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(MaxPool2D(
        pool_size=(3,3),  
        strides=(3,3),  
        padding='same', 
    ))


    model.add(Conv2D(
        filters=128,  
        kernel_size=(3,3),
        strides=(1,1),  
        padding='same'  
        
    ))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(MaxPool2D(
        pool_size=(2,2), 
        strides=(2,2),  
        padding='same', 
    ))


    model.add(Conv2D(
        filters=256,  
        kernel_size=(3,3),
        strides=(1,1),  
        padding='same'  
       
    ))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(MaxPool2D(
        pool_size=(2,2), 
        strides=(2,2),  
        padding='same', 
    ))


    model.add(Conv2D(
        filters=256,  
        kernel_size=(3,3),
        strides=(1,1),  
        padding='same'  
       
    ))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(MaxPool2D(
        pool_size=(2,2),  
        strides=(2,2),  
        padding='same', 
    ))


    model.add(Conv2D(
        filters=512,  
        kernel_size=(3,3),
        strides=(1,1),  
        padding='same'  
       
    ))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(MaxPool2D(
        pool_size=(2,2),  
        strides=(2,2),  
        padding='same', 
    ))
    
    model.add(Flatten())  
    model.add(Dense(1024))  
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dense(1024))  
    model.add(BatchNormalization())
    model.add(Activation(activation))

    model.add(Dense(4))  
    model.add(Activation('softmax'))  
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    print("training ==========================")
    checkpointer = ModelCheckpoint(filepath="Model/ToneNet.hdf5", verbose=1, save_best_only=True)
    model.fit(train_data, train_label, validation_split=0.1,shuffle=True, \
                epochs=epochs,verbose=1,batch_size=batch_size,callbacks=[checkpointer])  
    
   
    print("Testing ===========================")
    loss, accuracy = model.evaluate(test_data, test_label)
    print("loss:", loss)
    print("Test:", accuracy)


def predict(model, file_path):

    model = load_model(model)
    
    test_data,test_label=get_data(file_path)

    output_o = model.predict(test_data, batch_size=len(test_data))
    output = np.argmax(output_o,axis=1)

    confusion_matrix = metrics.confusion_matrix(test_label, output)
    accuracy = metrics.accuracy_score(test_label, output)

    precision = metrics.precision_score(test_label, output,average='macro')
    recall = metrics.recall_score(test_label, output,average='macro')
    f1_score = 2*recall*precision / (recall+precision)

    print(confusion_matrix)
    print('accuracy:',accuracy,'precision:',precision,'recall:',recall,'f1_score:',f1_score)
    return output_o



def main():
    train_data,train_label=get_data('train')
    test_data,test_label=get_data('test')
    wigth = 225
    heigth = 225
    channels = 3
    lr = 0.001
    activation = 'relu'
    epochs = 50
    batch_size = 128
    ToneNet(train_data,train_label,test_data,test_label, wigth,heigth,channels,lr,activation,epochs,batch_size)


main()
