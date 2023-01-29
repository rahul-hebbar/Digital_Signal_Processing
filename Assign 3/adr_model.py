import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # turning of logging WARNINGS and INFO

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    data = np.load('d.npy') # loading the data images
    label = np.load('l.npy') # loading corresponding labels
    traind,testd,trainl,testl = train_test_split(data,label,test_size = 0.1,random_state=42)
    """
    the above code helps split the data and labels into test and train data 
    and shuffle the contents in the respective numpy arrays
    """
    print(traind.shape,trainl.shape,testd.shape,testl.shape) # get the shape of all four arrays after split

    traind = traind/255.0 # normalizing the BGR channel values
    testd = testd/255.0 # normalizing the BGR channel values

    # the neural network is defined
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64 ,(5,5),activation = 'relu', input_shape = (100,100,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax")])
        
    # all the layers defined are complied with the optimizer and loss functions
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss = 'sparse_categorical_crossentropy',
                  metrics = ['accuracy'])
    
    #training the model with train data and corresponding labels
    model.fit(traind, trainl, epochs = 6, validation_split = 0.1)

    #testing our model with test data
    test_loss, test_acc = model.evaluate(testd, testl)

    #printing model accuracy percentage
    print(test_acc*100)

    #saving the model as a h5 file for further use
    model.save("adr_model.h5")            