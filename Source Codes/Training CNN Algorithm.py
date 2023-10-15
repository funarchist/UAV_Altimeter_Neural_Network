from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from keras.models import model_from_json
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import json
import numpy as np
import pickle

arrays = []

with open('../input/inputs/trainImagesX.pickle', 'rb') as trainImagesX:
    trainImagesX = pickle.load(trainImagesX)

with open('../input/inputs/validationImagesX.pickle', 'rb') as validationImagesX:
    validationImagesX = pickle.load(validationImagesX)

with open('../input/inputs/trainAttrX.pickle', 'rb') as trainAttrX:
    trainAttrX = pickle.load(trainAttrX)

with open('../input/inputs/validationAttrX.pickle', 'rb') as validationAttrX:
    validationAttrX = pickle.load(validationAttrX)

def create_cnn(width, height, depth, filters=(16, 32, 64, 128)):
    inputShape = (height, width, depth)
    chanDim = -1
    inputs = Input(shape=inputShape)
    for (i, f) in enumerate(filters):
        if i == 0:
            x = inputs
    x = Conv2D(f, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)
    x = Dense(4)(x)
    x = Activation("relu")(x)
    x = Dense(1, activation="linear")(x)
    model = Model(inputs, x)
    return model

trainImagesX = np.asarray(trainImagesX)
validationImagesX = np.asarray(validationImagesX)
trainAttrX = np.asarray(trainAttrX)
validationAttrX = np.asarray(validationAttrX)

model = create_cnn(64, 64, 3)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

print(model.summary())

history = model.fit(x=trainImagesX, y=trainAttrX,
validation_data=(validationImagesX, validationAttrX), epochs=50, batch_size=8)
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,51)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model_json = model.to_json()

with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")16