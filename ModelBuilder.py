import os
from keras.callbacks import EarlyStopping
from keras.layers import Dropout,Flatten,Dense
from keras.preprocessing.image import ImageDataGenerator
training_dir='Data'
validation_dir='Validation'
training_rock_dir='Data\Rock'
training_paper_dir='Data\Paper'
training_scissors_dir='Data\Scissors'
train_datagen = ImageDataGenerator(rescale=1./255 ,fill_mode="nearest",rotation_range=30,  width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True,vertical_flip=True,brightness_range=[-0.9,+0.9],shear_range=0.2,
zoom_range=0.2,
)
test_datagen = ImageDataGenerator(rescale=1./255)
print('The total training images of rock is',len(os.listdir(training_rock_dir)))

print('The total training images of paper is',len(os.listdir(training_paper_dir)))

print('The total training images of scissors is',len(os.listdir(training_scissors_dir)))
import keras
from keras import layers
from keras import models
keras.backend.clear_session()
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
input_shape=(300, 200, 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(Dropout(0.25))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
#model.add(layers.Conv2D(128, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(Dropout(0.25))
#model.add(layers.Conv2D(128, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))

#model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#model.add(layers.MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
import keras
model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])

train_generator = train_datagen.flow_from_directory(
training_dir,
target_size=(300, 200),
batch_size=64,
color_mode="grayscale",
class_mode='categorical')
validation_generator=test_datagen.flow_from_directory(validation_dir,target_size=(300,200),
batch_size=64,class_mode='categorical',color_mode="grayscale",)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=45)

history = model.fit_generator(
train_generator,callbacks= [es],
epochs=160,
validation_data=validation_generator,)
model.save('RockPaperScissors5.h5')
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
