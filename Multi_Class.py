import matplotlib.pyplot as plt
import numpy as np
import sys
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation
#from keras.layers import Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
#from keras import callbacks
from keras import regularizers
DEV = False
argvs = sys.argv
argc = len(argvs)

epochs = 200

train_data_path = 'F:/paper2019/CNN_OUT/training'
validation_data_path = 'F:/paper2019/CNN_OUT/validation'

"""
Parameters
"""
img_width, img_height = 64, 64 # when running on CPU
#img_width, img_height =128, 128 # when running on GPU
batch_size = 32
samples_per_epoch = 1000  # no. of total images in training set
validation_steps = 300   # no. of images in vaidation set
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 2
pool_size = 2
classes_num = 3
lr = 0.0001

model = Sequential()
model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, border_mode ="same", input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, border_mode ="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size), dim_ordering='th'))

model.add(Flatten())
model.add(Dense(256,kernel_regularizer = regularizers.l2(.01)))

################################
#model.add(Dense(128,kernel_regularizer = regularizers.l2(.01)))
###############################################################

model.add(Activation("relu"))

##########################
#model.add(Dense(units = 128, activation = 'relu',kernel_regularizer = regularizers.l2(.1)))
#model.add(Dropout(0.5))
##########################
model.add(Dense(classes_num, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])


trainAug = ImageDataGenerator(
	rescale=1 / 255.0,
	rotation_range=20,
	zoom_range=0.05,
	width_shift_range=0.05,
	height_shift_range=0.05,
	shear_range=0.05,
	horizontal_flip=True,
 #   vertical_flip = True,
	fill_mode="nearest")

valAug = ImageDataGenerator(rescale=1 / 255.0)


train_generator = trainAug.flow_from_directory(
	train_data_path,
	class_mode="categorical",
	target_size=(64, 64),
#    target_size=(128,128),
	color_mode="rgb",
	shuffle=True,
	batch_size=batch_size)

validation_generator = valAug.flow_from_directory(
	validation_data_path,
	class_mode="categorical",
	target_size=(64, 64),
 #   target_size=(128,128),
	color_mode="rgb",
	shuffle=False,
	batch_size=batch_size)



history = model.fit_generator(
    train_generator,
    samples_per_epoch=samples_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps)

#plt.figure(figsize=[8,6])
#plt.plot(history.history['loss'],'r',linewidth=3.0)
#plt.plot(history.history['val_loss'],'b',linewidth=3.0)
#plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
#plt.xlabel('Epochs ',fontsize=16)
#plt.ylabel('Loss',fontsize=16)
#plt.title('Loss Curves',fontsize=16)

N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
#plt.savefig(args["plot"])  
