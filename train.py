from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

data_path = "dataset/"
model = "covid.model"
plot = "plot.png"

INIT_LR = 1e-3
EPOCHS = 25
BS = 8

# load the images from the dataset folder
print("Loading Images")
data = []
labels = []

# loop over the images
for folder in os.listdir(data_path):
    if folder[:3] != ".DS":
        imagePath = data_path + folder + "/"
        for filename in os.listdir(imagePath):
            
            # get the class label
            labels.append(folder)

            # get the image, swap color channels, and resize to 224x224
            image = cv2.imread(os.path.join(imagePath, filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            data.append(image)


# convert the data and labels to NumPy arrays
# and scale the pixel rnage to 0-1
data = np.array(data)/ 255.0
labels = np.array(labels)

# one-hot encode the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# split into train and test set
print("Splitting Dataset")
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels)
print("Training Images: {} images \nTesting Images: {} images".format(len(trainX), len(testX)))

# add data augmentation
data_aug = ImageDataGenerator(rotation_range=15, fill_mode="nearest")

# create the neural network model
print("Building Neural Network")

# load the VGG16 base model without the fully connected layers
base = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# freeze the layers of the base model
for layer in base.layers:
    layer.trainable = False

head = base.output
head = AveragePooling2D(name="pool", pool_size=(4, 4))(head)
head = Flatten(name="flatten")(head)
head = Dense(64, activation="relu")(head)
head = Dropout(0.5)(head)
head = Dense(2, activation="softmax")(head)

# complete the model
model = Model(inputs=base.input, outputs=head)

# compile the model
print("Compiling The Model")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])

# train the network
print("Training The Network")
hist = model.fit_generator(data_aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS, validation_data=(testX, testY),
    validation_steps=len(testX) // BS, epochs=EPOCHS)

# save the model
print("Saving The Model")
model.save(model, save_format="h5")

# evaluate the network
print("Evaluating The Network")
preds = model.predict(testX, batch_size=BS)
preds = np.argmax(preds, axis=1)

# classification report
print(classification_report(testY.argmax(axis=1), preds, target_names=lb.classes_))

# confusion matrix
matrix = confusion_matrix(testY.argmax(axis=1), preds)
total = sum(sum(matrix))
acc = (matrix[0, 0] + matrix[1, 1]) / total
sensitivity = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
specificity = matrix[1, 1] / (matrix[1, 0] + matrix[1, 1])

print(matrix)
print("accuracy: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

# plot the loss and accuracy
plt.figure()
plt.plot(np.arange(0, EPOCHS), hist.history["loss"], label="Training Loss")
plt.plot(np.arange(0, EPOCHS), hist.history["val_loss"], label="Validation Loss")
plt.plot(np.arange(0, EPOCHS), hist.history["accuracy"], label="Training Acc")
plt.plot(np.arange(0, EPOCHS), hist.history["val_accuracy"], label="Validation Acc")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.title("Training Loss and Accuracy", fontsize=15)
plt.legend(loc="lower left")
plt.grid(True)
plt.savefig(plot)










            

