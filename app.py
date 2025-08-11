import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

# Konfiguracja generatorów danych
train_datagen = ImageDataGenerator(rescale=1./255.0,
                                   rotation_range=30,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'C:...train',
    target_size=(100, 100),
    batch_size=64,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    'C:...test,
    target_size=(100, 100),
    batch_size=64,
    class_mode='categorical'
)

# Budowa modelu CNN
model = Sequential()

model.add(Conv2D(32, (3,3), padding='same', input_shape=(100,100,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(64, (3,3), padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(128, (3,3), padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())

model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.3))

model.add(Dense(5, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(train_generator,
                    steps_per_epoch=20,
                    epochs=20,
                    validation_data=validation_generator,
                    validation_steps=200)

# Ewaluacja modelu
test_loss, test_acc = model.evaluate(validation_generator)
print('Dokładność testu:', test_acc)

# Przewidywanie etykiet dla danych walidacyjnych
y_pred = model.predict(validation_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Obliczanie macierzy pomyłek
confMatrix = confusion_matrix(validation_generator.classes, y_pred_classes)

# Tworzenie ramki danych (DataFrame) z macierzy pomyłek
confMatrix = pd.DataFrame(confMatrix, index=range(5), columns=range(5))

# Wyświetlanie macierzy pomyłek
print('\nMacierz pomyłek:')
print(confMatrix)

# ROC wykres
# Uzyskaj prawdopodobieństwa przewidywane przez model dla danych testowych
y_pred_prob = model.predict(validation_generator)

# Konwertuj etykiety kategorii na binarne etykiety
lb = LabelBinarizer()
lb.fit(validation_generator.classes)
y_test_bin = lb.transform(validation_generator.classes)

# Oblicz krzywe ROC dla każdej klasy
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(validation_generator.class_indices)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Narysuj krzywe ROC dla każdej klasy
plt.figure(figsize=(10, 6))
for i in range(len(validation_generator.class_indices)):
    plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve (class %d, AUC = %0.2f)' % (i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()


#new prediction
#krzywa uczenia narysować val_accuracy i val_loss
history.history.keys()
plt.figure()
plt.plot(history.history["loss"],label = "Train Loss", color = "black")
plt.plot(history.history["val_loss"],label = "Validation Loss", color = "darkred", linestyle="dashed",markeredgecolor = "purple", markeredgewidth = 2)
plt.title("Model Loss", color = "darkred", size = 13)
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history["accuracy"],label = "Train Accuracy", color = "black")
plt.plot(history.history["val_accuracy"],label = "Validation Accuracy", color = "darkred", linestyle="dashed",markeredgecolor = "purple", markeredgewidth = 2)
plt.title("Model Accuracy", color = "darkred", size = 13)
plt.legend()
plt.show()