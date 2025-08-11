import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

# Konfiguracja generatorów danych
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'C:...train',
    target_size=(224, 224),  # Wymagany rozmiar wejściowy dla ResNet50
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    'C:...validation',
    target_size=(224, 224),  # Wymagany rozmiar wejściowy dla ResNet50
    batch_size=32,
    class_mode='categorical'
)

# Załaduj model ResNet50 (bez warstw Dense na szczycie)
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Zamroź wagi modelu ResNet50
for layer in resnet_model.layers:
    layer.trainable = False

# Budowa nowego modelu na bazie ResNet50
model = Sequential([
    resnet_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(5, activation='softmax')
])

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trenowanie modelu
model.fit(
    train_generator,
    epochs=10,
    steps_per_epoch=None,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Ostateczna ocena modelu na zestawie testowym (opcjonalne)
test_loss, test_acc = model.evaluate(validation_generator)
print('Dokładność testu:', test_acc)

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

# Oblicz krzywe ROC dla każdej klasy
y_pred_prob = model.predict(validation_generator)
lb = LabelBinarizer()
lb.fit(validation_generator.classes)
y_test_bin = lb.transform(validation_generator.classes)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(validation_generator.class_indices)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

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
