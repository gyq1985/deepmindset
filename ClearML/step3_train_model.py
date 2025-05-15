import os
os.environ["MPLBACKEND"] = "Agg" 
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from PIL import Image
import seaborn as sns

from clearml import Task, Logger


import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import confusion_matrix, classification_report


warnings.filterwarnings('ignore')

Task.set_credentials(
    web_host='https://app.clear.ml',
    api_host='https://api.clear.ml',
    files_host='https://files.clear.ml',
    key='Q9JQUW7NG3L7UIGUQ99CH6APXN7CWX',
    secret='ZDyUBTNev2TSyADi6gkFMEPRwMUBughNu4uVKUPjH7UsImaBOTLh2B6nVU2CwvYQTcw'
)

task = Task.init(project_name="VGG16-v2", task_name="Pipeline Step 3 - Train Pneumonia Model")
task.set_base_docker("tensorflow/tensorflow:2.18.0-gpu")

logger = Logger.current_logger()


args = {
    'dataset_task_id': '405caed14d034630b33cf083a9fcc28d',
    'img_size': (224, 224),
    'batch_size': 32,
    'epochs_stage1': 20,
    'epochs_stage2': 10,
    'learning_rate_stage1': 0.01,
    'learning_rate_stage2': 1e-5,
    'dropout_rate': 0.2,
    'dense1_units': 198,
    'dense2_units': 128,
    'freeze_until_layer': -4,
    'earlystop_patience': 3,
    'num_classes': 4
}
task.connect(args)
# task.execute_remotely()


dataset_task = Task.get_task(task_id=args['dataset_task_id'])
train_dir = dataset_task.artifacts['train_dir'].get()
val_dir = dataset_task.artifacts['val_dir'].get()
test_dir = dataset_task.artifacts['test_dir'].get()
class_indices = dataset_task.artifacts['class_indices'].get()

IMG_SIZE = args['img_size']
BATCH_SIZE = args['batch_size']

train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)


input_shape = IMG_SIZE + (3,)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

model = Sequential([
    base_model,
    Flatten(),
    Dense(args['dense1_units'], activation='relu'),
    Dense(args['dense2_units'], activation='relu'),
    Dropout(args['dropout_rate']),
    Dense(args['num_classes'], activation='sigmoid')
])


for layer in base_model.layers:
    layer.trainable = False


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args['learning_rate_stage1']),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=args['earlystop_patience'], restore_best_weights=True)
checkpoint_stage1 = ModelCheckpoint("best_model_stage1.keras", monitor='val_accuracy', save_best_only=True)

history_stage1 = model.fit(
    train_generator,
    epochs=args['epochs_stage1'],
    validation_data=val_generator,
    callbacks=[early_stop, checkpoint_stage1]
)


for layer in base_model.layers[args['freeze_until_layer']:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args['learning_rate_stage2']),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint_stage2 = ModelCheckpoint("best_model_stage2.keras", monitor='val_accuracy', save_best_only=True)

history_stage2 = model.fit(
    train_generator,
    epochs=args['epochs_stage2'],
    validation_data=val_generator,
    callbacks=[early_stop, checkpoint_stage2]
)


task.upload_artifact(name="best_model_stage2", artifact_object="best_model_stage2.keras")


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history_stage2.history['accuracy'], label='Train Acc')
plt.plot(history_stage2.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_stage2.history['loss'], label='Train Loss')
plt.plot(history_stage2.history['val_loss'], label='Val Loss')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves_vgg16.png")
logger.report_image("training_curves_vgg16", "Accuracy and Loss", iteration=0, image=Image.open("training_curves_vgg16.png"))


test_generator.reset()
predictions = model.predict(test_generator)
pred_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

cm = confusion_matrix(true_classes, pred_classes)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.savefig("confusion_matrix_vgg16.png")
logger.report_image("confusion_matrix_vgg16", "Confusion Matrix", iteration=0, image=Image.open("confusion_matrix_vgg16.png"))

report = classification_report(true_classes, pred_classes, target_names=class_labels)
with open("classification_report_vgg16.txt", "w") as f:
    f.write(report)
task.upload_artifact("classification_report_vgg16", artifact_object="classification_report_vgg16.txt")