import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import numpy as np
import tensorflow as tf
from collections import Counter
from pathlib import Path
assert sys.version_info >= (3, 7)
from packaging import version
assert version.parse(tf.__version__) >= version.parse("2.8.0")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

#Set paths, batch, and imahe size
DATA_DIR = Path(r"d:\MACHINE_LEARNING\FINAL_ASSN\DATASET")
BATCH_SIZE = 16
IMG_SIZE = (224, 224) 

#load datatsets from the two CLEAN and BLOCK folder, and split them into training/validation, then turn them into classes
print("Loading Data")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
class_names = train_dataset.class_names

#datasets are autoned to increase preformance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

#carve out a held out TEST set from the validation set for final generalization metrics
#take half of validation as test, leave the other half as validation, this gets us roughly 80/10/10 train/val/test
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 2)
validation_dataset = validation_dataset.skip(val_batches // 2)
print(f"Validation batches: {int(val_batches) - int(val_batches // 2)}, Test batches: {int(val_batches // 2)}")

#Data Augmentation to help with the lack of data leading to issues such as overfitting 
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=42),
    tf.keras.layers.RandomRotation(0.3, seed=42), #Increased from 0.2
    tf.keras.layers.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), seed=42), #Wider zoom range
    tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2, seed=42), #Increased from 0.1
    tf.keras.layers.RandomContrast(factor=0.3, seed=42), #contrast variation
    tf.keras.layers.RandomBrightness(factor=0.2, value_range=(0.0, 255.0), seed=42), #brightness variation
])

#this is the model path for the created model, may or may not used the saved version for each run as there are issues
MODEL_PATH = "basketball_foul_model.keras"
#-----------------------------------------------------------------------------------------------------------------------
#COMMENTED OUT: Loading existing model                                                                                 |
# if os.path.exists(MODEL_PATH):                                                                                       |
#     print(f"\nFound existing model at '{MODEL_PATH}'.")                                                              |
#     model = tf.keras.models.load_model(MODEL_PATH)                                                                   |
# else:                                                                                                                |
#     print("\nNo saved model found.")                                                                                 |
#-----------------------------------------------------------------------------------------------------------------------

#load pre-trained ResNet50 model 
#we dont include the top layer because we are replacing it with our own 1-node output
base_model = tf.keras.applications.ResNet50(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
#freeze base model layers to stop them from being updated during train
base_model.trainable = False 

#build the new Model using Keras Functional API
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))

#use data augmentation only during training and apply RESNET50  to ensure the input is in the correct format for the pre-trained model
#pass through the base model to extract features then add a custom head for our specific classification task 
x = data_augmentation(inputs)
x = tf.keras.applications.resnet50.preprocess_input(x)
x = base_model(x, training=False)

#squish the 2D output down to 1D
x = tf.keras.layers.GlobalAveragePooling2D()(x)

#Dense Head using DROPOUT to prevent overfitting and a final sigmoid activation for classification
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x) 
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x) 
model = tf.keras.Model(inputs, outputs)

#compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

#Model summary before the fine tuning (for presnetaions)
#model.summary()

#Learning Rate Reduction = (lr) and Early Stopping Callbacks = (early_stopping) 
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    
    #keep best preforming weights
    restore_best_weights=True 
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=2, 
    min_lr=1e-6
)

#train Model
print("TRAINING MODEL NOW!")
epochs = 18 
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[early_stopping, reduce_lr]
)

#evaluate the mode using validationloss and validationaccuracy as the metrics
print("\nEvaluating model on validation data")
loss, accuracy = model.evaluate(validation_dataset)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

#Fine-Tuning for Model => unfreeze the top portion of ResNet50 so the pre trained features can adapt to basketball images
print("\nFINE TUNING PHASE")
base_model.trainable = True

#ResNet50 has 175 layers, freezing the first 143 means only the last residual block && our custom head get updated
fine_tune_at = 143
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

#recompile with learning rate (1e-5) due to interacting with frozen pre-trained weights
#using same loss and metric so the history can be combined with the initial run
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

#continue training initial_epoch keeps epoch numbering consistent so we can plot both phases on a single chart
fine_tune_epochs = 10
total_epochs = len(history.epoch) + fine_tune_epochs
history_fine = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=total_epochs,
    initial_epoch=len(history.epoch),
    callbacks=[early_stopping, reduce_lr]
)

#evaluate the fine tuned model on validation again to compare against the initial frozen run
print("\nEvaluating fine tuned model on validation data")
loss, accuracy = model.evaluate(validation_dataset)
print(f"Validation Loss (fine tuned): {loss:.4f}")
print(f"Validation Accuracy (fine tuned): {accuracy:.4f}")

#Test-Set validation for model accuracy validation
print("\nEvaluating fine tuned model on TEST SET (held out, never seen during training)")
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

#collect every prediction score and true label from the test set so we can build the metric plots below
y_true = []
y_scores = []
for images, labels in test_dataset:
    preds = model.predict(images, verbose=0).flatten()
    y_scores.extend(preds)
    y_true.extend(labels.numpy())
y_true = np.array(y_true)
y_scores = np.array(y_scores)

#use the same 0.6 threshold that predict_play uses so the confusion matrix matches real predictions
y_pred = (y_scores >= 0.6).astype(int)

#combine the initial training history with the fine tuning history so we can plot both phases on one chart
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss_hist = history.history['loss'] + history_fine.history['loss']
val_loss_hist = history.history['val_loss'] + history_fine.history['val_loss']


#PLOTTING ALL THE METRICS
#accuracy through each epoch
#the red dashed line marks where fine tuning started
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.axvline(x=len(history.epoch) - 1, color='r', linestyle='--', label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss_hist, label='Training Loss')
plt.plot(val_loss_hist, label='Validation Loss')
plt.axvline(x=len(history.epoch) - 1, color='r', linestyle='--', label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

#Confusion Matrix => shows where the model got it right vs where it mixed up fouls and blocks
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Test Set)')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

#write the count number inside each cell so the matrix is readable at a glance
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

#ROC Curve => tradeoff between true positive rate and false positive rate at every threshold => AUC of 1.0 is perfect <->0.5 is guessing
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Test Set)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.show()

#Precision Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_scores)
avg_precision = average_precision_score(y_true, y_scores)
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve (Test Set)')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('pr_curve.png')
plt.show()

#print all the test set metrics together so they are easy to copy into the report
print("\nTEST SET METRICS SUMMARY")
print(f"  Test Accuracy:              {test_accuracy:.4f}")
print(f"  Test Loss:                  {test_loss:.4f}")
print(f"  ROC AUC:                    {roc_auc:.4f}")
print(f"  Average Precision (PR AUC): {avg_precision:.4f}")
print(f"  Confusion Matrix:")
print(f"    True {class_names[0]} (correctly classified):  {cm[0][0]}")
print(f"    False Positives:                  {cm[0][1]}")
print(f"    False Negatives:                  {cm[1][0]}")
print(f"    True {class_names[1]} (correctly classified):   {cm[1][1]}")

#save the model for future use (maybe0)
model.save(MODEL_PATH)

#function that determines the playtype using a single image as the input
def predict_play(image_path):
    print(f"\nAnalyzing image: {image_path}...")
    try:
        #load the images and predict them using our model
        img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) 
        predictions = model.predict(img_array, verbose=0)
        score = predictions[0][0]

        #score > 0.5 is a foul or it is a block (SIGMOID FUNCTION)
        #confidence is the confidence of a model in its prediciton, SCORES WILL BE CHANGED BASED OFF MODEL ACCURACY
        if score >= 0.5:
            label = class_names[1] 
            confidence = score * 100
        else:
            label = class_names[0]  
            confidence = (1 - score) * 100

        print(f"Result: {label} (Confidence: {confidence:.2f}%)")
    except Exception as e:
        print(f"Error analyzing image: {e}")

#test a new image
sample_image_to_test = r"d:\MACHINE_LEARNING\FINAL_ASSN\INPUT\test_f.jpg"
predict_play(sample_image_to_test)