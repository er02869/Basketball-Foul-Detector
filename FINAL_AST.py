#all necessary libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
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

#configure GPU acceleration when TensorFlow is running in a GPU-capable environment
physical_gpus = tf.config.list_physical_devices('GPU')
for gpu in physical_gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as error:
        print(f"Could not configure GPU memory growth: {error}")

if physical_gpus:
    tf.config.set_soft_device_placement(True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    COMPUTE_DEVICE = '/GPU:0'
    strategy = tf.distribute.OneDeviceStrategy(COMPUTE_DEVICE)
    logical_gpus = tf.config.list_logical_devices('GPU')
    build_info = tf.sysconfig.get_build_info()
    gpu_build = 'ROCm' if build_info.get('is_rocm_build') else 'CUDA' if build_info.get('is_cuda_build') else 'unknown'
    with tf.device(COMPUTE_DEVICE):
        tf.linalg.matmul(tf.ones((2, 2)), tf.ones((2, 2)))
    print(
        f"GPU acceleration enabled: {len(physical_gpus)} physical / "
        f"{len(logical_gpus)} logical GPU(s) on {COMPUTE_DEVICE} "
        f"({gpu_build} TensorFlow build)"
    )
else:
    COMPUTE_DEVICE = '/CPU:0'
    strategy = tf.distribute.OneDeviceStrategy(COMPUTE_DEVICE)
    print(
        "GPU not detected"
    )

#Set paths, batch, and video sizes
PROJECT_DIR = Path(__file__).resolve().parent / "Fouls"
DATA_DIR = PROJECT_DIR / "IMAGE_DATASET"
VIDEO_DIR = PROJECT_DIR / "VIDEO_DATASET"
VIDEO_FRAME_DIR = PROJECT_DIR / "VIDEO_FRAMES"
VIDEO_DEFAULT_CLASS = "FOULS"
BATCH_SIZE = 16
IMG_SIZE = (224, 224) 

#train or test a previous model
parser = argparse.ArgumentParser(description="Train or test the basketball foul classifier.")
parser.add_argument(
    "--use-existing-model",
    action="store_true",
    help="Load a saved Keras model and skip training."
)
parser.add_argument(
    "--model",
    type=Path,
    help="Path to a saved .keras model."
)
parser.add_argument(
    "--test",
    type=Path,
    help="Image or video to classify when using an existing model."
)
args = parser.parse_args()


def read_video_frames(video_path, maximum_frames=32):
    """Read sample frames without depending on unreliable video metadata."""
    import cv2

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        video.release()
        raise ValueError(f"OpenCV could not open the video: {video_path}")

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count > 0:
        frame_indexes = np.linspace(
            0, frame_count - 1, min(frame_count, maximum_frames), dtype=int
        )
        frames = []
        for frame_index in frame_indexes:
            video.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = video.read()
            if success:
                frames.append(frame)
    else:
        frames = []
        while len(frames) < maximum_frames:
            success, frame = video.read()
            if not success:
                break
            frames.append(frame)
    video.release()

    if not frames:
        raise ValueError(
            f"The video has no readable frames: {video_path}. "
            "Check that the file is valid and supported by OpenCV/FFmpeg."
        )
    return frames


def test_saved_model(model_path, input_path):
    """Load a saved model and classify one image or video without retraining."""
    import cv2

    if not model_path.is_file():
        raise FileNotFoundError(f"Saved model was not found: {model_path}")
    if not input_path.is_file():
        raise FileNotFoundError(f"Test input was not found: {input_path}")

    class_names = sorted(
        path.name for path in DATA_DIR.iterdir() if path.is_dir()
    )
    if len(class_names) != 2:
        raise ValueError(f"Expected two class folders in {DATA_DIR}, found {class_names}")

    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"Loaded saved model: {model_path}")
    print(f"Testing input: {input_path}")

    video_extensions = {'.avi', '.m4v', '.mkv', '.mov', '.mp4', '.mpeg', '.wmv'}
    if input_path.suffix.lower() in video_extensions:
        frames = read_video_frames(input_path)
        scores = []
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = tf.image.resize(frame, IMG_SIZE)
            frame_array = tf.expand_dims(tf.cast(frame, tf.float32), 0)
            scores.append(float(model.predict(frame_array, verbose=0)[0][0]))
        score = float(np.mean(scores))
        print(f"Decoded {len(frames)} video frame(s)")
    else:
        image = tf.keras.utils.load_img(input_path, target_size=IMG_SIZE)
        image_array = tf.expand_dims(tf.keras.utils.img_to_array(image), 0)
        score = float(model.predict(image_array, verbose=0)[0][0])

    label = class_names[1] if score >= 0.5 else class_names[0]
    confidence = score * 100 if score >= 0.5 else (1 - score) * 100
    print(f"Result: {label} (Confidence: {confidence:.2f}%)")


if args.use_existing_model:
    model_path = args.model
    if model_path is None:
        model_candidates = [
            PROJECT_DIR.parent / "BEST-RUNS" / "basketball_foul_model.keras",
            PROJECT_DIR.parent / "BEST-RUNS" / "basketball_foul_model(1).keras",
            PROJECT_DIR / "basketball_foul_model.keras",
        ]
        model_path = next((path for path in model_candidates if path.is_file()), None)
        if model_path is None:
            raise FileNotFoundError("No saved .keras model was found in BEST-RUNS or Fouls")
    if args.test is None:
        raise ValueError("--test is required with --use-existing-model")
    test_saved_model(model_path.resolve(), args.test.resolve())
    raise SystemExit(0)

#extract labeled video frames so they can be trained with the image dataset
def extract_video_frames(video_dir, frame_dir, class_names, default_class):
    import cv2
    #available formats for videos
    video_extensions = {'.avi', '.m4v', '.mkv', '.mov', '.mp4', '.mpeg', '.wmv'}
    for class_name in class_names:
        (frame_dir / class_name).mkdir(parents=True, exist_ok=True)
    video_paths = [path for path in video_dir.rglob('*') if path.suffix.lower() in video_extensions]
    
    #go through the paths including the videos, and ensure that they are properly formatted and able to be prcessed
    for video_path in video_paths:
        parent_class = video_path.parent.name
        class_name = parent_class if parent_class in class_names else default_class
        if class_name not in class_names:
            raise ValueError(f"Video label '{class_name}' is not one of {class_names}")

        #retreive the video, then cut it into frames so that it can be trained into the AI model if need be
        video = cv2.VideoCapture(str(video_path))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            video.release()
            continue

        #retreive the framed out video and then sort it into the VIDEO_FRAMES folder
        output_dir = frame_dir / class_name
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_indexes = np.linspace(0, frame_count - 1, min(frame_count, 32), dtype=int)
        for frame_index in frame_indexes:
            video.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = video.read()
            if not success:
                continue

            frame_path = output_dir / f"{video_path.stem}_{frame_index}.jpg"
            cv2.imwrite(str(frame_path), frame)
        video.release()

#load datatsets from the two CLEAN and BLOCK folder and split them into training/validation then turn them into classes
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

#add representative video frames to the training data using the same class names
extract_video_frames(VIDEO_DIR, VIDEO_FRAME_DIR, class_names, VIDEO_DEFAULT_CLASS)
video_dataset = tf.keras.utils.image_dataset_from_directory(
    VIDEO_FRAME_DIR,
    class_names=class_names,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=123
)
train_dataset = train_dataset.concatenate(video_dataset)

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

#data Augmentation to help with the lack of data leading to issues such as overfitting 
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

#load pre-trained ResNet50 model and compile it on the selected device
with strategy.scope():
    base_model = tf.keras.applications.ResNet50(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

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
print("TRAINING MODEL")
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
print("\nUNFREEZE TOP LAYERS OF RESNET50 FOR MORE TRAINING")
base_model.trainable = True

#ResNet50 has 175 layers freezing the first 143 means only the last residual block && custom head get updated
fine_tune_at = 143
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

#recompile with learning rate (1e-5) due to interacting with frozen pre-trained weights
#using same loss and metric so the history can be combined with the initial run
with strategy.scope():
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
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

#collect every prediction score and true label from the test set so we can build the metric plots below
y_true = []
y_scores = []
for images, labels in test_dataset:
    preds = model.predict(images, verbose=0).flatten()
    y_scores.extend(preds)
    y_true.extend(labels.numpy())
y_true = np.array(y_true)
y_scores = np.array(y_scores)

#use the same 0.5 threshold that predict_play uses so the confusion matrix matches real predictions
y_pred = (y_scores >= 0.5).astype(int)

#combine the initial training history with the fine tuning history so we can plot both phases on one chart
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss_hist = history.history['loss'] + history_fine.history['loss']
val_loss_hist = history.history['val_loss'] + history_fine.history['val_loss']


#PLOTTING ALL THE METRICS USING THE TEST SET CREATED
"""
#Confusion Matrix => shows where the model got it right vs where it mixed up fouls and blocks
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

#write the count number inside each cell so the matrix is readable
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('c_matrix.png')
plt.show()

#ROC Curve => tradeoff between true positive rate and false positive rate at every threshold => AUC of 1.0 is perfect <->0.5 is guessing
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.show()

#print all the test set metrics together
print("\nTEST SET METRICS")
print(f"  ROC AUC:                    {roc_auc:.4f}")
print(f"  Confusion Matrix:")
print(f"    True {class_names[0]}   {cm[0][0]}")
print(f"    False {class_names[0]}                  {cm[0][1]}")
print(f"    False {class_names[1]}                  {cm[1][0]}")
print(f"    True {class_names[1]}  {cm[1][1]}")

#save the model for future use (maybe0)
model.save(MODEL_PATH)
"""


#function that determines the playtype using a single image as the input
def predict_play(image_path):
    print(f"\nAnalyzing input: {image_path}")
    try:
        #load the image or sample video frames and predict them using our model
        video_extensions = {'.avi', '.m4v', '.mkv', '.mov', '.mp4', '.mpeg', '.wmv'}
        if Path(image_path).suffix.lower() in video_extensions:
            import cv2

            frame_scores = []
            for frame in read_video_frames(Path(image_path)):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = tf.image.resize(frame, IMG_SIZE)
                frame_array = tf.expand_dims(tf.cast(frame, tf.float32), 0)
                frame_scores.append(model.predict(frame_array, verbose=0)[0][0])

            if not frame_scores:
                raise ValueError("The video frames could not be decoded")
            score = float(np.mean(frame_scores))
        else:
            #load the image and predict it using our model
            img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            score = float(model.predict(img_array, verbose=0)[0][0])

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

#testing video output
video_testing = VIDEO_DIR / "FOULS" / "FOUL2.mp4"
predict_play(video_testing)
