import numpy as np
import tensorflow as tf
import glob
import os
import natsort
from tensorflow import MobileNetV2
from tensorflow import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow import Model
from tensorflow import Adam
from tensorflow import ModelCheckpoint

BATCH_SIZE = 16
EPOCHS_INITIAL = 30
EPOCHS_FINE_TUNE = 120
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2
INPUT_SHAPE = (270, 480, 3)
NUM_CLASSES = 1
TRAINABLE_LAYERS = False
INITIAL_LEARNING_RATE = 1e-5
FINE_TUNE_LEARNING_RATE = 1e-5
CHECKPOINT_PATH = "//users/gowtham/downloads/majorproject/checkpoints/approach_3.1/cp.ckpt"

PATH_VIDEOS = '/users/gowtham/downloads/majorproject/Data/Videos/'
PATH_LABELS_CSV = '/users/gowtham/downloads/majorproject/Data/labels_framewise_csv.csv'
PATH_LABELS_LIST = '/users/gowtham/downloads/majorproject/Data/labels_framewise_list.pkl'
PATH_FRAMES = '/users/gowtham/downloads/majorproject/Data/Frames/'

def load_data(video_ids, path_frames, path_labels):
    filenames = []
    labels = []
    for vid in video_ids:
        folder = os.path.join(path_frames, f"video{vid}/")
        frames = glob.glob(os.path.join(folder, 'frame*.jpg'))
        frames = natsort.natsorted(frames)
        filenames.extend(frames)
        labels_path = os.path.join(path_frames, f"video{vid}/", f"labels{vid}.npy")
        labels_array = np.load(labels_path)
        labels.extend(labels_array)
    return np.array(filenames), np.array(labels)

video_ids = np.arange(1, 105)
np.random.seed(42)
np.random.shuffle(video_ids)
videos_validation = video_ids[:16]
videos_test = video_ids[16: 16+22]
videos_train = video_ids[16+22: ]

train_filenames, train_labels = load_data(videos_train, PATH_FRAMES, PATH_LABELS_CSV)
val_filenames, val_labels = load_data(videos_validation, PATH_FRAMES, PATH_LABELS_CSV)
test_filenames, test_labels = load_data(videos_test, PATH_FRAMES, PATH_LABELS_CSV)

def parse_function(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_image(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, INPUT_SHAPE[:2], method=tf.image.ResizeMethod.AREA, 
                            preserve_aspect_ratio=True)
    return image, label

def train_preprocess(image, label):
    image = tf.image.stateless_random_brightness(image, (0.15,), 42)
    image = tf.image.stateless_random_contrast(image, (0.8, 1.5), 42)
    image = tf.image.stateless_random_saturation(image, (0.6, 3), 42)
    return image, label

train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
train_dataset = train_dataset.shuffle(len(train_filenames))
train_dataset = train_dataset.map(parse_function, num_parallel_calls=4)
train_dataset = train_dataset.map(train_preprocess, num_parallel_calls=4)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(1)

val_dataset = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
val_dataset = val_dataset.shuffle(len(val_filenames))
val_dataset = val_dataset.map(parse_function, num_parallel_calls=4)
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.prefetch(1)

test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
test_dataset = test_dataset.shuffle(len(test_filenames))
test_dataset = test_dataset.map(parse_function, num_parallel_calls=4)
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.prefetch(1)

def create_model(input_shape, num_classes):
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False, 
        weights='imagenet', 
        pooling=None,
    )
    base_model.trainable = False
    inputs = Input(input_shape, name="input_layer")
    inputs_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    x = base_model(inputs_preprocessed, training=False)
    x = GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = Dropout(0.4, name="dropout_layer")(x)

    if num_classes == 1:
        activation = "sigmoid"
    else:
        activation = "softmax"

    outputs = Dense(num_classes, activation=activation, name="output_layer")(x)
    model = Model(inputs, outputs, name="mobilenet_v2_classifier")
    return model


def compile_model(model, learning_rate):
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=Adam(learning_rate),
        metrics=[tf.keras.metrics.RecallAtPrecision(precision=0.9, name='recallAtPrecision'), 
                 tf.keras.metrics.BinaryAccuracy(threshold=0.6, name='binaryAccuracy')]
    )
    return model

model = create_model(INPUT_SHAPE, NUM_CLASSES)
model = compile_model(model, INITIAL_LEARNING_RATE)

history_initial = model.fit(
    train_dataset, 
    validation_data=val_dataset, 
    epochs=EPOCHS_INITIAL, 
    verbose=1, 
    class_weight={0: 1, 1: 1.92}
)

for layer in model.layers:
    layer.trainable = True

model = compile_model(model, FINE_TUNE_LEARNING_RATE)

cp_callback = ModelCheckpoint(
    filepath=CHECKPOINT_PATH, 
    save_weights_only=True, 
    monitor='val_recallAtPrecision', 
    verbose=1, 
    save_best_only=True, 
    mode='max'
)

history_fine_tune = model.fit(
    train_dataset, 
    validation_data=val_dataset, 
    epochs=EPOCHS_FINE_TUNE, 
    verbose=1, 
    callbacks=[cp_callback], 
    class_weight={0: 1, 1: 1.92}
)

model.load_weights(CHECKPOINT_PATH)
test_loss, test_recall, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.3f}, Test Recall: {test_recall:.3f}, Test Accuracy: {test_accuracy:.3f}")

train_loss, train_recall, train_accuracy = model.evaluate(train_dataset)
print(f"Train Loss: {train_loss:.3f}, Train Recall: {train_recall:.3f}, Train Accuracy: {train_accuracy:.3f}")
