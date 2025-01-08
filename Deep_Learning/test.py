import numpy as np
import glob
import os
import random
import tensorflow as tf
import natsort
import cv2

base_path = '/users/gowtham/downloads/majorproject'
path_videos = os.path.join(base_path, 'Videos/')
path_labels_csv = os.path.join(base_path, 'labels_framewise_csv.csv')
path_labels_list = os.path.join(base_path, 'labels_framewise_list.pkl')
path_frames = os.path.join(base_path, 'Frames/')
checkpoint_path = os.path.join(base_path, 'checkpoints/training_deploy/cp.ckpt')


np.random.seed(42)
x = np.arange(1, 105)
np.random.shuffle(x)

videos_validation = x[:16]
videos_test = x[16: 16 + 22]
videos_train = x[16 + 22:]

print(videos_train, len(videos_train))
print(videos_test, len(videos_test))
print(videos_validation, len(videos_validation))

def load_data(videos):
    filenames = []
    labels = []
    for vid in videos:
        folder = os.path.join(path_frames, f"video{vid}/")
        frames = natsort.natsorted(glob.glob(os.path.join(folder, 'frame*.jpg')))
        filenames.extend(frames)
        
        labels_path = os.path.join(folder, f"labels{vid}.npy")
        labels_array = np.load(labels_path)
        labels.extend(labels_array.tolist())
    
    return np.array(filenames), np.array(labels)

filenames_train, labels_train = load_data(videos_train)
filenames_test, labels_test = load_data(videos_test)
filenames_validation, labels_validation = load_data(videos_validation)

print(filenames_train.shape, filenames_validation.shape, filenames_test.shape)
print(labels_train.shape, labels_validation.shape, labels_test.shape)

def parse_function(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [270, 480], method=tf.image.ResizeMethod.AREA, preserve_aspect_ratio=True)
    return image, label

def train_preprocess(image, label):
    image = tf.image.random_brightness(image, 0.15)
    image = tf.image.random_contrast(image, 0.8, 1.5)
    image = tf.image.random_saturation(image, 0.6, 3)
    return image, label

# Create TensorFlow datasets
def create_dataset(filenames, labels, is_training=False):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    if is_training:
        dataset = dataset.shuffle(len(filenames))
        dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(train_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(16)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


dataset_train = create_dataset(filenames_train, labels_train, is_training=True)
dataset_test = create_dataset(filenames_test, labels_test)
dataset_val = create_dataset(filenames_validation, labels_validation)


tf.keras.backend.set_image_data_format('channels_last')

def create_model():
    inputs = tf.keras.layers.Input([270, 480, 3])

    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=None, dilation_rate=(3, 3), use_bias=False)(inputs)
    x = tf.keras.layers.ReLU(6.)(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=None, dilation_rate=(3, 3), use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=None, dilation_rate=(2, 2), use_bias=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=None, dilation_rate=(2, 2), use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=None, dilation_rate=(2, 2), use_bias=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation=None, dilation_rate=(2, 2), use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation=None, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(64, activation=None, kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x = tf.keras.layers.ReLU(6.)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(32, activation=None, kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x = tf.keras.layers.ReLU(6.)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

model = create_model()
model.summary()

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001 / 5),
    metrics=[
        tf.keras.metrics.RecallAtPrecision(precision=0.9, name='recallAtPrecision'),
        tf.keras.metrics.BinaryAccuracy(threshold=0.6, name='binaryAccuracy')
    ]
)

model.load_weights(checkpoint_path)
print("Loaded weights")

print("Evaluate on test data")
results = model.evaluate(dataset_test)
print("Test loss, test accuracy:", results)

print("Evaluate on train data")
results = model.evaluate(dataset_train)
print("Train loss, train accuracy:", results)

print("Evaluate on validation data")
results = model.evaluate(dataset_val)
print("Validation loss, validation accuracy:", results)

img = cv2.imread("/users/gowtham/downloads/majorproject/Data/Frames/video33/frame60.jpg")
frame_input = tf.image.convert_image_dtype(img, tf.float32)
frame_input = tf.image.resize(frame_input, [270, 480], method=tf.image.ResizeMethod.AREA, preserve_aspect_ratio=True)
frame_input = np.expand_dims(frame_input, axis=0)
print("Input shape: ", frame_input.shape)
print("Prediction: ", model.predict(frame_input))


model.save('/users/gowtham/downloads/majorproject/savedmodels/approach_3')
print("Saved model")

loaded_model = tf.keras.models.load_model('/users/gowtham/downloads/majorproject/savedmodels/approach_3')
print("Prediction from loaded model: ", loaded_model.predict(frame_input))
