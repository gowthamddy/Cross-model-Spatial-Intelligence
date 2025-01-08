# Importing necessary libraries
import numpy as np
import glob
import os
import random
import tensorflow as tf
import natsort

base_path = '/users/gowtham/downloads/majorproject'  # Change this to your base path
path_videos = os.path.join(base_path, 'Videos/')
path_labels_csv = os.path.join(base_path, 'labels_framewise_csv.csv')
path_labels_list = os.path.join(base_path, 'labels_framewise_list.pkl')
path_frames = os.path.join(base_path, 'Frames/')
checkpoint_path = os.path.join(base_path, 'checkpoints/approach_3.3/cp.ckpt')


# Perform train-test-validation split (66-22-16)
np.random.seed(42)
x = np.arange(1, 105)
np.random.shuffle(x)

videos_validation = x[:16]
videos_test = x[16: 16 + 22]
videos_train = x[16 + 22:]

print(videos_train, len(videos_train))
print(videos_test, len(videos_test))
print(videos_validation, len(videos_validation))

# Function to load filenames and labels
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

# Load data for train, test, and validation sets
filenames_train, labels_train = load_data(videos_train)
filenames_test, labels_test = load_data(videos_test)
filenames_validation, labels_validation = load_data(videos_validation)

print(filenames_train.shape, filenames_validation.shape, filenames_test.shape)
print(labels_train.shape, labels_validation.shape, labels_test.shape)

# Balancing the dataset
ind0 = np.where(labels_train == 0)[0]
ind1 = np.where(labels_train == 1)[0]
random.shuffle(ind0)
random.shuffle(ind1)

if (ind0.shape[0] / ind1.shape[0] > 1.4):
    print('Reducing the number of unsafe frames in the training dataset\n')
    len_ind0 = int(ind1.shape[0] * 1.4)
    ind0 = ind0[:len_ind0]

    indices_required = np.concatenate((ind0, ind1))
    filenames_train = filenames_train[indices_required]
    labels_train = labels_train[indices_required]

print(filenames_train.shape, labels_train.shape)
print(ind0.shape, ind1.shape)

# %%
# Data preprocessing functions
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

# Create datasets
dataset_train = create_dataset(filenames_train, labels_train, is_training=True)
dataset_test = create_dataset(filenames_test, labels_test)
dataset_val = create_dataset(filenames_validation, labels_validation)

# %%
tf.keras.backend.set_image_data_format('channels_last')

# Model creation function
def create_model():
    inputs = tf.keras.layers.Input([270, 480, 3])

    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=None, dilation_rate=(3, 3 ), use_bias=False)(inputs)
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

# Create and compile the model
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

# Set up model checkpointing
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor='val_recallAtPrecision',
    verbose=1,
    save_best_only=True,
    mode='max'
)

# Train the model
model.fit(
    x=dataset_train,
    validation_data=dataset_val,
    epochs=150,
    verbose=1,
    callbacks=[cp_callback],
    class_weight={0: 1, 1: 1.92}
)

# Evaluate the model on test data
print("Evaluate on test data")
results = model.evaluate(dataset_test)
print("Test loss, test accuracy:", results)

# Evaluate the model on training data
print("Evaluate on train data")
results = model.evaluate(dataset_train)
print("Train loss, train accuracy:", results)

# Load the best weights and evaluate again
model.load_weights(checkpoint_path)
print("Evaluate on test data after loading best weights")
results = model.evaluate(dataset_test)
print("Test loss, test accuracy after loading weights:", results)

print("Evaluate on train data after loading best weights")
results = model.evaluate(dataset_train)
print("Train loss, train accuracy after loading weights:", results)