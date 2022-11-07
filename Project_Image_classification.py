from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg19
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.models import Model
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.datasets import load_files
from keras.utils import np_utils, load_img, img_to_array
import numpy as np
from keras.applications.vgg19 import preprocess_input
from tqdm import tqdm


def data_batches():
    train_path = r'F:\Career\Work\CNN_Operations\Dataset\dogs_vs_cats_project\data\train'
    valid_path = r'F:\Career\Work\CNN_Operations\Dataset\dogs_vs_cats_project\data\valid'
    test_path = r'F:\Career\Work\CNN_Operations\Dataset\dogs_vs_cats_project\data\test'

    train_batches = ImageDataGenerator().flow_from_directory(train_path,
                                                             target_size=(224, 224),
                                                             batch_size=10)
    valid_batches = ImageDataGenerator().flow_from_directory(valid_path,
                                                             target_size=(224, 224),
                                                             batch_size=30)
    test_batches = ImageDataGenerator().flow_from_directory(test_path,
                                                            target_size=(224, 224),
                                                            shuffle=True)
    return train_batches, valid_batches, test_batches


def base_transfer_model():
    base_model = vgg19.VGG19(weights='imagenet', include_top=False,
                             input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False
    last_layer = base_model.get_layer('block5_pool')
    last_output = last_layer.output
    return last_output, base_model


def create_model():
    last_output, base_model = base_transfer_model()
    x = Flatten()(last_output)
    x = Dense(64, activation='relu', name='FC_2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax', name='softmax')(x)
    new_model = Model(inputs=base_model.input, outputs=x)
    return new_model


def define_model():
    new_model = create_model()
    train_batches, valid_batches, test_batches = data_batches()
    new_model.compile(Adam(learning_rate=0.001),
                      loss='categorical_crossentropy', metrics=['accuracy'])
    new_model.fit_generator(train_batches,
                            steps_per_epoch=4,
                            validation_data=valid_batches, epochs=20, verbose=2)
    return new_model


def load_dataset(path):
    data = load_files(path)
    paths = np.array(data['filenames'])
    targets = np_utils.to_categorical(np.array(data['target']))
    return paths, targets


def path_to_tensor(img_path):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img=img)
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def predcitions():
    new_model = define_model()
    path = r'F:\Career\Work\CNN_Operations\Dataset\dogs_vs_cats_project\data\test'
    test_files, test_targets = load_dataset(path)
    test_tensors = preprocess_input(paths_to_tensor(test_files))
    print('Testing loss: {:.4f}Testing accuracy:{: .4f}'.format(*new_model.evaluate(test_tensors, test_targets)))


predcitions()
