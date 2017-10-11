from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import os
import os.path as osp
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, CSVLogger, TensorBoard

from data_handling import load_data, get_network


def run(data_folder, data_file, metric, save_folder, batch_size=20):
    if not osp.exists(save_folder):
        os.makedirs(save_folder)

    y, X = load_data(data_folder, data_file, metric=metric)
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    metrics_test = np.zeros((cv.get_n_splits(X, y), 3))
    print(save_folder)
    print('Batch size: {}'.format(batch_size))

    for (i_cv, (train_id, test_id)) in enumerate(cv.split(X, y)):
        print('{}/{}'.format(i_cv + 1, cv.get_n_splits(X, y)))
        X_train, X_test = X[train_id], X[test_id]
        y_train, y_test = y[train_id], y[test_id]
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=30, stratify=y_train)

        print('Train: Class 1: {}; Class 0: {}'.format(np.sum(y_train == 1), np.sum(y_train == 0)))
        print('Valid: Class 1: {}; Class 0: {}'.format(np.sum(y_valid == 1), np.sum(y_valid == 0)))
        print('Test: Class 1: {}; Class 0: {}'.format(np.sum(y_test == 1), np.sum(y_test == 0)))

        input_shape = X_train.shape[1:]
        network = get_network(n_classes=np.unique(y).size, input_shape=input_shape)

        mean_train = X_train.mean(axis=0, keepdims=True)
        std_train = X_train.std(axis=0, keepdims=True)
        X_train = (X_train - mean_train)/(std_train + 0.0001)
        X_valid = (X_valid - mean_train)/(std_train + 0.0001)
        X_test = (X_test - mean_train)/(std_train + 0.0001)

        y_train = to_categorical(y_train, num_classes=2)
        y_valid = to_categorical(y_valid, num_classes=2)
        y_test = to_categorical(y_test, num_classes=2)

        early_stopping = EarlyStopping(monitor='val_balanced_accuracy', min_delta=0.001, patience=50, verbose=1,
                                       mode='max')
        csv_logger = CSVLogger(osp.join(save_folder, 'training{}.log'.format(i_cv + 1)))
        tensorboard = TensorBoard(log_dir=osp.join(save_folder, 'tensorboard{}'.format(i_cv + 1)), histogram_freq=0,
                                  write_graph=True, write_images=True)


        network.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=1000, verbose=1,
                    validation_data=(X_valid, y_valid), shuffle=True,
                    callbacks=[early_stopping, csv_logger, tensorboard])

        loss, acc, bal_acc = network.evaluate(x=X_test, y=y_test, batch_size=y_test.size, verbose=1)
        print('Test: ')
        print(loss, acc, bal_acc)
        metrics_test[i_cv] = [loss, acc, bal_acc]
    print('Avg. test metrics:')
    print(metrics_test.mean(axis=0))

    network = run_final_model(X, y, batch_size)

    network.save(osp.join(save_folder, 'final_weights.h5'))
    np.save(osp.join(save_folder, 'metrics_test_cv.npy'), metrics_test)


def run_final_model(X, y, batch_size):
    network = get_network(n_classes=np.unique(y).size)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=30, stratify=y)

    y_train = to_categorical(y_train, num_classes=2)
    y_valid = to_categorical(y_valid, num_classes=2)

    mean_train = X_train.mean(axis=0, keepdims=True)
    std_train = X_train.std(axis=0, keepdims=True)
    X_train = (X_train - mean_train) / (std_train + 0.0001)
    X_valid = (X_valid - mean_train) / (std_train + 0.0001)

    early_stopping = EarlyStopping(monitor='val_balanced_accuracy', min_delta=0.001, patience=50, verbose=1,
                                   mode='max')
    network.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=1000, verbose=1,
                validation_data=(X_valid, y_valid), shuffle=True,
                callbacks=[early_stopping])
    return network


if __name__ == '__main__':
    data_dir = '/data/local/deeplearning/DeepPsychNet/abide_I_data/hdf5_data'
    data_file = 'fmri_summary.hdf5'
    metric = 'entropy'

    save_dir = osp.join(data_dir, '5layers_10cv_dropout_all_fc')
    run(data_folder=data_dir, data_file=data_file, save_folder=save_dir, metric=metric)
