from glob import glob
import os
import os.path as osp

import h5py
import numpy as np
import pandas as pd
import nibabel as nib
from deepMRI_keras import init_network


def get_network(n_classes=2, input_shape=(61, 73, 61, 1)):
    return init_network(n_classes=n_classes, input_shape=input_shape)


def atleast_5d(arr):
    if len(arr.shape) != 5:
        arr = arr[..., np.newaxis]
    return arr


def get_func_labels():
    return ['alff', 'degree_weighted', 'eigenvector_weighted', 'falff', 'lfcd']


def load_data(data_folder, h5_filename, metric='entropy', standardize_subjects=False):
    """

    :param data_folder: Folder of the HDF5 file containing rsfMRI summaries
    :param h5_filename: file name of the HDF5 file
    :param metric: entropy
    :return: a tuple of labels, dataset[entropy, autocorr, etc..]
    """

    hFile = h5py.File(osp.join(data_folder, h5_filename), 'r')

    if metric == "all":
        func_metrics = get_func_labels()
        data_all = []
        for func_metric in func_metrics:
            data_loc = osp.join(u'summaries', u'data_{}'.format(func_metric))
            data_all.append(atleast_5d(hFile[data_loc][()]))  # [()] makes it numpy array
        data = np.concatenate(data_all, axis=4)
    else:
        if metric in get_func_labels():
            data_loc = osp.join(u'summaries', u'data_{}'.format(metric))
            data = atleast_5d(hFile[data_loc][()]) # [()] makes it numpy array
        else:
            raise RuntimeError('{} not implemented'.format(metric))

    if standardize_subjects:
        mask = np.array(nib.load(osp.join(data_folder, 'maskAll.nii.gz')).get_data(), dtype=np.bool)
        n_subj = data.shape[0]

        for i_subj in range(n_subj):
            data_subj = data[i_subj]
            mean_subj = data_subj[mask, :].mean(axis=0)
            std_subj = data_subj[mask, :].std(axis=0)
            if np.any(std_subj == 0) or np.any(np.isnan(mean_subj)) or np.any(np.isnan(std_subj)):
                import pdb; pdb.set_trace()
            data[i_subj] = mask[..., np.newaxis] * (data_subj - mean_subj)/std_subj

    if metric == 'structural':
        labels = hFile['summaries'].attrs['struct_labels']
    else:
        labels = hFile['summaries'].attrs['func_labels']

    return labels, data


def get_data(parent_folder, tag, file_name):
    return np.array(sorted(glob(osp.join(parent_folder, tag, file_name))))


def extract_subj_folder(data_path):
    """
    Assumes that the last element in the array data_path is the filename (and the folder is the element before that)
    :param data_path:
    :return:
    """

    folder_names = np.char.asarray(np.char.split(data_path, '/'))[:, -2]
    folder_names = np.char.asarray(np.char.split(folder_names, '_'))
    dx = folder_names[:, 0]
    gender = folder_names[:, 1]
    subj_ids = np.array(folder_names[:, 2], dtype=np.int)
    return dx, subj_ids, gender


def load_subj(subj_file):
    return np.array(nib.load(subj_file).get_data(), dtype=np.float32)


def run(parent_folder, tags, file_name, save_folder, postfix=''):
    X_all, dx_all, ids_all, gender_all = [], [], [], []

    if not osp.exists(save_folder):
        os.makedirs(save_folder)

    for tag in tags:
        print(tag)
        X, dx_tag, subj_ids_tag, gender_tag = prepare_data_tag(file_name, parent_folder, tag)
        X_all.append(X)
        dx_all.append(dx_tag)
        ids_all.append(subj_ids_tag)
        gender_all.append(gender_tag)
    X = np.concatenate(X_all, axis=0)
    dx = np.concatenate(dx_all, axis=0)
    ids_all = np.concatenate(ids_all, axis=0)
    gender_all = np.concatenate(gender_all, axis=0)

    df = pd.DataFrame(data={'subj_ids': ids_all, 'DX_str': dx, 'gender': gender_all})
    df['DX'] = pd.get_dummies(df.DX_str)['ad'].values
    np.save(osp.join(save_folder, 'data_{}.npy'.format(postfix)), X)
    df.to_csv(osp.join(save_folder, 'meta.csv'), index=False)


def prepare_data_tag(file_name, parent_folder, tag):
    data_tag = get_data(parent_folder, tag, file_name)
    X = np.zeros((len(data_tag),) + nib.load(data_tag[0]).shape)
    dx_tag, subj_ids_tag, gender_tag = extract_subj_folder(data_tag)
    for i_subj in xrange(len(data_tag)):
        print('{}/{}'.format(i_subj + 1, len(data_tag)))
        X[i_subj] = load_subj(data_tag[i_subj])
    return X, dx_tag, subj_ids_tag, gender_tag


if __name__ == '__main__':
    parent_folder = '/home/rthomas/Neuro_VUmc/asl_dementie'
    tags = ['ad*', 'smc*']
    file_name = 'asl2std_6mm.nii.gz'
    save_folder = '/home/paulgpu/git/DeepNeurologe'
    postfix = '6mm'
    run(parent_folder, tags, file_name, save_folder, postfix=postfix)
