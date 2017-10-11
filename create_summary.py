import os
from glob import glob
from time import time
from sys import stdout
import pandas as pd
import nibabel as nib
import numpy as np
import h5py

from scipy.special import xlogy


def subject_dict(data_dir, abide_fmri_path, abide_mask_path, pheno_file):
    """

    :param data_dir: path to main directory
    :param abide_fmri_path: path to abide fMRI specific directory
    :param abide_mask_path: path to abide functional mask specific directory
    :param pheno_file: file with the phenotypical variable

    :return: dictionary with subject_ids as keys and [file_path_fmri, file_path_mask, diagnosis] as value
    """
    all_fmri_files  = sorted(glob(os.path.join(data_dir, abide_fmri_path, '*.nii.gz')))
    all_mask_files  = sorted(glob(os.path.join(data_dir, abide_mask_path, '*.nii.gz')))

    pheno_data = pd.read_csv(os.path.join(data_dir, pheno_file))

    # Autism = 1, Control = 2
    autistic_ids = pheno_data[pheno_data['DX_GROUP'] == 1]['SUB_ID']
    control_ids = pheno_data[pheno_data['DX_GROUP'] == 2]['SUB_ID']

    subject_vars = dict()

    for f, m in zip(all_fmri_files, all_mask_files):
        sub_id = int(f.split('_')[-3])
        if autistic_ids.isin([sub_id]).any():
            subject_vars[sub_id] = [f, m, 0]

        if control_ids.isin([sub_id]).any():
            subject_vars[sub_id] = [f, m, 1]

    return subject_vars


def get_entropy(series, nbins=15):
    """
    :param series: a 1-D array for which we need to find the entropy
    :param nbins: number of bins for histogram
    :return: entropy
    """
    # https://www.mathworks.com/matlabcentral/answers/27235-finding-entropy-from-a-probability-distribution

    counts, bin_edges = np.histogram(series, bins=nbins)
    p = counts / np.sum(counts, dtype=float)
    bin_width = np.diff(bin_edges)
    entropy = -np.sum(xlogy(p, p / bin_width))

    return entropy

def get_autocorr(series, lag_cc = 0.5):
    """

    :param series: time course
    :param lag_cc: width to be determined at lag=lag_cc
    :return:
    """

    cc = np.abs(np.correlate(series, series, mode='full')-lag_cc)
    return np.argsort(cc)[0] - len(cc) # because max cc is at len(cc)

def convert2summary(fmri_nifti, mask_nifti, metric='entropy'):
    """

    :param fmri_nifti: path to the 4-D resting-state functional scan.
    :param mask_nifti: path to the 3-D functional mask.
    :return: 3D nifti-file with entropy values in each voxels.
    """

    fmri = nib.load(fmri_nifti).get_data()
    mask = nib.load(mask_nifti).get_data()

    voxelwise_measure = np.zeros_like(mask)

    x_axis, y_axis, z_axis = mask.nonzero()

    for i, j, k in zip(x_axis, y_axis, z_axis):
        if metric == 'entropy':
            voxelwise_measure[i, j, k] = get_entropy(fmri[i, j, k, :])

        if metric == 'autcorr':
            voxelwise_measure[i, j, k] = get_autocorr(fmri[i, j, k, :])

    return voxelwise_measure


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def create_hdf5_file(subject_vars, hdf5_dir, hdf5_file, metrics = ['entropy']):
    """

    :param subject_vars: Dictionary with keys=sub_id and values=[func, mask, diagnosis]
    :param hdf5_dir: directory where the hdf5_file is stored
    :param hdf5_file: filename of the hdf5_file
    :param metric: [entropy, ALFF, pALFF, mean or any others
    :return: hdf5 file with the metric of interest[entropy, ALFF, pALFF, mean]
    """

    ensure_folder(hdf5_dir)

    nsubjs = len(subject_vars)
    data_shape = get_data_shape(subject_vars)

    sub_labels = []
    abide_ids = []

    with h5py.File(os.path.join(hdf5_dir, hdf5_file), 'w') as hfile:

        entry = hfile.create_group(u'summaries')

        # Remember to refactor to include as many metrics with a loop
        if 'entropy' in metrics:
            fmri_entropy = entry.create_dataset(u'data_entropy', shape=(nsubjs, ) + data_shape + (1,), dtype=np.float32)

        if 'autocorr' in metrics:
            fmri_autocorr = entry.create_dataset(u'data_autocorr', shape=(nsubjs,) + data_shape + (1,), dtype=np.float32)

        for sub_id, (k, v) in enumerate(subject_vars.items()):
            t1 = time()

            if 'autocorr' in metrics:
                summary_img = convert2summary(v[0], v[1], metric='autocorr')
                summary_img = np.array(summary_img, dtype=np.float32)[:, :, :, np.newaxis]
                fmri_entropy[sub_id, :, :, :] = summary_img

            if 'entropy' in metrics:
                summary_img = convert2summary(v[0], v[1], metric='entropy')
                summary_img = np.array(summary_img, dtype=np.float32)[:, :, :, np.newaxis]
                fmri_autocorr[sub_id, :, :, :] = summary_img

            sub_labels.append(v[2])  # Diagnosis
            abide_ids.append(k)
            t2 = time()
            stdout.write('\r{}/{}: {:0.2}sec'.format(sub_id + 1, nsubjs, t2 - t1))
            stdout.flush()

        entry.attrs['labels'] = sub_labels
        entry.attrs['sub_ids'] = abide_ids


def get_data_shape(subject_vars):
    """

    :param subject_vars: Dictionary with keys=sub_id and values=[func, mask, diagnosis]
    :return: shape of the 3D output
    """
    # Get any value
    _, m, _ = next(iter(subject_vars.values()))

    return nib.load(m).shape


def run():

    data_dir   = '/data/local/deeplearning/DeepPyschNet/abide_I_data'
    abide_rsfmri_dir  = 'Outputs/cpac/filt_noglobal/func_preproc/'
    abide_funcmask_dir  = 'Outputs/cpac/filt_noglobal/func_mask/'
    pheno_file = 'Phenotypic_V1_0b_preprocessed1.csv'

    output_hdf5_dir  = '/data/local/deeplearning/DeepPyschNet/abide_I_data/hdf5_data'
    output_hdf5_file = 'fmri_summary.hdf5'

    subject_vars = subject_dict(data_dir, abide_rsfmri_dir, abide_funcmask_dir, pheno_file)
    create_hdf5_file(subject_vars, output_hdf5_dir, output_hdf5_file, metrics=['entropy', 'autocorr'])

if __name__ == '__main__':
    run()



