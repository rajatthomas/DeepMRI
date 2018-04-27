import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_path',
        default='/data_local/deeplearning/data/ABIDE_hdf5', # '/data_local/deeplearning/DeepNeurologe/',  # '/data_local/deeplearning/PAC_competition/numpy_data',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--data_file',
        default='fmri_summary.hdf5', # ''data_6mm.npz',  # 'data4mm.npz', # data.npz (data for PAC competition 2mm resolution)
        type=str,
        help='Numpy binary file with train, test and validation datasets')
    parser.add_argument(
        '--result_path',
        default='/data_local/deeplearning/data/ABIDE_output',
        type=str,
        help='Result directory path')
    parser.add_argument(
        '--metric',
        default='alff',
        type=str,
        help='alff | autocorr | eigenvector_weighted | entropy | falff | lfcd | structural')
    parser.add_argument(
        '--n_classes',
        default=2,
        type=int,
        help=
        'Number of classes (depressed vs controls as default)'
    )
    parser.add_argument(
        '--image_size',
        default=(45, 54, 45),  # (30, 36, 30),  # (121, 145, 121), # (45, 54, 45),
        type=int,
        help='tuple of x-, y- and z- dimensions, e.g., (109, 91, 109)')
    parser.add_argument(
        '--standardize', action='store_false', help='standardize across subject dimension')
    parser.add_argument(
        '--learning_rate',
        default=0.1,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument(
        '--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument(
        '--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument(
        '--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument(
        '--optimizer',
        default='sgd',
        type=str,
        help='Currently only support SGD')
    parser.add_argument(
        '--lr_patience',
        default=50,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    )
    parser.add_argument(
        '--batch_size', default=4, type=int, help='Batch Size')
    parser.add_argument(
        '--n_epochs',
        default=70,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--begin_epoch',
        default=1,
        type=int,
        help=
        'Training begins at this epoch. Previous trained model indicated by resume_path is loaded.'
    )
    parser.add_argument(
        '--n_val_samples',
        default=3,
        type=int,
        help='Number of validation samples for each activity')
    parser.add_argument('--resume_path', action='store_const', const='save_model') # help='Save data (.pth) of previous training')
    parser.add_argument(
        '--no_train',
        action='store_true',
        help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument(
        '--no_val',
        action='store_true',
        help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument(
        '--no_test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=False)
    parser.add_argument(
        '--test_subset',
        default='val',
        type=str,
        help='Used subset in test (val | test)')
    parser.add_argument(
        '--no_softmax_in_test',
        action='store_true',
        help='If true, output for each clip is not normalized using softmax.')
    parser.set_defaults(no_softmax_in_test=False)
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument(
        '--n_threads',
        default=4,
        type=int,
        help='Number of threads for multi-thread loading')
    parser.add_argument(
        '--checkpoint',
        default=10,
        type=int,
        help='Trained model is saved at every this epochs.')
    parser.add_argument(
        '--no_hflip',
        action='store_true',
        help='If true holizontal flipping is not performed.')
    parser.set_defaults(no_hflip=False)
    parser.add_argument(
        '--norm_value',
        default=1,
        type=int,
        help=
        'If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument(
        '--model_depth',
        default=34,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101) Depth of densenet (121 | 169 | 201 | 264)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument(
        '--resnext_cardinality',
        default=32,
        type=int,
        help='ResNeXt cardinality')
    parser.add_argument(
        '--manual_seed', default=42, type=int, help='Manually set random seed')

    args = parser.parse_args()

    return args
