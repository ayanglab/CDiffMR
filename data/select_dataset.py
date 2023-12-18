


def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()


    # -----------------------------------------------------------
    # FastMRI
    # -----------------------------------------------------------
    if dataset_type in ['fastmri.d.1.0.complex']:
        raise NotImplementedError
        from data.dataset_FastMRI_complex_d10 import DatasetFastMRI as D

    elif dataset_type in ['fastmri.d.1.1.complex']:
        from data.dataset_FastMRI_complex import DatasetFastMRI as D

    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
