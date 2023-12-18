

def define_Model(opt):
    model = opt['model']


    if model == 'cdiffmr_m.0.4.blur':
        from models.model.cdiffmr.model_cdiffmr_blur_m04 import CDiffMR as M

    elif model == 'cdiffmr_m.0.4.gnoise':
        from models.model.cdiffmr.model_cdiffmr_gnoise_m04 import CDiffMR as M

    elif model == 'cdiffmr_m.0.4.ksu':
        from models.model.cdiffmr.model_cdiffmr_ksu_m04 import CDiffMR as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
