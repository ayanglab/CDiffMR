

def define_diffusion_model(opt):
    diffusion_opt = opt['diffusion']
    diffusion_type = diffusion_opt['diffusion_type']

    # default version
    if diffusion_type == 'cold_diffusion_gnoise':
        from models.model.cdiffmr.diffusion_model.cdm_gnoise_m05 import GaussianDiffusion as model
    elif diffusion_type == 'cold_diffusion_blur':
        from models.model.cdiffmr.diffusion_model.cdm_blur_m05 import GaussianDiffusion as model
    elif diffusion_type == 'cold_diffusion_ksu':
        from models.model.cdiffmr.diffusion_model.cdm_ksu_m05 import GaussianDiffusion as model

    # m.0.5 (m.0.4 -- > m.0.5 Nothing change)
    elif diffusion_type == 'cold_diffusion_gnoise_m.0.5':
        from models.model.cdiffmr.diffusion_model.cdm_gnoise_m05 import GaussianDiffusion as model
    elif diffusion_type == 'cold_diffusion_blur_m.0.5':
        from models.model.cdiffmr.diffusion_model.cdm_blur_m05 import GaussianDiffusion as model
    elif diffusion_type == 'cold_diffusion_ksu_m.0.5':
        from models.model.cdiffmr.diffusion_model.cdm_ksu_m05 import GaussianDiffusion as model

    elif diffusion_type == 'cold_diffusion_gnoise_m.0.4':
        from models.model.cdiffmr.diffusion_model.cdm_gnoise_m05 import GaussianDiffusion as model
    elif diffusion_type == 'cold_diffusion_blur_m.0.4':
        from models.model.cdiffmr.diffusion_model.cdm_blur_m05 import GaussianDiffusion as model
    elif diffusion_type == 'cold_diffusion_ksu_m.0.4':
        from models.model.cdiffmr.diffusion_model.cdm_ksu_m05 import GaussianDiffusion as model

    else:
        raise NotImplementedError(f'Unknown diffusion type {diffusion_type}')

    return model