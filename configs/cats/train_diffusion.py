import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.setup = ml_collections.ConfigDict()
    config.data = ml_collections.ConfigDict()
    config.data.dataset_params = ml_collections.ConfigDict()
    config.data.dataloader_params = ml_collections.ConfigDict()
    config.sde = ml_collections.ConfigDict()
    config.diffusion_model = ml_collections.ConfigDict()
    config.genie_model = ml_collections.ConfigDict()
    config.sampler = ml_collections.ConfigDict()
    config.optim = ml_collections.ConfigDict()
    config.optim.params = ml_collections.ConfigDict()
    config.train = ml_collections.ConfigDict()

    config.setup.runner = 'train_diffusion_base'

    config.data.image_size = 128
    config.data.num_channels = 3
    config.data.fid_stats = ['assets/stats/cats.zip']
    config.data.path = 'data/processed/cats_128.zip'
    config.data.num_classes = None
    config.data.dataset_params.xflip = True
    config.data.dataloader_params.num_workers = 1
    config.data.dataloader_params.pin_memory = True
    config.data.dataloader_params.drop_last = True

    config.sde.beta_min = .1
    config.sde.beta_d = 19.9

    config.diffusion_model.name = 'openai'
    config.diffusion_model.ema_rate = 0.9999
    config.diffusion_model.num_in_channels = config.data.num_channels
    config.diffusion_model.num_out_channels = config.data.num_channels
    config.diffusion_model.nf = 192
    config.diffusion_model.ch_mult = (1, 2, 2, 3, 3)
    config.diffusion_model.num_res_blocks = 2
    config.diffusion_model.attn_resolutions = (8, 16)
    config.diffusion_model.resamp_with_conv = True
    config.diffusion_model.dropout = .1
    config.diffusion_model.image_size = config.data.image_size
    config.diffusion_model.num_heads = None
    config.diffusion_model.num_head_channels = 64
    config.diffusion_model.num_head_upsample = -1
    config.diffusion_model.use_scale_shift_norm = True
    config.diffusion_model.resblock_updown = True
    config.diffusion_model.use_new_attention_order = True
    config.diffusion_model.num_classes = None
    config.diffusion_model.fourier_scale = 16
    config.diffusion_model.pred = 'v'
    config.diffusion_model.M = 1.
    config.diffusion_model.ckpt_path = 'none'

    config.sampler.name = 'ddim'
    config.sampler.batch_size = 64
    config.sampler.n_steps = 16
    config.sampler.denoising = False
    config.sampler.quadratic_striding = False
    config.sampler.eps = 1e-3
    config.sampler.afs = False
    config.sampler.denoising = False

    config.optim.optimizer = 'Adam'
    config.optim.params.learning_rate = 1e-4
    config.optim.params.weight_decay = 0.
    config.optim.params.grad_clip = 1.

    config.train.seed = 0
    config.train.eps = 1e-3
    config.train.n_iters = 400000
    config.train.n_warmup_iters = 100000
    config.train.batch_size = 16
    config.train.autocast = True
    config.train.log_freq = 100
    config.train.snapshot_freq = 10000
    config.train.snapshot_threshold = 1
    config.train.save_freq = 50000
    config.train.save_threshold = 1
    config.train.fid_freq = 50000
    config.train.fid_threshold = 1
    config.train.fid_samples = 10000

    return config
