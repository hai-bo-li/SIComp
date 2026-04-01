from yacs.config import CfgNode as CN
_CN = CN()
# loss
_CN.train_loss = 'l1+ssim+diff+lpips'
_CN.batch_size = 4
_CN.num_workers = 4
_CN.image_size = [256, 256]
_CN.surf_indices = [125]
_CN.transformer = 'latentcostformer'
_CN.restore_ckpt = r'/home/haibo/FF_CompenUltra_lab/checkpoint/FlowFormer/sintel.pth'
# _CN.CompenNeSt_ckpt = r"/home/haibo/FF_CompenUltra_lab/checkpoint/FF_CompenNeSt/SL_CompenNeSt_final.pth"
# latentcostformer
_CN.latentcostformer = CN()
_CN.latentcostformer.pe = 'linear'
_CN.latentcostformer.dropout = 0.0
_CN.latentcostformer.encoder_latent_dim = 256  # in twins, this is 256
_CN.latentcostformer.query_latent_dim = 64
_CN.latentcostformer.cost_latent_input_dim = 64
_CN.latentcostformer.cost_latent_token_num = 8
_CN.latentcostformer.cost_latent_dim = 128
_CN.latentcostformer.arc_type = 'transformer'
_CN.latentcostformer.cost_heads_num = 1
# encoder
_CN.latentcostformer.pretrain = True
_CN.latentcostformer.context_concat = False
_CN.latentcostformer.encoder_depth = 3
_CN.latentcostformer.feat_cross_attn = False
_CN.latentcostformer.patch_size = 8
_CN.latentcostformer.patch_embed = 'single'
_CN.latentcostformer.no_pe = False
_CN.latentcostformer.gma = "GMA"
_CN.latentcostformer.kernel_size = 9
_CN.latentcostformer.rm_res = True
_CN.latentcostformer.vert_c_dim = 64
_CN.latentcostformer.cost_encoder_res = True
_CN.latentcostformer.cnet = 'twins'
_CN.latentcostformer.fnet = 'twins'
_CN.latentcostformer.no_sc = False
_CN.latentcostformer.only_global = False
_CN.latentcostformer.add_flow_token = True
_CN.latentcostformer.use_mlp = False
_CN.latentcostformer.vertical_conv = False

# decoder
_CN.latentcostformer.decoder_depth = 12
_CN.latentcostformer.critical_params = ['cost_heads_num', 'vert_c_dim', 'cnet', 'pretrain', 'add_flow_token',
                                        'encoder_depth', 'gma', 'cost_encoder_res']

_CN.test = CN()
### TRAINER
_CN.trainer = CN()
_CN.trainer.l1_pretrain = False
_CN.trainer.l1_pretrain_num = 200

# Vis settings
_CN.vis_port = 1234
_CN.trainer.train_vis_num = 500
_CN.test.test_vis_num = 500
# Checkpoint saving
_CN.trainer.save_num = 1000
# General hyperparameters
_CN.trainer.randseed = 0
_CN.trainer.num_epochs = 10
_CN.trainer.Max_iters = 18000

# Network training parameters
_CN.trainer.flow_learning_rate = 0.000035
_CN.trainer.flow_weight_decay = 1e-05
_CN.trainer.cmp_learning_rate = 0.001
_CN.trainer.cmp_weight_decay = 1e-4

# Scheduler settings
_CN.trainer.flow_scheduler = CN()
_CN.trainer.flow_scheduler.type = 'step_lr'  # Optional: 'step_lr', 'lambda_lr', 'cosine_annealing'
_CN.trainer.flow_scheduler.step_size = 5000  # Used only when type is 'step_lr'   #32000
_CN.trainer.flow_scheduler.gamma = 0.90  # Used only when type is 'step_lr'

_CN.trainer.comp_scheduler = CN()
_CN.trainer.comp_scheduler.type = 'step_lr'  # Optional: 'step_lr', 'lambda_lr', 'cosine_annealing'
_CN.trainer.comp_scheduler.step_size = 12000  # Used only when type is 'step_lr' #32000
_CN.trainer.comp_scheduler.gamma = 0.2

def get_cfg():
    return _CN.clone()
