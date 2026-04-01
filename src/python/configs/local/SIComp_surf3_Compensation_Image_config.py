from yacs.config import CfgNode as CN
_CN = CN()
_CN.name = "SIComp(#surf=3)"
_CN.model_name = "SIComp(#surf=3)"
_CN.transformer = 'latentcostformer'
_CN.train_loss = 'l1+ssim+diff+lpips'
_CN.batch_size = 1
_CN.num_train = 500
_CN.num_valid = 100
_CN.surf_indices = [0, 62, 124]
_CN.OmniCompNet_pretrian_path = r"../../checkpoint/SIComp/SIComp_surf3/l1+ssim_step_12000_final.pth"
# _CN.OmniCompNet_pretrian_path = r"/home/haibo/FF_CompenUltra_lab/checkpoint/SIComp/SIComp_surf3/l1+ssim_step_12000_final.pth"

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


def get_cfg():
    return _CN.clone()