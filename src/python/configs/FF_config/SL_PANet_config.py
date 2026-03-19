from yacs.config import CfgNode as CN
_CN = CN()
# loss
# _CN.l1_pretrain = False
_CN.train_loss = 'l1+ssim'
_CN.batch_size = 24
_CN.surf_indices = [125]
_CN.test = CN()
### TRAINER
_CN.trainer = CN()
# _CN.trainer.l1_pretrain = 200
# Vis settings
_CN.trainer.train_vis_num = 1000
_CN.test.test_vis_num = 1000

_CN.vis_port = 1234
_CN.trainer.randseed = 0
_CN.trainer.num_epochs = 10
_CN.trainer.Max_iters = 80000

# Learning-rate scheduler settings
_CN.trainer.cmp_learning_rate = 0.0015
_CN.trainer.cmp_weight_decay = 1e-4

_CN.trainer.comp_scheduler = CN()
_CN.trainer.comp_scheduler.type = 'step_lr'    # Optional: 'step_lr', 'lambda_lr', 'cosine_annealing'
_CN.trainer.comp_scheduler.step_size = 24000   # Used only when type is 'step_lr'
_CN.trainer.comp_scheduler.gamma = 0.2
# Checkpoint settings
_CN.trainer.save_num = 1000

def get_cfg():
    return _CN.clone()