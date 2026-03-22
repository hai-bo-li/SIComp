from .utils_all.swin_transformer_v2 import *
from .TransformerLZH.Transformer.SwinTransformer.MySwinBlocks import MySwinFormerBlocks

def predict_flow_highres(FlowFormer, GT, x, target_size=(512, 512), train_size=(256, 256)):
    """
    Predict optical flow for high-resolution images using a FlowFormer model trained at a lower resolution.

    Args:
        FlowFormer: Trained FlowFormer model.
        GT: Target image at high resolution, for example (B, 3, 512, 512).
        x: Input image at high resolution, for example (B, 3, 512, 512).
        target_size: Original image size, for example (512, 512).
        train_size: Training resolution, for example (256, 256).

    Returns:
        Interpolated high-resolution flow resized to match ``target_size``.
    """
    # Step 1: resize the input images to the training resolution (for example 256x256)
    GT_low = F.interpolate(GT, size=train_size, mode='bilinear', align_corners=True)
    x_low = F.interpolate(x, size=train_size, mode='bilinear', align_corners=True)

    # Step 2: run FlowFormer inference at low resolution
    with torch.no_grad():
        flow_low = FlowFormer(GT_low, x_low)  # e.g. [B, 2, 256, 256]

        # Step 3: resize the flow to the high resolution and scale the displacement values
        flow_high = F.interpolate(flow_low, size=target_size, mode='bilinear', align_corners=True)

        # Scale the flow displacement values at the same time (for example from 256 to 512)
        scale_y = target_size[0] / train_size[0]
        scale_x = target_size[1] / train_size[1]
        flow_high[:, 0, :, :] *= scale_x  # dx
        flow_high[:, 1, :, :] *= scale_y  # dy

    return flow_high

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # The MLP (Multilayer Perceptron) is implemented with 1x1 convolutions
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

# Spatial attention module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))


# Integrated CBAM module
class CBAMBlock(nn.Module):
    def __init__(self, channels, ratio=1, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class SCNet_surf5(nn.Module):
    def __init__(self):
        super(SCNet_surf5, self).__init__()
        self.name = self.__class__.__name__
        self.relu = nn.ReLU()
        self.simplified = False
        # CBAM blocks for res connections
        self.cbam_res2_s = CBAMBlock(64)
        self.cbam_res3_s = CBAMBlock(128)
        self.cbam_res2 = CBAMBlock(64)
        self.cbam_res3 = CBAMBlock(128)
        self.cbam_res4 = CBAMBlock(256)
        self.cbam_res5 = CBAMBlock(128)
        self.cbam_trans1 = CBAMBlock(64)
        self.cbam_trans2 = CBAMBlock(32)
        self.cbam_conv6 = CBAMBlock(3)
        self.Upsample_and_conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        )
        # siamese encoder
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.conv1_surf = nn.Conv2d(15, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 128, 3, 1, 1)
        self.Swin_skip_21 = MySwinFormerBlocks(input_feature_size=(128, 128),
                                               input_feature_channels=32,
                                               skiped_patch_embed=False,
                                               block_depths=[2],
                                               out_indices=[0],
                                               nums_head=[2],
                                               patch_size=(1, 1),
                                               downsample=False,
                                               embedd_dim=64,
                                               use_ape=False,
                                               frozen_stage=-1,
                                               use_prenorm=True,
                                               norm_layer=nn.LayerNorm).cuda()

        # transposed conv
        self.transConv2 = nn.ConvTranspose2d(64, 32, 2, 2, 0)
        self.conv6 = nn.Conv2d(32, 3, 3, 1, 1)
        # s1
        self.skipConv11 = nn.Conv2d(3, 3, 1, 1, 0)
        self.skipConv11_surf = nn.Conv2d(15, 3, 1, 1, 0)
        self.skipConv12 = nn.Conv2d(3, 3, 3, 1, 1)
        self.skipConv13 = nn.Conv2d(3, 3, 3, 1, 1)
        # s2
        # self.skipConv21 = nn.Conv2d(32, 64, 1, 1, 0)
        self.skipConv22 = nn.Conv2d(64, 64, 3, 1, 1)

        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                torch.manual_seed(0)
                nn.init.kaiming_normal_(m.weight)

        self.apply(_initialize_weights)

    # x is the input uncompensated image, s is a 1x2sx2s56x2s56 surface image
    def forward(self, x, s):
        res1_s = self.relu(self.skipConv11_surf(s))
        res1_s = self.relu(self.skipConv12(res1_s))
        res1_s = self.skipConv13(res1_s)
        s = self.relu(self.conv1_surf(s))
        res2_s = self.Swin_skip_21(s)[0]
        # res2_s = self.skipConv21(s)
        res2_s = self.relu(res2_s)
        res2_s = self.skipConv22(res2_s)
        res2_s = self.cbam_res2_s(res2_s)  # Apply CBAMBlock
        s = self.relu(self.conv2(s))
        s = self.relu(self.conv3(s))
        res3_s = self.cbam_res3_s(s)  # Apply CBAMBlock

        res1 = self.relu(self.skipConv11(x))
        res1 = self.relu(self.skipConv12(res1))
        res1 = self.skipConv13(res1)
        res1 = res1 - res1_s
        x = self.relu(self.conv1(x))

        res2 = self.Swin_skip_21(x)[0]
        # res2 = self.skipConv21(x)
        res2 = self.relu(res2)

        res2 = self.skipConv22(res2)
        res2 = self.cbam_res2(res2)  # Apply CBAMBlock
        res2 = res2 - res2_s
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.cbam_res3(x)
        x = x - res3_s  # s3

        x = self.relu(self.conv4(x))
        x = self.cbam_res4(x)
        x = self.relu(self.conv5(x))
        x = self.cbam_res5(x)

        x = self.relu(self.Upsample_and_conv1(x) + res2)
        x = self.cbam_trans1(x)
        x = self.relu(self.transConv2(x))
        x = self.cbam_trans2(x)
        x = self.relu(self.conv6(x) + res1)
        x = self.cbam_conv6(x)
        x = torch.clamp(x, max=1)
        return x

class SCNet_surf3(nn.Module):
    def __init__(self):
        super(SCNet_surf3, self).__init__()
        self.name = self.__class__.__name__
        self.relu = nn.ReLU()
        self.simplified = False
        # CBAM blocks for res connections
        self.cbam_res2_s = CBAMBlock(64)
        self.cbam_res3_s = CBAMBlock(128)
        self.cbam_res2 = CBAMBlock(64)
        self.cbam_res3 = CBAMBlock(128)
        self.cbam_res4 = CBAMBlock(256)
        self.cbam_res5 = CBAMBlock(128)
        self.cbam_trans1 = CBAMBlock(64)
        self.cbam_trans2 = CBAMBlock(32)
        self.cbam_conv6 = CBAMBlock(3)
        self.Upsample_and_conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        )
        # siamese encoder
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.conv1_surf = nn.Conv2d(9, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 128, 3, 1, 1)

        self.Swin_skip_21 = MySwinFormerBlocks(input_feature_size=(128, 128),
                                               input_feature_channels=32,
                                               skiped_patch_embed=False,
                                               block_depths=[2],
                                               out_indices=[0],
                                               nums_head=[2],
                                               patch_size=(1, 1),
                                               downsample=False,
                                               embedd_dim=64,
                                               use_ape=False,
                                               frozen_stage=-1,
                                               use_prenorm=True,
                                               norm_layer=nn.LayerNorm).cuda()

        # transposed conv
        self.transConv2 = nn.ConvTranspose2d(64, 32, 2, 2, 0)
        self.conv6 = nn.Conv2d(32, 3, 3, 1, 1)
        # s1
        self.skipConv11 = nn.Conv2d(3, 3, 1, 1, 0)
        self.skipConv11_surf = nn.Conv2d(9, 3, 1, 1, 0)
        self.skipConv12 = nn.Conv2d(3, 3, 3, 1, 1)
        self.skipConv13 = nn.Conv2d(3, 3, 3, 1, 1)
        # s2
        # self.skipConv21 = nn.Conv2d(32, 64, 1, 1, 0)
        self.skipConv22 = nn.Conv2d(64, 64, 3, 1, 1)

        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                torch.manual_seed(0)
                nn.init.kaiming_normal_(m.weight)

        self.apply(_initialize_weights)

    # x is the input uncompensated image, s is a 1x2sx2s56x2s56 surface image
    def forward(self, x, s):
        res1_s = self.relu(self.skipConv11_surf(s))
        res1_s = self.relu(self.skipConv12(res1_s))
        res1_s = self.skipConv13(res1_s)
        s = self.relu(self.conv1_surf(s))
        res2_s = self.Swin_skip_21(s)[0]
        # res2_s = self.skipConv21(s)
        res2_s = self.relu(res2_s)
        res2_s = self.skipConv22(res2_s)
        res2_s = self.cbam_res2_s(res2_s)  # Apply CBAMBlock
        s = self.relu(self.conv2(s))
        s = self.relu(self.conv3(s))
        res3_s = self.cbam_res3_s(s)  # Apply CBAMBlock

        res1 = self.relu(self.skipConv11(x))
        res1 = self.relu(self.skipConv12(res1))
        res1 = self.skipConv13(res1)
        res1 = res1 - res1_s
        x = self.relu(self.conv1(x))

        res2 = self.Swin_skip_21(x)[0]
        # res2 = self.skipConv21(x)
        res2 = self.relu(res2)

        res2 = self.skipConv22(res2)
        res2 = self.cbam_res2(res2)  # Apply CBAMBlock
        res2 = res2 - res2_s
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.cbam_res3(x)
        x = x - res3_s  # s3

        x = self.relu(self.conv4(x))
        x = self.cbam_res4(x)
        x = self.relu(self.conv5(x))
        x = self.cbam_res5(x)

        x = self.relu(self.Upsample_and_conv1(x) + res2)
        x = self.cbam_trans1(x)
        x = self.relu(self.transConv2(x))
        x = self.cbam_trans2(x)
        x = self.relu(self.conv6(x) + res1)
        x = self.cbam_conv6(x)
        x = torch.clamp(x, max=1)
        return x


class SCNet_surf1(nn.Module):
    def __init__(self):
        super(SCNet_surf1, self).__init__()
        self.name = self.__class__.__name__
        self.relu = nn.ReLU()
        self.cbam_res2_s = CBAMBlock(64)
        self.cbam_res3_s = CBAMBlock(128)
        self.cbam_res2 = CBAMBlock(64)
        self.cbam_res3 = CBAMBlock(128)
        self.cbam_res4 = CBAMBlock(256)
        self.cbam_res5 = CBAMBlock(128)
        self.cbam_trans1 = CBAMBlock(64)
        self.cbam_trans2 = CBAMBlock(32)
        self.cbam_conv6 = CBAMBlock(3)
        self.Upsample_and_conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        )
        # self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # siamese encoder
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        # self.conv1_surf = nn.Conv2d(15, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 128, 3, 1, 1)

        self.Swin_skip_21 = MySwinFormerBlocks(input_feature_size=(128, 128),
                                               input_feature_channels=32,
                                               skiped_patch_embed=False,
                                               block_depths=[2],
                                               out_indices=[0],
                                               nums_head=[2],
                                               patch_size=(1, 1),
                                               downsample=False,
                                               embedd_dim=64,
                                               use_ape=False,
                                               frozen_stage=-1,
                                               use_prenorm=True,
                                               norm_layer=nn.LayerNorm).cuda()

        # transposed conv
        self.transConv2 = nn.ConvTranspose2d(64, 32, 2, 2, 0)
        self.conv6 = nn.Conv2d(32, 3, 3, 1, 1)
        # s1
        self.skipConv11 = nn.Conv2d(3, 3, 1, 1, 0)
        self.skipConv12 = nn.Conv2d(3, 3, 3, 1, 1)
        self.skipConv13 = nn.Conv2d(3, 3, 3, 1, 1)
        # s2
        self.skipConv22 = nn.Conv2d(64, 64, 3, 1, 1)

        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                torch.manual_seed(0)
                nn.init.kaiming_normal_(m.weight)

        self.apply(_initialize_weights)

    # x is the input uncompensated image, s is a 1x2sx2s56x2s56 surface image
    def forward(self, x, s):
        res1_s = self.relu(self.skipConv11(s))
        res1_s = self.relu(self.skipConv12(res1_s))
        res1_s = self.skipConv13(res1_s)
        s = self.relu(self.conv1(s))
        res2_s = self.Swin_skip_21(s)[0]

        res2_s = self.relu(res2_s)
        res2_s = self.skipConv22(res2_s)
        res2_s = self.cbam_res2_s(res2_s)  # Apply CBAMBlock
        s = self.relu(self.conv2(s))
        s = self.relu(self.conv3(s))
        res3_s = self.cbam_res3_s(s)  # Apply CBAMBlock

        res1 = self.relu(self.skipConv11(x))
        res1 = self.relu(self.skipConv12(res1))
        res1 = self.skipConv13(res1)
        res1 = res1 - res1_s
        x = self.relu(self.conv1(x))

        res2 = self.Swin_skip_21(x)[0]
        res2 = self.relu(res2)

        res2 = self.skipConv22(res2)
        res2 = self.cbam_res2(res2)  # Apply CBAMBlock
        res2 = res2 - res2_s
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.cbam_res3(x)
        x = x - res3_s  # s3

        x = self.relu(self.conv4(x))
        x = self.cbam_res4(x)
        x = self.relu(self.conv5(x))
        x = self.cbam_res5(x)

        x = self.relu(self.Upsample_and_conv1(x) + res2)
        x = self.cbam_trans1(x)
        x = self.relu(self.transConv2(x))
        x = self.cbam_trans2(x)
        x = self.relu(self.conv6(x) + res1)
        x = self.cbam_conv6(x)
        x = torch.clamp(x, max=1)
        return x


class PA(nn.Module):
    def __init__(self, channel):
        super(PA, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sig = nn.Sigmoid()

        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)

        self.apply(_initialize_weights)

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.sig(c1_)

        return x * c1


class PANet(nn.Module):
    def __init__(self):
        super(PANet, self).__init__()
        self.name = self.__class__.__name__
        self.relu = nn.ReLU()
        self.scale = 4

        self.simplified = False
        self.unshuffle = nn.PixelUnshuffle(self.scale)
        self.pa1 = PA(48)
        self.pa2 = PA(48)

        # siamese encoder
        self.conv1 = nn.Conv2d(3 * self.scale * self.scale, 48, 3, 2, 1)
        self.conv2 = nn.Conv2d(48, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 128, 3, 1, 1)

        # transposed conv
        self.transConv1 = nn.ConvTranspose2d(128, 64, 2, 2, 0)
        self.transConv2 = nn.ConvTranspose2d(64, 48, 2, 2, 0)

        # output layer
        self.conv6 = nn.Conv2d(48, 3 * self.scale * self.scale, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(self.scale)

        # skip layers (see s3 in forward)
        # s1
        self.skipConv11 = nn.Conv2d(48, 48, 1, 1, 0)
        self.skipConv12 = nn.Conv2d(48, 48, 3, 1, 1)
        self.skipConv13 = nn.Conv2d(48, 48, 3, 1, 1)

        # s2
        self.skipConv21 = nn.Conv2d(48, 64, 1, 1, 0)
        self.skipConv22 = nn.Conv2d(64, 64, 3, 1, 1)

        # stores biases of surface feature branch (net simplification)
        self.register_buffer('res1_s_pre', None)
        self.register_buffer('res2_s_pre', None)
        self.register_buffer('res3_s_pre', None)

        # initialization function, first checks the module type,
        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)

        self.apply(_initialize_weights)

    # simplify trained model by trimming surface branch to biases

    def simplify(self, s):
        s = (self.unshuffle(s))
        s = self.pa1(s)

        res1_s = self.relu(self.skipConv11(s))
        res1_s = self.relu(self.skipConv12(res1_s))
        res1_s = self.skipConv13(res1_s)
        self.res1_s_pre = res1_s

        s = (self.conv1(s))
        s = self.pa2(s)

        res2_s = self.skipConv21(s)
        res2_s = self.relu(res2_s)
        res2_s = self.skipConv22(res2_s)
        self.res2_s_pre = res2_s

        s = self.relu(self.conv2(s))
        s = self.relu(self.conv3(s))
        self.res3_s_pre = s

        self.res1_s_pre = self.res1_s_pre.squeeze()
        self.res2_s_pre = self.res2_s_pre.squeeze()
        self.res3_s_pre = self.res3_s_pre.squeeze()

        self.simplified = True

    # x is the input uncompensated image, s is a surface image
    def forward(self, x, s):
        # surface feature extraction

        # alternate between surface and image branch
        if self.simplified:
            res1_s = self.res1_s_pre
            res2_s = self.res2_s_pre
            res3_s = self.res3_s_pre
        else:
            s = self.unshuffle(s)
            s = self.pa1(s)

            res1_s = self.relu(self.skipConv11(s))
            res1_s = self.relu(self.skipConv12(res1_s))
            res1_s = self.skipConv13(res1_s)

            s = (self.conv1(s))
            s = self.pa2(s)

            res2_s = self.skipConv21(s)
            res2_s = self.relu(res2_s)
            res2_s = self.skipConv22(res2_s)

            s = self.relu(self.conv2(s))
            res3_s = self.relu(self.conv3(s))

        x = self.unshuffle(x)
        x = self.pa1(x)

        res1 = self.relu(self.skipConv11(x))
        res1 = self.relu(self.skipConv12(res1))
        res1 = self.skipConv13(res1)
        res1 = res1 - res1_s

        x = (self.conv1(x))
        x = self.pa2(x)

        res2 = self.skipConv21(x)
        res2 = self.relu(res2)
        res2 = self.skipConv22(res2)
        res2 = res2 - res2_s

        x = self.relu(self.conv2(x))

        x = self.relu(self.conv3(x))
        x = x - res3_s  # s3

        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))

        # x = self.relu(self.transConv1(x) + res2)
        out1 = self.transConv1(x)
        if out1.shape != res2.shape:
            min_h = min(out1.shape[2], res2.shape[2])
            min_w = min(out1.shape[3], res2.shape[3])
            out1 = out1[:, :, :min_h, :min_w]
            res2 = res2[:, :, :min_h, :min_w]
        x = self.relu(out1 + res2)
        x = self.relu(self.transConv2(x) + res1)
        x = (self.conv6(x))
        x = self.shuffle(x)
        x = torch.clamp(x, max=1)
        x = torch.clamp(x, min=0)

        return x


class CompenNeSt(nn.Module):
    def __init__(self):
        super(CompenNeSt, self).__init__()
        self.name = self.__class__.__name__
        self.relu = nn.ReLU()

        self.simplified = False

        # siamese encoder
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 128, 3, 1, 1)

        # transposed conv
        self.transConv1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.transConv2 = nn.ConvTranspose2d(64, 32, 2, 2, 0)

        # output layer
        self.conv6 = nn.Conv2d(32, 3, 3, 1, 1)

        # skip layers (see s3 in forward)
        # s1
        self.skipConv11 = nn.Conv2d(3, 3, 1, 1, 0)
        self.skipConv12 = nn.Conv2d(3, 3, 3, 1, 1)
        self.skipConv13 = nn.Conv2d(3, 3, 3, 1, 1)

        # s2
        self.skipConv21 = nn.Conv2d(32, 64, 1, 1, 0)
        self.skipConv22 = nn.Conv2d(64, 64, 3, 1, 1)

        # stores biases of surface feature branch (net simplification)
        self.register_buffer('res1_s_pre', None)
        self.register_buffer('res2_s_pre', None)
        self.register_buffer('res3_s_pre', None)

        # initialization function, first checks the module type,
        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)

        self.apply(_initialize_weights)


    def simplify(self, s):
        res1_s = self.relu(self.skipConv11(s))
        res1_s = self.relu(self.skipConv12(res1_s))
        res1_s = self.skipConv13(res1_s)
        self.res1_s_pre = res1_s

        s = self.relu(self.conv1(s))

        res2_s = self.skipConv21(s)
        res2_s = self.relu(res2_s)
        res2_s = self.skipConv22(res2_s)
        self.res2_s_pre = res2_s

        s = self.relu(self.conv2(s))
        s = self.relu(self.conv3(s))
        self.res3_s_pre = s

        self.res1_s_pre = self.res1_s_pre.squeeze()
        self.res2_s_pre = self.res2_s_pre.squeeze()
        self.res3_s_pre = self.res3_s_pre.squeeze()

        self.simplified = True

    # x is the input uncompensated image, s is a 1x2sx2s56x2s56 surface image
    def forward(self, x, s):
        # surface feature extraction

        # alternate between surface and image branch
        if self.simplified:
            res1_s = self.res1_s_pre
            res2_s = self.res2_s_pre
            res3_s = self.res3_s_pre
        else:
            res1_s = self.relu(self.skipConv11(s))
            res1_s = self.relu(self.skipConv12(res1_s))
            res1_s = self.skipConv13(res1_s)

            s = self.relu(self.conv1(s))

            res2_s = self.skipConv21(s)
            res2_s = self.relu(res2_s)
            res2_s = self.skipConv22(res2_s)

            s = self.relu(self.conv2(s))
            res3_s = self.relu(self.conv3(s))

        res1 = self.relu(self.skipConv11(x))
        res1 = self.relu(self.skipConv12(res1))
        res1 = self.skipConv13(res1)
        # res1 -= res1_s
        res1 = res1 - res1_s
        x = self.relu(self.conv1(x))

        res2 = self.skipConv21(x)
        res2 = self.relu(res2)
        res2 = self.skipConv22(res2)
        # res2 -= res2_s
        res2 = res2 - res2_s
        x = self.relu(self.conv2(x))

        x = self.relu(self.conv3(x))
        # x -= res3_s # s3
        x = x - res3_s
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))

        x = self.relu(self.transConv1(x) + res2)
        x = self.relu(self.transConv2(x))
        x = self.relu(self.conv6(x) + res1)

        x = torch.clamp(x, max=1)

        return x

def print_mem(tag):
    print(f"[{tag}] Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB | Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")


import cv2
import numpy as np


def tensor_to_cvimg(tensor, img_index=0):
    """
    Convert the RGB image at index ``img_index`` from a tensor of shape [B, 15, H, W]
    into an OpenCV image in BGR format.

    By default, this extracts the 4th image (index 3), corresponding to channels 9 to 11.

    Args:
    - tensor: [B, 15, H, W]
    - img_index: Image index to extract, starting from 0

    Returns:
    - OpenCV-format BGR image (uint8)
    """
    assert tensor.shape[1] % 3 == 0, "The number of channels must be a multiple of 3 (3 channels per image)"
    num_imgs = tensor.shape[1] // 3
    assert 0 <= img_index < num_imgs, f"img_index should be within the range [0, {num_imgs-1}]"

    start_c = img_index * 3
    end_c = start_c + 3

    img = tensor[0, start_c:end_c].detach().cpu().clamp(0, 1).numpy()  # [3, H, W]
    img = np.transpose(img, (1, 2, 0))  # [H, W, 3]
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel
def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image

def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False, max_flow=None):
    """
    Expects a two-dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    if max_flow is None:
        rad = np.sqrt(np.square(u) + np.square(v))
        rad_max = np.max(rad)
    else:
        rad_max = max_flow
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)

class Connection(nn.Module):
    def __init__(self, G_Net, P_Net, warp_func, actual_cmp=False, train_size=(256, 256)):
        super(Connection, self).__init__()
        self.FlowFormer = G_Net
        self.CompenNeSt = P_Net
        self.warp = warp_func
        self.actual_cmp = actual_cmp
        self.train_size = train_size

        self.register_buffer('fixed_flow', None)

    def forward(self, GT, x, s, desire_x=None, step=1):
        b, c, h, w = x.shape

        if self.fixed_flow is not None:
            # If a cached flow already exists, expand it from [1, 2, H, W] to the current batch size [B, 2, H, W]
            flow = self.fixed_flow.repeat(b, 1, 1, 1)
        else:
            # If there is no cached flow, compute it on the fly
            input_size = (h, w)
            if input_size != self.train_size:
                flow = predict_flow_highres(self.FlowFormer, GT, x,
                                            target_size=input_size,
                                            train_size=self.train_size)
            else:
                flow = self.FlowFormer(GT, x)

        # Compensation / projection logic
        if self.actual_cmp:
            if desire_x is None:
                raise ValueError("desire_x is required for actual compensation mode.")
            desire_warped = self.warp(desire_x, flow)
            s_warped = self.warp(s, flow)
            actual_predict = self.CompenNeSt(desire_warped, s_warped)
            return actual_predict
        else:
            x_warped = self.warp(x, flow)
            s_warped = self.warp(s, flow)
            predict = self.CompenNeSt(x_warped, s_warped)
            return predict

class FlowCompensationModel(nn.Module):
    def __init__(self, flow_model, compensation_model, warp_func, actual_compensation=False, train_size=(256, 256)):
        """
        Generic Flow Compensation model that follows the buffer-based caching logic
        used in the Connection class.
        """
        super(FlowCompensationModel, self).__init__()
        self.flow_model = flow_model
        self.compensation_model = compensation_model
        self.warp = warp_func
        self.actual_compensation = actual_compensation
        self.train_size = train_size

        # Core update: use register_buffer to store a fixed flow tensor
        # This allows it to move with the model between GPU/CPU and keeps it out of gradient updates
        self.register_buffer('fixed_flow', None)

    def forward(self, GT, x, s, desire_x=None):
        b, c, h, w = x.shape
        input_size = (h, w)

        # Core update: optical flow logic
        if self.fixed_flow is not None:
            # If a cached flow already exists, expand it from [1, 2, H, W] to the current batch size [B, 2, H, W]
            flow = self.fixed_flow.repeat(b, 1, 1, 1)
        else:
            # If there is no cached flow, compute it on the fly
            if input_size != self.train_size:
                flow = predict_flow_highres(self.flow_model, GT, x,
                                            target_size=input_size,
                                            train_size=self.train_size)
            else:
                flow = self.flow_model(GT, x)

        # Compensation / projection logic
        if self.actual_compensation:
            if desire_x is None:
                raise ValueError("desire_x is required for actual compensation mode.")
            desire_warped = self.warp(desire_x, flow)
            s_warped = self.warp(s, flow)
            actual_predict = self.compensation_model(desire_warped, s_warped)
            return actual_predict
        else:
            x_warped = self.warp(x, flow)
            s_warped = self.warp(s, flow)
            predict = self.compensation_model(x_warped, s_warped)
            return predict

