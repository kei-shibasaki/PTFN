import torch
import torch.nn as nn
import numpy as np

class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, shift_type='TSM', inplace=False, enable_past_buffer=True,
                 **kwargs):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.shift_type = shift_type
        self.inplace = inplace
        self.enable_past_buffer = enable_past_buffer
        # print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        if 'TSM' in self.shift_type:
            x = shift(x, self.n_segment, self.shift_type, fold_div=self.fold_div, inplace = self.inplace)

        return self.net(x)

def shift(x, n_segment, shift_type, fold_div=3, stride=1, inplace=False):
    nt, c, h, w = x.size()
    n_batch = nt // n_segment
    x = x.view(n_batch, n_segment, c, h, w)

    fold = c // fold_div # 32/8 = 4

    if inplace:
        # Due to some out of order error when performing parallel computing. 
        # May need to write a CUDA kernel.
        print("WARNING: use inplace shift. it has bugs")
        raise NotImplementedError  
        
    else:
        out = torch.zeros_like(x)
        if not 'toFutureOnly' in shift_type:
            out[:, :-stride, :fold] = x[:, stride:, :fold]  # backward (left shift)
            out[:, stride:, fold: 2 * fold] = x[:, :-stride, fold: 2 * fold]  # forward (right shift)
        else:
            out[:, stride:, : 2 * fold] = x[:, :-stride, : 2 * fold] # right shift only
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

    return out.view(nt, c, h, w)

class CvBlock(nn.Module):
    '''(Conv2d => BN => ReLU) x 2'''

    def __init__(self, in_ch, out_ch, norm='bn', bias=True, act='relu'):
        super(CvBlock, self).__init__()
        norm_fn = get_norm_function(norm)
        act_fn = get_act_function(act)
        self.c1 = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                            padding=1, bias=bias)
        self.b1 = norm_fn(out_ch)
        self.relu1 = act_fn(inplace=True)
        self.c2 = nn.Conv2d(out_ch, out_ch, kernel_size=3,
                            padding=1, bias=bias)
        self.b2 = norm_fn(out_ch)
        self.relu2 = act_fn(inplace=True)

    def forward(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.relu1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.relu2(x)
        return x

def get_norm_function(norm):
    if norm == "bn":
        norm_fn = nn.BatchNorm2d
    elif norm == "in":
        norm_fn = nn.InstanceNorm2d
    elif norm == 'none':
        norm_fn =nn.Identity
    else:
        norm_fn =nn.Identity
    return norm_fn

def get_act_function(act):
    if act == "relu":
        act_fn = nn.ReLU
    elif act == "relu6":
        act_fn = nn.ReLU6
    elif act == 'none':
        act_fn =nn.Identity
    return act_fn

class InputCvBlock(nn.Module):
    '''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''

    def __init__(self, num_in_frames, out_ch, in_ch=4, norm='bn', bias=True, act='relu', interm_ch = 30, blind=False):
    # def __init__(self, num_in_frames, out_ch, in_ch=4, norm='bn', bias=True, act='relu', blind=False):
        super(InputCvBlock, self).__init__()
        self.interm_ch = interm_ch
        norm_fn = get_norm_function(norm)
        act_fn = get_act_function(act)
        if blind:
            in_ch = 3
        self.convblock = nn.Sequential(
            nn.Conv2d(num_in_frames*in_ch, num_in_frames*self.interm_ch,
                      kernel_size=3, padding=1, groups=num_in_frames, bias=bias),
            norm_fn(num_in_frames*self.interm_ch),
            act_fn(inplace=True),
            nn.Conv2d(num_in_frames*self.interm_ch, out_ch,
                      kernel_size=3, padding=1, bias=bias),
            norm_fn(out_ch),
            act_fn(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)

class DownBlock(nn.Module):
    '''Downscale + (Conv2d => BN => ReLU)*2'''

    def __init__(self, in_ch, out_ch, norm='bn', bias=True, act='relu'):
        super(DownBlock, self).__init__()
        norm_fn = get_norm_function(norm)
        act_fn = get_act_function(act)
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      padding=1, stride=2, bias=bias),
            norm_fn(out_ch),
            act_fn(inplace=True),
            CvBlock(out_ch, out_ch, norm=norm, bias=bias, act=act)
        )

    def forward(self, x):
        return self.convblock(x)


class UpBlock(nn.Module):
    '''(Conv2d => BN => ReLU)*2 + Upscale'''

    def __init__(self, in_ch, out_ch, norm='bn', bias=True, act='relu'):
        super(UpBlock, self).__init__()
        self.convblock = nn.Sequential(
            CvBlock(in_ch, in_ch, norm=norm, bias=bias, act=act),
            nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=bias),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.convblock(x)


class OutputCvBlock(nn.Module):
    '''Conv2d => BN => ReLU => Conv2d'''

    def __init__(self, in_ch, out_ch, norm='bn', bias=True, act='relu'):
        super(OutputCvBlock, self).__init__()
        norm_fn = get_norm_function(norm)
        act_fn = get_act_function(act)
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=bias),
            norm_fn(in_ch),
            act_fn(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=bias)
        )

    def forward(self, x):
        return self.convblock(x)


class DenBlock(nn.Module):
    """ Definition of the denosing block of FastDVDnet.
    Inputs of constructor:
        num_input_frames: int. number of input frames
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
        noise_map: array with noise map of dim [N, 1, H, W]
    """

    def __init__(self, chns=[32, 64, 128], out_ch=3, in_ch=4, shift_input=False, norm='bn', bias=True,  act='relu', interm_ch=30, blind=False):
    # def __init__(self, chns=[32, 64, 128], out_ch=3, in_ch=4, shift_input=False, norm='bn', bias=True,  act='relu', blind=False):
        super(DenBlock, self).__init__()
        self.chs_lyr0, self.chs_lyr1, self.chs_lyr2 = chns
        
        # if stage2: in_ch=3
        if shift_input:
            self.inc = CvBlock(in_ch=in_ch, out_ch=self.chs_lyr0, norm=norm, bias=bias, act=act)
        else:
            self.inc = InputCvBlock(
                num_in_frames=1, out_ch=self.chs_lyr0, in_ch=in_ch, norm=norm, bias=bias, act=act, interm_ch=interm_ch, blind=blind)
                # num_in_frames=1, out_ch=self.chs_lyr0, in_ch=in_ch, norm=norm, bias=bias, act=act, blind=blind)
        
        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1, norm=norm, bias=bias, act=act)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2, norm=norm, bias=bias, act=act)
        self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1, norm=norm, bias=bias,    act=act)
        self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0, norm=norm, bias=bias,    act=act)
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=out_ch, norm=norm, bias=bias,     act=act)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, in1):
        '''Args:
            inX: Tensor, [N, C, H, W] in the [0., 1.] range
            noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
        '''
        # Input convolution block
        x0 = self.inc(in1)
        # Downsampling
        x1 = self.downc0(x0)
        x2 = self.downc1(x1)
        # Upsampling
        x2 = self.upc2(x2)
        x1 = self.upc1(x1+x2)
        # Estimation
        x = self.outc(x0+x1)

        # Residual
        x[:, :3, :, :] = in1[:, :3, :, :] - x[:, :3, :, :]

        return x

class WNet(nn.Module):
    def __init__(self, chns=[64, 128, 256], mid_ch=64, shift_input=False, stage_num=2, in_ch=4, out_ch=3, norm='none', act='relu6', interm_ch=64, blind=False):
        super(WNet, self).__init__()
        
        self.stage_num = stage_num
        self.nets_list = nn.ModuleList()
        for i in np.arange(stage_num):
            if i == 0:
                stage_in_ch = in_ch
            else:
                stage_in_ch = mid_ch
            if i == (stage_num-1):
                stage_out_ch = out_ch
            else:
                stage_out_ch = mid_ch
                
            # self.nets_list.append(DenBlock(chns=chns, out_ch=stage_out_ch, in_ch=stage_in_ch, shift_input=shift_input, norm=norm, act=act, interm_ch=interm_ch))
            
            if i == 0:
                self.nets_list.append(DenBlock(chns=chns, out_ch=stage_out_ch, in_ch=stage_in_ch, shift_input=shift_input, norm=norm, act=act, blind=blind, interm_ch=interm_ch))
            else:
                self.nets_list.append(DenBlock(chns=chns, out_ch=stage_out_ch,
                                           in_ch=stage_in_ch, shift_input=shift_input, norm=norm, act=act, interm_ch=interm_ch))

        # Init weights
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        for i in np.arange(self.stage_num):
            x = self.nets_list[i](x)
        return x

class TSN(nn.Module):
    """
        Temporal-shift denoiser during training
    """
    def __init__(self, 
                 num_segments=11,
                 base_model='WNet_multistage', 
                 shift_type='TSM', 
                 shift_div=8,
                 inplace=False,
                 net2d_opt={},
                 enable_past_buffer=True,
                 **kwargs):

        super(TSN, self).__init__()

        self.reshape = True
        self.num_segments = num_segments
        self.shift_type = shift_type
        self.shift_div = shift_div
        self.base_model_name = base_model
        self.net2d_opt = net2d_opt
        self.enable_past_buffer = enable_past_buffer        
        self.inplace = inplace
        self._prepare_base_model(base_model)

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))
        ShiftOutputCvBlock = int

        if base_model == 'WNet_multistage':
            self.base_model = WNet(**self.net2d_opt)

        else:
            print('No such model')
            raise NotImplementedError

        if self.shift_type != 'no_temporal_shift':
            for m in self.base_model.modules():
                if isinstance(m, CvBlock) or isinstance(m, ShiftOutputCvBlock):
                    #print('Adding temporal shift... {} {}'.format(m.c1, m.c2))
                    m.c1 = TemporalShift(m.c1, self.num_segments, self.shift_div, self.shift_type,
                                         inplace=self.inplace, enable_past_buffer=self.enable_past_buffer)
                    m.c2 = TemporalShift(m.c2, self.num_segments, self.shift_div, self.shift_type,
                                         inplace=self.inplace, enable_past_buffer=self.enable_past_buffer)

    def forward(self, input, noise_map=None):
        # N, F, C, H, W -> (N*F, C, H, W)
        if noise_map != None:
            input = torch.cat([input, noise_map], dim=2)
        if len(input.shape)==5:
            N, F, C, H, W = input.shape
            model_input = input.reshape(N*F, C, H, W)
        else:
            model_input = input
        base_out = self.base_model(model_input)
        if len(input.shape)==5:
            NF, C, H, W = base_out.shape
            base_out = base_out.reshape(N, F, C, H, W)
        return base_out

