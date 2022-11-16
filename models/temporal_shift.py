import torch
import torch.nn as nn
import torch.nn.functional as F

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
        #print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        if 'TSM' in self.shift_type:
            if self.net.training:
                x = shift(x, self.n_segment, self.shift_type, fold_div=self.fold_div, inplace = self.inplace)
            else:
                #x = batch_shift(x, self.shift_type, fold_div=self.fold_div, enable_past_buffer=self.enable_past_buffer)
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


    # Use batch_shift during validating or testing.
def batch_shift(x, shift_type, fold_div=3, stride=1, enable_past_buffer=True):
    nt, c, h, w = x.size()

    fold = c // fold_div
    
    out = torch.zeros_like(x)
    if not 'toFutureOnly' in shift_type: 
        out[:-stride, :fold] = x[stride:, :fold]  # backward (left) shift
        out[stride:, fold: 2 * fold] = x[:-stride, fold: 2 * fold] # forward (right) shift
        
        if enable_past_buffer:
            # memory-based inference
            if global_queue_buffer.get_batch_index() > 0:
                out[:stride, fold: 2 * fold] = global_queue_buffer.get()
            # Keep stride=1, future_buffer_length is abandened
            global_queue_buffer.put(x[-stride-global_queue_buffer.get_future_buffer_length(), fold: 2 * fold])
    else:
        out[stride:, : 2 * fold] = x[:-stride, : 2 * fold] # forward (right) shift only
        
        if enable_past_buffer:
            # memory-based inference
            if global_queue_buffer.get_batch_index() > 0:
                out[:stride, : 2 * fold] = global_queue_buffer.get()
            global_queue_buffer.put(x[-stride-global_queue_buffer.get_future_buffer_length(), : 2 * fold])
    out[:, 2 * fold:] = x[:, 2 * fold:]  # not shift

    
    return out