import numpy as np
import mindspore
from mindspore import nn, Tensor, ops, context
import mindspore.dataset as ds
from mindspore.common.initializer import HeNormal

class Hook():
    def __init__(self, module, idx, backward=False):
        self.idx = idx
        if not backward:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    
    def close(self):
        self.hook.remove()

def compute_entropy_1st(modelx, train_loader, hookF, classes):
    H = [None] * len(hookF)
    H_classwise = [None] * len(hookF)
    P = [None] * len(hookF)
    N = [None] * len(hookF)
    M = Tensor(np.zeros(classes), mindspore.float32)
    
    for idx in range(len(hookF)):
        H_classwise[idx] = Tensor(np.zeros((np.prod(hookF[idx].output.shape[1:]), classes)), mindspore.float32)
        P[idx] = Tensor(np.zeros((np.prod(hookF[idx].output.shape[1:]), classes)), mindspore.float32)
        N[idx] = Tensor(np.zeros((np.prod(hookF[idx].output.shape[1:]), classes)), mindspore.float32)
    
    for data in train_loader.create_dict_iterator():
        xb = data['image']
        yb = ops.Cast()(data['label'], mindspore.float32)
        for this_idx_1 in range(classes):
            condition = (yb == this_idx_1)
            condition = ops.Cast()(condition, mindspore.float32)
            M[this_idx_1] += ops.ReduceSum()(condition)
        
        modelx(xb)
        for idx in range(len(hookF)):
            for this_idx_1 in range(classes):
                this_yb_1 = (yb == this_idx_1).astype(mindspore.float32).expand_dims(axis=1)
                P[idx][:, this_idx_1] += ops.ReduceSum()((hookF[idx].output.view(this_yb_1.shape[0], -1) > 0).astype(mindspore.float32) * this_yb_1, 0)
                N[idx][:, this_idx_1] += ops.ReduceSum()((hookF[idx].output.view(this_yb_1.shape[0], -1) <= 0).astype(mindspore.float32) * this_yb_1, 0)
    
    for idx in range(len(hookF)):
        P[idx] = ops.clip_by_value(P[idx] / M.expand_dims(0), Tensor(0.0001, mindspore.float32), Tensor(0.9999, mindspore.float32))
        N[idx] = ops.clip_by_value(N[idx] / M.expand_dims(0), Tensor(0.0001, mindspore.float32), Tensor(0.9999, mindspore.float32))
        for this_idx_1 in range(classes):
            H_classwise[idx][:, this_idx_1] -= P[idx][:, this_idx_1] * ops.log2(P[idx][:, this_idx_1]) + (N[idx][:, this_idx_1] * ops.log2(N[idx][:, this_idx_1]))
        H[idx] = ops.ReduceSum()(H_classwise[idx], 1)
    
    return H, H_classwise

def compute_PSPentropy(model, dataset, order=1, classes=10):
    hookF = [Hook(layer, idx) for idx, layer in enumerate(model.cells()) if isinstance(layer, (nn.Conv2d, nn.Dense))]
    
    for data in dataset.create_dict_iterator():
        inputs = data['image']
        model(inputs)
        break
    
    if order == 1:
        H, H_classwise = compute_entropy_1st(model, dataset, hookF, classes)
        for h in hookF:
            h.close()
        return H, H_classwise
    else:
        raise NotImplementedError("Higher order entropy computation not yet implemented")
