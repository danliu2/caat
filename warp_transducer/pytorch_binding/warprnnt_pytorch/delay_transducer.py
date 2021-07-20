import torch
import warprnnt_pytorch as warp_rnnt
from torch.autograd import Function
from torch.nn import Module

from .warp_rnnt import *

from .rnnt import check_type, check_contiguous, check_dim

def certify_inputs(log_probs, labels, lengths, label_lengths,delay_values):
    # check_type(log_probs, torch.float32, "log_probs")
    check_type(labels, torch.int32, "labels")
    check_type(label_lengths, torch.int32, "label_lengths")
    check_type(lengths, torch.int32, "lengths")
    check_type(delay_values, torch.float32, "delay_values")
    check_contiguous(log_probs, "log_probs")
    check_contiguous(labels, "labels")
    check_contiguous(label_lengths, "label_lengths")
    check_contiguous(lengths, "lengths")
    check_contiguous(delay_values, "delay_values")

    if lengths.shape[0] != log_probs.shape[0]:
        raise ValueError("must have a length per example.")
    if label_lengths.shape[0] != log_probs.shape[0]:
        raise ValueError("must have a label length per example.")

    check_dim(log_probs, 4, "log_probs")
    check_dim(labels, 2, "labels")
    check_dim(lengths, 1, "lenghts")
    check_dim(label_lengths, 1, "label_lenghts")
    check_dim(delay_values, 3, "delay_values")
    max_T = torch.max(lengths)
    max_U = torch.max(label_lengths)
    T, U = log_probs.shape[1:3]
    # we split one batch into pieces, max length of each piece may be less than batch length
    """ if T != max_T:
        raise ValueError("Input length mismatch")
    if U != max_U + 1:
        raise ValueError("Output length mismatch") """

class _DelayTransducer(Function):
    @staticmethod
    def forward(ctx, acts, labels, act_lens, label_lens,delay_values,delay_scale, blank, temperature, reduction):
        """
        acts: Tensor of (batch x seqLength x labelLength x outputDim) containing output from network
        labels: 2 dimensional Tensor containing all the targets of the batch with zero padded
        act_lens: Tensor of size (batch) containing size of each output sequence from the network
        label_lens: Tensor of (batch) containing label length of each example
        """
        is_cuda = acts.is_cuda
        if not is_cuda:
            raise NotImplementedError("only gpu version now")

        certify_inputs(acts, labels, act_lens, label_lens, delay_values)

        loss_func = warp_rnnt.gpu_rnnt_delay
        grads = torch.zeros_like(acts) if acts.requires_grad else torch.zeros(0).to(acts)
        minibatch_size = acts.size(0)
        costs = torch.zeros(3, minibatch_size, dtype=acts.dtype)
        loss_func(acts,
                  labels,
                  act_lens,
                  label_lens,
                  delay_values,
                  costs,
                  grads,
                  delay_scale,
                  temperature,
                  blank,
                  0)
        costs = costs.to(acts.device)
        loss_rnnt= costs[0]
        loss_delay = costs[1]
        loss_total= costs[2]
        
        if reduction in ['sum', 'mean']:
            loss_rnnt= loss_rnnt.sum()
            loss_delay= loss_delay.sum()
            loss_total= loss_total.sum()
            if reduction == 'mean':
                loss_rnnt /= minibatch_size
                loss_delay /=minibatch_size
                loss_total /= minibatch_size
                grads /= minibatch_size

        
        ctx.grads = grads

        return loss_total, loss_rnnt,loss_delay

    @staticmethod
    def backward(ctx, grad_output, g2,g3):
        grad_output = grad_output.view(-1, 1, 1, 1).to(ctx.grads)
        
        return ctx.grads.mul_(grad_output), None, None, None, None, None, None,None, None


def delay_transducer_loss(
    acts, labels, act_lens, label_lens, delay_values, 
    delay_scale=1.0, 
    temperature=1.0, blank=0, reduction='sum'
):
    return _DelayTransducer.apply(acts, labels, act_lens, label_lens,delay_values, delay_scale, blank,temperature, reduction)


def delay_cost_zero(acts, src_lens, tgt_lens,):
    B, S,T = acts.shape[:3]
    delay_values= torch.arange(S).unsqueeze(0).repeat(B,1).contiguous().to(acts)
    delay_values= delay_values/(src_lens.unsqueeze(1).to(acts))
    delay_values= delay_values.unsqueeze(2).repeat(1,1, T)
    return delay_values

def delay_cost_diag_positive(acts, src_lens, tgt_lens):
    B,S,T = acts.shape[:3]
    src_lens = src_lens.float()
    tgt_lens= tgt_lens.float()
    gamma = tgt_lens/src_lens
    #BxT
    tgt_pos= torch.arange(T).repeat(B,1).contiguous().to(acts)
    # BxS
    src_pos = torch.arange(S).repeat(B,1).contiguous().to(acts)
    delay_values= ((src_pos+1)*gamma.unsqueeze(1)).unsqueeze(2) - (tgt_pos.unsqueeze(1)+1)
    delay_values= delay_values.clamp(0,) 
    delay_values = delay_values/ tgt_lens.view(-1,1,1)
    return delay_values

def delay_cost_diagonal(acts, src_lens, tgt_lens,):
    """
        have diagonal as golden, cost as diff from diagonal
    """
    B,S,T = acts.shape[:3]
    src_lens = src_lens.float()
    tgt_lens= tgt_lens.float()
    gamma = tgt_lens/src_lens
    #BxT
    tgt_pos= torch.arange(T).repeat(B,1).contiguous().to(acts)
    # BxS
    src_pos = torch.arange(S).repeat(B,1).contiguous().to(acts)
    delay_values= ((src_pos+1)*gamma.unsqueeze(1)).unsqueeze(2) - (tgt_pos.unsqueeze(1)+1)
    delay_values= torch.abs(delay_values)
    delay_values = delay_values/ tgt_lens.view(-1,1,1)
    return delay_values


class DelayTLoss(Module):
    """
    Parameters:
        blank (int, optional): blank label. Default: 0.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 
            'mean': the output losses will be divided by the target lengths and
            then the mean over the batch is taken. Default: 'sum'
        delay_func: may be one of "zero", "diagonal","diag_positive"
    """
    def __init__(self, blank=0, delay_scale=1.0, temperature=1.0, reduction='sum', delay_func="zero"):
        super(DelayTLoss, self).__init__()
        self.delay_scale= delay_scale
        self.blank = blank
        self.reduction = reduction
        self.temperature=temperature
        self.loss = _DelayTransducer.apply
        delay_funcs={
            "zero":delay_cost_zero,
            "diagonal":delay_cost_diagonal,
            "diag_positive":delay_cost_diag_positive,
        }
        if delay_func not in delay_funcs:
            raise NotImplementedError(f"{delay_func} not implemented")
        self.delay_func= delay_funcs[delay_func]

    def forward(self, acts, labels, act_lens, label_lens):
        bsz, maxT= acts.shape[:2]
        with torch.no_grad():
            delay_values= self.delay_func(acts, act_lens, label_lens)
        return self.loss(acts, labels, act_lens, label_lens,delay_values, self.delay_scale, self.blank,self.temperature, self.reduction)

