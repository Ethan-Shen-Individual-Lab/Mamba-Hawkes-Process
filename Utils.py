import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Constants
from transformer.Models import get_non_pad_mask


def get_per_sequence_counts(event_type):
    """Per-sequence event / prediction counts, shape (B,)."""
    non_pad = event_type.ne(Constants.PAD).float()
    num_events = non_pad.sum(dim=1).clamp(min=1)
    num_type_pred = (num_events - 1).clamp(min=1)
    num_time_gaps = (non_pad[:, 1:] * non_pad[:, :-1]).sum(dim=1).clamp(min=1)
    return num_events, num_type_pred, num_time_gaps


def batch_invariant_event_loss(event_ll, non_event_ll, event_type):
    """Per-sequence mean NLL, averaged over the batch."""
    num_events, _, _ = get_per_sequence_counts(event_type)
    per_seq = -(event_ll - non_event_ll) / num_events
    return per_seq.mean()


def batch_invariant_type_loss(prediction, types, loss_func, event_type):
    """Per-sequence mean type loss, averaged over the batch."""
    truth = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]

    if isinstance(loss_func, LabelSmoothingLoss):
        loss = loss_func(prediction, truth)
    else:
        loss = loss_func(prediction.transpose(1, 2), truth)

    _, num_type_pred, _ = get_per_sequence_counts(event_type)
    per_seq = loss.sum(dim=1) / num_type_pred
    return per_seq.mean()


def batch_invariant_time_loss(prediction, event_time, event_type, scaling_factor=None):
    """Per-sequence mean time loss, averaged over the batch."""
    prediction = prediction.squeeze(-1).clone()
    true = event_time[:, 1:] - event_time[:, :-1]
    prediction = prediction[:, :-1]

    if scaling_factor is not None:
        prediction = prediction / scaling_factor.unsqueeze(1).expand_as(prediction)

    diff = prediction - true
    non_pad = event_type.ne(Constants.PAD).float()
    valid = non_pad[:, 1:] * non_pad[:, :-1]
    se = diff * diff * valid
    _, _, num_time_gaps = get_per_sequence_counts(event_type)
    per_seq = se.sum(dim=1) / num_time_gaps
    return per_seq.mean()


def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))


def compute_event(event, non_pad_mask):
    """ Log-likelihood of events. """

    # add 1e-9 in case some events have 0 likelihood
    event += math.pow(10, -9)
    event.masked_fill_(~non_pad_mask.bool(), 1.0)

    result = torch.log(event)
    return result


def compute_integral_biased(all_lambda, time, non_pad_mask):
    """ Log-likelihood of non-events, using linear interpolation. """

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    diff_lambda = (all_lambda[:, 1:] + all_lambda[:, :-1]) * non_pad_mask[:, 1:]

    biased_integral = diff_lambda * diff_time
    result = 0.5 * biased_integral
    return result


def compute_integral_unbiased(model, data, time, non_pad_mask, type_mask, model_type):
    """ Log-likelihood of non-events, using Monte Carlo integration. """

    num_samples = 100

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    temp_time = diff_time.unsqueeze(2) * \
                torch.rand([*diff_time.size(), num_samples], device=data.device)
    if model_type == 'Pre' or model_type == 'Mamba_mix' or model_type == 'Mamba_mm':
        temp_time /= (time[:, :-1] + 1).unsqueeze(2)
    if model_type == 'RoPE_linear':
        temp_time /= (time[:, :-1] - time[:, 0] + 1).unsqueeze(2)

    temp_hid = model.linear(data)[:, 1:, :]
    temp_hid = torch.sum(temp_hid * type_mask[:, 1:, :], dim=2, keepdim=True)

    all_lambda = softplus(temp_hid + model.alpha * temp_time, model.beta)
    all_lambda = torch.sum(all_lambda, dim=2) / num_samples

    unbiased_integral = all_lambda * diff_time
    return unbiased_integral


def log_likelihood(model, data, time, types, model_type):
    """ Log-likelihood of sequence. """

    non_pad_mask = get_non_pad_mask(types).squeeze(2)

    type_mask = torch.zeros([*types.size(), model.num_types], device=data.device)
    for i in range(model.num_types):
        type_mask[:, :, i] = (types == i + 1).bool().to(data.device)

    all_hid = model.linear(data)
    all_lambda = softplus(all_hid, model.beta)
    type_lambda = torch.sum(all_lambda * type_mask, dim=2)

    # event log-likelihood
    event_ll = compute_event(type_lambda, non_pad_mask)
    event_ll = torch.sum(event_ll, dim=-1)

    # non-event log-likelihood, either numerical integration or MC integration
    #non_event_ll = compute_integral_biased(type_lambda, time, non_pad_mask)
    non_event_ll = compute_integral_unbiased(model, data, time, non_pad_mask, type_mask, model_type)
    non_event_ll = torch.sum(non_event_ll, dim=-1)

    return event_ll, non_event_ll


def type_loss(prediction, types, loss_func):
    """ Event prediction loss, cross entropy or label smoothing. """

    # convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
    truth = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]

    pred_type = torch.max(prediction, dim=-1)[1]
    correct_num = torch.sum(pred_type == truth)

    # compute cross entropy loss
    if isinstance(loss_func, LabelSmoothingLoss):
        loss = loss_func(prediction, truth)
    else:
        loss = loss_func(prediction.transpose(1, 2), truth)

    loss = torch.sum(loss)
    return loss, correct_num



def time_loss(prediction, event_time, event_type=None, scaling_factor=None):
    """ Time prediction loss over valid inter-event gaps only. """

    prediction.squeeze_(-1)

    true = event_time[:, 1:] - event_time[:, :-1]
    prediction = prediction[:, :-1]
    
    # If scaling_factor is provided, adjust the prediction
    # The model outputs scaled time intervals, so we need to unscale them
    if scaling_factor is not None:
        prediction = prediction / scaling_factor.unsqueeze(1).expand_as(prediction)

    diff = prediction - true
    if event_type is not None:
        non_pad = event_type.ne(Constants.PAD).float()
        valid = non_pad[:, 1:] * non_pad[:, :-1]
        diff = diff * valid

    se = torch.sum(diff * diff)
    return se

def rmse_loss(prediction, event_time):
    """ Time prediction loss. """

    prediction.squeeze_(-1)

    true = event_time[:, 1:] - event_time[:, 0].unsqueeze(1)
    prediction = prediction[:, :-1]

    # event time gap prediction
    diff = prediction - true
    se = torch.mean(diff * diff)
    return torch.sqrt(se)

def RMSE_loss(prediction, event_time, event_type=None):
    """ Time prediction RMSE over valid positions only. """

    prediction.squeeze_(-1)

    true = event_time[:, 1:] - event_time[:, 0].unsqueeze(1)
    prediction = torch.cumsum(prediction[:, :-1], dim=-1)

    diff = prediction - true
    if event_type is not None:
        non_pad = event_type.ne(Constants.PAD).float()
        valid = non_pad[:, 1:]
        diff = diff * valid
        se = torch.sum(diff * diff) / valid.sum().clamp(min=1)
    else:
        se = torch.mean(diff * diff)
    return torch.sqrt(se)

class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        output (FloatTensor): (batch_size) x n_classes
        target (LongTensor): batch_size
        """

        non_pad_mask = target.ne(self.ignore_index).float()

        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes

        log_prb = F.log_softmax(output, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss
