import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.Constants as Constants
from transformer.Layers import EncoderLayer
from transformer.mamba import MambaLayer, MambaEncoder

# Import with error handling for cloud environment
try:
    from transformer.mambapy import mamba
except ImportError as e:
    print(f"Import warning in Models.py: {e}")
    import transformer.mambapy.mamba as mamba



def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self, config,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout, model_type):
        super().__init__()

        self.d_model = d_model

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cuda'))

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, model_type, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

        self.model_type = model_type

        self.position_ratio = nn.Parameter(torch.tensor(1.0))
        self.SSM = MambaEncoder(d_model, d_model, False, True, 4, d_model // 16, 16, 2, num_types)
        self.mamba = mamba.Mambadelta(config)

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def temporal_enc_RoPE(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1)# / self.position_vec
        #result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        #result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_type, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        tem_enc_RoPE = self.temporal_enc_RoPE(event_time, non_pad_mask)
        if self.model_type == "Pre" or self.model_type == "RoPE":
            enc_output = self.event_emb(event_type)
        elif self.model_type == 'Mamba_old' or self.model_type == 'Mamba_ro':
            enc_output = self.SSM(event_type)
        elif self.model_type == 'Mamba':
            enc_output = self.event_emb(event_type)
            enc_output = self.mamba(enc_output, event_time)
            #enc_output = self.mamba(enc_output)
        else:
            enc_output = torch.zeros(tem_enc.shape).to('cuda')

        for enc_layer in self.layer_stack:
            if self.model_type == 'Pre' or self.model_type == 'Mamba_mix':
                enc_output += tem_enc
            if self.model_type == 'New':
                enc_output += tem_enc * self.position_ratio
            if self.model_type == 'New':
                enc_output, _ = enc_layer(
                enc_output, tem_enc_RoPE * (1 - self.position_ratio),
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                )
            else:
                enc_output, _ = enc_layer(
                enc_output, tem_enc_RoPE,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                )
        return enc_output


class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data, non_pad_mask):
        out = self.linear(data)
        #out = nn.Softplus(out)
        out = out * non_pad_mask
        return out
    
class Predictor_RoPE(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data, non_pad_mask):
        out = self.linear(data)
        #out = F.softplus(out)
        out = out * non_pad_mask
        return out



class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)
        return out


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            config, num_types, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1, model_type='Pre'):
        super().__init__()

        self.encoder = Encoder(
            config=config,
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            model_type=model_type
        )

        self.num_types = num_types

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_model, d_rnn)
        """
        if model_type == 'Pre':

            # prediction of next time stamp
            self.time_predictor = Predictor(d_model, 1)

            # prediction of next event type
            self.type_predictor = Predictor(d_model, num_types)

        else:
            self.time_predictor = Predictor_RoPE(d_model, 1)

            # prediction of next event type
            self.type_predictor = Predictor_RoPE(d_model, num_types)
        """
        self.time_predictor = Predictor(d_model, 1)

        self.model_type = model_type

            # prediction of next event type
        self.type_predictor = Predictor(d_model, num_types)

        self.SSM = MambaEncoder(d_model // 2, d_model, False, True, 4, d_model // 16, 16, 2, num_types)

    def forward(self, event_type, event_time):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """

        non_pad_mask = get_non_pad_mask(event_type)

        enc_output = self.encoder(event_type, event_time, non_pad_mask)
        #enc_output = self.rnn(enc_output, non_pad_mask)
        #if self.model_type == 'Mamba':
        #    enc_output = self.SSM(enc_output)

        time_prediction = self.time_predictor(enc_output, non_pad_mask)

        type_prediction = self.type_predictor(enc_output, non_pad_mask)

        return enc_output, (type_prediction, time_prediction)
    

class Encoder_pure(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self, config,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout, model_type):
        super().__init__()

        self.d_model = d_model

        # position vector, used for temporal encoding
        
        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)

        #self.layer_stack = nn.ModuleList([
        #    EncoderLayer(d_model, d_inner, n_head, d_k, d_v, model_type, dropout=dropout, normalize_before=False)
        #    for _ in range(n_layers)])

        self.model_type = model_type

        #self.position_ratio = nn.Parameter(torch.tensor(1.0))
        #self.SSM = MambaEncoder(d_model, d_model, False, True, 4, d_model // 16, 16, 2, num_types)
        self.mamba = mamba.Mambadelta(config)


    def forward(self, event_type, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        latest_scaling_factor = None
        #tem_enc = self.temporal_enc(event_time, non_pad_mask)
        #tem_enc_RoPE = self.temporal_enc_RoPE(event_time, non_pad_mask)
        if self.model_type == "Pre" or self.model_type == "RoPE":
            enc_output = self.event_emb(event_type)
        elif self.model_type == 'Mamba_old' or self.model_type == 'Mamba_ro':
            enc_output = self.SSM(event_type)
        elif self.model_type == 'Mamba':
            enc_output = self.event_emb(event_type)
            enc_output, latest_scaling_factor = self.mamba(enc_output, event_time)
        else:
            enc_output = self.event_emb(event_type)
            enc_output, latest_scaling_factor = self.mamba(enc_output, event_time)

        
        return enc_output, latest_scaling_factor

class Mamba_pure(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            config, num_types, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1, model_type='Pre'):
        super().__init__()

        self.encoder = Encoder_pure(
            config=config,
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            model_type=model_type
        )

        self.num_types = num_types

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # OPTIONAL recurrent layer, this sometimes helps
        #self.rnn = RNN_layers(d_model, d_rnn)
        """
        if model_type == 'Pre':

            # prediction of next time stamp
            self.time_predictor = Predictor(d_model, 1)

            # prediction of next event type
            self.type_predictor = Predictor(d_model, num_types)

        else:
            self.time_predictor = Predictor_RoPE(d_model, 1)

            # prediction of next event type
            self.type_predictor = Predictor_RoPE(d_model, num_types)
        """
        self.time_predictor = Predictor(d_model, 1)

        self.model_type = model_type

            # prediction of next event type
        self.type_predictor = Predictor(d_model, num_types)

        #self.SSM = MambaEncoder(d_model // 2, d_model, False, True, 4, d_model // 16, 16, 2, num_types)

    def forward(self, event_type, event_time):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """

        non_pad_mask = get_non_pad_mask(event_type)

        enc_output, latest_scaling_factor = self.encoder(event_type, event_time, non_pad_mask)
        #enc_output = self.rnn(enc_output, non_pad_mask)
        #if self.model_type == 'Mamba':
        #    enc_output = self.SSM(enc_output)

        time_prediction = self.time_predictor(enc_output, non_pad_mask)
        
        # Adjust time prediction by latest scaling factor if available (ablation-aware)
        if latest_scaling_factor is not None:
            # Check if this is a time-scaling ablation mode
            mamba_config = getattr(self.encoder, 'mamba', None)
            if mamba_config and hasattr(mamba_config, 'config'):
                ablation_mode = getattr(mamba_config.config, 'ablation_mode', 'full')
                # Only apply scaling factor division for time-scaling modes
                if ablation_mode in ['time_scaling_only', 'full']:
                    time_prediction = time_prediction / latest_scaling_factor.unsqueeze(1).expand_as(time_prediction)

        type_prediction = self.type_predictor(enc_output, non_pad_mask)

        return enc_output, (type_prediction, time_prediction)