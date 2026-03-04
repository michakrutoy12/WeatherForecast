import torch
import torch.nn as nn
import numpy as np

class ProbAttention(nn.Module):
    def __init__(self, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape

        V_sum = V.mean(dim=-2)
        contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()

        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        attn = torch.softmax(scores, dim=-1)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        scale = self.scale or 1./np.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale


        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn

class InformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, factor):
        super().__init__()
        self.attn = ProbAttention(factor=factor)

        self.n_heads = n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, L, D = x.shape
        D_head = D // self.n_heads
        
        q = self.q_proj(x).view(B, L, self.n_heads, D_head)
        k = self.k_proj(x).view(B, L, self.n_heads, D_head)
        v = self.v_proj(x).view(B, L, self.n_heads, D_head)
        
        new_x, _ = self.attn(q, k, v, attn_mask=None)
        new_x = new_x.reshape(B, L, -1)
        
        x = self.norm1(x + self.dropout(new_x))
        
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class LSTMInformer(nn.Module):
    def __init__(self, enc_in, c_out, d_model=256, n_heads=8, n_layers=3, factor=5, out_len=30):
        super().__init__()
        self.out_len = out_len
        
        self.lstm = nn.LSTM(enc_in, d_model, num_layers=2, batch_first=True, dropout=0.1)
        
        self.layers = nn.ModuleList([
            InformerBlock(d_model, n_heads, factor) for _ in range(n_layers)
        ])
        
        self.projection = nn.Linear(d_model, c_out)

    def forward(self, x):
        x, _ = self.lstm(x)
        
        for layer in self.layers:
            x = layer(x)
            
        return self.projection(x[:, -self.out_len:, :])