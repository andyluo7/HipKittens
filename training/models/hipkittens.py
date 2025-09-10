import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.autograd import Function

import tk_kernel_fwd
import tk_kernel_bkwd


class HipKittensFlashAttnFn(Function):
    """
    Inputs/outputs are BNHD (batch, seq, heads, dim), like your harness.
    Forward:  O, L  via tk_kernel_fwd.dispatch_fwd
    Backward: dQ,dK,dV via tk_kernel_bkwd.{dispatch_prep,dispatch_bwd_combined,dispatch_dq_shuffle}
    Compute in bf16, save L and O for backward, return O in input dtype.
    """

    @staticmethod
    def forward(ctx, q_bnhd: torch.Tensor, k_bnhd: torch.Tensor, v_bnhd: torch.Tensor):
        assert q_bnhd.dim() == 4 and k_bnhd.shape == q_bnhd.shape and v_bnhd.shape == q_bnhd.shape, \
            "Expected q,k,v as [B, N, H, D] (BNHD) with matching shapes."
        B, N, H, D = q_bnhd.shape
        dev = q_bnhd.device
        out_dtype = q_bnhd.dtype  # remember caller dtype

        # Cast to bf16 for kernels (compute) and make contiguous
        q = q_bnhd.to(torch.bfloat16).contiguous()
        k = k_bnhd.to(torch.bfloat16).contiguous()
        v = v_bnhd.to(torch.bfloat16).contiguous()

        # Allocate outputs for the kernels
        O = torch.empty((B, N, H, D), dtype=torch.bfloat16, device=dev).contiguous()    # BNHD
        L = torch.empty((B, H, N, 1), dtype=torch.float32,  device=dev).contiguous()    # BHN1 (matches your harness)

        # Forward kernel
        tk_kernel_fwd.dispatch_fwd(q, k, v, O, L)

        # Save for backward
        ctx.save_for_backward(q, k, v, O, L)
        # Return in caller dtype
        return O.to(out_dtype)

    @staticmethod
    def backward(ctx, dO_bnhd: torch.Tensor):
        q, k, v, O, L = ctx.saved_tensors
        B, N, H, D = O.shape
        dev = dO_bnhd.device

        # Cast grad to bf16 for kernels
        dO = dO_bnhd.to(torch.bfloat16).contiguous()

        # Allocate grads and workspaces
        dQ_in = torch.empty((B, H, N, D), dtype=torch.bfloat16, device=dev).contiguous()  # BHND (pre-shuffle)
        dQ    = torch.empty((B, N, H, D), dtype=torch.bfloat16, device=dev).contiguous()  # BNHD
        dK    = torch.empty((B, N, H, D), dtype=torch.bfloat16, device=dev).contiguous()  # BNHD
        dV    = torch.empty((B, N, H, D), dtype=torch.bfloat16, device=dev).contiguous()  # BNHD
        delta = torch.empty((B, H, 1, N), dtype=torch.float32,  device=dev).contiguous()  # BH1N (matches your harness)

        # Backward kernels
        tk_kernel_bkwd.dispatch_prep(O, dO, delta)
        tk_kernel_bkwd.dispatch_bwd_combined(q, k, v, O, dO, dQ_in, dK, dV, L, delta)
        tk_kernel_bkwd.dispatch_dq_shuffle(dQ_in, dQ)
        return dQ.to(dO_bnhd.dtype), dK.to(dO_bnhd.dtype), dV.to(dO_bnhd.dtype)


class HipKittensBertSelfAttention(nn.Module):
    """
    Uses HipKittensFlashAttnFn when there is NO padding.
    Falls back to MHA-style expansion if num_key_value_heads < num_attention_heads (GQA).
    Expects HF additive mask: [B,1,1,N] (0 keep, -inf mask)
    """
    def __init__(self, config, layer_idx=None, deterministic: bool = False):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError("hidden_size must be multiple of num_attention_heads")
        
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads                  # h_q
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_attention_heads)  # h_kv
        self.attention_head_size = config.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Linear layers for Q, K, V
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.num_key_value_heads * self.attention_head_size)
        self.value = nn.Linear(config.hidden_size, self.num_key_value_heads * self.attention_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.deterministic = deterministic
        self.is_causal = False

        print(f"HipKittens BertSelfAttention layer {layer_idx}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[tuple] = None,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = hidden_states.shape
        H = self.num_attention_heads
        HKV = self.num_key_value_heads
        D = self.attention_head_size

        q = self.query(hidden_states).view(B, N, H, D).to(torch.bfloat16).contiguous()
        k = self.key(hidden_states).view(B, N, HKV, D).to(torch.bfloat16).contiguous()
        v = self.value(hidden_states).view(B, N, HKV, D).to(torch.bfloat16).contiguous()

        if HKV != H:
            group_size = H // HKV
            k = k.unsqueeze(2).expand(B, N, group_size, HKV, D).reshape(B, N, H, D)
            v = v.unsqueeze(2).expand(B, N, group_size, HKV, D).reshape(B, N, H, D)

        out_bnhd = HipKittensFlashAttnFn.apply(q, k, v)  # BNHD
        ctx = out_bnhd.to(q.dtype).contiguous().view(B, N, H * D)
        return ctx, None

