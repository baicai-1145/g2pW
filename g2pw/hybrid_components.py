"""
Hybrid BERT-Qwen Components
Enhanced components from Qwen to improve BERT performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """RMSNorm from Qwen - more stable than LayerNorm"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN from Qwen - better than GELU"""
    def __init__(self, hidden_size, intermediate_size=None):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = int(hidden_size * 8 / 3)  # Qwen's ratio
            
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class RotaryPositionalEmbedding(nn.Module):
    """RoPE from Qwen - better positional encoding"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
            
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary positional embedding"""
    if position_ids is None:
        cos = cos[:q.shape[-2], :]
        sin = sin[:q.shape[-2], :]
    else:
        cos = cos[position_ids]
        sin = sin[position_ids]
        
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class HybridBertLayer(nn.Module):
    """Enhanced BERT Layer with Qwen components"""
    def __init__(self, config, use_rope=True, use_rmsnorm=True, use_swiglu=True):
        super().__init__()
        self.config = config
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.use_swiglu = use_swiglu
        
        # Import BERT components
        from transformers.models.bert.modeling_bert import BertSelfAttention, BertSelfOutput
        
        # Attention (keep BERT's proven mechanism)
        self.attention = BertSelfAttention(config)
        self.attention_output = BertSelfOutput(config)
        
        # RoPE if enabled
        if self.use_rope:
            head_dim = config.hidden_size // config.num_attention_heads
            self.rotary_emb = RotaryPositionalEmbedding(head_dim)
        
        # Normalization layers
        if self.use_rmsnorm:
            self.attention_norm = RMSNorm(config.hidden_size)
            self.ffn_norm = RMSNorm(config.hidden_size)
        else:
            self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # FFN
        if self.use_swiglu:
            self.ffn = SwiGLUFFN(config.hidden_size, config.intermediate_size)
        else:
            from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput
            self.intermediate = BertIntermediate(config)
            self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_value=None, output_attentions=False, **kwargs):
        
        # Store residual
        residual = hidden_states
        
        # Pre-norm (Qwen style) vs Post-norm (BERT style)
        if self.use_rmsnorm:
            # Pre-norm
            normed_hidden_states = self.attention_norm(hidden_states)
        else:
            # Post-norm (keep BERT's original behavior)
            normed_hidden_states = hidden_states
        
        # Apply RoPE if enabled
        if self.use_rope:
            # This is a simplified RoPE integration
            # In practice, we'd need to modify the attention mechanism more deeply
            pass  # For now, keep BERT's attention as-is
        
        # Self-attention
        self_attention_outputs = self.attention(
            normed_hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self_attention_outputs[0]
        
        # Attention output projection
        attention_output = self.attention_output(attention_output, residual)
        
        # Store for FFN residual
        ffn_residual = attention_output
        
        # FFN with normalization
        if self.use_rmsnorm:
            # Pre-norm
            normed_attention_output = self.ffn_norm(attention_output)
        else:
            # Post-norm will be handled in the FFN layers
            normed_attention_output = attention_output
        
        # FFN forward
        if self.use_swiglu:
            ffn_output = self.ffn(normed_attention_output)
            # Manual residual connection for SwiGLU
            layer_output = ffn_residual + ffn_output
        else:
            # Use BERT's original FFN
            intermediate_output = self.intermediate(normed_attention_output)
            layer_output = self.output(intermediate_output, attention_output)
        
        outputs = (layer_output,) + self_attention_outputs[1:]  # add attentions if we output them
        return outputs


class HybridBertEncoder(nn.Module):
    """BERT Encoder with hybrid layers"""
    def __init__(self, config, use_rope=True, use_rmsnorm=True, use_swiglu=True):
        super().__init__()
        self.config = config
        
        # Create hybrid layers
        self.layer = nn.ModuleList([
            HybridBertLayer(config, use_rope, use_rmsnorm, use_swiglu) 
            for _ in range(config.num_hidden_layers)
        ])
        
        self.gradient_checkpointing = False

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_values=None, use_cache=None, output_attentions=False,
                output_hidden_states=False, return_dict=True, **kwargs):
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        
        from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
