import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Optional

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:xq_.shape[1], :]
    xq_out = torch.view_as_real(xq_ * freqs_cis.unsqueeze(0)).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis.unsqueeze(0)).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class MsingiConfig:
    def __init__(
        self,
        vocab_size=50000,
        max_position_embeddings=2048,  # Increased for longer context
        hidden_size=768,  # Increased for better capacity
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        gradient_checkpointing=False,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.gradient_checkpointing = gradient_checkpointing

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def forward(self, hidden_states, freqs_cis, attention_mask=None):
        batch_size, seq_length = hidden_states.shape[:2]
        
        q = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        
        # Apply rotary embeddings
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        
        # Efficient attention computation
        q = q.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Flash attention-like computation (simplified version)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        if attention_mask is not None:
            scores = scores + attention_mask
            
        attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(q)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        
        return self.out_proj(output)

class MsingiBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(approximate='tanh'),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
    def forward(self, hidden_states, freqs_cis, attention_mask=None):
        # Pre-norm architecture
        attn_output = self.attention(self.ln1(hidden_states), freqs_cis, attention_mask)
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.ln2(hidden_states))
        return hidden_states

class Msingi1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MsingiBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Compute rotary position embeddings
        self.freqs_cis = precompute_freqs_cis(
            self.config.hidden_size // self.config.num_attention_heads,
            self.config.max_position_embeddings,
        )
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights if needed
        self.lm_head.weight = self.embeddings.weight
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.shape
        
        hidden_states = self.embeddings(input_ids)
        
        freqs_cis = self.freqs_cis.to(hidden_states.device)
        
        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
            
        for layer in self.layers:
            if self.config.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, freqs_cis, attention_mask
                )
            else:
                hidden_states = layer(hidden_states, freqs_cis, attention_mask)
            
        hidden_states = self.ln_f(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        
        return lm_logits
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        """
        Generate text using various sampling strategies
        """
        self.eval()
        batch_size = input_ids.shape[0]
        
        for _ in range(max_length - input_ids.size(1)):
            if input_ids.size(1) > self.config.max_position_embeddings:
                input_ids = input_ids[:, -self.config.max_position_embeddings:]
                
            outputs = self(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
                
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
        return input_ids
