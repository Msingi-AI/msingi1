import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Optional

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute the frequency tensor for complex exponentials (rotary embeddings)."""
    # Simplified implementation that works on all PyTorch versions
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    t = torch.arange(end, dtype=torch.float)
    freqs = torch.outer(t, freqs)  # [seq_len, dim/2]
    
    # Compute cos and sin for the frequencies
    cos = torch.cos(freqs)  # [seq_len, dim/2]
    sin = torch.sin(freqs)  # [seq_len, dim/2]
    
    return cos, sin

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary embeddings to input tensors using precomputed cos and sin.
    This is a simplified implementation that avoids complex reshaping.
    
    Args:
        x: Input tensor of shape [batch_size, seq_len, n_heads, head_dim]
        cos: Cosine part of the frequency tensor [seq_len, dim/2]
        sin: Sine part of the frequency tensor [seq_len, dim/2]
    """
    # Extract dimensions
    batch, seq_len, n_heads, d = x.shape
    
    # Handle odd dimensions
    d_2 = d // 2
    
    # Limit to seq_len
    cos = cos[:seq_len]  # [seq_len, dim/2]
    sin = sin[:seq_len]  # [seq_len, dim/2]
    
    # Reshape for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim/2]
    sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim/2]
    
    # Split the input tensor into two halves along the last dimension
    x_1 = x[..., :d_2]  # [batch, seq_len, n_heads, dim/2]
    x_2 = x[..., d_2:2*d_2]  # [batch, seq_len, n_heads, dim/2]
    
    # Apply the rotation using the rotation matrix:
    # [cos, -sin]
    # [sin,  cos]
    rotated_x_1 = x_1 * cos - x_2 * sin
    rotated_x_2 = x_2 * cos + x_1 * sin
    
    # Concatenate the two parts
    rotated_x = torch.cat([rotated_x_1, rotated_x_2], dim=-1)
    
    # If the dimension is odd, keep the last element unchanged
    if d != d_2 * 2:
        rotated_x = torch.cat([rotated_x, x[..., 2*d_2:]], dim=-1)
    
    return rotated_x

class MsingiConfig:
    def __init__(
        self,
        vocab_size=32000,
        max_position_embeddings=1024,
        n_layer=8,
        n_head=8,
        n_embd=512,
        intermediate_size=2048,
        dropout=0.1,
        rotary_emb=True,
        gradient_checkpointing=True,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.rotary_emb = rotary_emb
        self.gradient_checkpointing = gradient_checkpointing
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        
        # Ensure hidden size is divisible by num_attention_heads
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        # Ensure hidden size is divisible by 2 for rotary embeddings
        assert n_embd % 2 == 0, "n_embd must be even for rotary embeddings"

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.n_head
        self.hidden_size = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, hidden_states, cos, sin, attention_mask=None):
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project and reshape
        q = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        
        # Apply rotary embeddings
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        if attention_mask is not None:
            scores = scores + attention_mask
            
        attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(q)
        attention_weights = self.dropout(attention_weights)
        
        # Compute output
        output = torch.matmul(attention_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        
        return self.out_proj(output)

class MsingiBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.intermediate_size),
            nn.GELU(approximate='tanh'),
            nn.Linear(config.intermediate_size, config.n_embd),
            nn.Dropout(config.dropout),
        )
        self.ln1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
    def forward(self, hidden_states, cos, sin, attention_mask=None):
        # Pre-norm architecture
        attn_output = self.attention(self.ln1(hidden_states), cos, sin, attention_mask)
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.ln2(hidden_states))
        return hidden_states

class Msingi1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([MsingiBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Compute rotary position embeddings and register as buffer
        # This ensures it's properly saved and moved to the right device
        cos, sin = precompute_freqs_cis(
            self.config.n_embd // self.config.n_head,
            self.config.max_position_embeddings,
        )
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embeddings.weight
        
        # Gradient checkpointing flag
        self.gradient_checkpointing = False
    
    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        
    def forward(self, input_ids, attention_mask=None):
        # Get embeddings
        hidden_states = self.embeddings(input_ids)
        
        # Process through transformer layers with optional gradient checkpointing
        if self.gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward

            for layer in self.layers:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states, self.cos, self.sin, attention_mask,
                    use_reentrant=False
                )
        else:
            for layer in self.layers:
                hidden_states = layer(hidden_states, self.cos, self.sin, attention_mask)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        return logits
    
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
