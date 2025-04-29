import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Optional

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device="cpu"):
    """Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
    
    Args:
        dim: Dimension of the rotation embeddings
        end: Maximum sequence length
        theta: Base value for frequencies
        device: Device to store the frequencies on
    
    Returns:
        Complex tensor of shape [end, dim // 2] for applying rotary embeddings
    """
    # Make sure dim is even for rotary embeddings
    assert dim % 2 == 0, "Dimension must be even for rotary embeddings"
    
    # Create position indices tensor
    pos = torch.arange(end, device=device)
    
    # Create frequency indices tensor
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device) / dim))
    
    # Compute frequencies for each position and frequency index
    # Ensure shape compatibility with broadcasting
    freqs_cis = torch.outer(pos, freqs)
    
    # Convert to complex exponentials
    freqs_cis = torch.polar(torch.ones_like(freqs_cis), freqs_cis)
    
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to input tensors using the given frequency tensor.
    
    Args:
        xq: Query states tensor of shape [batch_size, seq_length, n_heads, head_dim]
        xk: Key states tensor of shape [batch_size, seq_length, n_heads, head_dim]
        freqs_cis: Complex tensor of shape [seq_length, head_dim/2]
        
    Returns:
        Tuple of (query_states, key_states) with rotary embeddings applied
    """
    # Ensure all tensors are on the same device
    if freqs_cis.device != xq.device:
        freqs_cis = freqs_cis.to(xq.device)
    
    # Extract tensor shapes
    batch, seq_len, n_heads, head_dim = xq.shape
    
    # Reshape for broadcasting the rotary embeddings
    xq_r = xq.reshape(batch, seq_len, n_heads, head_dim // 2, 2)
    xk_r = xk.reshape(batch, seq_len, n_heads, head_dim // 2, 2)
    
    # Convert to complex values
    xq_complex = torch.complex(xq_r[..., 0], xq_r[..., 1])
    xk_complex = torch.complex(xk_r[..., 0], xk_r[..., 1])
    
    # Ensure freqs_cis has the right sequence length
    # If input sequence is longer than precomputed, truncate
    # If input sequence is shorter, use only what's needed
    if seq_len > freqs_cis.shape[0]:
        # Handle case where sequence length exceeds precomputed frequencies
        # This should be avoided by ensuring freqs_cis is precomputed for max length
        print(f"Warning: Input sequence length {seq_len} exceeds precomputed frequency length {freqs_cis.shape[0]}")
        # Safely handle by using modulo to repeat frequencies
        freqs_cis_effective = freqs_cis[torch.arange(seq_len, device=freqs_cis.device) % freqs_cis.shape[0]]
    else:
        # Normal case - just use the needed frequencies
        freqs_cis_effective = freqs_cis[:seq_len]
    
    # Apply rotary embeddings by complex multiplication
    # Reshape freqs_cis for broadcasting
    freqs_cis_reshaped = freqs_cis_effective.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]
    
    # Apply complex multiplication
    xq_out = torch.view_as_real(xq_complex * freqs_cis_reshaped)
    xk_out = torch.view_as_real(xk_complex * freqs_cis_reshaped)
    
    # Reshape back to the original shape
    xq_out = xq_out.reshape(batch, seq_len, n_heads, head_dim)
    xk_out = xk_out.reshape(batch, seq_len, n_heads, head_dim)
    
    # Ensure output has the same dtype as input
    return xq_out.type_as(xq), xk_out.type_as(xk)


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
        
    def forward(self, hidden_states, freqs_cis, attention_mask=None):
        """Forward pass of the MultiHeadAttention module.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            freqs_cis: Precomputed frequencies for rotary embeddings
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor of shape [batch_size, seq_length, hidden_size]
        """
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project and reshape
        q = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        
        # Apply rotary embeddings
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        if attention_mask is not None:
            # Expand mask for broadcasting to attention heads
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Convert mask values to a large negative number to mask out padding tokens
            attention_mask = (1.0 - attention_mask) * -10000.0
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
        
    def forward(self, hidden_states, freqs_cis, attention_mask=None):
        """Forward pass of the transformer block.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            freqs_cis: Precomputed frequencies for rotary embeddings
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor of shape [batch_size, seq_length, hidden_size]
        """
        # Pre-norm architecture with safety checks
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        attn_output = self.attention(hidden_states, freqs_cis, attention_mask)
        hidden_states = residual + attn_output
        
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        
        return hidden_states


class Msingi1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Create token embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Create transformer layers
        self.layers = nn.ModuleList([MsingiBlock(config) for _ in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Output projection (tied with embeddings)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embeddings.weight
        
        # Precompute rotary embeddings
        # Add a safety margin to max_seq_len to avoid index errors
        max_seq_len = config.max_position_embeddings + 128  # Add safety margin
        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(config.n_embd, max_seq_len), persistent=False
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Gradient checkpointing flag
        self.gradient_checkpointing = False
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Use Xavier initialization for linear layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass of the model.
        
        Args:
            input_ids: Input token ids, shape [batch_size, seq_length]
            attention_mask: Optional attention mask, shape [batch_size, seq_length]
            
        Returns:
            Logits for next token prediction, shape [batch_size, seq_length, vocab_size]
        """
        # Get input shape
        batch_size, seq_length = input_ids.shape
        
        # Ensure sequence length doesn't exceed our precomputed frequencies
        if seq_length > self.freqs_cis.shape[0]:
            # Dynamically extend freqs_cis if needed
            max_seq_len = seq_length + 128  # Add safety margin
            self.freqs_cis = precompute_freqs_cis(
                self.config.n_embd, max_seq_len, device=input_ids.device
            )
            print(f"Extended rotary embeddings to length {max_seq_len}")
        
        # Get embeddings
        hidden_states = self.embeddings(input_ids)
        
        # Process through transformer layers
        if self.gradient_checkpointing and self.training:
            # Use gradient checkpointing to save memory during training
            for layer in self.layers:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, self.freqs_cis, attention_mask
                )
        else:
            # Standard forward pass
            for layer in self.layers:
                hidden_states = layer(hidden_states, self.freqs_cis, attention_mask)
        
        # Apply final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
    
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
