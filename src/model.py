import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Optional

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device="cpu"):
    """Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
    
    Args:
        dim: Dimension of the rotation embeddings (head dimension)
        end: Maximum sequence length
        theta: Base value for frequencies
        device: Device to store the frequencies on
    
    Returns:
        Tuple of (cos, sin) tensors of shape [end, dim // 2]
    """
    # Make sure dim is even for rotary embeddings
    if dim % 2 != 0:
        raise ValueError(f"Dimension must be even for rotary embeddings, got {dim}")
    
    # Compute frequencies
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    
    # Create position indices
    t = torch.arange(end, device=device).float()
    
    # Compute frequency matrix
    freqs = torch.outer(t, freqs)
    
    # Compute cos and sin directly (no complex numbers)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    
    return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary position embeddings to query and key tensors.
    
    Args:
        q: Query tensor of shape [batch_size, seq_length, n_heads, head_dim]
        k: Key tensor of shape [batch_size, seq_length, n_heads, head_dim]
        cos: Cosine part of the frequency tensor [seq_length, dim//2]
        sin: Sine part of the frequency tensor [seq_length, dim//2]
        position_ids: Optional position indices, defaults to None
        
    Returns:
        Tuple of rotated query and key tensors
    """
    # Get dimensions
    batch, seq_len, n_heads, d_head = q.shape
    
    # Handle position_ids or use default sequence
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=q.device)
    
    # Ensure position_ids are valid - clip to the maximum available position
    max_position = cos.shape[0] - 1
    
    # Safely clamp position_ids to valid range
    safe_position_ids = torch.clamp(position_ids, 0, max_position)
    
    if position_ids.max() > max_position:
        print(f"Warning: Position ids max {position_ids.max().item()} exceeds precomputed frequency length {max_position}")
        print(f"Using clamped position ids. Consider increasing max_position_embeddings in config.")
    
    # Get the cos and sin values for the positions (safely)
    cos_pos = cos[safe_position_ids]  # [seq_len, dim//2]
    sin_pos = sin[safe_position_ids]  # [seq_len, dim//2]
    
    # Reshape for broadcasting
    cos_pos = cos_pos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim//2]
    sin_pos = sin_pos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim//2]
    
    # Ensure head dimension is even
    if d_head % 2 != 0:
        raise ValueError(f"Head dimension must be even for rotary embeddings, got {d_head}")
    
    # Split q and k into two parts along the last dimension
    q_split = torch.reshape(q, (batch, seq_len, n_heads, 2, d_head // 2))
    k_split = torch.reshape(k, (batch, seq_len, n_heads, 2, d_head // 2))
    
    # Apply rotation using the following identities:
    # q_rot = [q1 * cos - q2 * sin, q2 * cos + q1 * sin]
    q1, q2 = q_split[:, :, :, 0], q_split[:, :, :, 1]
    k1, k2 = k_split[:, :, :, 0], k_split[:, :, :, 1]
    
    # Apply rotation
    q1_rot = q1 * cos_pos - q2 * sin_pos
    q2_rot = q2 * cos_pos + q1 * sin_pos
    k1_rot = k1 * cos_pos - k2 * sin_pos
    k2_rot = k2 * cos_pos + k1 * sin_pos
    
    # Concatenate back
    q_rot = torch.cat([q1_rot, q2_rot], dim=-1)
    k_rot = torch.cat([k1_rot, k2_rot], dim=-1)
    
    # Reshape back to original shape
    q_rot = torch.reshape(q_rot, (batch, seq_len, n_heads, d_head))
    k_rot = torch.reshape(k_rot, (batch, seq_len, n_heads, d_head))
    
    return q_rot, k_rot


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
        """Forward pass of the MultiHeadAttention module.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            cos: Cosine part of the frequency tensor [seq_length, dim//2]
            sin: Sine part of the frequency tensor [seq_length, dim//2]
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor of shape [batch_size, seq_length, hidden_size]
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project to query, key, value
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape to [batch_size, seq_length, n_heads, head_dim]
        q = q.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        
        # Apply rotary embeddings to query and key
        try:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        except Exception as e:
            print(f"Error in rotary embeddings: {e}")
            print(f"q shape: {q.shape}, k shape: {k.shape}")
            print(f"cos shape: {cos.shape}, sin shape: {sin.shape}")
            # Fall back to no rotary embeddings
            pass
        
        # Transpose to [batch_size, n_heads, seq_length, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scale query
        q = q * self.scaling
        
        # Compute attention scores
        # [batch_size, n_heads, seq_length, seq_length]
        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Compute output
        # [batch_size, n_heads, seq_length, head_dim]
        output = torch.matmul(attention_probs, v)
        
        # Transpose back to [batch_size, seq_length, n_heads, head_dim]
        output = output.transpose(1, 2).contiguous()
        
        # Reshape to [batch_size, seq_length, hidden_size]
        output = output.view(batch_size, seq_length, self.hidden_size)
        
        # Project to output
        output = self.out_proj(output)
        
        return output


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
        """Forward pass of the transformer block.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            cos: Cosine part of the frequency tensor [seq_length, dim//2]
            sin: Sine part of the frequency tensor [seq_length, dim//2]
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor of shape [batch_size, seq_length, hidden_size]
        """
        # Pre-norm architecture with safety checks
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        attn_output = self.attention(hidden_states, cos, sin, attention_mask)
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
        
        # Precompute rotary embeddings with a very large safety margin
        # This avoids having to dynamically extend during forward passes
        head_dim = config.n_embd // config.n_head
        max_seq_len = max(2048, config.max_position_embeddings * 2)  # Use at least 2048 or double the config
        print(f"Precomputing rotary embeddings for sequence length {max_seq_len}")
        cos, sin = precompute_freqs_cis(head_dim, max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        
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
        if seq_length > self.cos.shape[0]:
            # Precompute new rotary embeddings with a larger size
            max_seq_len = seq_length + 512  # Add much larger safety margin
            print(f"Warning: Input sequence length {seq_length} exceeds precomputed frequency length {self.cos.shape[0]}")
            print(f"Precomputing new rotary embeddings with length {max_seq_len}")
            
            # Compute on CPU first to avoid CUDA errors, then transfer
            cos, sin = precompute_freqs_cis(
                self.config.n_embd // self.config.n_head, 
                max_seq_len, 
                device="cpu"
            )
            # Move to the same device as input_ids
            self.register_buffer("cos", cos.to(input_ids.device), persistent=False)
            self.register_buffer("sin", sin.to(input_ids.device), persistent=False)
        
        # Get embeddings
        hidden_states = self.embeddings(input_ids)
        
        # Create attention mask if provided
        if attention_mask is not None:
            # Convert mask from [batch_size, seq_length] to [batch_size, 1, 1, seq_length]
            # with 1 for tokens to attend to and 0 for tokens to ignore
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        
        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory during training
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    self.cos,
                    self.sin,
                    attention_mask,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    self.cos,
                    self.sin,
                    attention_mask,
                )
        
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
