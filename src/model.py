import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Optional, Tuple
from torch.cuda.amp import autocast

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to input tensors using precomputed frequencies"""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Expand freqs_cis to match the batch size and number of heads
    # freqs_cis shape: [seq_len, dim//2]
    # xq_ shape: [batch, seq_len, n_heads, dim//2]
    seq_len = xq_.shape[1]
    freqs_cis = freqs_cis[:seq_len]  # [seq_len, dim//2]
    
    # Reshape for broadcasting
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim//2]
    
    # Apply rotation
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class MsingiConfig:
    def __init__(
        self,
        vocab_size=32000,  # Set to match ByteLevelBPE tokenizer vocabulary size
        max_position_embeddings=2048,  # Context length
        n_layer=12,  # Number of transformer layers
        n_head=12,  # Number of attention heads
        n_embd=768,  # Embedding dimension
        intermediate_size=3072,  # Feed-forward intermediate size
        # Total params: ~110M (with 32K vocab)
        dropout=0.1,
        rotary_emb=True,
        gradient_checkpointing=True,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        use_flash_attn=False,  # New parameter for flash attention
        use_mixed_precision=True  # New parameter for mixed precision
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
        self.use_flash_attn = use_flash_attn
        self.use_mixed_precision = use_mixed_precision
        
        # Ensure hidden size is divisible by num_attention_heads
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        # Ensure hidden size is divisible by 2 for rotary embeddings
        assert n_embd % 2 == 0, "n_embd must be even for rotary embeddings"

class FlashAttention(nn.Module):
    """Optimized attention implementation for faster computation"""
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
        
        # Optimized attention computation - more memory efficient
        # Compute attention scores in chunks to save memory
        chunk_size = 1024
        output = torch.zeros_like(q)
        
        for i in range(0, seq_length, chunk_size):
            chunk_end = min(i + chunk_size, seq_length)
            attn_weights = torch.matmul(q, k[:, :, i:chunk_end].transpose(-2, -1)) * self.scaling
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask[:, :, :, i:chunk_end]
                
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)
            attn_weights = self.dropout(attn_weights)
            
            output = output + torch.matmul(attn_weights, v[:, :, i:chunk_end])
        
        # Reshape output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        
        return self.out_proj(output)

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
        if config.use_flash_attn:
            self.attention = FlashAttention(config)
        else:
            self.attention = MultiHeadAttention(config)
            
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.intermediate_size),
            nn.GELU(approximate='tanh'),
            nn.Linear(config.intermediate_size, config.n_embd),
            nn.Dropout(config.dropout),
        )
        self.ln1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.use_mixed_precision = config.use_mixed_precision
        
    def forward(self, hidden_states, freqs_cis, attention_mask=None):
        # Pre-norm architecture
        residual = hidden_states
        
        # Apply mixed precision if enabled
        if self.use_mixed_precision and torch.cuda.is_available():
            with autocast():
                hidden_states = self.ln1(hidden_states)
                attn_output = self.attention(hidden_states, freqs_cis, attention_mask)
        else:
            hidden_states = self.ln1(hidden_states)
            attn_output = self.attention(hidden_states, freqs_cis, attention_mask)
            
        hidden_states = residual + attn_output
        residual = hidden_states
        
        # Apply mixed precision if enabled
        if self.use_mixed_precision and torch.cuda.is_available():
            with autocast():
                hidden_states = self.ln2(hidden_states)
                mlp_output = self.mlp(hidden_states)
        else:
            hidden_states = self.ln2(hidden_states)
            mlp_output = self.mlp(hidden_states)
            
        hidden_states = residual + mlp_output
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
        
        # Compute rotary position embeddings
        self.freqs_cis = precompute_freqs_cis(
            self.config.n_embd // self.config.n_head,
            self.config.max_position_embeddings,
        )
        
        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embeddings.weight
        
        # Gradient checkpointing flag
        self.gradient_checkpointing = config.gradient_checkpointing
        
        # Mixed precision flag
        self.use_mixed_precision = config.use_mixed_precision
    
    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        
    def forward(self, input_ids, attention_mask=None):
        # Apply mixed precision if enabled
        if self.use_mixed_precision and torch.cuda.is_available():
            with autocast():
                return self._forward_impl(input_ids, attention_mask)
        else:
            return self._forward_impl(input_ids, attention_mask)
    
    def _forward_impl(self, input_ids, attention_mask=None):
        hidden_states = self.embeddings(input_ids)
        
        # Move freqs_cis to the same device as hidden states
        freqs_cis = self.freqs_cis.to(hidden_states.device)
        
        # Process through transformer layers with optional gradient checkpointing
        if self.gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward

            for layer in self.layers:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states, freqs_cis, attention_mask
                )
        else:
            for layer in self.layers:
                hidden_states = layer(hidden_states, freqs_cis, attention_mask)
        
        hidden_states = self.ln_f(hidden_states)
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
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
    ):
        """
        Generate text using various sampling strategies with additional controls
        """
        self.eval()
        batch_size = input_ids.shape[0]
        
        # Track generated token IDs for repetition penalty
        generated_tokens = input_ids.clone()
        
        for _ in range(max_length - input_ids.size(1)):
            if input_ids.size(1) > self.config.max_position_embeddings:
                input_ids = input_ids[:, -self.config.max_position_embeddings:]
                generated_tokens = generated_tokens[:, -self.config.max_position_embeddings:]
                
            outputs = self(input_ids)
            next_token_logits = outputs[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(generated_tokens[i].tolist()):
                        # If the token appears in the generated text, penalize it
                        next_token_logits[i, token_id] /= repetition_penalty
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
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
            
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
            
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
            
    def save_pretrained(self, save_directory: str):
        """Save model weights and configuration to directory"""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save configuration
        config_dict = self.config.__dict__.copy()
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
    @classmethod
    def from_pretrained(cls, load_directory: str, map_location=None):
        """Load model weights and configuration from directory"""
        import os
        import json
        
        # Load configuration
        with open(os.path.join(load_directory, "config.json"), "r") as f:
            config_dict = json.load(f)
        
        # Create config object
        config = MsingiConfig(**config_dict)
        
        # Create model
        model = cls(config)
        
        # Load weights
        model.load_state_dict(
            torch.load(
                os.path.join(load_directory, "pytorch_model.bin"),
                map_location=map_location
            )
        )
        
        return model