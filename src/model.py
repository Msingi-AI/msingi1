import torch
import torch.nn as nn
from torch.nn import functional as F

class MsingiConfig:
    def __init__(
        self,
        vocab_size=50000,
        max_position_embeddings=1024,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_experts=4,  # Number of expert networks
        expert_capacity=2,  # How many tokens each expert can process
        moe_layer_frequency=2,  # Add MoE every N layers
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.moe_layer_frequency = moe_layer_frequency

class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return self.dropout(x)

class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        self.hidden_size = config.hidden_size
        
        # Create experts
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])
        
        # Router network (gates)
        self.router = nn.Linear(config.hidden_size, config.num_experts)
        
    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)  # Combine batch and sequence
        
        # Get router scores and probabilities
        router_logits = self.router(hidden_states)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts for each token
        top_k_probs, top_k_indices = torch.topk(router_probs, k=2, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities
        
        # Initialize output tensor
        final_output = torch.zeros_like(hidden_states)
        
        # Dispatch to experts
        for expert_idx, expert in enumerate(self.experts):
            # Find which tokens go to this expert
            expert_mask = top_k_indices[:, 0] == expert_idx
            if expert_mask.any():
                # Process tokens with this expert
                expert_input = hidden_states[expert_mask]
                expert_output = expert(expert_input)
                final_output[expert_mask] = expert_output * top_k_probs[expert_mask, 0].unsqueeze(-1)
        
        # Reshape back to original dimensions
        final_output = final_output.view(batch_size, sequence_length, hidden_size)
        return final_output

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float))
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer

class MsingiBlock(nn.Module):
    def __init__(self, config, use_moe=False):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.use_moe = use_moe
        
        if use_moe:
            self.feed_forward = MoELayer(config)
        else:
            self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
            self.output = nn.Linear(config.intermediate_size, config.hidden_size)
            
        self.layernorm1 = nn.LayerNorm(config.hidden_size)
        self.layernorm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.layernorm1(hidden_states + attention_output)
        
        if self.use_moe:
            layer_output = self.feed_forward(hidden_states)
        else:
            intermediate_output = F.gelu(self.intermediate(hidden_states))
            layer_output = self.output(intermediate_output)
            
        layer_output = self.dropout(layer_output)
        layer_output = self.layernorm2(hidden_states + layer_output)
        
        return layer_output

class Msingi1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Create layers with MoE at specified frequency
        self.layers = nn.ModuleList([
            MsingiBlock(config, use_moe=(i % config.moe_layer_frequency == 0))
            for i in range(config.num_hidden_layers)
        ])
        
        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(self, input_ids, attention_mask=None):
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        inputs_embeds = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        hidden_states = inputs_embeds + position_embeddings
        hidden_states = self.dropout(hidden_states)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
            
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            
        hidden_states = self.layernorm(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        
        return lm_logits
    
    def generate(self, input_ids, max_length, temperature=1.0):
        self.eval()
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                outputs = self(input_ids)
                next_token_logits = outputs[:, -1, :] / temperature
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
