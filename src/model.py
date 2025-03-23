import torch
import torch.nn as nn
from torch.nn import functional as F
from fmoe import FMoE
from fmoe.gates import NaiveGate

class MsingiConfig:
    def __init__(
        self,
        vocab_size=32000,  
        max_position_embeddings=1024,
        hidden_size=768,   
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_experts=8,     
        expert_capacity=32,  
        moe_layers=[2, 4]  
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
        self.moe_layers = set(moe_layers)  

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

class ExpertMLP(nn.Module):
    """Expert feedforward network"""
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, x):
        x = F.gelu(self.dense1(x))
        x = self.dense2(x)
        return self.dropout(x)

class MsingiBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layernorm1 = nn.LayerNorm(config.hidden_size)
        self.layernorm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        if layer_idx in config.moe_layers:
            experts = [ExpertMLP(config) for _ in range(config.num_experts)]
            self.feed_forward = FMoE(
                num_expert=config.num_experts,
                d_model=config.hidden_size,
                gate=NaiveGate,
                expert_fn=lambda: experts[len(experts)],
                world_size=1  
            )
        else:
            self.feed_forward = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.GELU(),
                nn.Linear(config.intermediate_size, config.hidden_size),
                nn.Dropout(config.hidden_dropout_prob)
            )
        
    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.layernorm1(hidden_states + attention_output)
        
        layer_output = self.feed_forward(hidden_states)
        layer_output = self.layernorm2(hidden_states + layer_output)
        
        return layer_output

class Msingi1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        self.layers = nn.ModuleList([
            MsingiBlock(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(self, input_ids, attention_mask=None):
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        hidden_states = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        lm_logits = self.lm_head(hidden_states)
        
        return lm_logits
    
    def generate(self, prompt, max_length=100, temperature=0.7, top_k=50):
        self.eval()
        with torch.no_grad():
            if isinstance(prompt, str):
                input_ids = torch.tensor(prompt).unsqueeze(0)
            else:
                input_ids = torch.tensor(prompt).unsqueeze(0)
            
            input_ids = input_ids.to(next(self.parameters()).device)
            
            for _ in range(max_length):
                outputs = self(input_ids)
                next_token_logits = outputs[:, -1, :] / temperature
                
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                probs = F.softmax(top_k_logits, dim=-1)
                next_token = top_k_indices[0, torch.multinomial(probs[0], 1)]
                
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                
                if next_token.item() == 1:
                    break
            
            return input_ids
