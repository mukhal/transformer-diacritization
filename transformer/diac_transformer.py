import torch
import torch.nn as nn

from base_transformer import Transformer

class TransformerWithDiacritizationHead(nn.Module):
    def __init__(self, config):
        """ Transformer with a diacritization head on top"""
        super().__init__()
        self.config = config
        self.transformer = Transformer(config.embed_dim, config.hidden_dim, config.num_embeddings,
                                       config.num_max_positions, config.num_heads, config.num_layers,
                                       config.dropout, causal=not config.mlm)
        
        
        self.diac_head = nn.Linear(config.embed_dim, config.num_diac_labels, bias=False)
        print(self.diac_head)
        
    def init_weights(self, module):
        """ initialize weights - nn.MultiheadAttention is already initalized by PyTorch (xavier) """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, labels=None, padding_mask=None):
        """ x has shape [seq length, batch], padding_mask has shape [batch, seq length] """
        hidden_states = self.transformer(x, padding_mask)    
        logits = self.diac_head(hidden_states) # seq_len x batch x num_labels
     
        if labels is not None: # training stage
            
            assert labels.size(0) == logits.size(0), "logits and labels dimension mismatch"
            
            #shift_logits = logits[:-1] if self.transformer.causal else logits
            #shift_labels = labels[1:] if self.transformer.causal else labels
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.padding_idx)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return logits, loss

        return logits