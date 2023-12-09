import torch
import torch.nn as nn
import math
import numpy as np 
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):
    matmul_qk = torch.matmul(Q, K.transpose(-2, -1))
    d_k = Q.size(-1)
    scaled_attention_logits = matmul_qk / math.sqrt(d_k)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)

    output = torch.matmul(attention_weights, V)
    return output, attention_weights

class CustomMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(CustomMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)
        
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, Q, K, V, mask):
        batch_size = Q.size(0)
        
        Q = self.q_linear(Q)
        K = self.k_linear(K)
        V = self.v_linear(V)

        Q_split = self.split_heads(Q, batch_size)
        K_split = self.split_heads(K, batch_size)
        V_split = self.split_heads(V, batch_size)

        output, attention_weights = scaled_dot_product_attention(Q_split, K_split, V_split, mask)
        
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        output = self.dense(output)

        return {'output': output, 'attention_weights': attention_weights, 'Q': Q, 'K': K, 'V': V}
    
class CustomEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, dropout):
        super(CustomEncoderBlock, self).__init__()

        self.self_attn = CustomMultiHeadAttention(d_model, num_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.feed_forward1 = nn.Linear(d_model, ffn_dim)
        self.feed_forward2 = nn.Linear(ffn_dim, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Save input for residual connections
        input_residual = x

        # Attention mechanism
        attn_result = self.self_attn(x, x, x, None)

        # Apply dropout to the attention output, then add the residual (input x)
        attn_output = self.dropout(attn_result['output'])
        attn_output += input_residual  # Intermediate residual stream
        # Apply layer norm after adding residual
        attn_output_norm = self.layer_norm1(attn_output)

        # Save attention output before layer norm for residual connection in MLP
        mlp_residual = attn_output_norm

        # MLP / Feed-forward network
        ff_output = self.feed_forward1(mlp_residual)
        ff_output_relu = F.relu(ff_output)  # Output after ReLU
        ff_output = self.feed_forward2(ff_output_relu)

        # Apply dropout to the feed-forward output, then add the residual
        ff_output = self.dropout(ff_output)
        ff_output += mlp_residual  # Intermediate residual stream
        # Apply layer norm after adding residual
        ff_output_norm = self.layer_norm2(ff_output)

        # Output dictionary
        output_dict = {
            'Q': attn_result['Q'],
            'K': attn_result['K'],
            'V': attn_result['V'],
            'attention_output': attn_output_norm,
            'intermediate_residual_stream': input_residual,
            'attention_weights': attn_result['attention_weights'],
            'feed_forward_output_relu': ff_output_relu,
            'feed_forward_output': ff_output_norm,
        }

        return output_dict
    
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, d_model)
        # pe shape from (1, max_len, d_model) sliced to (1, sequence_length, d_model)
        # Output shape will be same as x: (batch_size, sequence_length, d_model)
        return x + self.pe[:, :x.size(1), :]


class TweetyBERT(nn.Module):
    def __init__(self, d_transformer, nhead_transformer, embedding_dim, num_labels, tau=0.1, dropout=0.1, transformer_layers=3, dim_feedforward=128, m = 33, p = 0.01, alpha = 1, length = 1000, sigma=1):
        super(TweetyBERT, self).__init__()
        self.tau = tau
        self.num_labels = num_labels
        self.dropout = dropout
        self.m = m
        self.p = p 
        self.alpha = alpha 
        self.sigma = sigma 
        self.d_transformer = d_transformer
        self.embedding_dim = embedding_dim

        # TweetyNet Front End
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=(14, 1), stride=(14, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=(14, 1), stride=(14, 1))

        self.pos_enc = PositionalEncoding(d_transformer)
        self.learned_pos_enc = nn.Embedding(length, d_transformer)
        self.label_embedding = nn.Embedding(num_labels, embedding_dim)

        # transformer
        self.transformerProjection = nn.Linear(64, d_transformer)
        self.transformer_encoder = nn.ModuleList([CustomEncoderBlock(d_model=d_transformer, num_heads=nhead_transformer, ffn_dim=dim_feedforward, dropout=dropout) for _ in range(transformer_layers)])        
        self.transformerDeProjection = nn.Linear(d_transformer, embedding_dim)


    def feature_extractor_forward(self, x):
        x = F.gelu(self.conv1(x))
        x = self.pool1(x)
        x = F.gelu(self.conv2(x))
        x = self.pool2(x)
        x = x.flatten(1,2)
        return x

    def transformer_forward(self, x):
        all_outputs = []
        for layer in self.transformer_encoder:
            output_dict = layer(x) 
            all_outputs.append(output_dict)
            x = output_dict['feed_forward_output']  

        return x, all_outputs


    def masking_operation(self, x, p=0.01, m=10, noise_weight=20.0):
        
        """
        Apply a mask to the input tensor `x` and replace masked parts with weighted uniform noise.
        
        Parameters:
            x (torch.Tensor): Input tensor with shape [batch, dim, length]
            p (float): Probability of masking a particular element
            m (int): Number of consecutive elements to mask
            noise_weight (float): The weight factor for the noise
            
        Returns:
            torch.Tensor: Tensor with replaced noise, same shape as `x`
            torch.Tensor: Mask tensor with shape [batch, dim, length]
        """
        batch, dim, length = x.size()

        while True:
            prob_mask = torch.rand(length, device=x.device) < p
            if torch.any(prob_mask):  
                break

        expanded_mask = torch.zeros_like(prob_mask)
        for i in range(length):
            if prob_mask[i]:
                expanded_mask[i:min(i+m, length)] = 1

        mask = expanded_mask.view(1, 1, -1).expand_as(x)
        mask_float = mask.float()

        # Generate uniform noise with the same shape as x
        noise = torch.rand(x.size(), device=x.device)  # Uniform noise between 0 and 1
        noise = torch.abs(noise)

        # Replace masked areas with weighted uniform noise
        x = (noise) * mask_float + x * (1-mask_float)
        # x = normalize(x) * mask_float + x * (1-mask_float)

        return x, mask_float
    
    def learned_pos_embedding(self, x):
        # learned pos encoding
        # First, generate a sequence of positional indices
        position_ids = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
        # Then, use the embedding layer to get the positional encodings
        pos_enc = self.learned_pos_enc(position_ids)
        # Add the positional encodings to the input
        x = x + pos_enc
        return x 
    
    def train_forward(self, spec):
        intermediate_outputs = {}

        x = spec.clone()
        intermediate_outputs["input_spec"] = x.clone()

        x = self.feature_extractor_forward(x)
        intermediate_outputs["feature_extractor_forward"] = x.clone()
        x, mask = self.masking_operation(x, p=self.p, m=self.m)
        intermediate_outputs["masking_operation"] = x.clone()
        x = x.permute(0,2,1)
        x = self.transformerProjection(x)

        intermediate_outputs["transformerProjection"] = x.clone().permute(0,2,1)

        # sin pos enc
        # x = self.pos_enc(x)

        # learned pos encoding 
        x = self.learned_pos_embedding(x)

        intermediate_outputs["pos_enc"] = x.clone().permute(0,2,1)

        x, all_outputs = self.transformer_forward(x)

        intermediate_outputs1 = intermediate_outputs.copy()
        for i, layer in enumerate(all_outputs):
            # Add key-value pairs from dict2, modifying keys that overlap
            for key, value in layer.items():
                new_key = key
                # Modify the key if it already exists in dict1
                new_key = f"{key}_{i}"

                if value.dim() <= 3:
                    intermediate_outputs1[new_key] = value.permute(0,2,1)
        
        intermediate_outputs = intermediate_outputs1
        x = self.transformerDeProjection(x)
        intermediate_outputs["transformerDeProjection"] = x.clone().permute(0,2,1)

        return x, mask, spec, intermediate_outputs
    
    def inference_forward(self, spec):
        x = spec.clone()

        x = self.feature_extractor_forward(x)
        x = x.permute(0,2,1)
        x = self.transformerProjection(x)
        x = self.learned_pos_embedding(x)
        x, all_outputs = self.transformer_forward(x)
        x = self.transformerDeProjection(x)

        return x, all_outputs

    def compute_loss(self, predictions, targets, mask):
        epsilon = 1e-6
        alpha = self.alpha

        targets = targets.argmax(dim=-1)

        all_labels = torch.arange(self.num_labels).to(predictions.device)
        all_labels = self.label_embedding(all_labels)

        csim = torch.einsum('bse,le->bsl', predictions, all_labels)
        csim /= (predictions.norm(dim=-1, keepdim=True) * all_labels.norm(dim=-1, keepdim=True).T)

        # Scale the logits by the temperature
        scaled_csim = csim / self.tau

        softmax_csim = F.softmax(scaled_csim, dim=-1)  # use scaled cosine similarity

        predicted_labels = torch.argmax(softmax_csim, dim=-1)
        correct_predictions = (predicted_labels == targets).float()

        masked_indices = mask[:, 0, :] == 1.0
        unmasked_indices = ~masked_indices

        if masked_indices.sum() > 0:
            masked_correct_predictions = correct_predictions[masked_indices]
            masked_sequence_accuracy = masked_correct_predictions.mean()
        else:
            masked_sequence_accuracy = torch.tensor(0.0).to(predictions.device)

        if unmasked_indices.sum() > 0:
            unmasked_correct_predictions = correct_predictions[unmasked_indices]
            unmasked_sequence_accuracy = unmasked_correct_predictions.mean()
        else:
            unmasked_sequence_accuracy = torch.tensor(0.0).to(predictions.device)

        log_softmax_csim = -torch.log(softmax_csim + epsilon)
        loss = torch.gather(log_softmax_csim, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        loss_heatmap = torch.gather(log_softmax_csim, dim=-1, index=targets.unsqueeze(-1))

        if masked_indices.sum() > 0:
            masked_loss = loss[masked_indices].mean()
        else:
            masked_loss = epsilon

        if unmasked_indices.sum() > 0:
            unmasked_loss = loss[unmasked_indices].mean()
        else:
            unmasked_loss = epsilon

        masked_loss += epsilon
        unmasked_loss += epsilon
        combined_loss = alpha * masked_loss + (1 - alpha) * unmasked_loss

        return combined_loss, masked_sequence_accuracy, unmasked_sequence_accuracy, targets, predicted_labels, loss_heatmap, softmax_csim

    def cross_entropy_loss(self, predictions, targets):
        """loss function for TweetyNet
        Parameters
        ----------
        y_pred : torch.Tensor
            output of TweetyNet model, shape (batch, classes, timebins)
        y_true : torch.Tensor
            one-hot encoded labels, shape (batch, classes, timebins)
        Returns
        -------
        loss : torch.Tensor
            mean cross entropy loss
        """
        loss = nn.CrossEntropyLoss()
        return loss(predictions, targets)