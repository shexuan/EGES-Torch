import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os, pickle
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict, Counter


def create_embedding_matrix(sparse_columns, varlen_sparse_columns, embed_dim,
                            init_std=0.0001, padding=True, device='cpu', mode='mean'):
    # sparse_columns => dict{'name':vocab_size}
    # Return nn.ModuleDict: for sparse features, {embedding_name: nn.Embedding}
    padding_idx = 0 if padding else None
    sparse_embedding_dict = {
        feat: nn.Embedding(sparse_columns[feat], embed_dim, padding_idx=padding_idx)
                             for feat in sparse_columns
    }
    
    if varlen_sparse_columns:
        varlen_sparse_embedding_dict = {
            feat:nn.EmbeddingBag(varlen_sparse_columns[feat], embed_dim, padding_idx=padding_idx,
                                 mode=mode) for feat in varlen_sparse_columns
        }
        sparse_embedding_dict.update(varlen_sparse_embedding_dict)
        
    embedding_dict = nn.ModuleDict(sparse_embedding_dict)
    
    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)
        # nn.init.kaiming_uniform_(tensor.weight, mode='fan_in', nonlinearity='relu')

    return embedding_dict.to(device)


class EGES(nn.Module):
    def __init__(self, sparse_dict, varlen_sparse_dict=None, target_col='sku_id',
                 n_embed=64, k_side=3, noise_dist=None, device='cpu', padding=True):
        """sparse_dict: dict, {feature_name: vocab_size}
        """
        super().__init__()
        self.n_embed = n_embed
        self.k_side = k_side
        self.device = device
        self.padding = padding
        self.target_col = target_col
        self.features = list(sparse_dict.keys())
        if varlen_sparse_dict:
            self.features = self.features + list(varlen_sparse_dict.keys())
        # 如果padding了的话，则负采样出来的index均需要+1
        self.sample_word_offset = 1 if padding else 0
        # input embedding dict, include item and side info
        self.input_embedding_dict = create_embedding_matrix(
            sparse_dict, varlen_sparse_dict, n_embed,
            init_std=0.0001, padding=padding, device=device, mode='mean')
        self.out_embed = nn.Embedding(sparse_dict[target_col], n_embed,
                                      padding_idx=0 if padding else None)
        self.attn_embed = nn.Embedding(sparse_dict[target_col], k_side+1, 
                                       padding_idx=0 if padding else None)
        
        # Initialize out embedding tables with uniform distribution
        nn.init.normal_(self.out_embed.weight, mean=0, std=0.0001)
        nn.init.normal_(self.attn_embed.weight, mean=0, std=0.0001)

        if noise_dist is None:
            # sampling words uniformly
            self.noise_dist = torch.ones(self.n_vocab)
        else:
            self.noise_dist = noise_dist
        self.noise_dist = self.noise_dist.to(device)

    def forward_input(self, input_dict):
        # return input vector embeddings
        embed_lst = []
        for col in self.features:
            if col in input_dict:
                input_vector = self.input_embedding_dict[col](input_dict[col])
                embed_lst.append(input_vector)

        batch_size = input_vector.shape[0]
        # embeds => [batch_size, k_side+1, n_embed]
        embeds = torch.cat(embed_lst, dim=1).reshape(batch_size, self.k_side+1, self.n_embed)
        
        # attation => [batch_size, k_side+1]
        attn_w = self.attn_embed(input_dict[self.target_col])
        attn_w = torch.exp(attn_w)
        attn_s = torch.sum(attn_w, dim=1).reshape(-1, 1)
        attn_w = (attn_w/attn_s).reshape(batch_size, 1, self.k_side+1) # 归一化
        
        # attw => [batch_size, 1, k_side+1]
        # embeds => [batch_size, k_side+1, embed_size]
        # matmul out => [batch_size, 1, embed_size]
        input_vector = torch.matmul(attn_w, embeds).squeeze(1)
        
        return input_vector

    def forward_output(self, output_words):
        # return output vector embeddings 
        output_vector = self.out_embed(output_words)
        return output_vector
    
    def forward_noise(self, batch_size, n_samples):
        """Generate noise vectors with shape [batch_size, n_samples, n_embed]
        """
        # sample words from our noise distribution 
        noise_words = torch.multinomial(self.noise_dist, batch_size*n_samples, 
                                        replacement=True) + self.sample_word_offset
        noise_vector = self.out_embed(noise_words).view(batch_size, n_samples, self.n_embed)
        
        return noise_vector
    
    def forward_cold(self, input_dict):
        """处理冷启动item，使用其side info Embedding的均值
        """
        # return input vector embeddings
        embed_lst = []
        for col in self.features:
            if col in input_dict:
                input_vector = self.input_embedding_dict[col](input_dict[col])
                embed_lst.append(input_vector)

        batch_size = input_vector.shape[0]
        # embeds => [batch_size, k_side, n_embed]
        embeds = torch.cat(embed_lst, dim=1).reshape(batch_size, self.k_side, self.n_embed)
        return torch.mean(embeds, dim=1)


class NegativeSamplingLoss(nn.Module):
    """这里用的是负对数似然, 而不是sampled softmax
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, input_vectors, output_vectors, noise_vectors):
        batch_size, embed_size = input_vectors.shape
        
        # input vectors should be a batch of column vectors
        input_vectors = input_vectors.view(batch_size, embed_size, 1)
        
        # output vectors should be a batch of row vectors
        output_vectors = output_vectors.view(batch_size, 1, embed_size)
        
        # bmm = batch matrix multiplication
        # target words log-sigmoid loss
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()
        
        # negative sampling words log-sigmoid loss
        # negative words sigmoid optmize to small, thus here noise_vectors.neg()
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
        # sum the losses over the sample of noise vectors
        noise_loss = noise_loss.squeeze().sum(1)
        
        # sum target and negative loss
        return -(out_loss + noise_loss).mean()
