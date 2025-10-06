import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.headdropout import HeadDropout


class TreeNode(nn.Module):
    """Individual tree node with learned parameters"""
    def __init__(self, input_dim, depth, max_depth):
        super(TreeNode, self).__init__()
        self.depth = depth
        self.max_depth = max_depth
        self.is_leaf = depth == max_depth
        
        if not self.is_leaf:
            # Internal node - learnable split
            self.split_layer = nn.Linear(input_dim, 1)
            self.left_child = TreeNode(input_dim, depth + 1, max_depth)
            self.right_child = TreeNode(input_dim, depth + 1, max_depth)
        else:
            # Leaf node - output prediction
            self.leaf_output = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        if self.is_leaf:
            return self.leaf_output(x)
        
        # Compute split probability
        split_prob = torch.sigmoid(self.split_layer(x))
        
        # Get outputs from both children
        left_output = self.left_child(x)
        right_output = self.right_child(x)
        
        # Soft routing based on split probability
        output = split_prob * right_output + (1 - split_prob) * left_output
        return output


class DifferentiableDecisionTree(nn.Module):
    """Differentiable decision tree"""
    def __init__(self, input_dim, tree_depth):
        super(DifferentiableDecisionTree, self).__init__()
        self.tree_depth = tree_depth
        self.root = TreeNode(input_dim, 0, tree_depth)
    
    def forward(self, x):
        return self.root(x)


class Model(nn.Module):
    """
    MoLE_TreeNN: Mixture of Linear Experts with Tree-based routing
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.configs = configs
        
        self.num_predictions = configs.t_dim
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        
        # Tree-specific parameters
        self.tree_depth = getattr(configs, 'tree_depth', 3)
        self.num_trees = getattr(configs, 'num_trees', 2)
        
        # Linear experts for each channel
        self.Linear = nn.ModuleList([
            nn.Linear(configs.seq_len, configs.pred_len * self.num_predictions) 
            for _ in range(configs.enc_in)
        ]) if configs.individual else nn.Linear(configs.seq_len, configs.pred_len * self.num_predictions)
        
        # Tree ensemble for temporal routing
        self.trees = nn.ModuleList([
            DifferentiableDecisionTree(4, self.tree_depth) 
            for _ in range(self.num_trees)
        ])
        
        # Combine tree outputs for final gating
        self.tree_combiner = nn.Linear(self.num_trees, self.num_predictions * self.channels)
        
        # Dropout and normalization
        self.dropout = nn.Dropout(configs.drop)
        self.head_dropout = HeadDropout(configs.head_dropout)
        
        # Optional reversible normalization
        if not getattr(configs, 'disable_rev', False):
            try:
                from layers.Invertible import RevIN
                self.rev = RevIN(configs.enc_in)
            except ImportError:
                print("Warning: RevIN not available, using standard normalization")
                self.rev = None
        else:
            self.rev = None
        
        self.individual = configs.individual

    def forward(self, x, x_mark, return_gating_weights=False, return_seperate_head=False):
        # x: [Batch, Input length, Channel]
        x_mark_initial = x_mark[:, 0]  # Use initial time features
        
        # Apply reversible normalization if available
        if self.rev:
            x = self.rev(x, 'norm')
        
        # Apply dropout
        x = self.dropout(x)
        
        # Get predictions from linear layers
        if self.individual:
            pred_list = []
            for i, linear_layer in enumerate(self.Linear):
                channel_pred = linear_layer(x[:, :, i])
                pred_list.append(channel_pred)
            pred = torch.stack(pred_list, dim=1)  # [B, C, pred_len * num_predictions]
        else:
            pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)  # [B, pred_len * num_predictions, C]
            pred = pred.transpose(1, 2)  # [B, C, pred_len * num_predictions]
        
        # Tree-based temporal routing
        tree_outputs = []
        for tree in self.trees:
            tree_out = tree(x_mark_initial)  # [B, 1]
            tree_outputs.append(tree_out)
        
        tree_combined = torch.cat(tree_outputs, dim=-1)  # [B, num_trees]
        
        # Generate gating weights
        temporal_out = self.tree_combiner(tree_combined)  # [B, num_predictions * channels]
        temporal_out = temporal_out.reshape(-1, self.num_predictions, self.channels)
        
        # Apply head dropout
        temporal_out = self.head_dropout(temporal_out)
        temporal_out = F.softmax(temporal_out, dim=1)
        
        # Reshape predictions for mixture
        pred_raw = pred.reshape(-1, self.channels, self.pred_len, self.num_predictions)
        pred_raw = pred_raw.permute(0, 3, 1, 2)  # [B, num_predictions, C, pred_len]
        
        # Apply mixture weights
        pred = pred_raw * temporal_out.unsqueeze(-1)
        pred = pred.sum(dim=1).permute(0, 2, 1)  # [B, pred_len, C]
        
        # Apply reverse normalization if available
        if self.rev:
            pred = self.rev(pred, 'denorm')
        
        if return_gating_weights:
            return pred, temporal_out
        
        return pred