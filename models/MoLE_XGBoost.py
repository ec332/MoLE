import torch
import torch.nn as nn
import numpy as np
from utils.headdropout import HeadDropout

class Model(nn.Module):
    """
    Hybrid model that combines XGBoost-style decision logic with neural networks
    Uses decision tree-like routing but with differentiable operations
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.configs = configs
        self.num_experts = configs.t_dim
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        
        # Gradient-boosting inspired sequential experts
        self.sequential_experts = nn.ModuleList([
            self._create_boosting_expert(configs, expert_id)
            for expert_id in range(self.num_experts)
        ])
        
        # Random forest inspired parallel experts
        self.parallel_experts = nn.ModuleList([
            self._create_forest_expert(configs, expert_id)
            for expert_id in range(self.num_experts)
        ])
        
        # Temporal gating (like XGBoost feature selection)
        self.feature_selector = nn.Sequential(
            nn.Linear(4, configs.t_dim * configs.enc_in),
            nn.ReLU(),
            nn.Linear(configs.t_dim * configs.enc_in, configs.t_dim * configs.enc_in),
            nn.Dropout(0.1)
        )
        
        # Ensemble combiner (like XGBoost final prediction)
        self.ensemble_weights = nn.Parameter(torch.ones(self.num_experts * 2))
        self.head_dropout = HeadDropout(configs.head_dropout)
        
    def _create_boosting_expert(self, configs, expert_id):
        """Create expert that mimics gradient boosting behavior"""
        return nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(configs.d_model // 2, configs.d_model // 4),
            nn.ReLU(),
            nn.Linear(configs.d_model // 4, configs.pred_len),
        )
    
    def _create_forest_expert(self, configs, expert_id):
        """Create expert that mimics random forest behavior"""
        # Random feature selection (like random forest)
        feature_subset_size = max(1, configs.seq_len // 3)
        
        return nn.Sequential(
            nn.Linear(feature_subset_size, configs.d_model // 4),
            nn.ReLU(),
            nn.Linear(configs.d_model // 4, configs.pred_len),
        )
    
    def _apply_feature_bagging(self, x, expert_id):
        """Simulate random forest feature bagging"""
        seq_len = x.shape[1]
        feature_subset_size = max(1, seq_len // 3)
        
        # Use expert_id as seed for consistent feature selection
        torch.manual_seed(expert_id + 42)
        indices = torch.randperm(seq_len)[:feature_subset_size]
        torch.manual_seed(torch.initial_seed())  # Reset seed
        
        return x[:, indices, :]
    
    def forward(self, x, x_mark, return_gating_weights=False, return_seperate_head=False):
        # x: [Batch, Input length, Channel]
        x_mark_initial = x_mark[:, 0]
        batch_size = x.shape[0]
        
        # Feature selection and gating (XGBoost-style)
        temporal_features = self.feature_selector(x_mark_initial)
        temporal_features = temporal_features.reshape(batch_size, self.num_experts, self.channels)
        temporal_features = self.head_dropout(temporal_features)
        temporal_weights = torch.softmax(temporal_features, dim=1)
        
        # Sequential experts (Gradient Boosting style)
        sequential_outputs = []
        residual = x  # [B, L, C] - input sequence
        
        for i, expert in enumerate(self.sequential_experts):
            # Each expert learns from residuals (like gradient boosting)
            expert_input = residual.transpose(1, 2)  # [B, C, L]
            expert_pred = expert(expert_input).transpose(1, 2)  # [B, pred_len, C]
            sequential_outputs.append(expert_pred)
            
            # Update residual for next expert (boosting behavior)
            # Only update if we have more experts and shapes are compatible
            if i < len(self.sequential_experts) - 1:
                # Project prediction back to input sequence length for residual update
                if expert_pred.shape[1] != residual.shape[1]:
                    # Interpolate prediction to match input sequence length
                    pred_resized = torch.nn.functional.interpolate(
                        expert_pred.transpose(1, 2), 
                        size=residual.shape[1], 
                        mode='linear', 
                        align_corners=False
                    ).transpose(1, 2)
                    residual = residual - 0.1 * pred_resized  # Small learning rate
                else:
                    residual = residual - 0.1 * expert_pred
        
        # Parallel experts (Random Forest style)
        parallel_outputs = []
        for i, expert in enumerate(self.parallel_experts):
            # Apply feature bagging
            bagged_x = self._apply_feature_bagging(x, i)
            expert_input = bagged_x.transpose(1, 2)  # [B, C, reduced_seq_len]
            expert_pred = expert(expert_input).transpose(1, 2)  # [B, pred_len, C]
            
            # Ensure output matches prediction length
            if expert_pred.shape[1] != self.pred_len:
                expert_pred = torch.nn.functional.interpolate(
                    expert_pred.transpose(1, 2), 
                    size=self.pred_len, 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2)
            
            parallel_outputs.append(expert_pred)
        
        # Combine all expert outputs
        all_outputs = sequential_outputs + parallel_outputs
        stacked_outputs = torch.stack(all_outputs, dim=1)  # [B, num_experts*2, pred_len, C]
        
        # Ensemble weighting (like XGBoost final combination)
        ensemble_weights = torch.softmax(self.ensemble_weights, dim=0)
        weighted_output = torch.sum(
            stacked_outputs * ensemble_weights.view(1, -1, 1, 1), 
            dim=1
        )
        
        # Apply temporal gating for final adjustment
        temporal_adjustment = temporal_weights.mean(dim=1, keepdim=True)  # [B, 1, C]
        final_output = weighted_output * temporal_adjustment.unsqueeze(1)  # [B, pred_len, C]
        
        # Ensure output shape is correct: [B, pred_len, C]
        if final_output.shape[1] != self.pred_len:
            final_output = torch.nn.functional.interpolate(
                final_output.transpose(1, 2), 
                size=self.pred_len, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        
        if return_gating_weights:
            return final_output, temporal_weights
        
        return final_output


# Usage in the model dictionary would be:
# 'MoLE_XGBoost': MoLE_XGBoost_Hybrid,