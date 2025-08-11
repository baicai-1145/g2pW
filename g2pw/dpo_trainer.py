"""
DPO (Direct Preference Optimization) Trainer for G2PW
Implements preference learning to improve model accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime


class DPOLoss(nn.Module):
    """DPO Loss function"""
    
    def __init__(self, beta=0.1, reference_free=False):
        super().__init__()
        self.beta = beta
        self.reference_free = reference_free
    
    def forward(self, policy_chosen_logps, policy_rejected_logps, 
                reference_chosen_logps=None, reference_rejected_logps=None):
        """
        Compute DPO loss
        
        Args:
            policy_chosen_logps: Log probabilities of chosen responses from policy model
            policy_rejected_logps: Log probabilities of rejected responses from policy model
            reference_chosen_logps: Log probabilities of chosen responses from reference model
            reference_rejected_logps: Log probabilities of rejected responses from reference model
        """
        
        if self.reference_free:
            # Reference-free DPO (simpler)
            logits = self.beta * (policy_chosen_logps - policy_rejected_logps)
        else:
            # Standard DPO with reference model
            if reference_chosen_logps is None or reference_rejected_logps is None:
                raise ValueError("Reference log probabilities required for standard DPO")
            
            policy_ratio = policy_chosen_logps - policy_rejected_logps
            reference_ratio = reference_chosen_logps - reference_rejected_logps
            logits = self.beta * (policy_ratio - reference_ratio)
        
        # DPO loss: -log(sigmoid(logits))
        loss = -F.logsigmoid(logits).mean()
        
        # Compute accuracy (how often chosen > rejected)
        accuracy = (policy_chosen_logps > policy_rejected_logps).float().mean()
        
        return loss, accuracy, logits.mean()


class DPOTrainer:
    """DPO Trainer for G2PW models"""
    
    def __init__(self, model, reference_model=None, beta=0.1, 
                 learning_rate=1e-5, weight_decay=0.01):
        self.model = model
        self.reference_model = reference_model
        self.beta = beta
        
        # Setup DPO loss
        self.dpo_loss = DPOLoss(beta=beta, reference_free=(reference_model is None))
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Move models to device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        if self.reference_model:
            self.reference_model.to(self.device)
            self.reference_model.eval()  # Keep reference model frozen
        
        print(f"🎯 DPO Trainer initialized:")
        print(f"  - Beta: {beta}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Reference model: {'Yes' if reference_model else 'No (reference-free)'}")
        print(f"  - Device: {self.device}")
    
    def get_log_probabilities(self, model, batch):
        """Get log probabilities for chosen and rejected responses"""

        # Debug info (can be removed later)
        # print(f"🔍 DEBUG: Batch shapes:")
        # for key, value in batch.items():
        #     if isinstance(value, torch.Tensor):
        #         print(f"  - {key}: {value.shape}")

        # Forward pass for chosen labels
        with torch.no_grad() if model == self.reference_model else torch.enable_grad():
            probs_chosen, _, _ = model(
                input_ids=batch['input_ids'],
                token_type_ids=batch['token_type_ids'],
                attention_mask=batch['attention_mask'],
                phoneme_mask=batch['phoneme_mask'],
                char_ids=batch['char_ids'],
                position_ids=batch['position_ids'],
                label_ids=batch['chosen_label_ids'],  # Pass chosen labels
                pos_ids=batch.get('pos_ids')
            )

        # Forward pass for rejected labels
        with torch.no_grad() if model == self.reference_model else torch.enable_grad():
            probs_rejected, _, _ = model(
                input_ids=batch['input_ids'],
                token_type_ids=batch['token_type_ids'],
                attention_mask=batch['attention_mask'],
                phoneme_mask=batch['phoneme_mask'],
                char_ids=batch['char_ids'],
                position_ids=batch['position_ids'],
                label_ids=batch['rejected_label_ids'],  # Pass rejected labels
                pos_ids=batch.get('pos_ids')
            )

        # Get log probabilities for chosen and rejected labels
        # probs_chosen/rejected shape: [batch_size, num_labels]
        # chosen/rejected_label_ids shape: [batch_size]
        chosen_indices = batch['chosen_label_ids'].unsqueeze(1)  # [batch_size, 1]
        rejected_indices = batch['rejected_label_ids'].unsqueeze(1)  # [batch_size, 1]

        chosen_logps = torch.log(probs_chosen.gather(1, chosen_indices) + 1e-8).squeeze(1)
        rejected_logps = torch.log(probs_rejected.gather(1, rejected_indices) + 1e-8).squeeze(1)

        return chosen_logps, rejected_logps
    
    def train_step(self, batch):
        """Single training step"""
        self.model.train()

        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)

        # Get policy model log probabilities
        policy_chosen_logps, policy_rejected_logps = self.get_log_probabilities(self.model, batch)

        # Get reference model log probabilities (if available)
        reference_chosen_logps, reference_rejected_logps = None, None
        if self.reference_model:
            reference_chosen_logps, reference_rejected_logps = self.get_log_probabilities(self.reference_model, batch)

        # Compute DPO loss
        loss, accuracy, logits_mean = self.dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'logits_mean': logits_mean.item(),
            'policy_chosen_mean': policy_chosen_logps.mean().item(),
            'policy_rejected_mean': policy_rejected_logps.mean().item()
        }
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"DPO Epoch {epoch}")
        
        for batch in progress_bar:
            try:
                metrics = self.train_step(batch)
                
                total_loss += metrics['loss']
                total_accuracy += metrics['accuracy']
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'acc': f"{metrics['accuracy']:.3f}",
                    'logits': f"{metrics['logits_mean']:.3f}"
                })
                
            except Exception as e:
                print(f"Error in batch: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_accuracy = total_accuracy / max(num_batches, 1)
        
        return {
            'avg_loss': avg_loss,
            'avg_accuracy': avg_accuracy,
            'num_batches': num_batches
        }
    
    def evaluate(self, dataloader):
        """Evaluate model on validation set"""
        self.model.eval()
        
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="DPO Evaluation"):
                try:
                    # Move batch to device
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(self.device)
                    
                    # Get log probabilities
                    policy_chosen_logps, policy_rejected_logps = self.get_log_probabilities(self.model, batch)
                    
                    reference_chosen_logps, reference_rejected_logps = None, None
                    if self.reference_model:
                        reference_chosen_logps, reference_rejected_logps = self.get_log_probabilities(self.reference_model, batch)
                    
                    # Compute loss
                    loss, accuracy, _ = self.dpo_loss(
                        policy_chosen_logps, policy_rejected_logps,
                        reference_chosen_logps, reference_rejected_logps
                    )
                    
                    total_loss += loss.item()
                    total_accuracy += accuracy.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"Error in evaluation batch: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_accuracy = total_accuracy / max(num_batches, 1)
        
        return {
            'avg_loss': avg_loss,
            'avg_accuracy': avg_accuracy
        }
    
    def save_model(self, save_path, epoch, metrics):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'beta': self.beta,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, save_path)
        print(f"✓ Model saved to {save_path}")
    
    def load_model(self, load_path):
        """Load model checkpoint"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"✓ Model loaded from {load_path}")
        print(f"  - Epoch: {checkpoint['epoch']}")
        print(f"  - Metrics: {checkpoint['metrics']}")
        
        return checkpoint['epoch'], checkpoint['metrics']


def create_dpo_trainer(model, reference_model=None, config=None):
    """Create DPO trainer with configuration"""
    
    # Default config
    default_config = {
        'beta': 0.1,
        'learning_rate': 1e-5,
        'weight_decay': 0.01
    }
    
    if config:
        default_config.update(config)
    
    trainer = DPOTrainer(
        model=model,
        reference_model=reference_model,
        beta=default_config['beta'],
        learning_rate=default_config['learning_rate'],
        weight_decay=default_config['weight_decay']
    )
    
    return trainer
