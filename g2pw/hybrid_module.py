"""
Hybrid BERT-Qwen G2PW Module
Based on original g2pW but enhanced with Qwen components
"""

import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel, BertPreTrainedModel, BertConfig

from .module import ModifiedFocalLoss
from .hybrid_components import HybridBertEncoder


class HybridG2PW(BertPreTrainedModel):
    """
    Hybrid G2PW model combining BERT's proven performance with Qwen's innovations
    Based on the original G2PW architecture but enhanced with:
    - RMSNorm (more stable than LayerNorm)
    - SwiGLU FFN (better than GELU)
    - Optional RoPE (better positional encoding)
    """
    
    def __init__(self, config, labels, chars, pos_tags,
                 use_conditional=False, param_conditional=None,
                 use_focal=False, param_focal=None,
                 use_pos=False, param_pos=None,
                 # Hybrid enhancements
                 use_rope=True, use_rmsnorm=True, use_swiglu=True):
        
        # Initialize with BERT config
        if isinstance(config, str):
            config = BertConfig.from_pretrained(config)
        super().__init__(config)

        self.num_labels = len(labels)
        self.num_chars = len(chars)
        self.num_pos_tags = len(pos_tags)
        
        # Store enhancement flags
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.use_swiglu = use_swiglu

        # Create BERT model with hybrid encoder
        self.bert = BertModel(config)
        
        # Replace encoder with hybrid version
        if any([use_rope, use_rmsnorm, use_swiglu]):
            print(f"🔧 Creating Hybrid BERT with Qwen enhancements:")
            print(f"  - RoPE: {use_rope}")
            print(f"  - RMSNorm: {use_rmsnorm}")
            print(f"  - SwiGLU: {use_swiglu}")

            self.bert.encoder = HybridBertEncoder(
                config, use_rope, use_rmsnorm, use_swiglu
            )

            # Enable gradient checkpointing for memory saving
            self.bert.encoder.gradient_checkpointing = True
            print(f"  - Gradient checkpointing: Enabled")

        # Classification head
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

        # Conditional mechanism (from original G2PW)
        self.use_conditional = use_conditional
        self.param_conditional = param_conditional or {}
        if self.use_conditional:
            self._setup_conditional_mechanism(labels, chars, pos_tags)

        # Focal loss (from original G2PW)
        self.use_focal = use_focal
        self.param_focal = param_focal or {}

        # POS joint training (from original G2PW)
        self.use_pos = use_pos
        self.param_pos = param_pos or {}
        if self.use_pos and self.param_pos.get('pos_joint_training', False):
            self.pos_classifier = nn.Linear(self.config.hidden_size, self.num_pos_tags)

    def _setup_conditional_mechanism(self, labels, chars, pos_tags):
        """Setup conditional mechanism from original G2PW"""
        conditional_affect_location = self.param_conditional.get('affect_location', 'softmax')
        target_size = self.config.hidden_size if conditional_affect_location == 'emb' else self.num_labels

        if self.param_conditional.get('bias', False):
            self.descriptor_bias = nn.Embedding(1, target_size)
        if self.param_conditional.get('char-linear', False):
            self.char_descriptor = nn.Embedding(self.num_chars, target_size)
        if self.param_conditional.get('pos-linear', False):
            self.pos_descriptor = nn.Embedding(self.num_pos_tags, target_size)
        if self.param_conditional.get('char+pos-second', False):
            self.second_order_descriptor = nn.Embedding(self.num_chars * self.num_pos_tags, target_size)
        if self.param_conditional.get('char+pos-second_lowrank', False):
            assert not self.param_conditional.get('char+pos-second', False)
            assert 0 < self.param_conditional.get('lowrank_size', 0) < target_size
            self.second_lowrank_descriptor = nn.Sequential(
                nn.Embedding(self.num_chars * self.num_pos_tags, self.param_conditional['lowrank_size']),
                nn.Linear(self.param_conditional['lowrank_size'], target_size)
            )
        if self.param_conditional.get('char+pos-second_fm', False):
            assert not self.param_conditional.get('char+pos-second', False)
            assert 0 < self.param_conditional.get('fm_size', 0)
            self.second_fm_char_emb = nn.Sequential(
                nn.Embedding(self.num_chars, self.param_conditional['fm_size'] * target_size),
                nn.Unflatten(1, (target_size, self.param_conditional['fm_size']))
            )
            self.second_fm_pos_emb = nn.Sequential(
                nn.Embedding(self.num_pos_tags, self.param_conditional['fm_size'] * target_size),
                nn.Unflatten(1, (target_size, self.param_conditional['fm_size']))
            )
        if self.param_conditional.get('fix_mode'):
            self._setup_fix_mode(labels, chars, pos_tags)

    def _setup_fix_mode(self, labels, chars, pos_tags):
        """Setup fix mode from original G2PW"""
        assert all([not self.param_conditional.get(x, False) for x in 
                   ['bias', 'char-linear', 'pos-linear', 'char+pos-second', 
                    'char+pos-second_lowrank', 'char+pos-second_fm']])
        assert self.param_conditional.get('affect_location') == 'softmax'
        
        count_dict = json.load(open(self.param_conditional['count_json']))
        if self.param_conditional['fix_mode'] == 'count_distr:char':
            char_fix_count = torch.tensor(
                [[count_dict['by_char'][char].get(label, 0.) for label in labels] for char in chars]
            )
            self.char_fix_emb = nn.parameter.Parameter(
                char_fix_count / char_fix_count.sum(dim=-1, keepdim=True),
                requires_grad=False)
        elif self.param_conditional['fix_mode'] == 'count_distr:char+pos':
            char_pos_fix_count = torch.tensor(
                [[count_dict['by_char_pos'][f'{char}-{pos}'].get(label, 0.)
                  if f'{char}-{pos}' in count_dict['by_char_pos'] else 0.
                  for label in labels]
                 for char in chars for pos in pos_tags]
            )
            self.char_pos_fix_emb = nn.parameter.Parameter(
                char_pos_fix_count / char_pos_fix_count.sum(dim=-1, keepdim=True),
                requires_grad=False)
        else:
            raise Exception(f"Unknown fix_mode: {self.param_conditional['fix_mode']}")

    def _weighted_softmax(self, logits, weights, eps=1e-6):
        """Weighted softmax from original G2PW"""
        max_logits, _ = torch.max(logits, dim=-1, keepdim=True)
        weighted_exp_logits = torch.exp(logits - max_logits) * weights
        norm = torch.sum(weighted_exp_logits, dim=-1, keepdim=True)
        probs = weighted_exp_logits / norm
        probs = torch.clamp(probs, min=eps, max=1-eps)
        return probs

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                phoneme_mask=None, char_ids=None, position_ids=None,
                label_ids=None, pos_ids=None):

        # Fix position_ids dimension issue
        # position_ids should be [batch_size, seq_len] but we have [batch_size]
        # We need to extract the character position from the sequence
        batch_size, seq_len = input_ids.shape

        if position_ids is not None and position_ids.dim() == 1:
            # position_ids contains the target character position for each sample
            # We need to create proper position_ids for BERT
            bert_position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        else:
            bert_position_ids = position_ids

        # BERT encoding with hybrid enhancements
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=bert_position_ids
        )
        
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]

        # Extract target character representations
        batch_size, seq_len, _ = sequence_output.size()

        if position_ids is not None and position_ids.dim() == 1:
            # position_ids contains the target character position for each sample in the batch
            # Extract the hidden states at those positions
            batch_indices = torch.arange(batch_size, device=sequence_output.device)
            char_output = sequence_output[batch_indices, position_ids]  # [batch_size, hidden_size]
        else:
            # Fallback: use the first token (CLS)
            char_output = sequence_output[:, 0, :]  # [batch_size, hidden_size]
        
        # Classification
        logits = self.classifier(char_output)  # [batch_size, num_labels]
        
        # Apply conditional mechanism
        if self.use_conditional:
            logits = self._apply_conditional_mechanism(logits, char_ids, pos_ids)
        
        # Apply phoneme mask
        if phoneme_mask is not None:
            logits = logits.masked_fill(~phoneme_mask.bool(), -1e9)
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # POS prediction
        pos_logits = None
        if self.use_pos and hasattr(self, 'pos_classifier'):
            pos_logits = self.pos_classifier(char_output)
        
        # Calculate loss
        loss = None
        if label_ids is not None:
            if self.use_focal:
                # Use modified focal loss from original G2PW
                focal_loss = ModifiedFocalLoss(
                    alpha=self.param_focal.get('alpha', 0),
                    gamma=self.param_focal.get('gamma', 0.7),
                    reduction='mean'
                )
                loss = focal_loss(probs, label_ids)
            else:
                loss = F.cross_entropy(logits, label_ids)
            
            # Add POS loss if available
            if self.use_pos and pos_ids is not None and pos_logits is not None:
                pos_loss = F.cross_entropy(pos_logits, pos_ids)
                pos_weight = self.param_pos.get('weight', 0.1)
                loss = loss + pos_weight * pos_loss
        
        return probs, loss, pos_logits

    def _apply_conditional_mechanism(self, logits, char_ids, pos_ids):
        """Apply conditional mechanism from original G2PW"""
        conditional_affect_location = self.param_conditional.get('affect_location', 'softmax')
        
        if conditional_affect_location == 'softmax':
            weights = torch.ones_like(logits)
            
            if self.param_conditional.get('bias', False):
                bias_weight = self.descriptor_bias(torch.zeros_like(char_ids))
                weights = weights + bias_weight
            
            if self.param_conditional.get('char-linear', False):
                char_weight = self.char_descriptor(char_ids)
                weights = weights + char_weight
            
            if self.param_conditional.get('pos-linear', False) and pos_ids is not None:
                pos_weight = self.pos_descriptor(pos_ids)
                weights = weights + pos_weight
            
            if self.param_conditional.get('char+pos-second', False) and pos_ids is not None:
                second_order_ids = char_ids * self.num_pos_tags + pos_ids
                second_weight = self.second_order_descriptor(second_order_ids)
                weights = weights + second_weight
            
            if self.param_conditional.get('char+pos-second_lowrank', False) and pos_ids is not None:
                second_order_ids = char_ids * self.num_pos_tags + pos_ids
                second_weight = self.second_lowrank_descriptor(second_order_ids)
                weights = weights + second_weight
            
            if self.param_conditional.get('char+pos-second_fm', False) and pos_ids is not None:
                char_fm_emb = self.second_fm_char_emb(char_ids)
                pos_fm_emb = self.second_fm_pos_emb(pos_ids)
                second_weight = torch.sum(char_fm_emb * pos_fm_emb, dim=-1)
                weights = weights + second_weight
            
            if self.param_conditional.get('fix_mode'):
                if self.param_conditional['fix_mode'] == 'count_distr:char':
                    fix_weight = self.char_fix_emb[char_ids]
                    weights = weights * fix_weight
                elif self.param_conditional['fix_mode'] == 'count_distr:char+pos' and pos_ids is not None:
                    fix_ids = char_ids * self.num_pos_tags + pos_ids
                    fix_weight = self.char_pos_fix_emb[fix_ids]
                    weights = weights * fix_weight
            
            # Apply weighted softmax
            probs = self._weighted_softmax(logits, weights)
            # Convert back to logits for loss calculation
            logits = torch.log(probs + 1e-8)
        
        return logits

    @classmethod
    def from_pretrained_bert(cls, bert_model_path, labels, chars, pos_tags, **kwargs):
        """Create hybrid model from pretrained BERT"""
        config = BertConfig.from_pretrained(bert_model_path)
        model = cls(config, labels, chars, pos_tags, **kwargs)
        
        # Load BERT weights
        bert_model = BertModel.from_pretrained(bert_model_path)
        
        # Transfer compatible weights
        model.bert.embeddings.load_state_dict(bert_model.embeddings.state_dict())
        model.bert.pooler.load_state_dict(bert_model.pooler.state_dict())
        
        # Transfer encoder weights if not using hybrid components
        if not any([kwargs.get('use_rope', True), kwargs.get('use_rmsnorm', True), kwargs.get('use_swiglu', True)]):
            model.bert.encoder.load_state_dict(bert_model.encoder.state_dict())
        else:
            print("🔄 Hybrid components enabled - encoder weights will be partially transferred during training")
        
        print(f"✓ Hybrid G2PW model created from {bert_model_path}")
        return model
