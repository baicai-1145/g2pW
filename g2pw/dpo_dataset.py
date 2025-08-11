"""
DPO (Direct Preference Optimization) Dataset for G2PW
Generate preference pairs for better training
"""

import torch
from torch.utils.data import Dataset
from collections import defaultdict, Counter
import random
from .dataset import prepare_data, prepare_pos, get_phoneme_labels, ANCHOR_CHAR


class DPODatasetGenerator:
    """Generate DPO preference pairs from G2PW dataset"""
    
    def __init__(self, sent_path, lb_path, pos_path):
        self.sent_path = sent_path
        self.lb_path = lb_path
        self.pos_path = pos_path
        
        # Load data
        self.texts, self.query_ids, self.phonemes = prepare_data(sent_path, lb_path)
        self.pos_tags = prepare_pos(pos_path)
        
        # Analyze character-phoneme mappings
        self.char_phoneme_stats = self._analyze_char_phoneme_mappings()
        self.char_pos_stats = self._analyze_char_pos_mappings()
        
        print(f"📊 DPO Dataset Analysis:")
        print(f"  - Total samples: {len(self.texts)}")
        print(f"  - Characters with multiple phonemes: {len(self.char_phoneme_stats)}")
        print(f"  - Characters with multiple POS: {len(self.char_pos_stats)}")
    
    def _analyze_char_phoneme_mappings(self):
        """Analyze character-phoneme mappings to find alternatives"""
        char_phoneme_map = defaultdict(Counter)
        
        for i, (text, query_id, phoneme) in enumerate(zip(self.texts, self.query_ids, self.phonemes)):
            # Extract target character
            raw_text = text[:query_id] + ANCHOR_CHAR + text[query_id:]
            target_char = text[query_id] if query_id < len(text) else None
            
            if target_char:
                char_phoneme_map[target_char][phoneme] += 1
        
        # Filter characters with multiple phonemes
        multi_phoneme_chars = {}
        for char, phoneme_counts in char_phoneme_map.items():
            if len(phoneme_counts) > 1:
                multi_phoneme_chars[char] = dict(phoneme_counts)
        
        return multi_phoneme_chars
    
    def _analyze_char_pos_mappings(self):
        """Analyze character-POS mappings to find alternatives"""
        char_pos_map = defaultdict(Counter)
        
        for i, (text, query_id, pos_tag) in enumerate(zip(self.texts, self.query_ids, self.pos_tags)):
            # Extract target character
            target_char = text[query_id] if query_id < len(text) else None
            
            if target_char:
                char_pos_map[target_char][pos_tag] += 1
        
        # Filter characters with multiple POS tags
        multi_pos_chars = {}
        for char, pos_counts in char_pos_map.items():
            if len(pos_counts) > 1:
                multi_pos_chars[char] = dict(pos_counts)
        
        return multi_pos_chars
    
    def generate_phoneme_preference_pairs(self):
        """Generate phoneme preference pairs"""
        preference_pairs = []
        
        for i, (text, query_id, correct_phoneme) in enumerate(zip(self.texts, self.query_ids, self.phonemes)):
            target_char = text[query_id] if query_id < len(text) else None
            
            if target_char and target_char in self.char_phoneme_stats:
                # Get alternative phonemes for this character
                alternative_phonemes = list(self.char_phoneme_stats[target_char].keys())
                alternative_phonemes = [p for p in alternative_phonemes if p != correct_phoneme]
                
                if alternative_phonemes:
                    # Create preference pairs
                    for wrong_phoneme in alternative_phonemes:
                        preference_pairs.append({
                            'text': text,
                            'query_id': query_id,
                            'target_char': target_char,
                            'chosen_phoneme': correct_phoneme,
                            'rejected_phoneme': wrong_phoneme,
                            'pos_tag': self.pos_tags[i] if i < len(self.pos_tags) else 'UNK',
                            'type': 'phoneme'
                        })
        
        return preference_pairs
    
    def generate_pos_preference_pairs(self):
        """Generate POS preference pairs"""
        preference_pairs = []
        all_pos_tags = ['UNK', 'A', 'C', 'D', 'I', 'N', 'P', 'T', 'V', 'DE', 'SHI']
        
        for i, (text, query_id, correct_pos) in enumerate(zip(self.texts, self.query_ids, self.pos_tags)):
            target_char = text[query_id] if query_id < len(text) else None
            
            if target_char:
                # Get alternative POS tags
                wrong_pos_tags = [pos for pos in all_pos_tags if pos != correct_pos]
                
                # Create preference pairs for each wrong POS
                for wrong_pos in wrong_pos_tags:
                    preference_pairs.append({
                        'text': text,
                        'query_id': query_id,
                        'target_char': target_char,
                        'chosen_pos': correct_pos,
                        'rejected_pos': wrong_pos,
                        'phoneme': self.phonemes[i],
                        'type': 'pos'
                    })
        
        return preference_pairs
    
    def generate_all_preference_pairs(self, max_pairs_per_sample=3):
        """Generate all preference pairs with sampling"""
        phoneme_pairs = self.generate_phoneme_preference_pairs()
        pos_pairs = self.generate_pos_preference_pairs()
        
        # Sample to avoid too many pairs
        if len(phoneme_pairs) > len(self.texts) * max_pairs_per_sample:
            phoneme_pairs = random.sample(phoneme_pairs, len(self.texts) * max_pairs_per_sample)
        
        if len(pos_pairs) > len(self.texts) * max_pairs_per_sample:
            pos_pairs = random.sample(pos_pairs, len(self.texts) * max_pairs_per_sample)
        
        all_pairs = phoneme_pairs + pos_pairs
        random.shuffle(all_pairs)
        
        print(f"📊 Generated DPO preference pairs:")
        print(f"  - Phoneme pairs: {len(phoneme_pairs)}")
        print(f"  - POS pairs: {len(pos_pairs)}")
        print(f"  - Total pairs: {len(all_pairs)}")
        
        return all_pairs


class DPODataset(Dataset):
    """DPO Dataset for preference learning"""
    
    def __init__(self, tokenizer, preference_pairs, labels, char2phonemes, chars, 
                 max_len=512, use_pos=True):
        self.tokenizer = tokenizer
        self.preference_pairs = preference_pairs
        self.labels = labels
        self.char2phonemes = char2phonemes
        self.chars = chars
        self.max_len = max_len
        self.use_pos = use_pos
        
        # Create label mappings
        self.label2id = {label: i for i, label in enumerate(labels)}
        self.char2id = {char: i for i, char in enumerate(chars)}
        self.pos_tags = ['UNK', 'A', 'C', 'D', 'I', 'N', 'P', 'T', 'V', 'DE', 'SHI']
        self.pos2id = {pos: i for i, pos in enumerate(self.pos_tags)}
    
    def __len__(self):
        return len(self.preference_pairs)
    
    def __getitem__(self, idx):
        pair = self.preference_pairs[idx]

        # Tokenize text
        text = pair['text']
        query_id = pair['query_id']

        # Add special token at query position
        text_with_anchor = text[:query_id] + ANCHOR_CHAR + text[query_id:]

        # Tokenize
        encoding = self.tokenizer(
            text_with_anchor,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )

        # Find anchor position in tokenized text
        anchor_token_id = self.tokenizer.convert_tokens_to_ids(ANCHOR_CHAR)
        input_ids = encoding['input_ids'].squeeze(0)  # Remove batch dimension
        anchor_positions = (input_ids == anchor_token_id).nonzero(as_tuple=True)[0]

        if len(anchor_positions) > 0:
            position_id = anchor_positions[0].item()
        else:
            position_id = min(query_id, len(input_ids) - 1)  # Safe fallback

        # Prepare labels and masks
        target_char = pair['target_char']
        char_id = self.char2id.get(target_char, 0)

        # Create phoneme mask
        if target_char in self.char2phonemes:
            phoneme_mask = torch.zeros(len(self.labels), dtype=torch.bool)
            for phoneme_idx in self.char2phonemes[target_char]:
                phoneme_mask[phoneme_idx] = True
        else:
            phoneme_mask = torch.ones(len(self.labels), dtype=torch.bool)

        # Prepare chosen and rejected labels
        if pair['type'] == 'phoneme':
            chosen_label = self.label2id.get(pair['chosen_phoneme'], 0)
            rejected_label = self.label2id.get(pair['rejected_phoneme'], 0)
            pos_id = self.pos2id.get(pair['pos_tag'], 0)
        else:  # POS type
            chosen_label = self.label2id.get(pair['phoneme'], 0)
            rejected_label = chosen_label  # Same phoneme, different POS
            pos_id = self.pos2id.get(pair['chosen_pos'], 0)

        return {
            'input_ids': input_ids,  # [seq_len]
            'token_type_ids': encoding['token_type_ids'].squeeze(0),  # [seq_len]
            'attention_mask': encoding['attention_mask'].squeeze(0),  # [seq_len]
            'phoneme_mask': phoneme_mask,  # [num_labels]
            'char_ids': torch.tensor(char_id, dtype=torch.long),  # scalar
            'position_ids': torch.tensor(position_id, dtype=torch.long),  # scalar
            'chosen_label_ids': torch.tensor(chosen_label, dtype=torch.long),  # scalar
            'rejected_label_ids': torch.tensor(rejected_label, dtype=torch.long),  # scalar
            'pos_ids': torch.tensor(pos_id, dtype=torch.long),  # scalar
            'pair_type': pair['type']
        }


def create_dpo_dataset(sent_path, lb_path, pos_path, tokenizer, labels, char2phonemes, chars):
    """Create DPO dataset from G2PW data"""
    
    # Generate preference pairs
    generator = DPODatasetGenerator(sent_path, lb_path, pos_path)
    preference_pairs = generator.generate_all_preference_pairs()
    
    # Create dataset
    dataset = DPODataset(
        tokenizer=tokenizer,
        preference_pairs=preference_pairs,
        labels=labels,
        char2phonemes=char2phonemes,
        chars=chars
    )
    
    return dataset


def dpo_collate_fn(batch):
    """Collate function for DPO dataset"""
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        if key == 'pair_type':
            collated[key] = [item[key] for item in batch]
        elif key in ['input_ids', 'token_type_ids', 'attention_mask']:
            # These are sequences, stack them
            collated[key] = torch.stack([item[key] for item in batch])
        elif key == 'phoneme_mask':
            # Stack phoneme masks
            collated[key] = torch.stack([item[key] for item in batch])
        else:
            # These are scalars, stack them
            collated[key] = torch.stack([item[key] for item in batch])

    return collated
