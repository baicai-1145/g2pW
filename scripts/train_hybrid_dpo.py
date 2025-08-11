"""
Hybrid BERT-Qwen with DPO Training Script
Combines BERT's proven performance with Qwen enhancements and DPO preference learning
"""

import sys
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datetime import datetime
import json

# Add paths
sys.path.insert(0, '.')
sys.path.insert(0, 'g2pw')

from g2pw.hybrid_module import HybridG2PW
from g2pw.dpo_dataset import create_dpo_dataset, dpo_collate_fn
from g2pw.dpo_trainer import create_dpo_trainer
from g2pw.dataset import prepare_data, prepare_pos, get_phoneme_labels


def load_polyphonic_chars():
    """Load polyphonic characters"""
    polyphonic_chars = []
    polyphonic_chars_path = 'cpp_dataset/POLYPHONIC_CHARS.txt'

    with open(polyphonic_chars_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and '\t' in line:
                char, phoneme = line.split('\t', 1)
                polyphonic_chars.append((char, phoneme))
            elif line and ' ' in line:
                char, phoneme = line.split(' ', 1)
                polyphonic_chars.append((char, phoneme))

    print(f"✓ Loaded {len(polyphonic_chars)} polyphonic character-phoneme pairs")
    return polyphonic_chars


def create_hybrid_dpo_model(model_type='bert', use_enhancements=True):
    """Create hybrid model with optional enhancements"""
    
    print(f"🔧 Creating Hybrid G2PW Model:")
    print(f"  - Base model: {model_type}")
    print(f"  - Qwen enhancements: {use_enhancements}")
    
    # Load data for model initialization
    polyphonic_chars = load_polyphonic_chars()
    labels, char2phonemes = get_phoneme_labels(polyphonic_chars)
    chars = sorted(list(set([char for char, _ in polyphonic_chars])))
    pos_tags = ['UNK', 'A', 'C', 'D', 'I', 'N', 'P', 'T', 'V', 'DE', 'SHI']
    
    print(f"  - Labels: {len(labels)}")
    print(f"  - Characters: {len(chars)}")
    print(f"  - POS tags: {len(pos_tags)}")
    
    # Model configuration
    if model_type == 'bert':
        bert_model_path = 'bert-base-chinese'
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create hybrid model
    model = HybridG2PW.from_pretrained_bert(
        bert_model_path=bert_model_path,
        labels=labels,
        chars=chars,
        pos_tags=pos_tags,
        # G2PW features
        use_conditional=True,
        param_conditional={
            'affect_location': 'softmax',
            'bias': True,
            'char-linear': True,
            'pos-linear': False,  # Paper optimal
            'char+pos-second': True,
            'char+pos-second_lowrank': False,
            'lowrank_size': 0,
            'char+pos-second_fm': False,
            'fm_size': 0,
            'fix_mode': None
        },
        use_focal=True,
        param_focal={
            'alpha': 1.0,
            'gamma': 2.0
        },
        use_pos=True,
        param_pos={
            'pos_joint_training': True,
            'weight': 0.1
        },
        # Qwen enhancements
        use_rope=use_enhancements,
        use_rmsnorm=use_enhancements,
        use_swiglu=use_enhancements
    )
    
    return model, labels, chars, char2phonemes, pos_tags


def create_datasets_and_loaders(labels, chars, char2phonemes):
    """Create DPO datasets and data loaders"""
    
    print("📊 Creating DPO datasets...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    
    # Create training dataset
    train_dataset = create_dpo_dataset(
        sent_path='cpp_dataset/train.sent',
        lb_path='cpp_dataset/train.lb',
        pos_path='cpp_dataset/train.pos',
        tokenizer=tokenizer,
        labels=labels,
        char2phonemes=char2phonemes,
        chars=chars
    )
    
    # Create validation dataset
    valid_dataset = create_dpo_dataset(
        sent_path='cpp_dataset/dev.sent',
        lb_path='cpp_dataset/dev.lb',
        pos_path='cpp_dataset/dev.pos',
        tokenizer=tokenizer,
        labels=labels,
        char2phonemes=char2phonemes,
        chars=chars
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,  # Smaller batch size for DPO
        shuffle=True,
        collate_fn=dpo_collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=dpo_collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"✓ DPO datasets created:")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Validation samples: {len(valid_dataset)}")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(valid_loader)}")
    
    return train_loader, valid_loader, tokenizer


def train_hybrid_dpo_model(model, train_loader, valid_loader, config):
    """Train hybrid model with DPO"""
    
    print(f"🚀 Starting Hybrid BERT-Qwen + DPO Training:")
    print(f"  - Epochs: {config['epochs']}")
    print(f"  - Learning rate: {config['learning_rate']}")
    print(f"  - DPO beta: {config['dpo_beta']}")
    print(f"  - Use reference model: {config['use_reference_model']}")
    
    # Create reference model if needed
    reference_model = None
    if config['use_reference_model']:
        print("📋 Creating reference model...")
        reference_model, _, _, _, _ = create_hybrid_dpo_model(
            model_type='bert', 
            use_enhancements=False  # Use vanilla BERT as reference
        )
        reference_model.eval()
        print("✓ Reference model created (vanilla BERT)")
    
    # Create DPO trainer
    trainer = create_dpo_trainer(
        model=model,
        reference_model=reference_model,
        config={
            'beta': config['dpo_beta'],
            'learning_rate': config['learning_rate'],
            'weight_decay': config['weight_decay']
        }
    )
    
    # Training loop
    best_accuracy = 0.0
    best_model_path = None
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"HYBRID DPO EPOCH {epoch}/{config['epochs']}")
        print(f"{'='*60}")
        
        # Training
        train_metrics = trainer.train_epoch(train_loader, epoch)
        
        print(f"\n📊 Training Results:")
        print(f"  - Average Loss: {train_metrics['avg_loss']:.4f}")
        print(f"  - Average Accuracy: {train_metrics['avg_accuracy']:.3f}")
        print(f"  - Processed Batches: {train_metrics['num_batches']}")
        
        # Validation
        if epoch % config['eval_interval'] == 0:
            print(f"\n🔍 Validation...")
            valid_metrics = trainer.evaluate(valid_loader)
            
            print(f"📊 Validation Results:")
            print(f"  - Average Loss: {valid_metrics['avg_loss']:.4f}")
            print(f"  - Average Accuracy: {valid_metrics['avg_accuracy']:.3f}")
            
            # Save best model
            if valid_metrics['avg_accuracy'] > best_accuracy:
                best_accuracy = valid_metrics['avg_accuracy']
                best_model_path = f"{config['output_dir']}/best_hybrid_dpo_model.pth"
                trainer.save_model(best_model_path, epoch, valid_metrics)
                print(f"🏆 New best model saved! Accuracy: {best_accuracy:.3f}")
        
        # Save checkpoint
        if epoch % config['save_interval'] == 0:
            checkpoint_path = f"{config['output_dir']}/hybrid_dpo_epoch_{epoch}.pth"
            trainer.save_model(checkpoint_path, epoch, train_metrics)
    
    print(f"\n🎉 Training completed!")
    print(f"  - Best accuracy: {best_accuracy:.3f}")
    print(f"  - Best model: {best_model_path}")
    
    return model, best_model_path, best_accuracy


def main():
    """Main training function"""
    print("=" * 60)
    print("HYBRID BERT-QWEN + DPO TRAINING")
    print("=" * 60)
    print(f"Training started at: {datetime.now()}")
    
    # Configuration
    config = {
        'epochs': 20,
        'learning_rate': 1e-5,
        'weight_decay': 0.01,
        'dpo_beta': 0.1,
        'use_reference_model': True,
        'eval_interval': 2,
        'save_interval': 5,
        'output_dir': 'hybrid_dpo_output',
        'log_dir': 'hybrid_dpo_logs'
    }
    
    # Create output directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Save configuration
    with open(f"{config['output_dir']}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    try:
        # Create hybrid model
        print("\n🔧 Step 1: Creating Hybrid Model...")
        model, labels, chars, char2phonemes, pos_tags = create_hybrid_dpo_model(
            model_type='bert',
            use_enhancements=True
        )
        
        # Create datasets
        print("\n📊 Step 2: Creating DPO Datasets...")
        train_loader, valid_loader, tokenizer = create_datasets_and_loaders(
            labels, chars, char2phonemes
        )
        
        # Train model
        print("\n🚀 Step 3: Starting DPO Training...")
        trained_model, best_model_path, best_accuracy = train_hybrid_dpo_model(
            model, train_loader, valid_loader, config
        )
        
        # Final results
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"✓ Best accuracy: {best_accuracy:.3f}")
        print(f"✓ Best model saved to: {best_model_path}")
        print(f"✓ Training completed at: {datetime.now()}")
        
        # Expected performance improvement
        print(f"\n🎯 Expected Performance:")
        print(f"  - BERT baseline: 98.71%")
        print(f"  - Hybrid + DPO target: 99.0-99.5%")
        print(f"  - Actual result: {best_accuracy*100:.2f}%")
        
        if best_accuracy > 0.99:
            print(f"🏆 EXCELLENT! Achieved target performance!")
        elif best_accuracy > 0.985:
            print(f"🎉 GREAT! Significant improvement achieved!")
        else:
            print(f"📈 Good progress, consider more training or hyperparameter tuning")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎉 Hybrid BERT-Qwen + DPO training completed successfully!")
    else:
        print(f"\n❌ Training failed. Please check the logs.")
