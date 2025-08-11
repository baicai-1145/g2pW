"""
Simple launcher for Hybrid BERT-Qwen + DPO training
"""

import os
import sys
import subprocess
from datetime import datetime

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'transformers', 'numpy', 'tqdm', 'pandas', 'tensorboard'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {missing_packages}")
        print(f"Please install them with: pip install {' '.join(missing_packages)}")
        return False
    
    print("✓ All required packages are installed")
    return True

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        'cpp_dataset/POLYPHONIC_CHARS.txt',
        'cpp_dataset/train.sent',
        'cpp_dataset/train.lb',
        'cpp_dataset/train.pos',
        'cpp_dataset/dev.sent',
        'cpp_dataset/dev.lb',
        'cpp_dataset/dev.pos'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required data files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    print("✓ All required data files are present")
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'saved_models',
        'logs',
        'runs',
        'saved_models/Hybrid_BERT_Qwen_DPO',
        'logs/Hybrid_BERT_Qwen_DPO',
        'runs/Hybrid_BERT_Qwen_DPO'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✓ Created necessary directories")

def main():
    """Main launcher function"""
    print("=" * 60)
    print("HYBRID BERT-QWEN + DPO TRAINING LAUNCHER")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    # Check requirements
    print("\n🔍 Checking requirements...")
    if not check_requirements():
        return False
    
    # Check data files
    print("\n📁 Checking data files...")
    if not check_data_files():
        print("\n💡 To download the dataset, run:")
        print("cd cpp_dataset && bash download.sh")
        return False
    
    # Create directories
    print("\n📂 Creating directories...")
    create_directories()
    
    # Launch training
    print("\n🚀 Launching hybrid DPO training...")
    try:
        # Import and run training
        sys.path.insert(0, '.')
        sys.path.insert(0, 'g2pw')
        
        from scripts.train_hybrid_dpo import main as train_main
        success = train_main()
        
        if success:
            print(f"\n🎉 Training completed successfully!")
            return True
        else:
            print(f"\n❌ Training failed!")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n✅ All done! Check the saved_models directory for results.")
    else:
        print(f"\n❌ Setup or training failed. Please check the error messages above.")
