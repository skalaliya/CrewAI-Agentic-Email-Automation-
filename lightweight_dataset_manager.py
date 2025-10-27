#!/usr/bin/env python3
"""
Lightweight dataset manager optimized for limited disk space.
Smart approach to handle multiple email datasets efficiently.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.mail_agents.config import settings


def check_disk_space():
    """Check available disk space and recommend actions."""
    
    import shutil
    
    total, used, free = shutil.disk_usage("/")
    free_gb = free // (1024**3)
    used_percent = (used / total) * 100
    
    print(f"💾 Disk Space Status:")
    print(f"   Free: {free_gb} GB ({100-used_percent:.1f}% available)")
    print(f"   Used: {used_percent:.1f}%")
    
    if free_gb < 1:
        print(f"⚠️  LOW DISK SPACE WARNING!")
        print(f"   Enron dataset (~2GB uncompressed) requires more space")
        print(f"   Proceeding with existing spam dataset only")
        return False
    else:
        print(f"✅ Sufficient space for large datasets")
        return True


def enhance_existing_spam_dataset():
    """Enhance the existing spam dataset with better processing."""
    
    print("\n🔧 Enhancing Existing Spam Dataset")
    print("=" * 50)
    
    data_dir = Path("data")
    spam_file = data_dir / "spam_email_dataset.csv"
    
    if not spam_file.exists():
        print("❌ Spam dataset not found. Run download_kaggle_data.py first")
        return None
    
    # Load existing dataset
    print(f"📖 Loading existing dataset: {spam_file}")
    df = pd.read_csv(spam_file)
    
    print(f"📊 Original dataset shape: {df.shape}")
    print(f"📈 Label distribution: {df['label'].value_counts().to_dict()}")
    
    # Enhanced preprocessing
    print("\n🔧 Applying enhanced preprocessing...")
    
    # 1. Text cleaning and standardization
    df['text_length'] = df['Message'].str.len()
    df['word_count'] = df['Message'].str.split().str.len()
    
    # 2. Enhanced categorization
    df['email_type'] = df['Category'].map({
        'spam': 'spam',
        'ham': 'legitimate',
        '{"mode":"full"': 'malformed'  # Clean up the data issue we saw
    }).fillna('unknown')
    
    # 3. Filter out malformed entries
    df = df[df['email_type'] != 'malformed']
    
    # 4. Add email characteristics
    df['has_urls'] = df['Message'].str.contains(r'http[s]?://', case=False, na=False)
    df['has_money'] = df['Message'].str.contains(r'\$\d+|\d+\s*dollar', case=False, na=False)
    df['urgency_words'] = df['Message'].str.contains(r'urgent|immediately|act now|limited time', case=False, na=False)
    df['all_caps_ratio'] = df['Message'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1))
    
    # 5. Create enhanced samples
    enhanced_samples = create_smart_samples(df)
    
    # 6. Save enhanced dataset
    enhanced_file = data_dir / "enhanced_spam_dataset.csv"
    df.to_csv(enhanced_file, index=False)
    
    print(f"✅ Enhanced dataset created: {df.shape}")
    print(f"💾 Saved to: {enhanced_file}")
    
    return df, enhanced_samples


def create_smart_samples(df):
    """Create intelligent sample sets for different use cases."""
    
    print(f"\n📋 Creating Smart Sample Sets")
    print("=" * 40)
    
    data_dir = Path("data")
    
    # 1. Balanced samples for testing
    spam_samples = df[df['email_type'] == 'spam'].sample(n=min(10, len(df[df['email_type'] == 'spam'])), random_state=42)
    ham_samples = df[df['email_type'] == 'legitimate'].sample(n=min(10, len(df[df['email_type'] == 'legitimate'])), random_state=42)
    
    balanced_samples = pd.concat([spam_samples, ham_samples], ignore_index=True)
    balanced_file = data_dir / "balanced_test_samples.csv"
    balanced_samples.to_csv(balanced_file, index=False)
    
    # 2. Edge case samples (unusual characteristics)
    edge_cases = df[
        (df['text_length'] > df['text_length'].quantile(0.95)) |  # Very long emails
        (df['text_length'] < df['text_length'].quantile(0.05)) |  # Very short emails
        (df['all_caps_ratio'] > 0.3) |  # Lots of caps
        (df['has_urls'] & df['has_money'])  # URLs + money mentions
    ].sample(n=min(20, len(df)), random_state=42)
    
    edge_file = data_dir / "edge_case_samples.csv"
    edge_cases.to_csv(edge_file, index=False)
    
    # 3. High-quality samples for CrewAI testing
    quality_samples = df[
        (df['text_length'] >= 50) &  # Not too short
        (df['text_length'] <= 1000) &  # Not too long
        (df['word_count'] >= 10)  # Substantial content
    ].sample(n=min(30, len(df)), random_state=42)
    
    quality_file = data_dir / "quality_test_samples.csv"
    quality_samples.to_csv(quality_file, index=False)
    
    print(f"✅ Created sample sets:")
    print(f"   📊 Balanced samples: {len(balanced_samples)} ({balanced_file})")
    print(f"   ⚡ Edge cases: {len(edge_cases)} ({edge_file})")
    print(f"   🎯 Quality samples: {len(quality_samples)} ({quality_file})")
    
    return {
        'balanced': balanced_samples,
        'edge_cases': edge_cases,
        'quality': quality_samples
    }


def prepare_ml_training_data(df):
    """Prepare data specifically for ML model training."""
    
    print(f"\n🤖 Preparing ML Training Data")
    print("=" * 40)
    
    data_dir = Path("data")
    
    # Create training/validation split
    from sklearn.model_selection import train_test_split
    
    # Prepare features and labels
    X = df['Message']
    y = df['email_type']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create training datasets
    train_data = pd.DataFrame({
        'text': X_train,
        'label': y_train,
        'split': 'train'
    })
    
    test_data = pd.DataFrame({
        'text': X_test,
        'label': y_test,
        'split': 'test'
    })
    
    # Combine and save
    ml_data = pd.concat([train_data, test_data], ignore_index=True)
    ml_file = data_dir / "ml_training_data.csv"
    ml_data.to_csv(ml_file, index=False)
    
    print(f"✅ ML training data prepared:")
    print(f"   📚 Training samples: {len(train_data)}")
    print(f"   🧪 Test samples: {len(test_data)}")
    print(f"   💾 Saved to: {ml_file}")
    
    return ml_data


def test_enhanced_samples_with_crewai(samples):
    """Test enhanced samples with CrewAI crew."""
    
    print(f"\n🤖 Testing Enhanced Samples with CrewAI")
    print("=" * 50)
    
    try:
        from src.mail_agents.crew.email_crew import EmailProcessingCrew
        
        crew = EmailProcessingCrew()
        
        # Test quality samples
        quality_samples = samples['quality'].head(3)  # Test 3 samples
        
        for i, (_, sample) in enumerate(quality_samples.iterrows(), 1):
            print(f"\n🧪 Test {i}: {sample['email_type']} email")
            print(f"📊 Length: {sample['text_length']} chars, Words: {sample['word_count']}")
            print(f"📝 Text preview: {sample['Message'][:100]}...")
            
            try:
                result = crew.classify_email(sample['Message'])
                print(f"✅ Classification completed!")
                print(f"📊 Result preview: {result[:150]}...")
                
            except Exception as e:
                print(f"⚠️  Classification failed: {e}")
        
    except Exception as e:
        print(f"❌ CrewAI testing failed: {e}")


def create_future_enron_integration():
    """Create a placeholder for future Enron dataset integration."""
    
    print(f"\n🔮 Future Enron Integration Preparation")
    print("=" * 50)
    
    # Create a simple script for future use
    enron_script = '''#!/usr/bin/env python3
"""
Future Enron Dataset Integration
Run this when you have more disk space (need ~2GB free)
"""

import kagglehub
import pandas as pd

def download_enron_when_ready():
    """Download Enron dataset when disk space allows."""
    
    print("📥 Downloading Enron Email Dataset...")
    path = kagglehub.dataset_download("wcukierski/enron-email-dataset")
    print(f"✅ Downloaded to: {path}")
    
    # Process and save to data/enron_email_dataset.csv
    # Add to combined_email_dataset.csv for ML training
    
if __name__ == "__main__":
    download_enron_when_ready()
'''
    
    with open("download_enron_future.py", "w") as f:
        f.write(enron_script)
    
    print(f"✅ Created download_enron_future.py for when disk space is available")
    print(f"💡 Run this script after freeing up ~2GB of disk space")


def main():
    """Main function for lightweight dataset management."""
    
    print("🚀 Lightweight Email Dataset Manager")
    print("📊 Optimized for Limited Disk Space")
    print()
    
    # Check disk space
    has_space = check_disk_space()
    
    if not has_space:
        print(f"\n💡 Proceeding with space-efficient approach...")
    
    # Enhance existing spam dataset
    result = enhance_existing_spam_dataset()
    
    if result:
        df, samples = result
        
        # Prepare ML training data
        ml_data = prepare_ml_training_data(df)
        
        # Test with CrewAI
        test_enhanced_samples_with_crewai(samples)
        
        # Prepare for future Enron integration
        create_future_enron_integration()
    
    print("\n" + "=" * 60)
    print("🎉 Lightweight Dataset Enhancement Complete!")
    print()
    print("📂 Enhanced Files Created:")
    print("   ✅ data/enhanced_spam_dataset.csv (enhanced features)")
    print("   ✅ data/balanced_test_samples.csv (10 spam + 10 ham)")
    print("   ✅ data/edge_case_samples.csv (unusual emails)")
    print("   ✅ data/quality_test_samples.csv (high-quality samples)")
    print("   ✅ data/ml_training_data.csv (ML-ready dataset)")
    print("   ⏳ download_enron_future.py (for when space allows)")
    print()
    print("🚀 Next Steps:")
    print("   1. Train ML model with enhanced dataset")
    print("   2. Test CrewAI with quality samples")
    print("   3. Free up disk space for Enron dataset later")
    print("   4. Use enhanced features for better classification")


if __name__ == "__main__":
    main()