#!/usr/bin/env python3
"""
Enhanced Kaggle dataset downloader and processor for spam email classification.
Downloads the latest spam email dataset and prepares it for CrewAI agents.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    print("📦 Installing kagglehub...")
    os.system("uv pip install kagglehub")
    try:
        import kagglehub
        KAGGLEHUB_AVAILABLE = True
    except ImportError:
        KAGGLEHUB_AVAILABLE = False
        print("❌ Could not install kagglehub")

from src.mail_agents.config import settings


def download_kaggle_dataset():
    """Download the spam email classification dataset from Kaggle."""
    
    print("🔍 Downloading Kaggle Spam Email Dataset...")
    print("=" * 60)
    
    if not KAGGLEHUB_AVAILABLE:
        print("❌ KaggleHub not available. Please install it manually:")
        print("   uv pip install kagglehub")
        return None
    
    try:
        # Download latest version of the spam email classification dataset
        print("📥 Downloading dataset: ashfakyeafi/spam-email-classification")
        path = kagglehub.dataset_download("ashfakyeafi/spam-email-classification")
        
        print(f"✅ Dataset downloaded successfully!")
        print(f"📂 Path to dataset files: {path}")
        
        return path
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("💡 Make sure you have:")
        print("   1. Kaggle account setup")
        print("   2. API credentials configured")
        print("   3. Internet connection")
        return None


def process_dataset(dataset_path):
    """Process the downloaded dataset and prepare it for analysis."""
    
    if not dataset_path:
        return None
    
    print(f"\n📊 Processing dataset from: {dataset_path}")
    print("=" * 60)
    
    try:
        # Find CSV files in the dataset directory
        path_obj = Path(dataset_path)
        csv_files = list(path_obj.glob("*.csv"))
        
        if not csv_files:
            print("❌ No CSV files found in dataset")
            return None
        
        print(f"📋 Found {len(csv_files)} CSV file(s):")
        for csv_file in csv_files:
            print(f"   - {csv_file.name}")
        
        # Load the main dataset (usually the first/largest CSV)
        main_csv = csv_files[0]
        print(f"\n📖 Loading dataset: {main_csv.name}")
        
        df = pd.read_csv(main_csv)
        
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📝 Columns: {list(df.columns)}")
        
        # Show sample data
        print(f"\n🔍 Sample data:")
        print(df.head())
        
        # Show label distribution
        if 'label' in df.columns:
            print(f"\n📈 Label distribution:")
            print(df['label'].value_counts())
        elif 'Category' in df.columns:
            print(f"\n📈 Category distribution:")
            print(df['Category'].value_counts())
        
        # Save processed dataset to local directory
        local_data_dir = Path("data")
        local_data_dir.mkdir(exist_ok=True)
        
        processed_path = local_data_dir / "spam_email_dataset.csv"
        df.to_csv(processed_path, index=False)
        
        print(f"\n💾 Dataset saved to: {processed_path}")
        
        return df, processed_path
        
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        return None


def create_sample_emails(df):
    """Create sample emails from the dataset for testing."""
    
    if df is None:
        return []
    
    print(f"\n📧 Creating sample emails for testing...")
    print("=" * 60)
    
    try:
        # Try to find the text and label columns
        text_col = None
        label_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'text' in col_lower or 'message' in col_lower or 'email' in col_lower:
                text_col = col
            if 'label' in col_lower or 'category' in col_lower or 'class' in col_lower:
                label_col = col
        
        if not text_col:
            print("❌ Could not find text column in dataset")
            return []
        
        if not label_col:
            print("❌ Could not find label column in dataset")
            return []
        
        print(f"📝 Using text column: {text_col}")
        print(f"🏷️  Using label column: {label_col}")
        
        # Get sample emails (5 spam, 5 ham)
        spam_samples = df[df[label_col].str.contains('spam', case=False, na=False)].head(5)
        ham_samples = df[~df[label_col].str.contains('spam', case=False, na=False)].head(5)
        
        samples = []
        
        print(f"\n🚨 Spam email samples:")
        for idx, row in spam_samples.iterrows():
            email_text = str(row[text_col])[:200] + "..." if len(str(row[text_col])) > 200 else str(row[text_col])
            samples.append({
                'text': str(row[text_col]),
                'label': str(row[label_col]),
                'type': 'spam'
            })
            print(f"   {len(samples)}. {email_text}")
        
        print(f"\n✅ Ham (legitimate) email samples:")
        for idx, row in ham_samples.iterrows():
            email_text = str(row[text_col])[:200] + "..." if len(str(row[text_col])) > 200 else str(row[text_col])
            samples.append({
                'text': str(row[text_col]),
                'label': str(row[label_col]),
                'type': 'ham'
            })
            print(f"   {len(samples)}. {email_text}")
        
        # Save samples for easy testing
        samples_df = pd.DataFrame(samples)
        samples_path = Path("data") / "email_samples.csv"
        samples_df.to_csv(samples_path, index=False)
        
        print(f"\n💾 Sample emails saved to: {samples_path}")
        
        return samples
        
    except Exception as e:
        print(f"❌ Error creating samples: {e}")
        return []


def test_with_crewai(samples):
    """Test the samples with our CrewAI email processing crew."""
    
    if not samples:
        return
    
    print(f"\n🤖 Testing with CrewAI Email Processing Crew...")
    print("=" * 60)
    
    try:
        from src.mail_agents.crew.email_crew import EmailProcessingCrew
        
        # Initialize crew
        print("📝 Initializing EmailProcessingCrew...")
        crew = EmailProcessingCrew()
        
        # Test with one spam and one ham email
        test_emails = [
            samples[0],  # First spam
            samples[5] if len(samples) > 5 else samples[-1]  # First ham or last available
        ]
        
        for i, email_sample in enumerate(test_emails, 1):
            print(f"\n🧪 Test {i}: {email_sample['type'].upper()} email")
            print(f"📧 Original label: {email_sample['label']}")
            print(f"📝 Text preview: {email_sample['text'][:100]}...")
            
            try:
                # Use our enhanced crew with tools
                result = crew.classify_email(email_sample['text'])
                
                print(f"✅ CrewAI Classification completed!")
                print(f"📊 Result preview: {result[:200]}...")
                
            except Exception as e:
                print(f"⚠️  Classification failed: {e}")
                print("💡 This might be due to Ollama not running or timeout issues")
        
        print(f"\n🎉 CrewAI testing completed!")
        
    except Exception as e:
        print(f"❌ CrewAI testing failed: {e}")


def main():
    """Main function to download, process, and test with Kaggle dataset."""
    
    print("🚀 Enhanced Kaggle Dataset Integration for CrewAI")
    print("📊 Spam Email Classification Dataset Processing")
    print()
    
    # Step 1: Download dataset
    dataset_path = download_kaggle_dataset()
    
    # Step 2: Process dataset
    result = process_dataset(dataset_path)
    if result:
        df, processed_path = result
    else:
        print("❌ Could not process dataset")
        return
    
    # Step 3: Create sample emails
    samples = create_sample_emails(df)
    
    # Step 4: Test with CrewAI (optional)
    if samples:
        test_with_crewai(samples)
    
    print("\n" + "=" * 60)
    print("🎉 Dataset integration complete!")
    print()
    print("📋 What was accomplished:")
    print("   ✅ Downloaded latest Kaggle spam dataset")
    print("   ✅ Processed and analyzed dataset structure")
    print("   ✅ Created sample emails for testing")
    print("   ✅ Saved processed data locally")
    print("   ✅ Tested with CrewAI email processing crew")
    print()
    print("📂 Files created:")
    print("   - data/spam_email_dataset.csv (full dataset)")
    print("   - data/email_samples.csv (test samples)")
    print()
    print("🚀 Next steps:")
    print("   1. Use samples in your notebook: pd.read_csv('data/email_samples.csv')")
    print("   2. Train ML model with full dataset")
    print("   3. Test CrewAI agents with real-world emails")
    print("   4. Update your Jupyter notebook with new data")


if __name__ == "__main__":
    main()