#!/usr/bin/env python3
"""
Enhanced dataset manager for CrewAI Email Automation.
Downloads and processes multiple email datasets including:
1. Spam Email Classification Dataset (ashfakyeafi/spam-email-classification)
2. Enron Email Dataset (wcukierski/enron-email-dataset)
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import email
import email.utils
from email.message import EmailMessage
import re
from datetime import datetime

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


class EmailDatasetManager:
    """Comprehensive email dataset manager for multiple sources."""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            'spam_classification': {
                'kaggle_id': 'ashfakyeafi/spam-email-classification',
                'description': 'Spam vs Ham email classification dataset',
                'local_name': 'spam_email_dataset.csv',
                'processor': self._process_spam_dataset
            },
            'enron_emails': {
                'kaggle_id': 'wcukierski/enron-email-dataset',
                'description': 'Enron corporate email communications',
                'local_name': 'enron_email_dataset.csv',
                'processor': self._process_enron_dataset
            }
        }
    
    def download_all_datasets(self):
        """Download and process all configured datasets."""
        
        print("📊 Enhanced Email Dataset Manager")
        print("=" * 60)
        print("🎯 Downloading multiple email datasets for comprehensive ML training:")
        print("   1. Spam Classification Dataset (5,573 emails)")
        print("   2. Enron Corporate Dataset (500K+ emails)")
        print()
        
        results = {}
        
        for dataset_key, config in self.datasets.items():
            print(f"\n📥 Processing {config['description']}...")
            print("-" * 50)
            
            try:
                # Download dataset
                path = self._download_dataset(config['kaggle_id'])
                if not path:
                    print(f"❌ Failed to download {dataset_key}")
                    results[dataset_key] = None
                    continue
                
                # Process dataset
                processed_data = config['processor'](path)
                if processed_data is None:
                    print(f"❌ Failed to process {dataset_key}")
                    results[dataset_key] = None
                    continue
                
                # Save processed data
                output_path = self.data_dir / config['local_name']
                processed_data.to_csv(output_path, index=False)
                
                print(f"✅ {config['description']} processed successfully!")
                print(f"📊 Shape: {processed_data.shape}")
                print(f"💾 Saved to: {output_path}")
                
                results[dataset_key] = {
                    'data': processed_data,
                    'path': output_path,
                    'shape': processed_data.shape
                }
                
            except Exception as e:
                print(f"❌ Error processing {dataset_key}: {e}")
                results[dataset_key] = None
        
        return results
    
    def _download_dataset(self, kaggle_id):
        """Download a dataset from Kaggle."""
        
        if not KAGGLEHUB_AVAILABLE:
            print("❌ KaggleHub not available")
            return None
        
        try:
            print(f"📥 Downloading {kaggle_id}...")
            path = kagglehub.dataset_download(kaggle_id)
            print(f"✅ Downloaded to: {path}")
            return path
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return None
    
    def _process_spam_dataset(self, dataset_path):
        """Process the spam classification dataset."""
        
        try:
            path_obj = Path(dataset_path)
            csv_files = list(path_obj.glob("*.csv"))
            
            if not csv_files:
                print("❌ No CSV files found in spam dataset")
                return None
            
            # Load the main CSV
            main_csv = csv_files[0]
            print(f"📖 Loading: {main_csv.name}")
            
            df = pd.read_csv(main_csv)
            
            # Standardize columns
            if 'Category' in df.columns and 'Message' in df.columns:
                df = df.rename(columns={'Category': 'label', 'Message': 'text'})
            
            # Add metadata
            df['source'] = 'spam_classification'
            df['dataset_type'] = 'labeled'
            df['processed_at'] = datetime.now().isoformat()
            
            # Clean labels
            df['label'] = df['label'].str.lower().str.strip()
            
            print(f"📊 Processed spam dataset: {df.shape}")
            print(f"📈 Label distribution:")
            print(df['label'].value_counts())
            
            return df
            
        except Exception as e:
            print(f"❌ Error processing spam dataset: {e}")
            return None
    
    def _process_enron_dataset(self, dataset_path):
        """Process the Enron email dataset."""
        
        try:
            path_obj = Path(dataset_path)
            
            # Look for emails.csv or similar files
            csv_files = list(path_obj.glob("*.csv"))
            
            if csv_files:
                print(f"📖 Found CSV files: {[f.name for f in csv_files]}")
                
                # Try to load the main emails file
                main_csv = None
                for csv_file in csv_files:
                    if 'email' in csv_file.name.lower():
                        main_csv = csv_file
                        break
                
                if not main_csv:
                    main_csv = csv_files[0]  # Use first CSV if no obvious match
                
                print(f"📖 Loading: {main_csv.name}")
                df = pd.read_csv(main_csv)
                
            else:
                # Look for raw email files
                print("📧 Looking for raw email files...")
                email_files = []
                
                # Search for .txt or email files
                for pattern in ["*.txt", "*.eml", "**/*.txt", "**/*.eml"]:
                    email_files.extend(list(path_obj.glob(pattern)))
                
                if not email_files:
                    print("❌ No email files found in Enron dataset")
                    return None
                
                print(f"📧 Found {len(email_files)} email files")
                
                # Process raw email files (sample first 1000 for efficiency)
                sample_size = min(1000, len(email_files))
                print(f"📊 Processing sample of {sample_size} emails...")
                
                processed_emails = []
                
                for i, email_file in enumerate(email_files[:sample_size]):
                    if i % 100 == 0:
                        print(f"   Processing: {i}/{sample_size}")
                    
                    try:
                        with open(email_file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Parse email
                        parsed_email = self._parse_raw_email(content)
                        if parsed_email:
                            parsed_email['source_file'] = str(email_file)
                            processed_emails.append(parsed_email)
                    
                    except Exception as e:
                        continue  # Skip problematic files
                
                if not processed_emails:
                    print("❌ No emails could be processed")
                    return None
                
                df = pd.DataFrame(processed_emails)
            
            # Standardize Enron dataset columns
            if 'message' in df.columns:
                df = df.rename(columns={'message': 'text'})
            elif 'content' in df.columns:
                df = df.rename(columns={'content': 'text'})
            elif 'body' in df.columns:
                df = df.rename(columns={'body': 'text'})
            
            # Add metadata
            df['source'] = 'enron_dataset'
            df['dataset_type'] = 'corporate'
            df['label'] = 'ham'  # Enron emails are legitimate corporate communications
            df['processed_at'] = datetime.now().isoformat()
            
            # Clean and filter
            df = df.dropna(subset=['text'])
            df = df[df['text'].str.len() > 50]  # Filter very short emails
            
            # Limit size for processing efficiency
            if len(df) > 5000:
                df = df.sample(n=5000, random_state=42)
                print(f"📊 Sampled 5000 emails for processing efficiency")
            
            print(f"📊 Processed Enron dataset: {df.shape}")
            print(f"📈 Sample subjects: {df['subject'].head().tolist() if 'subject' in df.columns else 'N/A'}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error processing Enron dataset: {e}")
            return None
    
    def _parse_raw_email(self, content):
        """Parse raw email content."""
        
        try:
            # Try to parse as email message
            msg = email.message_from_string(content)
            
            result = {
                'subject': msg.get('Subject', ''),
                'from': msg.get('From', ''),
                'to': msg.get('To', ''),
                'date': msg.get('Date', ''),
                'text': ''
            }
            
            # Extract body
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        result['text'] = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        break
            else:
                result['text'] = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            
            return result if result['text'] else None
            
        except:
            # Fallback: simple text parsing
            lines = content.split('\n')
            
            result = {
                'subject': '',
                'from': '',
                'to': '',
                'date': '',
                'text': content
            }
            
            # Extract headers if present
            for line in lines[:20]:  # Check first 20 lines for headers
                if line.startswith('Subject:'):
                    result['subject'] = line.replace('Subject:', '').strip()
                elif line.startswith('From:'):
                    result['from'] = line.replace('From:', '').strip()
                elif line.startswith('To:'):
                    result['to'] = line.replace('To:', '').strip()
                elif line.startswith('Date:'):
                    result['date'] = line.replace('Date:', '').strip()
            
            return result if len(result['text']) > 50 else None
    
    def create_combined_dataset(self, results):
        """Combine multiple datasets into a unified training set."""
        
        print("\n🔗 Creating Combined Training Dataset")
        print("=" * 50)
        
        combined_data = []
        
        for dataset_key, result in results.items():
            if result and result['data'] is not None:
                data = result['data']
                print(f"✅ Adding {dataset_key}: {data.shape[0]} samples")
                combined_data.append(data)
            else:
                print(f"⚠️  Skipping {dataset_key}: No data available")
        
        if not combined_data:
            print("❌ No datasets available for combination")
            return None
        
        # Combine datasets
        combined_df = pd.concat(combined_data, ignore_index=True, sort=False)
        
        # Standardize columns
        required_columns = ['text', 'label', 'source']
        for col in required_columns:
            if col not in combined_df.columns:
                combined_df[col] = 'unknown'
        
        # Clean and prepare
        combined_df = combined_df.dropna(subset=['text'])
        combined_df = combined_df[combined_df['text'].str.len() > 20]
        
        # Add training metadata
        combined_df['combined_at'] = datetime.now().isoformat()
        combined_df['dataset_id'] = range(len(combined_df))
        
        # Save combined dataset
        combined_path = self.data_dir / 'combined_email_dataset.csv'
        combined_df.to_csv(combined_path, index=False)
        
        print(f"\n📊 Combined Dataset Summary:")
        print(f"   Total samples: {len(combined_df)}")
        print(f"   Sources: {combined_df['source'].value_counts().to_dict()}")
        print(f"   Labels: {combined_df['label'].value_counts().to_dict()}")
        print(f"   💾 Saved to: {combined_path}")
        
        return combined_df, combined_path
    
    def create_enhanced_samples(self, combined_df):
        """Create enhanced sample sets from combined data."""
        
        print(f"\n📋 Creating Enhanced Sample Sets")
        print("=" * 40)
        
        # Create balanced sample set
        samples_per_category = 10
        
        sample_data = []
        
        # Get spam samples
        spam_emails = combined_df[combined_df['label'] == 'spam']
        if len(spam_emails) > 0:
            spam_sample = spam_emails.sample(n=min(samples_per_category, len(spam_emails)), random_state=42)
            sample_data.append(spam_sample)
        
        # Get ham samples from different sources
        ham_emails = combined_df[combined_df['label'] == 'ham']
        if len(ham_emails) > 0:
            # Try to get diverse samples from different sources
            sources = ham_emails['source'].unique()
            samples_per_source = samples_per_category // len(sources) if len(sources) > 0 else samples_per_category
            
            for source in sources:
                source_emails = ham_emails[ham_emails['source'] == source]
                if len(source_emails) > 0:
                    source_sample = source_emails.sample(n=min(samples_per_source, len(source_emails)), random_state=42)
                    sample_data.append(source_sample)
        
        if sample_data:
            enhanced_samples = pd.concat(sample_data, ignore_index=True)
            
            # Save enhanced samples
            samples_path = self.data_dir / 'enhanced_email_samples.csv'
            enhanced_samples.to_csv(samples_path, index=False)
            
            print(f"✅ Enhanced samples created: {len(enhanced_samples)} emails")
            print(f"📊 Sample distribution:")
            print(f"   By label: {enhanced_samples['label'].value_counts().to_dict()}")
            print(f"   By source: {enhanced_samples['source'].value_counts().to_dict()}")
            print(f"💾 Saved to: {samples_path}")
            
            return enhanced_samples
        
        return None


def main():
    """Main function to download and process all email datasets."""
    
    print("🚀 Enhanced Email Dataset Integration")
    print("📊 Multi-Source Dataset Processing for Advanced ML Training")
    print()
    
    # Initialize dataset manager
    manager = EmailDatasetManager()
    
    # Download and process all datasets
    results = manager.download_all_datasets()
    
    # Create combined training dataset
    combined_result = manager.create_combined_dataset(results)
    
    if combined_result:
        combined_df, combined_path = combined_result
        
        # Create enhanced samples
        enhanced_samples = manager.create_enhanced_samples(combined_df)
        
        # Test with CrewAI if available
        if enhanced_samples is not None:
            test_with_crewai(enhanced_samples)
    
    print("\n" + "=" * 70)
    print("🎉 Enhanced Dataset Integration Complete!")
    print()
    print("📂 Files Created:")
    print("   - data/spam_email_dataset.csv (spam classification data)")
    print("   - data/enron_email_dataset.csv (corporate email data)")
    print("   - data/combined_email_dataset.csv (unified training set)")
    print("   - data/enhanced_email_samples.csv (balanced test samples)")
    print()
    print("🚀 Next Steps:")
    print("   1. Train ML model with combined dataset")
    print("   2. Test CrewAI agents with diverse email types")
    print("   3. Analyze corporate vs spam email patterns")
    print("   4. Enhance agent training with real corporate communications")


def test_with_crewai(samples):
    """Test enhanced samples with CrewAI crew."""
    
    print(f"\n🤖 Testing Enhanced Samples with CrewAI")
    print("=" * 50)
    
    try:
        from src.mail_agents.crew.email_crew import EmailProcessingCrew
        
        crew = EmailProcessingCrew()
        
        # Test one sample from each source
        test_samples = []
        for source in samples['source'].unique():
            source_sample = samples[samples['source'] == source].iloc[0]
            test_samples.append(source_sample)
        
        for i, sample in enumerate(test_samples, 1):
            print(f"\n🧪 Test {i}: {sample['source']} - {sample['label']}")
            print(f"📧 Subject: {sample.get('subject', 'N/A')}")
            print(f"📝 Text preview: {sample['text'][:100]}...")
            
            try:
                result = crew.classify_email(sample['text'])
                print(f"✅ Classification completed!")
                print(f"📊 Result preview: {result[:150]}...")
                
            except Exception as e:
                print(f"⚠️  Classification failed: {e}")
        
    except Exception as e:
        print(f"❌ CrewAI testing failed: {e}")


if __name__ == "__main__":
    main()