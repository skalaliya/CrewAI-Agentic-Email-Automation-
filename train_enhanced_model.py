#!/usr/bin/env python3
"""
Enhanced ML Model Training with Combined Dataset
Trains a hybrid spam classification model using the combined email dataset.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime

# ML libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.mail_agents.config import settings


class EnhancedSpamClassifier:
    """Enhanced spam classifier using combined dataset and ensemble methods."""
    
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.best_model = None
        self.feature_names = None
        self.training_stats = {}
        
    def load_combined_dataset(self):
        """Load and prepare the combined email dataset."""
        
        print("📊 Loading Combined Email Dataset")
        print("=" * 50)
        
        # Load combined dataset
        data_path = Path("data/combined_email_dataset.csv")
        if not data_path.exists():
            raise FileNotFoundError(f"Combined dataset not found: {data_path}")
        
        df = pd.read_csv(data_path)
        print(f"✅ Loaded dataset: {df.shape[0]} emails")
        
        # Data preprocessing
        df = df.dropna(subset=['text', 'label'])
        df['text'] = df['text'].astype(str)
        df['label'] = df['label'].str.lower().str.strip()
        
        # Filter valid labels
        valid_labels = ['spam', 'ham']
        df = df[df['label'].isin(valid_labels)]
        
        print(f"📈 Dataset after cleaning: {df.shape[0]} emails")
        print(f"📊 Label distribution:")
        print(df['label'].value_counts())
        print(f"📂 Source distribution:")
        print(df['source'].value_counts())
        
        return df
    
    def create_features(self, df):
        """Create enhanced features from email text."""
        
        print(f"\n🔧 Creating Enhanced Features")
        print("-" * 30)
        
        # TF-IDF Vectorizer with optimized parameters
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,           # Ignore terms that appear in less than 2 documents
            max_df=0.95,        # Ignore terms that appear in more than 95% of documents
            sublinear_tf=True   # Use sublinear TF scaling
        )
        
        # Fit and transform text data
        X_tfidf = self.vectorizer.fit_transform(df['text'])
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"✅ TF-IDF features created: {X_tfidf.shape[1]} features")
        
        # Additional manual features
        additional_features = []
        
        # Email length features
        df['email_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        additional_features.extend(['email_length', 'word_count'])
        
        # Spam indicator features
        df['has_urls'] = df['text'].str.contains(r'http[s]?://', case=False, na=False).astype(int)
        df['has_money'] = df['text'].str.contains(r'\$\d+|\d+\s*dollar|money|cash|prize', case=False, na=False).astype(int)
        df['urgency_words'] = df['text'].str.contains(r'urgent|immediately|act now|limited time|hurry', case=False, na=False).astype(int)
        df['caps_ratio'] = df['text'].apply(lambda x: sum(c.isupper() for c in x) / len(x) if len(x) > 0 else 0)
        df['exclamation_count'] = df['text'].str.count('!')
        additional_features.extend(['has_urls', 'has_money', 'urgency_words', 'caps_ratio', 'exclamation_count'])
        
        # Source features (one-hot encoding)
        source_dummies = pd.get_dummies(df['source'], prefix='source')
        additional_features.extend(source_dummies.columns.tolist())
        
        # Combine TF-IDF with additional features
        from scipy.sparse import hstack
        additional_matrix = np.column_stack([df[feat].values for feat in additional_features if feat in df.columns] + 
                                          [source_dummies.values])
        
        X_combined = hstack([X_tfidf, additional_matrix])
        
        print(f"✅ Combined features: {X_combined.shape[1]} total features")
        print(f"   - TF-IDF features: {X_tfidf.shape[1]}")
        print(f"   - Additional features: {additional_matrix.shape[1]}")
        
        return X_combined, df['label']
    
    def train_models(self, X, y):
        """Train multiple models and create ensemble."""
        
        print(f"\n🤖 Training Enhanced Models")
        print("=" * 40)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training set: {X_train.shape[0]} emails")
        print(f"📊 Test set: {X_test.shape[0]} emails")
        
        # Define models
        models_config = {
            'logistic_regression': LogisticRegression(
                random_state=42, max_iter=1000, class_weight='balanced'
            ),
            'svm': LinearSVC(
                random_state=42, class_weight='balanced', max_iter=2000
            ),
            'naive_bayes': MultinomialNB(alpha=0.1),
            'random_forest': RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced'
            )
        }
        
        # Train individual models
        results = {}
        
        for name, model in models_config.items():
            print(f"\n🔄 Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'predictions': y_pred
                }
                
                if y_proba is not None:
                    auc = roc_auc_score(y_test == 'spam', y_proba)
                    results[name]['auc'] = auc
                
                print(f"   ✅ Accuracy: {accuracy:.3f}")
                print(f"   ✅ F1-Score: {f1:.3f}")
                
                # Store model
                self.models[name] = model
                
            except Exception as e:
                print(f"   ❌ Failed to train {name}: {e}")
        
        # Create ensemble model
        print(f"\n🔗 Creating Ensemble Model...")
        
        # Select best performing models for ensemble
        ensemble_models = []
        for name, result in results.items():
            if result['accuracy'] > 0.85:  # Only include good models
                ensemble_models.append((name, result['model']))
        
        if len(ensemble_models) >= 2:
            ensemble = VotingClassifier(
                estimators=ensemble_models,
                voting='soft' if all(hasattr(model[1], 'predict_proba') for model in ensemble_models) else 'hard'
            )
            
            ensemble.fit(X_train, y_train)
            y_pred_ensemble = ensemble.predict(X_test)
            
            ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
            ensemble_precision, ensemble_recall, ensemble_f1, _ = precision_recall_fscore_support(y_test, y_pred_ensemble, average='weighted')
            
            results['ensemble'] = {
                'model': ensemble,
                'accuracy': ensemble_accuracy,
                'precision': ensemble_precision,
                'recall': ensemble_recall,
                'f1': ensemble_f1,
                'predictions': y_pred_ensemble
            }
            
            print(f"   ✅ Ensemble Accuracy: {ensemble_accuracy:.3f}")
            print(f"   ✅ Ensemble F1-Score: {ensemble_f1:.3f}")
            
            self.models['ensemble'] = ensemble
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['f1'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   📊 Accuracy: {results[best_model_name]['accuracy']:.3f}")
        print(f"   📊 F1-Score: {results[best_model_name]['f1']:.3f}")
        
        # Store training statistics
        self.training_stats = {
            'best_model': best_model_name,
            'results': {k: {key: val for key, val in v.items() if key != 'model' and key != 'predictions'} 
                       for k, v in results.items()},
            'training_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'features': X.shape[1],
            'trained_at': datetime.now().isoformat()
        }
        
        # Detailed evaluation
        self._evaluate_model(X_test, y_test, results)
        
        return results
    
    def _evaluate_model(self, X_test, y_test, results):
        """Detailed model evaluation."""
        
        print(f"\n📊 Detailed Model Evaluation")
        print("=" * 40)
        
        best_model_name = max(results.keys(), key=lambda k: results[k]['f1'])
        best_model = results[best_model_name]['model']
        best_predictions = results[best_model_name]['predictions']
        
        # Classification report
        print(f"\n📋 Classification Report ({best_model_name}):")
        print(classification_report(y_test, best_predictions))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, best_predictions)
        print(f"\n📊 Confusion Matrix:")
        print(f"          Predicted")
        print(f"Actual    Ham  Spam")
        print(f"Ham      {cm[0][0]:4d} {cm[0][1]:4d}")
        print(f"Spam     {cm[1][0]:4d} {cm[1][1]:4d}")
        
        # Feature importance (if available)
        if hasattr(best_model, 'feature_importances_'):
            print(f"\n🔍 Top 10 Important Features:")
            feature_importance = best_model.feature_importances_
            top_features = np.argsort(feature_importance)[-10:][::-1]
            
            for i, feat_idx in enumerate(top_features):
                if feat_idx < len(self.feature_names):
                    feat_name = self.feature_names[feat_idx]
                else:
                    feat_name = f"additional_feature_{feat_idx - len(self.feature_names)}"
                print(f"   {i+1:2d}. {feat_name}: {feature_importance[feat_idx]:.4f}")
        
        elif hasattr(best_model, 'coef_'):
            print(f"\n🔍 Top 10 Spam Indicators (highest coefficients):")
            coef = best_model.coef_[0]
            top_spam_indices = np.argsort(coef)[-10:][::-1]
            
            for i, feat_idx in enumerate(top_spam_indices):
                if feat_idx < len(self.feature_names):
                    feat_name = self.feature_names[feat_idx]
                    print(f"   {i+1:2d}. {feat_name}: {coef[feat_idx]:.4f}")
    
    def save_models(self):
        """Save trained models and vectorizer."""
        
        print(f"\n💾 Saving Enhanced Models")
        print("-" * 30)
        
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save vectorizer
        vectorizer_path = models_dir / "enhanced_vectorizer.pkl"
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"✅ Vectorizer saved: {vectorizer_path}")
        
        # Save best model
        model_path = models_dir / "enhanced_spam_classifier.pkl"
        joblib.dump(self.best_model, model_path)
        print(f"✅ Best model saved: {model_path}")
        
        # Save all models
        all_models_path = models_dir / "all_enhanced_models.pkl"
        joblib.dump(self.models, all_models_path)
        print(f"✅ All models saved: {all_models_path}")
        
        # Save training statistics
        stats_path = models_dir / "training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        print(f"✅ Training stats saved: {stats_path}")
        
        return {
            'vectorizer_path': str(vectorizer_path),
            'model_path': str(model_path),
            'all_models_path': str(all_models_path),
            'stats_path': str(stats_path)
        }
    
    def test_with_samples(self):
        """Test the trained model with sample emails."""
        
        print(f"\n🧪 Testing Enhanced Model")
        print("=" * 30)
        
        test_emails = [
            ("FREE MONEY! Click now to claim your $1000000 prize! No questions asked!", "spam"),
            ("Hi John, can we schedule our quarterly review meeting for next Tuesday at 3pm?", "ham"),
            ("URGENT! Your account will be suspended unless you verify immediately!", "spam"),
            ("Please find attached the Q4 financial report for review.", "ham"),
            ("Congratulations! You've won a lottery you never entered! Send bank details now!", "spam")
        ]
        
        correct_predictions = 0
        
        for email_text, expected_label in test_emails:
            # Transform text
            email_features = self.vectorizer.transform([email_text])
            
            # Add additional features (simplified for testing)
            additional_feats = np.array([[
                len(email_text),  # email_length
                len(email_text.split()),  # word_count
                1 if 'http' in email_text.lower() else 0,  # has_urls
                1 if any(word in email_text.lower() for word in ['$', 'money', 'cash', 'prize']) else 0,  # has_money
                1 if any(word in email_text.lower() for word in ['urgent', 'immediately', 'act now']) else 0,  # urgency_words
                sum(c.isupper() for c in email_text) / len(email_text) if len(email_text) > 0 else 0,  # caps_ratio
                email_text.count('!'),  # exclamation_count
                0, 1  # source features (assuming spam_classification source)
            ]])
            
            # Combine features
            from scipy.sparse import hstack
            combined_features = hstack([email_features, additional_feats])
            
            # Predict
            prediction = self.best_model.predict(combined_features)[0]
            
            # Get confidence if available
            confidence = ""
            if hasattr(self.best_model, 'predict_proba'):
                proba = self.best_model.predict_proba(combined_features)[0]
                max_proba = max(proba)
                confidence = f" ({max_proba:.3f} confidence)"
            
            print(f"📧 Email: {email_text[:50]}...")
            print(f"   Expected: {expected_label}")
            print(f"   Predicted: {prediction}{confidence}")
            print(f"   {'✅ Correct' if prediction == expected_label else '❌ Wrong'}")
            print()
            
            if prediction == expected_label:
                correct_predictions += 1
        
        accuracy = correct_predictions / len(test_emails)
        print(f"🎯 Test Accuracy: {accuracy:.1%} ({correct_predictions}/{len(test_emails)})")


def main():
    """Main training function."""
    
    print("🚀 Enhanced ML Model Training")
    print("📊 Using Combined Email Dataset for Superior Performance")
    print("=" * 60)
    
    # Initialize classifier
    classifier = EnhancedSpamClassifier()
    
    try:
        # Load dataset
        df = classifier.load_combined_dataset()
        
        # Create features
        X, y = classifier.create_features(df)
        
        # Train models
        results = classifier.train_models(X, y)
        
        # Save models
        saved_paths = classifier.save_models()
        
        # Test with samples
        classifier.test_with_samples()
        
        print(f"\n🎉 Enhanced ML Training Complete!")
        print("=" * 40)
        print(f"✅ Best Model: {classifier.training_stats['best_model']}")
        print(f"✅ Training Samples: {classifier.training_stats['training_samples']:,}")
        print(f"✅ Features: {classifier.training_stats['features']:,}")
        print(f"✅ Models saved to: models/")
        print()
        print("🔗 Integration with CrewAI:")
        print("   Your agents can now use these ML predictions")
        print("   for faster and more accurate email classification!")
        print()
        print("🚀 Next Steps:")
        print("   1. Test the enhanced system: uv run python test_enhanced_model.py")
        print("   2. Update CrewAI agents to use ML predictions")
        print("   3. Deploy enhanced system to production")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()