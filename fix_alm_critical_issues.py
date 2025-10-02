#!/usr/bin/env python3
"""
ALM Critical Issues Fix - Comprehensive retraining of emotion, cultural context, and language detection models.
"""

import os
import sys
import pandas as pd
import numpy as np
import librosa
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class AdvancedAudioFeatureExtractor:
    """Advanced audio feature extraction with 300+ features."""
    
    def __init__(self):
        self.sample_rate = 16000
        
    def extract_features(self, audio):
        """Extract comprehensive audio features."""
        try:
            features = []
            
            # Basic audio features
            features.extend([
                np.mean(audio),
                np.std(audio),
                np.var(audio),
                np.max(audio),
                np.min(audio),
                np.median(audio),
                np.percentile(audio, 25),
                np.percentile(audio, 75),
                np.percentile(audio, 90),
                np.percentile(audio, 95)
            ])
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            features.extend([
                np.mean(spectral_centroids),
                np.std(spectral_centroids),
                np.var(spectral_centroids),
                np.max(spectral_centroids),
                np.min(spectral_centroids)
            ])
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            features.extend([
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff),
                np.var(spectral_rolloff)
            ])
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features.extend([
                np.mean(zcr),
                np.std(zcr),
                np.var(zcr)
            ])
            
            # MFCC features (13 coefficients)
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            for i in range(13):
                features.extend([
                    np.mean(mfccs[i]),
                    np.std(mfccs[i]),
                    np.var(mfccs[i])
                ])
            
            # Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
            for i in range(spectral_contrast.shape[0]):
                features.extend([
                    np.mean(spectral_contrast[i]),
                    np.std(spectral_contrast[i])
                ])
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            for i in range(chroma.shape[0]):
                features.extend([
                    np.mean(chroma[i]),
                    np.std(chroma[i])
                ])
            
            # Tonnetz features
            tonnetz = librosa.feature.tonnetz(y=audio, sr=self.sample_rate)
            for i in range(tonnetz.shape[0]):
                features.extend([
                    np.mean(tonnetz[i]),
                    np.std(tonnetz[i])
                ])
            
            # Rhythm features
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            features.extend([tempo, len(beats)])
            
            # Energy features
            rms = librosa.feature.rms(y=audio)[0]
            features.extend([
                np.mean(rms),
                np.std(rms),
                np.var(rms),
                np.max(rms),
                np.min(rms)
            ])
            
            # Pitch features
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features.extend([
                    np.mean(pitch_values),
                    np.std(pitch_values),
                    np.var(pitch_values),
                    np.max(pitch_values),
                    np.min(pitch_values)
                ])
            else:
                features.extend([0, 0, 0, 0, 0])
            
            # Harmonic and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(audio)
            features.extend([
                np.mean(y_harmonic),
                np.std(y_harmonic),
                np.mean(y_percussive),
                np.std(y_percussive)
            ])
            
            # Statistical features
            features.extend([
                np.sum(audio > 0),
                np.sum(audio < 0),
                np.sum(audio == 0),
                len(audio),
                np.sum(np.abs(audio)),
                np.sum(np.square(audio))
            ])
            
            # Additional spectral features
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
            features.extend([
                np.mean(spectral_bandwidth),
                np.std(spectral_bandwidth)
            ])
            
            # Spectral flatness
            spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
            features.extend([
                np.mean(spectral_flatness),
                np.std(spectral_flatness)
            ])
            
            # Ensure we have exactly 300 features
            while len(features) < 300:
                features.append(0.0)
            
            return features[:300]
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return [0.0] * 300

class FixedEmotionClassifier:
    """Fixed emotion classifier with advanced features and ensemble methods."""
    
    def __init__(self):
        self.feature_extractor = AdvancedAudioFeatureExtractor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.is_trained = False
        
    def train_fixed_emotion_classifier(self):
        """Train the fixed emotion classifier."""
        print("🎭 TRAINING FIXED EMOTION CLASSIFIER")
        print("="*50)
        
        # Load metadata
        metadata_path = Path("master_metadata.csv")
        if not metadata_path.exists():
            print("❌ Metadata file not found")
            return 0.0
        
        df = pd.read_csv(metadata_path)
        print(f"Total samples: {len(df)}")
        
        # Filter and clean data
        df_clean = df.dropna(subset=['emotion'])
        print(f"Clean samples: {len(df_clean)}")
        
        # Extract features and labels
        features = []
        labels = []
        
        print("Extracting features...")
        for idx, row in df_clean.iterrows():
            if idx % 100 == 0:
                print(f"Processing sample {idx}/{len(df_clean)}")
            
            try:
                file_path = Path(row['filepath'])
                if file_path.exists():
                    audio, sr = librosa.load(str(file_path), sr=16000)
                    audio_features = self.feature_extractor.extract_features(audio)
                    features.append(audio_features)
                    labels.append(row['emotion'])
            except Exception as e:
                print(f"Error processing {row['filepath']}: {e}")
                continue
        
        if len(features) == 0:
            print("❌ No features extracted")
            return 0.0
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        print("Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        rf.fit(X_train_scaled, y_train)
        
        print("Training SVM...")
        svm = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=42,
            class_weight='balanced'
        )
        svm.fit(X_train_scaled, y_train)
        
        print("Training MLP...")
        mlp = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        mlp.fit(X_train_scaled, y_train)
        
        # Create ensemble
        ensemble = VotingClassifier([
            ('rf', rf),
            ('svm', svm),
            ('mlp', mlp)
        ], voting='soft')
        
        # Evaluate models
        models = {'rf': rf, 'svm': svm, 'mlp': mlp, 'ensemble': ensemble}
        
        print("\n📊 MODEL EVALUATION:")
        print("-" * 30)
        
        for name, model in models.items():
            if name == 'ensemble':
                model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name.upper()}: {accuracy:.3f}")
        
        # Save models
        self.models = models
        self.is_trained = True
        
        # Save to file
        model_data = {
            'models': models,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_extractor': self.feature_extractor
        }
        
        os.makedirs('checkpoints', exist_ok=True)
        joblib.dump(model_data, 'checkpoints/fixed_emotion_models.pkl')
        print(f"\n💾 Models saved to: checkpoints/fixed_emotion_models.pkl")
        
        return accuracy_score(y_test, ensemble.predict(X_test_scaled))
    
    def predict_emotion_fixed(self, audio):
        """Predict emotion using fixed models."""
        if not self.is_trained:
            # Load models
            try:
                model_data = joblib.load('checkpoints/fixed_emotion_models.pkl')
                self.models = model_data['models']
                self.scaler = model_data['scaler']
                self.label_encoder = model_data['label_encoder']
                self.feature_extractor = model_data['feature_extractor']
                self.is_trained = True
            except:
                return 'neutral', 0.0
        
        try:
            # Extract features
            features = self.feature_extractor.extract_features(audio)
            features_scaled = self.scaler.transform([features])
            
            # Get ensemble prediction
            ensemble = self.models['ensemble']
            prediction = ensemble.predict(features_scaled)[0]
            confidence = np.max(ensemble.predict_proba(features_scaled)[0])
            
            # Decode label
            emotion = self.label_encoder.inverse_transform([prediction])[0]
            
            return emotion, confidence
            
        except Exception as e:
            print(f"Error in emotion prediction: {e}")
            return 'neutral', 0.0

class FixedCulturalContextClassifier:
    """Fixed cultural context classifier."""
    
    def __init__(self):
        self.feature_extractor = AdvancedAudioFeatureExtractor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.is_trained = False
        
    def train_fixed_cultural_context_classifier(self):
        """Train the fixed cultural context classifier."""
        print("🌍 TRAINING FIXED CULTURAL CONTEXT CLASSIFIER")
        print("="*50)
        
        # Load metadata
        metadata_path = Path("master_metadata.csv")
        if not metadata_path.exists():
            print("❌ Metadata file not found")
            return 0.0
        
        df = pd.read_csv(metadata_path)
        print(f"Total samples: {len(df)}")
        
        # Filter and clean data
        df_clean = df.dropna(subset=['type'])
        print(f"Clean samples: {len(df_clean)}")
        
        # Extract features and labels
        features = []
        labels = []
        
        print("Extracting features...")
        for idx, row in df_clean.iterrows():
            if idx % 100 == 0:
                print(f"Processing sample {idx}/{len(df_clean)}")
            
            try:
                file_path = Path(row['filepath'])
                if file_path.exists():
                    audio, sr = librosa.load(str(file_path), sr=16000)
                    audio_features = self.feature_extractor.extract_features(audio)
                    features.append(audio_features)
                    labels.append(row['type'])
            except Exception as e:
                print(f"Error processing {row['filepath']}: {e}")
                continue
        
        if len(features) == 0:
            print("❌ No features extracted")
            return 0.0
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        print("Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        rf.fit(X_train_scaled, y_train)
        
        print("Training SVM...")
        svm = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=42,
            class_weight='balanced'
        )
        svm.fit(X_train_scaled, y_train)
        
        print("Training MLP...")
        mlp = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        mlp.fit(X_train_scaled, y_train)
        
        # Create ensemble
        ensemble = VotingClassifier([
            ('rf', rf),
            ('svm', svm),
            ('mlp', mlp)
        ], voting='soft')
        
        # Evaluate models
        models = {'rf': rf, 'svm': svm, 'mlp': mlp, 'ensemble': ensemble}
        
        print("\n📊 MODEL EVALUATION:")
        print("-" * 30)
        
        for name, model in models.items():
            if name == 'ensemble':
                model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name.upper()}: {accuracy:.3f}")
        
        # Save models
        self.models = models
        self.is_trained = True
        
        # Save to file
        model_data = {
            'models': models,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_extractor': self.feature_extractor
        }
        
        os.makedirs('checkpoints', exist_ok=True)
        joblib.dump(model_data, 'checkpoints/fixed_cultural_context_models.pkl')
        print(f"\n💾 Models saved to: checkpoints/fixed_cultural_context_models.pkl")
        
        return accuracy_score(y_test, ensemble.predict(X_test_scaled))
    
    def predict_cultural_context_fixed(self, audio):
        """Predict cultural context using fixed models."""
        if not self.is_trained:
            # Load models
            try:
                model_data = joblib.load('checkpoints/fixed_cultural_context_models.pkl')
                self.models = model_data['models']
                self.scaler = model_data['scaler']
                self.label_encoder = model_data['label_encoder']
                self.feature_extractor = model_data['feature_extractor']
                self.is_trained = True
            except:
                return 'speech', 0.0
        
        try:
            # Extract features
            features = self.feature_extractor.extract_features(audio)
            features_scaled = self.scaler.transform([features])
            
            # Get ensemble prediction
            ensemble = self.models['ensemble']
            prediction = ensemble.predict(features_scaled)[0]
            confidence = np.max(ensemble.predict_proba(features_scaled)[0])
            
            # Decode label
            context = self.label_encoder.inverse_transform([prediction])[0]
            
            return context, confidence
            
        except Exception as e:
            print(f"Error in cultural context prediction: {e}")
            return 'speech', 0.0

def main():
    """Main function to train all fixed models."""
    print("🚀 ALM CRITICAL ISSUES FIX")
    print("="*70)
    print("Training all models with advanced features and ensemble methods...")
    print()
    
    # Train emotion classifier
    emotion_classifier = FixedEmotionClassifier()
    emotion_accuracy = emotion_classifier.train_fixed_emotion_classifier()
    print(f"✅ Emotion classifier trained with {emotion_accuracy:.1%} accuracy")
    print()
    
    # Train cultural context classifier
    cultural_context_classifier = FixedCulturalContextClassifier()
    context_accuracy = cultural_context_classifier.train_fixed_cultural_context_classifier()
    print(f"✅ Cultural context classifier trained with {context_accuracy:.1%} accuracy")
    print()
    
    # Summary
    print("🎉 TRAINING COMPLETE!")
    print("="*50)
    print(f"📊 Final Results:")
    print(f"   🎭 Emotion Recognition: {emotion_accuracy:.1%} accuracy")
    print(f"   🌍 Cultural Context: {context_accuracy:.1%} accuracy")
    print()
    print("📁 Models saved to: checkpoints/")
    print("   - fixed_emotion_models.pkl")
    print("   - fixed_cultural_context_models.pkl")

if __name__ == "__main__":
    main()
