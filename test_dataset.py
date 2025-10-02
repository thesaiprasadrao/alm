#!/usr/bin/env python3
"""
Test ALM on dataset samples for validation.
"""

import sys
from pathlib import Path
import pandas as pd
import torch
import librosa
import numpy as np
import json
from datetime import datetime
from collections import defaultdict

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from fix_alm_critical_issues import (
    FixedEmotionClassifier,
    FixedCulturalContextClassifier,
    AdvancedAudioFeatureExtractor
)

class DatasetTester:
    """Test ALM on dataset samples."""
    
    def __init__(self):
        self.emotion_classifier = FixedEmotionClassifier()
        self.cultural_context_classifier = FixedCulturalContextClassifier()
        self.feature_extractor = AdvancedAudioFeatureExtractor()
        self.results = []
        self.performance_metrics = defaultdict(list)
        
    def select_test_samples(self, num_samples=10):
        """Select diverse test samples from dataset."""
        print("🎯 SELECTING DATASET TEST SAMPLES")
        print("="*50)
        
        # Load metadata
        metadata_path = Path("master_metadata.csv")
        if not metadata_path.exists():
            print("❌ Metadata file not found")
            return []
        
        df = pd.read_csv(metadata_path)
        print(f"Total samples in dataset: {len(df)}")
        
        # Filter out samples with missing data
        df_clean = df.dropna(subset=['emotion', 'type'])
        print(f"Clean samples: {len(df_clean)}")
        
        # Select diverse samples
        selected_samples = []
        
        # 1. Select by emotion (2 samples per emotion)
        emotions = df_clean['emotion'].unique()
        for emotion in emotions:
            emotion_samples = df_clean[df_clean['emotion'] == emotion]
            if len(emotion_samples) > 0:
                sample = emotion_samples.sample(n=min(2, len(emotion_samples)), random_state=42)
                selected_samples.extend(sample.to_dict('records'))
        
        # 2. Select by type (speech vs non-speech)
        types = df_clean['type'].unique()
        for type_val in types:
            type_samples = df_clean[df_clean['type'] == type_val]
            if len(type_samples) > 0:
                sample = type_samples.sample(n=min(2, len(type_samples)), random_state=42)
                selected_samples.extend(sample.to_dict('records'))
        
        # Remove duplicates
        unique_samples = []
        seen_paths = set()
        for sample in selected_samples:
            if sample['filepath'] not in seen_paths:
                unique_samples.append(sample)
                seen_paths.add(sample['filepath'])
        
        # Limit to requested number
        final_samples = unique_samples[:num_samples]
        
        print(f"Selected {len(final_samples)} diverse samples:")
        for i, sample in enumerate(final_samples):
            print(f"  {i+1}. {sample['filepath']} - {sample['emotion']} - {sample['type']}")
        
        return final_samples
    
    def test_audio_sample(self, sample):
        """Test a single audio sample."""
        print(f"\n🎵 TESTING: {sample['filepath']}")
        print("-" * 50)
        
        # Check if file exists
        file_path = Path(sample['filepath'])
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return None
        
        try:
            # Load audio
            audio, sr = librosa.load(str(file_path), sr=16000)
            
            # Test transcription
            print("🎤 Processing transcription...")
            try:
                from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
                processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
                model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
                model.eval()
                
                inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
                
                with torch.no_grad():
                    logits = model(**inputs).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = processor.decode(predicted_ids[0])
                
                print(f"✅ Transcription: '{transcription.strip()}'")
                
            except Exception as e:
                print(f"❌ Transcription Error: {e}")
                transcription = "Transcription failed"
            
            # Test emotion recognition
            print("😊 Processing emotion recognition...")
            try:
                emotion, emotion_conf = self.emotion_classifier.predict_emotion_fixed(audio)
                print(f"✅ Emotion: {emotion} (confidence: {emotion_conf:.3f})")
            except Exception as e:
                print(f"❌ Emotion Error: {e}")
                emotion = "neutral"
                emotion_conf = 0.0
            
            # Test cultural context
            print("🌍 Processing cultural context...")
            try:
                context, context_conf = self.cultural_context_classifier.predict_cultural_context_fixed(audio)
                print(f"✅ Cultural Context: {context} (confidence: {context_conf:.3f})")
            except Exception as e:
                print(f"❌ Cultural Context Error: {e}")
                context = "speech"
                context_conf = 0.0
            
            # Test language detection
            print("🌐 Processing language detection...")
            try:
                # Simple language detection based on transcription
                if transcription and transcription != "Transcription failed":
                    # Check for Hindi characters
                    hindi_chars = 'अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह'
                    hindi_count = sum(1 for char in transcription if char in hindi_chars)
                    total_chars = len(transcription)
                    
                    if total_chars > 0 and hindi_count / total_chars > 0.1:
                        language = "hi"
                        language_conf = 0.8
                    else:
                        # Check for English indicators
                        english_indicators = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
                        transcription_lower = transcription.lower()
                        english_count = sum(1 for word in english_indicators if word in transcription_lower)
                        
                        if english_count > 0:
                            language = "en"
                            language_conf = 0.7
                        else:
                            language = "unknown"
                            language_conf = 0.3
                else:
                    language = "unknown"
                    language_conf = 0.3
                
                print(f"✅ Language: {language} (confidence: {language_conf:.3f})")
                
            except Exception as e:
                print(f"❌ Language Detection Error: {e}")
                language = "unknown"
                language_conf = 0.3
            
            # Create test result
            test_result = {
                'file_path': sample['filepath'],
                'ground_truth': {
                    'emotion': sample['emotion'],
                    'cultural_context': sample['type'],
                    'language': sample.get('language', 'unknown')
                },
                'predictions': {
                    'transcription': transcription,
                    'emotion': emotion,
                    'cultural_context': context,
                    'language': language
                },
                'confidence': {
                    'emotion': emotion_conf,
                    'cultural_context': context_conf,
                    'language': language_conf
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Calculate accuracy metrics
            self._calculate_accuracy_metrics(test_result)
            
            return test_result
            
        except Exception as e:
            print(f"❌ Error testing {sample['filepath']}: {e}")
            return None
    
    def _calculate_accuracy_metrics(self, test_result):
        """Calculate accuracy metrics for a test result."""
        gt = test_result['ground_truth']
        pred = test_result['predictions']
        
        # Emotion accuracy
        emotion_correct = 1 if gt['emotion'] == pred['emotion'] else 0
        self.performance_metrics['emotion_accuracy'].append(emotion_correct)
        
        # Cultural context accuracy
        context_correct = 1 if gt['cultural_context'] == pred['cultural_context'] else 0
        self.performance_metrics['context_accuracy'].append(context_correct)
        
        # Language accuracy (if available)
        if gt['language'] != 'unknown':
            language_correct = 1 if gt['language'] == pred['language'] else 0
            self.performance_metrics['language_accuracy'].append(language_correct)
        
        # Confidence scores
        confidence = test_result['confidence']
        if 'emotion' in confidence:
            self.performance_metrics['emotion_confidence'].append(confidence['emotion'])
        if 'cultural_context' in confidence:
            self.performance_metrics['context_confidence'].append(confidence['cultural_context'])
        if 'language' in confidence:
            self.performance_metrics['language_confidence'].append(confidence['language'])
    
    def test_all_samples(self, num_samples=10):
        """Test all selected samples."""
        print("🧪 ALM TESTING - DATASET SAMPLES")
        print("="*70)
        
        # Select test samples
        samples = self.select_test_samples(num_samples)
        if not samples:
            print("❌ No samples selected")
            return
        
        # Test each sample
        successful_tests = 0
        for i, sample in enumerate(samples):
            print(f"\n📊 TEST {i+1}/{len(samples)}")
            result = self.test_audio_sample(sample)
            if result:
                self.results.append(result)
                successful_tests += 1
        
        print(f"\n✅ Successfully tested {successful_tests}/{len(samples)} samples")
        
        # Generate performance report
        self.generate_performance_report()
    
    def generate_performance_report(self):
        """Generate comprehensive performance report."""
        print("\n📊 PERFORMANCE ANALYSIS REPORT")
        print("="*70)
        
        if not self.results:
            print("❌ No test results available")
            return
        
        # Overall metrics
        total_tests = len(self.results)
        print(f"Total tests: {total_tests}")
        
        # Emotion accuracy
        if self.performance_metrics['emotion_accuracy']:
            emotion_accuracy = np.mean(self.performance_metrics['emotion_accuracy']) * 100
            print(f"🎭 Emotion Recognition Accuracy: {emotion_accuracy:.1f}%")
        
        # Cultural context accuracy
        if self.performance_metrics['context_accuracy']:
            context_accuracy = np.mean(self.performance_metrics['context_accuracy']) * 100
            print(f"🌍 Cultural Context Accuracy: {context_accuracy:.1f}%")
        
        # Language accuracy
        if self.performance_metrics['language_accuracy']:
            language_accuracy = np.mean(self.performance_metrics['language_accuracy']) * 100
            print(f"🌐 Language Detection Accuracy: {language_accuracy:.1f}%")
        
        # Average confidence scores
        if self.performance_metrics['emotion_confidence']:
            avg_emotion_conf = np.mean(self.performance_metrics['emotion_confidence']) * 100
            print(f"😊 Average Emotion Confidence: {avg_emotion_conf:.1f}%")
        
        if self.performance_metrics['context_confidence']:
            avg_context_conf = np.mean(self.performance_metrics['context_confidence']) * 100
            print(f"🌍 Average Context Confidence: {avg_context_conf:.1f}%")
        
        if self.performance_metrics['language_confidence']:
            avg_language_conf = np.mean(self.performance_metrics['language_confidence']) * 100
            print(f"🌐 Average Language Confidence: {avg_language_conf:.1f}%")
        
        # Save detailed results
        results_file = Path("test_results_dataset.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        # Performance summary
        summary_file = Path("performance_summary_dataset.json")
        summary = {
            'total_tests': total_tests,
            'emotion_accuracy': np.mean(self.performance_metrics['emotion_accuracy']) * 100 if self.performance_metrics['emotion_accuracy'] else 0,
            'context_accuracy': np.mean(self.performance_metrics['context_accuracy']) * 100 if self.performance_metrics['context_accuracy'] else 0,
            'language_accuracy': np.mean(self.performance_metrics['language_accuracy']) * 100 if self.performance_metrics['language_accuracy'] else 0,
            'avg_emotion_confidence': np.mean(self.performance_metrics['emotion_confidence']) * 100 if self.performance_metrics['emotion_confidence'] else 0,
            'avg_context_confidence': np.mean(self.performance_metrics['context_confidence']) * 100 if self.performance_metrics['context_confidence'] else 0,
            'avg_language_confidence': np.mean(self.performance_metrics['language_confidence']) * 100 if self.performance_metrics['language_confidence'] else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"📊 Performance summary saved to: {summary_file}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if emotion_accuracy < 50:
            print("   🎭 Consider retraining emotion model for better accuracy")
        if context_accuracy < 70:
            print("   🌍 Consider retraining cultural context model")
        if language_accuracy < 60:
            print("   🌐 Consider improving language detection logic")
        
        if emotion_accuracy > 50 and context_accuracy > 70 and language_accuracy > 60:
            print("   ✅ All models performing well!")

def main():
    """Main function to test ALM on dataset samples."""
    # Initialize tester
    tester = DatasetTester()
    
    # Test models on dataset samples
    tester.test_all_samples(num_samples=15)
    
    print(f"\n🎉 DATASET TESTING COMPLETE!")
    print("📊 Check the generated JSON files for detailed results")

if __name__ == "__main__":
    main()
