#!/usr/bin/env python3
"""
Test ALM on audio files in the root directory.
"""

import sys
from pathlib import Path
import torch
import librosa
import numpy as np
import json
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from fix_alm_critical_issues import (
    FixedEmotionClassifier,
    FixedCulturalContextClassifier,
    AdvancedAudioFeatureExtractor
)

class ALMTester:
    """Test ALM on root audio files."""
    
    def __init__(self):
        self.emotion_classifier = FixedEmotionClassifier()
        self.cultural_context_classifier = FixedCulturalContextClassifier()
        self.feature_extractor = AdvancedAudioFeatureExtractor()
        
    def test_audio_file(self, audio_file):
        """Test a single audio file."""
        print(f"\n🎵 TESTING: {audio_file}")
        print("="*60)
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_file, sr=16000)
            
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
            
            # Generate reasoning
            print("💡 Generating reasoning...")
            reasoning_parts = []
            
            if emotion in ['fear', 'anger']:
                reasoning_parts.append(f"Person appears {emotion}")
            elif emotion in ['happiness', 'sadness']:
                reasoning_parts.append(f"Person seems {emotion}")
            
            if context == 'non-speech':
                reasoning_parts.append("Audio contains primarily non-speech content")
            else:
                reasoning_parts.append("Audio contains clear speech content")
            
            if language == 'hi':
                reasoning_parts.append("Audio contains Hindi speech")
            elif language == 'en':
                reasoning_parts.append("Audio contains English speech")
            
            if transcription and transcription != "Transcription failed":
                if len(transcription) > 20:
                    reasoning_parts.append("Audio contains substantial speech content")
                elif len(transcription) < 10:
                    reasoning_parts.append("Audio contains brief speech content")
            
            reasoning = ". ".join(reasoning_parts) + "." if reasoning_parts else "Unable to determine context from available information."
            print(f"✅ Reasoning: {reasoning}")
            
            # Results summary
            print(f"\n📊 ALM ANALYSIS RESULTS")
            print("="*40)
            print(f"🎤 Transcription: {transcription[:50]}...")
            print(f"😊 Emotion: {emotion} ({emotion_conf:.1%} confidence)")
            print(f"🌍 Cultural Context: {context} ({context_conf:.1%} confidence)")
            print(f"🌐 Language: {language} ({language_conf:.1%} confidence)")
            print(f"💡 Reasoning: {reasoning}")
            
            return {
                'file': audio_file,
                'transcription': transcription,
                'emotion': emotion,
                'emotion_confidence': emotion_conf,
                'cultural_context': context,
                'cultural_context_confidence': context_conf,
                'language': language,
                'language_confidence': language_conf,
                'reasoning': reasoning,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error processing {audio_file}: {e}")
            return None

def main():
    """Main function to test ALM on root audio files."""
    print("🧪 ALM TESTING - ROOT AUDIO FILES")
    print("="*70)
    
    # Initialize tester
    tester = ALMTester()
    
    # Find audio files in root directory
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
    root_files = []
    
    for file_path in Path('.').iterdir():
        if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
            root_files.append(str(file_path))
    
    if not root_files:
        print("❌ No audio files found in root directory")
        print("💡 Place audio files (.mp3, .wav, .m4a, .flac, .ogg) in the root directory")
        return
    
    print(f"📁 Found {len(root_files)} audio files:")
    for file in root_files:
        print(f"   - {file}")
    
    # Test each file
    results = []
    for audio_file in root_files:
        result = tester.test_audio_file(audio_file)
        if result:
            results.append(result)
    
    # Summary
    if results:
        print(f"\n🎉 TESTING COMPLETE!")
        print("="*50)
        print(f"📊 Tested {len(results)} files successfully")
        
        # Calculate average confidence
        avg_emotion_conf = np.mean([r['emotion_confidence'] for r in results])
        avg_context_conf = np.mean([r['cultural_context_confidence'] for r in results])
        avg_language_conf = np.mean([r['language_confidence'] for r in results])
        
        print(f"\n📈 Average Confidence Scores:")
        print(f"   🎭 Emotion: {avg_emotion_conf:.1%}")
        print(f"   🌍 Cultural Context: {avg_context_conf:.1%}")
        print(f"   🌐 Language: {avg_language_conf:.1%}")
        
        # Save results
        results_file = Path("test_results_root.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {results_file}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if avg_emotion_conf < 0.5:
            print("   🎭 Consider retraining emotion model for better accuracy")
        if avg_context_conf < 0.7:
            print("   🌍 Consider retraining cultural context model")
        if avg_language_conf < 0.6:
            print("   🌐 Consider improving language detection logic")
        
        if avg_emotion_conf > 0.5 and avg_context_conf > 0.7 and avg_language_conf > 0.6:
            print("   ✅ All models performing well!")
    else:
        print("❌ No files were successfully processed")

if __name__ == "__main__":
    main()
