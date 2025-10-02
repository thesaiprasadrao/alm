#!/usr/bin/env python3
"""
Audio Dataset Metadata Builder
Scans audio dataset and creates master_metadata.csv with unified metadata from all audio files.
"""

import os
import csv
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioMetadataBuilder:
    def __init__(self, root_dir: str, output_file: str = "master_metadata.csv"):
        self.root_dir = Path(root_dir)
        self.output_file = output_file
        self.metadata_rows = []
        
        # Emotion mapping
        self.emotion_map = {
            'a': 'anger',
            'd': 'disgust', 
            'f': 'fear',
            'h': 'happiness',
            's': 'sadness'
        }
        
        # Dataset statistics
        self.stats = {
            'speech_emotional': 0,
            'non_speech_urban': 0,
            'environmental_esc50': 0,
            'environmental_tut': 0,
            'environmental_urban8k': 0,
            'multilingual_en': 0,
            'multilingual_hi': 0
        }

    def parse_emotional_speech_filename(self, filename: str) -> Tuple[str, str, str]:
        """
        Parse emotional speech filename: xAA (B).wav or xAA (B)b.wav
        Returns: (emotion, utterance_id, speaker_id)
        """
        # Remove extension
        name = filename.replace('.wav', '')
        
        # Handle duplicate takes (e.g., f18 (5)b.wav)
        if name.endswith(('a', 'b', 'c', 'd', 'e', 'f')):
            name = name[:-1]  # Remove suffix
        
        # Extract emotion, utterance, speaker
        pattern = r'([adfhs])(\d+)\s*\((\d+)\)'
        match = re.match(pattern, name)
        
        if match:
            emotion_letter = match.group(1)
            utterance = match.group(2)
            speaker = match.group(3)
            emotion = self.emotion_map.get(emotion_letter, 'unknown')
            return emotion, utterance, speaker
        
        return 'unknown', 'unknown', 'unknown'

    def process_speech_emotional(self) -> None:
        """Process Acted Emotional Speech dataset"""
        logger.info("Processing Speech Audio - Emotional Speech...")
        
        speech_dir = self.root_dir / "Speech Audio" / "Acted Emotional Speech Dynamic Database" / "Acted Emotional Speech Dynamic Database"
        
        if not speech_dir.exists():
            logger.warning(f"Speech emotional directory not found: {speech_dir}")
            return
        
        for emotion_folder in speech_dir.iterdir():
            if emotion_folder.is_dir():
                emotion_name = emotion_folder.name
                logger.info(f"Processing emotion: {emotion_name}")
                
                for audio_file in emotion_folder.glob("*.wav"):
                    emotion, utterance_id, speaker_id = self.parse_emotional_speech_filename(audio_file.name)
                    
                    row = {
                        'filename': audio_file.name,
                        'filepath': str(audio_file.relative_to(self.root_dir)),
                        'type': 'speech',
                        'category': 'emotional',
                        'subcategory': emotion_name,
                        'emotion': emotion,
                        'language': '',
                        'speaker_id': speaker_id,
                        'utterance_id': utterance_id,
                        'label': emotion,
                        'transcription': '',
                        'dataset': 'ActedEmo'
                    }
                    self.metadata_rows.append(row)
                    self.stats['speech_emotional'] += 1

    def process_non_speech_audio(self) -> None:
        """Process Non-Speech Audio dataset"""
        logger.info("Processing Non-Speech Audio...")
        
        non_speech_dir = self.root_dir / "Non Speech Audio"
        
        if not non_speech_dir.exists():
            logger.warning(f"Non-Speech directory not found: {non_speech_dir}")
            return
        
        # Look for CSV metadata file
        csv_files = list(non_speech_dir.glob("*.csv"))
        if not csv_files:
            logger.warning("No CSV metadata file found in Non-Speech Audio")
            return
        
        csv_file = csv_files[0]  # Use first CSV found
        logger.info(f"Using metadata file: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
            
            for _, row in df.iterrows():
                filename = row.get('slice_file_name', '')
                if filename and filename.endswith('.wav'):
                    audio_path = non_speech_dir / filename
                    if audio_path.exists():
                        metadata_row = {
                            'filename': filename,
                            'filepath': str(audio_path.relative_to(self.root_dir)),
                            'type': 'non-speech',
                            'category': 'environmental',
                            'subcategory': row.get('class', ''),
                            'emotion': '',
                            'language': '',
                            'speaker_id': '',
                            'utterance_id': '',
                            'label': row.get('class', ''),
                            'transcription': '',
                            'dataset': 'UrbanSound8K-NonSpeech'
                        }
                        self.metadata_rows.append(metadata_row)
                        self.stats['non_speech_urban'] += 1
                        
        except Exception as e:
            logger.error(f"Error processing Non-Speech CSV: {e}")

    def process_environmental_esc50(self) -> None:
        """Process ESC-50 Environmental Audio dataset"""
        logger.info("Processing Environmental Audio - ESC-50...")
        
        esc50_dir = self.root_dir / "Mixed Audio" / "Environmental Audio" / "ESC-50-master"
        audio_dir = esc50_dir / "audio"
        meta_file = esc50_dir / "meta" / "esc50.csv"
        
        if not audio_dir.exists() or not meta_file.exists():
            logger.warning(f"ESC-50 directory or metadata not found")
            return
        
        try:
            df = pd.read_csv(meta_file)
            
            for _, row in df.iterrows():
                filename = row.get('filename', '')
                if filename and filename.endswith('.wav'):
                    audio_path = audio_dir / filename
                    if audio_path.exists():
                        metadata_row = {
                            'filename': filename,
                            'filepath': str(audio_path.relative_to(self.root_dir)),
                            'type': 'non-speech',
                            'category': 'environmental',
                            'subcategory': row.get('category', ''),
                            'emotion': '',
                            'language': '',
                            'speaker_id': '',
                            'utterance_id': '',
                            'label': row.get('category', ''),
                            'transcription': '',
                            'dataset': 'ESC-50'
                        }
                        self.metadata_rows.append(metadata_row)
                        self.stats['environmental_esc50'] += 1
                        
        except Exception as e:
            logger.error(f"Error processing ESC-50: {e}")

    def process_environmental_tut(self) -> None:
        """Process TUT Acoustic Scenes dataset"""
        logger.info("Processing Environmental Audio - TUT...")
        
        tut_dir = self.root_dir / "Mixed Audio" / "Environmental Audio" / "TUT-acoustic-scenes-2017-development"
        audio_dir = tut_dir / "audio"
        meta_file = tut_dir / "meta.txt"
        
        if not audio_dir.exists() or not meta_file.exists():
            logger.warning(f"TUT directory or metadata not found")
            return
        
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    audio_path_str = parts[0]
                    scene_label = parts[1]
                    
                    # Extract filename from path
                    filename = os.path.basename(audio_path_str)
                    audio_path = audio_dir / filename
                    
                    if audio_path.exists():
                        metadata_row = {
                            'filename': filename,
                            'filepath': str(audio_path.relative_to(self.root_dir)),
                            'type': 'non-speech',
                            'category': 'environmental',
                            'subcategory': scene_label,
                            'emotion': '',
                            'language': '',
                            'speaker_id': '',
                            'utterance_id': '',
                            'label': scene_label,
                            'transcription': '',
                            'dataset': 'TUT-AcousticScenes'
                        }
                        self.metadata_rows.append(metadata_row)
                        self.stats['environmental_tut'] += 1
                        
        except Exception as e:
            logger.error(f"Error processing TUT: {e}")

    def process_environmental_urban8k(self) -> None:
        """Process UrbanSound8K dataset"""
        logger.info("Processing Environmental Audio - UrbanSound8K...")
        
        urban8k_dir = self.root_dir / "Mixed Audio" / "Environmental Audio" / "UrbanSound8K"
        audio_dir = urban8k_dir / "audio"
        meta_file = urban8k_dir / "UrbanSound8K.csv"
        
        if not audio_dir.exists() or not meta_file.exists():
            logger.warning(f"UrbanSound8K directory or metadata not found")
            return
        
        try:
            df = pd.read_csv(meta_file)
            
            for _, row in df.iterrows():
                filename = row.get('slice_file_name', '')
                if filename and filename.endswith('.wav'):
                    audio_path = audio_dir / filename
                    if audio_path.exists():
                        metadata_row = {
                            'filename': filename,
                            'filepath': str(audio_path.relative_to(self.root_dir)),
                            'type': 'non-speech',
                            'category': 'environmental',
                            'subcategory': row.get('class', ''),
                            'emotion': '',
                            'language': '',
                            'speaker_id': '',
                            'utterance_id': '',
                            'label': row.get('class', ''),
                            'transcription': '',
                            'dataset': 'UrbanSound8K'
                        }
                        self.metadata_rows.append(metadata_row)
                        self.stats['environmental_urban8k'] += 1
                        
        except Exception as e:
            logger.error(f"Error processing UrbanSound8K: {e}")

    def process_multilingual_speech(self, language: str) -> None:
        """Process Multilingual Speech dataset for given language"""
        logger.info(f"Processing Multilingual Speech - {language.upper()}...")
        
        lang_dir = self.root_dir / "Mixed Audio" / "Multilingual Speech" / language
        clips_dir = lang_dir / "clips"
        
        if not clips_dir.exists():
            logger.warning(f"Multilingual {language} clips directory not found")
            return
        
        # Look for TSV metadata files
        tsv_files = list(lang_dir.glob("*.tsv"))
        if not tsv_files:
            logger.warning(f"No TSV metadata files found for {language}")
            return
        
        # Use validated.tsv as primary metadata source, fallback to train.tsv for Hindi
        primary_tsv = lang_dir / "validated.tsv"
        
        # Check if validated.tsv is empty or very small, or if it's Hindi (use train.tsv)
        if (not primary_tsv.exists() or primary_tsv.stat().st_size <= 200) or language == 'hi':
            if language == 'hi' and (lang_dir / "train.tsv").exists():
                primary_tsv = lang_dir / "train.tsv"
                logger.info(f"Using train.tsv for Hindi language")
            else:
                primary_tsv = tsv_files[0]  # Use first available TSV
        
        try:
            df = pd.read_csv(primary_tsv, sep='\t')
            
            for _, row in df.iterrows():
                path = row.get('path', '')
                if path and path.endswith('.mp3'):
                    audio_path = clips_dir / path
                    if audio_path.exists():
                        # Extract speaker info if available
                        speaker_id = row.get('client_id', '')
                        if not speaker_id:
                            speaker_id = row.get('speaker_id', '')
                        
                        metadata_row = {
                            'filename': os.path.basename(path),
                            'filepath': str(audio_path.relative_to(self.root_dir)),
                            'type': 'speech',
                            'category': 'multilingual',
                            'subcategory': language,
                            'emotion': '',
                            'language': language,
                            'speaker_id': speaker_id,
                            'utterance_id': '',
                            'label': language,
                            'transcription': row.get('sentence', ''),
                            'dataset': 'CommonVoice'
                        }
                        self.metadata_rows.append(metadata_row)
                        
                        if language == 'en':
                            self.stats['multilingual_en'] += 1
                        else:
                            self.stats['multilingual_hi'] += 1
                            
        except Exception as e:
            logger.error(f"Error processing Multilingual {language}: {e}")

    def save_metadata(self) -> None:
        """Save all metadata to CSV file"""
        logger.info(f"Saving metadata to {self.output_file}...")
        
        if not self.metadata_rows:
            logger.warning("No metadata rows to save")
            return
        
        # Define column order
        columns = [
            'filename', 'filepath', 'type', 'category', 'subcategory', 
            'emotion', 'language', 'speaker_id', 'utterance_id', 
            'label', 'transcription', 'dataset'
        ]
        
        # Create DataFrame and save
        df = pd.DataFrame(self.metadata_rows)
        df = df.reindex(columns=columns, fill_value='')
        df.to_csv(self.output_file, index=False, encoding='utf-8')
        
        logger.info(f"Saved {len(self.metadata_rows)} records to {self.output_file}")

    def print_summary(self) -> None:
        """Print processing summary"""
        print("\n" + "="*60)
        print("AUDIO DATASET METADATA PROCESSING SUMMARY")
        print("="*60)
        
        total_files = sum(self.stats.values())
        print(f"Total files processed: {total_files}")
        print("\nBreakdown by dataset:")
        print(f"  Speech Emotional:     {self.stats['speech_emotional']:>6}")
        print(f"  Non-Speech Urban:    {self.stats['non_speech_urban']:>6}")
        print(f"  Environmental ESC-50: {self.stats['environmental_esc50']:>6}")
        print(f"  Environmental TUT:   {self.stats['environmental_tut']:>6}")
        print(f"  Environmental Urban8K:{self.stats['environmental_urban8k']:>6}")
        print(f"  Multilingual EN:     {self.stats['multilingual_en']:>6}")
        print(f"  Multilingual HI:     {self.stats['multilingual_hi']:>6}")
        
        print(f"\nOutput file: {self.output_file}")
        print("="*60)

    def run(self) -> None:
        """Main execution method"""
        logger.info(f"Starting metadata extraction from: {self.root_dir}")
        
        # Process all dataset types
        self.process_speech_emotional()
        self.process_non_speech_audio()
        self.process_environmental_esc50()
        self.process_environmental_tut()
        self.process_environmental_urban8k()
        self.process_multilingual_speech('en')
        self.process_multilingual_speech('hi')
        
        # Save results
        self.save_metadata()
        self.print_summary()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build master metadata CSV from audio dataset')
    parser.add_argument('--root-dir', '-r', default='.', 
                       help='Root directory of audio dataset (default: current directory)')
    parser.add_argument('--output', '-o', default='master_metadata.csv',
                       help='Output CSV filename (default: master_metadata.csv)')
    
    args = parser.parse_args()
    
    # Validate root directory
    root_path = Path(args.root_dir)
    if not root_path.exists():
        print(f"Error: Root directory does not exist: {root_path}")
        return 1
    
    # Create builder and run
    builder = AudioMetadataBuilder(str(root_path), args.output)
    builder.run()
    
    return 0


if __name__ == "__main__":
    exit(main())
