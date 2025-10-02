"""
Transcription model for ALM project.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
from typing import Dict, Any, Optional, List
import logging
import numpy as np


class TranscriptionModel(nn.Module):
    """Transcription model using Wav2Vec2 for speech-to-text conversion."""
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        vocab_size: Optional[int] = None,
        device: str = "cuda"
    ):
        """Initialize transcription model.
        
        Args:
            model_name: HuggingFace model name
            vocab_size: Vocabulary size for custom tokenizer
            device: Device to run model on
        """
        super().__init__()
        
        self.model_name = model_name
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Load model and processor
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        
        # Move to device
        self.model.to(device)
        
        # Get vocabulary
        self.vocab = self.processor.tokenizer.get_vocab()
        self.vocab_size = len(self.vocab)
        
        self.logger.info(f"Initialized transcription model: {model_name}")
        self.logger.info(f"Vocabulary size: {self.vocab_size}")
    
    def forward(self, audio_inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            audio_inputs: Input audio tensor [batch_size, sequence_length]
            
        Returns:
            Logits tensor [batch_size, sequence_length, vocab_size]
        """
        # Ensure audio is 1D for each sample in the batch
        if audio_inputs.dim() == 2:
            # Batch of 1D audio samples
            batch_size = audio_inputs.size(0)
            # Convert to list of numpy arrays for the processor
            audio_list = [audio_inputs[i].numpy() for i in range(batch_size)]
        else:
            # Single 1D audio sample
            audio_list = [audio_inputs.numpy()]
        
        # Process audio inputs
        inputs = self.processor(
            audio_list,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # Fix the input_values shape if it has extra dimensions
        if 'input_values' in inputs:
            input_values = inputs['input_values']
            if input_values.dim() == 3 and input_values.size(0) == 1:
                # Remove the extra batch dimension
                inputs['input_values'] = input_values.squeeze(0)
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        return logits
    
    def transcribe(
        self,
        audio_inputs: torch.Tensor,
        return_confidence: bool = False
    ) -> List[str]:
        """Transcribe audio to text.
        
        Args:
            audio_inputs: Input audio tensor
            return_confidence: Whether to return confidence scores
            
        Returns:
            List of transcribed texts
        """
        with torch.no_grad():
            # Get logits
            logits = self.forward(audio_inputs)
            
            # Decode predictions
            predicted_ids = torch.argmax(logits, dim=-1)
            transcriptions = self.processor.batch_decode(predicted_ids)
            
            if return_confidence:
                # Calculate confidence scores
                probs = torch.softmax(logits, dim=-1)
                max_probs = torch.max(probs, dim=-1)[0]
                confidences = torch.mean(max_probs, dim=-1)
                
                return transcriptions, confidences.cpu().numpy()
            
            return transcriptions
    
    def transcribe_file(
        self,
        file_path: str,
        return_confidence: bool = False
    ) -> str:
        """Transcribe single audio file.
        
        Args:
            file_path: Path to audio file
            return_confidence: Whether to return confidence score
            
        Returns:
            Transcribed text
        """
        # Load and preprocess audio
        from ..utils.audio_utils import AudioUtils
        audio_utils = AudioUtils()
        
        audio, sr = audio_utils.load_audio(file_path, sample_rate=16000)
        audio_tensor = audio_utils.audio_to_tensor(audio)
        audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
        
        # Transcribe
        if return_confidence:
            transcriptions, confidences = self.transcribe(
                audio_tensor, return_confidence=True
            )
            return transcriptions[0], confidences[0]
        else:
            transcriptions = self.transcribe(audio_tensor)
            return transcriptions[0]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'vocab_size': self.vocab_size,
            'device': self.device,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def save_model(self, save_path: str) -> None:
        """Save model to disk.
        
        Args:
            save_path: Path to save model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'vocab_size': self.vocab_size
        }, save_path)
        
        self.logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str) -> None:
        """Load model from disk.
        
        Args:
            load_path: Path to load model from
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        self.logger.info(f"Model loaded from {load_path}")
    
    def fine_tune(
        self,
        train_dataloader,
        val_dataloader,
        num_epochs: int = 3,
        learning_rate: float = 1e-4,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """Fine-tune the model on custom data.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            save_path: Path to save fine-tuned model
            
        Returns:
            Dictionary with training history
        """
        # Set up optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_wer': []
        }
        
        self.train()
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss = 0.0
            for batch in train_dataloader:
                optimizer.zero_grad()
                
                audio = batch['audio'].to(self.device)
                transcriptions = batch['transcription']
                
                # Forward pass
                logits = self.forward(audio)
                
                # Calculate loss (simplified - would need proper CTC loss)
                loss = self._calculate_ctc_loss(logits, transcriptions)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            val_loss, val_wer = self._evaluate(val_dataloader)
            
            # Update history
            history['train_loss'].append(train_loss / len(train_dataloader))
            history['val_loss'].append(val_loss)
            history['val_wer'].append(val_wer)
            
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {history['train_loss'][-1]:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val WER: {val_wer:.4f}"
            )
        
        # Save model if path provided
        if save_path:
            self.save_model(save_path)
        
        return history
    
    def _calculate_ctc_loss(
        self,
        logits: torch.Tensor,
        transcriptions: List[str]
    ) -> torch.Tensor:
        """Calculate CTC loss for training.
        
        Args:
            logits: Model logits
            transcriptions: Ground truth transcriptions
            
        Returns:
            CTC loss
        """
        # This is a simplified implementation
        # In practice, you would need to properly encode the transcriptions
        # and use torch.nn.CTCLoss
        
        # For now, return a dummy loss
        return torch.tensor(0.0, requires_grad=True)
    
    def _evaluate(self, dataloader) -> tuple:
        """Evaluate model on validation data.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Tuple of (average_loss, wer)
        """
        self.eval()
        total_loss = 0.0
        total_wer = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                audio = batch['audio'].to(self.device)
                transcriptions = batch['transcription']
                
                # Forward pass
                logits = self.forward(audio)
                
                # Calculate loss
                loss = self._calculate_ctc_loss(logits, transcriptions)
                total_loss += loss.item()
                
                # Calculate WER (simplified)
                predicted = self.transcribe(audio)
                wer = self._calculate_wer(predicted, transcriptions)
                total_wer += wer
        
        return total_loss / len(dataloader), total_wer / len(dataloader)
    
    def _calculate_wer(self, predicted: List[str], ground_truth: List[str]) -> float:
        """Calculate Word Error Rate.
        
        Args:
            predicted: Predicted transcriptions
            ground_truth: Ground truth transcriptions
            
        Returns:
            Word Error Rate
        """
        # Simplified WER calculation
        # In practice, you would use a proper WER implementation
        return 0.0
