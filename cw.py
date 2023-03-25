# Importing necessary modules/packages
import torchaudio
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import Levenshtein
from colorama import Fore
from tqdm.autonotebook import tqdm

# Class for generating adversarial attack 
class AdversarialAttack():

  def __init__(self, model, device):
    # Constructor method
    self.model = model
    self.device = device

  def is_successful_attack(self, adversarial_audio: torch.Tensor, target_transcription: str, threshold: float = 0.5) -> bool:
    # Forward pass
    predicted_transcription = output, _ = self.model(adversarial_audio.to(self.device))
    
    # Return the WER using Levenshtein formula 
    return Levenshtein.ratio(predicted_transcription, target_transcription) < threshold
  
  def encode_transcription(self, transcription):
    # Define the dictionary
    dictionary = {'-': 0, '|': 1, 'E': 2, 'T': 3, 'A': 4, 'O': 5, 'N': 6, 'I': 7, 'H': 8, 'S': 9, 'R': 10, 'D': 11, 'L': 12, 'U': 13, 'M': 14, 'W': 15, 'C': 16, 'F': 17, 'G': 18, 'Y': 19, 'P': 20, 'B': 21, 'V': 22, 'K': 23, "'": 24, 'X': 25, 'J': 26, 'Q': 27, 'Z': 28}

    # Convert transcription string to list of characters
    chars = list(transcription)

    # Encode each character using the dictionary
    encoded_chars = [dictionary[char] for char in chars]

    # Concatenate the encoded characters to form the final encoded transcription
    encoded_transcription = torch.tensor(encoded_chars)

    # Returning the encoded transcription
    return encoded_transcription
  
  def CW_ASR(self, input_audio: np.ndarray, target_transcription: str,
                         eps: float = 0.0, c: float = 1e-4, learning_rate: float = 0.01,
                         max_iter: int = 1000, decrease_factor_eps: float = 0.8,
                         num_iter_decrease_eps: int = 10, opt: str = None) -> np.ndarray:
    # Convert the input audio to a PyTorch tensor
    input_audio = torch.from_numpy(input_audio).to(self.device).float()
    input_audio.requires_grad_()
    input_audio_orig = input_audio.clone()
    
    # Encode the target transcription
    encoded_transcription = self.encode_transcription(target_transcription)
    
    # Convert the target transcription to a PyTorch tensor
    target = torch.from_numpy(np.array(encoded_transcription)).to(self.device).long()
    
    if opt == "Adam":
    # Define the optimizer
        optimizer = torch.optim.Adam([input_audio], lr=learning_rate)
    else:
        optimizer = torch.optim.SGD([input_audio], lr=learning_rate)
    # Run the optimization loop
    successful_attack = False
    num_successful_attacks = 0
    
    for i in tqdm(range(max_iter), colour="red"):
        # Zero the gradients
        optimizer.zero_grad()
        
        # Compute the model’s prediction
        output, _ = self.model(input_audio)
        output = F.log_softmax(output, dim=-1)
        
        # Compute the CTC loss function
        output_lengths = torch.tensor([output.shape[1]], dtype=torch.long).to(self.device)
        output = output.transpose(0, 1)
        target_lengths = torch.tensor([len(encoded_transcription)], dtype=torch.long).to(self.device)
        loss_classifier = F.ctc_loss(output, target, output_lengths, target_lengths, blank=0, reduction='mean')
        
        # Regularization term to minimize the perturbation
        loss_regularizer = c * torch.norm(input_audio - input_audio_orig)
        
        # Combine the losses and compute gradients
        loss = loss_classifier + loss_regularizer
        loss.backward()
        
        # Update the input audio with gradients
        optimizer.step()
        
        # Project the perturbation onto the epsilon ball and clip to audio range using PGD technique which author used in his paper
        perturbation = input_audio - input_audio_orig
        perturbation = torch.clamp(perturbation, -eps, eps)
        input_audio.data = torch.clamp(input_audio_orig + perturbation, -1, 1)
        
        # Check if the attack is successful (for targetted attacks only)
        if self.is_successful_attack(input_audio, target_transcription):
            num_successful_attacks += 1
            if num_successful_attacks >= num_iter_decrease_eps:
                successful_attack = True
                eps *= decrease_factor_eps
                num_successful_attacks = 0
            else:
                successful_attack = False
                num_successful_attacks = 0
        if successful_attack and eps <= 0:
            break
    
    # Return the adversarial audio
    return input_audio.detach().cpu().numpy()
