import torch
import torch.nn.functional as F

DEFAULT_BLANK = 0
EPSILON = 1e-8

def sequence_length_regularization(
    input_logprob: torch.Tensor,   
    target_lengths: torch.Tensor, 
    blank_token_id: int = DEFAULT_BLANK
) -> torch.Tensor:
    """
    Accounting for blank tokens, calculates the length of the predicted 
    sequence and penalizes sequences that are too short.

    Args:
        input_logprob (torch.Tensor): The logprob from the model (Batch, L1_max, Classes)
        target_lengths (torch.Tensor): The true, unpadded length of each target sequence (Batch,)
        blank_token_id (int): The token ID in input_seqs whose deletion cost should be approximately zero 

    Returns:
        torch.Tensor: A tensor of shape (Batch,) containing the sequence length
        regularization penalty
    """

    _, L1, _ = input_logprob.shape


    seq_len = L1 - torch.sum(torch.exp(input_logprob[:, :, blank_token_id]), dim = -1)
    too_short = F.relu(target_lengths - seq_len) #squared distance
    too_long = F.relu(seq_len - target_lengths)

    return (too_short + too_long) ** 2
