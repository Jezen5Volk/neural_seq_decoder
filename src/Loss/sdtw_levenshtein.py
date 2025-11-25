import torch
import torch.nn.functional as F

# Default regularization parameter
DEFAULT_GAMMA = 5.0
DEFAULT_BLANK = 0

def logsumexp_k(tensors, gamma=DEFAULT_GAMMA):
    """
    Computes the log-sum-exp (soft-minimum) for a list of tensors.
    
    This is the mathematically stable, differentiable replacement for min(*tensors).
    The soft-minimum is calculated as: -gamma * log(sum(exp(-t_i / gamma)))
    """
    if not tensors:
        raise ValueError("Input list of tensors cannot be empty.")
    
    # Initialize LSE with the first tensor
    lse_result = tensors[0] / gamma
    
    # Iteratively apply torch.logaddexp for stability
    for i in range(1, len(tensors)):
        lse_result = torch.logaddexp(lse_result, tensors[i] / gamma)
        
    return lse_result * gamma

def batched_soft_edit_distance(
    input_seqs: torch.Tensor,
    input_lengths: torch.Tensor,   
    target_seqs: torch.Tensor,
    target_lengths: torch.Tensor, 
    blank_token_id: int = DEFAULT_BLANK,
    gamma: float = DEFAULT_GAMMA
) -> torch.Tensor:
    """
    Calculates the Batched Differentiable Soft Levensshtein Edit Distance (Soft-SED), 
    handling padded sequences for both input and target.

    Args:
        input_seqs (torch.Tensor): The source sequences (Batch, L1_max), potentially padded.
        input_lengths (torch.Tensor): The true, unpadded length of each input sequence (Batch,).
        target_seqs (torch.Tensor): The target sequences (Batch, L2_max), potentially padded.
        target_lengths (torch.Tensor): The true, unpadded length of each target sequence (Batch,).
        blank_token_id (int): The token ID in input_seqs whose deletion cost is 0.
        gamma (float): Regularization parameter for the soft-minimum.

    Returns:
        torch.Tensor: A tensor of shape (Batch,) containing the soft edit distance 
                      for each sequence pair, evaluated at its true end point.
    """
    B, L1 = input_seqs.shape
    _, L2 = target_seqs.shape
    
    # --- Initialize DP Table D ---
    # D[b, i, j] holds the soft edit distance for the b-th pair of sequences 
    # seq1[:i] and seq2[:j]. D is initialized with zeros and has dimensions (B, L1+1, L2+1).
    D = torch.zeros((B, L1 + 1, L2 + 1), device=input_seqs.device, dtype=input_seqs.dtype)
    
    # Create masks for length checks
    # Index tensors for i and j (1-based indices corresponding to D table)
    i_indices = torch.arange(1, L1 + 1, device=input_seqs.device).unsqueeze(0).repeat(B, 1) # (B, L1)
    j_indices = torch.arange(1, L2 + 1, device=input_seqs.device).unsqueeze(0).repeat(B, 1) # (B, L2)
    
    # Mask: 1.0 if (i, j) is inside the unpadded sequence boundaries, 0.0 otherwise.
    # Note: We only need the j-mask in the loop, but i-mask is useful for border init.
    input_mask = (i_indices <= input_lengths.unsqueeze(1)).float()  # (B, L1)
    target_mask = (j_indices <= target_lengths.unsqueeze(1)).float() # (B, L2)
    
    # --- Initialize Borders (Insertion/Deletion) ---
    # Calculate DelCost for all input tokens (1 for non-blank, 0 for blank)
    is_blank = (input_seqs == blank_token_id).float()
    del_costs_raw = 1.0 - is_blank  # (B, L1) - DelCost[b, i-1]
    
    # Crucial: Zero out costs corresponding to padded parts of the input sequence
    del_costs = del_costs_raw * input_mask # (B, L1) - Cost is only applied if within true length

    # Calculate the cumulative deletion cost for the first column D[i, 0]
    D[:, 1:, 0] = torch.cumsum(del_costs, dim=1)
    
    # Initial costs for the first row (i = 0, j > 0) are accumulated insertions.
    # Insertion cost is 1.0. We mask this cost based on the target length.
    insertion_costs = torch.ones_like(j_indices, dtype=input_seqs.dtype) # (B, L2)
    insertion_costs_masked = insertion_costs * target_mask
    D[:, 0, 1:] = torch.cumsum(insertion_costs_masked, dim=1)

    # --- Compute Substitution/Match Cost C ---
    # Check if tokens mismatch (1.0 for mismatch, 0.0 for match)
    mismatch_indicator = (input_seqs.unsqueeze(2) != target_seqs.unsqueeze(1)).float() # (B, L1, L2)
    
    # Sub/Match cost is 1.0 only if mismatch AND both tokens are within their true lengths.
    # The mask for C[i, j] is input_mask[i-1] * target_mask[j-1]
    cost_mask = input_mask.unsqueeze(2) * target_mask.unsqueeze(1) # (B, L1, 1) * (B, 1, L2) -> (B, L1, L2)

    C_raw = mismatch_indicator * cost_mask # Only calculate cost within true boundaries
    
    # C is padded by 1 at the start of L1 and L2 for easier indexing
    C = F.pad(C_raw, (1, 0, 1, 0)) # (B, L1+1, L2+1)
    
    # --- Dynamic Programming (Batched) ---
    for i in range(1, L1 + 1):
        for j in range(1, L2 + 1):
            
            # Mask checks if the current cell D[i, j] should be calculated
            # We only calculate if i is within L1 AND j is within L2 (for this batch element)
            current_mask = (i_indices[:, i-1] <= input_lengths) * (j_indices[:, j-1] <= target_lengths)
            current_mask = current_mask.float() # (B,) mask for updates
            
            # If current_mask is 0, we should skip the calculation for that batch element,
            # but since we cannot skip steps in a batched DP, we ensure T_del, T_ins, T_sub
            # are extremely large (or the previous D value is maintained) when masked.
            
            # --- Recurrence Paths ---
            current_del_cost = del_costs_raw[:, i - 1] # Cost of deleting input token i (0 or 1)
            
            # Deletion Path (from D[i-1, j])
            # Note: We use del_costs_raw here because the DP table D[:, i-1, j] already holds 
            # the accumulated cost up to the boundary. We only add the cost of deleting token i.
            T_del = D[:, i - 1, j] + current_del_cost  # (B,)

            # Insertion Path (from D[i, j-1])
            T_ins = D[:, i, j - 1] + 1.0  # (B,)
            
            # Substitution/Match Path (from D[i-1, j-1])
            T_sub = D[:, i - 1, j - 1] + C_raw[:, i-1, j-1] # (B,)
            
            # Calculate the smooth minimum (log-sum-exp)
            D_new = logsumexp_k([T_del, T_ins, T_sub], gamma=gamma)

            # Apply the mask: only update D[i, j] for sequences where i and j are within the true length.
            # If masked (current_mask=0), D[i, j] retains its previous value (0.0).
            # This is technically fine since the *final* gather operation will only read from the true end cell.
            # However, for robustness, let's only update if the cell is valid.
            
            # Use torch.where to apply the mask explicitly to stop accumulating costs in the padded area:
            D[:, i, j] = torch.where(
                current_mask.bool(), # Condition: Is the (i,j) cell valid for this batch element?
                D_new,              # True: Use the calculated soft distance D_new
                D[:, i, j]          # False: Keep the initial value (0.0) or (optionally) use a large value
            )


    # --- Final Distance Extraction ---
    # We want to extract D[b, input_lengths[b], target_lengths[b]].
    # This point (L1_true, L2_true) is the only required distance path endpoint.
    # Create indices for all batch elements, then index through D
    batch_indices = torch.arange(B, device=input_seqs.device).long()
    final_distances = D[batch_indices, input_lengths.long(), target_lengths.long()]
    
    return final_distances
