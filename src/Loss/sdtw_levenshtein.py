import torch
import torch.nn.functional as F

# Default regularization parameter
DEFAULT_GAMMA = 0.5
DEFAULT_BLANK = 0

def logsumexp_k(tensors, gamma=DEFAULT_GAMMA):
    """
    Computes the log-sum-exp (soft-minimum) for a list of tensors.
    
    This is the differentiable replacement for the min operation.
    The soft-minimum is calculated as: -gamma * log(sum(exp(-t_i / gamma)))

    To ensure stability, use the trick of subtracting the maximum, as in the paper
    """

    scaled_tensors = -1*torch.stack(tensors, dim = 0)/gamma
    M, _ = torch.max(scaled_tensors, dim = 0, keepdim = True)
    lse_result = torch.logsumexp(scaled_tensors - M, dim = 0) + M.squeeze(dim = 0)

    return lse_result*gamma*-1
        


def batched_soft_edit_distance(
    input_logprob: torch.Tensor,
    input_lengths: torch.Tensor,   
    target_seqs: torch.Tensor,
    target_lengths: torch.Tensor, 
    blank_token_id: int = DEFAULT_BLANK,
    gamma: float = DEFAULT_GAMMA
) -> torch.Tensor:
    """
    Calculates the Batched Soft Levenshtein Edit Distance, 
    handling padded sequences for both input and target.

    Args:
        input_logprob (torch.Tensor): The source log probabilities (as calculated by softmax) (Batch, L1_max, C)
        input_lengths (torch.Tensor): The true, unpadded length of each input sequence (Batch,)
        target_seqs (torch.Tensor): The target sequences (Batch, L2_max)
        target_lengths (torch.Tensor): The true, unpadded length of each target sequence (Batch,)
        blank_token_id (int): The token ID in input_seqs whose deletion cost is 0
        gamma (float): Regularization parameter for the soft-minimum

    Returns:
        torch.Tensor: A tensor of shape (Batch,) containing the soft edit distance 
                      for each sequence pair, evaluated at its true end point.
    """
    B, L1, C = input_logprob.shape
    _, L2 = target_seqs.shape
    dtype = input_logprob.dtype 
    device = input_logprob.device
    target_1hot = F.one_hot(target_seqs.long(), C).to(dtype)

    # D[b, i, j] holds the soft edit distance for the b-th pair of sequences 
    # seq1[:i] and seq2[:j]. D is initialized with zeros and has dimensions (B, L1+1, L2+1).
    D = torch.zeros((B, L1 + 1, L2 + 1), device=input_logprob.device, dtype=dtype)
    
    # Create masks for length checks
    # Index tensors for i and j (1-based indices corresponding to D table)
    i_indices = torch.arange(1, L1 + 1, device=input_logprob.device).unsqueeze(0).repeat(B, 1) # (B, L1)
    j_indices = torch.arange(1, L2 + 1, device=input_logprob.device).unsqueeze(0).repeat(B, 1) # (B, L2)
    
    # Mask: 1.0 if (i, j) is inside the unpadded sequence boundaries, 0.0 otherwise.
    # Note: Only need the j-mask in the loop, but i-mask is useful for border init.
    input_mask = (i_indices <= input_lengths.unsqueeze(1)).to(dtype)  # (B, L1)
    target_mask = (j_indices <= target_lengths.unsqueeze(1)).to(dtype) # (B, L2)
    
    # Calculate del cost for all input tokens from probability of token being blank token
    del_costs_raw = 1.0 - torch.exp(input_logprob[:, :, blank_token_id]) # (B, L1) 
    del_costs = del_costs_raw * input_mask # Zero out costs corresponding to padded/masked parts of the input sequence

    # Calculate the cumulative deletion cost for the first column D[i, 0]
    D[:, 1:, 0] = torch.cumsum(del_costs, dim=1)
    
    # Initial costs for the first row (i = 0, j > 0) are accumulated insertions.
    insertion_costs = torch.ones_like(j_indices, dtype=dtype) # (B, L2)
    insertion_costs_masked = insertion_costs * target_mask # Mask this cost based on the target length.
    D[:, 0, 1:] = torch.cumsum(insertion_costs_masked, dim=1)

    # Substitution/Match Cost C (ensure high cost when input logits do not align with 1hot target)
    # The mask for C[i, j] is input_mask[i-1] * target_mask[j-1]
    cost_mask = input_mask.unsqueeze(2) * target_mask.unsqueeze(1) # (B, L1, 1) * (B, 1, L2) -> (B, L1, L2)
    C_ij_tilde_raw = -input_logprob @ target_1hot.mT # (B, L1, C) @ (B, C, L2) --> (B, L1, L2)
    C_ij_tilde =  C_ij_tilde_raw*cost_mask # Only calculate cost within true boundaries
    

    #'''
    # Recurrent Programming to Generate D[i, j]
    for k in range(2, L1 + L2 + 1):
        
        #Anti-diagonal indices constrained by i + j = k
        i = torch.arange(max(1, k - L2), min(k-1, L1) + 1, device = device) 
        j = k*torch.ones_like(i) - i

        # Mask checks if the current cell D[i, j] should be calculated
        current_mask = (i_indices[:, i-1].T <= input_lengths).T * (j_indices[:, j-1].T <= target_lengths).T
        
        #Deletion Path (from D[i-1, j])
        T_del = D[:, i - 1, j] + del_costs[:, i - 1]  # (B,)

        # Insertion Path (from D[i, j-1])
        T_ins = D[:, i, j - 1] + 1.0  # (B,)
        
        # Substitution/Match Path (from D[i-1, j-1])
        T_sub = D[:, i - 1, j - 1] + C_ij_tilde[:, i-1, j-1] # (B,)
        
        # Calculate the smooth minimum (log-sum-exp)
        D_ij = logsumexp_k([T_del, T_ins, T_sub], gamma=gamma)

        # apply the mask explicitly to stop accumulating costs in the padded area:
        D[:, i, j] = torch.where(
            current_mask.bool(),
            D_ij,              
            torch.zeros_like(D_ij) #when false, will keep D_ij init value of zero
        )
    #'''




    '''
    # Recurrent Programming to Generate D[i,j]
    for i in range(1, L1 + 1):
        for j in range(1, L2 + 1):
            
            # Mask checks if the current cell D[i, j] should be calculated
            current_mask = (i_indices[:, i-1] <= input_lengths) * (j_indices[:, j-1] <= target_lengths)
            
            #Deletion Path (from D[i-1, j])
            T_del = D[:, i - 1, j] + del_costs[:, i - 1]  # (B,)

            # Insertion Path (from D[i, j-1])
            T_ins = D[:, i, j - 1] + 1.0  # (B,)
            
            # Substitution/Match Path (from D[i-1, j-1])
            T_sub = D[:, i - 1, j - 1] + C_ij_tilde[:, i-1, j-1] # (B,)
            
            # Calculate the smooth minimum (log-sum-exp)
            D_ij = logsumexp_k([T_del, T_ins, T_sub], gamma=gamma) 

            # apply the mask explicitly to stop accumulating costs in the padded area:
            D[:, i, j] = torch.where(
                current_mask.bool(),
                D_ij,              
                torch.zeros_like(D_ij) #when false, will keep D_ij init value of zero
            )
    '''

    # Distance extraction from D matrix
    batch_indices = torch.arange(B, device=input_logprob.device).long()
    final_distances = D[batch_indices, input_lengths.long(), target_lengths.long()]


    return final_distances
