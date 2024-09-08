import torch
import torch.nn.functional as F

def flatten_embeddings(enc_out):
    # Flatten the embeddings into a long 1D tensor
    return enc_out.view(-1)

def min_max_normalize(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    return (tensor - min_val) / (max_val - min_val)

def compute_distances(enc_out, channel_ids, label_encoder, matrix = True):
    """
    Compute the average normalized L2 distance of each channel to channel 4.
    """
    num_channels = 8

    distance_matrix = torch.zeros((num_channels, num_channels)) if matrix else None 
    
    # Split enc_out into stems (each stem has 8 channels)
    stems = enc_out.split(num_channels)  
    channels = channel_ids.split(num_channels) 
    
    channel_order = []
    
    for stem, channel_id_set in zip(stems, channels):
        # Compute mean pooled embeddings for each channel in the stem
        flattend = [flatten_embeddings(channel) for channel in stem]
        
        # Normalize each vector using min-max normalization
        normalized_vectors = [min_max_normalize(vector) for vector in flattend]
        
            
        for idx, (channel_id, vector) in enumerate(zip(channel_id_set, normalized_vectors)):
            
            if matrix:
                for jdx, other_vector in enumerate(normalized_vectors):
                    if jdx < idx:  # Only fill the lower triangle
                        distance = F.pairwise_distance(vector.unsqueeze(0), other_vector.unsqueeze(0), p=2).item()
                        distance_matrix[channel_id.item(), channel_id_set[jdx].item()] += distance
                        distance_matrix[channel_id_set[jdx].item(), channel_id.item()] += distance
                
    for channel_id, decoded_name in channel_order:
        print(f"Channel ID: {channel_id}, Decoded: {decoded_name}")
    
    if matrix:
        return distance_matrix
