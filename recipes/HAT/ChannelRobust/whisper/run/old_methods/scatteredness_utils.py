import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from collections import defaultdict
import matplotlib.cm as cm
from scipy.stats.stats import pearsonr 
import logging



def cal_scatteredness(emb_dicts_list, output_path):
    """
    Called after each validation/test epoch.
    Arguments
    ---------
    emb_dicts_list: list of dicts,
        takes the form:
        [
            {"stem_id": stem_encoded, "channel_id": channel_id_encoded, "embedding": tensor},
            {...},
            ...
        ]
    `scatteredness` is defined as the expected determinant of the correlation matrix
    for each stem, averaged over all stems.

    Each embedding as $x \in \R^d$, let number of stems as $N$.

    A stem is $X \in \R^{d\times c}$, has $c$ channels per stem,
    each channel in a stem has a specific embedding.

    scatteredness is defined by:
        $ \mathbb{E}_{N} [\det(\text{Corr}_{d\times d}(X))] $

    i.e., the expected value over the number of stems of the determinant
    of the correlation matrix of the embeddings with the same `stem_id`.
    """

    # Organize embeddings by stem_id
    stem_embeddings = {}
    for item in emb_dicts_list:
        stem_id = item['stem_id']
        embedding = item['embedding'].cpu().numpy()  # Convert tensor to numpy array
        if stem_id not in stem_embeddings:
            stem_embeddings[stem_id] = []
        stem_embeddings[stem_id].append(embedding)


    for stem_id in stem_embeddings.keys():
        num_embeddings = len(stem_embeddings[stem_id])

        # Create an empty 8x8 correlation matrix for this stem
        correlation_matrix = np.zeros((num_embeddings, num_embeddings))
        correlation_matrix_sum = np.zeros((num_embeddings, num_embeddings))
        num_correlation_matrices = 0

        # Calculate correlation coefficient (scalar) for each pair of embeddings
        for i in range(num_embeddings):
            for j in range(i + 1, num_embeddings):  # Avoid redundancy (i, j) and (j, i)
                embedding_i = stem_embeddings[stem_id][i]
                embedding_j = stem_embeddings[stem_id][j]
                correlation, _ = pearsonr(embedding_i, embedding_j)  # Extract only correlation
                correlation_matrix[i][j] = correlation
                correlation_matrix[j][i] = correlation  # Fill symmetric entries
            correlation_matrix[i][i] = 1

        correlation_matrix_sum += correlation_matrix
        num_correlation_matrices += 1
        
    mean_correlation_matrix = correlation_matrix_sum / num_correlation_matrices

    # Compute the expected value of the determinant (mean of correlation determinants)
    scatteredness = np.linalg.det(mean_correlation_matrix)

    print("scatteredness:", scatteredness)
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(mean_correlation_matrix, annot=True, cmap='coolwarm', center=0, cbar_kws={'label': 'Correlation'})
    plt.title('Heatmap of Mean Correlation Matrix')
    plt.xlabel('Embedding Index')
    plt.ylabel('Embedding Index')
    plt.show()
    
    print(f'Plot saved at {output_path}/heatmap.png')
    
    return scatteredness, mean_correlation_matrix

def plotTSNE(emb_dicts_list, output_path):
    """
    Plot t-SNE for all the embeddings and save to output_path, color-marked by the channel_id.
    Arguments
    ---------
    emb_dicts_list: list of dicts,
    takes the form:
    [
        {"stem_id": stem_encoded, "channel_id": channel_id_encoded, "embedding": tensor}, 
        {...}, 
        ...
    ]
    output_path: str, path to save the plot
    """
    embeddings = []
    channel_ids = []
    
    # Extract embeddings and channel_ids
    for emb_dict in emb_dicts_list:
        embeddings.append(emb_dict['embedding'].cpu().numpy())
        channel_ids.append(emb_dict['channel_id'])

    embeddings = np.array(embeddings)
    channel_ids = np.array(channel_ids)
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_fit = tsne.fit_transform(embeddings)
    
    # Create a color map with distinct colors for channel_id values
    unique_channel_ids = np.unique(channel_ids)
    colors = cm.get_cmap('Dark2', len(unique_channel_ids))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot t-SNE
    scatter = ax.scatter(
        embeddings_fit[:, 0], 
        embeddings_fit[:, 1], 
        c=channel_ids, 
        cmap=colors,
        alpha=0.6,
        s=20  # Set the dot size smaller
    )
    
    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Channel ID')
    cbar.set_ticks(np.arange(len(unique_channel_ids)) + 0.5)
    cbar.set_ticklabels(unique_channel_ids)
    
    # Add title and labels
    ax.set_title('t-SNE of Embeddings')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    
    # Save the plot
    plt.savefig(f"{output_path}/tSNE.png")
    plt.close()
    
    print(f'Plot saved at {output_path}/tSNE.png')
