"""Move some basic utils in distill.py in VL-Distill here"""
import os
import numpy as np
import torch

import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment
import joblib
from PIL import Image
from huggingface_hub import login
from diffusers import StableUnCLIPImg2ImgPipeline

__all__ = [
    "nearest_neighbor",
    "load_or_process_file",
]


def nearest_neighbor(sentences, query_embeddings, database_embeddings):
    """
    Find the nearest neighbors for a batch of embeddings.
    """
    nearest_neighbors = []
    
    
    similarities = cosine_similarity(query_embeddings, database_embeddings)

    most_similar_indices = np.argmax(similarities, axis=1)

    nearest_neighbors = [sentences[i] for i in most_similar_indices]
        
    return nearest_neighbors



def load_or_process_file(file_type, process_func, args, data_source):
    """
    Load the processed file if it exists, otherwise process the data source and create the file.

    Args:
    file_type: The type of the file (e.g., 'train', 'test').
    process_func: The function to process the data source.
    args: The arguments required by the process function and to build the filename.
    data_source: The source data to be processed.

    Returns:
    The loaded data from the file.
    """
    if 'img' in file_type:
        filename = f'{args.embed_path}/{args.dataset}_{args.image_encoder}_{file_type}_embed.npz'
    elif 'text' in file_type:
        filename = f'{args.embed_path}/{args.dataset}_{args.text_encoder}_{file_type}_embed.npz'

    if not os.path.exists(filename):
        print(f'Creating {filename}')
        process_func(args, data_source)
    else:
        print(f'Loading {filename}')
    
    return np.load(filename)



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def kmeans_clustering(img_embeds, txt_embeds, args):

    n_clusters_over = args.num_pairs * getattr(args, 'over_cluster_factor', 3)
    
    if not os.path.exists(f'data/center/{args.image_encoder}-{args.text_encoder}'):
        os.makedirs(f'data/center/{args.image_encoder}-{args.text_encoder}', exist_ok=True)
        
    img_center_path = f'data/center/{args.image_encoder}-{args.text_encoder}/{args.dataset}_img_kmeans_centers_{args.num_pairs}.pkl'
    txt_center_path = f'data/center/{args.image_encoder}-{args.text_encoder}/{args.dataset}_text_kmeans_centers_{args.num_pairs}.pkl'
    
        
    if not os.path.exists(img_center_path) or not os.path.exists(txt_center_path):    

        if args.normalize_embedding:
            image_embeds_norm = normalize(img_embeds, axis=1)
            text_embeds_norm = normalize(txt_embeds, axis=1)

            # === The normalize branch also uses fused feature clustering ===
            joint_alpha = getattr(args, 'joint_alpha', 0.5)
            joint_feats = joint_alpha * image_embeds_norm + (1 - joint_alpha) * text_embeds_norm
            joint_feats = joint_feats / np.linalg.norm(joint_feats, axis=1, keepdims=True)

            # Two-stage clustering: coarse-to-fine
            n_coarse = min(n_clusters_over * 5, len(joint_feats) // 2)
            kmeans_coarse = MiniBatchKMeans(n_clusters=n_coarse, random_state=42, batch_size=10000, n_init=10)
            coarse_labels = kmeans_coarse.fit_predict(joint_feats)

            # The coarse cluster is then clustered back to num_pairs.
            coarse_centers = kmeans_coarse.cluster_centers_.astype(np.float32)
            coarse_centers_norm = coarse_centers / np.linalg.norm(coarse_centers, axis=1, keepdims=True)
            kmeans_fine = MiniBatchKMeans(n_clusters=n_clusters_over, random_state=42, batch_size=n_coarse, n_init=20)
            fine_labels_of_coarse = kmeans_fine.fit_predict(coarse_centers_norm)

            # Map the fine labels back to the original samples
            joint_labels = fine_labels_of_coarse[coarse_labels]


            img_labels = joint_labels
            txt_labels = joint_labels

            img_centers = np.zeros((args.num_pairs, img_embeds.shape[1]), dtype=np.float32)
            txt_centers = np.zeros((args.num_pairs, txt_embeds.shape[1]), dtype=np.float32)
            for k in range(args.num_pairs):
                idx = np.where(joint_labels == k)[0]
                if len(idx) > 0:
                    img_centers[k] = img_embeds[idx].mean(axis=0)
                    txt_centers[k] = txt_embeds[idx].mean(axis=0)


        else:
            img_norm = img_embeds / np.linalg.norm(img_embeds, axis=1, keepdims=True)
            txt_norm = txt_embeds / np.linalg.norm(txt_embeds, axis=1, keepdims=True)

            joint_alpha = getattr(args, 'joint_alpha', 0.5)
            joint_feats = joint_alpha * img_norm + (1 - joint_alpha) * txt_norm
            joint_feats = joint_feats / np.linalg.norm(joint_feats, axis=1, keepdims=True)

            kmeans_joint = MiniBatchKMeans(n_clusters=n_clusters_over, random_state=42, batch_size=10000, n_init=20)
            joint_labels = kmeans_joint.fit_predict(joint_feats)

            img_labels = joint_labels
            txt_labels = joint_labels

            img_centers = np.zeros((args.num_pairs, img_embeds.shape[1]), dtype=np.float32)
            txt_centers = np.zeros((args.num_pairs, txt_embeds.shape[1]), dtype=np.float32)
            for k in range(args.num_pairs):
                idx = np.where(joint_labels == k)[0]
                if len(idx) > 0:
                    img_centers[k] = img_embeds[idx].mean(axis=0)
                    txt_centers[k] = txt_embeds[idx].mean(axis=0)

        norm_img_all = img_embeds / np.linalg.norm(img_embeds, axis=1, keepdims=True)
        norm_txt_all = txt_embeds / np.linalg.norm(txt_embeds, axis=1, keepdims=True)
        pair_sims = np.sum(norm_img_all * norm_txt_all, axis=1)  # shape: (N,)
        num_img_clusters = len(np.unique(img_labels))
        num_txt_clusters = len(np.unique(txt_labels))
        weighted_cost_table = np.zeros((num_img_clusters, num_txt_clusters), dtype=np.float64)
        for idx in range(len(img_labels)):
            weighted_cost_table[img_labels[idx], txt_labels[idx]] += pair_sims[idx]
        cost_matrix = -weighted_cost_table
        img_idxs, txt_idxs = linear_sum_assignment(cost_matrix)
        
        matched_img_centers, matched_txt_centers = match_and_sort_centers(img_labels, txt_labels, img_idxs, txt_idxs, 
                                                                          img_centers, txt_centers, img_embeds, txt_embeds, 
                                                                          args) 
        
        n_candidates = len(matched_img_centers)
        if n_candidates > args.num_pairs:
            selected_idx = greedy_max_min_select(matched_img_centers, matched_txt_centers, args.num_pairs)
            matched_img_centers = matched_img_centers[selected_idx]
            matched_txt_centers = matched_txt_centers[selected_idx]
            print(f"Diversity selection: {n_candidates} -> {args.num_pairs} prototypes")

        joblib.dump(matched_img_centers, img_center_path)
        joblib.dump(matched_txt_centers, txt_center_path)
        
        print(f"Cluster centers are saved")          
            
    return joblib.load(img_center_path), joblib.load(txt_center_path)

def greedy_max_min_select(img_centers, txt_centers, num_select):
    n = len(img_centers)
    assert n >= num_select, f"候选数 {n} 少于需要选的数量 {num_select}"

    norm_img = img_centers / (np.linalg.norm(img_centers, axis=1, keepdims=True) + 1e-8)
    norm_txt = txt_centers / (np.linalg.norm(txt_centers, axis=1, keepdims=True) + 1e-8)
    joint = np.concatenate([norm_img, norm_txt], axis=1)  # shape: (n, 2*dim)

    selected = []
    mean_joint = joint.mean(axis=0, keepdims=True)
    first_idx = int(np.argmin(np.linalg.norm(joint - mean_joint, axis=1)))
    selected.append(first_idx)

    min_dists = np.linalg.norm(joint - joint[first_idx], axis=1)
    min_dists[first_idx] = -1.0

    for _ in range(num_select - 1):
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)
        new_dists = np.linalg.norm(joint - joint[next_idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)
        min_dists[next_idx] = -1.0

    return np.array(selected)

def match_and_sort_centers(img_labels, txt_labels, img_idxs, txt_idxs, img_centers, txt_centers, img_embeds, txt_embeds, args):


    matched_img_centers = []
    matched_txt_centers = []

    for i, j in zip(img_idxs, txt_idxs):
        mask = (img_labels == i) & (txt_labels == j)
        num_matched = np.sum(mask)
        print(f"Matched Image cluster {i} with Text cluster {j} -> {num_matched} samples")

        if num_matched == 0 and args.num_pairs < 300:
            matched_img_centers.append(img_centers[i])
            matched_txt_centers.append(txt_centers[j])

        else:
            masked_img = img_embeds[mask]
            masked_txt = txt_embeds[mask]
            norm_i = masked_img / np.linalg.norm(masked_img, axis=1, keepdims=True)
            norm_t = masked_txt / np.linalg.norm(masked_txt, axis=1, keepdims=True)
            weights = np.sum(norm_i * norm_t, axis=1)  # shape: (num_matched,)
            weights = np.clip(weights, 0, None)

            if len(weights) >= 4:
                local_median = np.median(weights)
                local_keep = weights >= local_median * 0.85
                if local_keep.sum() >= 2:
                    masked_img = masked_img[local_keep]
                    masked_txt = masked_txt[local_keep]
                    weights = weights[local_keep]

            weight_sum = weights.sum()
            if weight_sum > 0:
                weights = weights / weight_sum
            else:
                weights = np.ones(len(weights)) / len(weights)
            matched_img_centers.append((masked_img * weights[:, None]).sum(axis=0))
            matched_txt_centers.append((masked_txt * weights[:, None]).sum(axis=0))


    matched_img_centers = np.stack(matched_img_centers, axis=0)
    matched_txt_centers = np.stack(matched_txt_centers, axis=0)

    norm_img = matched_img_centers / np.linalg.norm(matched_img_centers, axis=1, keepdims=True)
    norm_txt = matched_txt_centers / np.linalg.norm(matched_txt_centers, axis=1, keepdims=True)
    sims_np = np.sum(norm_img * norm_txt, axis=1)
    sorted_indices = np.argsort(sims_np)[::-1]

    return matched_img_centers[sorted_indices], matched_txt_centers[sorted_indices]



def load_rep_embed(args, embed_type='text', get_origin=False):
    

    img_center_path = f'data/center/{args.image_encoder}-{args.text_encoder}/{args.dataset}_img_kmeans_centers_{args.num_pairs}.pkl'
    txt_center_path = f'data/center/{args.image_encoder}-{args.text_encoder}/{args.dataset}_text_kmeans_centers_{args.num_pairs}.pkl'

    
    if embed_type == 'text':
        if not os.path.exists(txt_center_path):
            raise FileNotFoundError(f"Text centers file not found: {txt_center_path}")
        return joblib.load(txt_center_path)
    
    elif embed_type == 'image':
        if not os.path.exists(img_center_path):
            raise FileNotFoundError(f"Image centers file not found: {img_center_path}")
        return joblib.load(img_center_path)



def remove_low_sim_pairs(img_embeds, txt_embeds, sim, remove_ratio=0.1):

    assert len(img_embeds) == len(sim)
    assert 0 <= remove_ratio < 1

    num_to_remove = int(len(sim) * remove_ratio)
    if num_to_remove == 0:
        return img_embeds, txt_embeds

    sorted_indices = np.argsort(sim)
    remove_indices = sorted_indices[:num_to_remove]

    keep_mask = torch.ones(len(sim), dtype=torch.bool)
    keep_mask[remove_indices] = False

    return img_embeds[keep_mask], txt_embeds[keep_mask]


def compute_self_sim(img_embeds, txt_embeds, args, prune=False):
    

    assert len(img_embeds) == len(txt_embeds), "List lengths must match"

    norm_img = img_embeds / np.linalg.norm(img_embeds, axis=1, keepdims=True)
    norm_txt = txt_embeds / np.linalg.norm(txt_embeds, axis=1, keepdims=True)
    sims_np = np.sum(norm_img * norm_txt, axis=1) 
    
    return sims_np 






def generate_syn_img(img_emdeds, sentence_list, img_path, args):
    if sentence_list is not None:
        assert len(img_emdeds) == len(sentence_list), "Image and text embeddings must have the same length"

    decoder_pipe = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip-small", torch_dtype=torch.float16).to(args.device)

    if sentence_list is None:
        sentence_list = [""]*len(img_emdeds)  # Default empty prompt if none provided
    
    os.makedirs(f'{img_path}', exist_ok=True)
    for idx, (img_emded, txt_promt) in enumerate(zip(img_emdeds, sentence_list)):
        save_path = f'{img_path}/{idx}.png'
        img_emded = torch.tensor(img_emded, dtype=torch.float16).to(args.device)
        
        # Image generation using Unclip
        negative_prompt= "text, watermark" 
        decoder_output = decoder_pipe(prompt=txt_promt, negative_prompt=negative_prompt, \
                                      image_embeds=img_emded.unsqueeze(0), num_inference_steps=args.infer_num_steps, \
                                      guidance_scale=args.guidance_scale, noise_level=args.noise_level)

        
        img_generated = decoder_output.images[0]
            
        # Resize and save
        img_resized = img_generated.resize((args.image_size, args.image_size), resample=Image.LANCZOS)  #Image.NEAREST, Image.BILINEAR, Image.BICUBIC
        img_resized.save(save_path)
        
