## Overview

<img src='./assets/animation_for_rag2.gif'/>

This work takes embeddings (e.g., from one-shot models with Siamese networks) and generates descriptive captions.  Given an input image representing a patch of terrain (32x32), the model predicts the composition of the terrain, including percentages of different elements (e.g., "30% fucus and 70% asco") and the presence of nearby features (e.g., "with water nearby"). This uses results from my research work with one-shot models for multiband signals.

## Retrieval-Augmented Generation (RAG) and FAISS Indexing

This project utilizes a Retrieval-Augmented Generation (RAG) approach to enhance caption generation (see prompts.yaml).

### How it Works

1.  **Embedding Generation:**  The Siamese network processes the input terrain image (32x32) and produces an embedding vector. For example: -0.2879038 ,  0.00277467, -0.23789813, -0.16470747, -0.36744052,
        -0.19136755, -0.0723161 ,  0.11992071, -0.00212295, -0.23359874,
         0.2599904 ,  0.00269173,  0.17385477, -0.15069602, -0.01112346,
        -0.26239097,  0.10531765,  0.1856052 , -0.05377749, -0.11307743,
         0.11436549,  0.18217856, -0.24487908,  0.19925837,  0.23600164,
        -0.26388025,  0.0616321 ,  0.10541023, -0.08139204,  0.11356585,
         0.13498148, -0.03214637
   
2.  **FAISS Index Search:** This embedding vector is used as a query to search a pre-built FAISS index

3.  **Nearest Neighbor Retrieval:** FAISS efficiently finds the *k*-nearest neighbors (most similar images) in the index based on the query embedding.  We retrieve both the embedding vectors and the associated captions of these neighbors.

4.  **Caption Fusion/Augmentation:** The retrieved captions (from the nearest neighbors) are used to augment the caption generation process via prompting
  
5.  **Final Caption Generation:** Generate the final caption, taking into account both the input image embedding and the retrieved information

## Usage

Supply paths to FAISS index of embeddings and a CSV that maps embeddings to captions in config.toml. Naturally, if there are lots of embeddings, use Parquet instead.

```
cd src/main; streamlit run main.py
```


