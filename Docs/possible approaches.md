# Possible Approaches

Taken from [Gemini](https://gemini.google.com/share/b275e3e3af96)

## Unsupervised methods

- DBSCAN
- Isolation Forest
    - Gradient boosting methods to boost performance of this tree method
    -  
- some NN based on autoencoder structure
    - may not work since we only have 2 input and 1 output feature
    - bench on reconstruction error, if threshold exceeds, its anomaly
- PCA
- Local Outlier Factor (LOF)
- Boltzmann machines
- Graph neural networks (like GraphSAGE or GAT)
    - create embeddings for existing users based on item interactions
    - figure out which purchase is anomalous

AI recommends to use an ensemble training pipeline to handle the new data over time

## About autoencoders

- <https://arxiv.org/pdf/2501.13864>
    - its possible to bullshit a reconstruction loss despite being an anomaly (phenomena characterised as out-of-bounds reconstruction loss)
