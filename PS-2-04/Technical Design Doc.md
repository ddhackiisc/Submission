# IISc UG - 8414 - SMILES Variational AutoEncoder - Technical Design Document

Our goal is to take a dataset of SMILES strings representing molecules with inhibitory effect on SARS-CoV and generate new molecules that may form potential drugs against COVID-19, using a variational autoencoder (VAE).

We now describe the model specifications

The model has three parts, an LSTM based autoencoder that maps SMILES strings into a vector representation, a VAE that maps this vector space to itself via a continuous latent space, and a decoder that converts the resulting vector representations back into SMILES strings.

The architecture is as follows:

## Encoder - Bidirectional LSTM 
Maps input space (30 or so possible SMILES characters in training data) to hidden space of size 300 using a single embedding layers followed by a one-layer Gated Recurrent Unit (GRU).

## VAE
Uses two fully connected layers of width 300 to map encoded vector representations to a continuous latent space. The vector is reparameterized by defining a mean and standard deviation based on the input, and using it to scale a normally-distributed random variable. This gives us a new vector representation which is passed to the decoder after two more fully connected layers.

## Decoder
Maps the vector representation back to a SMILES srting using a 1-layer GRU with 300 hidden layer neurons followed by an embedding into the space of SMILES characters.

## Training

We use the architecture described above to train on a dataset of smiles by minimizing the weighted sum of the reconstruction loss obtained by trying to predict the input SMILES string and the analytic form for KL divergence loss as described in the VAE paper[1]

## Generation
Finally, we obtain new samples from the trained model by feeding in random numbers to the VAE and decoding the corresponding vector in the continuous latent space.

We then check the generated SMILES for chemical validity and diversity from the training set.

[1] Diederik P Kingma, Max Welling. Auto encoding Variational Bayes. https://arxiv.org/abs/1312.6114
