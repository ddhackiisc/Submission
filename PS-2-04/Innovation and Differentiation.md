# IISc UG - 8414 - SMILES Variational AutoEncoder - Innovation

Our Variational Autoencoder (VAE) to generate potential SARS-CoV-2 inhibitors displays several innovations.

## LSTM Architecture

Most importantly, we solve the problem of translating between a text based non linear representation of a molecule (by a SMILES string) by using a Long Short Term Memory (LSTM) Network to learn an embedding which can be used to map the molecules into a vector space. This LSTM is bidirectional and reads the SMILES string both forward and backwards, so we can make use of all the information in the string in ann efficient manner.

### Batching

In order to speed up training, we also utilize pytorch methods that allow the model to train on multiple inputs at the same time, using a custom loss function to combine results from strings of different lengths.

### Loss function
We also alter the loss function from the standard loss mentioned in the VAE paper in order to encourage the model to produce more diverse outputs. We do this by downweighting the KL loss component by a factor of 10^4.

## Data Curation
We canonicalize the input SMILES strings received using the rdkit ChemInformatics Library, which makes it easier to train the model as it does not need to learn different way of representing the same molecule.

## Quality Control and Output Visualization
After generating random samples from the model, we use rdkit to check tanimoto similarity witht the entire dataset to verify diversity. WE also generate images of the molecules for easy visualisation. 

Our code is also well documented and presented in an IPython notebook (SMILES VAE.ipynb), which makes it it easy for other researchers to build on our work.
