# How to run the model to generate samples

## requirements
You need to have rdkit available as well as standard ML and pytorch libraries

## generation
In order to generate (say) 10 molecules using the VAE:

run the following command
$ python generate_samples.py 10

in the folder containing encoder300.pt, vae300.pt and decoder300.pt and cleancovid19dataset.csv .

Note that sometimes less strings will be generated than the command-line input, because the script deletes those strings which do not form valid molecules or are already present in the training data

