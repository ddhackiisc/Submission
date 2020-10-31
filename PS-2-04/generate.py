import sys 
  
num_smiles = eval(sys.argv[1]) #this is a gaping security flaw btw, but this is python code so they could execute anything they wanted anyway
print("Attempting to generate", num_smiles, "SMILES strings")

#!/usr/bin/env python
# coding: utf-8

#This file can be used to generate molecules directly without running the training notebook

import pandas as pd
import numpy as np
import random

from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, Descriptors, AllChem

def canonicalize(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

def validSMILES(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return True

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence#, masked_cross_entropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

traindata = pd.read_csv('cleancovid19dataset.csv')
traindata['cleanSMILES'] = traindata['SMILES'].apply(canonicalize)
data = list(zip(traindata['cleanSMILES'],traindata['cleanSMILES']))

PAD_token = 0
SOS_token = 1
EOS_token = 2

class Lang:
    def __init__(self, name):
        self.name = name
        self.char2index = {}
        self.char2count = {}
        self.index2char = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_chars = 3  # Count SOS and EOS and PAD

    def addSMILES(self, smiles):
        for char in smiles:
            self.addChar(char)

    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1

MIN_LENGTH = 3
MAX_LENGTH = 63

def filter_pairs(pairs):
    filtered_pairs = []
    for pair in pairs:
        if len(pair[0]) >= MIN_LENGTH and len(pair[0]) <= MAX_LENGTH\
        and len(pair[1]) >= MIN_LENGTH and len(pair[1]) <= MAX_LENGTH:
                filtered_pairs.append(pair)
    return filtered_pairs

def prepareData(lang1, lang2, data):
    """takes data as list of pairs for now
    returns the dictionaries for input and output languages, and the pairs"""
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    pairs = data
    print("Read %s smiles pairs" % len(pairs))
    
    pairs = filter_pairs(pairs)
    print("Filtered to %d pairs" % len(pairs))
    
    print("Counting chars...")
    for pair in pairs:
        input_lang.addSMILES(pair[0])
        output_lang.addSMILES(pair[1])
    print("Counted chars:")
    print(input_lang.name, input_lang.n_chars)
    print(output_lang.name, output_lang.n_chars)
    return input_lang, output_lang, pairs

#input_lang, output_lang, pairs = prepareData("Random SMILES","Canonical SMILES", data)
input_lang, output_lang, pairs = prepareData("Canonical SMILES","Canonical SMILES", data)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=True)
        
    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
        return outputs, hidden

class VAE(nn.Module):
    def __init__(self,hidden_size, lat_int_size = 60, lat_size = 30):
        super(VAE, self).__init__()
        self.hidden_size = hidden_size
        self.lat_int_size = lat_int_size
        self.lat_size = lat_size
        self.fc1 = nn.Linear(hidden_size, lat_int_size)
        self.fc21 = nn.Linear(lat_int_size, lat_size)
        self.fc22 = nn.Linear(lat_int_size, lat_size)
        self.fc3 = nn.Linear(lat_size, lat_int_size)
        self.fc4 = nn.Linear(lat_int_size, hidden_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        #mu, logvar = self.encode(x.view(-1, hidden_size))
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, last_hidden):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = embedded.view(1, batch_size, self.hidden_size) # S=1 x B x N

        # Get current hidden state from input word and last hidden state
        output, hidden = self.gru(embedded, last_hidden)
        output = self.out(output)

        # Return final output, hidden state
        return output, hidden



def indexesFromSmiles(lang, smiles):
    return [lang.char2index[char] for char in smiles] + [EOS_token]

def tensorFromSmiles(lang, smiles):
    indexes = indexesFromSmiles(lang, smiles)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

def random_batch(batch_size, whichpairs):
    input_seqs = []
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(whichpairs)
        input_seqs.append(indexesFromSmiles(input_lang, pair[0]))
        target_seqs.append(indexesFromSmiles(output_lang, pair[1]))

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs) 
    """the * unzips"""
    
    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1).to(device)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1).to(device)
        
    return input_var, input_lengths, target_var, target_lengths

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long() #note arange is [) unlike torch.range which is []
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):
    length = Variable(torch.LongTensor(length))

    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss



def evaluate(encoder,vae,decoder,input_seq, max_length=MAX_LENGTH):
    input_lengths = [len(input_seq)]
    input_seqs = [indexesFromSmiles(input_lang, input_seq)]
    input_batches = Variable(torch.LongTensor(input_seqs)).transpose(0, 1).to(device)
        
    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)
    
    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    
    #run through vae
    new_hidden, mu, logvar = vae(encoder_hidden[:decoder.n_layers])

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([SOS_token])).to(device) # SOS
    decoder_hidden = new_hidden
    # Use last (forward) hidden state from encoder

    # Store output words and attention states
    decoded_chars = []
    
    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_hidden= decoder(decoder_input, decoder_hidden)

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0].item()
        #print("ni",ni)
        if ni == EOS_token:
            #decoded_chars.append('<EOS>') there's no point adding it, just screws up rdkit
            break
        else:
            decoded_chars.append(output_lang.index2char[ni])
            
        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([ni])).to(device)
        

    # Set back to training mode
    encoder.train(True)
    decoder.train(True)
    
    return ''.join(decoded_chars)


# In[405]:


def evaluateRandomly(encoder,vae, decoder, n=5):
    for i in range(n):
        pair = random.choice(pairs)
        print('original smiles', pair[0])
        print('actual canonical SMILES', pair[1])
        output_smiles = evaluate(encoder,vae, decoder, pair[0])
        print('predicted canonical SMILES', output_smiles)
        print('')

def quality(encoder,vae,decoder, n=20):
    valids = 0
    for i in range(n):
        pair = random.choice(pairs)
        output_smiles = evaluate(encoder,vae, decoder, pair[0])
        if validSMILES(output_smiles):
            valids +=1
            print('predicted canonical SMILES', output_smiles, 'n: ',i)
    print("Fraction of valid SMILES is: ",valids/n)
        


def train(input_batches, input_lengths, target_batches, target_lengths,          encoder,vae, decoder, encoder_optimizer,vae_optimizer, decoder_optimizer,          max_length=MAX_LENGTH):
    
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    vae_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    
    # run hidden vector through vae, also get the mu and logvar
    new_hidden, mu, logvar = vae(encoder_hidden[:decoder.n_layers])
    
    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size)).to(device)
    decoder_hidden = new_hidden
    # Use last (forward) hidden state from encoder

    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size)).to(device)

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t] # Next input is current target (this is teacher forcing?)

    # Loss calculation and backpropagation
    recon_loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
        target_batches.transpose(0, 1).contiguous(), # -> batch x seq
        target_lengths
    )
    
    #apparently this is the analytical solution for KL divergence of gaussians
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    loss = recon_loss + 0.00005*kl_loss #had to reduce the weight give to the KL #divded by 2
    #to prevent it from making every single molecule the same
    loss.backward()
    
    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    vc = torch.nn.utils.clip_grad_norm_(vae.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    
    
    # Update parameters with optimizers
    encoder_optimizer.step()
    vae_optimizer.step()
    decoder_optimizer.step()
    #print("recon loss: ", recon_loss, "and kl loss:", kl_loss)
    return loss.item()



# Configure models
hidden_size = 300
lat_int_size = 300
lat_size = 300
n_layers = 1

encoder = EncoderRNN(input_lang.n_chars, hidden_size, n_layers=1).to(device)
vae = VAE(hidden_size, lat_int_size, lat_size).to(device)
decoder = DecoderRNN(hidden_size, output_lang.n_chars, n_layers=1).to(device)

from pathlib import Path
savedencoderfile = "SMILESencoder300.pt"
savedvaefile = "SMILESvae300.pt"
saveddecoderfile = "SMILESdecoder300.pt"
if Path(savedencoderfile).is_file():
    # file exists
    encoder.load_state_dict(torch.load(savedencoderfile))
    #model.eval() wtf is this
    print("=== Encoder was loaded from " + savedencoderfile)
    
if Path(savedvaefile).is_file():
    # file exists
    vae.load_state_dict(torch.load(savedvaefile))
    #model.eval() wtf is this
    print("=== Encoder was loaded from " + savedvaefile)
    
if Path(saveddecoderfile).is_file():
    # file exists
    decoder.load_state_dict(torch.load(saveddecoderfile))
    #model.eval() wtf is this
    print("=== Decoder was loaded from " + saveddecoderfile)

def randomSample(vae,decoder):
    #get random hidden vector from the p(x/z) distribution
    new_hidden = vae.decode(torch.randn(1,1,vae.lat_size).to(device))
    
    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([SOS_token])).to(device) # SOS
    decoder_hidden = new_hidden
    # Use last (forward) hidden state from encoder

    # Store output words and attention states
    decoded_chars = []
    
    # Run through decoder
    for di in range(MAX_LENGTH):
        decoder_output, decoder_hidden= decoder(decoder_input, decoder_hidden)

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0].item()
        #print("ni",ni)
        if ni == EOS_token:
            #decoded_chars.append('<EOS>') there's no point adding it, just screws up rdkit
            break
        else:
            decoded_chars.append(output_lang.index2char[ni])
            
        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([ni])).to(device)
    
    return ''.join(decoded_chars)

def samplepics(encoder,vae,decoder,n=10):
    sms = []
    ms = []
    for i in range(n):
        output_smiles = randomSample(vae,decoder)
        if validSMILES(output_smiles):
            sms.append(output_smiles)
            ms.append(Chem.MolFromSmiles(output_smiles))
            #print('generated canonical SMILES', output_smiles, 'n: ',i)
    return ms, sms



ms, sms = samplepics(encoder,vae,decoder,num_smiles)

def diversemol(smiles, dataset, level):
    mol = Chem.MolFromSmiles(smiles)
    fpmol = Chem.RDKFingerprint(mol)
    for datasmiles in dataset:
        datamol = Chem.MolFromSmiles(datasmiles)
        fpdatamol = Chem.RDKFingerprint(datamol)
        if DataStructs.TanimotoSimilarity(fpmol, fpdatamol) >= level:
            return False
    return True

divsmiles = [m for m in set(sms) if diversemol(m,traindata['cleanSMILES'], 1)]
divms = [Chem.MolFromSmiles(m) for m in divsmiles]
print("\n\nHere are the SMILES strings generated:")
for s in divsmiles:
    print(s,"\n")

Draw.MolsToGridImage(divms,molsPerRow=4,subImgSize=(200,200),legends=[Chem.MolToSmiles(x) for x in ms])


# # References

# https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb

# Auto-Encoding Variational Bayes
# Diederik P Kingma, Max Welling https://arxiv.org/abs/1312.6114

# Novel Inhibitors of Severe Acute Respiratory Syndrome Coronavirus Entry That Act by Three Distinct Mechanisms https://dx.doi.org/10.1128%2FJVI.00998-13