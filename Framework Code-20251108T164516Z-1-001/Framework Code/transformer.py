import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *
import math  # For sqrt in scaling
import torch.nn.functional as F

# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.arange(0, input_size, dtype=torch.long)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_internal)
        self.k_proj = nn.Linear(d_model, d_internal)
        self.v_proj = nn.Linear(d_model, d_model)  # V remains d_model for output dim match
        self.ff1 = nn.Linear(d_model, d_internal)
        self.ff2 = nn.Linear(d_internal, d_model)
        self.d_k = d_internal

    def forward(self, input_vecs, mask=None):
        # Self-attention (single-head)
        Q = self.q_proj(input_vecs)  # [seq, d_internal]
        K = self.k_proj(input_vecs)  # [seq, d_internal]
        V = self.v_proj(input_vecs)  # [seq, d_model]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [seq, seq]
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)  # Causal mask for BEFORE
        attn_weights = F.softmax(scores, dim=-1)  # [seq, seq]
        attn_out = torch.matmul(attn_weights, V)  # [seq, d_model]
        # Residual 1
        after_attn = input_vecs + attn_out
        # FFN
        ff_hidden = F.relu(self.ff1(after_attn))  # [seq, d_internal]
        ff_out = self.ff2(ff_hidden)  # [seq, d_model]
        # Residual 2
        output = after_attn + ff_out
        return output, attn_weights


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, backward_only=False):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        :param backward_only: True for causal (look-back) attention in BEFORE task
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, num_positions)
        self.layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for _ in range(num_layers)])
        self.output = nn.Linear(d_model, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.backward_only = backward_only
        self.num_positions = num_positions

    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        device = indices.device
        x = self.embed(indices)  # [20, d_model]
        x = self.pos_enc(x)
        mask = None
        if self.backward_only:
            mask = torch.triu(torch.ones(self.num_positions, self.num_positions, device=device), diagonal=1).bool()
        attns = []
        for layer in self.layers:
            x, attn = layer(x, mask=mask)
            attns.append(attn)
        log_probs = self.log_softmax(self.output(x))  # [20, 3]
        return log_probs, attns


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    vocab_size = 27  # Fixed from vocab
    num_positions = 20
    d_model = 64
    d_internal = 64
    num_classes = 3
    num_layers = 1  # Experiment with 2-4 for Q2
    backward_only = (args.task == 'BEFORE')  # Causal for main task

    model = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, backward_only)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Tune if needed (e.g., 5e-5)
    loss_fcn = nn.NLLLoss()

    num_epochs = 20  # Increase if loss plateaus
    ex_idxs = list(range(len(train)))
    for t in range(num_epochs):
        random.shuffle(ex_idxs)
        loss_this_epoch = 0.0
        for ex_idx in ex_idxs:
            model.zero_grad()
            ex = train[ex_idx]
            log_probs, _ = model(ex.input_tensor)
            loss = loss_fcn(log_probs, ex.output_tensor)
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
        avg_loss = loss_this_epoch / len(train)
        if t % 5 == 0:
            print(f"Epoch {t}: avg loss {avg_loss:.4f}")
            # Optional: Quick dev eval (FIXED: Unpack [0] for log_probs)
            model.eval()
            with torch.no_grad():
                dev_correct = sum([sum(torch.argmax(model(ex.input_tensor)[0], dim=1) == ex.output_tensor) for ex in dev[:100]])
                dev_acc = dev_correct / (100 * 20)
                print(f"  Dev acc (100 exs): {dev_acc:.4f}")
            model.train()

    model.eval()
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))