
from vocabulary import Vocabulary, iVocabulary

import streamlit as st
import re
import torch
import torch.nn as nn

# Initialize the Streamlit app
st.title("Next Word Prediction")

# User input for text and model configuration
input_text = st.text_input("Enter the input text:", "Once upon a time")
max_len = st.slider("Length of predicted text", 60, 150, value=30)

# Clean input text
input_text = re.sub(r'[^a-zA-Z0-9 \.]', '', input_text).lower()

# Model parameters
embedding_dim, context_length, activation_function = 64, 10, 'relu'

# Load model checkpoint to get vocab size
checkpoint_path = f"models/model_{embedding_dim}_{context_length}_{activation_function}.pth"
checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
vocab_size = checkpoint['emb.weight'].shape[0]

# Define Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class NextWord(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, vocab_size)
        self.act = nn.ReLU() if activation_function == 'relu' else nn.Tanh()

    def forward(self, x):
        x = self.emb(x).view(x.shape[0], -1)
        x = self.act(self.lin1(x))
        return self.lin2(x)

# Instantiate the model with the correct vocab size
pred_model = NextWord(context_length, vocab_size, embedding_dim, 1024).to(device)
pred_model.load_state_dict(checkpoint)

# Generate text function remains the same
def generate_text(model, vocab, ivocab, block_size, user_input, max_len=30):
    context = [vocab.get(word, 0) for word in user_input.split()]
    context = context[-block_size:] if len(context) >= block_size else [0] * (block_size - len(context)) + context
    generated = ' '.join([ivocab.get(idx, '') for idx in context]).strip()

    for _ in range(max_len):
        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        word = ivocab[ix]
        generated += " " + word
        context = context[1:] + [ix]

    return generated

# Generate and display text on button click
if st.button("Generate"):
    predicted_text = generate_text(pred_model, Vocabulary, iVocabulary, context_length, input_text, max_len)
    st.write(predicted_text)
