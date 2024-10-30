import streamlit as st
import re
from vocabulary import Vocabulary, iVocabulary
import torch
import torch.nn as nn

# Initialize the Streamlit app
st.title("Next Word Prediction Application")

# User input
input_text = st.text_input("Enter the input text:", "There was a king")
context_length = st.selectbox("Context length", [5, 10, 15])
embedding_dim = st.selectbox("Embedding dimension", [64, 128])
activation_function = st.selectbox("Activation function", ["relu", "tanh"])
max_len = st.number_input("Maximum length of predicted text", min_value=0, max_value=1000, value=30)
input_text = re.sub(r'[^a-zA-Z0-9 \.]', '', input_text).lower()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if activation_function == 'relu':
    class NextWord(nn.Module):
        def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim)
            self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
            self.lin2 = nn.Linear(hidden_size, vocab_size)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.emb(x).view(x.shape[0], -1)
            x = self.relu(self.lin1(x))
            return self.lin2(x)

    pred_model = NextWord(context_length, len(Vocabulary), embedding_dim, 1024).to(device)
    pred_model.load_state_dict(torch.load(f"models/model_{embedding_dim}_{context_length}_relu.pth", map_location=device))

if activation_function == 'tanh':
    class NextWord(nn.Module):
        def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim)
            self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
            self.lin2 = nn.Linear(hidden_size, vocab_size)
            self.tanh = nn.Tanh()

        def forward(self, x):
            x = self.emb(x).view(x.shape[0], -1)
            x = self.tanh(self.lin1(x))
            return self.lin2(x)

    pred_model = NextWord(context_length, len(Vocabulary), embedding_dim, 1024).to(device)
    pred_model.load_state_dict(torch.load(f"models/model_{embedding_dim}_{context_length}_tanh.pth", map_location=device))

def generate_para(model, Vocabulary, iVocabulary, block_size, user_input=None, max_len=30):
    if user_input:
        context = [Vocabulary.get(word, 0) for word in user_input.split()]
        context = context[-block_size:] if len(context) >= block_size else [0] * (block_size - len(context)) + context
    else:
        context = [0] * block_size

    new_para = ' '.join([iVocabulary.get(idx, '') for idx in context]).strip()
    for i in range(max_len):
        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        word = iVocabulary[ix]
        new_para = new_para + " " + word
        context = context[1:] + [ix]

    return new_para

if st.button("Generate Prediction"):
    predicted_text = generate_para(pred_model, Vocabulary, iVocabulary, context_length, input_text, max_len)
    st.write("Predicted Text:", predicted_text)
