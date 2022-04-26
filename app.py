import streamlit as st
from clf1 import predict
import torch
import torch.nn as nn
from model import LSTMRating
# streamlit run app.py
import __main__

__main__.LSTMRating = LSTMRating

class LSTMRating(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_items, num_output,device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, num_output)
        self.hidden = self.init_hidden(device)

    def init_hidden(self,device):
    	# initialize both hidden layers
        return (Variable(torch.zeros(1, 1, self.hidden_dim).to(device)),
                Variable(torch.zeros(1, 1, self.hidden_dim)).to(device))

    def forward(self, sequence):
        embeddings = self.item_embeddings(sequence)
        output, self.hidden = self.lstm(embeddings.view(len(sequence), 1, -1),
                                        self.hidden)
        rating_scores = self.linear(output.view(len(sequence), -1))
        return rating_scores

    def predict(self, sequence):
        rating_scores = self.forward(sequence)
        return rating_scores


st.set_option("deprecation.showfileUploaderEncoding", False)

st.title("LSTM recommendation system")
st.write("")
model = torch.load('fullmodel.pt',map_location = torch.device("cpu"))
UserId = st.text_input("please input your User Id",value = None)

if UserId is not None:

    st.write("")
    st.write("Just a second...")
    recommendation = predict(UserId,model)


    st.write("top_5_recommendation:", recommendation)
