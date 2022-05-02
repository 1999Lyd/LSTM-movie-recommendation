import torch
import torch.nn as nn
from torch.autograd import Variable
import math

class lstm_from_scratch(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        
        #parameters for input gate layer
        self.W_i = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))
        
        #parameters for forget gate layer
        self.W_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))
        
        #parameter for cell state generation
        self.W_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))
        
        #parameter for ouput gate layer
        self.W_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))
        
        #initialize all parameters
        self.init_weights()
    
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self,
                x,
                init_states):
        
    
        seq_sz, _, _ = x.size()
        hidden_seq = []
        h_t, c_t = init_states
            
        for t in range(seq_sz):
            x_t = x[t, :, :]
            
            # input gate layer(filter)
            i_t = torch.sigmoid(x_t @ self.W_i + h_t @ self.U_i + self.b_i)
            
            # forget gate layer(filter)
            f_t = torch.sigmoid(x_t @ self.W_f + h_t @ self.U_f + self.b_f)
            
            # input new feature vector
            g_t = torch.tanh(x_t @ self.W_c + h_t @ self.U_c + self.b_c)
            
            # output gate layer(filter)
            o_t = torch.sigmoid(x_t @ self.W_o + h_t @ self.U_o + self.b_o)
            
            # cell state generation
            c_t = f_t * c_t + i_t * g_t
            
            # hidden state and output generation
            h_t = o_t * torch.tanh(c_t)
            
            hidden_seq.append(h_t.unsqueeze(0))
        
       
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


class LSTMRating(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_items, num_output,device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.lstm = lstm_from_scratch(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, num_output)
        self.hidden = self.init_hidden(device)

    def init_hidden(self,device):
        
    	# initialize both hidden layers
        return (Variable(torch.zeros(1, 1, self.hidden_dim).to(device)),
                Variable(torch.zeros(1, 1, self.hidden_dim)).to(device))

    def forward(self, sequence):
        
        # create embeddings for every item in a sequence
        embeddings = self.item_embeddings(sequence)
        output, self.hidden = self.lstm(embeddings.view(len(sequence), 1, -1),
                                        self.hidden)
        rating_scores = self.linear(output.view(len(sequence), -1))
        return rating_scores

    def predict(self, sequence):
        rating_scores = self.forward(sequence)
        return rating_scores
