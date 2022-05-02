from torch import nn
import torch

class MFRecommender(nn.Module):
    
    def __init__(self,n_users, n_items, embedding_dim ,rating_range):
        super().__init__() 
        self.user_embeddings = nn.Embedding(num_embeddings=n_users,embedding_dim=embedding_dim) # user embeddings
        self.user_bias = nn.Embedding(num_embeddings=n_users,embedding_dim=1) # user bias
        self.item_embeddings = nn.Embedding(num_embeddings=n_items,embedding_dim=embedding_dim) # item embeddings
        self.item_bias = nn.Embedding(num_embeddings=n_items,embedding_dim=1) # item bias
        self.rating_range = rating_range # range of expected ratings e.g. 0-5

    def forward(self, X):
        embedded_users = self.user_embeddings(X[:,0]) # dims = [batch_size, embedding_dim]
        embedded_items = self.item_embeddings(X[:,1]) # dims = [batch_size, embedding_dim]
        # Take dot product of each user embedding with the embedding of item to be rated to get the predicted rating
        preds = torch.sum(embedded_users * embedded_items, dim=1, keepdim=True) 
        # Add user and item bias to rating
        preds = preds.view(-1,1) + self.user_bias(X[:,0]) + self.item_bias(X[:,1])
        # Scale predicted ratings to target-range [low,high]
        preds = torch.sigmoid(preds) * (self.rating_range[1]-self.rating_range[0]) + self.rating_range[0]
        return preds
