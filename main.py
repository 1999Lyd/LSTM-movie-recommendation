from torch.autograd import Variable
import torch
from model import LSTMRating
from torch.utils.data import DataLoader,TensorDataset
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import warnings


df = pd.read_csv('allrev.csv')

# get purchase sequence for every user
s = pd.DataFrame(columns=['UserId', 'rseq', 'pseq'])
sub1 = pd.DataFrame(columns=['UserId', 'rseq', 'pseq'])
for i in range(int(n_user)):
    sub = df[df['UserId']==i]
    n_reviews = len(sub)
    if n_reviews>=1 & n_reviews<=50:
      sub = sub.sort_values(by="time" , ascending=True)
      sub1['UserId'] = i
      sub1['rseq'] = [sub['ratings'].values]
      sub1['pseq'] = [sub['ProductId'].values]
    
    
        
      s = s.append(sub1)
    else:
        continue

s = s.iloc[1:,:]

#pad every sequence to fixed length
reviews = []
labels = []
max_pad_length = 50
for i in range(len(s['rseq'])):
    if len(s['rseq'].iloc[i]) < max_pad_length:
        reviews.append(torch.Tensor(s['pseq'].iloc[i]))
        labels.append(torch.Tensor(s['rseq'].iloc[i]))

X = pad_sequence(reviews)
X = torch.transpose(X,0,1)
y = pad_sequence(labels)
y = torch.transpose(y,0,1)

#prepare dataset for training
train_iter = [(label,text) for label,text in zip(y,X)]


# Create PyTorch Datasets from iterators
train_dataset = to_map_style_dataset(train_iter)


# Split training data to get a validation set
num_train = int(len(train_dataset) * 0.8)
split_train_dataset, split_valid_dataset = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

batch_size = 64
train_dataloader = DataLoader(split_train_dataset, batch_size=batch_size,
                              shuffle=False)
val_dataloader = DataLoader(split_valid_dataset, batch_size=batch_size,
                              shuffle=False)

train_dataloaders = {'train':train_dataloader,'val':val_dataloader}
dataset_sizes = {'train':len(split_train_dataset),'val':len(split_valid_dataset)}

# define parameters for training
embedding_dim = 64
hidden_dim = 128
n_output = 1
n_items = df['ProductId'].max()
n_epochs=10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMRating(embedding_dim, hidden_dim, n_items+1, n_output,device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, train_dataset, device, num_epochs=5, scheduler=None):
     # Send model to GPU if available
    since = time.time()

    costpaths = {'train':[],'val':[]}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Get the inputs and labels, and send to GPU if available
            for (labels,inputs) in train_dataset:
                inputs = inputs
                labels = labels.to(device)
                model.zero_grad()
                # initialize hidden layers
                model.hidden = model.init_hidden(device)
                model = model.to(device)
                # convert sequence to PyTorch variables
                sequence_var = Variable(torch.LongTensor(inputs.numpy().astype('int64'))).to(device)
                # Zero the weight gradients
                optimizer.zero_grad()

                # Forward pass to get outputs and calculate loss
                # Track gradient only for training data
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(sequence_var).view(-1)
                    loss = criterion(outputs, labels)

                    # Backpropagation to get the gradients with respect to each weight
                    # Only if in train
                    if phase == 'train':
                        loss.backward()
                        # Update the weights
                        optimizer.step()

                # Convert loss into a scalar and add it to running_loss
                running_loss += np.sqrt(loss.item()) 

            # Step along learning rate scheduler when in train
            if (phase == 'train') and (scheduler is not None):
                scheduler.step()

            # Calculate and display average loss and accuracy for the epoch
            epoch_loss = running_loss / len(train_dataset)
            costpaths[phase].append(epoch_loss)
            print('{} loss: {:.4f}'.format(phase, epoch_loss))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return costpaths

# start training
cost_paths = train_model(model,criterion,optimizer,train_dataset, device,n_epochs, scheduler=None)

model_dir = 'models/'
os.makedirs(os.path.dirname(model_dir), exist_ok=True)
filename = 'fullmodel.pt'

# Save the entire model
torch.save(model, model_dir+filename)
