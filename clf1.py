from torch.autograd import Variable
import numpy as np
import pandas as pd
import torch

def predict(UserId,model):
    df = pd.read_csv('allrev.csv')
    sub = df[df['UserId'] == UserId]
    sub = sub.sort_values(by="time", ascending=True)
    pseq = list(sub['ProductId'].values)

    n_items = df['ProductId'].max() + 1
    score_df = pd.DataFrame(columns=['ProdId', 'Score'])

    for i in range(10000):
        pseq.append(i)

        tensor_seq = Variable(torch.LongTensor(np.array(pseq).astype('int64')))
        out = model.predict(tensor_seq)
        out = out.cpu().detach().numpy().tolist()
        pseq = pseq[:-1]

        score = out[-1]
        sub1 = pd.DataFrame({'ProdId': [i], 'Score': [score]})

        score_df = score_df.append(sub1)
    score_df = score_df.sort_values(by="Score", ascending=False)
    top_5_list = score_df.head()['ProdId'].values
    one = top_5_list[0]
    two = top_5_list[1]
    three = top_5_list[2]
    four = top_5_list[3]
    five = top_5_list[4]

    return one, two, three, four, five

