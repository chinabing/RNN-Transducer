
"""
一个RNN Transducer的示例，类似一个语言翻译任务，输入为一段文本序列X，输出为另一个文本序列Y；
将Y序列中的特殊字符和元音字符去除即为X。例如：
X：Wll, Prnc, s Gn nd Lcc r nw jst fmly stts f t
Y：Well, Prince, so Genoa and Lucca are now just family estates of the
"""

import torch
import string
import numpy as np
import itertools
from collections import Counter
from tqdm import tqdm
import unidecode

NULL_INDEX = 0

encoder_dim   = 1024
predictor_dim = 1024
joiner_dim    = 1024

class Encoder(torch.nn.Module):
    def __init__(self, num_inputs):
        super(Encoder, self).__init__()
        self.embed = torch.nn.Embedding(num_inputs, encoder_dim)
        self.rnn = torch.nn.GRU(input_size=encoder_dim, hidden_size=encoder_dim, num_layers=3, batch_first=True,bidirectional=True, dropout=0.1)
        self.linear = torch.nn.Linear(encoder_dim*2, joiner_dim)

    def forward(self, x):
        out = x
        out = self.embed(out)
        out = self.rnn(out)[0]
        out = self.linear(out)
        return out
    

class Predictor(torch.nn.Module):
    def __init__(self, num_outputs):
        super(Predictor, self).__init__()
        self.embed = torch.nn.Embedding(num_outputs, predictor_dim)
        self.rnn = torch.nn.GRUCell(input_size=predictor_dim, hidden_size=predictor_dim)
        self.linear = torch.nn.Linear(predictor_dim, joiner_dim)

        self.initial_state = torch.nn.Parameter(torch.randn(predictor_dim))
        self.start_symbol = NULL_INDEX #原始论文中，使用0向量，这里采用使用null index


    def forward_one_step(self, input, previous_state):
        embedding = self.embed(input)
        state = self.rnn.forward(embedding, previous_state)
        out = self.linear(state)
        return out, state
    
    def forward(self, y):
        batch_size = y.shape[0]
        U = y.shape[1]
        outs = []
        state = torch.stack([self.initial_state] * batch_size).to(y.device)
        for u in range(U+1):
            if u == 0:
                decoder_input = torch.tensor([self.start_symbol]*batch_size, device=y.device)
            else:
                decoder_input = y[:,u-1]
            out, state = self.forward_one_step(decoder_input, state)
            outs.append(out)
        out = torch.stack(outs, dim=1)
        return out
    

class Joiner(torch.nn.Module):
    def __init__(self, num_outputs):
        super(Joiner, self).__init__()
        self.linear = torch.nn.Linear(joiner_dim, num_outputs)

    def forward(self, encoder_out, predictor_out):
        out = encoder_out + predictor_out
        out = torch.nn.functional.relu(out)
        out = self.linear(out)
        return out
    

class Transducer(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Transducer, self).__init__()
        self.encoder = Encoder(num_inputs)
        self.predictor = Predictor(num_outputs)
        self.joiner = Joiner(num_outputs)

        if torch.cuda.is_available(): self.device = "cuda:0"
        else: self.device = "cpu"
        self.to(self.device)

    def comput_forward_prob(self, joiner_out, T, U, y):
        """
        joiner_out: tensor of shape (B, T_max, U_max+1, #labels)
        T: list of input lengths
        U: list of output lengths
        y: label tensor (B, U_max+1)
        """

        B = joiner_out.shape[0]
        T_max = joiner_out.shape[1]
        U_max = joiner_out.shape[2] - 1
        log_alpha = torch.zeros(B, T_max, U_max+1, device=model.device)
        for t in range(T_max):
            for u in range(U_max+1):
                if u == 0:
                    if t == 0:
                        log_alpha[:,t,u] = 0.
                    else: # t > 0
                        log_alpha[:, t, u] = log_alpha[:, t-1, u] + joiner_out[:, t-1, 0, NULL_INDEX]

                else: # u > 0
                    if t == 0:
                        log_alpha[:, t, u] = log_alpha[:, t, u-1] + torch.gather(joiner_out[:,t, u-1], dim=1, index=y[:,u-1].view(-1,1)).reshape(-1)
                    else: # t > 0
                        log_alpha[:, t, u] = torch.logsumexp(torch.stack([
                            log_alpha[:, t-1, u] + joiner_out[:, t-1, u, NULL_INDEX],
                            log_alpha[:, t, u-1] + torch.gather(joiner_out[:,t, u-1], dim=1, index=y[:,u-1].view(-1,1) ).reshape(-1)
                        ]), dim=0)

        log_probs = []
        for b in range(B):
            log_prob = log_alpha[b, T[b]-1, U[b]] + joiner_out[b, T[b]-1, U[b], NULL_INDEX]
            log_probs.append(log_prob)
        log_probs = torch.stack(log_probs)
        return log_probs

    def compute_loss(self, x, y, T, U):
        encoder_out = self.encoder.forward(x)
        predictor_out = self.predictor.forward(y)
        joiner_out = self.joiner.forward(encoder_out.unsqueeze(2), predictor_out.unsqueeze(1)).log_softmax(3)
        loss = -self.comput_forward_prob(joiner_out, T, U, y).mean()
        return loss

    def compute_single_alignment_prob(self, encoder_out, predictor_out, T, U, z, y):
        """
        Computes the probability of one alignment, z.
        """ 
        t = 0; u = 0
        t_u_indices = []
        y_expanded = []
        for step in z:
            t_u_indices.append((t,u))
            if step == 0: 
                y_expanded.append(NULL_INDEX)
                t +=1
            if step == 1:
                y_expanded.append(y[u])
                u +=1
        t_u_indices.append((T-1,U))
        y_expanded.append(NULL_INDEX)

        t_indices = [t for (t,u) in t_u_indices]
        u_indices = [u for (t,u) in t_u_indices]
        encoder_out_expanded = encoder_out[t_indices]
        predictor_out_expanded = predictor_out[u_indices]
        joiner_out = self.joiner.forward(encoder_out_expanded, predictor_out_expanded).log_softmax(1)
        logprob = -torch.nn.functional.nll_loss(input=joiner_out, target=torch.tensor(y_expanded).long().to(self.device), reduction="sum")
        return logprob
    
#    Transducer.comput_single_alignment_prob = comput_single_alignment_prob
    def greedy_search(self, x, T):
        y_batch = []
        B = len(x)
        encoder_out = self.encoder.forward(x)
        U_max = 200
        for b in range(B):
            t = 0; u = 0; y = [self.predictor.start_symbol]; 
            predictor_state = self.predictor.initial_state.unsqueeze(0)
            while t < T[b] and u < U_max:
                predictor_input = torch.tensor([ y[-1] ], device = x.device)
                g_u, predictor_state = self.predictor.forward_one_step(predictor_input, predictor_state)
                f_t = encoder_out[b, t]
                h_t_u = self.joiner.forward(f_t, g_u)
                argmax = h_t_u.max(-1)[1].item()
                if argmax == NULL_INDEX:
                    t += 1
                else: # argmax == a label
                    u += 1
                    y.append(argmax)
            y_batch.append(y[1:]) # remove start symbol
        return y_batch

class Collate:
    def __call__(self, batch):
        """
        batch: list of tuples (input string, output string)
        Returns a minibatch of strings, encoded as labels and padded to have the same length.
        """
        x = []; y = []
        batch_size = len(batch)
        for index in range(batch_size):
            x_, y_ = batch[index]
            x.append(encode_string(x_))
            y.append(encode_string(y_))

        # pad all sequences to have same length
        T = [len(x_) for x_ in x]
        U = [len(y_) for y_ in y]
        T_max = max(T)
        U_max = max(U)

        for index in range(batch_size):
            x[index] += [NULL_INDEX] * (T_max - len(x[index]))
            x[index] = torch.tensor(x[index])
            y[index] += [NULL_INDEX] * (U_max - len(y[index]))
            y[index] = torch.tensor(y[index])

        # stack into single tensor
        x = torch.stack(x)
        y = torch.stack(y)
        T = torch.tensor(T)
        U = torch.tensor(U)

        return (x,y,T,U)
    

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, lines, batch_size):
        lines = list(filter(("\n").__ne__, lines))
        self.lines = lines
        collate = Collate()
        self.loader = torch.utils.data.DataLoader(self, batch_size=batch_size,num_workers=0,collate_fn=collate)

    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        line = self.lines[idx].replace("\n", "")
        line = unidecode.unidecode(line)                        # 去除特殊字符 
        x = "".join(c for c in line if c not in "AEIOUaeiou")   # 去除元音字符
        y = line
        return (x,y)
    
def encode_string(s):
    for c in s:
        if c not in string.printable:
            print(s)
    return [string.printable.index(c) + 1 for c in s]

def decode_labels(l):
    return "".join([string.printable[c-1]  for c in l])

class Trainer:
  def __init__(self, model, lr):
    self.model = model
    self.lr = lr
    self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
  
  def train(self, dataset, print_interval = 20):
    train_loss = 0
    num_samples = 0
    self.model.train()
    pbar = tqdm(dataset.loader)
    for idx, batch in enumerate(pbar):
      x,y,T,U = batch
      x = x.to(self.model.device); y = y.to(self.model.device)
      batch_size = len(x)
      num_samples += batch_size
      loss = self.model.compute_loss(x,y,T,U)
      self.optimizer.zero_grad()
      pbar.set_description("%.2f" % loss.item())
      loss.backward()
      self.optimizer.step()
      train_loss += loss.item() * batch_size
      if idx % print_interval == 0:
        self.model.eval()
        guesses = self.model.greedy_search(x,T)
        self.model.train()
        print("\n")
        for b in range(2):
          print("input:", decode_labels(x[b,:T[b]]))
          print("guess:", decode_labels(guesses[b]))
          print("truth:", decode_labels(y[b,:U[b]]))
          print("")
    train_loss /= num_samples
    return train_loss

  def test(self, dataset, print_interval=1):
    test_loss = 0
    num_samples = 0
    self.model.eval()
    pbar = tqdm(dataset.loader)
    with torch.no_grad():
        for idx, batch in enumerate(pbar):
          x,y,T,U = batch
          x = x.to(self.model.device); y = y.to(self.model.device)
          batch_size = len(x)
          num_samples += batch_size
          loss = self.model.compute_loss(x,y,T,U)
          pbar.set_description("%.2f" % loss.item())
          test_loss += loss.item() * batch_size
          if idx % print_interval == 0:
            print("\n")
            print("input:", decode_labels(x[0,:T[0]]))
            print("guess:", decode_labels(self.model.greedy_search(x,T)[0]))
            print("truth:", decode_labels(y[0,:U[0]]))
            print("")
    test_loss /= num_samples
    return test_loss
    

if __name__ == '__main__':


    with open("war_and_peace.txt", "r") as f:
        lines = f.readlines()

    end = round(0.9 * len(lines))
    train_lines = lines[:end]
    test_lines = lines[end:]
    train_set = TextDataset(train_lines, batch_size=8)
    test_set = TextDataset(test_lines, batch_size=8)
    train_set.__getitem__(0)
    

    num_chars = len(string.printable)
    model = Transducer(num_inputs=num_chars+1, num_outputs=num_chars+1)
    trainer = Trainer(model=model, lr=0.0003)

    num_epochs = 1
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        train_loss = trainer.train(train_set)
        test_loss = trainer.test(test_set)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print("Epoch %d: train loss = %f, test loss = %f" %(epoch, train_loss, test_loss))








        


        