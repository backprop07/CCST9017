import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
model="res" # "res" or "plain"
class block_change(nn.Module):
    def __init__(self,in_features,out_features):
        super(block_change, self).__init__()
        self.fc1=nn.Linear(in_features, out_features)
        self.bn1=nn.BatchNorm1d(out_features)
        self.relu1=nn.ReLU()
        self.dropout1=nn.Dropout(0.1)
        self.fc2=nn.Linear(out_features, out_features)
        self.bn2=nn.BatchNorm1d(out_features)
        self.dropout2=nn.Dropout(0.1)
        self.shortcut=nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features),nn.Dropout(drop_out))
        self.relu2=nn.ReLU()
        
    def forward(self, x):
        temp=x.clone()
        x=self.fc1(x)
        x=self.bn1(x)
        x=self.relu1(x)
        x=self.dropout1(x)
        x=self.fc2(x)
        x=self.bn2(x)
        x=self.dropout2(x)
        x=x+self.shortcut(temp)
        x=self.relu2(x)
        return x
if model == "plain":
    network = torch.load(r"plain_23-12-01.pth")["model"]
    network.load_state_dict(torch.load(r'plain_23-12-01.pth')["weight"])
else:
    network = torch.load(r"res_small_23-12-02.pth")["model"]
    network.load_state_dict(torch.load(r'res_small_23-12-02.pth')["weight"])
from itertools import combinations
network.eval()
list_tar=[(1,2),(1,3),(1,4),(1,5),(1,6),(2,3),(2,4),(2,5),(2,6),(3,4),(3,5),(3,6),(4,5),(4,6),(5,6)]
enu=enumerate(list_tar)
encode_dict={tar:idx for idx,tar in enu}
encode_dict[-1]=-1
def encode(inp):
    return encode_dict[inp]
def encode_list(inp_list):
    return [encode_dict[tar] for tar in inp_list]
enu=enumerate(list_tar)
decode_dict={idx:tar for idx,tar in enu}
decode_dict[-1]=-1
def decode(inp):
    return decode_dict[inp]
def decode_list(inp):
    return [decode_dict[tar] for tar in inp]
def form_tri(lst: list,inp: set):
    line_list=[x for x in lst+[inp] if x != -1]
    combs = list(combinations(line_list, 3))
    for comb in combs:
        (a,b),(c,d),(e,f)=comb
        if len(set([a,b,c,d,e,f]))==3:
            return True
    return False
def game_over(inp: list):
    if form_tri(decode_list(inp[::2][:-1]),decode(inp[::2][-1])) or form_tri(decode_list(inp[1::2][:-1]),decode(inp[1::2][-1])):
        return True
def one_hot_list_inp(inp_list):
    temp=[[0]*15 for _ in range(15)]
    for j in range(15):
        if inp_list[j]!=-1:
            temp[j][int(inp_list[j])]=1
    return temp
def make_move(inp: list): # inptut should be encoded.
    softmax = nn.Softmax(dim=1)
    network.eval()
    inp1=one_hot_list_inp(inp)
    out=softmax(network(torch.tensor([inp1],dtype=torch.float32).to('cuda'))).tolist()[0]
    descending_indices = sorted(list(range(len(out))), key=lambda i: out[i],reverse=True)
    for idx in descending_indices:
        if (idx in inp) or form_tri(decode_list(inp[1::2]),decode(idx)):
            continue
        else:
            return decode(idx)
    for idx in descending_indices:
        if (idx in inp):
            continue
        else:
            return decode(idx)
    return -1

def make_move_no_assistance(inp: list):
    softmax = nn.Softmax(dim=1)
    network.eval()
    out=softmax(network(torch.tensor([inp],dtype=torch.float32).to(device))).tolist()[0]
    descending_indices = sorted(list(range(len(out))), key=lambda i: out[i],reverse=True)
    for idx in descending_indices:
        if (idx in inp):
            continue
        else:
            return decode(idx)
from random import choice
def find_pos(inp_list):
    for i in range(2,7):
        lst=inp_list[:i+1]
        combs = list(combinations(lst, 3))
        combs=sorted(combs)
        for comb in combs:
            (a,b),(c,d),(e,f)=comb
            if len(set([a,b,c,d,e,f]))==3:
                return i
    return 7
# 0 stands for the the player one win, 1 stands for the player two win
def determine_winner(inp_list):
    pos_1=find_pos(inp_list[::2])
    pos_2=find_pos(inp_list[1::2])
    if pos_1<=pos_2:
        return 1
    else:
        return 0
game=[-1 for _ in range(15)]
for i in range(2):
    x=input("Input starting point, starting point must be smaller than ending point:")
    y=input("Input ending point:")
    game[2*i]=encode((int(x),int(y)))
    out = make_move(game)
    print(out)
    game[2*i+1]=encode(out)
for i in range(2,8):
    x=input("Input starting point, starting point must be smaller than ending point:")
    y=input("Input ending point:")
    game[2*i]=encode((int(x),int(y)))
    out = make_move(game)
    print(out)
    game[2*i+1]=encode(out)
    if game_over(game):
        break
print("Game over!")
game = [x for x in game if x != -1]
game = decode_list(game)
if determine_winner(game) == 0:
    print("Player 1 wins!")
else:
    print("Player 2 wins!")
input()