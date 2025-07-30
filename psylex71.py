import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

import numpy as np
import os
import sys
import pandas as pd
from collections import OrderedDict

# from IPython import get_ipython
# isColab =  'google.colab' in str(get_ipython())

class Psylex71_Dataset(torch.utils.data.Dataset):
    '''ニューラルネットワークモデルに Psylex71 を学習させるための PyTorch 用データセットのクラス'''

    def __init__(
        self,
        inp_minlen:int = 2,   # 最短文字列長
        inp_maxlen:int = 2,   # 最長文字列長
        input_tokenizer=None, # gakushu_tokenizer,   # 入力データのトークナイザ
        output_tokenizer=None, # mora_tokenizer,     # 出力データのトークナイザ
        special_tokens:list = ['PAD', 'UNK', 'SOW', 'EOW'],                                                                                                                       device:str=device,
        display:bool=True,
        add_special_tokens:bool=True,
        excel_fname:str='Psylex71.xlsx',
        is_padding:bool=True,
        isColab:bool=False):
        
        super().__init__()

        self.excel_fname = excel_fname
        psylex71_dic = pd.read_excel(os.path.join(os.path.dirname(__file__), excel_fname)).to_dict(orient='index')
        if psylex71_dic is None:
            raise ValueError("psylex71_dic must be provided as a dictionary.")

        self.inp_minlen = inp_minlen
        self.inp_maxlen = inp_maxlen
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        self.is_padding = is_padding

        self.dic = {}
        for k,v in psylex71_dic.items():
            wrd = v['単語']
            wrd_len = len(wrd)

            # 単語長が条件範囲内であればデータとして採用
            if (inp_minlen <= wrd_len) and (wrd_len <= inp_maxlen):

                is_valid_ch = True
                for ch in wrd:
                    if not ch in input_tokenizer.tokens:
                        is_valid_ch = False
                if is_valid_ch:
                    self.dic[k] = v
                
        self.inputs = [v['単語'] for v in self.dic.values()]
        self.targets = [v['ヨミ'] for v in self.dic.values()]
        self.special_tokens = special_tokens
        self.device = device
        self.add_special_tokens = add_special_tokens
        
        out_maxlen = 0
        for k, v in self.dic.items():
            _len = len(self.output_tokenizer(v['ヨミ']))
            out_maxlen = _len if _len > out_maxlen else out_maxlen

        # ＋2 しているのは <SOW>,<EOW> という 2 つのスペシャルトークンを付加するため            
        self.out_maxlen = out_maxlen + 2
        self.inp_maxlen = inp_maxlen + 2

        if display:
            print(f'Psylex71_Dataset(): inp_minlen:{inp_minlen}, inp_maxlen:{inp_maxlen}, len(self.dic):{len(self.dic)}, out_maxlen:{self.out_maxlen}')
            # print(f'input_tokenizer.tokens: {input_tokenizer.tokens}')
            # print(f'output_tokenizer.tokens: {self.output_tokenizer.tokens}')
            print(f'special_tokens: {self.special_tokens}')
            print('')


    def __len__(self):
        return len(self.dic)

    def __getitem__(self, idx):
        inp, tgt = self.inputs[idx], self.targets[idx]

        if self.add_special_tokens:
            # 入力信号にスペシャルトークン <SOW>, <EOW> トークンを付与する場合
            #inp = [self.input_cands.index('<SOW>')]  + [self.input_cands.index(x) for x in inp]  + [self.input_cands.index('<EOW>')]
            inp = [self.input_tokenizer.tokens.index('<SOW>')] + self.input_tokenizer(inp) + [self.input_tokenizer.tokens.index('<EOW>')]
            tgt = [self.output_tokenizer.tokens.index('<SOW>')] + self.output_tokenizer(tgt) + [self.output_tokenizer.tokens.index('<EOW>')]
        else:
            # 入力信号に スペシャルトークンを付与しない場合
            #inp = [self.input_tokenizer.tokens.index(x) for x in inp]
            inp = self.input_tokenizer(inp)
            tgt = self.output_tokenizer(tgt)

        # ターゲット (教師)信号 に <SOW>, <EOW> を付与する
        #tgt = [self.target_tokecands.index('<SOW>')] + [self.target_cands.index(x) for x in tgt] + [self.target_cands.index('<EOW>')]
        #tgt = self.output_tokenizer(tgt)

        if self.is_padding:
            while len(inp) < self.inp_maxlen:
                inp = inp + [self.input_tokenizer.tokens.index('<PAD>')]

            while len(tgt) < self.out_maxlen:
                tgt = tgt + [self.output_tokenizer.tokens.index('<PAD>')]
                #tgt = tgt + [self.target_cands.index('<PAD>')]

        inp, tgt = torch.LongTensor(inp), torch.LongTensor(tgt)
        inp, tgt = inp.to(self.device), tgt.to(self.device)
        return inp, tgt

    def getitem(self, idx):
        #inp, tgt = self.inputs[idx], self.targets[idx]
        wrd = self.inputs[idx]
        phn = self.targets[idx]
        return wrd, phn

    def ids2argmax(self, ids):
        out = np.array([torch.argmax(idx).numpy() for idx in ids], dtype=np.int32)
        return out

    def ids2tgt(self, ids):
        # out = [self.target_cands[idx - len(self.special_tokens)] for idx in ids]
        out = self.output_tokenizer.decode(ids)
        return out

    def ids2inp(self, ids):
        #out = [self.input_cands[idx] for idx in ids]
        out = self.input_tokenizer.decode(ids)
        return out

    def target_ids2target(self, ids:list):
        ret = self.output_tokenizer.decode(ids)
        return ret

# for inplen_min in [2]:
#     for inplen_max in [2, 3, 4,5,6,7,8,9,10,20]:
#         psylex71_ds = Psylex71_Dataset(inplen_min=2,inplen_max=inplen_max)
#         print(f'inplen_min:{inplen_min}, inplen_max:{inplen_max} psylex71_ds.__len__():{psylex71_ds.__len__()}')
#         #print(f'psylex71_ds.__len__():{psylex71_ds.__len__()}')

#psyex71_ds = Psylex71_Dataset(dic=Psylex71, input_tokenizer=gakushu_tokenizer, output_tokenizer=mora_tokenizer, device=device)