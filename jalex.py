import numpy as np
import os
import sys
from collections import OrderedDict
import pandas as pd
import jaconv

# Mecab を使ってヨミを得るために MeCab を import
#from ccap.mecab_settings import wakati, yomi
import jaconv
import MeCab

hostname = os.uname().nodename.split('.')[0]
if hostname == 'Leda':
    mecab_dir = '/usr/local/lib/mecab/dic/mecab-ipadic-neologd'
else:
    mecab_dir = '/opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd'

wakati = MeCab.Tagger(f'-Owakati -d {mecab_dir}').parse
#wakati = MeCab.Tagger('-Owakati -d /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd/').parse
yomi = MeCab.Tagger(f'-Oyomi -d {mecab_dir}').parse
#yomi = MeCab.Tagger('-Oyomi -d /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd').parse
HOME = os.environ['HOME']

import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

class jalex_Dataset(torch.utils.data.Dataset):
    '''ニューラルネットワークモデルに jalex を学習させるための PyTorch 用データセットのクラス'''

    def __init__(
        self,
        inp_minlen:int = 2,   # 最短文字列長
        inp_maxlen:int = 2,   # 最長文字列長
        input_tokenizer=None, # gakushu_tokenizer,   # 入力データのトークナイザ
        output_tokenizer=None, # mora_tokenizer,     # 出力データのトークナイザ
        special_tokens = ['<PAD>', '<EOW>', '<SOW>', '<UNK>', '<CLS>'], 
        # それぞれ, 埋草, 語頭, 語尾, 未定義, トークン。最後の '<CLS>' は、BERT などのモデルで使用されることを想定している。
        add_special_tokens:bool=True,
        jalex_fname='JALEX_utf8.csv',
        is_padding:bool=True,
        isColab:bool=False,
        display:bool=True,
        device=device):
        
        super().__init__()

        #self.jalex_fname = jalex_fname
        _jalex_fname = os.path.join(os.path.dirname(__file__), jalex_fname)
        #_jalex_fname = os.path.join(HOME, 'study/2025notebooks/CDP_Ja', jalex_fname)

        # print(f'_jalex_fname:{_jalex_fname}')
        df = pd.read_csv(_jalex_fname)
        word_list = df['Target'].to_list()

        _dic = {}
        for word in word_list:
            _yomi = yomi(word).strip()
            _wakati = wakati(_yomi).strip()
            _inp_ids = input_tokenizer(_yomi)
            _out_ids = output_tokenizer(_yomi)
            i = len(_dic)
            _dic[i] = {
                '単語': word, 'ヨミ':_yomi, 'inp_ids':_inp_ids, 'out_ids':_out_ids}
            _d = df.iloc[i].to_dict()
            for k in list(_d.keys())[:]:
                _dic[i][k] = _d[k]
                
        self.inp_minlen = inp_minlen
        self.inp_maxlen = inp_maxlen
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        self.is_padding = is_padding

        self.dic = {}
        for k,v in _dic.items():
            wrd = _dic[k]['単語']
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
            print(f'jalex_Dataset(): inp_minlen:{self.inp_minlen}, inp_maxlen:{self.inp_maxlen}, len(self.dic):{len(self.dic)}, out_maxlen:{self.out_maxlen}')


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

# jalex_dss = []
# for inp_minlen in [2]:
#     for inp_maxlen in [2,3,5,8,13]:
#         _ds = jalex_Dataset(inp_minlen=inp_minlen, inp_maxlen=inp_maxlen,
#                             input_tokenizer=input_tokenizer, output_tokenizer=output_tokenizer)
#         jalex_dss.append(_ds)
