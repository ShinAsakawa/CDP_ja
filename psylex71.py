import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# from IPython import get_ipython
# isColab =  'google.colab' in str(get_ipython())

import pandas as pd


class Psylex71_Dataset(torch.utils.data.Dataset):
    '''ニューラルネットワークモデルに Psylex71 を学習させるための PyTorch 用データセットのクラス'''

    def __init__(self,
                 inplen_min:int = 2,   # 最短文字列長
                 inplen_max:int = 2,   # 最長文字列長
                 psylex71_dic:dict=None,  # Psylex71_dic を仮定
                 input_tokenizer = None, # gakushu_tokenizer,   # 入力データのトークナイザ
                 output_tokenizer = None, # mora_tokenizer,     # 出力データのトークナイザ
                 special_tokens:list = ['PAD', 'UNK', 'SOW', 'EOW'],                                                                                                                       device:str=device,
                 display:bool=True):
        super().__init__()

        if psylex71_dic is None:
            psylex71_dic = pd.read_excel('CDP_ja/Psylex71.xlsx').to_dict(orient='index')

        self.dic = {}
        for k,v in psylex71_dic.items():
            wrd = v['単語']
            wrd_len = len(wrd)
            if (inplen_min <= wrd_len) and (wrd_len <= inplen_max):  # 単語長が条件範囲内であればデータとして採用
                self.dic[k] = v
                
        self.inputs = [v['単語'] for v in self.dic.values()]
        self.targets = [v['ヨミ'] for v in self.dic.values()]
        self.special_tokens = special_tokens
        self.device = device
        
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer

        maxlen_out = 0
        for k, v in self.dic.items():
            _len = len(self.output_tokenizer(v['ヨミ']))
            maxlen_out = _len if _len > maxlen_out else maxlen_out

        # ＋2 しているのは <SOW>,<EOW> という 2 つのスペシャルトークンを付加するため            
        self.maxlen_out = maxlen_out + 2

        if display:
            print(f'Psylex71_Dataset(): inplen_min:{inplen_min}, inplen_max:{inplen_max}, len(self.dic):{len(self.dic)}, maxlen_out:{self.maxlen_out}')
            print(f'input_tokenizer.tokens: {input_tokenizer.tokens}')
            print(f'output_tokenizer.tokens: {self.output_tokenizer.tokens}')
            print(f'special_tokens: {self.special_tokens}')
            print('')


    def __len__(self):
        return len(self.dic)

    def __getitem__(self, idx):
        inp, tgt = self.inputs[idx], self.targets[idx]

        # 入力信号にスペシャルトークン <SOW>, <EOW> トークンを付与する場合
        #inp = [self.input_cands.index('<SOW>')]  + [self.input_cands.index(x) for x in inp]  + [self.input_cands.index('<EOW>')]
        inp = [self.input_tokenizer.tokens.index('<SOW>')] + [self.input_tokenizer(inp)] + [self.input_tokenizer.tokens.index('<EOW>')]

        # 入力信号に スペシャルトークンを付与しない場合
        #inp = [self.input_tokenizer.index(x) for x in inp]
        #inp = self.input_tokenizer(inp)

        # ターゲット (教師)信号 に <SOW>, <EOW> を付与する
        #tgt = [self.target_tokecands.index('<SOW>')] + [self.target_cands.index(x) for x in tgt] + [self.target_cands.index('<EOW>')]
        #tgt = self.output_tokenizer(tgt)
        tgt = [self.output_tokenizer.tokens.index('<SOW>')] + [self.output_tokenizer(tgt)] + [self.output_tokenizer.tokens.index('<EOW>')]

        while len(tgt) < self.maxlen_out:
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