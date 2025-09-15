import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

import numpy as np
import os
import sys
import pandas as pd
from collections import OrderedDict

import requests
import jaconv
import datetime
from copy import deepcopy

from .tokenizers import gakushu_Tokenizer
from .tokenizers import mora_Tokenizer
gakushu_tokenizer = gakushu_Tokenizer()
mora_tokenizer = mora_Tokenizer()

"""
日本語を読むための語彙データベース（VDRJ） Ver. 1.1　（＝日本語を読むための”ＴＭ語彙リスト”（総合版）　Ver.4.0）
"""
class vdrj_Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        inp_minlen:int = 2,   # 最短文字列長
        inp_maxlen:int = 2,   # 最長文字列長
        input_tokenizer=gakushu_tokenizer,   # 入力データのトークナイザ
        output_tokenizer=mora_tokenizer,     # 出力データのトークナイザ
        special_tokens = ['<PAD>', '<EOW>', '<SOW>', '<UNK>', '<CLS>'], # そぞれ, 埋草, 語頭, 語尾, 未定義, トークン。最後の '<CLS>' は、BERT などのモデルで使用されることを想定している。
        device:str=device,
        display:bool=True,
        add_special_tokens:bool=True,
        is_padding:bool=True):

        self.inp_minlen = inp_minlen
        self.inp_maxlen = inp_maxlen
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        self.is_padding = is_padding
        self.display=display
        self.add_special_tokens=add_special_tokens
        self.devcie=device


        vdrj_url='http://www17408ui.sakura.ne.jp/tatsum/database/VDRJ_Ver1_1_Research_Top60894.xlsx'
        excel_fname = vdrj_url.split('/')[-1]  # 直上行の url からエクセルファイル名を切り出す

        # もしエクセルファイルが存在しなかったら ダウンロードする
        excel_fname = os.path.join(os.path.dirname(__file__), excel_fname)    
        if not os.path.exists(excel_fname):
            print(f'エクセルファイルのダウンロード {datetime.datetime.now()}...')
            r = requests.get(vdrj_url)
            with open(excel_fname, 'wb') as f:
                total_length = int(r.headers.get('content-length'))
                print('Downloading {0} - {1} bytes'.format(excel_fname, (total_length)))
                f.write(r.content)
            print(f'done {datetime.datetime.now()}')

        # 実際のエクセルファイルの読み込み
        sheet_name='重要度順語彙リスト60894語'  # シート名を指定
        print(f'エクセルファイルの読み込み {datetime.datetime.now()}...')
        df = pd.read_excel(excel_fname, sheet_name=sheet_name)
        print(f'done. {datetime.datetime.now()}')

        df = df[[
            '見出し語彙素\nLexeme',
            '標準的（新聞）表記\nStandard (Newspaper) Orthography',
            '標準的読み方（カタカナ）\nStandard Reading (Katakana)',
            '品詞\nPart of Speech',
            '(Fw)累積テキストカバー率（想定既知語彙分を含む）\nFw Cumulative Text Coverage including Assumed Known Words', 
            'ID']] #.dropna()
            # .dropna() により NaN を含む行を削除
        #print(f'df.__len__():{df.__len__()}')

        _df = df.rename(columns={
            '見出し語彙素\nLexeme': 'Lexeme', 
            '標準的（新聞）表記\nStandard (Newspaper) Orthography': 'Orth',
            '標準的読み方（カタカナ）\nStandard Reading (Katakana)': 'Kata',
            '品詞\nPart of Speech':'POS',
            '(Fw)累積テキストカバー率（想定既知語彙分を含む）\nFw Cumulative Text Coverage including Assumed Known Words':'CumCR'})
        vdrj_words = _df['Lexeme'].to_list()

        vdrj_words, vdrj_nonwords = [], []
        max_wordlen, max_katalen = 0, 0

        for idx, row in _df.iterrows():
            wrd = row['Orth']
            kata = row['Kata']

            if isinstance(wrd, float) or isinstance(kata, int) or isinstance(kata, float):
                vdrj_nonwords.append((idx,dict(row)))
            else:

                if '＊' == wrd:
                    row['Orth'] = row['Lexeme']
        
                kata_dups, word_dups = False, False
                if '／' in kata:
                    kata1, kata2 = kata.split('／')
                    kata_dups = True

                if '／' in wrd:
                    word1, word2 = wrd.split('／')
                    word_dups = True

                if kata_dups or word_dups:
                    row1 = deepcopy(row)
                    row2 = deepcopy(row)

                    if kata_dups:
                        row1['Kata'] = kata1
                        row2['Kata'] = kata2
                        for _k in [kata1, kata2]:
                            katalen = len(_k)
                            max_katalen = katalen if katalen > max_katalen else max_katalen
                    if word_dups:
                        row1['Orth'] = word1
                        row2['Orth'] = word2
                        for _w in [word1, word2]:
                            wordlen = len(_w)
                            max_wordlen = wordlen if wordlen > max_wordlen else max_wordlen

                    vdrj_words.append((idx,row1))
                    vdrj_words.append((idx,row2))
                    #print(idx, dict(row), dict(row1), dict(row2))
                else:
                    wordlen = len(wrd)
                    katalen = len(kata)
                    max_wordlen = wordlen if wordlen > max_wordlen else max_wordlen
                    max_katalen = katalen if katalen > max_katalen else max_katalen
                    vdrj_words.append((idx,dict(row)))
        
            #print(f'len(vdrj_words):{len(vdrj_words)}', f'len(vdrj_nonwords):{len(vdrj_nonwords)}')
            #print(f'max_wordlen:{max_wordlen}', f'max_katalen:{max_katalen}')    

        valid_words = []
        invalid_words = []

        dic = {}
        for idx, X in enumerate(vdrj_words):
            idx, x = X
            wrd = x['Orth']
            wrd_len = len(wrd)

            # 単語長が条件範囲内であればデータとして採用
            if (inp_minlen <= wrd_len) and (wrd_len <= inp_maxlen):

                is_valid = (np.array([c in input_tokenizer.tokens for c in wrd]) * 1).sum() == len(wrd)
                if is_valid:
                    dic[idx] = x

        #self.valid_words = valid_words
        self.dic = dic

    def __len__(self):
        return len(self.dic)
    
    def __getitem__(self, idx):
        X = self.dic[idx]
        _, x = X
        inp = x['Orth']
        tgt = x['Kata']
    
        if self.add_special_tokens:  # 入力信号にスペシャルトークン <SOW>, <EOW> トークンを付与する場合
            #inp = [self.input_cands.index('<SOW>')]  + [self.input_cands.index(x) for x in inp]  + [self.input_cands.index('<EOW>')]
            inp = [self.input_tokenizer.tokens.index('<SOW>')] + self.input_tokenizer(inp) + [self.input_tokenizer.tokens.index('<EOW>')]
            tgt = [self.output_tokenizer.tokens.index('<SOW>')] + self.output_tokenizer(tgt) + [self.output_tokenizer.tokens.index('<EOW>')]
        else:  # 入力信号に スペシャルトークンを付与しない場合
            inp = self.input_tokenizer(inp)
            tgt = self.output_tokenizer(tgt)

        if self.is_padding:
            while len(inp) < self.inp_maxlen:
                inp = inp + [self.input_tokenizer.tokens.index('<PAD>')]

            while len(tgt) < self.out_maxlen:
                tgt = tgt + [self.output_tokenizer.tokens.index('<PAD>')]
 
        inp, tgt = torch.LongTensor(inp), torch.LongTensor(tgt)
        inp, tgt = inp.to(self.device), tgt.to(self.device)
        return inp, tgt

    def getitem(self, idx):
        #inp, tgt = self.inputs[idx], self.targets[idx]
        _, X = self.dic[idx]
        wrd = X['orth']
        phn = X['Kata']
        return wrd, phn
