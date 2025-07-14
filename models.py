import torch
# 全モデル共通使用するライブラリの輸入
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class vanilla_TLA(torch.nn.Module):
    def __init__(self,
                 inp_vocab_size:int=None, # len(psylex71_ds.input_tokenizer.tokens),
                 out_vocab_size:int=None, # len(psylex71_ds.output_tokenizer.tokens),
                 inp_len:int=None, # maxlen_inp,
                 out_len:int=None, # psylex71_ds.maxlen_out,
                 n_hid:int=1024,
                 device:str='cpu'):
        
        super().__init__()
        self.inp_vocab_size=inp_vocab_size
        self.inp_len=inp_len
        self.out_vocab_size=out_vocab_size
        self.out_len=out_len
        self.n_hid=n_hid

        self.emb_layer = torch.nn.Linear(in_features=inp_vocab_size * inp_len, out_features=n_hid).to(device)
        #self.sigmoid = torch.nn.Sigmoid()
        #self.tanh = torch.nn.Tanh()
        #self.relu = torch.nn.ReLU()
        self.emb_outf = torch.nn.Tanh()
        self.out_outf = torch.nn.Sigmoid()
        #self.out_outf = torch.nn.Tanh()
        
        self.out_layer = torch.nn.Linear(in_features=n_hid, out_features=out_vocab_size * out_len).to(device)

    def forward(self, X, Y):
        '''互換性のため Y を入力としているが実際には使っていない'''

        # 入力 X はトークン ID リストであるので，ワンホットベクトル化する
        X = torch.nn.functional.one_hot(X, num_classes=self.inp_vocab_size)

        X = X.reshape(X.size(0),-1) # ワンホットベクトルを連接して行ベクトルに変換
        X = X.float()               # ワンホットベクトルは整数 int64 なので浮動小数点に変換
        
        X = self.emb_layer(X)       # 埋め込み層への信号伝搬
        X = self.emb_outf(X)        # 埋め込み層の非線形変換
        
        X = self.out_layer(X)       # 出力層への信号伝搬
        X = self.out_outf(X)        # 出力層での非線形変換

        # 各出力ニューロンに分割
        X = X.reshape(X.size(0), self.out_len, self.out_vocab_size)

        return X

# vanilla_tla = vanilla_TLA(device=device)
# print(vanilla_tla.eval())
# print(f'vanilla_tla.out_len:{vanilla_tla.out_len}')


class Seq2Seq_wAtt(nn.Module):
    """ 注意つき符号化器‐復号化器モデル
    Bahdanau, Cho, & Bengio (2015) NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE, arXiv:1409.0473
    """
    def __init__(self,
                 enc_vocab_size:int,
                 dec_vocab_size:int,
                 n_hid:int,
                 n_layers:int=2,
                 bidirectional:bool=False,
                 device='cpu'):
        
        super().__init__()

        # Encoder 側の入力トークン id を多次元ベクトルに変換
        self.encoder_emb = nn.Embedding(num_embeddings=enc_vocab_size,
                                        embedding_dim=n_hid,
                                        padding_idx=0)

        # Decoder 側の入力トークン id を多次元ベクトルに変換
        self.decoder_emb = nn.Embedding(num_embeddings=dec_vocab_size,
                                        embedding_dim=n_hid,
                                        padding_idx=0)

        # Encoder LSTM 本体
        self.encoder = nn.LSTM(input_size=n_hid,
                               hidden_size=n_hid,
                               num_layers=n_layers,
                               batch_first=True,
                               bidirectional=bidirectional)

        # Decoder LSTM 本体
        self.decoder = nn.LSTM(input_size=n_hid,
                               hidden_size=n_hid,
                               num_layers=n_layers,
                               batch_first=True,
                               bidirectional=bidirectional)

        # 文脈ベクトルと出力ベクトルの合成を合成する層
        bi_fact = 2 if bidirectional else 1
        self.combine_layer = nn.Linear(bi_fact * 2 * n_hid, n_hid)

        # 最終出力層
        self.out_layer = nn.Linear(n_hid, dec_vocab_size)

    def forward(self, enc_inp, dec_inp):

        enc_emb = self.encoder_emb(enc_inp)
        enc_out, (hnx, cnx) = self.encoder(enc_emb)

        dec_emb = self.decoder_emb(dec_inp)
        dec_out, (hny, cny) = self.decoder(dec_emb,(hnx, cnx))

        # enc_out は (バッチサイズ，ソースの単語数，中間層の次元数)
        # ソース側 (enc_out) の各単語とターゲット側 (dec_out) の各単語との類似度を測定するため
        # 両テンソルの内積をとるため ソース側 (enc_out) の軸を入れ替え
        enc_outP = enc_out.permute(0,2,1)

        # sim の形状は (バッチサイズ, 中間層の次元数，ソースの単語数)
        sim = torch.bmm(dec_out, enc_outP)

        # sim の各次元のサイズを記録
        batch_size, dec_word_size, enc_word_size = sim.shape

        # sim に対して，ソフトマックスを行うため形状を変更
        simP = sim.reshape(batch_size * dec_word_size, enc_word_size)

        # simP のソフトマックスを用いて注意の重み alpha を算出
        alpha = F.softmax(simP,dim=1).reshape(batch_size, dec_word_size, enc_word_size)

        # 注意の重み alpha に encoder の出力を乗じて，文脈ベクトル c_t とする
        c_t = torch.bmm(alpha, enc_out)

        # torch.cat だから c_t と dec_out とで合成
        dec_out_ = torch.cat([c_t, dec_out], dim=2)
        dec_out_ = self.combine_layer(dec_out_)

        return self.out_layer(dec_out_)

    def evaluate(self, enc_inp, dec_inp):

        enc_emb = self.encoder_emb(enc_inp)
        enc_out, (hnx, cnx) = self.encoder(enc_emb)

        dec_emb = self.decoder_emb(dec_inp)
        dec_out, (hny, cny) = self.decoder(dec_emb,(hnx, cnx))
        return self.out_layer(dec_out)

        # enc_out は (バッチサイズ，ソースの単語数，中間層の次元数)
        # ソース側 (enc_out) の各単語とターゲット側 (dec_out) の各単語との類似度を測定するため
        # 両テンソルの内積をとるため ソース側 (enc_out) の軸を入れ替え
        # enc_outP = enc_out.permute(0,2,1)

        # # sim の形状は (バッチサイズ, 中間層の次元数，ソースの単語数)
        # sim = torch.bmm(dec_out, enc_outP)

        # # sim の各次元のサイズを記録
        # batch_size, dec_word_size, enc_word_size = sim.shape

        # # sim に対して，ソフトマックスを行うため形状を変更
        # simP = sim.reshape(batch_size * dec_word_size, enc_word_size)

        # # simP のソフトマックスを用いて注意の重み alpha を算出
        # alpha = F.softmax(simP,dim=1).reshape(batch_size, dec_word_size, enc_word_size)

        # # 注意の重み alpha に encoder の出力を乗じて，文脈ベクトル c_t とする
        # c_t = torch.bmm(alpha, enc_out)

        # # torch.cat だから c_t と dec_out とで合成
        # dec_out_ = torch.cat([c_t, dec_out], dim=2)
        # dec_out_ = self.combine_layer(dec_out_)

        #return self.out_layer(dec_out_)


# # # 以下確認作業
# # ds = train_ds
# n_layers=1
# bidirectional=False
# n_hid=128
# tla_seq2seq = Seq2Seq_wAtt(enc_vocab_size=len(gakushu_tokenizer.tokens),
#                            dec_vocab_size=len(mora_tokenizer.tokens),
#                            n_layers=n_layers,
#                            bidirectional=bidirectional,
#                            n_hid=n_hid).to(device)
# print(tla_seq2seq.eval())

class Seq2Seq_woAtt(nn.Module):
    """ 注意つき符号化器‐復号化器モデル
    Bahdanau, Cho, & Bengio (2015) NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE, arXiv:1409.0473
    """
    def __init__(self,
                 enc_vocab_size:int,
                 dec_vocab_size:int,
                 n_hid:int,
                 n_layers:int=2,
                 bidirectional:bool=False,
                 device='cpu'):
        super().__init__()

        # Encoder 側の入力トークン id を多次元ベクトルに変換
        self.encoder_emb = nn.Embedding(num_embeddings=enc_vocab_size,
                                        embedding_dim=n_hid,
                                        padding_idx=0)

        # Decoder 側の入力トークン id を多次元ベクトルに変換
        self.decoder_emb = nn.Embedding(num_embeddings=dec_vocab_size,
                                        embedding_dim=n_hid,
                                        padding_idx=0)

        # Encoder LSTM 本体
        self.encoder = nn.LSTM(input_size=n_hid,
                               hidden_size=n_hid,
                               num_layers=n_layers,
                               batch_first=True,
                               bidirectional=bidirectional)

        # Decoder LSTM 本体
        self.decoder = nn.LSTM(input_size=n_hid,
                               hidden_size=n_hid,
                               num_layers=n_layers,
                               batch_first=True,
                               bidirectional=bidirectional)

        # 文脈ベクトルと出力ベクトルの合成を合成する層
        bi_fact = 2 if bidirectional else 1
        self.combine_layer = nn.Linear(bi_fact * 2 * n_hid, n_hid)

        # 最終出力層
        self.out_layer = nn.Linear(n_hid, dec_vocab_size)

    def forward(self, enc_inp, dec_inp):

        enc_emb = self.encoder_emb(enc_inp)
        enc_out, (hnx, cnx) = self.encoder(enc_emb)

        dec_emb = self.decoder_emb(dec_inp)
        dec_out, (hny, cny) = self.decoder(dec_emb,(hnx, cnx))

        return self.out_layer(dec_out)

    def evaluate(self, enc_inp, dec_inp):
        return self.forward(enc_inp, dec_inp)


# # # 以下確認作業
# # ds = train_ds
# n_layers=1
# bidirectional=False
# n_hid=128
# tla_seq2seq = Seq2Seq_wAtt(enc_vocab_size=len(gakushu_tokenizer.tokens),
#                            dec_vocab_size=len(mora_tokenizer.tokens),
#                            n_layers=n_layers,
#                            bidirectional=bidirectional,
#                            n_hid=n_hid).to(device)
# print(tla_seq2seq.eval())

# tla_seq2seq0 = Seq2Seq_woAtt(enc_vocab_size=len(gakushu_tokenizer.tokens),
#                              dec_vocab_size=len(mora_tokenizer.tokens),
#                              n_layers=n_layers,
#                              bidirectional=bidirectional,
#                              n_hid=n_hid).to(device)
# print(tla_seq2seq0.eval())