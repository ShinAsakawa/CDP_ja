import torch
import torch.optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random

def init_seed(seed:int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def fit_an_epoch(model:torch.nn.Module=None,
                 optimizer:torch.optim=None,
                 loss_f:torch.nn.modules=None,
                 _dl:torch.utils.data.dataloader.DataLoader=None,
                 is_seq2seq:bool=True,
                 device:str="cpu"):
    """改訂版 1 エポック分の学習を行う"""

    model.train()  # モデルを訓練モードに変更

    sum_loss=0
    count=0
    N = 0

    for inps, tchs in _dl:
        inps = pad_sequence(inps, batch_first=True).to(device)
        tchs = pad_sequence(tchs, batch_first=True).to(device)

        if is_seq2seq:
            enc_inps, enc_tchs = inps[:,:-1], inps[:,1:]
            dec_inps, dec_tchs = tchs[:,:-1], tchs[:,1:]
            dec_outs, enc_outs = model(enc_inps, dec_inps)
        else:
            dec_outs = model(inps, tchs)
            dec_tchs = tchs

        # 正解のカウント
        out_ids = [out.argmax(dim=1) for out in dec_outs]
        for tch, out in zip(dec_tchs[:], out_ids[:]):
            yesno = ((tch==out) * 1).sum().cpu().numpy() == len(tch)
            count += 1 if yesno else 0

        # 学習の実行
        optimizer.zero_grad()
        loss = 0.
        if is_seq2seq:
            for j in range(len(enc_tchs)):
                loss += loss_f(enc_outs[j], enc_tchs[j])
            for j in range(len(dec_tchs)):
                loss += loss_f(dec_outs[j], dec_tchs[j])
            sum_loss += loss.item()
        else:
            for j in range(len(dec_tchs)):
                loss += loss_f(dec_outs[j], dec_tchs[j])
            sum_loss += loss.item()

        loss.backward()  # 損失値の計算
        optimizer.step() # 学習

        N += len(tchs)
    p_ = count / N
    return {'sum_loss':sum_loss, 'count':count, 'N':N, 'P':p_}, model, optimizer


def eval_an_epoch(model:torch.nn.Module=None,
                  loss_f:torch.nn.modules=None,
                  _dl:torch.utils.data.dataloader.DataLoader=None,
                  is_seq2seq:bool=True,
                  device:str='cpu' ):
    """1 エポック分の検証,評価を行う"""

    model.eval()  # モデルを評価モードに変更

    sum_loss=0
    count=0
    N = 0

    for inps, tchs in _dl:
        inps = pad_sequence(inps, batch_first=True).to(device)
        tchs = pad_sequence(tchs, batch_first=True).to(device)

        if is_seq2seq:
            enc_inps, enc_tchs = inps[:,:-1], inps[:,1:]
            dec_inps, dec_tchs = tchs[:,:-1], tchs[:,1:]

            dec_outs, enc_outs = model(enc_inps, dec_inps)
        else:
            dec_outs = model(inps, tchs)
            dec_tchs = tchs

        # 正解のカウント
        out_ids = [out.argmax(dim=1) for out in dec_outs]
        for tch, out in zip(dec_tchs[:], out_ids[:]):
            yesno = ((tch==out) * 1).sum().cpu().numpy() == len(tch)
            count += 1 if yesno else 0

        loss = 0.
        if is_seq2seq:
            for j in range(len(enc_tchs)):
                loss += loss_f(enc_outs[j], enc_tchs[j])
            for j in range(len(dec_tchs)):
                loss += loss_f(dec_outs[j], dec_tchs[j])
            sum_loss += loss.item()
        else:
            for j in range(len(dec_tchs)):
                loss += loss_f(dec_outs[j], dec_tchs[j])
            sum_loss += loss.item()

        N += len(tchs)
    p_ = count / N
    return {'sum_loss':sum_loss, 'count':count, 'N':N, 'P':p_}, model