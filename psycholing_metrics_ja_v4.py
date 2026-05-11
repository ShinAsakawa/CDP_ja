"""
psycholing_metrics_ja.py  ―  日本語心理言語学変数計算モジュール
=================================================================
バージョン: 2.3  (多読み対応版 + log1p_freq + PNS(word,yomi)化 + NFKC順序修正 + em_max_chars=6)

主な変更点 (v1 → v2):
  - load_psylex71: sep=' '/shift-jis/skipinitialspace/usecols修正
  - load_psylex71: _KATA_CHARS による明示的カタカナフィルタ
  - load_psylex71: サブコード重複除去 (word+yomi 単位)
  - load_psylex71: 頻度は word→sum(freq) の合計値を返す
  - PsychoLingEngine: ONS/OLD20 は語形集合(set)で構築 (読み不要)
  - PsychoLingEngine: PNS は全 (word,yomi) ペアで構築するが
    近傍集合は set(words) — 多読み語の二重カウント防止
  - PsychoLingEngine: PLD20 は word 単位でmin距離を取る
  - PsychoLingEngine: SNF は freq_dict[word] (合計頻度) を参照
  - PsychoLingEngine: EM 訓練は全 (word,yomi) ペアを使用

使い方:
    from psycholing_metrics_ja import PsychoLingEngine, load_psylex71

    # WLSP で初期化
    engine = PsychoLingEngine(wlsp_path='bunruidb.txt')

    # psylex71 で初期化
    pairs, freq_dict = load_psylex71('psylex71.txt')
    engine = PsychoLingEngine(pairs=pairs, freq_dict=freq_dict,
                              corpus_name='psylex71')

    # 計算
    m = engine.compute('研究', 'ケンキュウ')
    df = engine.compute_batch([('研究','ケンキュウ'), ('先生','センセイ')])
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

# ╔═══════════════════════════════════════════════════════════╗
# ║  モジュールレベル定数                                      ║
# ╚═══════════════════════════════════════════════════════════╝

SMALL_KANA: frozenset = frozenset('ゃゅょぁぃぅぇぉャュョァィゥェォ')
LONG_VOWEL: str = 'ー'
SOKUON: frozenset = frozenset({'っ', 'ッ'})
HATSUON: frozenset = frozenset({'ん', 'ン'})

# psylex71 用有効カタカナ文字集合
_KATA_CHARS: frozenset = frozenset(
    'ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾ'
    'タダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポ'
    'マミムメモャヤュユョヨラリルレロヮワヰヱヲンヴヵヶー')

# 長音→母音変換表 (WLSP規則)
_CHOUON_MAP: Dict[str, str] = {
    'あ': 'あ', 'い': 'い', 'う': 'う', 'え': 'え', 'お': 'う',
    'ア': 'ア', 'イ': 'イ', 'ウ': 'ウ', 'エ': 'エ', 'オ': 'ウ',
    'か': 'あ', 'き': 'い', 'く': 'う', 'け': 'え', 'こ': 'う',
    'が': 'あ', 'ぎ': 'い', 'ぐ': 'う', 'げ': 'え', 'ご': 'う',
    'さ': 'あ', 'し': 'い', 'す': 'う', 'せ': 'え', 'そ': 'う',
    'ざ': 'あ', 'じ': 'い', 'ず': 'う', 'ぜ': 'え', 'ぞ': 'う',
    'た': 'あ', 'ち': 'い', 'つ': 'う', 'て': 'え', 'と': 'う',
    'だ': 'あ', 'ぢ': 'い', 'づ': 'う', 'で': 'え', 'ど': 'う',
    'な': 'あ', 'に': 'い', 'ぬ': 'う', 'ね': 'え', 'の': 'う',
    'は': 'あ', 'ひ': 'い', 'ふ': 'う', 'へ': 'え', 'ほ': 'う',
    'ば': 'あ', 'び': 'い', 'ぶ': 'う', 'べ': 'え', 'ぼ': 'う',
    'ぱ': 'あ', 'ぴ': 'い', 'ぷ': 'う', 'ぺ': 'え', 'ぽ': 'う',
    'ま': 'あ', 'み': 'い', 'む': 'う', 'め': 'え', 'も': 'う',
    'や': 'あ', 'ゆ': 'う', 'よ': 'う',
    'ら': 'あ', 'り': 'い', 'る': 'う', 'れ': 'え', 'ろ': 'う',
    'わ': 'あ', 'ゐ': 'い', 'ゑ': 'え', 'を': 'う',
    'ぁ': 'あ', 'ぃ': 'い', 'ぅ': 'う', 'ぇ': 'え', 'ぉ': 'う',
    'ゃ': 'あ', 'ゅ': 'う', 'ょ': 'う',
    'ヵ': 'ア', 'ヶ': 'エ',
    'ティ': 'イ', 'ウィ': 'イ', 'ウェ': 'エ', 'ウォ': 'オ',
    'ファ': 'ア', 'フィ': 'イ', 'フェ': 'エ', 'フォ': 'オ',
}

# ╔═══════════════════════════════════════════════════════════╗
# ║  読み正規化ユーティリティ                                  ║
# ╚═══════════════════════════════════════════════════════════╝

def resolve_chouon(reading: str) -> str:
    """長音符号ーを直前の仮名の母音段に変換する (WLSP規則)。"""
    if not reading:
        return reading
    result: List[str] = []
    i = 0
    while i < len(reading):
        ch = reading[i]
        if ch != LONG_VOWEL:
            result.append(ch)
            i += 1
            continue
        # ー を解決
        resolved = None
        for j in range(len(result) - 1, -1, -1):
            candidate = result[j]
            if candidate in SOKUON or candidate in HATSUON:
                continue
            if candidate in _CHOUON_MAP:
                resolved = _CHOUON_MAP[candidate]
                break
        result.append(resolved if resolved else ch)
        i += 1
    return ''.join(result)


def kata_to_hira(text: str) -> str:
    """カタカナをひらがなに変換する (ー等は保持)。"""
    result = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:          # ァ-ヶ
            result.append(chr(code - 0x60))
        else:
            result.append(ch)
    return ''.join(result)


def normalize_reading(reading: str) -> str:
    """カタカナ→ひらがな変換 + 長音解決を一括適用。"""
    return resolve_chouon(kata_to_hira(reading))


def tokenize_mora(hira: str) -> Tuple[str, ...]:
    """ひらがな正規化済み読みをモーラ列に分解する。"""
    morae: List[str] = []
    i = 0
    while i < len(hira):
        ch = hira[i]
        if i + 1 < len(hira) and hira[i + 1] in SMALL_KANA:
            morae.append(ch + hira[i + 1])
            i += 2
        else:
            morae.append(ch)
            i += 1
    return tuple(morae)


def kana_to_mora(reading: str) -> List[str]:
    """カタカナ/ひらがな混在読みをモーラリストに変換 (ーは1モーラ)。"""
    morae: List[str] = []
    i = 0
    while i < len(reading):
        ch = reading[i]
        if i + 1 < len(reading) and reading[i + 1] in SMALL_KANA:
            morae.append(ch + reading[i + 1])
            i += 2
        else:
            morae.append(ch)
            i += 1
    return morae


# ╔═══════════════════════════════════════════════════════════╗
# ║  EM アライメント                                           ║
# ╚═══════════════════════════════════════════════════════════╝

def _log_sum_exp(vals: List[float]) -> float:
    if not vals:
        return -math.inf
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


def _entropy(probs: List[float]) -> float:
    return -sum(p * math.log(p) for p in probs if p > 0)


def _confidence(probs: List[float]) -> float:
    return max(probs) if probs else 0.0


from functools import lru_cache as _lru_cache


@_lru_cache(maxsize=None)
def _enumerate_segmentations(M: int, N: int) -> Tuple[Tuple[int, ...], ...]:
    """M文字をN個のセグメントに分割する全ての境界点リスト。

    lru_cache により同一 (M,N) の再計算をゼロにする。
    hashable な tuple で返す。
    """
    if N == 1:
        return ((M,),)
    result = []
    for cut in range(1, M - N + 2):
        for rest in _enumerate_segmentations(M - cut, N - 1):
            result.append((cut,) + rest)
    return tuple(result)


def _splits_to_segments(M: int,
                         splits: Tuple[int, ...]) -> List[Tuple[int, int]]:
    segs = []
    start = 0
    for length in splits:
        segs.append((start, start + length))
        start += length
    return segs


def _build_entries(
        data: List[Tuple[str, str]],
        freq_weights: Optional[Dict[str, float]] = None,
        max_chars: int = 6,
) -> List[Tuple[str, str, float]]:
    """(word, yomi) ペアから EM 訓練用エントリを構築する。

    Returns: [(word, yomi_hira_normalized, weight), ...]
    """
    import re
    _has_kanji = re.compile(r'[\u4E00-\u9FFF\u3400-\u4DBF]')
    _only_kanji_hira = re.compile(r'^[\u4E00-\u9FFF\u3400-\u4DBF\u3041-\u309F]+$')

    entries = []
    for word, yomi in data:
        if not isinstance(word, str) or not isinstance(yomi, str):
            continue
        if len(word) < 2 or len(word) > max_chars:
            continue
        if not _has_kanji.search(word):
            continue
        if not _only_kanji_hira.match(word):
            continue
        yomi_n = normalize_reading(yomi)
        if not yomi_n:
            continue
        w = float(freq_weights.get(word, 1.0)) if freq_weights else 1.0
        entries.append((word, yomi_n, w))
    return entries


def em_align(
        data: List[Tuple[str, str]],
        n_iter: int = 20,
        smoothing: float = 0.1,
        freq_weights: Optional[Dict[str, float]] = None,
        max_chars: int = 6,
) -> Dict[str, Counter]:
    """IBM Model 1 スタイルの EM でカナ→漢字アライメントを学習する。

    Parameters
    ----------
    data : list of (word, yomi)
    n_iter : EM 反復回数
    smoothing : 加算スムージング量
    freq_weights : word→float の頻度重み辞書 (per-(word,yomi) 重みも可)
    max_chars : 学習対象の最大文字数

    Returns
    -------
    kanji_dist : dict  {kanji_char: Counter({mora_tuple: count})}
    """
    entries = _build_entries(data, freq_weights, max_chars)

    # 初期化: 一様分布
    kanji_dist: Dict[str, Counter] = defaultdict(Counter)

    for _ in range(n_iter):
        new_counts: Dict[str, Counter] = defaultdict(Counter)

        for word, yomi_n, weight in entries:
            chars = list(word)
            morae = tokenize_mora(yomi_n)
            M, N = len(chars), len(morae)
            if M == 0 or N == 0 or N < M:
                continue

            # 全セグメンテーションの確率を計算
            seg_log_probs = []
            segs_list = []
            for splits in _enumerate_segmentations(N, M):
                segs = _splits_to_segments(N, splits)
                log_p = 0.0
                for char, (s, e) in zip(chars, segs):
                    segment = morae[s:e]
                    total = sum(kanji_dist[char].values()) + smoothing * 100
                    cnt = kanji_dist[char].get(segment, 0) + smoothing
                    log_p += math.log(cnt / total)
                seg_log_probs.append(log_p)
                segs_list.append(segs)

            log_Z = _log_sum_exp(seg_log_probs)

            for log_p, segs in zip(seg_log_probs, segs_list):
                p_seg = math.exp(log_p - log_Z)
                for char, (s, e) in zip(chars, segs):
                    segment = morae[s:e]
                    new_counts[char][segment] += p_seg * weight

        # 正規化
        kanji_dist = defaultdict(Counter, {k: Counter(v)
                                           for k, v in new_counts.items()})

    return dict(kanji_dist)


# ╔═══════════════════════════════════════════════════════════╗
# ║  漢字読み統計                                              ║
# ╚═══════════════════════════════════════════════════════════╝

class KanjiStats(NamedTuple):
    consistency: float      # max P(r|K) = Jared 1990 type consistency
    entropy: float          # H(P(r|K))
    n_readings: int
    top_reading: Tuple[str, ...]
    top_prob: float


def _kanji_consistency(kanji: str,
                        kanji_dist: Dict[str, Counter]) -> Tuple[float, float]:
    """(consistency, entropy) を返す。kanji_dist に kanji がなければ (NaN, NaN)。"""
    if kanji not in kanji_dist:
        return float('nan'), float('nan')
    cnt = kanji_dist[kanji]
    total = sum(cnt.values())
    if total == 0:
        return float('nan'), float('nan')
    probs = [v / total for v in cnt.values()]
    return max(probs), _entropy(probs)


def kanji_stats(kanji: str,
                kanji_dist: Dict[str, Counter],
                min_prob: float = 0.01) -> Optional[KanjiStats]:
    if kanji not in kanji_dist:
        return None
    cnt = kanji_dist[kanji]
    total = sum(cnt.values())
    if total == 0:
        return None
    items = [(reading, c / total) for reading, c in cnt.most_common()
             if c / total >= min_prob]
    if not items:
        return None
    top_reading, top_prob = items[0]
    probs = [p for _, p in items]
    return KanjiStats(
        consistency=top_prob,
        entropy=_entropy(probs),
        n_readings=len(items),
        top_reading=top_reading,
        top_prob=top_prob,
    )


# ╔═══════════════════════════════════════════════════════════╗
# ║  非単語読み予測                                            ║
# ╚═══════════════════════════════════════════════════════════╝

class NonwordReading(NamedTuple):
    reading: str                    # 最尤読み (ひらがな)
    probability: float
    n_readings: int
    consistency: float


def predict_nonword(nonword: str,
                    kanji_dist: Dict[str, Counter],
                    smoothing: float = 0.01) -> NonwordReading:
    """非単語の最尤読みを EM 分布から予測する。"""
    chars = list(nonword)
    # 各漢字の最尤読みを結合
    parts = []
    total_prob = 1.0
    consistencies = []
    for ch in chars:
        if ch not in kanji_dist:
            parts.append(ch)
            consistencies.append(float('nan'))
            continue
        cnt = kanji_dist[ch]
        total = sum(cnt.values())
        if total == 0:
            parts.append(ch)
            consistencies.append(float('nan'))
            continue
        top_mora, top_cnt = cnt.most_common(1)[0]
        prob = top_cnt / total
        parts.append(''.join(top_mora))
        total_prob *= prob
        consistencies.append(prob)

    reading = ''.join(parts)
    cons = float(np.nanmean(consistencies)) if consistencies else float('nan')
    return NonwordReading(
        reading=reading,
        probability=total_prob,
        n_readings=1,
        consistency=cons,
    )


def predict_nonword_topn(nonword: str,
                          kanji_dist: Dict[str, Counter],
                          top_n: int = 5) -> List[Tuple[str, float]]:
    """非単語の読み候補を確率の高い順に返す。"""
    chars = list(nonword)
    candidates: List[Tuple[str, float]] = [('', 1.0)]
    for ch in chars:
        if ch not in kanji_dist:
            candidates = [(s + ch, p) for s, p in candidates]
            continue
        cnt = kanji_dist[ch]
        total = sum(cnt.values())
        new_candidates = []
        for top_mora, top_cnt in cnt.most_common(top_n):
            prob = top_cnt / total
            mora_str = ''.join(top_mora)
            for s, p in candidates:
                new_candidates.append((s + mora_str, p * prob))
        new_candidates.sort(key=lambda x: -x[1])
        candidates = new_candidates[:top_n]
    return candidates


# ╔═══════════════════════════════════════════════════════════╗
# ║  psylex71 ローダー                                         ║
# ╚═══════════════════════════════════════════════════════════╝

def load_psylex71(
        path: str,
) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    """psylex71.txt を読み込み (pairs, freq_dict) を返す。

    Parameters
    ----------
    path : str
        psylex71.txt のパス

    Returns
    -------
    pairs : list of (word, yomi)
        サブコード重複を除去した全 (word, yomi) ペア。
        同一 word に複数の正当な yomi がある場合は全て残す。
    freq_dict : dict  {word: total_freq}
        word ごとに全 yomi の頻度を合計した辞書。
    """
    # ── 読み込み ──────────────────────────────────────────────
    # psylex71.txt 列構成（skipinitialspace=True 適用後）:
    #   0: 語番号1, 1: 語番号2(サブコード), 2: 表記(word),
    #   3: 読み(yomi), 4: 品詞コード, 5: 総合頻度, 6-19: コーパス別頻度
    df = pd.read_csv(
        path,
        sep=' ',
        encoding='shift-jis',
        header=None,
        skipinitialspace=True,
        usecols=[2, 3, 5],
        dtype={2: str, 3: str, 5: float},
        on_bad_lines='skip',
    )
    df.columns = ['word', 'yomi', 'freq']
    df = df.dropna(subset=['word', 'yomi', 'freq'])
    df['freq'] = df['freq'].astype(int)

    # ── [1] カタカナフィルタ（NFKC 正規化より先に実施）───────────────
    # 元の文字列に対してフィルタをかける。
    # 理由: NFKC 正規化後にフィルタをかけると，ー（U+30FC）が
    #       別の文字へ変換されたあとで判定が行われ，
    #       「ー」のみからなる読みを正しく排除できない場合がある。
    df = df[df['yomi'].apply(
        lambda y: isinstance(y, str)
                  and len(y) > 0
                  and y != 'ー'
                  and all(c in _KATA_CHARS for c in y)
    )]

    # ── [2] 長音解決（カタカナフィルタ後・NFKC 前）──────────────────
    # ー を直前の仮名の母音段に変換する（WLSP 規則）
    df['yomi'] = df['yomi'].map(
        lambda y: resolve_chouon(y) if isinstance(y, str) else y
    )

    # ── [3] NFKC 正規化（カタカナフィルタ・長音解決の後）───────────
    df['word'] = df['word'].map(lambda x: unicodedata.normalize('NFKC', x)
                                if isinstance(x, str) else x)
    df['yomi'] = df['yomi'].map(lambda x: unicodedata.normalize('NFKC', x)
                                if isinstance(x, str) else x)

    # ── サブコード重複除去 ─────────────────────────────────────
    # 同一 (word, yomi) が col1(サブコード)違いで複数行 → 最大頻度を残す
    df = (df
          .sort_values('freq', ascending=False)
          .drop_duplicates(subset=['word', 'yomi'], keep='first')
          .reset_index(drop=True))

    # ── 出力の構築 ────────────────────────────────────────────
    # pairs: 全 (word, yomi) ペア — 同一 word の複数 yomi を保持
    pairs: List[Tuple[str, str]] = list(zip(df['word'], df['yomi']))

    # freq_dict: word → 全 yomi の頻度合計
    freq_dict: Dict[str, int] = (
        df.groupby('word')['freq'].sum().to_dict()
    )

    return pairs, freq_dict


# ╔═══════════════════════════════════════════════════════════╗
# ║  WLSP ローダー & 前処理                                    ║
# ╚═══════════════════════════════════════════════════════════╝

def _preprocess_wlsp(path: str) -> List[Tuple[str, str]]:
    """WLSP (bunruidb.txt) を読み込み前処理して (word, yomi) ペアを返す。"""
    col_names = [
        'レコードID番号', '見出し番号', 'レコード種別', '類', '部門',
        '中項目', '分類項目', '分類番号', '段落番号', '小段落番号', '語番号',
        '見出し', '見出し本体', '読み', '逆読み',
    ]
    df = pd.read_csv(path, header=None, encoding='shift-jis',
                     names=col_names, on_bad_lines='skip')

    # レコード種別 B を除外
    df = df[df['レコード種別'] != 'B']

    # 記号・接辞・特殊エントリを除外
    def _is_valid_midashi(s: str) -> bool:
        if not isinstance(s, str) or len(s) == 0:
            return False
        if s.strip('＊*') == '':
            return False
        if s.startswith('…') or s.startswith('・'):
            return False
        if s.startswith('−') or s.endswith('−'):
            return False
        if '〓' in s:
            return False
        return True

    df = df[df['見出し本体'].apply(_is_valid_midashi)]
    df = df.dropna(subset=['見出し本体', '読み'])

    # NFKC 正規化
    for col in ['見出し本体', '読み']:
        df[col] = df[col].map(
            lambda x: unicodedata.normalize('NFKC', x).strip()
            if isinstance(x, str) else x
        )

    # 既知のデータ誤りを修正
    df.loc[df['見出し本体'] == 'ウォッカ', '読み'] = 'うぉっか'
    df.loc[df['見出し本体'] == 'だけど',  '読み'] = 'だけど'

    # 「・」「／」で区切られた見出しを展開
    rows = []
    for _, row in df.iterrows():
        midashi = row['見出し本体']
        yomi    = row['読み']
        for m in str(midashi).replace('／', '・').split('・'):
            m = m.strip()
            if m:
                rows.append((m, yomi))

    return rows


# ╔═══════════════════════════════════════════════════════════╗
# ║  レーベンシュタイン距離                                    ║
# ╚═══════════════════════════════════════════════════════════╝

def _edit_distance(a: Tuple, b: Tuple) -> int:
    """編集距離 (Levenshtein)。タプル要素単位で計算。"""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], (
                prev if a[i - 1] == b[j - 1]
                else 1 + min(prev, dp[j], dp[j - 1])
            )
    return dp[n]


# ╔═══════════════════════════════════════════════════════════╗
# ║  PsychoLingEngine                                          ║
# ╚═══════════════════════════════════════════════════════════╝

class WordMetrics(NamedTuple):
    word:               str
    reading_normalized: str
    n_chars:            int
    n_morae:            int
    ons:                int
    pns:                int
    pns_segment:        int
    old20:              float
    pld20:              float
    summed_neighbor_freq: int
    log1p_snf:          float
    log1p_freq:         float   # 対象語自身の log(1+freq); freq = sum across all yomi
    consistency:        float
    reading_entropy:    float


class PsychoLingEngine:
    """日本語心理言語学変数計算エンジン (多読み対応版)。

    初期化方法:
        (A) WLSP を使用:
            engine = PsychoLingEngine(wlsp_path='bunruidb.txt')

        (B) psylex71 を使用:
            pairs, freq_dict = load_psylex71('psylex71.txt')
            engine = PsychoLingEngine(pairs=pairs, freq_dict=freq_dict,
                                      corpus_name='psylex71')

    設計方針:
        - ONS / OLD20: 書記的指標。word の一意集合で構築。読み不要。
        - PNS / PLD20: 音韻的指標。全 (word,yomi) ペアで構築するが
          近傍集合は set(words)。多読み語の二重カウントを防止。
        - freq: word → sum(freq across all yomi)。合計頻度を使用。
        - EM: 全 (word,yomi) ペアで訓練。多読みは正当な訓練シグナル。
    """

    def __init__(
            self,
            wlsp_path: Optional[str] = None,
            *,
            pairs: Optional[List[Tuple[str, str]]] = None,
            freq_dict: Optional[Dict[str, int]] = None,
            corpus_name: Optional[str] = None,
            em_n_iter: int = 20,
            em_max_chars: int = 6,
    ) -> None:
        if wlsp_path is not None and pairs is not None:
            raise ValueError('wlsp_path と pairs は排他。どちらか一方を指定。')

        if wlsp_path is not None:
            print(f'[1/5] WLSP 読み込み: {wlsp_path}')
            raw_pairs = _preprocess_wlsp(wlsp_path)
            self._corpus_name = corpus_name or 'wlsp'
            # WLSP は読み (ひらがな) を正規化してそのまま使う
            pairs_norm: List[Tuple[str, str]] = []
            for word, yomi in raw_pairs:
                yomi_n = normalize_reading(yomi)
                if yomi_n:
                    pairs_norm.append((word, yomi_n))
            # WLSP は多読みなし想定 → (word, yomi) 単位で重複除去
            pairs_norm = list(dict.fromkeys(pairs_norm))
            # freq_dict は均一 (WLSP は頻度なし)
            self._freq_dict: Dict[str, int] = {}
        else:
            if pairs is None:
                raise ValueError('wlsp_path か pairs のどちらかを指定してください。')
            self._corpus_name = corpus_name or 'custom'
            # pairs は load_psylex71 から来る場合，カタカナ → ひらがな変換
            pairs_norm = []
            for word, yomi in pairs:
                yomi_n = normalize_reading(yomi)
                if yomi_n:
                    pairs_norm.append((word, yomi_n))
            self._freq_dict = freq_dict or {}

        print(f'[2/5] インデックス構築 ({len(pairs_norm)} ペア) ...')
        self._build_indices(pairs_norm)

        print(f'[3/5] EM 訓練 (n_iter={em_n_iter}, max_chars={em_max_chars}) ...')
        # EM には freq_weights として per-word 合計頻度を使用
        self._kanji_dist: Dict[str, Counter] = em_align(
            data=pairs_norm,
            n_iter=em_n_iter,
            smoothing=0.1,
            freq_weights=self._freq_dict,
            max_chars=em_max_chars,
        )
        print('[4/5] 完了。')

    # ── 内部インデックス構築 ──────────────────────────────────

    def _build_indices(self, pairs: List[Tuple[str, str]]) -> None:
        """ONS / PNS / OLD20 / PLD20 / SNF 用インデックスを構築する。

        ONS / OLD20:
            書記的指標。語形の一意集合で構築。読みは不要。
        PNS / PLD20:
            音韻的指標。全 (word, yomi) ペアから構築。
            ただし近傍集合は set(words) — 多読み語の二重カウント防止。
        """
        # ── 書記的インデックス (ONS / OLD20) ────────────────
        # word の一意集合
        word_set: set = set(word for word, _ in pairs)

        # ONS テンプレートインデックス: template → set(words)
        self._ons_index: Dict[str, set] = defaultdict(set)
        for word in word_set:
            for i in range(len(word)):
                tmpl = word[:i] + '\x00' + word[i + 1:]
                self._ons_index[tmpl].add(word)

        # OLD20 用単語リスト (書記テンプレート)
        # 各語を文字タプルとして保存
        self._old20_entries: List[Tuple[Tuple[str, ...], str]] = [
            (tuple(w), w) for w in word_set
        ]

        # ── 音韻的インデックス (PNS / PLD20) ────────────────
        # 全 (word, yomi_normalized) ペアを使用
        # (word, yomi) ペア単位でカウント — 多読み語が複数の音韻近傍に
        # それぞれ貢献できるようにする。
        # 例:「明日」が あした・みょうにち 両読みで近傍なら PNS += 2

        # PNS テンプレートインデックス: mora_template → set((word, yomi_n))
        self._pns_index: Dict[str, set] = defaultdict(set)
        # PLD20 用エントリ: [(mora_tuple, word), ...]
        # 同一 word の複数 yomi はいずれも登録
        self._pld20_entries: List[Tuple[Tuple[str, ...], str]] = []

        seen_word_yomi: set = set()
        for word, yomi_n in pairs:
            if (word, yomi_n) in seen_word_yomi:
                continue
            seen_word_yomi.add((word, yomi_n))

            morae = tokenize_mora(yomi_n)
            if not morae:
                continue

            # PNS テンプレート: (word, yomi_n) ペアを格納
            for i in range(len(morae)):
                tmpl = morae[:i] + ('\x00',) + morae[i + 1:]
                key = '|'.join(tmpl)
                self._pns_index[key].add((word, yomi_n))   # (word,yomi) ペア

            # PLD20 用
            self._pld20_entries.append((morae, word))

        print(f'     ONS母集団: {len(word_set)} 語')
        print(f'     PNS母集団: {len(seen_word_yomi)} (word,yomi) ペア')

    # ── 単語メトリクス計算 ───────────────────────────────────

    def compute(self, word: str, reading: str) -> WordMetrics:
        """1語のメトリクスを計算する。

        Parameters
        ----------
        word : str   対象語の表記
        reading : str 対象語の読み (カタカナ or ひらがな)

        Returns
        -------
        WordMetrics NamedTuple
        """
        yomi = normalize_reading(reading)
        morae = tokenize_mora(yomi)
        n_chars = len(word)
        n_morae = len(morae)

        # ── ONS ────────────────────────────────────────────
        ons_neighbors: set = set()
        for i in range(n_chars):
            tmpl = word[:i] + '\x00' + word[i + 1:]
            ons_neighbors |= self._ons_index.get(tmpl, set())
        ons_neighbors.discard(word)
        ons = len(ons_neighbors)

        # ── PNS ────────────────────────────────────────────
        # _pns_index の値は set((word, yomi_n)) ペア。
        # 同一 word が複数 yomi で近傍に入れば複数カウント。
        pns_pairs: set = set()
        for i in range(n_morae):
            tmpl = morae[:i] + ('\x00',) + morae[i + 1:]
            key = '|'.join(str(m) for m in tmpl)
            pns_pairs |= self._pns_index.get(key, set())
        pns_pairs = {p for p in pns_pairs if p[0] != word}   # 対象語除外
        pns = len(pns_pairs)

        # ── PNS (セグメント: 前半/後半を独立に置換) ──────────
        mid = n_morae // 2
        pns_seg_pairs: set = set(pns_pairs)
        for half_morae in [morae[:mid], morae[mid:]]:
            for i in range(len(half_morae)):
                tmpl = half_morae[:i] + ('\x00',) + half_morae[i + 1:]
                key = '|'.join(str(m) for m in tmpl)
                pns_seg_pairs |= self._pns_index.get(key, set())
        pns_seg_pairs = {p for p in pns_seg_pairs if p[0] != word}
        pns_segment = len(pns_seg_pairs)

        # ── OLD20 ──────────────────────────────────────────
        word_chars = tuple(word)
        dists = sorted(
            _edit_distance(word_chars, w_chars)
            for w_chars, w in self._old20_entries
            if w != word
        )
        old20 = float(np.mean(dists[:20])) if len(dists) >= 20 else float('nan')

        # ── PLD20 ──────────────────────────────────────────
        # 多読み語への距離 = その語の全 yomi への最小距離
        word_pld_dists: Dict[str, int] = {}
        for other_morae, other_word in self._pld20_entries:
            if other_word == word:
                continue
            d = _edit_distance(morae, other_morae)
            if other_word not in word_pld_dists or d < word_pld_dists[other_word]:
                word_pld_dists[other_word] = d
        pld_sorted = sorted(word_pld_dists.values())
        pld20 = float(np.mean(pld_sorted[:20])) if len(pld_sorted) >= 20 \
            else float('nan')

        # ── Summed Neighbor Frequency (SNF) ────────────────
        # freq_dict は word → 合計頻度
        snf = sum(self._freq_dict.get(w, 0) for w in ons_neighbors)
        log1p_snf = math.log1p(snf)

        # ── 対象語自身の頻度 ────────────────────────────────
        log1p_freq = math.log1p(self._freq_dict.get(word, 0))

        # ── Consistency / Entropy ───────────────────────────
        chars = list(word)
        cons_vals, ent_vals = [], []
        for ch in chars:
            c, e = _kanji_consistency(ch, self._kanji_dist)
            if not math.isnan(c):
                cons_vals.append(c)
            if not math.isnan(e):
                ent_vals.append(e)
        consistency = float(np.mean(cons_vals)) if cons_vals else float('nan')
        reading_entropy = float(np.mean(ent_vals)) if ent_vals else float('nan')

        return WordMetrics(
            word=word,
            reading_normalized=yomi,
            n_chars=n_chars,
            n_morae=n_morae,
            ons=ons,
            pns=pns,
            pns_segment=pns_segment,
            old20=old20,
            pld20=pld20,
            summed_neighbor_freq=snf,
            log1p_snf=log1p_snf,
            log1p_freq=log1p_freq,
            consistency=consistency,
            reading_entropy=reading_entropy,
        )

    def compute_batch(
            self,
            pairs: List[Tuple[str, str]],
            verbose: bool = True,
    ) -> pd.DataFrame:
        """複数の (word, reading) ペアを一括計算して DataFrame で返す。"""
        rows = []
        total = len(pairs)
        for i, (word, reading) in enumerate(pairs):
            if verbose and (i + 1) % 100 == 0:
                print(f'  {i + 1}/{total} ...')
            rows.append(self.compute(word, reading)._asdict())
        return pd.DataFrame(rows)

    def kanji_info(self, kanji: str) -> Optional[dict]:
        """単一漢字の読み分布情報を返す。"""
        if kanji not in self._kanji_dist:
            return None
        cnt = self._kanji_dist[kanji]
        total = sum(cnt.values())
        readings = [(''.join(r), c / total)
                    for r, c in cnt.most_common(10)]
        cons, ent = _kanji_consistency(kanji, self._kanji_dist)
        return {
            'kanji': kanji,
            'consistency': cons,
            'entropy': ent,
            'n_readings': len(cnt),
            'top_readings': readings,
        }

    @property
    def kanji_dist(self) -> Dict[str, Counter]:
        """EM 学習済みカナ→漢字分布。"""
        return self._kanji_dist

    @property
    def corpus_name(self) -> str:
        return self._corpus_name


# ╔═══════════════════════════════════════════════════════════╗
# ║  デモ                                                      ║
# ╚═══════════════════════════════════════════════════════════╝

if __name__ == '__main__':
    import os, time

    HOME = os.environ['HOME']
    WLSP_PATH = os.path.join(HOME, 'study/2026masayu-a_WLSP.git/bunruidb.txt')
    PSYLEX_PATH = 'psylex71.txt'

    if os.path.exists(PSYLEX_PATH):
        print('=== psylex71 エンジン ===')
        t0 = time.perf_counter()
        pairs, freq_dict = load_psylex71(PSYLEX_PATH)
        print(f'  pairs: {len(pairs)}, unique words: {len(freq_dict)}')
        engine = PsychoLingEngine(pairs=pairs, freq_dict=freq_dict,
                                  corpus_name='psylex71')
        print(f'  初期化: {time.perf_counter() - t0:.1f}s')
    elif os.path.exists(WLSP_PATH):
        print('=== WLSP エンジン ===')
        engine = PsychoLingEngine(wlsp_path=WLSP_PATH)
    else:
        print('データファイルが見つかりません。')
        raise SystemExit(1)

    test_pairs = [
        ('研究', 'ケンキュウ'),
        ('国語', 'コクゴ'),
        ('今日', 'キョウ'),       # 多読み語
        ('今日', 'コンニチ'),     # 同語の別読み
        ('明日', 'アシタ'),       # 多読み語
        ('明日', 'ミョウニチ'),   # 同語の別読み
    ]
    df = engine.compute_batch(test_pairs, verbose=False)
    print(df[['word', 'reading_normalized', 'ons', 'pns', 'old20',
              'consistency', 'reading_entropy']].to_string())