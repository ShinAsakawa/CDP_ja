# モーラ分かち書きの定義 source https://qiita.com/shimajiroxyz/items/a133d990df2bc3affc12
import re

# # 各条件を正規表現で表す
# c1 = '[ウクスツヌフムユルグズヅブプヴ][ァィェォ]' #ウ段＋「ァ/ィ/ェ/ォ」
# c2 = '[イキシチニヒミリギジヂビピ][ャュェョ]' #イ段（「イ」を除く）＋「ャ/ュ/ェ/ョ」
# c3 = '[テデ][ィュ]' #「テ/デ」＋「ャ/ィ/ュ/ョ」
# c4 = '[ァ-ヴー]' #カタカナ１文字（長音含む）
#
# cond = '('+c1+'|'+c2+'|'+c3+'|'+c4+')'
# re_mora = re.compile(cond)
#
# def moraWakachi(kana_text):
#     return re_mora.findall(kana_text)


class mora_Tokenizer:
    def __init__(self, display:bool=True):


        special_tokens = ['<PAD>', '<EOW>', '<SOW>', '<UNK>']

        mora_list = ['ァ', 'ア', 'ィ', 'イ', 'イェ', 'ゥ', 'ウ', 'ウィ', 'ウェ', 'ウォ', 'ェ', 'エ', 'ォ', 'オ', 'カ', 'ガ', 'キ', 'キャ', 'キュ', 'キョ', 'ギ', 'ギャ', 'ギュ', 'ギョ', 'ク', 'クァ', 'クィ', 'クェ', 'クォ', 'グ', 'グァ', 'ケ', 'ゲ', 'コ', 'ゴ', 'サ', 'ザ', 'シ', 'シェ', 'シャ', 'シュ', 'ショ', 'ジ', 'ジェ', 'ジャ', 'ジュ', 'ジョ', 'ス', 'ズ', 'ズィ', 'セ', 'ゼ', 'ソ', 'ゾ', 'タ', 'ダ', 'チ', 'チェ', 'チャ', 'チュ', 'チョ', 'ヂ', 'ヂャ', 'ヂュ', 'ヂョ', 'ッ', 'ツ', 'ツァ', 'ツィ', 'ツェ', 'ツォ', 'ヅ', 'テ', 'ティ', 'テュ', 'デ', 'ディ', 'デュ', 'ト', 'ド', 'ナ', 'ニ', 'ニェ', 'ニャ', 'ニュ', 'ニョ', 'ヌ', 'ネ', 'ノ', 'ハ', 'バ', 'パ', 'ヒ', 'ヒェ', 'ヒャ', 'ヒュ', 'ヒョ', 'ビ', 'ビャ', 'ビュ', 'ビョ', 'ピ', 'ピャ', 'ピュ', 'ピョ', 'フ', 'ファ', 'フィ', 'フェ', 'フォ', 'ブ', 'ブィ', 'プ', 'ヘ', 'ベ', 'ペ', 'ホ', 'ボ', 'ポ', 'マ', 'ミ', 'ミャ', 'ミュ', 'ミョ', 'ム', 'メ', 'モ', 'ヤ', 'ュ', 'ユ', 'ョ', 'ヨ', 'ラ', 'リ', 'リェ', 'リャ', 'リュ', 'リョ', 'ル', 'レ', 'ロ', 'ヮ', 'ワ', 'ヲ', 'ン', 'ヴ', 'ヴァ', 'ヴィ', 'ヴェ', 'ヴォ', 'ー']

        self.tokens = special_tokens + mora_list

        # 各条件を正規表現で表す
        c1 = '[ウクスツヌフムユルグズヅブプヴ][ァィェォ]' #ウ段＋「ァ/ィ/ェ/ォ」
        c2 = '[イキシチニヒミリギジヂビピ][ャュェョ]' #イ段（「イ」を除く）＋「ャ/ュ/ェ/ョ」
        c3 = '[テデ][ィュ]' #「テ/デ」＋「ャ/ィ/ュ/ョ」
        c4 = '[ァ-ヴー]' #カタカナ１文字（長音含む）

        self.cond = '('+c1+'|'+c2+'|'+c3+'|'+c4+')'
        self.re_mora = re.compile(self.cond)

        if display:
            print("モーラ分かち書きトークナイザ: mora_Tokenizer():")
            print(f"self.tokens:{self.tokens}")
            print("""
    code:
        c1 = '[ウクスツヌフムユルグズヅブプヴ][ァィェォ]' #ウ段＋「ァ/ィ/ェ/ォ」
        c2 = '[イキシチニヒミリギジヂビピ][ャュェョ]' #イ段（「イ」を除く）＋「ャ/ュ/ェ/ョ」
        c3 = '[テデ][ィュ]' #「テ/デ」＋「ャ/ィ/ュ/ョ」
        c4 = '[ァ-ヴー]' #カタカナ１文字（長音含む）

        self.cond = '('+c1+'|'+c2+'|'+c3+'|'+c4+')'
        self.re_mora = re.compile(self.cond)""")
            print()


    def moraWakachi(self, kana_text):
        return self.re_mora.findall(kana_text)


    def wakachi(self, kana_text):
        kana_text = kana_text.replace('ヱ','エ').replace('ヰ','イ')
        morae = self.moraWakachi(kana_text)
        return morae

    def encode(self, kana_text):
        kana_text = kana_text.replace('ヱ','エ').replace('ヰ','イ')
        morae = self.moraWakachi(kana_text)
        ids = [self.tokens.index(_mora) for _mora in morae]
        return ids

    def decode(self, ids):
        out = []
        for idx in ids:
            m = self.tokens[idx]
            out.append(m)
        return out

    def __call__(self, kana_text):
        return self.encode(kana_text)

#mora_tokenizer = mora_Tokenizer()
# mora_tokenizer の検証
# for _w in ['キチジョージ', 'チキン', 'ガッッキューホーカイー']:
#     print(_w, mora_tokenizer(_w), mora_tokenizer.decode(mora_tokenizer(_w)))
#     print(_w, mora_tokenizer.encode(_w))

# 訓令式ローマ字トークナイザ kunrei_Tokenizer() の定義
class kunrei_Tokenizer():

    '''訓令式ローマ字 tokenizer の定義

        ア イ ウ エ オ     a i u e o              
        カ キ ク ケ コ  キャ キュ キョ ka ki ku ke ko  kya kyu kyo          
        サ シ ス セ ソ  シャ シュ ショ sa shi su se so  sha shu sho          
        タ チ ツ テ ト  チャ チュ チョ ta chi tsu te to  cha chu cho          
        ナ ニ ヌ ネ ノ  ニャ ニュ ニョ na ni nu ne no  nya nyu nyo          
        ハ ヒ フ へ ホ  ヒャ ヒュ ヒョ ha hi fu he ho  hya hyu hyo          
        マ ミ ム メ モ  ミャ ミュ ミョ ma mi mu me mo  mya myu myo          
        ヤ  ユ  ヨ     ya  yu  yo              
        ラ リ ル レ ロ  リャ リュ リョ ra ri ru re ro  rya  ryu ryo          
        ワ    ヲ     wa    o              
        ガ ギ グ ゲ ゴ  ギャ ギュ ギョ ga gi gu ge go  gya gyu gyo          
        ザ ジ ズ ゼ ゾ  ジャ ジュ ジョ za ji zu ze zo  ja ju jo          
        ダ ヂ ヅ デ ド  ヂャ ヂュ ヂョ da ji zu de do  ja ju jo          
        バ ビ ブ ベ ボ  ビャ ビュ ビョ ba bi bu be bo  bya byu byo          
        パ ピ プ ペ ポ  ピャ ピュ ピョ pa pi pu pe po  pya pyu pyo
    '''

    def __init__(self, display:bool=True):
    
        self.kunrei_trans_dict = {
            'ァ':'a',    'ア':'a',    'ィ':'i',    'イ':'i',    'イェ': 'i e',
            'ゥ':'u',    'ウ':'u',    'ウィ':'u i',    'ウェ':'u e',    'ウォ':'u o',
            'ェ':'e',    'エ':'e',    'ォ':'o',    'オ':'o',    'カ':'k a', 
            'ガ':'g a',    'キ':'k i',    'キャ':'ky a',    'キュ':'ky u',    'キョ':'ky o',
            'ギ':'g i',     'ギャ':'gy a',     'ギュ':'gy u',     'ギョ':'gy o',     'ク':'k u', 
            'クァ':'k u a',     'クィ':'k u i',     'クェ':'k u e',     'クォ':'k u o',     'グ':'g u', 
            'グァ':'g u a',     'ケ':'k e',     'ゲ':'g e',     'コ':'k o',     'ゴ':'g o',
            'サ':'s a',     'ザ':'z a',     'シ':'s a',     'シェ':'sy e',     'シャ':'sy a',
            'シュ':'sy u',     'ショ':'sy o',     'ジ':'g i',     'ジェ':'gy e',     'ジャ':'gy a',
            'ジュ':'gy u',     'ジョ':'gy o',     'ス':'s u',     'ズ':'z u',     'ズィ':'z i',
            'セ':'s e',     'ゼ':'z e',     'ソ':'s o',     'ゾ':'z o',     'タ':'t a',
            'ダ':'d a',     'チ':'t i',     'チェ':'ch e',     'チャ':'ch a',     'チュ':'ch u', 
            'チョ':'ch o',     'ヂ':'z i',     'ヂャ':'zy a',     'ヂュ':'zy u',     'ヂョ':'zy o',
            'ッ':'t u',    'ツ':'t u',    'ツァ':'ty a',    'ツィ':'ty i',    'ツェ':'ty e',
            'ツォ':'ty o',     'ヅ':'z u',     'テ':'t e',     'ティ':'t i',     'テュ':'t u', 
            'デ':'d e',     'ディ':'d i',     'デュ':'d u',     'ト':'t o',     'ド':'d o', 
            'ナ':'n a',     'ニ':'n i',     'ニェ':'ny e',    'ニャ':'ny a',    'ニュ':'ny u',
            'ニョ':'ny o',    'ヌ':'n u',    'ネ':'n e',    'ノ':'n o',
            'ハ':'h a',    'バ':'b a',    'パ':'p a',    'ヒ':'h i',    'ヒェ':'hy e',
            'ヒャ':'hy a',    'ヒュ':'hy u',    'ヒョ':'hy o',    'ビ':'b i',    'ビャ':'by a',
            'ビュ':'by u',    'ビョ':'by o',    'ピ':'p i',    'ピャ':'py a',    'ピュ':'py u', 
            'ピョ':'py o',    'フ':'f u',    'ファ':'f a',    'フィ':'f u i',    'フェ':'f u e',
            'フォ':'f u o',    'ブ':'b u',    'ブィ':'b i',    'プ':'p u',    'ヘ':'h e',
            'ベ':'b e',    'ペ':'p e',    'ホ':'h o',    'ボ':'b o',    'ポ':'p o',    'マ':'m a',
            'ミ':'m i',    'ミャ':'my a',    'ミュ':'my u',    'ミョ':'my o',
            'ム':'m u',    'メ':'m e',    'モ':'m o',    'ヤ':'y a',    'ュ':'y u',
            'ユ':'y o',    'ョ':'y o',    'ヨ':'y o',    'ラ':'r a',    'リ':'r i',
            'リェ':'ry e',    'リャ':'ry a',    'リュ':'ry u',    'リョ':'ry o',    'ル':'r u',
            'レ':'r e',    'ロ':'r o',    'ヮ':'w a',    'ワ':'w a',    'ヲ':'o',    'ン':'N',
            'ヴ':'b o',    'ヴァ':'b a',    'ヴィ':'b i',    'ヴェ':'b o',    'ヴォ':'b o', 
            'ー':':', '〜':':'}
        self.mora_tokenizer = mora_Tokenizer(display=False)

        tokens = [':', 'N', 'a', 'b', 'by', 'ch', 'd', 'e', 'f', 'g', 'gy', 'h', 'hy', 'i', 'k', 'ky', 'm', 'my', 'n', 'ny', 'o', 'p', 'py', 'r', 'ry', 's', 'sy', 't', 'ty', 'u', 'w', 'y', 'z', 'zy']
        special_tokens = ['<PAD>', '<EOW>', '<SOW>', '<UNK>']
        self.tokens = special_tokens + tokens

        if display:
            print("訓令式ローマ字トークナイザ: kunrei_Tokenizer():")
            print(f"self.tokens:{self.tokens}")
            print("""
ア イ ウ エ オ     a i u e o              
カ キ ク ケ コ  キャ キュ キョ ka ki ku ke ko  kya kyu kyo          
サ シ ス セ ソ  シャ シュ ショ sa shi su se so  sha shu sho          
タ チ ツ テ ト  チャ チュ チョ ta chi tsu te to  cha chu cho          
ナ ニ ヌ ネ ノ  ニャ ニュ ニョ na ni nu ne no  nya nyu nyo          
ハ ヒ フ へ ホ  ヒャ ヒュ ヒョ ha hi fu he ho  hya hyu hyo          
マ ミ ム メ モ  ミャ ミュ ミョ ma mi mu me mo  mya myu myo          
ヤ  ユ  ヨ     ya  yu  yo              
ラ リ ル レ ロ  リャ リュ リョ ra ri ru re ro  rya  ryu ryo          
ワ    ヲ     wa    o              
ガ ギ グ ゲ ゴ  ギャ ギュ ギョ ga gi gu ge go  gya gyu gyo          
ザ ジ ズ ゼ ゾ  ジャ ジュ ジョ za ji zu ze zo  ja ju jo          
ダ ヂ ヅ デ ド  ヂャ ヂュ ヂョ da ji zu de do  ja ju jo          
バ ビ ブ ベ ボ  ビャ ビュ ビョ ba bi bu be bo  bya byu byo          
パ ピ プ ペ ポ  ピャ ピュ ピョ pa pi pu pe po  pya pyu pyo
""")



    def encode(self, kana_text):
        # kana_text = kana_text.replace('ヱ','エ').replace('ヰ','イ')
        # mora_wakachi = self.mora_tokenizer.moraWakachi(kana_text)

        # phon = []
        # for mora in mora_wakachi:
        #     if not mora in self.kunrei_trans_dict:
        #         p = self.tokens.index('<UNK>')
        #     else:
        #         p = self.kunrei_trans_dict[mora].split(' ')
        #     phon = phon + p

        phon = self.wakachi(kana_text)
        out = []
        for p in phon:
            if not p in self.tokens:
                out.append(self.tokens.index('<UNK>'))
            else:
                out.append(self.tokens.index(p))
        return out

    def wakachi(self, kana_text):
        kana_text = kana_text.replace('ヱ','エ').replace('ヰ','イ')
        mora_wakachi = self.mora_tokenizer.moraWakachi(kana_text)

        phon = []
        for mora in mora_wakachi:
            if not mora in self.kunrei_trans_dict:
                p = self.tokens.index('<UNK>')
            else:
                p = self.kunrei_trans_dict[mora].split(' ')
            phon = phon + p
        return phon

    def decode(self, ids):
        out = []
        for idx in ids:
            m = self.tokens[idx]
            out.append(m)
        return out

    def __call__(self, kana_text):
        return self.encode(kana_text)

# kunrei_tokenizer = kunrei_Tokenizer()
# word = 'アカサタナ'
# ids = kunrei_tokenizer(word)
# print(word, ids, kunrei_tokenizer.decode(ids))
# print(kunrei_tokenizer.wakachi(word))

class gakushu_Tokenizer():
    def __init__(self, display:bool=True):

        special_tokens = ['<PAD>', '<EOW>', '<SOW>', '<UNK>']
        alphabet_upper_chars='ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ'
        alphabet_lower_chars='ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
        num_chars='０１２３４５６７８９'
        hira_chars='ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわゐゑをん'
        kata_chars='ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヰヱヲンヴヵヶ'

        # 学習漢字 学年別
        gakushu_chars = '一右雨円王音下火花貝学気休玉金九空月犬見五口校左三山四子糸字耳七車手十出女小上森人水正生青石赤先千川早草足村大男竹中虫町天田土二日入年白八百文本名木目夕立力林六' + '引羽雲園遠黄何夏家科歌画会回海絵外角楽活間丸岩顔帰汽記弓牛魚京強教近兄形計元原言古戸午後語交光公工広考行高合国黒今才細作算姉市思止紙寺時自室社弱首秋週春書少場色食心新親図数星晴声西切雪線船前組走多太体台谷知地池茶昼朝長鳥直通弟店点電冬刀東当答頭同道読内南肉馬買売麦半番父風分聞米歩母方北妹毎万明鳴毛門夜野矢友曜用来理里話' + '悪安暗委意医育員飲院運泳駅横屋温化荷界開階寒感漢館岸期起客宮急球究級去橋業局曲銀区苦具君係軽決血研県庫湖向幸港号根祭坂皿仕使始指死詩歯事持次式実写者主取守酒受州拾終習集住重宿所暑助勝商昭消章乗植深申真神身進世整昔全想相送息速族他打対待代第題炭短談着柱注丁帳調追定庭笛鉄転登都度島投湯等豆動童農波配倍箱畑発反板悲皮美鼻筆氷表病秒品夫負部服福物平返勉放味命面問役薬油有由遊予様洋羊葉陽落流旅両緑礼列練路和' + '愛案以位囲胃衣印栄英塩央億加果課貨芽改械害街各覚完官管観関願喜器希旗機季紀議救求泣給挙漁競共協鏡極訓軍郡型径景芸欠結健建験固候功好康航告差最菜材昨刷察札殺参散産残司史士氏試児治辞失借種周祝順初唱松焼照省笑象賞信臣成清静席積折節説戦浅選然倉巣争側束続卒孫帯隊達単置仲貯兆腸低停底的典伝徒努灯働堂得特毒熱念敗梅博飯費飛必標票不付府副粉兵別変辺便包法望牧末満未脈民無約勇要養浴利陸料良量輪類令例冷歴連労老録' + '圧易移因営永衛液益演往応恩仮価可河過賀解快格確額刊幹慣眼基寄規技義逆久旧居許境興均禁句群経潔件券検険減現限個故護効厚構耕講鉱混査再妻採災際在罪財桜雑賛酸師志支枝資飼似示識質舎謝授修術述準序承招証常情条状織職制勢性政精製税績責接設絶舌銭祖素総像増造則測属損態貸退団断築張提程敵適統導銅徳独任燃能破判版犯比肥非備俵評貧婦富布武復複仏編弁保墓報豊暴貿防務夢迷綿輸余預容率略留領' + '異遺域宇映延沿我灰拡閣革割株巻干看簡危揮机貴疑吸供胸郷勤筋敬系警劇激穴憲権絹厳源呼己誤后孝皇紅鋼降刻穀骨困砂座済裁策冊蚕姿私至視詞誌磁射捨尺若樹収宗就衆従縦縮熟純処署諸除傷将障城蒸針仁垂推寸盛聖誠宣専泉洗染善創奏層操窓装臓蔵存尊宅担探誕暖段値宙忠著庁潮頂賃痛展党糖討届難乳認納脳派俳拝背肺班晩否批秘腹奮並閉陛片補暮宝訪亡忘棒枚幕密盟模訳優郵幼欲翌乱卵覧裏律臨朗論'

        self.tokens = special_tokens + list(num_chars) + list(hira_chars) + list(kata_chars) + list(gakushu_chars)

        if display:
            print("学習漢字トークナイザ: gakushu_Tokenizer():")
            print(f'self.tokens:{"".join(c for c in self.tokens)}')
            print()


    def encode(self, chars):
        out = []
        for ch in chars:
            if not ch in self.tokens:
                out.append(self.tokens.index('<UNK>'))
            else:
                out.append(self.tokens.index(ch))
        return out

    def decode(self, ids):
        out = [self.tokens[idx] for idx in ids]
        return out

    def __call__(self, chars):
        return self.encode(chars)

# gakushu_tokenizer = gakushu_Tokenizer()
# 上記 gakushu_tokenizer の検証
# print(gakushu_tokenizer.char_list)
# #gakushu_tokenizer.encode('学校')
# print(gakushu_tokenizer('学校'))
# print(gakushu_tokenizer.decode(gakushu_tokenizer('学校')))
#print(len(gakushu_tokenizer.tokens), len(mora_tokenizer.tokens),)


class joyo_Tokenizer():
    def __init__(self, display:bool=True):

        special_tokens = ['<PAD>', '<EOW>', '<SOW>', '<UNK>']
        alphabet_upper_chars='ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ'
        alphabet_lower_chars='ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
        num_chars='０１２３４５６７８９'
        hira_chars='ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわゐゑをん'
        kata_chars='ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヰヱヲンヴヵヶ'

        # 学習漢字 学年別
        joyo_chars = '亜哀挨愛曖悪握圧扱宛嵐安案暗以衣位囲医依委威為畏胃尉異移萎偉椅彙意違維慰遺緯域育一壱逸茨芋引印因咽姻員院淫陰飲隠韻右宇羽雨唄鬱畝浦運雲永泳英映栄営詠影鋭衛易疫益液駅悦越謁閲円延沿炎怨宴媛援園煙猿遠鉛塩演縁艶汚王凹央応往押旺欧殴桜翁奥横岡屋億憶臆虞乙俺卸音恩温穏下化火加可仮何花佳価果河苛科架夏家荷華菓貨渦過嫁暇禍靴寡歌箇稼課蚊牙瓦我画芽賀雅餓介回灰会快戒改怪拐悔海界皆械絵開階塊楷解潰壊懐諧貝外劾害崖涯街慨蓋該概骸垣柿各角拡革格核殻郭覚較隔閣確獲嚇穫学岳楽額顎掛潟括活喝渇割葛滑褐轄且株釜鎌刈干刊甘汗缶完肝官冠巻看陥乾勘患貫寒喚堪換敢棺款間閑勧寛幹感漢慣管関歓監緩憾還館環簡観韓艦鑑丸含岸岩玩眼頑顔願企伎危机気岐希忌汽奇祈季紀軌既記起飢鬼帰基寄規亀喜幾揮期棋貴棄毀旗器畿輝機騎技宜偽欺義疑儀戯擬犠議菊吉喫詰却客脚逆虐九久及弓丘旧休吸朽臼求究泣急級糾宮救球給嗅窮牛去巨居拒拠挙虚許距魚御漁凶共叫狂京享供協況峡挟狭恐恭胸脅強教郷境橋矯鏡競響驚仰暁業凝曲局極玉巾斤均近金菌勤琴筋僅禁緊錦謹襟吟銀区句苦駆具惧愚空偶遇隅串屈掘窟熊繰君訓勲薫軍郡群兄刑形系径茎係型契計恵啓掲渓経蛍敬景軽傾携継詣慶憬稽憩警鶏芸迎鯨隙劇撃激桁欠穴血決結傑潔月犬件見券肩建研県倹兼剣拳軒健険圏堅検嫌献絹遣権憲賢謙鍵繭顕験懸元幻玄言弦限原現舷減源厳己戸古呼固股虎孤弧故枯個庫湖雇誇鼓錮顧五互午呉後娯悟碁語誤護口工公勾孔功巧広甲交光向后好江考行坑孝抗攻更効幸拘肯侯厚恒洪皇紅荒郊香候校耕航貢降高康控梗黄喉慌港硬絞項溝鉱構綱酵稿興衡鋼講購乞号合拷剛傲豪克告谷刻国黒穀酷獄骨駒込頃今困昆恨根婚混痕紺魂墾懇左佐沙査砂唆差詐鎖座挫才再災妻采砕宰栽彩採済祭斎細菜最裁債催塞歳載際埼在材剤財罪崎作削昨柵索策酢搾錯咲冊札刷刹拶殺察撮擦雑皿三山参桟蚕惨産傘散算酸賛残斬暫士子支止氏仕史司四市矢旨死糸至伺志私使刺始姉枝祉肢姿思指施師恣紙脂視紫詞歯嗣試詩資飼誌雌摯賜諮示字寺次耳自似児事侍治持時滋慈辞磁餌璽鹿式識軸七𠮟失室疾執湿嫉漆質実芝写社車舎者射捨赦斜煮遮謝邪蛇尺借酌釈爵若弱寂手主守朱取狩首殊珠酒腫種趣寿受呪授需儒樹収囚州舟秀周宗拾秋臭修袖終羞習週就衆集愁酬醜蹴襲十汁充住柔重従渋銃獣縦叔祝宿淑粛縮塾熟出述術俊春瞬旬巡盾准殉純循順準潤遵処初所書庶暑署緒諸女如助序叙徐除小升少召匠床抄肖尚招承昇松沼昭宵将消症祥称笑唱商渉章紹訟勝掌晶焼焦硝粧詔証象傷奨照詳彰障憧衝賞償礁鐘上丈冗条状乗城浄剰常情場畳蒸縄壌嬢錠譲醸色拭食植殖飾触嘱織職辱尻心申伸臣芯身辛侵信津神唇娠振浸真針深紳進森診寝慎新審震薪親人刃仁尽迅甚陣尋腎須図水吹垂炊帥粋衰推酔遂睡穂随髄枢崇数据杉裾寸瀬是井世正生成西声制姓征性青斉政星牲省凄逝清盛婿晴勢聖誠精製誓静請整醒税夕斥石赤昔析席脊隻惜戚責跡積績籍切折拙窃接設雪摂節説舌絶千川仙占先宣専泉浅洗染扇栓旋船戦煎羨腺詮践箋銭潜線遷選薦繊鮮全前善然禅漸膳繕狙阻祖租素措粗組疎訴塑遡礎双壮早争走奏相荘草送倉捜挿桑巣掃曹曽爽窓創喪痩葬装僧想層総遭槽踪操燥霜騒藻造像増憎蔵贈臓即束足促則息捉速側測俗族属賊続卒率存村孫尊損遜他多汰打妥唾堕惰駄太対体耐待怠胎退帯泰堆袋逮替貸隊滞態戴大代台第題滝宅択沢卓拓託濯諾濁但達脱奪棚誰丹旦担単炭胆探淡短嘆端綻誕鍛団男段断弾暖談壇地池知値恥致遅痴稚置緻竹畜逐蓄築秩窒茶着嫡中仲虫沖宙忠抽注昼柱衷酎鋳駐著貯丁弔庁兆町長挑帳張彫眺釣頂鳥朝貼超腸跳徴嘲潮澄調聴懲直勅捗沈珍朕陳賃鎮追椎墜通痛塚漬坪爪鶴低呈廷弟定底抵邸亭貞帝訂庭逓停偵堤提程艇締諦泥的笛摘滴適敵溺迭哲鉄徹撤天典店点展添転塡田伝殿電斗吐妬徒途都渡塗賭土奴努度怒刀冬灯当投豆東到逃倒凍唐島桃討透党悼盗陶塔搭棟湯痘登答等筒統稲踏糖頭謄藤闘騰同洞胴動堂童道働銅導瞳峠匿特得督徳篤毒独読栃凸突届屯豚頓貪鈍曇丼那奈内梨謎鍋南軟難二尼弐匂肉虹日入乳尿任妊忍認寧熱年念捻粘燃悩納能脳農濃把波派破覇馬婆罵拝杯背肺俳配排敗廃輩売倍梅培陪媒買賠白伯拍泊迫剝舶博薄麦漠縛爆箱箸畑肌八鉢発髪伐抜罰閥反半氾犯帆汎伴判坂阪板版班畔般販斑飯搬煩頒範繁藩晩番蛮盤比皮妃否批彼披肥非卑飛疲秘被悲扉費碑罷避尾眉美備微鼻膝肘匹必泌筆姫百氷表俵票評漂標苗秒病描猫品浜貧賓頻敏瓶不夫父付布扶府怖阜附訃負赴浮婦符富普腐敷膚賦譜侮武部舞封風伏服副幅復福腹複覆払沸仏物粉紛雰噴墳憤奮分文聞丙平兵併並柄陛閉塀幣弊蔽餅米壁璧癖別蔑片辺返変偏遍編弁便勉歩保哺捕補舗母募墓慕暮簿方包芳邦奉宝抱放法泡胞俸倣峰砲崩訪報蜂豊飽褒縫亡乏忙坊妨忘防房肪某冒剖紡望傍帽棒貿貌暴膨謀頰北木朴牧睦僕墨撲没勃堀本奔翻凡盆麻摩磨魔毎妹枚昧埋幕膜枕又末抹万満慢漫未味魅岬密蜜脈妙民眠矛務無夢霧娘名命明迷冥盟銘鳴滅免面綿麺茂模毛妄盲耗猛網目黙門紋問冶夜野弥厄役約訳薬躍闇由油喩愉諭輸癒唯友有勇幽悠郵湧猶裕遊雄誘憂融優与予余誉預幼用羊妖洋要容庸揚揺葉陽溶腰様瘍踊窯養擁謡曜抑沃浴欲翌翼拉裸羅来雷頼絡落酪辣乱卵覧濫藍欄吏利里理痢裏履璃離陸立律慄略柳流留竜粒隆硫侶旅虜慮了両良料涼猟陵量僚領寮療瞭糧力緑林厘倫輪隣臨瑠涙累塁類令礼冷励戻例鈴零霊隷齢麗暦歴列劣烈裂恋連廉練錬呂炉賂路露老労弄郎朗浪廊楼漏籠六録麓論和話賄脇惑枠湾腕'

        self.tokens = special_tokens + list(num_chars) + list(hira_chars) + list(kata_chars) + list(joyo_chars)

        if display:
            print("常用漢字トークナイザ: gakushu_Tokenizer():")
            print(f"self.tokens:{"".join(c for c in self.tokens)}")
            print()


    def encode(self, chars):
        out = []
        for ch in chars:
            if not ch in self.tokens:
                out.append(self.tokens.index('<UNK>'))
            else:
                out.append(self.tokens.index(ch))
        return out

    def decode(self, ids):
        out = [self.tokens[idx] for idx in ids]
        return out

    def __call__(self, chars):
        return self.encode(chars)

# gakushu_tokenizer = gakushu_Tokenizer()
# 上記 gakushu_tokenizer の検証
# print(gakushu_tokenizer.char_list)
# #gakushu_tokenizer.encode('学校')
# print(gakushu_tokenizer('学校'))
# print(gakushu_tokenizer.decode(gakushu_tokenizer('学校')))
#print(len(gakushu_tokenizer.tokens), len(mora_tokenizer.tokens),)
