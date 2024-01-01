#!/usr/bin/env python3
import json
import re
import os.path
import pickle
import argparse
import gzip
from datetime import datetime, timezone, timedelta
from collections import Counter
from itertools import chain
from multiprocessing import Pool
from operator import itemgetter
from copy import copy
from bisect import bisect_left

from sudachipy import tokenizer, dictionary
import jaconv

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.image as mpimg
from matplotlib.ticker import MultipleLocator, PercentFormatter
from matplotlib.font_manager import FontProperties

from adjustText import adjust_text

from emoji import EMOJI_DATA

matplotlib.use("module://mplcairo.macosx")

TIMELINE = os.path.join(os.path.dirname(__file__), "timeline.pickle")
TIMEZONE = timezone(timedelta(hours=9), "JST")

matplotlib.rcParams["font.sans-serif"] = ["Hiragino Maru Gothic Pro", "Yu Gothic", "Meirio", "Takao", "IPAexGothic", "IPAPGothic", "VL PGothic", "Noto Sans CJK JP"]
emoji_prop = FontProperties(fname="/System/Library/Fonts/Apple Color Emoji.ttc")

UNICODE_EMOJI = EMOJI_DATA.keys()

# (ward to plot, line style, color)
RTA_EMOTES = (
    ("rtaClap", "-", "#ec7087"),
    (("rtaGg", "GG"), "-", "#ff381c"),
    (("rtaGl", "GL"), "-", "#5cc200"),
    ("rtaPray", "-", "#f7f97a"),
    ("rtaR", "-", "white"),
    ("rtaCheer", "-", "#ffbe00"),
    ("rtaHatena", "-", "#ffb5a1"),
    (("rtaCry", "BibleThump"), "-", "#5ec6ff"),

    ("rtaPokan", "-.", "#838187"),
    ("rtaKabe", "-.", "#bf927a"),
    ("rtaFear", "-.", "#8aa0ec"),
    ("rtaListen", "-.", "#5eb0ff"),
    (("rtaRedbull", "rtaRedbull2", "レッドブル"), "-.", "#98b0df"),
    # ("rtaGogo", "-.", "#df4f69"),
    # ("rtaBanana", ":", "#f3f905"),
    # ("rtaBatsu", ":", "#5aafdd"),
    # ("rtaShogi", ":", "#c68d46"),
    # ("rtaThink", ":", "#f3f905"),
    # ("rtaIizo", ":", "#0f9619"),
    ("rtaDeath", "-.", "#ffbe00"),
    # ("rtaDaba", "-.", "white"),

    ("rtaIce", ":", "#CAEEFA"),
    ("rtaPog", ":", "#f8c900"),
    ("rtaFire", ":", "#E56124"),
    # ("rtaHello", ":", "#ff3291"),
    ("rtaHeart", ":", "#ff3291"),
    # ("rtaHmm", "-.", "#fcc7b9"),
    # ("rtaMaru", ":", "#c80730"),
    ("rtaThunder", ":", "#F5D219"),
    # ("rtaPoison", ":", "#9F65B2"),
    ("rtaGlitch", ":", "#9F65B2"),
    # ("rtaWind", ":", "#C4F897"),
    # ("rtaOko", "-.", "#d20025"),
    # ("rtaWut", ":", "#d97f8d"),
    ("rtaPolice", ":", "#7891b8"),
    # ("rtaChan", "-.", "green"),
    # ("rtaKappa", "-.", "#ffeae2"),

    # ("rtaSleep", "-.", "#ff8000"),
    # ("rtaCafe", "--", "#a44242"),
    # ("rtaDot", "--", "#ff3291"),

    # ("rtaShi", ":", "#8aa0ec"),
    # ("rtaGift", ":", "white"),
    # ("rtaAnkimo", ":", "#f92218 "),

    (("草", "ｗｗｗ", "LUL"), "--", "#1e9100"),
    ("DinoDance", "--", "#00b994"),
    ("rtaFrameperfect", "--", "#ff7401"),
    # # ("rtaPixelperfect", "--", "#ffa300"),
    ("Cheer（ビッツ）", "--", "#bd62fe"),
    ("無敵時間", "--", "red"),
    ("石油王", "--", "yellow"),
    # ("かわいい", "--", "#ff3291")
)

VOCABULARY = set(w for w, _, _, in RTA_EMOTES if isinstance(w, str))
VOCABULARY |= set(chain(*(w for w, _, _, in RTA_EMOTES if isinstance(w, tuple))))

EXCLUDE_MESSAGE_TERMS = (
    " subscribed with Prime",
    " subscribed at Tier ",
    " gifted a Tier ",
    " is gifting ",
    " raiders from "
)

# (title, movie start time as timestamp, offset hour, min, sec)
GAMES = (
    ('開幕のあいさつ', 1703526807.591, 0, 6, 49),
    ('\nクラッシュ・バンディクー ブッとび3段もり！', 1703526807.591, 0, 14, 14),
    ('俺は魚だよ', 1703526807.591, 2, 51, 52),
    ('ドーナツ・ドド', 1703526807.591, 4, 30, 35),
    ('Hollow Knight', 1703526807.591, 5, 11, 15),
    ('Have a Nice Death', 1703526807.591, 6, 40, 38),
    ('メタルホーク', 1703526807.591, 7, 38, 41, "right"),
    ('Touhou\nLuna\nNights', 1703526807.591, 7, 55, 21),
    ('スーパー\nボンバーマン', 1703526807.591, 8, 15, 3),
    ('ワルキューレ\nの伝説', 1703526807.591, 8, 48, 16),
    ('SIREN: New Translation', 1703526807.591, 9, 11, 19),
    ('Devil May Cry 3: Special Edition', 1703526807.591, 10, 21, 15),
    ('Ib', 1703526807.591, 11, 59, 27),
    ('DEATH STRANDING', 1703526807.591, 12, 39, 36),

    ('クロックタワーゴーストヘッド', 1703576728.182, 3, 2, 43, "right"),
    ('Human: Fall Flat', 1703576728.182, 4, 0, 14, "right"),
    ('Stilt Fella', 1703576728.182, 4, 14, 2),
    ('煉獄弐 The Stairway to H.E.A.V.E.N.', 1703576728.182, 5, 9, 23),
    ('スターフォックス64', 1703576728.182, 6, 33, 34),
    ('ドラえもん のび太と3つの精霊石', 1703576728.182, 7, 39, 12),
    ('ドラえもん3 のび太と時の宝玉', 1703576728.182, 8, 31, 26),
    ('SaGa Frontier Remastered', 1703576728.182, 9, 38, 27),
    ('Forward to the Sky', 1703576728.182, 11, 42, 59),
    ('ファイナルファンタジー\n・クリスタルクロニクル', 1703576728.182, 12, 31, 11),
    ('ポケモンピンボール\nルビー＆サファイア', 1703576728.182, 14, 18, 33, "right"),
    ('フェアルーン2', 1703576728.182, 14, 46, 45),
    ('スライムもりもりドラゴンクエスト2\n大戦車としっぽ団', 1703576728.182, 16, 24, 42),
    ('モンスターハンター3G', 1703576728.182, 18, 30, 52),
    ('Light Infantry', 1703576728.182, 20, 32, 2),
    ('ゼルダの伝説\nムジュラの仮面', 1703576728.182, 21, 17, 3),
    ('ゼルダ無双 厄災の黙示録', 1703576728.182, 21, 47, 20),

    ('ポケットモンスター ブリリアントダイヤモンド・シャイニングパール', 1703665224.608, 0, 7, 4),
    ('ファイアーエムブレム 聖魔の光石', 1703665224.608, 3, 56, 19),
    ('星のカービィ スターアライズ', 1703665224.608, 5, 28, 41),
    ('星のカービィ 64', 1703665224.608, 6, 20, 8),
    ('ピクミン', 1703665224.608, 7, 47, 47),
    ('Carto', 1703665224.608, 9, 5, 38),
    ('風ノ旅ビト', 1703665224.608, 10, 14, 3),
    ('ソニックフォース', 1703665224.608, 10, 57, 7),
    ('shapez', 1703665224.608, 12, 9, 20),
    ('Unpacking', 1703665224.608, 14, 11, 26, "right"),
    ('海のぬし釣り\n宝島に向かって', 1703665224.608, 14, 45, 47, "right"),
    ('ミッキーとドナルド\nマジカル\nアドベンチャー3', 1703665224.608, 15, 8, 38),
    ('ワリオランドシェイク', 1703665224.608, 15, 43, 14),
    ('Pizza Tower', 1703665224.608, 17, 17, 23),
    ('美少女戦士\nセーラームーン', 1703665224.608, 18, 37, 38),
    ('マリーのアトリエGB', 1703665224.608, 19, 8, 8),
    ('ゼノギアス', 1703665224.608, 19, 50, 23),
    ('El Shaddai: Ascension of the Metatron HD Remastered', 1703665224.608, 25, 46, 16),
    ('ペーパーマリオRPG', 1703665224.608, 29, 10, 35),
    ('スーパーマリオ3Dランド', 1703665224.608, 31, 55, 28),
    ('スーパーマリオ\nランド', 1703665224.608, 33, 10, 37),
    ('スーパーマリオ\nブラザーズ2', 1703665224.608, 33, 42, 55),
    ('元祖西遊記\nスーパーモンキー\n大冒険', 1703665224.608, 34, 35, 10, "right"),
    ('ビビッドナイト', 1703665224.608, 34, 53, 53),
    ('ウナきり\nアクション！\n-きりたん砲の謎-', 1703665224.608, 35, 18, 35),
    ('ドラゴンボールZIII\n烈戦人造人間', 1703665224.608, 35, 46, 12),
    ('悪魔城ドラキュラ\nCircle of the moon', 1703665224.608, 36, 46, 56),
    ('Owlboy', 1703665224.608, 37, 23, 52),
    ('ウィザードリィⅣ\nワードナの逆襲', 1703665224.608, 38, 1, 6),
    ('メテオス\n(Straight Star Trip)', 1703665224.608, 38, 40, 52),
    ('\n\n(Multi Star Trip)', 1703665224.608, 38, 48, 30),
    ('ばくばくアニマル 世界飼育係選手権', 1703665224.608, 39, 1, 36),
    ('\nバベルの塔', 1703665224.608, 39, 19, 22),
    ('パズルボブル\nエブリバブル!', 1703665224.608, 40, 39, 33, "right"),
    ('BurgerTime\nDeluxe', 1703665224.608, 40, 59, 32),
    ('カービィボウル', 1703665224.608, 41, 23, 29),
    ('テトリスDX', 1703665224.608, 42, 17, 41),
    ('風のクロノア2 アンコール ～世界が望んだ忘れ物～', 1703665224.608, 42, 51, 49),

    ('My Friendly\nNeighborhood', 1703825575.673, 0, 1, 26),
    ('零～zero～', 1703825575.673, 0, 42, 22),
    ('ポポロクロイス物語', 1703825575.673, 2, 35, 47),
    ('ロックマンゼクスアドベント', 1703825575.673, 7, 43, 16),
    ('ガンヴォルト爪', 1703825575.673, 8, 45, 33),
    ('ウエーブレース64', 1703825575.673, 9, 32, 4),
    ('マリオゴルフ\nGBAツアー', 1703825575.673, 10, 34, 8, "right"),
    ('マリオカート8 デラックス', 1703825575.673, 11, 37, 14),
    ('激走トマランナー', 1703825575.673, 13, 39, 7),
    ('マリオテニス エース', 1703825575.673, 14, 9, 18),
    ('SHINOBI\nNON GRATA', 1703825575.673, 15, 18, 44),
    ('ICEY', 1703825575.673, 15, 42, 9),
    ('CHUNITHM\nLUMINOUS', 1703825575.673, 16, 33, 41, "right"),
    ('風来のシレン2 鬼襲来！シレン城！', 1703825575.673, 18, 9, 54),
    ('ENDER LILIES:\nQuietus of the Knights', 1703825575.673, 21, 11, 57),
    ('HADES', 1703825575.673, 22, 5, 40, "right"),
    ('ソニックフロンティア', 1703825575.673, 22, 45, 57, "right"),
    ('テイルスアドベンチャー', 1703825575.673, 23, 19, 45),
    ('LAPIN', 1703825575.673, 24, 5, 34),
    ('アーシャのアトリエ～黄昏の大地の錬金術士～DX', 1703825575.673, 25, 6, 43),
    ('BIOHAZARD 6', 1703825575.673, 27, 5, 44),
    ('世界樹の迷宮II 諸王の聖杯\nHD REMASTER', 1703825575.673, 30, 11, 29),
    ('One Hand Clapping', 1703825575.673, 31, 15, 40),
    ('Ultimate Doom', 1703825575.673, 34, 1, 6),
    ('ENTROPOLY', 1703825575.673, 34, 34, 20),
    ('Golf It!', 1703825575.673, 35, 0, 28),
    ('バラン\nワンダーワールド', 1703825575.673, 35, 36, 51, "right"),
    ('メトロイド ドレッド', 1703825575.673, 36, 47, 39),
    ('ポケットモンスター ハートゴールド・ソウルシルバー', 1703825575.673, 38, 20, 54),
    ('Sekiro: Shadows Die Twice', 1703825575.673, 40, 12, 53),
    ('閉幕のあいさつ', 1703825575.673, 41, 46, 5, "right")
)


class Game:
    def __init__(self, name, t, h, m, s, align="left"):
        self.name = name
        # self.startat = datetime.fromtimestamp(t + h * 3600 + m * 60 + s).replace(tzinfo=TIMEZONE)
        self.startat = datetime.fromtimestamp(t + h * 3600 + m * 60 + s)
        self.align = align


GAMES = tuple(Game(*args) for args in GAMES)

WINDOWSIZE = 1
WINDOW = timedelta(seconds=WINDOWSIZE)
AVR_WINDOW = 60
PER_SECONDS = 60
FIND_WINDOW = 15
DOMINATION_RATE = 0.6
COUNT_THRESHOLD = 30

DPI = 200
ROW = 5
PAGES = 4
YMAX = 700
WIDTH = 3840
HEIGHT = 2160

FONT_COLOR = "white"
FRAME_COLOR = "#ffff79"
BACKGROUND_COLOR = "#352319"
FACE_COLOR = "#482b1e"
ARROW_COLOR = "#ffff79"
MESSAGE_FILL_COLOR = "#1e0d0b"
MESSAGE_EDGE_COLOR = "#7f502f"

BACKGROUND = "2023w.png"

plt.rcParams['axes.facecolor'] = FACE_COLOR
plt.rcParams['savefig.facecolor'] = FACE_COLOR


class Message:
    _tokenizer = dictionary.Dictionary().create()
    _mode = tokenizer.Tokenizer.SplitMode.C

    pns = (
        "無敵時間",
        "石油王",
        "国境なき医師団",
        "ナイスセーヌ",
        "ハイプトレイン",
        "そうはならんやろ",
        "強制スクロール",
        "暗黒盆踊り",
        "ソードフィッシュ",
        "鬼いちゃん",
        "やったか",
        "昼ドラ",
        "ヴァグ技",
        "おじょうず",
        "いつもの",
        "テトリス仮面",
        "おじさん",
        "謎加速",
        "マンジカブラ",
        "泣けるぜ",
        "冷えピタ",
        "ファイナルアダム",
        "トゥートゥー"
    )
    pn_patterns = (
        (re.compile("[\u30A1-\u30FF]+ケンカ"), "〜ケンカ"),
        (re.compile("[a-zA-Z]+[0-9]+"), "Cheer（ビッツ）"),
        (re.compile("世界[1１一]位?"), "世界一"),
        (re.compile("ヨシ！+"), "ヨシ！"),
        (re.compile("[＃#]?ブリオを許すな",), "ブリオを許すな"),
        (re.compile("(はちみつ|ハチミツ)ください"), "ハチミツください"),
        (re.compile("[＾^][＾^][；;]?"), "＾＾"),
        (re.compile("(何|なに)もしてい?ないのに"), "何もしてないのに"),
        (re.compile("大丈夫だ.問題ない"), "大丈夫だ、問題ない。")
    )
    stop_words = (
        "Squid2",
        ''
    )

    @classmethod
    def _tokenize(cls, text):
        return cls._tokenizer.tokenize(text, cls._mode)

    def __init__(self, raw):
        # self.name = raw["author"]["name"]

        if "emotes" in raw:
            self.emotes = set(e["name"] for e in raw["emotes"]
                              if e["name"] not in self.stop_words)
        else:
            self.emotes = set()
        # self.datetime = datetime.fromtimestamp(int(raw["timestamp"]) // 1000000).replace(tzinfo=TIMEZONE)
        self.datetime = datetime.fromtimestamp(int(raw["timestamp"]) // 1000000)

        self.message = raw["message"]
        self.msg = set()

        message = self.message
        for emote in self.emotes:
            message = message.replace(emote, "")
        for stop in self.stop_words:
            message = message.replace(stop, "")
        message = re.sub(r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+", "", message)

        #
        for pattern, replace in self.pn_patterns:
            match = pattern.findall(message)
            if match:
                self.msg.add(replace)
                if pattern.pattern.startswith('^') and pattern.pattern.endswith('$'):
                    message = ''
                else:
                    for m in match:
                        message = message.replace(m, "")

        #
        for pn in self.pns:
            if pn in message:
                self.msg.add(pn)
                message = message.replace(pn, "")

        #
        message = jaconv.h2z(message)

        # (名詞 or 動詞) (+助動詞)を取り出す
        parts = []
        currentpart = None
        for m in self._tokenize(message):
            part = m.part_of_speech()[0]

            if currentpart:
                if part == "助動詞":
                    parts.append(m.surface())
                else:
                    self.msg.add(''.join(parts))
                    parts = []
                    if part in ("名詞", "動詞"):
                        currentpart = part
                        parts.append(m.surface())
                    else:
                        currentpart = None
            else:
                if part in ("名詞", "動詞"):
                    currentpart = part
                    parts.append(m.surface())

        if parts:
            self.msg.add(''.join(parts))

        #
        kusa = False
        for word in copy(self.msg):
            if set(word) & set(('w', 'ｗ')):
                kusa = True
                self.msg.remove(word)
        if kusa:
            self.msg.add("ｗｗｗ")

        message = message.strip()
        if not self.msg and message:
            self.msg.add(message)

    def __len__(self):
        return len(self.msg)

    @property
    def words(self):
        return self.msg | self.emotes


def _make_messages(raw_message):
    if "name" in raw_message["author"] and raw_message["author"]["name"] == "fossabot":
        return

    for term in EXCLUDE_MESSAGE_TERMS:
        if term in raw_message["message"]:
            return
    return Message(raw_message)


def _parse_chat(paths):
    messages = []
    for p in paths:
        if p.endswith(".gz"):
            opener = gzip.open
        else:
            opener = open

        with opener(p) as f, Pool() as pool:
            j = json.load(f)
            messages += [msg for msg in pool.map(_make_messages, j, len(j) // pool._processes)
                         if msg is not None]

    timeline = []
    currentwindow = messages[0].datetime.replace(microsecond=0) + WINDOW
    _messages = []
    for m in messages:
        if m.datetime <= currentwindow:
            _messages.append(m)
        else:
            timeline.append((currentwindow, *_make_timepoint(_messages)))
            while True:
                currentwindow += WINDOW
                if m.datetime <= currentwindow:
                    _messages = [m]
                    break
                else:
                    timeline.append((currentwindow, 0, Counter()))

    if _messages:
        timeline.append((currentwindow, *_make_timepoint(_messages)))

    return timeline


def _make_timepoint(messages):
    total = len(messages)
    counts = Counter(_ for _ in chain(*(m.words for m in messages)))

    return total, counts


def _load_timeline(paths):
    if os.path.exists(TIMELINE):
        with open(TIMELINE, "rb") as f:
            timeline = pickle.load(f)
    else:
        timeline = _parse_chat(paths)
        with open(TIMELINE, "wb") as f:
            pickle.dump(timeline, f)

    return timeline


def _save_counts(timeline):
    _, _, counters = zip(*timeline)

    counter = Counter()
    for c in counters:
        counter.update(c)

    with open("words.tab", 'w') as f:
        for w, c in sorted(counter.items(), key=itemgetter(1), reverse=True):
            print(w, c, sep='\t', file=f)


def _plot(timeline, normarize, pages):
    scales = False
    if normarize:
        x, totals, _ = tuple(zip(*timeline))

        breaks = [game.startat for game in GAMES]
        breaks = [bisect_left(x, b) for b in breaks]
        breaks = [0] + breaks + [len(x)]

        scales = np.array([])
        totals = moving_average(totals) * PER_SECONDS
        for begin, end in zip(breaks, breaks[1:]):
            max_msgs = max(totals[begin:end])
            scales = np.concatenate((scales, np.ones(end - begin) / max_msgs))

    for npage in range(1, 1 + PAGES):
        if npage not in pages:
            continue

        chunklen = int(len(timeline) / PAGES / ROW)

        fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI)
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        plt.rcParams["savefig.facecolor"] = BACKGROUND_COLOR
        ax = fig.add_axes((0, 0, 1, 1))
        background_image = mpimg.imread(BACKGROUND)
        ax.imshow(background_image)

        plt.subplots_adjust(left=0.07, bottom=0.05, top=0.92)

        for i in range(1, 1 + ROW):
            nrow = i + ROW * (npage - 1)
            f, t = chunklen * (nrow - 1), chunklen * nrow
            x, c, y = zip(*timeline[f:t])
            # _x = tuple(t.replace(tzinfo=None) for t in x)

            ax = fig.add_subplot(ROW, 1, i)
            scale = False if scales is False else scales[f:t]

            _plot_row(ax, x, y, c, i == 1, i == ROW, scale)

        fig.suptitle(f"RTA in Japan Winter 2023 チャット頻出スタンプ・単語 ({npage}/{PAGES})",
                     color=FONT_COLOR, size="x-large")

        desc = "" if scales is False else ", ゲームタイトルごとの最大値=100%"
        ytitle = f"単語 / 分 （同一メッセージ内の重複は除外{desc}）"
        fig.text(0.03, 0.5, ytitle,
                 ha="center", va="center", rotation="vertical", color=FONT_COLOR, size="large")
        fig.savefig(f"{npage}.png", dpi=DPI, transparent=True)
        plt.close()
        print(npage)


def moving_average(x, w=AVR_WINDOW):
    _x = np.convolve(x, np.ones(w), "same") / w
    return _x[:len(x)]


def _plot_row(ax, x, y, total_raw, add_upper_legend, add_lower_legend, scales):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M", tz=TIMEZONE))
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(range(0, 60, 5)))

    if scales is not False:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    else:
        ax.yaxis.set_minor_locator(MultipleLocator(50))

    ax.set_facecolor(FACE_COLOR)

    for axis in ("top", "bottom", "left", "right"):
        ax.spines[axis].set_color(FRAME_COLOR)

    ax.tick_params(colors=FONT_COLOR, which="both")
    ax.set_xlim(x[0], x[-1])
    if scales is not False:
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(0, YMAX)
    # ax.set_ylim(25, 800)
    # ax.set_yscale('log')

    ax.fill_between(x, 0, YMAX, color=BACKGROUND_COLOR, alpha=0.5)

    total = moving_average(total_raw) * PER_SECONDS
    if scales is not False:
        total *= scales
    total = ax.fill_between(x, 0, total, color=MESSAGE_FILL_COLOR,
                            edgecolor=MESSAGE_EDGE_COLOR, linewidth=0.5)

    text_spacing = (x[-1] - x[0]) / 250
    for i, game in enumerate(GAMES):
        annoat = YMAX if scales is False else 1
        if x[0] <= game.startat <= x[-1]:
            ax.axvline(x=game.startat, color=ARROW_COLOR, linestyle=":")
            # ax.annotate(game.name, xy=(game.startat, annoat), xytext=(game.startat, annoat * 0.85), verticalalignment="top",
            #             color=FONT_COLOR, arrowprops=dict(facecolor=ARROW_COLOR, shrink=0.05), ha=game.align)
            ax.text(game.startat, annoat * 0.98, '⭐', verticalalignment="top", horizontalalignment="center",
                    color=FONT_COLOR, fontproperties=emoji_prop)
            ax.text(game.startat + text_spacing * (1 if game.align == "left" else -1),
                    annoat * 0.9, game.name, verticalalignment="top", ha=game.align, color=FONT_COLOR)

    # ys = []
    # labels = []
    # colors = []
    for words, style, color in RTA_EMOTES:
        if isinstance(words, str):
            words = (words, )
        _y = np.fromiter((sum(c[w] for w in words) for c in y), int)
        if not sum(_y):
            continue
        _y = moving_average(_y) * PER_SECONDS
        if scales is not False:
            _y *= scales
        # ys.append(_y)
        # labels.append("\n".join(words))
        # colors.append(color if color else None)
        ax.plot(x, _y, label="\n".join(words), linestyle=style, color=(color if color else None))
    # ax.stackplot(x, ys, labels=labels, colors=colors)

    #
    avr_10min = moving_average(total_raw, FIND_WINDOW) * FIND_WINDOW
    words = Counter()
    for counter in y:
        words.update(counter)
    words = set(k for k, v in words.items() if v >= COUNT_THRESHOLD)
    words -= VOCABULARY

    annotations = []
    for word in words:
        at = []
        _ys = moving_average(np.fromiter((c[word] for c in y), int), FIND_WINDOW) * FIND_WINDOW
        for i, (_y, total_y) in enumerate(zip(_ys, avr_10min)):
            if _y >= total_y * DOMINATION_RATE and _y >= COUNT_THRESHOLD:
                ypoint = _y * PER_SECONDS / FIND_WINDOW * DOMINATION_RATE
                if scales is not False:
                    ypoint *= scales[i]
                at.append((i, ypoint))
        if at:
            at.sort(key=lambda x: x[1])
            at = at[-1]

            if any(c in UNICODE_EMOJI for c in word):
                text = ax.text(x[at[0]], at[1], word, color=FONT_COLOR, fontsize="xx-small", fontproperties=emoji_prop)
            else:
                text = ax.text(x[at[0]], at[1], word, color=FONT_COLOR, fontsize="xx-small")
            annotations.append(text)
    if annotations:
        adjust_text(annotations, only_move={"text": 'x'})

    if add_upper_legend:
        leg = ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        _set_legend(leg)
        frame = leg.get_frame()
        frame.set_facecolor(BACKGROUND_COLOR)
        frame.set_alpha(1)

    if add_lower_legend:
        leg = plt.legend([total], ["メッセージ / 分"], loc=(1.015, 0.4))
        _set_legend(leg)
        msg = "図中の単語は{}秒間で{}%の\nメッセージに含まれていた単語\n({:.1f}メッセージ / 秒 以上のもの)".format(
            FIND_WINDOW, int(DOMINATION_RATE * 100), COUNT_THRESHOLD / FIND_WINDOW
        )
        plt.gcf().text(0.915, 0.06, msg, fontsize="x-small", color=FONT_COLOR)


def _set_legend(leg):
    frame = leg.get_frame()
    frame.set_facecolor(FACE_COLOR)
    frame.set_edgecolor(FRAME_COLOR)
    frame.set_alpha(0.5)

    for text in leg.get_texts():
        text.set_color(FONT_COLOR)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json", nargs="+")
    parser.add_argument("-n", "--normarize", action="store_true")
    parser.add_argument("-p", "--pages", type=int, nargs='*', default=range(1, 1 + PAGES))
    args = parser.parse_args()

    timeline = _load_timeline(args.json)
    _save_counts(timeline)

    _plot(timeline, args.normarize, args.pages)


if __name__ == "__main__":
    _main()
