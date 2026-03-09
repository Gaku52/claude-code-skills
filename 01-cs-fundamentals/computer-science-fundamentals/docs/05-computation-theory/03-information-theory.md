# 情報理論

> シャノンの情報理論は、データ圧縮の限界と通信の信頼性の限界を数学的に明らかにした。
> 情報を定量的に扱うための数学的基盤であり、現代のデータ圧縮・通信・暗号・機械学習すべてに浸透している。

## この章で学ぶこと

- [ ] 情報量（ビット）の数学的定義を理解し、任意の確率分布に対して計算できる
- [ ] エントロピーの意味と、データ圧縮の理論的限界との関係を説明できる
- [ ] シャノンの第一基本定理（情報源符号化定理）の主張を正確に述べられる
- [ ] ハフマン符号の構築手順を実装し、最適前置符号の性質を理解する
- [ ] シャノンの第二基本定理（通信路符号化定理）の直感的理解を得る
- [ ] 誤り検出・訂正符号の原理と代表的手法を比較できる
- [ ] クロスエントロピーとKLダイバージェンスの定義と応用を理解する
- [ ] Python で情報理論の基本概念を実装できる


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [計算複雑性理論](./02-complexity-theory.md) の内容を理解していること

---

## 1. なぜ情報理論が必要か

### 1.1 「情報」を数学で扱う意味

日常語としての「情報」は曖昧である。「今日は天気がいい」という文と「明日、隕石が地球に衝突する」という文では、直感的に後者の方が「情報量が大きい」と感じる。しかし、この直感を工学に活かすには、情報を定量的に測定する枠組みが必要になる。

1948年、クロード・シャノン（Claude Shannon）は論文 "A Mathematical Theory of Communication" において、この問題に対する決定的な解答を与えた。シャノンは「情報」を確率論の言葉で定式化し、以下の根本的な問いに数学的解答を与えた。

1. **データをどこまで圧縮できるか？**（情報源符号化定理）
2. **雑音のある通信路でどこまで信頼性を保てるか？**（通信路符号化定理）

これら2つの定理は、工学のあらゆる分野に影響を与え続けている。

### 1.2 情報理論が関わる技術領域

```
+------------------------------------------------------------------+
|                     情報理論の応用領域                              |
+------------------------------------------------------------------+
|                                                                    |
|  データ圧縮          通信工学          暗号理論                     |
|  ┌──────────┐     ┌──────────┐     ┌──────────┐                  |
|  │ ZIP/gzip │     │ 5G / WiFi│     │ AES / RSA│                  |
|  │ JPEG/PNG │     │ 誤り訂正 │     │ エントロ │                  |
|  │ MP3/AAC  │     │ LDPC符号 │     │ ピーソース│                  |
|  │ H.264    │     │ Polar符号│     │ 鍵の安全性│                  |
|  └──────────┘     └──────────┘     └──────────┘                  |
|                                                                    |
|  機械学習            自然言語処理      情報検索                     |
|  ┌──────────┐     ┌──────────┐     ┌──────────┐                  |
|  │ 損失関数 │     │ パープレ │     │ TF-IDF   │                  |
|  │ 交差エント│     │ キシティ │     │ 情報利得 │                  |
|  │ ロピー   │     │ 言語モデル│     │ 相互情報量│                  |
|  │ KLダイバー│     │ 評価指標 │     │ 特徴選択 │                  |
|  │ ジェンス │     │          │     │          │                  |
|  └──────────┘     └──────────┘     └──────────┘                  |
+------------------------------------------------------------------+
```

### 1.3 歴史的背景

情報理論の成立には、いくつかの先駆的研究がある。

| 年 | 人物 | 貢献 |
|------|------|------|
| 1928 | ハートレー (Hartley) | 情報の対数的尺度を提案 |
| 1948 | シャノン (Shannon) | 情報理論の体系的構築。エントロピー、通信路容量の定式化 |
| 1949 | シャノン (Shannon) | 暗号理論への応用（"Communication Theory of Secrecy Systems"） |
| 1951 | ハフマン (Huffman) | 最適前置符号の構築アルゴリズム |
| 1952 | ハミング (Hamming) | ハミング符号（誤り訂正符号の先駆） |
| 1960 | リード & ソロモン | リードソロモン符号（バースト誤り訂正） |
| 1977 | レンペル & ジフ | LZ77 アルゴリズム（辞書式圧縮の基盤） |
| 1993 | ベルー, グラヴュー, ティティマ | ターボ符号（シャノン限界に迫る誤り訂正） |
| 2009 | アリカン (Arikan) | Polar符号（理論的にシャノン限界を達成） |

シャノンの論文は、情報を「意味」から切り離し、純粋に統計的な量として扱うという革命的な発想に基づいていた。ある記号列が伝える意味とは無関係に、その記号列がどの程度「予測しにくいか」だけに着目する。この抽象化により、圧縮・通信・暗号といった異なる分野を統一的に扱う理論が生まれた。

---

## 2. 情報量とエントロピー

### 2.1 自己情報量

確率 P(x) で生じる事象 x の **自己情報量**（self-information）は次のように定義される。

```
I(x) = -log₂(P(x))   [単位: ビット]
```

この定義は以下の直感に合致する。

- **確実な事象は情報を持たない**: P(x) = 1 ならば I(x) = 0
- **稀な事象ほど情報量が大きい**: P(x) が小さいほど I(x) は大きい
- **独立事象の情報量は加法的**: I(x, y) = I(x) + I(y)（x, y が独立のとき）

```
自己情報量の例:

  事象                確率P(x)    情報量 I(x) = -log₂(P(x))
  ─────────────────────────────────────────────────────────
  公平なコイン（表）    0.5         1.0 ビット
  公平なサイコロ（1）   1/6         2.585 ビット
  確実な事象            1.0         0.0 ビット
  稀な事象              0.01        6.644 ビット
  非常に稀な事象        0.001       9.966 ビット

  グラフ（P vs I）:
  I(x)
  10 |*
   8 | *
   6 |   *
   4 |      *
   2 |            *
   0 |________________________*___
     0   0.2   0.4   0.6   0.8  1.0  P(x)

  → P(x)が0に近づくと情報量は無限大に発散
  → P(x)が1に近づくと情報量は0に収束
```

対数の底を変えると単位が変わる。底2ならビット（bit）、底 e なら **ナット**（nat）、底10なら **ハートレー**（hartley）である。情報工学では慣例的にビットが用いられる。

### 2.2 エントロピー

離散確率変数 X が確率分布 {p₁, p₂, ..., pₙ} に従うとき、**シャノンエントロピー**（Shannon entropy）は次のように定義される。

```
H(X) = -Σᵢ pᵢ log₂(pᵢ)   [単位: ビット]
```

エントロピーは「平均的な情報量」すなわち「情報源の不確実性の尺度」である。

### 2.3 エントロピーの性質

エントロピーには以下の重要な性質がある。

**性質1: 非負性**
```
H(X) ≥ 0
```
等号は X が確定的（ある事象の確率が 1）のときに成立する。

**性質2: 最大値**
```
H(X) ≤ log₂(n)   （n は X の取りうる値の数）
```
等号は X が一様分布に従うときに成立する。一様分布は「最もランダムな」分布であり、圧縮が最も困難な状態に対応する。

**性質3: 連鎖律**
```
H(X, Y) = H(X) + H(Y|X)
```
同時エントロピーは、X のエントロピーと、X を知った上での Y の条件付きエントロピーの和に分解される。

**性質4: 凹性（concavity）**
エントロピーは確率分布の凹関数である。すなわち、2つの分布を混合すると、エントロピーは増加する（または変わらない）。

### 2.4 エントロピーの計算例

```
例1: 公平なコイン投げ
  P(表) = 0.5, P(裏) = 0.5
  H = -0.5 × log₂(0.5) - 0.5 × log₂(0.5)
    = -0.5 × (-1) - 0.5 × (-1)
    = 0.5 + 0.5 = 1.0 ビット

  → 1回の試行あたり1ビットの情報が得られる
  → 1ビットで完全に表現可能

例2: 偏ったコイン
  P(表) = 0.9, P(裏) = 0.1
  H = -0.9 × log₂(0.9) - 0.1 × log₂(0.1)
    = -0.9 × (-0.152) - 0.1 × (-3.322)
    = 0.137 + 0.332 = 0.469 ビット

  → 公平なコインより情報量が少ない（予測しやすいため）
  → 理論上は0.469ビット/試行まで圧縮可能

例3: 英語テキスト（26文字 + 空白）
  各文字の出現確率は不均一（'e'が約12.7%, 'z'が約0.07%）
  一様分布ならば: log₂(27) ≈ 4.75 ビット/文字
  実際のエントロピー: H ≈ 4.11 ビット/文字（1次統計量のみ）
  文脈を考慮すると: H ≈ 1.0〜1.5 ビット/文字

  → ASCII 8ビットから大幅に圧縮可能
```

### 2.5 Python による情報量・エントロピーの計算

```python
"""
情報量とエントロピーの計算
"""
import math
from collections import Counter
from typing import Dict, List


def self_information(probability: float) -> float:
    """
    自己情報量を計算する。

    Args:
        probability: 事象の確率 (0 < p <= 1)

    Returns:
        自己情報量（ビット単位）

    Raises:
        ValueError: 確率が範囲外の場合
    """
    if not (0 < probability <= 1):
        raise ValueError(f"確率は (0, 1] の範囲: {probability}")
    return -math.log2(probability)


def entropy(probabilities: List[float]) -> float:
    """
    シャノンエントロピーを計算する。

    Args:
        probabilities: 確率分布（合計が1になるリスト）

    Returns:
        エントロピー（ビット単位）

    Raises:
        ValueError: 確率の合計が1でない場合
    """
    if abs(sum(probabilities) - 1.0) > 1e-9:
        raise ValueError(f"確率の合計が1でない: {sum(probabilities)}")

    h = 0.0
    for p in probabilities:
        if p > 0:
            h -= p * math.log2(p)
    return h


def text_entropy(text: str) -> float:
    """
    テキストの1次エントロピーを計算する。

    各文字の出現頻度から確率分布を推定し、エントロピーを算出する。

    Args:
        text: 入力テキスト

    Returns:
        1文字あたりのエントロピー（ビット単位）
    """
    if not text:
        return 0.0

    counter = Counter(text)
    total = len(text)
    probs = [count / total for count in counter.values()]
    return entropy(probs)


def entropy_analysis(text: str) -> Dict[str, float]:
    """
    テキストのエントロピー分析を行う。

    Args:
        text: 入力テキスト

    Returns:
        分析結果を含む辞書
    """
    if not text:
        return {"entropy": 0.0, "max_entropy": 0.0, "redundancy": 0.0}

    counter = Counter(text)
    n_symbols = len(counter)
    total = len(text)

    h = text_entropy(text)
    h_max = math.log2(n_symbols) if n_symbols > 1 else 0.0
    redundancy = 1.0 - (h / h_max) if h_max > 0 else 0.0

    return {
        "entropy": h,
        "max_entropy": h_max,
        "redundancy": redundancy,
        "n_symbols": n_symbols,
        "total_chars": total,
        "theoretical_min_bits": h * total,
        "ascii_bits": 8 * total,
        "compression_ratio": h / 8.0,
    }


# === 動作確認 ===
if __name__ == "__main__":
    # 自己情報量
    print("=== 自己情報量 ===")
    for p in [0.5, 1/6, 0.01, 0.001, 1.0]:
        print(f"  P={p:.4f}  I={self_information(p):.3f} ビット")

    # エントロピー
    print("\n=== エントロピー ===")
    print(f"  公平なコイン: {entropy([0.5, 0.5]):.3f} ビット")
    print(f"  偏ったコイン (0.9/0.1): {entropy([0.9, 0.1]):.3f} ビット")
    print(f"  公平なサイコロ: {entropy([1/6]*6):.3f} ビット")
    print(f"  確定的: {entropy([1.0]):.3f} ビット")

    # テキストのエントロピー分析
    sample_text = (
        "information theory is a mathematical framework for "
        "quantifying information content and communication"
    )
    print(f"\n=== テキスト分析 ===")
    analysis = entropy_analysis(sample_text)
    for key, value in analysis.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
```

上記のプログラムを実行すると、以下のような出力が想定される。

```
=== 自己情報量 ===
  P=0.5000  I=1.000 ビット
  P=0.1667  I=2.585 ビット
  P=0.0100  I=6.644 ビット
  P=0.0010  I=9.966 ビット
  P=1.0000  I=0.000 ビット

=== エントロピー ===
  公平なコイン: 1.000 ビット
  偏ったコイン (0.9/0.1): 0.469 ビット
  公平なサイコロ: 2.585 ビット
  確定的: 0.000 ビット

=== テキスト分析 ===
  entropy: 3.937
  max_entropy: 4.585
  redundancy: 0.141
  n_symbols: 24
  total_chars: 95
  theoretical_min_bits: 374.010
  ascii_bits: 760
  compression_ratio: 0.492
```

### 2.6 条件付きエントロピーと相互情報量

2つの確率変数 X, Y の関係を記述する概念を整理する。

```
同時エントロピー:      H(X, Y) = -Σ P(x,y) log₂ P(x,y)
条件付きエントロピー:  H(Y|X)  = -Σ P(x,y) log₂ P(y|x)
相互情報量:            I(X;Y)  = H(X) + H(Y) - H(X,Y)
                                = H(X) - H(X|Y)
                                = H(Y) - H(Y|X)

エントロピーのベン図:

  ┌───────────────────────────────────┐
  │               H(X,Y)              │
  │  ┌──────────┬──────────┐          │
  │  │          │          │          │
  │  │  H(X|Y)  │  I(X;Y)  │ H(Y|X)  │
  │  │          │          │          │
  │  └──────────┴──────────┘          │
  │     H(X)         H(Y)             │
  └───────────────────────────────────┘

  H(X,Y) = H(X|Y) + I(X;Y) + H(Y|X)
  I(X;Y) = H(X) + H(Y) - H(X,Y)

  直感的解釈:
  - H(X|Y): Y を知っても残る X の不確実性
  - I(X;Y): X と Y が共有する情報量
  - I(X;Y) = 0 ⟺ X と Y が独立
```

相互情報量は、特徴選択・独立性検定・情報利得（決定木）など、幅広い応用を持つ。

---

## 3. 情報源符号化（シャノン第一基本定理）

### 3.1 情報源符号化定理の主張

**シャノンの第一基本定理（情報源符号化定理, 1948）:**

> 無記憶情報源 X のエントロピーを H(X) とする。X から生成される記号列を可逆的に符号化するとき、1記号あたりの平均符号長 L は以下を満たす。
>
> **H(X) ≤ L < H(X) + 1**
>
> ただし、n 個の記号をまとめてブロック符号化すれば、1記号あたりの平均符号長を H(X) に任意に近づけることができる。

この定理は2つの主張を含んでいる。

1. **達成不能性**: エントロピー H(X) より短い平均符号長で可逆圧縮することは不可能である。
2. **達成可能性**: エントロピー H(X) に任意に近い平均符号長を実現する符号が存在する。

### 3.2 前置符号（Prefix Code）

**前置符号**（prefix-free code）とは、どの符号語も他の符号語の接頭辞になっていない符号である。この性質により、符号の区切りを示す特別な記号なしに一意復号が可能になる。

```
前置符号の例:

  記号    符号語
  ─────────────
  A       0
  B       10
  C       110
  D       111

  復号の一意性:
  符号列 "010110111" を左から読む:
    0    → A
    10   → B
    110  → C
    111  → D
  結果: "ABCD"

  非前置符号の反例:
  記号    符号語
  ─────────────
  A       0
  B       01      ← A の符号語 "0" が B の接頭辞
  C       011

  "011" は "A, B" とも "C" とも解釈できる → 一意復号不可能
```

**クラフトの不等式**: 符号語長が d₁, d₂, ..., dₙ の前置符号が存在するための必要十分条件は次の通りである。

```
Σᵢ 2^(-dᵢ) ≤ 1
```

### 3.3 ハフマン符号

ハフマン符号（Huffman, 1952）は、与えられた確率分布に対して最適な前置符号を構築するアルゴリズムである。「最適」とは、平均符号長が前置符号の中で最小であることを意味する。

**アルゴリズム:**

```
ハフマン木の構築手順:

  入力: 各記号とその確率
  出力: 最適前置符号

  1. 各記号を確率値を持つ葉ノードとして初期化
  2. 確率が最小の2つのノードを取り出す
  3. それらを子とする新しい内部ノードを作成し、
     確率は2つの子の確率の和とする
  4. 新しいノードをリストに戻す
  5. ノードが1つになるまで 2-4 を繰り返す
  6. 根から各葉へのパスを辿り、
     左の枝に0、右の枝に1を割り当てて符号語とする

  例: {A:0.4, B:0.3, C:0.2, D:0.1}

  ステップ1: C(0.2)とD(0.1)を結合 → CD
  ステップ2: B(0.3)とCDを結合 → BCD
  ステップ3: A(0.4)とBCDを結合 → ABCD

  ハフマン木:
              [1.0]
             /     \
           0/       \1
          A(0.4)   [0.6]
                   /     \
                 0/       \1
               B(0.3)   [0.3]
                         /     \
                       0/       \1
                     C(0.2)   D(0.1)

  符号割り当て:
  A → 0      (1ビット)
  B → 10     (2ビット)
  C → 110    (3ビット)
  D → 111    (3ビット)

  平均符号長:
  L = 0.4×1 + 0.3×2 + 0.2×3 + 0.1×3
    = 0.4 + 0.6 + 0.6 + 0.3
    = 1.9 ビット

  エントロピー:
  H = -(0.4×log₂0.4 + 0.3×log₂0.3 + 0.2×log₂0.2 + 0.1×log₂0.1)
    ≈ 1.846 ビット

  効率: H/L ≈ 1.846/1.9 ≈ 97.2%
```

### 3.4 算術符号

ハフマン符号は各記号に整数ビット長の符号語を割り当てるため、確率が2のべき乗でない場合に効率が落ちる。**算術符号**（arithmetic coding）は、メッセージ全体を [0, 1) 区間内の1つの数値として表現することで、この制約を回避する。

```
算術符号化の概念:

  メッセージ "BAC" を符号化する
  確率: A=0.4, B=0.3, C=0.3

  初期区間: [0, 1)

  B を符号化: B の範囲は [0.4, 0.7)
    → 区間 [0.4, 0.7)

  A を符号化: A の範囲は [0, 0.4)
    → 新しい区間: [0.4, 0.4 + 0.3×0.4) = [0.4, 0.52)

  C を符号化: C の範囲は [0.7, 1.0)
    → 新しい区間: [0.4 + 0.12×0.7, 0.4 + 0.12×1.0) = [0.484, 0.52)

  最終区間 [0.484, 0.52) 内の任意の値で復号可能
  例えば 0.5 を二進表現すると → "1" の後に必要な精度分のビット

  長所: ブロック符号化なしでエントロピーに漸近
  短所: 計算量が大きい、特許問題（歴史的に）
```

### 3.5 ANS（Asymmetric Numeral Systems）

ANS は Jarek Duda（2009年）が提案した比較的新しいエントロピー符号化方式で、算術符号に匹敵する圧縮率を、ハフマン符号に匹敵する速度で達成する。Facebook が開発した zstd（Zstandard）で採用されている。

```
エントロピー符号化の比較:

  手法          圧縮率      速度      複雑さ    採用例
  ──────────────────────────────────────────────────────
  ハフマン符号  良好        高速      低       DEFLATE, JPEG
  算術符号      最良        低速      高       JPEG2000, H.265
  ANS (rANS)   最良        高速      中       zstd, LZFSE
  ANS (tANS)   最良        最高速    中       zstd, Brotli

  圧縮率: ハフマン ≤ ANS ≈ 算術符号 ≤ シャノン限界
```

---

## 4. 通信路符号化（シャノン第二基本定理）

### 4.1 通信路モデル

通信路とは、送信者から受信者へ情報を伝送する経路を抽象化したモデルである。現実の通信路には雑音（ノイズ）が存在し、送信したデータが正しく届かない可能性がある。

```
通信の一般モデル（シャノンモデル）:

  情報源 → [符号器] → [通信路] → [復号器] → 受信者
                        ↑ 雑音

  情報源符号化: データ圧縮（冗長性の除去）
  通信路符号化: 誤り訂正のための冗長性の付加
  → 情報源符号化と通信路符号化は分離して最適化可能
    （分離定理）
```

最も基本的な通信路モデルは **二元対称通信路**（Binary Symmetric Channel, BSC）である。

```
二元対称通信路 (BSC):

  送信                 受信
  ─────                ─────
         1-p
  0 ─────────────── 0
    \              /
     \   p      p /
      \          /
       ×        ×
      /          \
     /   p      p \
    /              \
  1 ─────────────── 1
         1-p

  p: ビット反転確率（誤り率）
  1-p: 正しく伝送される確率

  p = 0: 完全な通信路（雑音なし）
  p = 0.5: 完全にランダム（通信不能）
  p = 1: 全ビット反転（反転すれば完全な通信路）
```

### 4.2 通信路容量

**通信路容量** C は、その通信路を通じて信頼性のある通信が可能な最大速度（ビット/使用）である。

```
通信路容量の定義:

  C = max_{p(x)} I(X;Y)

  BSC の場合:
  C = 1 - H(p) = 1 + p×log₂(p) + (1-p)×log₂(1-p)

  加法的白色ガウス雑音通信路（AWGN）の場合:
  C = (1/2) × log₂(1 + S/N)  [ビット/使用]

  ここで S/N は信号対雑音比（SNR）

  通信路容量の例:

  BSC (p=0.1):  C = 1 - H(0.1) ≈ 1 - 0.469 = 0.531 ビット/使用
  BSC (p=0.01): C = 1 - H(0.01) ≈ 1 - 0.081 = 0.919 ビット/使用
  BSC (p=0.5):  C = 1 - H(0.5) = 1 - 1.0 = 0.0 ビット/使用

  AWGN (SNR=10, つまり10dB):
  C = 0.5 × log₂(1 + 10) ≈ 1.73 ビット/使用
```

### 4.3 シャノンの第二基本定理（通信路符号化定理）

**シャノンの第二基本定理（通信路符号化定理, 1948）:**

> 通信路容量 C の通信路において、任意の通信速度 R < C に対して、復号誤り率を任意に小さくできる符号が存在する。
> 逆に、R > C の通信速度では、復号誤り率を0に近づけることは不可能である。

この定理は「存在定理」であり、具体的な符号の構成法を示していない点が重要である。シャノンはランダム符号帳を用いた確率的議論によってこの定理を証明した。実用的な符号（ターボ符号、LDPC符号、Polar符号）がシャノン限界に近い性能を達成するまでには、定理の発表から約50年を要した。

```
シャノン限界への収束（歴史的推移）:

  通信路容量からのギャップ（dB）
  10 |  ● ハミング符号 (1950)
   8 |
   6 |     ● BCH符号 (1960)
   4 |        ● リードソロモン (1960)
   3 |
   2 |              ● 連接符号 (1970)
   1 |
   0.5|                    ● ターボ符号 (1993)
   0.3|                       ● LDPC符号 (再発見, 1996)
   0.1|                          ● Polar符号 (2009)
   0  |────────────────────────────── シャノン限界
      1950    1960    1970    1990    2000    2010
```

### 4.4 誤り検出と誤り訂正

#### パリティ検査

最も単純な誤り検出方式はパリティビットの付加である。

```
パリティビット:

  データ: 1011001
  偶数パリティ: 1の個数が偶数になるようにビットを追加
  → 10110011 (パリティビット = 1)

  受信: 10110111 (1ビットの誤り)
  パリティ検査: 1の個数 = 7 (奇数) → 誤り検出

  制限:
  - 1ビットの誤りは検出可能
  - 2ビットの誤りは検出不能
  - 誤りの位置は特定不能（訂正不能）
```

#### ハミング符号

ハミング符号（Hamming, 1950）は、1ビットの誤り訂正と2ビットの誤り検出を可能にする符号である。

```
ハミング(7,4)符号:

  4ビットのデータ → 7ビットの符号語（3ビットの冗長ビットを追加）

  位置:     1  2  3  4  5  6  7
  種類:     p1 p2 d1 p3 d2 d3 d4
  (p=パリティ, d=データ)

  パリティの計算:
  p1: 位置1,3,5,7をカバー（二進表現の最下位ビットが1）
  p2: 位置2,3,6,7をカバー（二進表現の第2ビットが1）
  p3: 位置4,5,6,7をカバー（二進表現の第3ビットが1）

  例: データ = 1011
  d1=1, d2=0, d3=1, d4=1

  p1 = d1 ⊕ d2 ⊕ d4 = 1 ⊕ 0 ⊕ 1 = 0
  p2 = d1 ⊕ d3 ⊕ d4 = 1 ⊕ 1 ⊕ 1 = 1
  p3 = d2 ⊕ d3 ⊕ d4 = 0 ⊕ 1 ⊕ 1 = 0

  符号語: 0 1 1 0 0 1 1

  誤り訂正: 位置5のビットが反転したとする
  受信: 0 1 1 0 1 1 1
                ^
  シンドローム計算:
  s1 = p1 ⊕ d1 ⊕ d2 ⊕ d4 = 0 ⊕ 1 ⊕ 1 ⊕ 1 = 1
  s2 = p2 ⊕ d1 ⊕ d3 ⊕ d4 = 1 ⊕ 1 ⊕ 1 ⊕ 1 = 0
  s3 = p3 ⊕ d2 ⊕ d3 ⊕ d4 = 0 ⊕ 1 ⊕ 1 ⊕ 1 = 1

  シンドローム = s3 s2 s1 = 101 (二進) = 5 (十進)
  → 位置5のビットを反転して訂正
```

#### リードソロモン符号

リードソロモン符号（Reed-Solomon, 1960）は、**バースト誤り**（連続するビットの誤り）の訂正に優れた符号である。

```
リードソロモン符号の特徴:

  RS(n, k): n シンボルの符号語、k シンボルのデータ
  訂正能力: t = (n - k) / 2 シンボルの誤りを訂正可能

  応用例:
  ┌─────────────────┬──────────────┬─────────────┐
  │ 応用             │ パラメータ   │ 訂正能力     │
  ├─────────────────┼──────────────┼─────────────┤
  │ CD               │ RS(32,28)    │ 2シンボル   │
  │ DVD              │ RS(208,192)  │ 8シンボル   │
  │ QRコード         │ 4段階        │ 7%〜30%     │
  │ 深宇宙通信       │ RS(255,223)  │ 16シンボル  │
  │ デジタル放送     │ RS(204,188)  │ 8シンボル   │
  └─────────────────┴──────────────┴─────────────┘

  QRコードの誤り訂正レベル:
  L: 約 7% のコードワードを復元可能
  M: 約15% のコードワードを復元可能
  Q: 約25% のコードワードを復元可能
  H: 約30% のコードワードを復元可能
```

#### 現代の誤り訂正符号

```
現代の誤り訂正符号の比較:

  符号           年    シャノン限界   計算量     採用例
                       からのギャップ
  ─────────────────────────────────────────────────────
  ハミング符号   1950  大きい         O(n)       メモリ ECC
  BCH符号        1960  中程度         O(n²)      フラッシュメモリ
  RS符号         1960  中程度         O(n²)      CD/DVD, QR
  畳み込み符号   1955  中程度         O(2^k)     旧世代通信
  ターボ符号     1993  ~0.5 dB        O(n log n) 3G/4G
  LDPC符号       1996  ~0.04 dB       O(n)       Wi-Fi 6, 5G
  Polar符号      2009  0 dB(理論)     O(n log n) 5G制御チャネル

  → ターボ符号の発見は「符号化理論のブレイクスルー」と呼ばれた
  → LDPC は Gallager(1962) が提案し、MacKay(1996) が再発見
  → Polar 符号は初めて「構成的に」シャノン限界を達成
```

---

## 5. データ圧縮の実践

### 5.1 可逆圧縮と非可逆圧縮

データ圧縮は大きく2つに分類される。

```
データ圧縮の分類:

  ┌───────────────────────────────────────────┐
  │           データ圧縮                       │
  ├─────────────────────┬─────────────────────┤
  │   可逆圧縮          │   非可逆圧縮         │
  │   (Lossless)        │   (Lossy)            │
  ├─────────────────────┼─────────────────────┤
  │ 元のデータを完全に  │ 元のデータの近似を   │
  │ 復元可能            │ 復元                 │
  ├─────────────────────┼─────────────────────┤
  │ ZIP, gzip, bzip2    │ JPEG, MP3, H.264     │
  │ PNG, FLAC           │ AAC, HEVC, AV1       │
  │ zstd, brotli        │ Opus, WebP           │
  ├─────────────────────┼─────────────────────┤
  │ 圧縮率: 2:1〜10:1   │ 圧縮率: 10:1〜100:1  │
  │ (データ依存)        │ (品質設定に依存)     │
  ├─────────────────────┼─────────────────────┤
  │ 用途: テキスト,      │ 用途: 画像, 音声,     │
  │ プログラム, DB      │ 動画                 │
  └─────────────────────┴─────────────────────┘
```

### 5.2 LZ77 アルゴリズム

LZ77（Lempel-Ziv, 1977）は辞書式圧縮の基礎であり、gzip, DEFLATE, zstd などの現代的な圧縮アルゴリズムの核心部分である。

```
LZ77 の基本原理:

  スライディングウィンドウ内の過去のデータから、
  現在位置以降と一致する最長の部分文字列を見つけ、
  (距離, 長さ, 次の文字) の三つ組で表現する。

  例: "ABRACADABRA" を圧縮

  位置  入力文字  検索バッファ  一致       出力
  ─────────────────────────────────────────────
  0     A         (空)          なし       (0, 0, 'A')
  1     B         A             なし       (0, 0, 'B')
  2     R         AB            なし       (0, 0, 'R')
  3     A         ABR           A(距離3)   (3, 1, 'C')
  5     A         ABRAC         A(距離5)   (5, 1, 'D')
  7     A         ABRACAD       ABRA(距離7)(7, 4, '\0')

  圧縮の直感:
  "ABRACADABRA" (11文字) → 6つのトークン
  繰り返しパターンが多いほど圧縮率が向上する
```

### 5.3 DEFLATE アルゴリズム

DEFLATE は LZ77 とハフマン符号を組み合わせた圧縮アルゴリズムであり、ZIP, gzip, PNG で使用されている。RFC 1951 で標準化されている。

```
DEFLATE の処理フロー:

  入力データ
      │
      ▼
  ┌──────────┐
  │  LZ77    │  重複パターンの検出と参照に置換
  │  圧縮    │  (距離, 長さ) ペアの生成
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │ ハフマン  │  リテラル/長さ/距離の
  │ 符号化   │  頻度に基づく符号化
  └────┬─────┘
       │
       ▼
  圧縮データ

  DEFLATE ブロック構成:
  - 非圧縮ブロック: そのままコピー
  - 固定ハフマン: 定義済みハフマン木を使用
  - 動的ハフマン: データから最適なハフマン木を構築
```

### 5.4 現代の圧縮アルゴリズム比較

```
  アルゴリズム  年    圧縮率    圧縮速度   展開速度   採用例
  ──────────────────────────────────────────────────────────
  gzip/DEFLATE  1993  普通      普通       高速      HTTP, ZIP
  bzip2         1996  良好      低速       低速      UNIX圧縮
  LZ4           2011  低い      最高速     最高速    ファイルシステム
  Brotli        2015  良好      低速       高速      HTTPS圧縮
  zstd          2016  良好      高速       最高速    Facebook, Linux
  ──────────────────────────────────────────────────────────

  選択の指針:
  - 速度重視: LZ4 or zstd (低圧縮レベル)
  - 圧縮率重視: zstd (高圧縮レベル) or Brotli
  - 互換性重視: gzip (最も広くサポート)
  - リアルタイム用途: LZ4 (ゲーム, データベース)
```

### 5.5 Python による LZ77 圧縮の実装

```python
"""
LZ77 圧縮・展開の実装

辞書式圧縮の基本概念を示すための教育的な実装。
実用的な圧縮にはzlibやzstdの使用を推奨する。
"""
from typing import List, Tuple


class LZ77Compressor:
    """LZ77 圧縮アルゴリズムの実装"""

    def __init__(self, window_size: int = 256, lookahead_size: int = 15):
        """
        Args:
            window_size: スライディングウィンドウのサイズ
            lookahead_size: 先読みバッファのサイズ
        """
        self.window_size = window_size
        self.lookahead_size = lookahead_size

    def compress(self, data: str) -> List[Tuple[int, int, str]]:
        """
        LZ77 でデータを圧縮する。

        Args:
            data: 入力文字列

        Returns:
            (距離, 長さ, 次の文字) のトークンリスト
        """
        tokens: List[Tuple[int, int, str]] = []
        pos = 0

        while pos < len(data):
            best_distance = 0
            best_length = 0

            # 検索バッファの範囲
            search_start = max(0, pos - self.window_size)

            # 先読みバッファの範囲
            lookahead_end = min(pos + self.lookahead_size, len(data))

            # 最長一致を検索
            for i in range(search_start, pos):
                length = 0
                while (pos + length < lookahead_end
                       and data[i + length] == data[pos + length]):
                    length += 1
                    # 検索バッファの末尾を超える場合の処理
                    if i + length >= pos:
                        break

                if length > best_length:
                    best_distance = pos - i
                    best_length = length

            # トークンの生成
            if best_length > 0 and pos + best_length < len(data):
                next_char = data[pos + best_length]
                tokens.append((best_distance, best_length, next_char))
                pos += best_length + 1
            else:
                tokens.append((0, 0, data[pos]))
                pos += 1

        return tokens

    def decompress(self, tokens: List[Tuple[int, int, str]]) -> str:
        """
        LZ77 トークン列を展開する。

        Args:
            tokens: (距離, 長さ, 次の文字) のトークンリスト

        Returns:
            復元された文字列
        """
        result: List[str] = []

        for distance, length, next_char in tokens:
            if distance > 0 and length > 0:
                start = len(result) - distance
                for i in range(length):
                    result.append(result[start + i])
            result.append(next_char)

        return "".join(result)


# === 動作確認 ===
if __name__ == "__main__":
    compressor = LZ77Compressor(window_size=32, lookahead_size=8)

    test_cases = [
        "ABRACADABRA",
        "AAAAAAAAA",
        "ABCABCABCABC",
        "HELLO WORLD",
    ]

    for text in test_cases:
        tokens = compressor.compress(text)
        restored = compressor.decompress(tokens)

        print(f"元のデータ:   '{text}' ({len(text)}文字)")
        print(f"トークン数:   {len(tokens)}")
        print(f"トークン:     {tokens}")
        print(f"復元データ:   '{restored}'")
        print(f"復元一致:     {text == restored}")
        print()
```

想定される出力:

```
元のデータ:   'ABRACADABRA' (11文字)
トークン数:   6
トークン:     [(0, 0, 'A'), (0, 0, 'B'), (0, 0, 'R'), (3, 1, 'C'), (5, 1, 'D'), (7, 4, 'A')]
復元データ:   'ABRACADABRA'
復元一致:     True

元のデータ:   'AAAAAAAAA' (9文字)
トークン数:   2
トークン:     [(0, 0, 'A'), (1, 7, 'A')]
復元データ:   'AAAAAAAAA'
復元一致:     True

元のデータ:   'ABCABCABCABC' (12文字)
トークン数:   4
トークン:     [(0, 0, 'A'), (0, 0, 'B'), (0, 0, 'C'), (3, 8, 'C')]
復元データ:   'ABCABCABCABC'
復元一致:     True

元のデータ:   'HELLO WORLD' (11文字)
トークン数:   10
トークン:     [(0, 0, 'H'), (0, 0, 'E'), (0, 0, 'L'), (1, 1, 'O'), (0, 0, ' '), (0, 0, 'W'), (0, 0, 'O'), (0, 0, 'R'), (0, 0, 'L'), (0, 0, 'D')]
復元データ:   'HELLO WORLD'
復元一致:     True
```

---

## 6. 情報理論の応用

### 6.1 クロスエントロピー

真の確率分布 P に対して、推定分布 Q を用いて符号化したときの平均符号長が **クロスエントロピー** H(P, Q) である。

```
クロスエントロピーの定義:

  H(P, Q) = -Σ P(x) × log₂(Q(x))

  性質:
  - H(P, Q) ≥ H(P)（ギブスの不等式）
  - H(P, Q) = H(P) ⟺ P = Q
  - H(P, Q) ≠ H(Q, P) 一般には非対称

  直感的解釈:
  真の分布 P からサンプリングされたデータを、
  分布 Q に基づく符号で符号化したときの平均ビット数。
  P ≠ Q ならば、H(P) より多くのビットが必要になる。
```

機械学習において、クロスエントロピーは**損失関数**として広く用いられる。分類問題では、正解ラベルの分布 P（one-hot）とモデルの予測分布 Q のクロスエントロピーを最小化する。

```
分類問題でのクロスエントロピー損失:

  正解ラベル（one-hot）: P = [0, 0, 1, 0, 0]  (クラス3が正解)
  モデルの予測:          Q = [0.05, 0.1, 0.7, 0.1, 0.05]

  H(P, Q) = -1 × log₂(0.7) ≈ 0.515 ビット

  予測が悪い場合:
  モデルの予測:          Q = [0.2, 0.2, 0.2, 0.2, 0.2]
  H(P, Q) = -1 × log₂(0.2) ≈ 2.322 ビット

  → 予測が正解に近いほどクロスエントロピーは小さい
  → 完璧な予測ならば H(P, Q) = 0
```

### 6.2 KLダイバージェンス

**KLダイバージェンス**（Kullback-Leibler divergence）は、2つの確率分布の「距離」を測る非対称な指標である。

```
KLダイバージェンスの定義:

  D_KL(P || Q) = Σ P(x) × log₂(P(x) / Q(x))
               = H(P, Q) - H(P)

  性質:
  - D_KL(P || Q) ≥ 0（ギブスの不等式）
  - D_KL(P || Q) = 0 ⟺ P = Q
  - D_KL(P || Q) ≠ D_KL(Q || P)（非対称、距離ではない）
  - D_KL(P || Q) は P を Q で近似する「情報損失量」

  直感的解釈:
  真の分布 P のデータを、Q に基づく符号で符号化したときに、
  最適符号（P に基づく符号）と比べて余分に必要になるビット数。
```

### 6.3 KLダイバージェンスの応用

```
KLダイバージェンスの主要な応用:

  1. 変分推論（VAE）:
     変分下界 ELBO の最大化
     = E[log p(x|z)] - D_KL(q(z|x) || p(z))
     事前分布 p(z) に変分事後分布 q(z|x) を近づける

  2. 方策勾配法（強化学習）:
     TRPO/PPO: 方策更新のKLダイバージェンスを制約
     → 学習の安定化

  3. 知識蒸留:
     大きなモデル(教師)の出力分布を
     小さなモデル(生徒)が模倣する際のKL最小化

  4. 言語モデル評価:
     パープレキシティ = 2^{H(P,Q)}
     → クロスエントロピー（KLダイバージェンス + エントロピー）の指数
```

### 6.4 Python によるクロスエントロピーとKLダイバージェンスの計算

```python
"""
クロスエントロピーとKLダイバージェンスの計算
"""
import math
from typing import List


def cross_entropy(p: List[float], q: List[float]) -> float:
    """
    クロスエントロピー H(P, Q) を計算する。

    Args:
        p: 真の確率分布
        q: 推定確率分布

    Returns:
        クロスエントロピー（ビット単位）
    """
    if len(p) != len(q):
        raise ValueError("分布の長さが一致しない")

    result = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float('inf')  # Q(x)=0 かつ P(x)>0 なら無限大
            result -= pi * math.log2(qi)
    return result


def kl_divergence(p: List[float], q: List[float]) -> float:
    """
    KLダイバージェンス D_KL(P || Q) を計算する。

    Args:
        p: 真の確率分布
        q: 推定確率分布

    Returns:
        KLダイバージェンス（ビット単位）
    """
    if len(p) != len(q):
        raise ValueError("分布の長さが一致しない")

    result = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float('inf')
            result += pi * math.log2(pi / qi)
    return result


def jensen_shannon_divergence(p: List[float], q: List[float]) -> float:
    """
    ジェンセン-シャノンダイバージェンス JSD(P || Q) を計算する。
    KLダイバージェンスの対称版。

    Args:
        p: 確率分布1
        q: 確率分布2

    Returns:
        JSD（ビット単位）、範囲は [0, 1]
    """
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    return (kl_divergence(p, m) + kl_divergence(q, m)) / 2


# === 動作確認 ===
if __name__ == "__main__":
    # 分類問題の例
    true_dist = [0.0, 0.0, 1.0, 0.0, 0.0]  # クラス3が正解

    good_pred = [0.05, 0.10, 0.70, 0.10, 0.05]
    bad_pred  = [0.20, 0.20, 0.20, 0.20, 0.20]

    print("=== 分類問題のクロスエントロピー ===")
    print(f"  良い予測: H(P,Q) = {cross_entropy(true_dist, good_pred):.4f} ビット")
    print(f"  悪い予測: H(P,Q) = {cross_entropy(true_dist, bad_pred):.4f} ビット")

    # 一般的な分布間の比較
    p = [0.4, 0.3, 0.2, 0.1]
    q = [0.25, 0.25, 0.25, 0.25]  # 一様分布

    print("\n=== 分布間の比較 ===")
    print(f"  P = {p}")
    print(f"  Q = {q}")
    print(f"  H(P)         = {cross_entropy(p, p):.4f} ビット（エントロピー）")
    print(f"  H(P, Q)      = {cross_entropy(p, q):.4f} ビット")
    print(f"  D_KL(P || Q) = {kl_divergence(p, q):.4f} ビット")
    print(f"  D_KL(Q || P) = {kl_divergence(q, p):.4f} ビット（非対称性）")
    print(f"  JSD(P || Q)  = {jensen_shannon_divergence(p, q):.4f} ビット")

    # KLダイバージェンスの非対称性の確認
    print("\n=== 非対称性の確認 ===")
    r = [0.9, 0.1]
    s = [0.5, 0.5]
    print(f"  R = {r}, S = {s}")
    print(f"  D_KL(R || S) = {kl_divergence(r, s):.4f}")
    print(f"  D_KL(S || R) = {kl_divergence(s, r):.4f}")
    print(f"  → D_KL(R||S) ≠ D_KL(S||R) なので距離の公理を満たさない")
```

想定される出力:

```
=== 分類問題のクロスエントロピー ===
  良い予測: H(P,Q) = 0.5146 ビット
  悪い予測: H(P,Q) = 2.3219 ビット

=== 分布間の比較 ===
  P = [0.4, 0.3, 0.2, 0.1]
  Q = [0.25, 0.25, 0.25, 0.25]
  H(P)         = 1.8464 ビット（エントロピー）
  H(P, Q)      = 2.0000 ビット
  D_KL(P || Q) = 0.1536 ビット
  D_KL(Q || P) = 0.1699 ビット（非対称性）
  JSD(P || Q)  = 0.0394 ビット

=== 非対称性の確認 ===
  R = [0.9, 0.1], S = [0.5, 0.5]
  D_KL(R || S) = 0.5310
  D_KL(S || R) = 0.3681
  → D_KL(R||S) ≠ D_KL(S||R) なので距離の公理を満たさない
```

### 6.5 相互情報量と特徴選択

相互情報量 I(X; Y) は特徴選択において重要な指標である。各特徴 X と目的変数 Y の相互情報量を計算し、I(X; Y) が大きい特徴を選択することで、予測に有用な特徴を特定できる。

```
特徴選択における相互情報量:

  特徴量     | I(X; Y) | 解釈
  ───────────┼─────────┼──────────────────────
  年齢       | 0.45    | 目的変数と強い関連
  収入       | 0.38    | 目的変数と中程度の関連
  性別       | 0.02    | 目的変数とほぼ無関係
  郵便番号   | 0.01    | 目的変数とほぼ無関係

  → 年齢と収入を選択、性別と郵便番号は除外

  決定木の情報利得:
  情報利得 = H(Y) - H(Y|X)  = I(X; Y)
  → 分割後のエントロピーが最も減少する特徴で分岐
```

### 6.6 レート歪み理論

非可逆圧縮の理論的限界を示すのが **レート歪み理論**（rate-distortion theory）である。

```
レート歪み関数:

  R(D) = min_{p(x̂|x): E[d(x,x̂)]≤D} I(X; X̂)

  R(D): 歪み D 以下で圧縮するために必要な最小レート
  d(x, x̂): 歪み関数（例: 二乗誤差）

  R(D) のグラフ:
  R
  │
  H │────*
    │     \
    │      \
    │       \
    │        --------
    │                 -----
  0 └──────────────────────── D
    0                    D_max

  - D=0: 可逆圧縮（レート = エントロピー H）
  - D 増大: 許容する歪みが大きいほど低レートで圧縮可能
  - D_max: 元データを無視しても達成できる歪み

  応用:
  - JPEG の品質パラメータ: R(D)曲線上の動作点を選択
  - 映像符号化のビットレート制御
  - 音声圧縮の品質-ビットレートトレードオフ
```

---

## 7. 実装: Python でハフマン符号化

### 7.1 完全な実装

```python
"""
ハフマン符号化の完全な実装

ハフマン木の構築、符号化、復号を行う。
ファイルの圧縮・展開にも対応する。
"""
import heapq
from collections import Counter
from typing import Dict, Optional, Tuple


class HuffmanNode:
    """ハフマン木のノード"""

    def __init__(
        self,
        char: Optional[str] = None,
        freq: int = 0,
        left: Optional["HuffmanNode"] = None,
        right: Optional["HuffmanNode"] = None,
    ):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other: "HuffmanNode") -> bool:
        """優先度キュー用の比較演算子"""
        return self.freq < other.freq

    def is_leaf(self) -> bool:
        """葉ノードかどうか"""
        return self.left is None and self.right is None


class HuffmanCoder:
    """ハフマン符号化・復号器"""

    def __init__(self):
        self.root: Optional[HuffmanNode] = None
        self.codes: Dict[str, str] = {}
        self.reverse_codes: Dict[str, str] = {}

    def build_tree(self, text: str) -> None:
        """
        テキストからハフマン木を構築する。

        Args:
            text: 入力テキスト
        """
        if not text:
            return

        # 文字頻度のカウント
        freq = Counter(text)

        # 優先度キューの初期化
        heap: list = []
        for char, count in freq.items():
            heapq.heappush(heap, HuffmanNode(char=char, freq=count))

        # 記号が1種類の場合の処理
        if len(heap) == 1:
            node = heapq.heappop(heap)
            self.root = HuffmanNode(freq=node.freq, left=node)
            self._generate_codes(self.root, "")
            return

        # ハフマン木の構築
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = HuffmanNode(
                freq=left.freq + right.freq,
                left=left,
                right=right,
            )
            heapq.heappush(heap, merged)

        self.root = heap[0]
        self._generate_codes(self.root, "")

    def _generate_codes(self, node: Optional[HuffmanNode], code: str) -> None:
        """
        ハフマン木を走査して各文字の符号を生成する。

        Args:
            node: 現在のノード
            code: 現在のパスに対応するビット列
        """
        if node is None:
            return

        if node.is_leaf() and node.char is not None:
            # 符号が空の場合（記号が1種類のとき）は "0" を割り当て
            self.codes[node.char] = code if code else "0"
            self.reverse_codes[code if code else "0"] = node.char
            return

        self._generate_codes(node.left, code + "0")
        self._generate_codes(node.right, code + "1")

    def encode(self, text: str) -> str:
        """
        テキストをハフマン符号に変換する。

        Args:
            text: 入力テキスト

        Returns:
            ビット列（文字列表現）
        """
        if not self.codes:
            self.build_tree(text)
        return "".join(self.codes[char] for char in text)

    def decode(self, encoded: str) -> str:
        """
        ハフマン符号を元のテキストに復号する。

        Args:
            encoded: ビット列（文字列表現）

        Returns:
            復号されたテキスト
        """
        if self.root is None:
            return ""

        result = []
        current = self.root

        for bit in encoded:
            if bit == "0":
                current = current.left
            else:
                current = current.right

            if current is not None and current.is_leaf():
                result.append(current.char)
                current = self.root

        return "".join(result)

    def get_statistics(self, text: str) -> Dict[str, float]:
        """
        符号化の統計情報を返す。

        Args:
            text: 入力テキスト

        Returns:
            統計情報の辞書
        """
        import math

        if not self.codes:
            self.build_tree(text)

        freq = Counter(text)
        total = len(text)

        # エントロピー
        entropy_val = 0.0
        for count in freq.values():
            p = count / total
            if p > 0:
                entropy_val -= p * math.log2(p)

        # 平均符号長
        avg_code_length = sum(
            (freq[char] / total) * len(self.codes[char])
            for char in freq
        )

        # 符号化後のビット数
        encoded_bits = sum(freq[char] * len(self.codes[char]) for char in freq)

        return {
            "entropy": entropy_val,
            "avg_code_length": avg_code_length,
            "efficiency": entropy_val / avg_code_length if avg_code_length > 0 else 0,
            "original_bits": total * 8,  # ASCII想定
            "encoded_bits": encoded_bits,
            "compression_ratio": encoded_bits / (total * 8) if total > 0 else 0,
            "n_symbols": len(freq),
        }

    def print_codes(self) -> None:
        """符号表を表示する"""
        print("文字  頻度  符号       符号長")
        print("─" * 40)
        for char in sorted(self.codes.keys()):
            code = self.codes[char]
            display_char = repr(char) if char in (' ', '\n', '\t') else char
            print(f"  {display_char:5s}      {code:12s}  {len(code)}")


# === 動作確認 ===
if __name__ == "__main__":
    # 例1: 基本的な使用
    text = "ABRACADABRA"
    coder = HuffmanCoder()
    coder.build_tree(text)

    print(f"入力: '{text}'")
    print(f"\n=== 符号表 ===")
    coder.print_codes()

    encoded = coder.encode(text)
    decoded = coder.decode(encoded)

    print(f"\n符号化: {encoded}")
    print(f"復号:   {decoded}")
    print(f"一致:   {text == decoded}")

    stats = coder.get_statistics(text)
    print(f"\n=== 統計情報 ===")
    print(f"  エントロピー:     {stats['entropy']:.4f} ビット/文字")
    print(f"  平均符号長:       {stats['avg_code_length']:.4f} ビット/文字")
    print(f"  符号化効率:       {stats['efficiency']:.4f}")
    print(f"  元のビット数:     {stats['original_bits']}")
    print(f"  符号化ビット数:   {stats['encoded_bits']}")
    print(f"  圧縮率:           {stats['compression_ratio']:.4f}")

    # 例2: 英文テキスト
    print("\n" + "=" * 60)
    english_text = (
        "information theory is a mathematical framework for "
        "quantifying information content and it provides fundamental "
        "limits on data compression and reliable communication"
    )
    coder2 = HuffmanCoder()
    coder2.build_tree(english_text)

    print(f"入力: '{english_text[:50]}...'")
    print(f"\n=== 符号表 ===")
    coder2.print_codes()

    stats2 = coder2.get_statistics(english_text)
    print(f"\n=== 統計情報 ===")
    for key, value in stats2.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    encoded2 = coder2.encode(english_text)
    decoded2 = coder2.decode(encoded2)
    print(f"\n復号一致: {english_text == decoded2}")
```

想定される出力:

```
入力: 'ABRACADABRA'

=== 符号表 ===
文字  頻度  符号       符号長
────────────────────────────────────────
  A            0             1
  B            110           3
  C            1110          4
  D            1111          4
  R            10            2

符号化: 01101001110011110110100
復号:   ABRACADABRA
一致:   True

=== 統計情報 ===
  エントロピー:     2.0399 ビット/文字
  平均符号長:       2.0909 ビット/文字
  符号化効率:       0.9756
  元のビット数:     88
  符号化ビット数:   23
  圧縮率:           0.2614
```

### 7.2 ハフマン木の可視化

```python
"""
ハフマン木のテキストベース可視化
"""
from typing import Optional


def visualize_huffman_tree(node: Optional["HuffmanNode"], prefix: str = "",
                           is_left: bool = True, is_root: bool = True) -> str:
    """
    ハフマン木をテキスト形式で可視化する。

    Args:
        node: ハフマンノード
        prefix: インデント用の接頭辞
        is_left: 左の子かどうか
        is_root: ルートノードかどうか

    Returns:
        ツリーの文字列表現
    """
    if node is None:
        return ""

    lines = []
    connector = "" if is_root else ("├── " if is_left else "└── ")
    extension = "" if is_root else ("│   " if is_left else "    ")

    if node.is_leaf():
        char_display = repr(node.char) if node.char in (' ', '\n', '\t') else node.char
        lines.append(f"{prefix}{connector}[{char_display}:{node.freq}]")
    else:
        lines.append(f"{prefix}{connector}({node.freq})")

    if node.left is not None:
        lines.append(
            visualize_huffman_tree(node.left, prefix + extension, True, False)
        )
    if node.right is not None:
        lines.append(
            visualize_huffman_tree(node.right, prefix + extension, False, False)
        )

    return "\n".join(line for line in lines if line)
```

上記のプログラムにおいて、"ABRACADABRA" に対するハフマン木は以下のように可視化される。

```
想定される出力:

  (11)
  ├── [A:5]
  └── (6)
      ├── [R:2]
      └── (4)
          ├── [B:2]
          └── (2)
              ├── [C:1]
              └── [D:1]
```

---

## 8. トレードオフと比較分析

### 8.1 エントロピー符号化の比較

各エントロピー符号化方式には、圧縮率・速度・実装複雑度のトレードオフが存在する。

| 特性 | ハフマン符号 | 算術符号 | ANS (rANS/tANS) |
|------|------------|---------|-----------------|
| 圧縮率 | H(X) ≤ L < H(X)+1 | H(X) に漸近 | H(X) に漸近 |
| 符号化速度 | 高速 | 低速 | 高速〜最高速 |
| 復号速度 | 高速 | 低速 | 最高速 |
| メモリ使用量 | 符号表サイズ | 小 | テーブルサイズ依存 |
| 実装複雑度 | 低 | 高 | 中 |
| 特許問題 | なし | 歴史的にあり | なし |
| 適応型対応 | 可能だが非効率 | 容易 | 可能 |
| 代表的用途 | DEFLATE, JPEG | JPEG2000, H.265 | zstd, LZFSE |

**選択の指針:**
- 互換性・単純さを重視する場合はハフマン符号が適切である
- 最高の圧縮率が必要で速度が許容できる場合は算術符号が適切である
- 高い圧縮率と高速性の両立が必要な場合は ANS が最適解となる

### 8.2 可逆圧縮アルゴリズムの比較

| 特性 | gzip | bzip2 | LZ4 | zstd | Brotli |
|------|------|-------|-----|------|--------|
| 圧縮率 | 中 | 高 | 低 | 高 | 最高 |
| 圧縮速度 | 中 | 低 | 最高 | 高 | 低 |
| 展開速度 | 高 | 中 | 最高 | 最高 | 高 |
| メモリ使用量 | 低 | 高 | 最低 | 中 | 中 |
| 標準化 | RFC 1952 | なし | BSD | RFC 8878 | RFC 7932 |
| エントロピー符号 | ハフマン | ハフマン | なし | ANS | ハフマン+ANS |
| 辞書式圧縮 | LZ77 | BWT | LZ77 | LZ77 | LZ77 |
| 主な用途 | 汎用 | アーカイブ | リアルタイム | 汎用 | Web |

```
圧縮率 vs 圧縮速度のトレードオフ（概念図）:

  圧縮率
  (高い)
    │        * Brotli (max)
    │      * bzip2
    │    * zstd (max)
    │   * gzip (max)
    │  * zstd (default)
    │ * gzip (default)
    │* zstd (fast)
    │
    │             * LZ4 (default)
  (低い)
    └──────────────────────────── 圧縮速度
    (低速)                     (高速)

  展開速度 vs 圧縮率のトレードオフ:
  展開が高速な順: LZ4 > zstd > gzip > Brotli > bzip2
  圧縮率が高い順: Brotli ≈ zstd > bzip2 > gzip > LZ4
```

### 8.3 誤り訂正符号の比較

| 特性 | ハミング符号 | RS符号 | ターボ符号 | LDPC符号 | Polar符号 |
|------|------------|--------|----------|---------|----------|
| 訂正能力 | 1ビット | tシンボル | 高い | 非常に高い | 理論的に最適 |
| シャノン限界 | 遠い | 中程度 | 0.5dB | 0.04dB | 0dB(理論) |
| 復号計算量 | O(n) | O(n²) | O(n log n) | O(n) | O(n log n) |
| 遅延 | 最小 | 小 | 大 | 中〜大 | 中 |
| バースト誤り | 不可 | 得意 | 対応可能 | 対応可能 | 対応可能 |
| 代表的用途 | メモリECC | CD/DVD/QR | 3G/4G | Wi-Fi 6/5G | 5G制御 |

### 8.4 情報理論的指標の使い分け

```
指標の使い分けガイド:

  目的                          推奨指標
  ──────────────────────────────────────────────────
  データの圧縮可能性の評価      エントロピー H(X)
  符号の最適性の評価            平均符号長 L vs H(X)
  通信路の能力評価              通信路容量 C
  分類モデルの訓練              クロスエントロピー H(P,Q)
  分布の近さの測定              KLダイバージェンス D_KL(P||Q)
  分布間の対称的距離            JSD(P||Q) or ワッサーシュタイン距離
  特徴選択                      相互情報量 I(X;Y)
  言語モデルの評価              パープレキシティ 2^{H(P,Q)}
  非可逆圧縮の限界評価          レート歪み関数 R(D)
```

---

## 9. アンチパターン

### アンチパターン1: エントロピーを無視した圧縮の過信

**問題**: データの性質を分析せずに「圧縮すればサイズが減る」と仮定する。

```
誤った前提:
  「このファイルを gzip で圧縮すれば必ず半分以下になる」

現実:
  - 既に圧縮済みのデータ（JPEG, MP3, ZIP）を再圧縮しても
    ほとんど縮まない。むしろ僅かに増大することがある。
  - ランダムデータ（暗号化データ、乱数列）は圧縮不能。
    エントロピーが最大（= log₂(アルファベットサイズ)）であり、
    これ以上冗長性を除去する余地がない。
  - 非常に短いデータは、ヘッダのオーバーヘッドにより
    「圧縮」後のサイズが元より大きくなる場合がある。

具体例:
  ┌────────────────────┬──────────┬──────────┬────────┐
  │ データ種類          │ 元サイズ │ gzip後   │ 比率   │
  ├────────────────────┼──────────┼──────────┼────────┤
  │ 英語テキスト        │ 100 KB   │ 35 KB    │ 35%    │
  │ ソースコード        │ 100 KB   │ 25 KB    │ 25%    │
  │ JPEG画像           │ 100 KB   │ 99 KB    │ 99%    │
  │ 暗号化データ        │ 100 KB   │ 101 KB   │ 101%   │
  │ /dev/urandom       │ 100 KB   │ 101 KB   │ 101%   │
  │ 10バイトテキスト    │ 10 B     │ 30 B     │ 300%   │
  └────────────────────┴──────────┴──────────┴────────┘
```

**対策**: 圧縮前にデータのエントロピーを推定し、圧縮効果が見込めるか判断する。既に高エントロピーなデータに対しては圧縮を適用しない。特に、暗号化パイプラインでは「暗号化の前に圧縮」が正しい順序である（暗号化後のデータは擬似ランダムであり圧縮できない）。

### アンチパターン2: 固定長符号への固執

**問題**: 可変長符号の利点を無視し、すべてのシンボルに同じビット数を割り当てる。

```
誤ったアプローチ:
  英語テキストの各文字を8ビット固定長（ASCII）で格納する。

分析:
  英語テキストの1次エントロピー ≈ 4.11 ビット/文字
  文脈を考慮したエントロピー ≈ 1.0〜1.5 ビット/文字
  ASCII: 8 ビット/文字

  冗長度 = 1 - H/L = 1 - 4.11/8 ≈ 49%
  → テキストの約半分は冗長な情報

  出現頻度が大きく異なる場合:
  'e' ≈ 12.7% (最頻) → 最適には約3ビット
  'z' ≈ 0.07% (最少) → 最適には約10ビット

  しかし ASCII では両方とも8ビット
  → 高頻度文字に短い符号を割り当てる方が効率的
```

**対策**: データの統計的性質に応じた可変長符号（ハフマン符号、算術符号、ANS）を使用する。ただし、ランダムアクセスが必要な場合や、符号化・復号のレイテンシが厳しい場合には、固定長符号が適切な選択肢となることもある。設計判断は要件に基づいて行う。

### アンチパターン3: 誤り訂正の過不足

**問題**: 通信路の特性を考慮せずに、誤り訂正符号を選択する。

```
パターンA: 誤り訂正の不足
  「インターネット通信だから TCP に任せれば大丈夫」
  → ストレージの長期保存では、TCP は関係ない
  → ビットロット（経年劣化によるデータ破損）に対する
     誤り訂正は別途必要

パターンB: 誤り訂正の過剰
  「安全のために最大冗長度で誤り訂正符号を付加する」
  → 帯域幅/ストレージの浪費
  → 通信路容量 C に対して冗長度を増やしても、
     R > C ならば信頼性は改善しない
  → 適切な誤り訂正レベルの選択が重要

対策:
  1. 通信路の誤り特性を分析する（BER, バースト長等）
  2. 必要な信頼性レベルを明確にする（BER < 10^-9 等）
  3. シャノン限界と実用的な符号の性能を比較する
  4. 複数の候補から計算量・遅延・冗長度のバランスを取る
```

---

## 10. 演習問題

### 基礎レベル

**演習1: エントロピーの計算**

4面のサイコロがあり、各面の出現確率が以下の通りである。

```
面1: P = 0.5
面2: P = 0.25
面3: P = 0.125
面4: P = 0.125
```

(a) このサイコロのエントロピー H を計算せよ。
(b) 一様分布（各面 P = 0.25）の場合のエントロピーと比較せよ。
(c) ハフマン符号を構築し、各面の符号語と平均符号長を求めよ。

<details>
<summary>解答</summary>

```
(a) エントロピーの計算:
  H = -(0.5×log₂0.5 + 0.25×log₂0.25 + 0.125×log₂0.125 + 0.125×log₂0.125)
    = -(0.5×(-1) + 0.25×(-2) + 0.125×(-3) + 0.125×(-3))
    = -(−0.5 − 0.5 − 0.375 − 0.375)
    = 1.75 ビット

(b) 一様分布の場合:
  H_uniform = log₂(4) = 2.0 ビット
  → 偏った分布 (1.75) < 一様分布 (2.0)
  → 偏りがある方がエントロピーは小さい（予測しやすい）

(c) ハフマン符号:
  ステップ1: 面3(0.125) と 面4(0.125) を結合 → 3,4
  ステップ2: 面2(0.25) と 3,4 を結合 → 2,3,4
  ステップ3: 面1(0.5) と 2,3,4 を結合 → ルート(1.0)

  符号割り当て:
    面1: 0        (1ビット)
    面2: 10       (2ビット)
    面3: 110      (3ビット)
    面4: 111      (3ビット)

  平均符号長:
  L = 0.5×1 + 0.25×2 + 0.125×3 + 0.125×3
    = 0.5 + 0.5 + 0.375 + 0.375
    = 1.75 ビット

  L = H = 1.75 → 確率が2のべき乗の場合、
  ハフマン符号はエントロピーに完全に一致する。
```

</details>

**演習2: 情報量の直感的理解**

以下の各事象について、自己情報量を計算し、日常的な直感と一致するか検討せよ。

```
(a) 天気予報で「東京で1月に雪が降る」（想定確率: 0.05）
(b) 天気予報で「東京で8月に晴れる」（想定確率: 0.7）
(c) 宝くじで1等が当たる（想定確率: 1/10,000,000）
(d) サイコロで6が出る（想定確率: 1/6）
```

<details>
<summary>解答</summary>

```
(a) I = -log₂(0.05) ≈ 4.32 ビット
    → 珍しい事象であり、情報量が大きい

(b) I = -log₂(0.7) ≈ 0.51 ビット
    → ありふれた事象であり、情報量が小さい
    → 「夏に晴れた」は驚きが少ない

(c) I = -log₂(1/10,000,000) ≈ 23.25 ビット
    → 非常に稀な事象であり、情報量が極めて大きい
    → 宝くじの当選はニュースになる

(d) I = -log₂(1/6) ≈ 2.58 ビット
    → 中程度の情報量
    → サイコロの1つの目はそこそこの驚き

直感との一致:
事象の「驚き」の大きさと自己情報量は概ね一致する。
日常的に「ニュースになる」事象ほど情報量が大きい。
```

</details>

### 応用レベル

**演習3: テキスト圧縮の分析**

以下のテキストに対して、ハフマン符号を構築し、圧縮率を分析せよ。

```
テキスト: "AABABCABCDABCDE"
```

(a) 各文字の出現頻度と確率を求めよ。
(b) エントロピーを計算せよ。
(c) ハフマン木を構築し、各文字の符号語を示せ。
(d) 平均符号長と符号化効率（H/L）を計算せよ。
(e) ASCII（8ビット/文字）と比較した圧縮率を求めよ。

<details>
<summary>解答</summary>

```
(a) 文字頻度:
  A: 5回 (P=5/15=0.333)
  B: 4回 (P=4/15=0.267)
  C: 3回 (P=3/15=0.200)
  D: 2回 (P=2/15=0.133)
  E: 1回 (P=1/15=0.067)

(b) エントロピー:
  H = -(0.333×log₂0.333 + 0.267×log₂0.267 + 0.200×log₂0.200
       + 0.133×log₂0.133 + 0.067×log₂0.067)
    ≈ -(0.333×(-1.585) + 0.267×(-1.907) + 0.200×(-2.322)
       + 0.133×(-2.907) + 0.067×(-3.907))
    ≈ 0.528 + 0.509 + 0.464 + 0.387 + 0.261
    ≈ 2.149 ビット

(c) ハフマン木の構築:
  ステップ1: D(0.133) と E(0.067) → DE
  ステップ2: C(0.200) と DE → CDE
  ステップ3: B(0.267) と CDE → BCDE
  （あるいは A(0.333) と B(0.267) を先に結合しても可）

  一つの有効な符号割り当て:
    A:  0       (1ビット)
    B:  10      (2ビット)
    C:  110     (3ビット)
    D:  1110    (4ビット)
    E:  1111    (4ビット)

(d) 平均符号長と効率:
  L = 0.333×1 + 0.267×2 + 0.200×3 + 0.133×4 + 0.067×4
    = 0.333 + 0.534 + 0.600 + 0.533 + 0.267
    = 2.267 ビット

  効率 = H/L = 2.149/2.267 ≈ 0.948 (94.8%)

(e) 圧縮率:
  ASCII: 15 × 8 = 120 ビット
  ハフマン: 5×1 + 4×2 + 3×3 + 2×4 + 1×4 = 5+8+9+8+4 = 34 ビット
  圧縮率: 34/120 = 28.3%
```

</details>

**演習4: 通信路容量の計算**

二元対称通信路（BSC）のビット反転確率が p = 0.05 であるとする。

(a) 通信路容量 C を計算せよ。
(b) 符号化率 R = 0.7 で通信した場合、信頼性のある通信は可能か。
(c) 符号化率 R = 0.8 では可能か。R = 0.9 ではどうか。

<details>
<summary>解答</summary>

```
(a) BSC の通信路容量:
  C = 1 - H(p) = 1 - H(0.05)
  H(0.05) = -(0.05×log₂0.05 + 0.95×log₂0.95)
           = -(0.05×(-4.322) + 0.95×(-0.074))
           = -(−0.216 − 0.070)
           = 0.286 ビット

  C = 1 - 0.286 = 0.714 ビット/使用

(b) R = 0.7 < C = 0.714
  → R < C なので、シャノンの第二基本定理により、
    適切な符号を用いれば信頼性のある通信が可能。

(c) R = 0.8 > C = 0.714
  → R > C なので、どのような符号を用いても
    復号誤り率を任意に小さくすることは不可能。

  R = 0.9 > C = 0.714
  → 同様に不可能。さらにギャップが大きいため、
    誤り率はより高くなる。
```

</details>

### 発展レベル

**演習5: ハフマン符号の最適性の証明**

ハフマン符号が最適前置符号であること（すなわち、前置符号の中で平均符号長が最小であること）を、以下の手順で証明せよ。

(a) 最適前置符号において、確率が最も小さい2つの記号が、最も長い符号語を持ち、かつ最後の1ビットだけが異なる（兄弟ノードである）ことを示せ。
(b) これらの2つの記号を1つにまとめた（確率を合算した）新しい情報源に対して、最適前置符号の平均符号長と元の情報源の最適前置符号の平均符号長の関係を示せ。
(c) 帰納法を用いて、ハフマンのアルゴリズムが常に最適な前置符号を生成することを証明せよ。

<details>
<summary>解答の方針</summary>

```
(a) 背理法による証明の方針:
  最適前置符号 C* において、最も確率の小さい2つの記号を
  x, y（P(x) ≤ P(y)）とする。

  仮に x の符号語が最長でないとする。すると、ある記号 z が
  x より長い符号語を持つ。P(x) ≤ P(z) のとき、
  x と z の符号語を交換すると、平均符号長は減少するか
  変わらない。これは C* の最適性に矛盾する。

  同様に、x と y が兄弟でないと仮定すると、
  他の兄弟ペアとの交換により平均符号長を減少させられ、
  矛盾が生じる。

(b) 新しい情報源 X' を、x と y を確率 P(x)+P(y) の
  新しい記号 z で置き換えたものとする。
  C' を X' の最適前置符号、C* を X の最適前置符号とすると:

  L(C*) = L(C') + P(x) + P(y)

  （z の符号語に0/1を追加して x, y の符号語を作るため、
   x と y の分だけ符号長が1ビット増える）

(c) 帰納法:
  基底: 記号数2のとき、ハフマン符号は {0, 1} であり明らかに最適。
  帰納ステップ: 記号数 n-1 でハフマン符号が最適と仮定する。
  記号数 n の情報源に対して:
  - (a) より、最適符号では最小確率の2記号は兄弟
  - ハフマンのアルゴリズムはこの2記号を結合
  - (b) より、結合後の n-1 記号の問題に帰着
  - 帰納法の仮定より、ハフマンは n-1 記号に対して最適
  - したがって、n 記号に対しても最適
```

</details>

**演習6: 情報理論と機械学習の接点**

2クラス分類問題において、正解分布 P と3つのモデルの予測分布 Q₁, Q₂, Q₃ が以下の通りであるとする。

```
P  = [0.8, 0.2]（クラス0が80%, クラス1が20%）
Q₁ = [0.9, 0.1]
Q₂ = [0.5, 0.5]
Q₃ = [0.2, 0.8]
```

(a) 各モデルのクロスエントロピー H(P, Qᵢ) を計算せよ。
(b) 各モデルの KL ダイバージェンス D_KL(P || Qᵢ) を計算せよ。
(c) どのモデルが最も良い予測をしているか、情報理論的に説明せよ。

<details>
<summary>解答</summary>

```
P のエントロピー:
H(P) = -(0.8×log₂0.8 + 0.2×log₂0.2)
     = -(0.8×(-0.322) + 0.2×(-2.322))
     = 0.258 + 0.464 = 0.722 ビット

(a) クロスエントロピー:
H(P, Q₁) = -(0.8×log₂0.9 + 0.2×log₂0.1)
          = -(0.8×(-0.152) + 0.2×(-3.322))
          = 0.122 + 0.664 = 0.786 ビット

H(P, Q₂) = -(0.8×log₂0.5 + 0.2×log₂0.5)
          = -(0.8×(-1) + 0.2×(-1))
          = 0.8 + 0.2 = 1.000 ビット

H(P, Q₃) = -(0.8×log₂0.2 + 0.2×log₂0.8)
          = -(0.8×(-2.322) + 0.2×(-0.322))
          = 1.858 + 0.064 = 1.922 ビット

(b) KLダイバージェンス:
D_KL(P || Q₁) = H(P, Q₁) - H(P) = 0.786 - 0.722 = 0.064 ビット
D_KL(P || Q₂) = H(P, Q₂) - H(P) = 1.000 - 0.722 = 0.278 ビット
D_KL(P || Q₃) = H(P, Q₃) - H(P) = 1.922 - 0.722 = 1.200 ビット

(c) Q₁ が最も良い予測:
  - クロスエントロピーが最小 (0.786)
  - KLダイバージェンスが最小 (0.064)
  - Q₁ は P に最も近い分布
  - Q₃ は P とほぼ逆の予測をしており、最も悪い

  クロスエントロピーの最小化 ⟺ KLダイバージェンスの最小化
  （H(P) は定数であるため）

  これが機械学習でクロスエントロピー損失を最小化する
  理論的根拠である。
```

</details>

---

## 11. FAQ

### Q1: エントロピーと熱力学的エントロピーの関係は？

シャノンが情報のエントロピーを定義したとき、フォン・ノイマンに相談したところ「エントロピーと呼ぶべきだ。第一に、その数学的形式が統計力学のエントロピーと同一だから。第二に、エントロピーという言葉を本当に理解している人は誰もいないから、議論で常に有利になれるだろう」と助言されたという逸話がある。

数学的形式の類似は偶然ではない。ボルツマンの統計力学エントロピー S = k_B × ln(W)（W はミクロ状態の数）とシャノンエントロピー H = -Σ p_i × log p_i は、どちらも「状態の不確実性」を定量化している。ボルツマンエントロピーは系のミクロ状態の数の対数であり、シャノンエントロピーは情報源の不確実性（平均的な驚きの度合い）である。

マクスウェルの悪魔のパラドックスは、この関係をさらに深める。情報の消去（ランダウアーの原理）にはエネルギーの散逸が伴うことが示されており、情報と物理は本質的に結びついている。ランダウアーの原理によれば、1ビットの情報を消去するために最低 k_B × T × ln(2) のエネルギーが必要である。

### Q2: データ圧縮においてエントロピーは本当に「限界」なのか？

はい。シャノンの情報源符号化定理は、エントロピーが可逆圧縮の絶対的な限界であることを数学的に証明している。ただし、いくつかの重要な注意点がある。

1. **モデルに依存する**: エントロピーは確率分布のモデルに基づいて計算される。より精密なモデルを使えば、推定されるエントロピーは下がりうる。例えば、英語テキストの1次統計量（各文字の出現頻度のみ）に基づくエントロピーは約4.1ビット/文字だが、文脈（n-gram）を考慮すると約1.0〜1.5ビット/文字まで下がる。
2. **無記憶性の仮定**: シャノンの第一定理の基本形は無記憶情報源を仮定している。実際のデータ（テキスト、画像等）には強い依存関係があり、これを利用することでエントロピーに近い圧縮が可能になる。LZ77 系のアルゴリズムはこの依存関係を辞書として活用している。
3. **計算量の制約**: エントロピーに近い圧縮を達成する符号は理論的に存在するが、符号化・復号の計算量が実用的に許容できるかは別問題である。

### Q3: 機械学習でクロスエントロピー損失を使う理由は？

機械学習における分類問題でクロスエントロピー損失が標準的に使われる理由は、情報理論的に明確な根拠がある。

1. **KLダイバージェンスの最小化**: クロスエントロピーの最小化は、正解分布 P とモデルの予測分布 Q の間の KL ダイバージェンスの最小化と等価である（H(P) は定数であるため）。KL ダイバージェンスは「モデルが正解分布にどれだけ近いか」の自然な指標である。

2. **最尤推定との等価性**: クロスエントロピーの最小化は、モデルパラメータの最尤推定と数学的に等価である。対数尤度 Σ log Q(y_i | x_i) の最大化は、クロスエントロピー -Σ P(y|x) log Q(y|x) の最小化と同じ演算である。

3. **勾配の性質**: クロスエントロピー損失の勾配は、予測の「間違い具合」に比例する。正解から大きく外れた予測に対しては大きな勾配が生じ、学習が加速する。これに対して、二乗誤差損失ではソフトマックス出力に対する勾配が飽和しやすい問題がある。

4. **情報理論的な正当性**: モデルの予測分布 Q で正解分布 P のデータを符号化するときに必要な「余分なビット数」を最小化するということは、モデルが正解分布をできるだけ正確に学習するということに他ならない。

### Q4: パープレキシティとは何か。なぜ言語モデルの評価に使われるのか。

パープレキシティ（perplexity）は、言語モデルの性能を評価する標準的な指標であり、情報理論に直接的な根拠を持つ。

```
定義:
  パープレキシティ = 2^{H(P, Q)}

  ここで H(P, Q) はテストデータ上のクロスエントロピー。
  実用的には:
  PP = 2^{-(1/N) Σ log₂ Q(w_i | w_1...w_{i-1})}

  直感的解釈:
  パープレキシティが PP ということは、
  モデルが各単語の予測において「PP 個の候補から
  一様にランダムに選んでいるのと同程度の不確実性」
  を持っていることを意味する。

  例:
  PP = 100 → 毎回100語の候補から選んでいるのと同等の困難さ
  PP = 10  → 毎回10語の候補から選んでいるのと同等の困難さ
  PP = 1   → 完璧な予測（次の単語を常に正しく当てられる）

  言語モデルの進化（想定されるパープレキシティの推移）:
  n-gram モデル: PP ≈ 100〜200
  LSTM:         PP ≈ 60〜80
  Transformer:  PP ≈ 20〜40
  GPT系:        PP ≈ 10〜20
```

パープレキシティが低いほど、モデルはテストデータをより良く予測できている。ただし、パープレキシティはモデルの「流暢さ」を測る指標であり、生成テキストの「正確さ」や「有用さ」は別途評価する必要がある。

### Q5: 量子情報理論とは何か。古典情報理論とどう違うのか。

量子情報理論は、量子力学の原理に基づいて情報を扱う理論である。

```
古典情報理論との主な違い:

  概念        古典情報理論        量子情報理論
  ─────────────────────────────────────────────
  基本単位    ビット (0 or 1)     量子ビット（|0⟩, |1⟩, 重ね合わせ）
  エントロピー シャノンエントロピー フォン・ノイマンエントロピー
               H = -Σ p log p     S = -Tr(ρ log ρ)
  通信路      古典通信路          量子通信路
  通信路容量  シャノン容量        ホレヴォ容量
  複製        可能                不可能（量子複製不可能定理）
  テレポート  不可能              量子テレポーテーション可能

  量子情報理論の応用:
  - 量子暗号（BB84プロトコル）: 盗聴の検出が物理的に保証される
  - 量子誤り訂正: デコヒーレンスからの量子状態の保護
  - 量子圧縮（シューマッハ圧縮）: 量子ビットの圧縮
  - 量子通信路符号化: 量子通信路の容量定理
```

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## 12. まとめ

### 12.1 主要概念の整理

| 概念 | 定義 | 要点 |
|------|------|------|
| 自己情報量 | I(x) = -log₂(P(x)) | 稀な事象ほど情報量が大きい |
| エントロピー | H(X) = -Σ P(x) log₂ P(x) | 情報源の不確実性の尺度。圧縮の理論的限界 |
| 条件付きエントロピー | H(Y\|X) = -Σ P(x,y) log₂ P(y\|x) | X を知った上での Y の残りの不確実性 |
| 相互情報量 | I(X;Y) = H(X) + H(Y) - H(X,Y) | X と Y が共有する情報量 |
| クロスエントロピー | H(P,Q) = -Σ P(x) log₂ Q(x) | P のデータを Q の符号で表すときの平均ビット数 |
| KLダイバージェンス | D_KL(P\|\|Q) = Σ P(x) log₂(P(x)/Q(x)) | P を Q で近似する情報損失量 |
| 通信路容量 | C = max I(X;Y) | 信頼性のある通信の最大速度 |

### 12.2 2つの基本定理

```
┌──────────────────────────────────────────────────────┐
│  シャノンの第一基本定理（情報源符号化定理）            │
│                                                        │
│  エントロピー H(X) は可逆圧縮の限界である。            │
│  H(X) ≤ L < H(X) + 1                                 │
│  ブロック符号化により H(X) に任意に近づけられる。       │
│                                                        │
│  応用: ハフマン符号, 算術符号, ANS, ZIP, gzip, zstd    │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  シャノンの第二基本定理（通信路符号化定理）            │
│                                                        │
│  通信路容量 C 以下の速度であれば、                      │
│  誤り率を任意に小さくする符号が存在する。              │
│  C を超える速度では、誤り率を 0 にはできない。         │
│                                                        │
│  応用: ハミング符号, RS符号, LDPC符号, Polar符号, 5G   │
└──────────────────────────────────────────────────────┘
```

### 12.3 情報理論の思想的意義

シャノンの情報理論が革命的だったのは、以下の3つの発想にある。

1. **意味からの分離**: 情報の「内容」や「意味」を無視し、統計的な性質のみに着目することで、普遍的な理論を構築した。
2. **限界の明示**: 圧縮の限界（エントロピー）と通信の限界（通信路容量）を数学的に示し、工学者に「どこまで改善の余地があるか」の指標を与えた。
3. **存在証明の力**: 特に第二定理は、具体的な符号の構成法を示さずに、良い符号の「存在」を証明した。この存在証明が、その後50年以上にわたる符号化理論の研究を動機づけた。

---

## 次に読むべきガイド


---

## 参考文献

1. Shannon, C. E. "A Mathematical Theory of Communication." *The Bell System Technical Journal*, Vol. 27, pp. 379-423, 623-656, 1948.
   - 情報理論の創始論文。エントロピー、通信路容量、情報源符号化定理、通信路符号化定理のすべてがこの1本の論文で提示された。情報理論を学ぶ者は原典に当たることを強く推奨する。

2. Cover, T. M. & Thomas, J. A. *Elements of Information Theory*, 2nd Edition. Wiley-Interscience, 2006.
   - 情報理論の標準的な教科書。理論から応用までを網羅的にカバーしている。大学院レベルの教科書として世界的に定評がある。エントロピー、通信路容量、レート歪み理論、ネットワーク情報理論まで幅広く扱う。

3. MacKay, D. J. C. *Information Theory, Inference, and Learning Algorithms*. Cambridge University Press, 2003.
   - 情報理論と機械学習の接点を重視した教科書。無料のオンライン版が著者のWebサイトで公開されている。誤り訂正符号、ベイズ推論、ニューラルネットワークを統一的な視点で解説する。

4. Huffman, D. A. "A Method for the Construction of Minimum-Redundancy Codes." *Proceedings of the IRE*, Vol. 40, No. 9, pp. 1098-1101, 1952.
   - ハフマン符号の原論文。MITの大学院生時代に、期末レポートの代わりとして提出された研究が、後に最も広く使われる符号化アルゴリズムの1つとなった。

5. Arikan, E. "Channel Polarization: A Method for Constructing Capacity-Achieving Codes for Symmetric Binary-Input Memoryless Channels." *IEEE Transactions on Information Theory*, Vol. 55, No. 7, pp. 3051-3073, 2009.
   - Polar符号の原論文。初めて「構成的に」シャノン限界を達成する符号を示した歴史的な成果であり、5Gの制御チャネルに採用されている。

6. Ziv, J. & Lempel, A. "A Universal Algorithm for Sequential Data Compression." *IEEE Transactions on Information Theory*, Vol. 23, No. 3, pp. 337-343, 1977.
   - LZ77の原論文。辞書式圧縮の基礎を確立し、gzip, ZIP, PNG などの現代的圧縮アルゴリズムの原型となった。
