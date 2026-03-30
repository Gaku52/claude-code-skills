# 圧縮アルゴリズム

> データ圧縮の本質は「冗長性の除去」であり、情報理論のシャノン限界が圧縮可能な理論的下限を定める。ハフマン符号は統計的冗長性を、LZ 系は辞書的冗長性を、DEFLATE はその両方を同時に排除する。これらの原理を理解し、用途に応じた圧縮手法を正しく選択できることは、ソフトウェアエンジニアにとって不可欠な素養である。

---

## この章で学ぶこと

- [ ] 情報エントロピーとシャノン限界から圧縮の理論的限界を説明できる
- [ ] ランレングス符号化（RLE）の仕組みと適用場面を理解する
- [ ] ハフマン符号の木構築から符号化・復号化まで実装できる
- [ ] LZ77 のスライディングウィンドウ方式を説明し簡易実装できる
- [ ] LZ78 / LZW の辞書構築方式を理解し GIF 特許問題の経緯を説明できる
- [ ] DEFLATE の二段階圧縮（LZ77 + ハフマン）の仕組みを説明できる
- [ ] 可逆圧縮と非可逆圧縮の違い・使い分けを即答できる
- [ ] 現代の圧縮アルゴリズム（zstd, Brotli, LZ4）の特性比較ができる
- [ ] JPEG / MP3 / H.264 の非可逆圧縮の原理を概説できる
- [ ] HTTP 圧縮・データベース圧縮など実務の圧縮設計を行える

## 前提知識

- 基本的なデータ構造（二分木、優先度付きキュー）の概念

---

## 1. 情報理論の基礎

### 1.1 シャノンのエントロピー

1948 年、クロード・シャノンは論文 "A Mathematical Theory of Communication" において、データに含まれる「情報量」を数学的に定義した。この概念はデータ圧縮の理論的基盤であり、あらゆる圧縮アルゴリズムの性能を評価する基準となる。

#### エントロピーの定義

情報源 X が n 個のシンボル x_1, x_2, ..., x_n を確率 p(x_1), p(x_2), ..., p(x_n) で出力するとき、情報エントロピー H(X) は次の式で定義される。

```
  情報エントロピー H(X):

    H(X) = - Σ p(x_i) × log₂(p(x_i))     [単位: ビット/シンボル]
            i=1..n

  直感的な意味:
    - H(X) は「次のシンボルを予測するのにどれだけ情報が必要か」を表す
    - エントロピーが低い → 予測しやすい → 冗長性が高い → よく圧縮できる
    - エントロピーが高い → 予測しにくい → ランダムに近い → 圧縮しにくい
```

#### 具体的な計算例

```
  例1: 公平なコイン投げ（表50%, 裏50%）

    H = -(0.5 × log₂(0.5) + 0.5 × log₂(0.5))
    H = -(0.5 × (-1) + 0.5 × (-1))
    H = -(-0.5 + (-0.5))
    H = 1.0 ビット/シンボル

    → 結果を表すのに最低1ビット必要（これは直感に合う）

  例2: 偏ったコイン（表90%, 裏10%）

    H = -(0.9 × log₂(0.9) + 0.1 × log₂(0.1))
    H = -(0.9 × (-0.152) + 0.1 × (-3.322))
    H ≈ -((-0.137) + (-0.332))
    H ≈ 0.469 ビット/シンボル

    → 1ビット未満で表現可能！偏りがある = 冗長性がある = 圧縮できる

  例3: 4シンボルの情報源（A:50%, B:25%, C:12.5%, D:12.5%）

    H = -(0.5 × log₂(0.5) + 0.25 × log₂(0.25)
         + 0.125 × log₂(0.125) + 0.125 × log₂(0.125))
    H = -(0.5 × (-1) + 0.25 × (-2) + 0.125 × (-3) + 0.125 × (-3))
    H = -(-0.5 + (-0.5) + (-0.375) + (-0.375))
    H = 1.75 ビット/シンボル

    → 固定長なら2ビット必要だが、エントロピーは1.75ビット
    → ハフマン符号で A=0, B=10, C=110, D=111 とすれば
       平均符号長 = 0.5×1 + 0.25×2 + 0.125×3 + 0.125×3 = 1.75 ビット
       → エントロピーに完全一致！（この場合ハフマン符号が最適解）
```

#### シャノン限界（情報源符号化定理）

```
  シャノンの情報源符号化定理（第一定理）:

    任意の無歪み（可逆）符号化において、
    シンボルあたりの平均符号長 L は必ずエントロピー H 以上である。

      L ≥ H(X)

    かつ、H(X) に任意に近い平均符号長を達成する符号が存在する。

      H(X) ≤ L < H(X) + 1

    意味:
    - どんなに優れたアルゴリズムでも、エントロピー以下には可逆圧縮できない
    - エントロピーに近い圧縮率を達成するアルゴリズムは存在する
    - これが「圧縮の理論的限界」

  実用的な含意:
    ┌──────────────────────────────────────────────────────────┐
    │  データの種類          │ エントロピー │ 圧縮可能性       │
    ├──────────────────────────────────────────────────────────┤
    │  自然言語テキスト      │ 約1-2 bit/字 │ 高い（2-5倍）    │
    │  ソースコード          │ 約2-3 bit/字 │ 中程度（2-4倍）  │
    │  暗号化データ          │ ≈8 bit/byte  │ ほぼ不可能       │
    │  既圧縮データ          │ ≈8 bit/byte  │ ほぼ不可能       │
    │  ランダムバイト列      │ 8 bit/byte   │ 不可能           │
    └──────────────────────────────────────────────────────────┘
```

#### Python によるエントロピー計算

```python
"""
情報エントロピーの計算

シャノンのエントロピー公式に基づいて、与えられたデータの
情報エントロピーを計算する。これにより理論的な圧縮限界を知ることができる。
"""
import math
from collections import Counter
from typing import Union


def calculate_entropy(data: Union[str, bytes]) -> float:
    """シャノンのエントロピーを計算する。

    Args:
        data: 解析対象のデータ（文字列またはバイト列）

    Returns:
        ビット/シンボル単位のエントロピー値

    Raises:
        ValueError: 空のデータが渡された場合

    >>> calculate_entropy("AAAA")
    0.0
    >>> round(calculate_entropy("AB" * 50), 2)
    1.0
    >>> round(calculate_entropy("ABRACADABRA"), 2)
    2.04
    """
    if not data:
        raise ValueError("データが空です")

    total = len(data)
    freq = Counter(data)

    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def analyze_compressibility(data: Union[str, bytes]) -> dict:
    """データの圧縮可能性を分析する。

    エントロピーに基づいて理論的な最小サイズと
    圧縮率の上限を算出する。

    Args:
        data: 解析対象のデータ

    Returns:
        分析結果を含む辞書
    """
    entropy = calculate_entropy(data)
    total = len(data)

    # 理論的最小ビット数
    min_bits = entropy * total
    min_bytes = math.ceil(min_bits / 8)

    # シンボルあたりのビット数（固定長の場合）
    unique_symbols = len(set(data))
    fixed_bits_per_symbol = math.ceil(math.log2(unique_symbols)) if unique_symbols > 1 else 1
    fixed_total_bits = fixed_bits_per_symbol * total

    return {
        "length": total,
        "unique_symbols": unique_symbols,
        "entropy_per_symbol": round(entropy, 4),
        "fixed_bits_per_symbol": fixed_bits_per_symbol,
        "theoretical_min_bits": round(min_bits, 2),
        "theoretical_min_bytes": min_bytes,
        "fixed_length_bits": fixed_total_bits,
        "max_compression_ratio": round(fixed_total_bits / min_bits, 2) if min_bits > 0 else float('inf'),
    }


# 使用例
if __name__ == "__main__":
    samples = [
        ("均等分布", "ABCD" * 25),
        ("偏り大", "A" * 90 + "B" * 10),
        ("自然言語風", "the quick brown fox jumps over the lazy dog"),
        ("最大冗長", "AAAAAAAAAA"),
    ]

    for label, data in samples:
        result = analyze_compressibility(data)
        print(f"\n--- {label}: '{data[:30]}...' ---")
        print(f"  長さ: {result['length']} シンボル")
        print(f"  ユニーク: {result['unique_symbols']} 種類")
        print(f"  エントロピー: {result['entropy_per_symbol']} bit/symbol")
        print(f"  理論最小: {result['theoretical_min_bits']} bits "
              f"({result['theoretical_min_bytes']} bytes)")
        print(f"  固定長: {result['fixed_length_bits']} bits")
        print(f"  最大圧縮率: {result['max_compression_ratio']}倍")
```

### 1.2 自己情報量と条件付きエントロピー

エントロピーの理解を深めるために、関連する概念を補足する。

```
  自己情報量（Self-Information）:

    事象 x が発生したとき、その情報量は次のように定義される:

      I(x) = -log₂(p(x))   [単位: ビット]

    直感: 珍しい事象ほど情報量が大きい
      - p(x) = 1（確実に起こる）→ I(x) = 0 ビット（情報なし）
      - p(x) = 0.5           → I(x) = 1 ビット
      - p(x) = 0.01（稀）    → I(x) ≈ 6.64 ビット（大きな情報量）

    エントロピーは自己情報量の期待値:
      H(X) = E[I(X)] = Σ p(x) × I(x)

  条件付きエントロピー H(Y|X):

    X を知った上での Y の不確実さ。
    データ圧縮では「文脈」を活用するほど条件付きエントロピーが下がり、
    より効率的な圧縮が可能になる。

    例: 英語テキストにおいて
      - H(次の文字) ≈ 4.0 bit/文字（文脈なし）
      - H(次の文字 | 直前1文字) ≈ 3.3 bit/文字
      - H(次の文字 | 直前2文字) ≈ 3.0 bit/文字
      ...
      - シャノンの推定: 英語の真のエントロピー ≈ 1.0-1.5 bit/文字

    → PPM, LZMA 等の高圧縮アルゴリズムは長い文脈を活用して
      条件付きエントロピーに近づこうとする
```

### 1.3 可逆圧縮 vs 非可逆圧縮

データ圧縮は大きく二つの範疇に分かれる。この区分は圧縮の本質に関わるものであり、あらゆる圧縮技術を理解する上での出発点となる。

```
  ┌────────────────────────────────────────────────────────────────┐
  │                     データ圧縮の分類                            │
  ├─────────────────────────┬──────────────────────────────────────┤
  │   可逆圧縮 (Lossless)   │   非可逆圧縮 (Lossy)                │
  ├─────────────────────────┼──────────────────────────────────────┤
  │ 元データを完全に復元可能  │ 元データの近似を復元                 │
  │ 1ビットの損失もない      │ 人間に知覚されにくい情報を除去        │
  │                         │                                      │
  │ 理論的限界:              │ 限界:                                │
  │  シャノン限界まで        │  品質の許容範囲による                │
  │                         │  （Rate-Distortion理論）             │
  │                         │                                      │
  │ 主要アルゴリズム:        │ 主要アルゴリズム:                    │
  │  - ハフマン符号          │  - DCT（JPEG, MPEG）               │
  │  - 算術符号              │  - ウェーブレット（JPEG 2000）      │
  │  - LZ77 / LZ78 / LZW   │  - 心理音響モデル（MP3, AAC）       │
  │  - DEFLATE              │  - 動き補償（H.264, H.265, AV1）   │
  │  - BWT + MTF            │  - 量子化                           │
  │                         │                                      │
  │ 用途:                   │ 用途:                                │
  │  - テキスト/ソースコード │  - 写真・画像（JPEG, WebP, AVIF）   │
  │  - 実行可能ファイル      │  - 音楽（MP3, AAC, Opus）           │
  │  - データベースバックアップ│  - 動画（H.264, H.265, AV1）       │
  │  - アーカイブ (ZIP等)    │  - 音声通話（Speex, Opus）          │
  │  - ネットワーク転送      │                                      │
  │                         │                                      │
  │ 圧縮率: 2〜5 倍程度     │ 圧縮率: 10〜1000 倍以上             │
  │ （データの性質に依存）   │ （許容品質に依存）                   │
  └─────────────────────────┴──────────────────────────────────────┘
```

#### 可逆圧縮が不可欠なケース

テキストファイルやプログラムのバイナリなど、1 ビットでも変化すると意味が変わるデータには可逆圧縮を使わなければならない。ZIP アーカイブでプログラムを配布するとき、展開後のファイルが元と 1 バイトでも異なれば動作しなくなる。

#### 非可逆圧縮が許容されるケース

画像・音声・動画など、人間の知覚特性を利用して「知覚できない情報」を除去しても実用上問題がないデータに対しては、非可逆圧縮が極めて高い圧縮率を達成する。JPEG 品質 75 の画像と非圧縮画像を比較して、多くの場合その差を肉眼で識別することは困難である。

---

## 2. 可逆圧縮アルゴリズム

### 2.1 ランレングス符号化（RLE: Run-Length Encoding）

RLE はデータ圧縮における最も単純なアルゴリズムであり、同じ値が連続する区間（ラン）を「値 + 繰り返し回数」の組で表現する。

#### 基本原理

```
  RLE の動作:

  入力:  AAAAAABBCCCCDDDDDDDD
         ~~~~~~^^~~~~^^^^^^^^
         A×6   B×2 C×4  D×8

  出力:  A6B2C4D8

  圧縮率: 20文字 → 8文字 = 60%削減

  ─────────────────────────────────────
  より現実的な例（バイナリデータ）:

  入力バイト列: FF FF FF FF FF 00 00 00 00 00 00 00 FF FF
                 ~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~~~~~~~  ~~~~~
                    FF × 5          00 × 7            FF × 2

  RLE出力: (FF, 5) (00, 7) (FF, 2)
           6バイト（3ペア × 2バイト）

  元: 14バイト → RLE: 6バイト = 57%削減
```

#### RLE の弱点

```
  RLE が逆効果となるケース:

  入力: ABCDEFGHIJ
  出力: A1B1C1D1E1F1G1H1I1J1

  元: 10文字 → RLE: 20文字 = 100%膨張！

  対策:
  1. リテラルモードの導入
     - 連続しない部分はそのまま出力（モードフラグで区別）
     - 例: [LIT:10]ABCDEFGHIJ  （リテラル10文字）
           [RUN:6]A             （ラン6回）

  2. 適用場面の選択
     - 同値の長い連続が頻出するデータにのみ使用
     - FAXデータ、単色領域の多い画像、スプレッドシートの空白セル等
```

#### Python による RLE 実装

```python
"""
ランレングス符号化（RLE）の実装

同一値の連続区間を (値, 回数) のペアに圧縮する。
最も単純な圧縮アルゴリズムであり、教育目的および
特定用途（FAX、BMP）で使用される。
"""
from typing import Union


def rle_encode(data: Union[str, bytes]) -> list[tuple]:
    """RLE でデータを符号化する。

    Args:
        data: 入力データ（文字列またはバイト列）

    Returns:
        (値, 連続回数) のタプルのリスト

    >>> rle_encode("AAABBBCCCCDD")
    [('A', 3), ('B', 3), ('C', 4), ('D', 2)]
    >>> rle_encode("ABCD")
    [('A', 1), ('B', 1), ('C', 1), ('D', 1)]
    >>> rle_encode("")
    []
    """
    if not data:
        return []

    result = []
    current = data[0]
    count = 1

    for i in range(1, len(data)):
        if data[i] == current:
            count += 1
        else:
            result.append((current, count))
            current = data[i]
            count = 1

    result.append((current, count))
    return result


def rle_decode(encoded: list[tuple]) -> str:
    """RLE 符号化されたデータを復号する。

    Args:
        encoded: (値, 連続回数) のタプルのリスト

    Returns:
        復号された文字列

    >>> rle_decode([('A', 3), ('B', 2), ('C', 4)])
    'AAABBCCCC'
    """
    return "".join(char * count for char, count in encoded)


def rle_compression_ratio(data: str) -> dict:
    """RLE 圧縮の効率を分析する。

    Args:
        data: 入力文字列

    Returns:
        圧縮効率の分析結果
    """
    encoded = rle_encode(data)
    # 各ペアを「文字 + 回数の桁数」で見積もる
    encoded_size = sum(1 + len(str(count)) for _, count in encoded)
    original_size = len(data)

    return {
        "original_size": original_size,
        "encoded_pairs": len(encoded),
        "encoded_size_estimate": encoded_size,
        "compression_ratio": round(original_size / encoded_size, 2) if encoded_size > 0 else 0,
        "effective": encoded_size < original_size,
    }


# 使用例
if __name__ == "__main__":
    test_cases = [
        "AAAAAABBCCCCDDDDDDDD",        # RLE向き
        "ABCDEFGHIJKLMNOP",             # RLE不向き
        "WWWWWWWWWWWWBWWWWWWWWWWWWBBB",  # 混在
        "A" * 1000,                     # 極端な繰り返し
    ]

    for data in test_cases:
        result = rle_compression_ratio(data)
        status = "効果的" if result["effective"] else "逆効果"
        print(f"入力: '{data[:40]}...' ({result['original_size']}文字)")
        print(f"  → {result['encoded_pairs']}ペア, "
              f"推定サイズ: {result['encoded_size_estimate']}文字, "
              f"圧縮率: {result['compression_ratio']}倍 [{status}]")
```

### 2.2 ハフマン符号（Huffman Coding）

1952 年に MIT の大学院生デヴィッド・ハフマンが発表したこのアルゴリズムは、可変長符号の中で最適な前置符号（prefix-free code）を構築する。出現頻度の高いシンボルに短い符号を、低いシンボルに長い符号を割り当てることで、全体の符号長を最小化する。

#### ハフマン符号の性質

```
  前置符号（Prefix-Free Code）の重要性:

    どの符号語も、他の符号語の接頭辞（prefix）になっていない。
    → 区切り記号なしに一意に復号可能（瞬時復号可能符号）

    例: A=0, B=10, C=110, D=111

      符号列: 01011001111100
      復号:   0|10|110|0|111|110|0
              A  B  C   A  D   C  A  ← 一意に決まる

    反例（前置符号でない場合）:
      A=0, B=01, C=1  ← A の符号「0」が B の符号「01」の接頭辞！
      符号列: 01 → A,C なのか B なのか曖昧
```

#### ハフマン木の構築手順

```
  入力: "ABRACADABRA" （11文字）

  ステップ1: 出現頻度を数える
  ┌──────┬──────┐
  │ 文字 │ 頻度 │
  ├──────┼──────┤
  │  A   │  5   │
  │  B   │  2   │
  │  R   │  2   │
  │  C   │  1   │
  │  D   │  1   │
  └──────┴──────┘

  ステップ2: 頻度の低いものから順にペアにして結合（優先度付きキュー使用）

    初期状態（頻度順）: C:1  D:1  B:2  R:2  A:5

    反復1: C(1) と D(1) を結合 → CD
           残り: B:2  [CD]:2  R:2  A:5

    反復2: B(2) と CD を結合 → B,CD
           残り: R:2  [B,CD]:4  A:5

    反復3: R(2) と B,CD を結合 → R,B,CD
           残り: A:5  [R,B,CD]:6

    反復4: A(5) と R,B,CD を結合 → A,R,B,CD
           根ノード完成

  ステップ3: 完成したハフマン木

              [11]
             /    \
           A(5)   [6]
                 /    \
              R(2)   [4]
                    /    \
                 B(2)   [2]
                       /    \
                    C(1)   D(1)

  ステップ4: 符号の割り当て（左の枝=0, 右の枝=1）

    A: 0        （1ビット）  ← 最頻出 → 最短符号
    R: 10       （2ビット）
    B: 110      （3ビット）
    C: 1110     （4ビット）
    D: 1111     （4ビット）  ← 最低頻度 → 最長符号

  ステップ5: 符号化

    A  B   R  A  C    A  D    A  B   R  A
    0  110 10 0  1110 0  1111 0  110 10 0

    = 0 110 10 0 1110 0 1111 0 110 10 0
    = 011010011100111101101 00
    = 23ビット

  結果:
    元データ:   11文字 × 8ビット = 88ビット
    ハフマン符号: 23ビット（+ 木の情報をヘッダとして保存する必要あり）
    符号部分の圧縮率: 88 / 23 ≈ 3.83倍

  平均符号長の計算:
    L = 5/11 × 1 + 2/11 × 2 + 2/11 × 3 + 1/11 × 4 + 1/11 × 4
      = 5/11 + 4/11 + 6/11 + 4/11 + 4/11
      = 23/11
      ≈ 2.09 ビット/シンボル

  エントロピーとの比較:
    H = -(5/11×log₂(5/11) + 2/11×log₂(2/11) + 2/11×log₂(2/11)
         + 1/11×log₂(1/11) + 1/11×log₂(1/11))
    H ≈ 2.04 ビット/シンボル

    → L = 2.09 ≥ H = 2.04 （シャノン限界を満たしている）
    → 差はわずか 0.05 ビット/シンボル（非常に効率的）
```

#### Python によるハフマン符号の完全実装

```python
"""
ハフマン符号の完全実装

ハフマン木の構築、符号化、復号化を行う。
優先度付きキュー（heapq）を使用して効率的に最適な前置符号を構築する。
"""
import heapq
from collections import Counter
from typing import Optional


class HuffmanNode:
    """ハフマン木のノード。"""

    def __init__(self, char: Optional[str], freq: int,
                 left: Optional['HuffmanNode'] = None,
                 right: Optional['HuffmanNode'] = None):
        self.char = char    # 葉ノードの場合のみ文字を持つ
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other: 'HuffmanNode') -> bool:
        """優先度付きキューでの比較用。"""
        return self.freq < other.freq

    def is_leaf(self) -> bool:
        return self.char is not None


class HuffmanCoding:
    """ハフマン符号化/復号化クラス。"""

    def __init__(self):
        self.root: Optional[HuffmanNode] = None
        self.codes: dict[str, str] = {}
        self.reverse_codes: dict[str, str] = {}

    def build_tree(self, text: str) -> HuffmanNode:
        """テキストからハフマン木を構築する。

        Args:
            text: 入力テキスト

        Returns:
            ハフマン木の根ノード

        Raises:
            ValueError: 空のテキストが渡された場合
        """
        if not text:
            raise ValueError("テキストが空です")

        freq = Counter(text)

        # 1文字しかない場合の特殊処理
        if len(freq) == 1:
            char = list(freq.keys())[0]
            self.root = HuffmanNode(char=None, freq=freq[char],
                                    left=HuffmanNode(char, freq[char]))
            self._generate_codes(self.root, "")
            return self.root

        # 優先度付きキューに葉ノードを追加
        heap: list[HuffmanNode] = []
        for char, count in freq.items():
            heapq.heappush(heap, HuffmanNode(char, count))

        # 頻度の低いノードから順に結合してハフマン木を構築
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = HuffmanNode(
                char=None,
                freq=left.freq + right.freq,
                left=left,
                right=right,
            )
            heapq.heappush(heap, merged)

        self.root = heap[0]
        self._generate_codes(self.root, "")
        return self.root

    def _generate_codes(self, node: Optional[HuffmanNode], code: str) -> None:
        """ハフマン木を走査して符号表を生成する。"""
        if node is None:
            return

        if node.is_leaf():
            # 木が1ノードしかない場合のフォールバック
            self.codes[node.char] = code if code else "0"
            self.reverse_codes[code if code else "0"] = node.char
            return

        self._generate_codes(node.left, code + "0")
        self._generate_codes(node.right, code + "1")

    def encode(self, text: str) -> str:
        """テキストをハフマン符号でエンコードする。

        Args:
            text: 入力テキスト

        Returns:
            ビット列を表す文字列（'0' と '1' の並び）

        >>> hc = HuffmanCoding()
        >>> hc.build_tree("ABRACADABRA")  # doctest: +ELLIPSIS
        <...>
        >>> encoded = hc.encode("ABRACADABRA")
        >>> len(encoded)  # 23ビット
        23
        """
        if not self.codes:
            self.build_tree(text)
        return "".join(self.codes[char] for char in text)

    def decode(self, encoded: str) -> str:
        """ハフマン符号化されたビット列を復号する。

        Args:
            encoded: '0' と '1' からなるビット列文字列

        Returns:
            復号されたテキスト
        """
        if self.root is None:
            raise ValueError("ハフマン木が構築されていません")

        result = []
        node = self.root

        for bit in encoded:
            node = node.left if bit == "0" else node.right

            if node.is_leaf():
                result.append(node.char)
                node = self.root

        return "".join(result)

    def get_code_table(self) -> dict[str, dict]:
        """符号表とその統計情報を返す。"""
        return {
            char: {
                "code": code,
                "length": len(code),
            }
            for char, code in sorted(self.codes.items(), key=lambda x: len(x[1]))
        }

    def print_tree(self, node: Optional[HuffmanNode] = None,
                   prefix: str = "", is_left: bool = True) -> None:
        """ハフマン木を視覚的に表示する。"""
        if node is None:
            node = self.root
        if node is None:
            return

        connector = "├── " if is_left else "└── "
        label = f"'{node.char}' ({node.freq})" if node.is_leaf() else f"[{node.freq}]"

        if prefix == "":
            print(label)
        else:
            print(prefix + connector + label)

        new_prefix = prefix + ("│   " if is_left else "    ")

        if not node.is_leaf():
            self.print_tree(node.left, new_prefix, True)
            self.print_tree(node.right, new_prefix, False)


# 使用例・動作確認
if __name__ == "__main__":
    text = "ABRACADABRA"
    print(f"入力テキスト: '{text}' ({len(text)}文字, {len(text)*8}ビット)")
    print()

    hc = HuffmanCoding()
    hc.build_tree(text)

    print("=== ハフマン木 ===")
    hc.print_tree()
    print()

    print("=== 符号表 ===")
    for char, info in hc.get_code_table().items():
        print(f"  '{char}': {info['code']} ({info['length']}ビット)")
    print()

    encoded = hc.encode(text)
    print(f"=== 符号化結果 ===")
    print(f"  ビット列: {encoded}")
    print(f"  長さ: {len(encoded)}ビット ({len(encoded)/8:.1f}バイト)")
    print(f"  圧縮率: {len(text)*8/len(encoded):.2f}倍")
    print()

    decoded = hc.decode(encoded)
    print(f"=== 復号結果 ===")
    print(f"  復号テキスト: '{decoded}'")
    print(f"  元テキストと一致: {decoded == text}")
```

### 2.3 算術符号化（Arithmetic Coding）

ハフマン符号は各シンボルに整数ビット長の符号語を割り当てるため、シンボルの確率が 2 のべき乗でない場合にエントロピーとの間に最大 1 ビット/シンボルの差が生じる。算術符号化はこの制約を克服し、エントロピーにより近い圧縮率を達成する。

```
  算術符号化の基本アイデア:

    メッセージ全体を [0, 1) の区間内の一つの実数として符号化する。

    例: シンボル確率 A=0.6, B=0.2, C=0.2  メッセージ: "BAC"

    ステップ1: 初期区間 [0.0, 1.0)
      A → [0.0, 0.6)
      B → [0.6, 0.8)
      C → [0.8, 1.0)

    ステップ2: 最初のシンボル B → 区間 [0.6, 0.8) を選択
      この区間内をさらに分割:
      A → [0.6, 0.72)     (0.6 + 0.6×0.2 = 0.72)
      B → [0.72, 0.76)
      C → [0.76, 0.8)

    ステップ3: 次のシンボル A → 区間 [0.6, 0.72) を選択
      この区間内をさらに分割:
      A → [0.6, 0.672)
      B → [0.672, 0.696)
      C → [0.696, 0.72)

    ステップ4: 次のシンボル C → 区間 [0.696, 0.72) を選択

    結果: "BAC" は区間 [0.696, 0.72) 内の任意の値で表現可能
           例えば 0.7 → 二進数で 0.1011... → 必要ビット数が最小化

  ハフマン符号との比較:
    ┌──────────────────┬────────────────────┬──────────────────────┐
    │ 特性              │ ハフマン符号        │ 算術符号化           │
    ├──────────────────┼────────────────────┼──────────────────────┤
    │ 符号単位          │ シンボル単位        │ メッセージ全体       │
    │ 最小符号長        │ 1ビット/シンボル    │ 理論的制限なし       │
    │ エントロピーとの差│ 最大1bit/symbol    │ 2ビット以内（全体）  │
    │ 適応型対応        │ 木の再構築が必要    │ 確率の更新のみ       │
    │ 計算コスト        │ 低い               │ やや高い             │
    │ 特許問題          │ なし               │ 過去に存在（現在失効）│
    │ 用途              │ DEFLATE, JPEG      │ JPEG2000, H.264/265 │
    └──────────────────┴────────────────────┴──────────────────────┘

  ANS（Asymmetric Numeral Systems）:
    2009年にヤレク・ドゥダが発表した新しいエントロピー符号化方式。
    算術符号化と同等の圧縮率を、ハフマン符号並みの高速で達成する。
    Zstandard, LZFSE (Apple), CRAM (バイオインフォマティクス) で採用。
```

### 2.4 LZ77（Lempel-Ziv 1977）

1977 年にアブラハム・レンペルとヤコブ・ジヴが発表した LZ77 は、辞書ベース圧縮の先駆けであり、現代の多くの圧縮アルゴリズム（DEFLATE, zstd, LZ4 等）の基盤となっている。

#### スライディングウィンドウの仕組み

```
  LZ77 の基本アイデア:
    過去に出現した文字列パターンを「(距離, 長さ)」の参照で置き換える。
    スライディングウィンドウ（検索バッファ + 先読みバッファ）を使用する。

  ┌─────────────────────────────────────────┐
  │          スライディングウィンドウ         │
  │  ┌──────────────────┬──────────────┐    │
  │  │  検索バッファ      │ 先読みバッファ │    │
  │  │  (既に処理済み)    │ (これから処理) │    │
  │  │  ← 32KB等 →      │ ← 258B等 →  │    │
  │  └──────────────────┴──────────────┘    │
  │         ↑ 最長一致を探す  ↑ 現在位置     │
  └─────────────────────────────────────────┘

  出力形式:
    一致あり → (distance, length)  距離と長さの参照
    一致なし → literal(byte)      リテラルバイト

  ─────────────────────────────────────────
  例: 入力 "AABCAABCAABC"

  処理の流れ:
  位置0: 'A' → 検索バッファに一致なし → リテラル 'A'
  位置1: 'A' → 距離1,長さ1の一致 → (1,1)
         ※ ただし短い一致はリテラルの方が効率的な場合もある
  位置2: 'B' → 一致なし → リテラル 'B'
  位置3: 'C' → 一致なし → リテラル 'C'
  位置4: 'AABCAABC' → 位置0からの"AABC"と一致
         → (4, 4) ← 距離4, 長さ4
  位置8: 'AABC' → 位置4からの"AABC"と一致
         → (4, 4)

  出力: lit(A), lit(A), lit(B), lit(C), (4,4), (4,4)

  ─────────────────────────────────────────
  自己参照の例（LZ77 の重要な特性）:

  入力: "ABABABABABAB"

  位置0: lit(A)
  位置1: lit(B)
  位置2: "ABABABABAB" → 位置0の "AB" と一致するが、
         先読みバッファが検索バッファを超えて一致を延長できる！
         → (2, 10) ← 距離2, 長さ10

  展開時: 位置2で(2,10)を展開
    → 位置0を参照 → A をコピー（位置2に書く）
    → 位置1を参照 → B をコピー（位置3に書く）
    → 位置2を参照 → A をコピー（位置4に書く）← 今書いたばかりの A！
    → ... と繰り返す

  これにより「ABABABABABAB」全体が lit(A), lit(B), (2,10) の
  わずか3トークンで表現できる。
```

#### Python による LZ77 の実装

```python
"""
LZ77 圧縮アルゴリズムの実装

スライディングウィンドウ方式で、過去に出現したパターンを
(距離, 長さ) の参照に置き換えることでデータを圧縮する。
DEFLATE, zstd 等の基盤となるアルゴリズム。
"""
from dataclasses import dataclass
from typing import Union


@dataclass
class LZ77Token:
    """LZ77 の出力トークン。"""
    is_literal: bool
    value: Union[str, None] = None      # リテラルの場合の文字
    distance: int = 0                    # 参照の場合の距離
    length: int = 0                      # 参照の場合の長さ

    def __repr__(self) -> str:
        if self.is_literal:
            return f"lit('{self.value}')"
        return f"({self.distance},{self.length})"


class LZ77:
    """LZ77 エンコーダ/デコーダ。

    Args:
        window_size: 検索バッファのサイズ（デフォルト: 4096）
        min_match: 参照として出力する最小一致長（デフォルト: 3）
    """

    def __init__(self, window_size: int = 4096, min_match: int = 3):
        self.window_size = window_size
        self.min_match = min_match

    def encode(self, data: str) -> list[LZ77Token]:
        """LZ77 でデータをエンコードする。

        Args:
            data: 入力文字列

        Returns:
            LZ77Token のリスト
        """
        tokens = []
        pos = 0

        while pos < len(data):
            best_distance = 0
            best_length = 0

            # 検索バッファの開始位置
            search_start = max(0, pos - self.window_size)

            # 検索バッファ内で最長一致を探す
            for i in range(search_start, pos):
                match_length = 0
                # 自己参照を許容するため、1文字ずつ比較
                while (pos + match_length < len(data) and
                       match_length < 258):  # 最大一致長の制限
                    # 自己参照: 距離内で循環参照
                    ref_pos = i + match_length
                    if ref_pos >= pos:
                        ref_pos = i + (match_length % (pos - i))
                    if data[ref_pos] == data[pos + match_length]:
                        match_length += 1
                    else:
                        break

                if match_length > best_length:
                    best_length = match_length
                    best_distance = pos - i

            if best_length >= self.min_match:
                tokens.append(LZ77Token(
                    is_literal=False,
                    distance=best_distance,
                    length=best_length,
                ))
                pos += best_length
            else:
                tokens.append(LZ77Token(
                    is_literal=True,
                    value=data[pos],
                ))
                pos += 1

        return tokens

    def decode(self, tokens: list[LZ77Token]) -> str:
        """LZ77 トークン列をデコードする。

        Args:
            tokens: LZ77Token のリスト

        Returns:
            復元された文字列
        """
        result = []

        for token in tokens:
            if token.is_literal:
                result.append(token.value)
            else:
                # 参照を展開（自己参照対応のため1文字ずつ）
                start = len(result) - token.distance
                for i in range(token.length):
                    result.append(result[start + i])

        return "".join(result)


# 使用例
if __name__ == "__main__":
    encoder = LZ77(window_size=256, min_match=3)

    test_cases = [
        "ABCABCABCABC",
        "AABCAABCAABC",
        "the cat sat on the mat",
        "ABABABABABABABABABAB",
    ]

    for data in test_cases:
        tokens = encoder.encode(data)
        decoded = encoder.decode(tokens)

        print(f"入力: '{data}'")
        print(f"  トークン: {tokens}")
        print(f"  トークン数: {len(tokens)} (元: {len(data)}文字)")
        print(f"  復号: '{decoded}'")
        print(f"  一致: {decoded == data}")
        print()
```

### 2.5 LZ78 / LZW

LZ77 がスライディングウィンドウを用いるのに対し、LZ78（1978 年）は明示的な辞書を構築するアプローチを取る。LZW（Lempel-Ziv-Welch, 1984 年）はその改良版であり、GIF 画像形式で広く使用された。

```
  LZ78 の辞書構築:

    入力: "ABABABCABC"

    辞書（初期状態: 空）:

    ステップ | 入力  | 辞書にある？ | 出力        | 辞書に追加
    ─────────┼───────┼──────────────┼─────────────┼──────────
    1        | A     | No           | (0, 'A')    | 1: "A"
    2        | B     | No           | (0, 'B')    | 2: "B"
    3        | AB    | Yes→No(AB)  | (1, 'B')    | 3: "AB"
    4        | A     | Yes          |             |
    4        | AB    | Yes          |             |
    4        | ABC   | No           | (3, 'C')    | 4: "ABC"
    5        | A     | Yes          |             |
    5        | AB    | Yes          |             |
    5        | ABC   | Yes→(done)  | (4, '')     | 5: "ABCA"

    出力: (0,A) (0,B) (1,B) (3,C) (4,EOF)

  LZW の改良点:
    - 辞書を全シンボル（0-255）で初期化
    - 出力は辞書インデックスのみ（文字を含まない）
    - より単純で高速な実装が可能

  ─────────────────────────────────────────
  LZW の GIF 特許問題:

    1985年: Unisys（旧Sperry）が LZW の特許を取得
    1994年: CompuServe が GIF に LZW を採用していることが判明
            → Unisys がライセンス料を要求
    1995年: PNG 形式の開発が開始（LZW の代わりに DEFLATE を使用）
    2003年: 米国での特許期限切れ
    2004年: 全世界での特許期限切れ

    教訓: ソフトウェア特許が標準技術に与えるリスクの代表例
          → オープンな標準とロイヤリティフリーの重要性

  LZ77 vs LZ78 の比較:
    ┌────────────┬──────────────────┬──────────────────┐
    │ 特性        │ LZ77             │ LZ78             │
    ├────────────┼──────────────────┼──────────────────┤
    │ 辞書の種類  │ 暗黙的（窓内）   │ 明示的（辞書表）  │
    │ メモリ使用  │ 窓サイズに比例   │ 辞書サイズに比例  │
    │ 圧縮率      │ 一般に良好       │ LZ77とほぼ同等   │
    │ 復号速度    │ 高速             │ 高速             │
    │ 代表的応用  │ DEFLATE, zstd    │ GIF (LZW)       │
    │ 特許問題    │ なし             │ LZW に過去あり   │
    └────────────┴──────────────────┴──────────────────┘
```

### 2.6 DEFLATE アルゴリズム

DEFLATE はフィル・カッツが 1993 年に発表したアルゴリズムであり、LZ77 とハフマン符号を組み合わせた二段階圧縮を行う。ZIP, gzip, zlib, PNG, HTTP 圧縮など極めて広範に使用されており、インターネット時代の最も重要な圧縮アルゴリズムの一つと言える。

```
  DEFLATE の処理フロー:

  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────┐
  │ 入力データ │ → │ LZ77 圧縮     │ → │ ハフマン符号化 │ → │ 圧縮データ │
  └──────────┘    │              │    │             │    └──────────┘
                  │ リテラルと    │    │ リテラル/   │
                  │ (距離,長さ)  │    │ 長さ/距離を │
                  │ のトークン列  │    │ ハフマン符号で│
                  │ に変換        │    │ さらに圧縮   │
                  └──────────────┘    └─────────────┘

  詳細:

  ステージ1 - LZ77:
    - 検索バッファ: 最大32KB
    - 先読みバッファ: 最大258バイト
    - 一致長: 3〜258 バイト
    - 出力: リテラルバイト (0-255) と 一致参照 (距離, 長さ)

  ステージ2 - ハフマン符号:
    DEFLATE は二つの独立したハフマン木を使用する:
    1. リテラル/長さ用の木（シンボル 0-285）
       - 0-255: リテラルバイト
       - 256: ブロック終端マーカー
       - 257-285: 一致長（3〜258のレンジに対応）
    2. 距離用の木（シンボル 0-29）
       - 各シンボルが距離レンジに対応

  ブロック構造:
    DEFLATE データは複数のブロックで構成される。
    各ブロックは以下の3タイプのいずれか:

    タイプ0: 無圧縮ブロック（データをそのまま格納）
    タイプ1: 固定ハフマン符号（事前定義されたハフマン木を使用）
    タイプ2: 動的ハフマン符号（ブロック固有のハフマン木を生成）

    通常のデータではタイプ2が最も効率的だが、
    ハフマン木自体のヘッダオーバーヘッドがあるため、
    小さなブロックではタイプ1の方が効率的な場合もある。

  ─────────────────────────────────────────
  DEFLATE を使用する形式:

    ┌──────────┬───────────────────────────────────────────┐
    │ 形式      │ 説明                                      │
    ├──────────┼───────────────────────────────────────────┤
    │ ZIP       │ アーカイブ + DEFLATE（複数ファイル対応）   │
    │ gzip      │ DEFLATE + CRC32 + ヘッダ（単一ストリーム）│
    │ zlib      │ DEFLATE + Adler-32（ライブラリ形式）      │
    │ PNG       │ フィルタリング + zlib（画像用）           │
    │ HTTP      │ Content-Encoding: gzip / deflate          │
    │ PDF       │ FlateDecode フィルタ                      │
    │ JAR/WAR   │ Java アーカイブ（ZIP 形式ベース）         │
    │ DOCX/XLSX │ Office Open XML（ZIP 形式ベース）         │
    └──────────┴───────────────────────────────────────────┘
```

### 2.7 バロウズ-ウィーラー変換（BWT）

BWT は圧縮アルゴリズムそのものではなく、データの並び替え変換（可逆）であり、後続の圧縮（MTF + エントロピー符号化）の効率を劇的に向上させる。bzip2 の中核技術である。

```
  BWT の仕組み:

  入力: "BANANA$" （$は終端記号）

  ステップ1: 全ての巡回シフトを生成
    BANANA$
    ANANA$B
    NANA$BA
    ANA$BAN
    NA$BANA
    A$BANAN
    $BANANA

  ステップ2: 辞書順にソート
    $BANANA     → 最後の文字: A
    A$BANAN     → 最後の文字: N
    ANA$BAN     → 最後の文字: N
    ANANA$B     → 最後の文字: B
    BANANA$     → 最後の文字: $ ← 元の文字列の位置（行番号4）
    NA$BANA     → 最後の文字: A
    NANA$BA     → 最後の文字: A

  ステップ3: 最後の列を取得
    BWT出力: "ANNB$AA" （+ 元の位置: 4）

  効果: 同じ文字が近くに集まる傾向がある!
    → "ANNB$AA" は "A" が近くに集中
    → Move-to-Front (MTF) 変換との相性が極めて良い
    → その後のエントロピー符号化が高効率に

  bzip2 の圧縮パイプライン:
    入力 → BWT → MTF → RLE → ハフマン符号 → 出力
```

### 2.8 現代の圧縮アルゴリズム比較

```
  アルゴリズムの進化と特性:

  ┌────────────┬──────┬────────┬────────┬────────┬──────────────────┐
  │ 名称        │ 年   │ 圧縮率 │圧縮速度│展開速度│ 主な用途          │
  │            │      │(高=良) │(MB/s)  │(MB/s)  │                  │
  ├────────────┼──────┼────────┼────────┼────────┼──────────────────┤
  │ gzip       │ 1992 │ ★★★   │ 36     │ 410    │ HTTP, 汎用       │
  │ bzip2      │ 1996 │ ★★★★  │ 14     │ 56     │ アーカイブ        │
  │ LZ4        │ 2011 │ ★★    │ 780    │ 4,560  │ DB, リアルタイム  │
  │ Snappy     │ 2011 │ ★★    │ 565    │ 1,800  │ BigTable, Kafka  │
  │ Brotli     │ 2015 │ ★★★★★ │ 5      │ 430    │ HTTP (Google)    │
  │ zstd       │ 2016 │ ★★★★  │ 510    │ 1,380  │ 汎用 (Meta)      │
  │ LZMA2      │ 2001 │ ★★★★★ │ 3      │ 120    │ 7z, xz           │
  └────────────┴──────┴────────┴────────┴────────┴──────────────────┘

  速度 vs 圧縮率のトレードオフ図:

  圧縮率（高い）
    ↑
    │  ×LZMA2
    │  ×Brotli
    │       ×bzip2
    │         ×zstd
    │    ×gzip
    │
    │                    ×Snappy
    │                     ×LZ4
    │
    └──────────────────────────────→ 圧縮速度（速い）

  Zstandard (zstd) の特徴:
    - Meta（旧Facebook）のヤン・コレによって開発
    - 圧縮レベル 1-22 で速度と圧縮率を柔軟に調整可能
    - レベル1: LZ4並みの速度で gzip以上の圧縮率
    - レベル19: LZMA に迫る圧縮率
    - 辞書圧縮: 小さなデータ（JSON等）に対して事前辞書を適用可能
    - RFC 8478, RFC 8878 で標準化済み

  Brotli の特徴:
    - Google が Web 配信向けに開発
    - 静的辞書を内蔵（HTML/CSS/JS の頻出パターン）
    - Web コンテンツに対して gzip より 20-26% 小さい出力
    - RFC 7932 で標準化、主要ブラウザが全対応
    - CDN での事前圧縮に最適（圧縮は遅いが展開は高速）

  選択ガイドライン:
  ┌──────────────────────┬──────────────────────────────┐
  │ 用途                  │ 推奨アルゴリズム             │
  ├──────────────────────┼──────────────────────────────┤
  │ DB内部/メッセージキュー│ LZ4 or Snappy（超低遅延）   │
  │ ファイルシステム       │ zstd（バランス最良）         │
  │ Web 静的配信          │ Brotli（最高圧縮率 + CDN）  │
  │ Web 動的圧縮          │ gzip or zstd（速度と互換性） │
  │ アーカイブ保存         │ LZMA2/7z（最高圧縮率）      │
  │ ログ圧縮              │ zstd（辞書圧縮が効果的）     │
  │ レガシー互換           │ gzip（最も広くサポート）     │
  └──────────────────────┴──────────────────────────────┘
```

---

## 3. 非可逆圧縮

非可逆圧縮は、人間の知覚特性を利用して、知覚されにくい情報を除去することで可逆圧縮を大幅に超える圧縮率を達成する。

### 3.1 画像圧縮

#### JPEG 圧縮の詳細

```
  JPEG 圧縮のパイプライン:

  ┌──────┐  ┌────────┐  ┌─────┐  ┌──────┐  ┌──────────┐  ┌──────┐
  │ RGB  │→│YCbCr   │→│ダウン│→│ DCT  │→│ 量子化    │→│ エント│
  │ 画像 │  │変換    │  │サンプ│  │ 変換 │  │          │  │ロピー│
  │      │  │        │  │リング│  │(8×8) │  │(情報損失)│  │符号化│
  └──────┘  └────────┘  └─────┘  └──────┘  └──────────┘  └──────┘

  各ステップの詳細:

  1. 色空間変換（RGB → YCbCr）:
     Y  = 輝度（明るさ）    → 人間の目が最も敏感
     Cb = 青色差（色差成分）  → 感度が低い
     Cr = 赤色差（色差成分）  → 感度が低い

     計算式:
       Y  =  0.299R + 0.587G + 0.114B
       Cb = -0.169R - 0.331G + 0.500B + 128
       Cr =  0.500R - 0.419G - 0.081B + 128

  2. クロマサブサンプリング（ダウンサンプリング）:
     4:4:4 → 色差を間引かない（最高品質）
     4:2:2 → 色差を水平方向に半分（放送品質）
     4:2:0 → 色差を水平・垂直とも半分（JPEG標準）
             → データ量が 4:4:4 の半分に

  3. DCT（離散コサイン変換）:
     8×8 ピクセルブロックごとに周波数領域へ変換
     → 左上: DC成分（ブロック平均輝度）
     → 右下に行くほど高周波成分（細かい模様）

     DCT変換前（空間領域）      DCT変換後（周波数領域）
     ┌─┬─┬─┬─┬─┬─┬─┬─┐      ┌──┬──┬──┬──┬──┬──┬──┬──┐
     │各│ピ│ク│セ│ル│の│輝│度│      │DC│低│低│ ：│ ：│ ：│ ：│高│
     ├─┼─┼─┼─┼─┼─┼─┼─┤      ├──┼──┼──┼──┼──┼──┼──┼──┤
     │値│が│空│間│的│に│並│ぶ│      │低│ ：│ ：│ ：│ ：│ ：│ ：│ ：│
     │ │ │ │ │ │ │ │ │      │ ：│ ：│ ：│ ：│ ：│ ：│ ：│ ：│
     │ │ │ │ │ │ │ │ │      │ ：│ ：│ ：│ ：│ ：│ ：│ ：│ ：│
     │ │ │ │ │ │ │ │ │      │高│ ：│ ：│ ：│ ：│ ：│ ：│高│
     └─┴─┴─┴─┴─┴─┴─┴─┘      └──┴──┴──┴──┴──┴──┴──┴──┘

  4. 量子化（ここで情報が失われる）:
     DCT 係数を量子化テーブルで割り、丸める。
     品質設定が低いほど大きな値で割る → 高周波が 0 になる → 情報損失大

     量子化前:              量子化テーブル(Q50):    量子化後:
     [120  -5  3  1]       [16  11  10  16]        [8  0  0  0]
     [ -8   2 -1  0]  ÷    [12  12  14  19]   =    [0  0  0  0]
     [  3  -1  0  0]       [14  13  16  24]        [0  0  0  0]
     [  0   0  0  0]       [14  17  22  29]        [0  0  0  0]
     （簡略化した4×4の例）

     → 大量のゼロが発生 → ジグザグスキャン → RLE で効率的に圧縮

  5. エントロピー符号化:
     - ジグザグスキャン（低周波→高周波の順にスキャン）
     - DC成分: 前ブロックとの差分をハフマン符号化
     - AC成分: ゼロラン + 非ゼロ係数をハフマン符号化

  JPEG 品質と圧縮率の関係（1920×1080 写真）:
    ┌────────┬──────────┬──────────┬──────────────────────┐
    │ 品質   │ サイズ   │ 圧縮率   │ 視覚的品質           │
    ├────────┼──────────┼──────────┼──────────────────────┤
    │ 100    │ 約2.0 MB │ 3:1      │ ほぼ無損失           │
    │ 95     │ 約800 KB │ 7.5:1   │ 違いをほぼ識別不能   │
    │ 85     │ 約500 KB │ 12:1    │ Web用途に十分        │
    │ 75     │ 約300 KB │ 20:1    │ 一般的なデフォルト   │
    │ 50     │ 約150 KB │ 40:1    │ やや劣化が見える     │
    │ 25     │ 約80 KB  │ 75:1    │ ブロックノイズ顕著   │
    │ 10     │ 約50 KB  │ 120:1   │ 大幅な劣化           │
    └────────┴──────────┴──────────┴──────────────────────┘
```

#### 画像フォーマットの包括的比較

```
  ┌────────┬────────┬───────┬────────┬──────────┬────────────────────┐
  │ 形式   │ 圧縮   │ 透過  │ アニメ │ 色深度   │ 主な用途           │
  ├────────┼────────┼───────┼────────┼──────────┼────────────────────┤
  │ JPEG   │ 非可逆 │ ×    │ ×     │ 24bit    │ 写真全般           │
  │ PNG    │ 可逆   │ α    │ APNG  │ 48bit+α │ UI, スクリーンショット│
  │ GIF    │ 可逆   │ 1bit │ ○     │ 8bit     │ 簡易アニメ(256色)  │
  │ WebP   │ 両対応 │ α    │ ○     │ 24bit+α │ Web画像(Google)    │
  │ AVIF   │ 非可逆 │ α    │ ○     │ 36bit+α │ 次世代Web画像      │
  │ JPEG XL│ 両対応 │ α    │ ○     │ 32bit+α │ JPEG後継(新規格)   │
  │ HEIF   │ 非可逆 │ α    │ ○     │ 30bit   │ Apple写真          │
  │ SVG    │ -      │ ○    │ CSS   │ -        │ ベクターグラフィック│
  │ TIFF   │ 両対応 │ α    │ ×     │ 64bit   │ 印刷, 医療画像     │
  │ BMP    │ 無/RLE │ ×    │ ×     │ 32bit   │ Windows(非推奨)    │
  └────────┴────────┴───────┴────────┴──────────┴────────────────────┘

  次世代フォーマットの圧縮効率（JPEG比）:
    WebP:    約25-35% 小さい（同等画質）
    AVIF:    約50% 小さい（同等画質）
    JPEG XL: 約60% 小さい + JPEG からの可逆変換が可能
```

### 3.2 音声圧縮

```
  音声データの基本:

    非圧縮 CD 品質 (PCM):
      サンプリングレート: 44,100 Hz
      量子化ビット数: 16 bit
      チャンネル: 2（ステレオ）
      ビットレート: 44,100 × 16 × 2 = 1,411,200 bps ≈ 1.4 Mbps
      1分間: 約10 MB

  心理音響モデル（Psychoacoustic Model）:

    人間の聴覚特性を利用して知覚できない音を除去する。

    1. 絶対聴力閾値:
       周波数によって最小可聴音圧が異なる。
       1-5 kHz が最も敏感、20 Hz 以下と 20 kHz 以上は聞こえない。

    2. 周波数マスキング:
       強い音の近くの周波数帯の弱い音は知覚されない。
       例: 1 kHz で 60 dB の音がある場合、
           1.1 kHz で 40 dB の音は聞こえない → 除去可能

    3. 時間マスキング:
       大きな音の直前・直後の小さな音は知覚されない。
       前方マスキング: 約50-200 ms
       後方マスキング: 約5-20 ms

  音声コーデックの比較:
    ┌──────────┬──────┬────────────┬────────────┬────────────────┐
    │ コーデック│ 年   │ 推奨レート │ 品質の目安 │ 主な用途       │
    ├──────────┼──────┼────────────┼────────────┼────────────────┤
    │ MP3      │ 1993 │ 128-320kbps│ 標準〜高  │ 音楽配信(レガシー)│
    │ AAC      │ 1997 │ 96-256kbps │ 高品質    │ Apple, YouTube │
    │ Vorbis   │ 2000 │ 96-320kbps │ 高品質    │ ゲーム, Web    │
    │ Opus     │ 2012 │ 64-256kbps │ 最高品質  │ VoIP, Web, 万能│
    │ FLAC     │ 2001 │ 800-1200kbps│完全復元  │ オーディオ保存 │
    │ ALAC     │ 2004 │ 800-1200kbps│完全復元  │ Apple Music    │
    └──────────┴──────┴────────────┴────────────┴────────────────┘

  Opus の特筆すべき特性:
    - 6 kbps（音声通話）〜 510 kbps（高音質音楽）まで対応
    - 超低遅延: 最小 2.5 ms（リアルタイム通信向き）
    - SILK（音声向け）と CELT（音楽向け）のハイブリッド
    - IETF 標準 (RFC 6716)、完全ロイヤリティフリー
    - Discord, WebRTC, Signal 等で採用
```

### 3.3 動画圧縮

```
  動画圧縮の基本概念:

    非圧縮 1080p 30fps:
      1920 × 1080 × 3 bytes × 30 fps = 186,624,000 B/s
      ≈ 186 MB/秒 ≈ 1.49 Gbps
      → 1分間 = 約 11.2 GB ← 実用不可能

  フレーム間予測:

    ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
    │  I   │   │  P   │   │  B   │   │  B   │   │  P   │
    │フレーム│   │フレーム│   │フレーム│   │フレーム│   │フレーム│
    │      │──→│      │   │      │   │      │──→│      │
    │(完全)│   │(差分)│   │(双方向)│   │(双方向)│   │(差分)│
    └──────┘   └──────┘   └──────┘   └──────┘   └──────┘
         ↑──────────────────↑──↑──────────↑
               参照関係

    I フレーム（Intra）:
      - 単独で完結する画像（JPEG に近い圧縮）
      - ランダムアクセスの基点
      - 最もサイズが大きい（通常 P の 3-10 倍）

    P フレーム（Predicted）:
      - 前の I/P フレームとの「差分」のみ保存
      - 動き補償: 「この 16×16 ブロックは (dx, dy) 移動した」
      - 残差: 動き補償後のずれを DCT + 量子化で圧縮

    B フレーム（Bi-predictive）:
      - 前後のフレームから予測（双方向）
      - 最もサイズが小さい（P の 50-70%）
      - 復号時に前後のフレームが必要 → 遅延が発生

  動画コーデックの進化:
    ┌──────────┬──────┬────────────┬──────────────┬──────────────┐
    │ コーデック│ 年   │ ブロック   │ 対MPEG-2比   │ 特記事項     │
    ├──────────┼──────┼────────────┼──────────────┼──────────────┤
    │ MPEG-2   │ 1995 │ 16×16     │ (基準)       │ DVD, 放送    │
    │ H.264    │ 2003 │ 4×4〜16×16│ 50%削減      │ Blu-ray, Web │
    │ VP9      │ 2013 │ 4×4〜64×64│ H.264比30%減 │ YouTube      │
    │ H.265    │ 2013 │ 4×4〜64×64│ 75%削減      │ 4K/8K, Apple │
    │ AV1      │ 2018 │ 4×4〜128×128│ 75-80%削減 │ YouTube,Netflix│
    │ VVC      │ 2020 │ 4×4〜128×128│ 85%削減    │ 8K, 次世代放送│
    └──────────┴──────┴────────────┴──────────────┴──────────────┘

  AV1 の意義:
    - Alliance for Open Media (AOMedia) による開発
      → Google, Netflix, Amazon, Apple, Meta, Microsoft 等が参加
    - H.265(HEVC) と同等以上の圧縮率
    - 完全ロイヤリティフリー（HEVC の高額ライセンス問題を解決）
    - ハードウェアデコーダの普及が進行中
    - YouTube, Netflix, Twitch 等で段階的に採用拡大中
```

---

## 4. 実務での圧縮技術

### 4.1 HTTP 圧縮

Web のパフォーマンスにおいて転送サイズの削減は最も効果的な最適化の一つであり、HTTP 圧縮はその中核を担う。

```
  HTTP 圧縮のネゴシエーション:

  ブラウザ                                    サーバー
  │                                          │
  │  GET /index.html HTTP/1.1                │
  │  Accept-Encoding: gzip, deflate, br      │
  │ ────────────────────────────────────────→ │
  │                                          │
  │  HTTP/1.1 200 OK                         │
  │  Content-Encoding: br                    │
  │  Content-Type: text/html                 │
  │  Vary: Accept-Encoding                  │
  │  [Brotli 圧縮されたレスポンスボディ]      │
  │ ←──────────────────────────────────────── │
  │                                          │

  Content-Type 別の圧縮効果:
    ┌──────────────────────┬──────────┬──────────────────────────┐
    │ Content-Type         │ 圧縮効果 │ 理由                     │
    ├──────────────────────┼──────────┼──────────────────────────┤
    │ text/html            │ 70-85%   │ 繰り返しタグが多い       │
    │ text/css             │ 75-85%   │ 規則的な構造             │
    │ application/javascript│ 70-80%  │ 変数名・構文の繰り返し   │
    │ application/json     │ 75-90%   │ キー名の繰り返しが多い   │
    │ image/jpeg           │ 0-2%     │ 既に非可逆圧縮済み       │
    │ image/png            │ 0-5%     │ 既にDEFLATE圧縮済み     │
    │ font/woff2           │ 0-1%     │ 既にBrotli圧縮済み      │
    │ image/svg+xml        │ 60-75%   │ XMLテキストなので効果的  │
    └──────────────────────┴──────────┴──────────────────────────┘

  Nginx での圧縮設定例:
    # gzip 設定
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;           # 1-9, 6がバランス良好
    gzip_min_length 256;         # 256バイト未満は圧縮しない
    gzip_types
        text/plain
        text/css
        text/xml
        application/json
        application/javascript
        application/xml
        image/svg+xml;

    # 画像・動画・フォントは圧縮対象外（既に圧縮済み）

  Brotli vs gzip の使い分け:
    - 静的ファイル（CDN 配信）: Brotli レベル11（事前圧縮）
    - 動的レスポンス（API）: gzip レベル6 または zstd（速度重視）
    - レガシーブラウザ対応: gzip をフォールバックとして併用
```

### 4.2 データベースの圧縮

```
  主要データベースの圧縮機能:

  PostgreSQL - TOAST (The Oversized-Attribute Storage Technique):
    - 2KB 超のカラムデータを自動的に LZ 系で圧縮
    - PostgreSQL 14+ では LZ4 も選択可能
    - ユーザーが意識せずとも透過的に動作
    - ALTER TABLE t ALTER COLUMN c SET COMPRESSION lz4;

  MySQL InnoDB:
    - ページ圧縮 (ROW_FORMAT=COMPRESSED)
    - 透過的ページ圧縮 (innodb_compression_algorithm)
    - パンチホール圧縮（ファイルシステムレベル）

  列指向データベースの圧縮:
    同じ型のデータが列方向に並ぶため、圧縮効率が非常に高い。

    ┌─────────────────────────────────────────────────┐
    │ 行指向（MySQL, PostgreSQL）:                     │
    │ [Name:Alice, Age:30, City:Tokyo]                │
    │ [Name:Bob,   Age:25, City:Osaka]                │
    │ [Name:Carol, Age:30, City:Tokyo]                │
    │ → 異なる型が混在 → 圧縮しにくい                  │
    │                                                  │
    │ 列指向（ClickHouse, Parquet）:                   │
    │ Name: [Alice, Bob, Carol, ...]   → 辞書圧縮     │
    │ Age:  [30, 25, 30, ...]          → Delta + RLE  │
    │ City: [Tokyo, Osaka, Tokyo, ...] → 辞書圧縮     │
    │ → 同型データが連続 → 圧縮が非常に効果的          │
    └─────────────────────────────────────────────────┘

    列指向圧縮の主な手法:
    - 辞書圧縮 (Dictionary Encoding):
      カーディナリティの低いカラム（性別、国名等）に有効
      ["Tokyo","Osaka","Tokyo","Tokyo"] → 辞書{0:"Tokyo",1:"Osaka"} + [0,1,0,0]

    - デルタ圧縮 (Delta Encoding):
      タイムスタンプ等の連続値に有効
      [1000, 1001, 1003, 1005] → [1000, +1, +2, +2]

    - ビットパッキング:
      値の範囲が狭い場合、必要最小限のビット数で格納
      [1, 3, 2, 0, 3] → 各値2ビットで格納（8ビットではなく）

    - RLE (Run-Length Encoding):
      ソート済みカラムで同値が連続する場合に有効
      [Tokyo, Tokyo, Tokyo, Osaka, Osaka] → [(Tokyo,3), (Osaka,2)]
```

### 4.3 Python による圧縮ライブラリの活用

```python
"""
Python 標準ライブラリ・サードパーティによる圧縮の比較

実務でよく使用する圧縮ライブラリの使い方と性能比較。
"""
import gzip
import zlib
import bz2
import lzma
import time
import os
from typing import Callable


def compress_benchmark(data: bytes,
                       compressors: dict[str, Callable]) -> list[dict]:
    """複数の圧縮アルゴリズムのベンチマークを実行する。

    Args:
        data: 圧縮対象のバイト列
        compressors: {名前: 圧縮関数} の辞書

    Returns:
        各アルゴリズムの結果リスト
    """
    results = []
    original_size = len(data)

    for name, compress_func in compressors.items():
        # 圧縮
        start = time.perf_counter()
        compressed = compress_func(data)
        compress_time = time.perf_counter() - start

        compressed_size = len(compressed)
        ratio = original_size / compressed_size if compressed_size > 0 else 0

        results.append({
            "name": name,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "ratio": round(ratio, 2),
            "reduction_pct": round((1 - compressed_size / original_size) * 100, 1),
            "compress_time_ms": round(compress_time * 1000, 2),
        })

    return sorted(results, key=lambda x: x["compressed_size"])


# 圧縮関数の定義
compressors = {
    "zlib (level 6)":   lambda d: zlib.compress(d, 6),
    "zlib (level 9)":   lambda d: zlib.compress(d, 9),
    "gzip (level 6)":   lambda d: gzip.compress(d, compresslevel=6),
    "gzip (level 9)":   lambda d: gzip.compress(d, compresslevel=9),
    "bz2 (level 9)":    lambda d: bz2.compress(d, compresslevel=9),
    "lzma (preset 6)":  lambda d: lzma.compress(d, preset=6),
}

# テストデータの生成
test_data = b"Hello, World! " * 10000  # 繰り返しの多いデータ

results = compress_benchmark(test_data, compressors)
print(f"元データ: {len(test_data):,} bytes")
print(f"{'アルゴリズム':<20} {'圧縮後':>10} {'圧縮率':>8} {'削減':>8} {'時間':>10}")
print("-" * 60)
for r in results:
    print(f"{r['name']:<20} {r['compressed_size']:>10,} {r['ratio']:>7.1f}x "
          f"{r['reduction_pct']:>6.1f}% {r['compress_time_ms']:>8.2f}ms")
```

---

## 5. アンチパターンと設計上の注意点

### アンチパターン 1: 二重圧縮の罠

```
  問題:
    既に圧縮されたデータをさらに圧縮しようとすること。
    結果として圧縮率の向上はほぼなく、CPU 時間だけが浪費される。

  典型的な誤り:
    ┌──────────────────────────────────────────────────────┐
    │ × JPEG画像をZIP圧縮してからgzip転送                   │
    │   → JPEG は既に圧縮済み。ZIP/gzip は効果なし         │
    │                                                      │
    │ × woff2 フォントに Content-Encoding: gzip を適用     │
    │   → woff2 は内部で Brotli 圧縮済み                  │
    │                                                      │
    │ × 暗号化データを圧縮                                 │
    │   → 暗号化データはランダムに近く、圧縮不可能         │
    │   → しかも圧縮率の変動が暗号の手がかりになる         │
    │     （CRIME/BREACH 攻撃）                            │
    └──────────────────────────────────────────────────────┘

  正しいアプローチ:
    ┌──────────────────────────────────────────────────────┐
    │ ○ 圧縮してから暗号化（順序が重要！）                 │
    │   → 元データ → 圧縮 → 暗号化 → 転送                │
    │                                                      │
    │ ○ Content-Type に応じて圧縮の適用/非適用を制御       │
    │   → テキスト系: 圧縮する                            │
    │   → 画像/動画/フォント: 圧縮しない                  │
    │                                                      │
    │ ○ 異なる原理の組み合わせは有効                       │
    │   → LZ77 + ハフマン = DEFLATE（辞書 + 統計の融合）  │
    │   → BWT + MTF + ハフマン = bzip2                    │
    └──────────────────────────────────────────────────────┘
```

### アンチパターン 2: 圧縮レベル至上主義

```
  問題:
    常に最高の圧縮レベルを選択し、処理速度を考慮しないこと。
    圧縮率と速度のトレードオフを無視した設計。

  具体例:
    ┌──────────────────────────────────────────────────────┐
    │ × API レスポンスに Brotli レベル11 を適用            │
    │   → 圧縮に数百ミリ秒かかり、レイテンシが悪化        │
    │   → レベル4-6 で十分な圧縮率が得られる              │
    │                                                      │
    │ × リアルタイム DB に bzip2 を使用                    │
    │   → 展開速度が遅すぎて読み取りレイテンシが悪化      │
    │   → LZ4 or Snappy が適切                            │
    │                                                      │
    │ × 小さなデータ（< 1KB）に高圧縮アルゴリズムを適用   │
    │   → ヘッダのオーバーヘッドで逆にサイズ増加          │
    │   → gzip_min_length 256; 等で下限を設定             │
    └──────────────────────────────────────────────────────┘

  正しいアプローチ:
    ┌──────────────────────────────────────────────────────┐
    │ 用途に応じた圧縮レベルの選択:                         │
    │                                                      │
    │ リアルタイム系（DB, メッセージキュー）:               │
    │   → LZ4 / Snappy / zstd level 1-3                   │
    │   → 圧縮・展開とも 1ms 以下を目標                   │
    │                                                      │
    │ オンライン系（API, Web配信）:                        │
    │   → gzip level 4-6 / Brotli level 4-6               │
    │   → 圧縮時間 < 10ms を目標                          │
    │                                                      │
    │ オフライン系（アーカイブ, バックアップ）:             │
    │   → LZMA / Brotli level 11 / zstd level 19+         │
    │   → 圧縮時間は許容、最高圧縮率を追求                │
    └──────────────────────────────────────────────────────┘
```

### アンチパターン 3: CRIME / BREACH 攻撃の無視

```
  問題:
    TLS 上で圧縮を有効にしたまま、ユーザー入力を含むレスポンスを返すこと。

  攻撃原理:
    1. 攻撃者がリクエストに推測文字列を注入
    2. サーバーがレスポンスにシークレット（CSRFトークン等）を含める
    3. レスポンスを圧縮して暗号化して返す
    4. 推測が正しい → シークレットと一致 → 圧縮率が上がる → サイズ減少
    5. サイズの変化から攻撃者がシークレットを1文字ずつ特定

  対策:
    - TLS レベルの圧縮は無効にする（現在はデフォルトで無効）
    - HTTP 圧縮は SameSite Cookie + CSRF トークンと併用
    - セキュリティトークンに対するレスポンスでは圧縮を無効化することも検討
```

---

## 6. 実践演習

### 演習1（基礎）: ハフマン符号の手計算

以下の文字列に対してハフマン符号を手で構築せよ。

```
  課題: "MISSISSIPPI" （11文字）に対して:

  (a) 各文字の出現頻度を数えよ
  (b) ハフマン木を構築せよ（構築過程を図示すること）
  (c) 各文字の符号語を求めよ
  (d) "MISSISSIPPI" 全体の符号化後のビット数を計算せよ
  (e) エントロピーと比較し、ハフマン符号の効率を評価せよ
  (f) 固定長符号（各文字に等しいビット数を割り当て）と比較せよ

  ヒント:
    文字の出現頻度: I=4, S=4, P=2, M=1
    ユニークな文字数: 4 → 固定長なら 2ビット/文字
```

### 演習2（応用）: 圧縮アルゴリズムの実装と比較

```
  課題: 以下のプログラムを Python で実装せよ。

  (a) RLE エンコーダ/デコーダを実装し、以下のテストケースで検証:
      - "AAABBBCCCDDD" → 効果的
      - "ABCDEFGH"     → 逆効果
      - バイナリデータ（bytes 型対応）

  (b) ハフマン符号エンコーダ/デコーダを実装し:
      - "ABRACADABRA" で構築した符号表を表示
      - 符号化・復号化の正しさを検証
      - エントロピーとの差を計算

  (c) Python 標準ライブラリの圧縮関数で以下のファイルを圧縮し比較:
      - テキストファイル（10KB以上）
      - 繰り返しの多い人工データ
      - ランダムバイト列（os.urandom）
      圧縮率と処理時間を表にまとめよ。
```

### 演習3（発展）: LZ77 と DEFLATE の理解

```
  課題:

  (a) LZ77 エンコーダ/デコーダを実装せよ。
      - スライディングウィンドウサイズは可変（デフォルト4096）
      - 最小一致長は3
      - 自己参照（距離 < 長さ）を正しく処理すること
      テスト: "ABCABCABCABC" → トークン列 → 復号 → 元と一致

  (b) 上記の LZ77 出力に対してハフマン符号化を適用し、
      DEFLATE の基本的な二段階圧縮を模擬するプログラムを作成せよ。

  (c) 以下のデータに対して LZ77 のウィンドウサイズを
      64, 256, 1024, 4096 と変えた場合の圧縮率を測定し、
      ウィンドウサイズと圧縮率の関係をグラフにまとめよ。
      - テキストデータ（繰り返し多い）
      - テキストデータ（繰り返し少ない）
      - ランダムデータ

  発展:
  (d) BWT（バロウズ-ウィーラー変換）を実装し、
      BWT → MTF → RLE → ハフマン のパイプラインで
      bzip2 の基本構造を再現せよ。
```

---

## 7. FAQ

### Q1: 既に圧縮されたデータをさらに圧縮できますか？

**A**: 原理的にほぼ不可能である。圧縮されたデータは冗長性が除去された状態にあり、エントロピーが限界に近い（ランダムに近い）。JPEG 画像を ZIP で圧縮してもファイルサイズはほぼ変わらないか、わずかに増加することすらある。

ただし、**異なる原理の圧縮を段階的に適用する**ことは効果的な場合がある。DEFLATE が LZ77（辞書ベース）とハフマン符号（統計ベース）を組み合わせるように、異なる種類の冗長性を段階的に除去する設計は有効である。

### Q2: ZIPとgzipとtar.gzの違いは何ですか？

**A**: それぞれ設計思想が異なる。

| 形式 | アーカイブ | 圧縮 | 個別ファイルアクセス | 文化圏 |
|------|-----------|------|---------------------|--------|
| ZIP | あり（複数ファイル統合） | あり（DEFLATE） | 可能（各ファイルが独立圧縮） | Windows |
| gzip | なし（単一ストリーム） | あり（DEFLATE） | N/A | Unix |
| tar | あり（複数ファイル統合） | なし | tar内を順次読み取り | Unix |
| tar.gz | あり（tar経由） | あり（全体を一括圧縮） | 不可（展開してからアクセス） | Unix |
| tar.zst | あり（tar経由） | あり（zstd） | Seekable zstd なら可能 | 新しい標準 |

ZIP は各ファイルを独立に圧縮するため個別アクセスが高速だが、ファイル間の冗長性を活用できない。tar.gz は全ファイルを連結してから一括圧縮するため圧縮率は高いが、途中のファイルへのアクセスには全体の展開が必要になる。

### Q3: なぜ動画ファイルは ZIP で圧縮しても小さくならないのですか？

**A**: 動画ファイル（MP4, MKV 等）は既に H.264 / H.265 / AV1 等の高度な非可逆圧縮が適用された状態にある。元の非圧縮動画データから通常 100〜1000 倍に圧縮されており、残存する冗長性は極めて少ない。ZIP や gzip の可逆圧縮アルゴリズムがこれ以上の冗長性を発見することは原理的に困難である。

同様の理由で、JPEG 画像、MP3 音声、woff2 フォント等の既圧縮データに対しても二重圧縮は効果がない。HTTP サーバーの圧縮設定では、これらの Content-Type を圧縮対象から除外するのが正しい設計である。

### Q4: 圧縮アルゴリズムが特に有効なデータの特徴は何ですか？

**A**: 圧縮効率はデータの冗長性に直結する。以下の特徴を持つデータは高い圧縮率を期待できる。

1. **繰り返しパターンが多い**: ログファイル（同じフォーマットの行が大量）、HTML/XML（タグの繰り返し）、ソースコード（構文パターンの繰り返し）
2. **シンボルの出現頻度に偏りがある**: 自然言語テキスト（'e' や 'the' の高頻度出現）
3. **局所的な相関が強い**: 画像データ（隣接ピクセルの類似性）、時系列データ（値の漸次変化）

逆に、暗号化データ、既圧縮データ、ランダムバイト列は本質的に圧縮困難である。

### Q5: Web サービスでの圧縮はどう設計すべきですか？

**A**: 以下の多層的アプローチが推奨される。

1. **静的アセット**: ビルド時に Brotli（レベル11）で事前圧縮し、CDN から配信。gzip もフォールバック用に生成。
2. **動的 API レスポンス**: gzip レベル4-6 またはzstd でオンライン圧縮。Brotli は圧縮が遅いため動的用途には不向き。
3. **画像**: 用途に応じて WebP / AVIF を使用。`<picture>` 要素でフォーマットフォールバックを実装。
4. **データベース**: TOAST（PostgreSQL）や InnoDB 圧縮を活用。列指向 DB ではカラム圧縮を活用。
5. **メッセージキュー / キャッシュ**: LZ4 や Snappy で低遅延圧縮。

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## 8. まとめ

### 知識の整理表

| 概念 | 核心ポイント |
|------|-------------|
| エントロピー | データの情報量を表す指標。可逆圧縮の理論的下限を定める |
| シャノン限界 | どんなに優れた可逆圧縮もエントロピー以下には圧縮できない |
| RLE | 同値の連続を (値, 回数) で表現。最も単純だが適用場面が限定的 |
| ハフマン符号 | 出現頻度に基づく最適前置符号。DEFLATE, JPEG 等の内部で使用 |
| 算術符号化 | メッセージ全体を一つの数値で表現。ハフマンより高効率 |
| LZ77 | スライディングウィンドウで過去のパターンを参照。DEFLATE の基盤 |
| LZ78 / LZW | 明示的辞書を構築。GIF で使用（特許問題の歴史あり） |
| DEFLATE | LZ77 + ハフマン。ZIP, gzip, PNG, HTTP の中核 |
| BWT | データ並べ替え変換。bzip2 の中核技術 |
| zstd | 速度と圧縮率のバランスに最も優れる現代的アルゴリズム |
| Brotli | Web 配信向け最高圧縮率。静的辞書内蔵 |
| LZ4 / Snappy | 超高速圧縮/展開。リアルタイム・DB 向け |
| JPEG | DCT + 量子化 + ハフマン。写真の標準フォーマット |
| H.264 / AV1 | フレーム間予測 + 動き補償。動画の標準コーデック |
| HTTP 圧縮 | テキスト系で 70-85% 削減。画像等は対象外 |

### 選択フローチャート

```
  圧縮アルゴリズムの選択:

  データの種類は？
  ├── テキスト/バイナリ（可逆必須）
  │   ├── リアルタイム性が必要？
  │   │   ├── Yes → LZ4 or Snappy
  │   │   └── No
  │   │       ├── Web配信？
  │   │       │   ├── 静的 → Brotli (level 11)
  │   │       │   └── 動的 → gzip (level 6) or zstd (level 3)
  │   │       ├── アーカイブ保存？
  │   │       │   └── LZMA2 (7z) or zstd (level 19)
  │   │       └── 汎用？
  │   │           └── zstd (level 3-6)
  │   │
  ├── 画像
  │   ├── 可逆必須？ → PNG or WebP (lossless)
  │   ├── 写真？ → JPEG (quality 75-85) or WebP or AVIF
  │   └── UI/ロゴ？ → SVG (ベクター) or PNG
  │
  ├── 音声
  │   ├── 可逆必須？ → FLAC
  │   ├── 音楽配信？ → AAC or Opus (128-256 kbps)
  │   └── 通話/VoIP？ → Opus (32-64 kbps)
  │
  └── 動画
      ├── 最大互換性？ → H.264
      ├── 高効率？ → H.265 or AV1
      └── ロイヤリティフリー？ → AV1 or VP9
```

---

## 9. 次に読むべきガイド


---

## 10. 参考文献

### 基礎理論

1. Shannon, C. E. "A Mathematical Theory of Communication." *Bell System Technical Journal*, Vol. 27, pp. 379-423, 623-656, 1948. -- 情報理論の原論文。エントロピーの概念と情報源符号化定理を定義した。
2. Huffman, D. A. "A Method for the Construction of Minimum-Redundancy Codes." *Proceedings of the IRE*, Vol. 40, No. 9, pp. 1098-1101, 1952. -- ハフマン符号の原論文。MIT での学生課題から生まれた。
3. Cover, T. M. & Thomas, J. A. *Elements of Information Theory*. 2nd Edition, Wiley-Interscience, 2006. -- 情報理論の標準的教科書。エントロピー、符号化定理、Rate-Distortion理論を包括的に解説。

### 辞書ベース圧縮

4. Ziv, J. & Lempel, A. "A Universal Algorithm for Sequential Data Compression." *IEEE Transactions on Information Theory*, Vol. 23, No. 3, pp. 337-343, 1977. -- LZ77 の原論文。
5. Ziv, J. & Lempel, A. "Compression of Individual Sequences via Variable-Rate Coding." *IEEE Transactions on Information Theory*, Vol. 24, No. 5, pp. 530-536, 1978. -- LZ78 の原論文。
6. Welch, T. A. "A Technique for High-Performance Data Compression." *Computer*, Vol. 17, No. 6, pp. 8-19, 1984. -- LZW の原論文。

### 現代の圧縮技術

7. Collet, Y. "Zstandard Compression and the application/zstd Media Type." RFC 8478, IETF, 2018. -- Zstandard の RFC 仕様。
8. Alakuijala, J. & Szabadka, Z. "Brotli Compressed Data Format." RFC 7932, IETF, 2016. -- Brotli の RFC 仕様。
9. Deutsch, P. "DEFLATE Compressed Data Format Specification version 1.3." RFC 1951, IETF, 1996. -- DEFLATE の公式仕様。

### 画像・動画圧縮

10. Wallace, G. K. "The JPEG Still Picture Compression Standard." *Communications of the ACM*, Vol. 34, No. 4, pp. 30-44, 1991. -- JPEG 標準の解説論文。
11. Wiegand, T. et al. "Overview of the H.264/AVC Video Coding Standard." *IEEE Transactions on Circuits and Systems for Video Technology*, Vol. 13, No. 7, pp. 560-576, 2003. -- H.264 の包括的解説。
12. Chen, Y. et al. "An Overview of Core Coding Tools in the AV1 Video Codec." *2018 Picture Coding Symposium (PCS)*, IEEE, 2018. -- AV1 の技術概要。

### 補足資料

13. Duda, J. "Asymmetric Numeral Systems: Entropy Coding Combining Speed of Huffman Coding with Compression Rate of Arithmetic Coding." *arXiv:1311.2540*, 2013. -- ANS の原論文。zstd の基盤技術。
14. Burrows, M. & Wheeler, D. J. "A Block-sorting Lossless Data Compression Algorithm." *Technical Report 124*, Digital Equipment Corporation, 1994. -- BWT の原論文。bzip2 の基盤技術。

---

## 次に読むべきガイド

- [ストレージ容量と単位](./05-storage-capacity.md) - 次のトピックへ進む

---

## 参考文献

- [MDN Web Docs](https://developer.mozilla.org/) - Web技術のリファレンス
- [Wikipedia](https://ja.wikipedia.org/) - 技術概念の概要
