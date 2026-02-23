# 2進数と数値表現

> コンピュータの世界では全てが0と1で表現される。この「制約」こそが、デジタル技術の信頼性と普遍性を生み出す源泉である。

## この章で学ぶこと

- [ ] 2進数、8進数、16進数の変換ができる
- [ ] ビット演算（AND, OR, XOR, NOT, シフト）を使いこなせる
- [ ] なぜコンピュータが2進数を採用しているか説明できる
- [ ] 各基数間の変換を高速に行う手法を身につける
- [ ] ビット演算を実務で活用するパターンを理解する
- [ ] 数値リテラルを各プログラミング言語で正しく表記できる

## 前提知識

- 基本的な数学（四則演算）

---

## 1. なぜ2進数か

### 1.1 物理的な理由

```
トランジスタ = スイッチ:

  ON  = 電圧高い = 1
  OFF = 電圧低い = 0

  2つの状態しかないので「2進数」が自然

  もし3進数を使おうとすると:
  - 電圧を3段階に正確に区別する必要がある
  - ノイズに弱くなる（境界が2箇所）
  - 回路が複雑化

  2進数の利点:
  - ノイズに強い（しきい値が1つだけ）
  - 回路が単純（ON/OFFだけ）
  - 論理演算とビット演算が直接対応
```

### 1.2 歴史的背景

コンピュータが2進数を採用するに至った歴史的背景は、数学・物理学・工学の交差点にある。

```
年代別の重要な出来事:

  1679年  ライプニッツが2進法の体系を発表
          - 「0と1による普遍的な計算」を提唱
          - 中国の易経（陰陽思想）からの影響も指摘される

  1847年  ジョージ・ブールがブール代数を考案
          - 論理をTRUE/FALSEの2値で数学的に表現
          - AND, OR, NOT の基本演算を定式化

  1937年  クロード・シャノンの修士論文
          - 「リレー回路とスイッチング回路の記号的解析」
          - ブール代数と電気回路の対応関係を証明
          - デジタル回路設計の理論的基盤を確立

  1945年  フォン・ノイマンのEDVAC報告書
          - 2進数ベースの stored-program 方式を提案
          - 「2進表現は回路設計を大幅に簡素化する」

  1947年  トランジスタの発明（ベル研究所）
          - 真空管に代わる高信頼・小型のスイッチング素子
          - ON/OFF の2状態が極めて安定

  1958年  集積回路（IC）の発明
          - トランジスタを大量に集積可能に
          - 2進数の並列処理が実用的になる
```

### 1.3 情報理論からの視点

```
シャノンの情報理論（1948年）:

  情報量の基本単位 = bit（binary digit）

  1 bit = 「2つの等確率の選択肢から1つを選ぶ」のに必要な情報量

  情報エントロピー H = -Σ p(x) log₂ p(x)

  例: コイン投げ（表/裏が等確率）
    H = -(0.5 × log₂(0.5) + 0.5 × log₂(0.5))
    H = -(0.5 × (-1) + 0.5 × (-1))
    H = 1 bit

  例: 8面サイコロ（各面が等確率）
    H = -8 × (1/8 × log₂(1/8))
    H = -8 × (1/8 × (-3))
    H = 3 bits

  → 2進数は情報の「最小単位」を直接表現する
```

### 1.4 なぜ10進数ではないのか

```
10進数コンピュータの問題点:

  1. 回路の複雑さ
     - 10状態を区別する回路は、2状態の回路より遥かに複雑
     - エラー率が桁違いに高くなる

  2. 論理演算との不整合
     - ブール論理（TRUE/FALSE）は自然に2進数と対応
     - 10進数では論理演算が非効率

  3. 歴史的な試み
     - ENIAC（1946年）は10進数ベースだった
     - 10進数の加算器は2進数の約3倍の回路が必要
     - EDVAC以降、2進数に移行

  実際の比較（加算器の回路規模）:
     2進数 全加算器: ANDゲート2個 + XORゲート2個 + ORゲート1個
     10進数 加算器: 数十個のゲートが必要 + 繰り上がり処理が複雑
```

---

## 2. 基数と位取り記数法

### 2.1 位取り記数法の一般理論

```
N進法（基数N）の一般形:

  数値 = Σ(i=0 to n) d_i × N^i

  ここで d_i は各桁の数字（0 ≦ d_i < N）

  例: 10進数 4273
    = 4×10³ + 2×10² + 7×10¹ + 3×10⁰
    = 4000  + 200   + 70    + 3
    = 4273

  例: 2進数 1101
    = 1×2³ + 1×2² + 0×2¹ + 1×2⁰
    = 8    + 4    + 0    + 1
    = 13 (10進)

  例: 16進数 0x2AF
    = 2×16² + A(10)×16¹ + F(15)×16⁰
    = 512   + 160       + 15
    = 687 (10進)

  例: 8進数 0o755
    = 7×8² + 5×8¹ + 5×8⁰
    = 448   + 40   + 5
    = 493 (10進)
```

### 2.2 基数変換

```
10進数 ⇔ 2進数 ⇔ 16進数:

  10進数  2進数          16進数   8進数
  ──────────────────────────────────────
     0    0000 0000      0x00     000
     1    0000 0001      0x01     001
    10    0000 1010      0x0A     012
    42    0010 1010      0x2A     052
   127    0111 1111      0x7F     177
   128    1000 0000      0x80     200
   255    1111 1111      0xFF     377
   256    1 0000 0000    0x100    400

  変換方法（10進→2進）: 2で割り続けて余りを逆から並べる
  42 ÷ 2 = 21 余り 0
  21 ÷ 2 = 10 余り 1
  10 ÷ 2 =  5 余り 0
   5 ÷ 2 =  2 余り 1
   2 ÷ 2 =  1 余り 0
   1 ÷ 2 =  0 余り 1
  → 101010 (2) = 42 (10) ✓

  16進数は2進数の4ビットまとめ:
  0010 1010 → 2  A → 0x2A ✓
```

### 2.3 10進数から各基数への変換（詳細手順）

```
【方法1: 除算法（10進 → N進）】

  10進数 173 → 2進数:
  173 ÷ 2 = 86 余り 1
   86 ÷ 2 = 43 余り 0
   43 ÷ 2 = 21 余り 1
   21 ÷ 2 = 10 余り 1
   10 ÷ 2 =  5 余り 0
    5 ÷ 2 =  2 余り 1
    2 ÷ 2 =  1 余り 0
    1 ÷ 2 =  0 余り 1
  → 10101101 (2) ✓

  10進数 173 → 16進数:
  173 ÷ 16 = 10 余り 13 (D)
   10 ÷ 16 =  0 余り 10 (A)
  → 0xAD ✓

  検算: 0xAD = 10×16 + 13 = 160 + 13 = 173 ✓

  10進数 4096 → 8進数:
  4096 ÷ 8 = 512 余り 0
   512 ÷ 8 =  64 余り 0
    64 ÷ 8 =   8 余り 0
     8 ÷ 8 =   1 余り 0
     1 ÷ 8 =   0 余り 1
  → 0o10000 ✓ (= 8⁴ = 4096)


【方法2: 減算法（10進 → 2進）】

  2のべき乗の表を使う（暗記推奨）:
  2⁰=1, 2¹=2, 2²=4, 2³=8, 2⁴=16, 2⁵=32, 2⁶=64, 2⁷=128, 2⁸=256

  10進数 173 を2進に変換:
  173 ≧ 128 → 1 (173-128=45)
   45 ≧  64 → × → 0
   45 ≧  32 → 1 (45-32=13)
   13 ≧  16 → × → 0
   13 ≧   8 → 1 (13-8=5)
    5 ≧   4 → 1 (5-4=1)
    1 ≧   2 → × → 0
    1 ≧   1 → 1 (1-1=0)
  → 10101101 (2) ✓

  この方法は慣れると除算法より速い
```

### 2.4 2進数と16進数の相互変換（高速法）

```
2進数 → 16進数: 右から4ビットずつ区切って変換

  暗記すべき対応表:
  0000=0  0001=1  0010=2  0011=3
  0100=4  0101=5  0110=6  0111=7
  1000=8  1001=9  1010=A  1011=B
  1100=C  1101=D  1110=E  1111=F

  例: 1011 1110 0100 1101
      B    E    4    D
  → 0xBE4D

  16進数 → 2進数: 各桁を4ビットに展開

  例: 0xCAFE
      C    A    F    E
      1100 1010 1111 1110
  → 1100 1010 1111 1110


2進数 → 8進数: 右から3ビットずつ区切って変換

  例: 10 101 101
      2  5   5
  → 0o255

  8進数 → 2進数: 各桁を3ビットに展開

  例: 0o755
      7   5   5
      111 101 101
  → 111 101 101


実務でよく見る変換パターン:
  0xFF     = 1111 1111               = 255
  0xFFFF   = 1111 1111 1111 1111     = 65535
  0xDEAD   = 1101 1110 1010 1101     = 57005
  0xBEEF   = 1011 1110 1110 1111     = 48879
  0xCAFE   = 1100 1010 1111 1110     = 51966
  0xC0FFEE = 1100 0000 1111 1111 1110 1110 = 12648430
```

### 2.5 小数の基数変換

```
10進小数 → 2進小数: 2を掛けて整数部を取り出す

  0.625 を2進数に変換:
  0.625 × 2 = 1.25  → 整数部 1
  0.25  × 2 = 0.5   → 整数部 0
  0.5   × 2 = 1.0   → 整数部 1
  0.0   → 終了
  → 0.101 (2) ✓

  検算: 0.101 = 1×2⁻¹ + 0×2⁻² + 1×2⁻³
              = 0.5   + 0     + 0.125
              = 0.625 ✓


  0.1 を2進数に変換（循環小数になる例）:
  0.1 × 2 = 0.2  → 0
  0.2 × 2 = 0.4  → 0
  0.4 × 2 = 0.8  → 0
  0.8 × 2 = 1.6  → 1
  0.6 × 2 = 1.2  → 1
  0.2 × 2 = 0.4  → 0  ← ここから繰り返し
  0.4 × 2 = 0.8  → 0
  ...
  → 0.0001100110011... (2) = 0.0(0011)の循環

  これが「0.1 + 0.2 ≠ 0.3」問題の根本原因!
  → 03-floating-point.md で詳細を解説


2進小数 → 10進小数:
  0.1101 (2) = 1×2⁻¹ + 1×2⁻² + 0×2⁻³ + 1×2⁻⁴
             = 0.5   + 0.25  + 0     + 0.0625
             = 0.8125 (10)
```

---

## 3. ビット演算

### 3.1 基本演算

```python
# Pythonでのビット演算

# AND: 両方1なら1
a = 0b1100  # 12
b = 0b1010  # 10
print(bin(a & b))   # 0b1000 = 8

# OR: どちらか1なら1
print(bin(a | b))   # 0b1110 = 14

# XOR: 異なれば1
print(bin(a ^ b))   # 0b0110 = 6

# NOT: ビット反転
print(bin(~a & 0xFF))  # 0b11110011 = 243 (8ビットの場合)

# 左シフト: 2倍（各ビットを左にずらす）
print(bin(a << 1))  # 0b11000 = 24 (12 × 2)
print(bin(a << 3))  # 0b1100000 = 96 (12 × 8)

# 右シフト: 半分（各ビットを右にずらす）
print(bin(a >> 1))  # 0b110 = 6 (12 ÷ 2)
```

### 3.2 真理値表

```
AND演算（論理積）: 両方1のときだけ1
  A  B  │ A & B
  ───────┼──────
  0  0  │  0
  0  1  │  0
  1  0  │  0
  1  1  │  1

  用途: ビットマスク（特定ビットの抽出）
  例: 0b1011_0110 & 0b0000_1111 = 0b0000_0110（下位4ビットを抽出）


OR演算（論理和）: どちらか1なら1
  A  B  │ A | B
  ───────┼──────
  0  0  │  0
  0  1  │  1
  1  0  │  1
  1  1  │  1

  用途: フラグのセット（特定ビットを1にする）
  例: 0b0000_0001 | 0b0000_0100 = 0b0000_0101（ビット0とビット2をセット）


XOR演算（排他的論理和）: 異なるとき1
  A  B  │ A ^ B
  ───────┼──────
  0  0  │  0
  0  1  │  1
  1  0  │  1
  1  1  │  0

  XORの重要な性質:
  - A ^ A = 0（自分自身とのXORは0）
  - A ^ 0 = A（0とのXORは元の値）
  - A ^ B ^ B = A（2回XORすると元に戻る → 暗号で利用）
  - 交換法則: A ^ B = B ^ A
  - 結合法則: (A ^ B) ^ C = A ^ (B ^ C)


NOT演算（ビット反転）:
  A  │ ~A
  ───┼────
  0  │  1
  1  │  0

  注意: Pythonでは ~n = -(n+1) （2の補数表現）
  8ビットの場合: ~0b0000_1111 = 0b1111_0000


NAND演算（否定論理積）:
  A  B  │ ~(A & B)
  ───────┼─────────
  0  0  │  1
  0  1  │  1
  1  0  │  1
  1  1  │  0

  NANDはユニバーサルゲート: NANDだけで全ての論理演算を構成可能
  - NOT(A) = NAND(A, A)
  - AND(A, B) = NOT(NAND(A, B))
  - OR(A, B) = NAND(NOT(A), NOT(B))
```

### 3.3 シフト演算の詳細

```
左シフト（<<）: ビットを左にずらし、右端に0を埋める

  0b0000_1100 << 1 = 0b0001_1000  (12 → 24 = 12 × 2¹)
  0b0000_1100 << 2 = 0b0011_0000  (12 → 48 = 12 × 2²)
  0b0000_1100 << 3 = 0b0110_0000  (12 → 96 = 12 × 2³)

  一般則: x << n = x × 2ⁿ（オーバーフローしない場合）


論理右シフト（>>>）: ビットを右にずらし、左端に0を埋める
  - Java, JavaScriptの >>> 演算子
  - 符号なし整数として扱う

  0b1000_0000 >>> 1 = 0b0100_0000  (128 → 64)
  0b1000_0000 >>> 2 = 0b0010_0000  (128 → 32)


算術右シフト（>>）: ビットを右にずらし、左端に符号ビットを埋める
  - C, Python, Javaの >> 演算子（符号付き整数）
  - 負の数の場合、左端に1が埋まる

  正の数の場合:
  0b0110_0000 >> 1 = 0b0011_0000  (96 → 48)

  負の数の場合（8ビット符号付き）:
  0b1100_0000 >> 1 = 0b1110_0000  (-64 → -32)
  0b1100_0000 >> 2 = 0b1111_0000  (-64 → -16)

  一般則: x >> n = x ÷ 2ⁿ（負の無限大方向への丸め）


シフト演算 vs 乗除算のパフォーマンス:
  - 現代のCPUでは乗算もほぼ同速度（1クロック）
  - コンパイラが最適化で自動的にシフトに置き換える
  - 可読性を優先して x * 2 と書いても問題ない
  - ただし組み込みシステムでは依然としてシフトが有利な場合あり
```

### 3.4 ビット演算の実務応用

```python
# ビット演算の実務的なユースケース

# 1. フラグ管理（ビットフィールド）
READ    = 0b001  # 1
WRITE   = 0b010  # 2
EXECUTE = 0b100  # 4

permissions = READ | WRITE  # 0b011 = 3
has_read = bool(permissions & READ)     # True
has_execute = bool(permissions & EXECUTE)  # False

# ファイルパーミッション: chmod 755 = rwxr-xr-x
# 7 = 111, 5 = 101, 5 = 101

# 2. 偶奇判定（最下位ビット）
is_even = (n & 1) == 0  # n % 2 == 0 と同じだが高速

# 3. 2の冪乗判定
is_power_of_2 = n > 0 and (n & (n - 1)) == 0
# 8 = 1000, 7 = 0111 → 1000 & 0111 = 0000 → True
# 6 = 0110, 5 = 0101 → 0110 & 0101 = 0100 → False

# 4. XORスワップ（一時変数なしで交換）
a ^= b
b ^= a
a ^= b
# 理論的に面白いが、実用では temp = a; a = b; b = temp が読みやすい

# 5. ビットマスク
ip_address = 0xC0A80164  # 192.168.1.100
subnet_mask = 0xFFFFFF00  # 255.255.255.0
network = ip_address & subnet_mask  # 192.168.1.0
```

### 3.5 ビット演算の高度な技法

```python
# === ビットカウント（popcount / ハミング重み） ===

# 方法1: 素朴な方法
def popcount_naive(n):
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count

# 方法2: Brian Kernighanのアルゴリズム（高速）
def popcount_kernighan(n):
    """n & (n-1) は最下位の1ビットを消す"""
    count = 0
    while n:
        n &= n - 1  # 最下位の1ビットをクリア
        count += 1
    return count

# 例: n = 0b1011_0100 (180)
# 1011_0100 & 1011_0011 = 1011_0000 (count=1)
# 1011_0000 & 1010_1111 = 1010_0000 (count=2)
# 1010_0000 & 1001_1111 = 1000_0000 (count=3)
# 1000_0000 & 0111_1111 = 0000_0000 (count=4)
# → 4個の1ビット

# 方法3: Python組み込み
bin(180).count('1')  # 4

# 方法4: CPUの命令を直接使用（最速）
# x86の POPCNT 命令、ARMの VCNT 命令


# === 最下位セットビットの抽出 ===
def lowest_set_bit(n):
    """最下位の1ビットだけを残す"""
    return n & (-n)  # 2の補数の性質を利用

# 例: n = 0b1010_1000
# -n = 0b0101_1000 (2の補数)
# n & (-n) = 0b0000_1000 → ビット3が最下位の1


# === 最下位セットビットのクリア ===
def clear_lowest_set_bit(n):
    return n & (n - 1)

# 例: n = 0b1010_1000
# n-1 = 0b1010_0111
# n & (n-1) = 0b1010_0000


# === ビット反転（特定範囲） ===
def toggle_bits(n, mask):
    """maskで指定したビットを反転"""
    return n ^ mask

# 例: n = 0b1100_0011, mask = 0b0000_1111
# n ^ mask = 0b1100_1100 (下位4ビットが反転)


# === 2つの値の中間値（オーバーフローなし） ===
def average_no_overflow(a, b):
    """(a + b) / 2 のオーバーフロー安全版"""
    return (a & b) + ((a ^ b) >> 1)

# 通常の (a + b) / 2 は a + b がオーバーフローする可能性
# この方法は共通ビット + 異なるビット/2 で計算


# === 絶対値（分岐なし） ===
def abs_branchless(n):
    """32ビット符号付き整数の絶対値"""
    mask = n >> 31  # 正なら0x00000000、負なら0xFFFFFFFF
    return (n ^ mask) - mask

# 負の場合: mask = -1
# n ^ (-1) = ~n (ビット反転)
# ~n - (-1) = ~n + 1 = -n (2の補数)
```

### 3.6 ビット演算のアルゴリズム応用

```python
# === ビットボード（チェス・将棋のAI） ===
# 8x8のボードを64ビット整数で表現

# チェスの例: ナイトの移動可能位置
def knight_moves(position):
    """ナイトの移動先を計算（ビットボード）"""
    # 端のマスクで折り返しを防止
    NOT_A_FILE = 0xFEFEFEFEFEFEFEFE
    NOT_H_FILE = 0x7F7F7F7F7F7F7F7F
    NOT_AB_FILE = 0xFCFCFCFCFCFCFCFC
    NOT_GH_FILE = 0x3F3F3F3F3F3F3F3F

    moves = 0
    moves |= (position << 17) & NOT_A_FILE   # 上2右1
    moves |= (position << 15) & NOT_H_FILE   # 上2左1
    moves |= (position << 10) & NOT_AB_FILE  # 上1右2
    moves |= (position <<  6) & NOT_GH_FILE  # 上1左2
    moves |= (position >> 17) & NOT_H_FILE   # 下2左1
    moves |= (position >> 15) & NOT_A_FILE   # 下2右1
    moves |= (position >> 10) & NOT_GH_FILE  # 下1左2
    moves |= (position >>  6) & NOT_AB_FILE  # 下1右2
    return moves


# === ブルームフィルタ（確率的データ構造） ===
class BloomFilter:
    """ビット配列を使った確率的集合メンバーシップテスト"""

    def __init__(self, size=1024):
        self.size = size
        self.bit_array = 0  # Python の無限精度整数をビット配列として使用

    def _hashes(self, item):
        """複数のハッシュ値を生成"""
        h1 = hash(item) % self.size
        h2 = hash(str(item) + "salt") % self.size
        h3 = hash(str(item) + "pepper") % self.size
        return [h1, h2, h3]

    def add(self, item):
        for h in self._hashes(item):
            self.bit_array |= (1 << h)  # 対応ビットをセット

    def might_contain(self, item):
        for h in self._hashes(item):
            if not (self.bit_array & (1 << h)):
                return False  # 確実に含まれない
        return True  # 含まれるかもしれない（偽陽性の可能性あり）


# === ビットマニピュレーションによる部分集合列挙 ===
def enumerate_subsets(s):
    """ビットマスク s の全部分集合を列挙"""
    subset = s
    subsets = []
    while subset > 0:
        subsets.append(subset)
        subset = (subset - 1) & s
    subsets.append(0)  # 空集合
    return subsets

# 例: s = 0b1010 (要素2と要素0を含む集合)
# 部分集合: 1010, 1000, 0010, 0000
# → {2,0}, {2}, {0}, {}


# === ビットリバース（FFTで使用） ===
def reverse_bits(n, bit_width=8):
    """ビット列を逆順にする"""
    result = 0
    for _ in range(bit_width):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result

# 例: reverse_bits(0b10110010, 8) = 0b01001101
```

### 3.7 各言語でのビット演算

```c
// C言語: 低レベル操作に最適

#include <stdint.h>

// 符号なし整数の使用が推奨（シフト演算の挙動が明確）
uint32_t set_bit(uint32_t n, int pos) {
    return n | (1U << pos);
}

uint32_t clear_bit(uint32_t n, int pos) {
    return n & ~(1U << pos);
}

uint32_t toggle_bit(uint32_t n, int pos) {
    return n ^ (1U << pos);
}

int test_bit(uint32_t n, int pos) {
    return (n >> pos) & 1;
}

// GCCのビルトイン関数
int count = __builtin_popcount(0xFF);     // 8
int leading_zeros = __builtin_clz(0x10);  // 27
int trailing_zeros = __builtin_ctz(0x10); // 4
```

```java
// Java: Integer / Long のユーティリティメソッド

int n = 0b1010_1100;

// ビットカウント
int count = Integer.bitCount(n);        // 4

// 先頭/末尾のゼロ数
int leadingZeros = Integer.numberOfLeadingZeros(n);   // 24
int trailingZeros = Integer.numberOfTrailingZeros(n);  // 2

// ビット反転
int reversed = Integer.reverse(n);

// 最上位/最下位のセットビット
int highest = Integer.highestOneBit(n);   // 128 (0b10000000)
int lowest = Integer.lowestOneBit(n);     // 4   (0b00000100)

// 論理右シフト（>>>）と算術右シフト（>>）の違い
int neg = -128;                  // 0xFFFFFF80
int arithmetic = neg >> 4;       // 0xFFFFFFF8 = -8（符号保持）
int logical = neg >>> 4;         // 0x0FFFFFF8 = 268435448（0埋め）
```

```go
// Go: math/bits パッケージ

package main

import (
    "fmt"
    "math/bits"
)

func main() {
    var n uint32 = 0b1010_1100

    // ビットカウント
    fmt.Println(bits.OnesCount32(n))     // 4

    // 先頭/末尾のゼロ数
    fmt.Println(bits.LeadingZeros32(n))  // 24
    fmt.Println(bits.TrailingZeros32(n)) // 2

    // ビット長
    fmt.Println(bits.Len32(n))           // 8

    // ビット反転
    fmt.Println(bits.Reverse32(n))

    // ローテーション
    fmt.Println(bits.RotateLeft32(n, 4))
}
```

```rust
// Rust: 型に組み込みメソッド

fn main() {
    let n: u32 = 0b1010_1100;

    // ビットカウント
    println!("{}", n.count_ones());       // 4
    println!("{}", n.count_zeros());      // 28

    // 先頭/末尾のゼロ数
    println!("{}", n.leading_zeros());    // 24
    println!("{}", n.trailing_zeros());   // 2

    // ビット反転
    println!("{}", n.reverse_bits());

    // ローテーション
    println!("{}", n.rotate_left(4));
    println!("{}", n.rotate_right(4));

    // オーバーフロー検出
    let (result, overflowed) = n.overflowing_add(u32::MAX);
    println!("result={}, overflow={}", result, overflowed);

    // チェック付き算術
    match n.checked_add(u32::MAX) {
        Some(v) => println!("sum = {}", v),
        None => println!("overflow!"),
    }
}
```

---

## 4. データサイズの単位

### 4.1 ビットとバイトの階層

```
ビットとバイトの階層:

  1 bit    = 0 or 1
  1 nibble = 4 bits   = 16進数1桁
  1 byte   = 8 bits   = 256通りの値（0-255）
  1 word   = 32 or 64 bits（CPU依存）

  ストレージ単位:
  ┌────────────┬──────────────┬───────────────────────┐
  │ 単位       │ 10進（SI）   │ 2進（IEC）             │
  ├────────────┼──────────────┼───────────────────────┤
  │ キロ (K)   │ 1,000        │ 1,024 (2^10) = KiB    │
  │ メガ (M)   │ 1,000,000    │ 1,048,576 (2^20) = MiB│
  │ ギガ (G)   │ 10^9         │ 2^30 = GiB            │
  │ テラ (T)   │ 10^12        │ 2^40 = TiB            │
  │ ペタ (P)   │ 10^15        │ 2^50 = PiB            │
  │ エクサ (E) │ 10^18        │ 2^60 = EiB            │
  └────────────┴──────────────┴───────────────────────┘

  注意: HDD/SSDメーカーはSI単位を使い、OSはIEC単位を使う
  → 1TB SSD がOS上で 931GB と表示される
  → 1,000,000,000,000 / 1,073,741,824 ≈ 931 GiB
```

### 4.2 2のべき乗の暗記表

```
プログラマが暗記すべき2のべき乗:

  2⁰  = 1
  2¹  = 2
  2²  = 4
  2³  = 8
  2⁴  = 16
  2⁵  = 32
  2⁶  = 64
  2⁷  = 128
  2⁸  = 256        ← 1バイトの範囲
  2⁹  = 512
  2¹⁰ = 1,024      ← 1 KiB
  2¹¹ = 2,048
  2¹² = 4,096      ← 一般的なメモリページサイズ
  2¹³ = 8,192
  2¹⁴ = 16,384
  2¹⁵ = 32,768
  2¹⁶ = 65,536     ← unsigned short の範囲
  2²⁰ = 1,048,576  ← 1 MiB
  2²⁴ = 16,777,216 ← RGB各色8ビット（TrueColor）
  2³⁰ = 1,073,741,824 ← 約10億 ≈ 1 GiB
  2³² = 4,294,967,296 ← unsigned int の範囲（約43億）
  2⁴⁰ = 1,099,511,627,776 ← 1 TiB
  2⁶⁴ = 18,446,744,073,709,551,616 ← unsigned long long の範囲

  近似値の覚え方:
  2¹⁰ ≈ 10³  (1024 ≈ 1000)
  2²⁰ ≈ 10⁶  (約100万)
  2³⁰ ≈ 10⁹  (約10億)
  2⁴⁰ ≈ 10¹²
  2⁵⁰ ≈ 10¹⁵
  2⁶⁰ ≈ 10¹⁸
```

### 4.3 実務での容量見積もり

```
よくあるデータサイズの目安:

  テキスト:
  - ASCII 1文字 = 1 byte
  - UTF-8 日本語 1文字 = 3 bytes
  - 1ページのテキスト（40行 × 80文字）≈ 3.2 KB
  - 小説1冊（約10万字）≈ 300 KB（UTF-8）
  - Wikipedia全文（英語版）≈ 22 GB（2024年時点）

  画像:
  - 1920×1080 非圧縮（24bit色）= 1920 × 1080 × 3 ≈ 6.2 MB
  - 同サイズ JPEG（品質80）≈ 200-500 KB
  - 同サイズ PNG ≈ 1-3 MB
  - 4K画像（3840×2160）非圧縮 ≈ 24.9 MB

  音声:
  - CD品質（44.1kHz, 16bit, ステレオ）= 44100 × 2 × 2 = 176.4 KB/s ≈ 10.6 MB/min
  - MP3 128kbps ≈ 960 KB/min ≈ 1 MB/min
  - FLAC（可逆圧縮）≈ 5 MB/min

  動画:
  - 1080p 非圧縮（30fps）≈ 186 MB/s ≈ 11.2 GB/min
  - 1080p H.264（ストリーミング品質）≈ 5 Mbps ≈ 37.5 MB/min
  - 4K H.265 ≈ 15-25 Mbps

  データベース:
  - 1行のユーザーレコード（ID, 名前, メール等）≈ 200-500 bytes
  - 100万ユーザー ≈ 200-500 MB
  - インデックスはテーブルの10-30%程度のサイズ
```

---

## 5. 各プログラミング言語での数値リテラル

### 5.1 Python

```python
# Python: 接頭辞で基数を指定
decimal = 42        # 10進数
binary  = 0b101010  # 2進数
octal   = 0o52      # 8進数
hexadec = 0x2A      # 16進数
# 全て42

# アンダースコア区切り（Python 3.6+）
large = 1_000_000_000  # 10億
binary_large = 0b1010_1010_1100_1100
hex_large = 0xFF_FF_FF_FF

# 変換関数
bin(42)   # '0b101010'
oct(42)   # '0o52'
hex(42)   # '0x2a'
int('101010', 2)  # 42 (2進文字列→整数)
int('2A', 16)     # 42 (16進文字列→整数)
int('52', 8)      # 42 (8進文字列→整数)

# フォーマット
f"{42:b}"     # '101010'   (2進)
f"{42:o}"     # '52'       (8進)
f"{42:x}"     # '2a'       (16進小文字)
f"{42:X}"     # '2A'       (16進大文字)
f"{42:08b}"   # '00101010' (8桁ゼロ埋め2進)
f"{42:#010x}" # '0x0000002a' (接頭辞付き10桁ゼロ埋め16進)

# Pythonは任意精度整数
huge = 2 ** 1000  # 問題なく計算可能
print(huge.bit_length())  # 1001 (ビット数)
```

### 5.2 JavaScript / TypeScript

```javascript
// JavaScript: 同様の接頭辞
const decimal = 42;
const binary  = 0b101010;
const octal   = 0o52;
const hexadec = 0x2A;

// BigInt: 大きな整数
const big = 9007199254740993n;  // 'n' サフィックス
const bigHex = 0x1FFFFFFFFFFFFFn;

// Number の限界
console.log(Number.MAX_SAFE_INTEGER);  // 9007199254740991 (2^53 - 1)
console.log(Number.MIN_SAFE_INTEGER);  // -9007199254740991
// これを超えると精度が失われる

// ビット演算は32ビット整数に変換される（注意！）
console.log(0xFFFFFFFF | 0);    // -1 (符号付き32ビット)
console.log(0xFFFFFFFF >>> 0);  // 4294967295 (符号なし32ビット)

// 変換
(42).toString(2);    // '101010'
(42).toString(8);    // '52'
(42).toString(16);   // '2a'
parseInt('101010', 2);  // 42
parseInt('2A', 16);     // 42

// TypedArray でバイナリデータを扱う
const buffer = new ArrayBuffer(4);
const view = new DataView(buffer);
view.setUint32(0, 0xDEADBEEF);
console.log(view.getUint8(0).toString(16));  // 'de'

// Uint8Array での直接操作
const bytes = new Uint8Array([0xCA, 0xFE, 0xBA, 0xBE]);
```

### 5.3 Rust

```rust
// Rust: 型アノテーション + アンダースコア区切り
let decimal: u32 = 42;
let binary: u32  = 0b0010_1010;  // アンダースコアで視認性向上
let octal: u32   = 0o52;
let hexadec: u32 = 0x2A;
let byte: u8     = b'A';  // ASCII値 (65)

// 型サフィックス
let a = 42u8;    // u8 型
let b = 42i32;   // i32 型
let c = 42usize; // usize 型

// フォーマット
println!("{:b}", 42);     // "101010"
println!("{:08b}", 42);   // "00101010"
println!("{:o}", 42);     // "52"
println!("{:x}", 42);     // "2a"
println!("{:X}", 42);     // "2A"
println!("{:#010x}", 42); // "0x0000002a"

// バイト配列
let bytes: [u8; 4] = [0xDE, 0xAD, 0xBE, 0xEF];

// from_str_radix で文字列から変換
let n = u32::from_str_radix("101010", 2).unwrap();   // 42
let m = u32::from_str_radix("2A", 16).unwrap();      // 42

// ビット操作メソッド
let n: u32 = 42;
println!("{}", n.count_ones());      // 3
println!("{}", n.count_zeros());     // 29
println!("{}", n.leading_zeros());   // 26
println!("{}", n.trailing_zeros());  // 1
println!("{}", n.reverse_bits());    // ビット反転
println!("{}", n.rotate_left(4));    // 左ローテーション
```

### 5.4 Go

```go
package main

import (
    "fmt"
    "strconv"
)

func main() {
    // リテラル
    decimal := 42
    binary := 0b101010     // Go 1.13+
    octal := 0o52          // Go 1.13+ (旧形式: 052 も可)
    hexadec := 0x2A

    fmt.Println(decimal, binary, octal, hexadec) // 42 42 42 42

    // アンダースコア区切り
    large := 1_000_000_000

    // フォーマット
    fmt.Printf("%b\n", 42)    // 101010
    fmt.Printf("%08b\n", 42)  // 00101010
    fmt.Printf("%o\n", 42)    // 52
    fmt.Printf("%x\n", 42)    // 2a
    fmt.Printf("%X\n", 42)    // 2A
    fmt.Printf("%#x\n", 42)   // 0x2a

    // 文字列変換
    s := strconv.FormatInt(42, 2)   // "101010"
    n, _ := strconv.ParseInt("101010", 2, 64)  // 42

    _ = large
    _ = s
    _ = n
}
```

### 5.5 C / C++

```c
// C言語
#include <stdio.h>
#include <stdint.h>

int main() {
    int decimal = 42;
    int binary  = 0b00101010;  // C23 / GCC拡張
    int octal   = 052;         // 先頭0で8進数（注意！）
    int hexadec = 0x2A;

    // フォーマット
    printf("%d\n", 42);    // 42 (10進)
    printf("%o\n", 42);    // 52 (8進)
    printf("%x\n", 42);    // 2a (16進小文字)
    printf("%X\n", 42);    // 2A (16進大文字)
    printf("%#x\n", 42);   // 0x2a (接頭辞付き)

    // 固定幅整数型（推奨）
    uint8_t  byte_val = 0xFF;       // 必ず8ビット
    uint16_t short_val = 0xFFFF;    // 必ず16ビット
    uint32_t int_val = 0xFFFFFFFF;  // 必ず32ビット
    uint64_t long_val = 0xFFFFFFFFFFFFFFFFULL;  // 必ず64ビット

    // リテラルサフィックス
    unsigned int u = 42U;
    long l = 42L;
    unsigned long ul = 42UL;
    long long ll = 42LL;
    unsigned long long ull = 42ULL;

    // C23のバイナリリテラル
    // int b = 0b1010; // C23標準

    return 0;
}
```

```cpp
// C++: std::bitset
#include <bitset>
#include <iostream>

int main() {
    std::bitset<8> bits(42);          // "00101010"
    std::bitset<8> mask("11110000");  // 文字列から構築

    std::cout << bits << std::endl;          // 00101010
    std::cout << bits.count() << std::endl;  // 3 (1の数)
    std::cout << bits.size() << std::endl;   // 8
    std::cout << bits.test(1) << std::endl;  // 1 (ビット1がセットされているか)

    bits.set(0);    // ビット0をセット
    bits.reset(3);  // ビット3をクリア
    bits.flip(5);   // ビット5を反転
    bits.flip();    // 全ビット反転

    // C++14以降: 2進リテラル
    auto b = 0b0010'1010;  // ' で桁区切り

    // C++20: std::bit_cast, std::popcount 等
    // #include <bit>
    // int pc = std::popcount(42u);  // 3

    return 0;
}
```

---

## 6. 実務で頻出する数値パターン

### 6.1 メモリアドレスとアラインメント

```
メモリアドレスのアラインメント:

  多くのCPUは、データが特定の境界に整列していることを要求/推奨する。

  4バイトアラインメント:
    アドレスの下位2ビットが00
    → アドレスが4の倍数 (0x00, 0x04, 0x08, 0x0C, ...)

  8バイトアラインメント:
    アドレスの下位3ビットが000
    → アドレスが8の倍数 (0x00, 0x08, 0x10, 0x18, ...)

  アラインメント計算（ビット演算）:
    アドレスを N バイト境界に切り上げ:
    aligned = (addr + (N - 1)) & ~(N - 1)

    例: addr = 0x13, N = 8
    aligned = (0x13 + 0x07) & ~0x07
            = 0x1A & 0xFFFFFFF8
            = 0x18

  アラインメント確認:
    is_aligned = (addr & (N - 1)) == 0

    例: addr = 0x18, N = 8
    0x18 & 0x07 = 0x00 → 整列されている ✓

  なぜアラインメントが重要か:
  - ミスアラインドアクセスはCPUによっては例外を発生
  - 正しくアラインされたアクセスは1回のメモリ読み取りで完了
  - ミスアラインドだと2回のメモリ読み取り + マージが必要
  - SIMD命令（SSE, AVX）は16/32バイトアラインメントが必須
```

### 6.2 ハッシュとビット演算

```python
# ハッシュテーブルのインデックス計算

# 方法1: モジュロ演算（一般的）
index = hash(key) % table_size

# 方法2: ビットマスク（テーブルサイズが2のべき乗の場合、高速）
# table_size = 2^n のとき
index = hash(key) & (table_size - 1)

# 例: table_size = 1024 = 2^10
# hash(key) & 0x3FF  ← 下位10ビットを取り出す（0-1023）

# これが HashMap/HashSet のパフォーマンスの秘密
# → テーブルサイズを常に2のべき乗に保つ理由

# FNV-1aハッシュ（ビット演算で実装）
def fnv1a_32(data: bytes) -> int:
    FNV_OFFSET_BASIS = 0x811C9DC5
    FNV_PRIME = 0x01000193
    h = FNV_OFFSET_BASIS
    for byte in data:
        h ^= byte
        h = (h * FNV_PRIME) & 0xFFFFFFFF  # 32ビットに制限
    return h

# MurmurHash の最終ミキシング（ビットシフト + XOR）
def murmur_finalizer(h):
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h
```

### 6.3 色コードとビット演算

```python
# RGBA色の操作（各8ビット = 合計32ビット）

# 色の構成: 0xAARRGGBB (Alpha, Red, Green, Blue)
color = 0xFF8040C0  # A=255, R=128, G=64, B=192

# 各チャンネルの抽出
alpha = (color >> 24) & 0xFF  # 255
red   = (color >> 16) & 0xFF  # 128
green = (color >>  8) & 0xFF  # 64
blue  = (color >>  0) & 0xFF  # 192

# チャンネルから色を合成
def rgba(r, g, b, a=255):
    return (a << 24) | (r << 16) | (g << 8) | b

white = rgba(255, 255, 255)       # 0xFFFFFFFF
red_color = rgba(255, 0, 0)       # 0xFFFF0000
transparent = rgba(0, 0, 0, 0)    # 0x00000000

# アルファブレンディング
def blend(fg, bg, alpha):
    """alpha: 0-255"""
    return ((fg * alpha) + (bg * (255 - alpha))) // 255

# 色の明度調整
def brighten(color, factor):
    """factor: 0.0-2.0 (1.0 = 変化なし)"""
    a = (color >> 24) & 0xFF
    r = min(int(((color >> 16) & 0xFF) * factor), 255)
    g = min(int(((color >>  8) & 0xFF) * factor), 255)
    b = min(int(((color >>  0) & 0xFF) * factor), 255)
    return (a << 24) | (r << 16) | (g << 8) | b

# Webカラー
# #FF8040 → RGB(255, 128, 64)
hex_str = "FF8040"
r = int(hex_str[0:2], 16)  # 255
g = int(hex_str[2:4], 16)  # 128
b = int(hex_str[4:6], 16)  # 64
```

### 6.4 ネットワークプログラミングでのビット演算

```python
# IPv4アドレスの操作

import struct
import socket

# ドット表記 → 32ビット整数
def ip_to_int(ip_str):
    """'192.168.1.100' → 0xC0A80164"""
    parts = ip_str.split('.')
    return (int(parts[0]) << 24) | (int(parts[1]) << 16) | \
           (int(parts[2]) << 8) | int(parts[3])

# 32ビット整数 → ドット表記
def int_to_ip(ip_int):
    """0xC0A80164 → '192.168.1.100'"""
    return f"{(ip_int >> 24) & 0xFF}.{(ip_int >> 16) & 0xFF}." \
           f"{(ip_int >> 8) & 0xFF}.{ip_int & 0xFF}"

# サブネット計算
ip = ip_to_int('192.168.1.100')        # 0xC0A80164
mask = ip_to_int('255.255.255.0')      # 0xFFFFFF00

network = ip & mask                     # 192.168.1.0
broadcast = ip | (~mask & 0xFFFFFFFF)   # 192.168.1.255
host_part = ip & (~mask & 0xFFFFFFFF)   # 0.0.0.100

# CIDRプレフィックスからマスクを生成
def cidr_to_mask(prefix_len):
    """24 → 0xFFFFFF00"""
    return (0xFFFFFFFF << (32 - prefix_len)) & 0xFFFFFFFF

# マスクからCIDRプレフィックスを取得
def mask_to_cidr(mask):
    """0xFFFFFF00 → 24"""
    return bin(mask).count('1')

# ホスト数の計算
def host_count(prefix_len):
    """使用可能なホスト数"""
    return (2 ** (32 - prefix_len)) - 2  # ネットワークとブロードキャストを除く

print(host_count(24))   # 254
print(host_count(16))   # 65534
print(host_count(8))    # 16777214

# 同じサブネットに属するか判定
def same_subnet(ip1, ip2, mask):
    return (ip_to_int(ip1) & ip_to_int(mask)) == \
           (ip_to_int(ip2) & ip_to_int(mask))

print(same_subnet('192.168.1.100', '192.168.1.200', '255.255.255.0'))  # True
print(same_subnet('192.168.1.100', '192.168.2.100', '255.255.255.0'))  # False


# TCPフラグ（ビットフィールド）
TCP_FIN = 0x01  # 接続終了
TCP_SYN = 0x02  # 接続開始
TCP_RST = 0x04  # リセット
TCP_PSH = 0x08  # プッシュ
TCP_ACK = 0x10  # 確認応答
TCP_URG = 0x20  # 緊急

# SYN-ACK パケットのフラグ
flags = TCP_SYN | TCP_ACK  # 0x12

# フラグチェック
is_syn = bool(flags & TCP_SYN)      # True
is_fin = bool(flags & TCP_FIN)      # False
is_syn_ack = (flags & (TCP_SYN | TCP_ACK)) == (TCP_SYN | TCP_ACK)  # True
```

### 6.5 暗号学でのビット演算

```python
# XOR暗号（ストリーム暗号の基礎）

def xor_encrypt(plaintext: bytes, key: bytes) -> bytes:
    """XOR暗号化（復号も同じ操作）"""
    return bytes(p ^ k for p, k in zip(plaintext, key * (len(plaintext) // len(key) + 1)))

# XORの性質: A ^ K ^ K = A
message = b"Hello, World!"
key = b"SECRET"
encrypted = xor_encrypt(message, key)
decrypted = xor_encrypt(encrypted, key)
print(decrypted)  # b'Hello, World!'


# ビットローテーション（暗号アルゴリズムで多用）
def rotate_left_32(n, d):
    """32ビット左ローテーション"""
    return ((n << d) | (n >> (32 - d))) & 0xFFFFFFFF

def rotate_right_32(n, d):
    """32ビット右ローテーション"""
    return ((n >> d) | (n << (32 - d))) & 0xFFFFFFFF

# SHA-256のCh関数とMaj関数
def ch(x, y, z):
    """Choice: xが1ならyのビット、0ならzのビット"""
    return (x & y) ^ (~x & z)

def maj(x, y, z):
    """Majority: 多数決"""
    return (x & y) ^ (x & z) ^ (y & z)


# Feistel構造（DES等のブロック暗号の基本構造）
def feistel_round(left, right, round_key):
    """1ラウンドのFeistel構造"""
    new_left = right
    new_right = left ^ f(right, round_key)  # f は暗号化関数
    return new_left, new_right

# 逆変換（復号）は鍵の順序を逆にするだけ
def feistel_round_inv(left, right, round_key):
    new_right = left
    new_left = right ^ f(left, round_key)
    return new_left, new_right
```

---

## 7. エンディアン（バイト順序）

### 7.1 ビッグエンディアンとリトルエンディアン

```
32ビット整数 0x12345678 のメモリ上の配置:

  ビッグエンディアン（BE）: 上位バイトが先頭アドレス
  アドレス: 0x00  0x01  0x02  0x03
  値:       0x12  0x34  0x56  0x78
  → 人間にとって自然な順序（左から大きい桁）

  リトルエンディアン（LE）: 下位バイトが先頭アドレス
  アドレス: 0x00  0x01  0x02  0x03
  値:       0x78  0x56  0x34  0x12
  → Intel/AMD (x86, x64) が採用

  なぜリトルエンディアンが主流か:
  - 8ビットデータも16ビットデータも先頭アドレスが同じ
  - 加算の際、下位バイトから処理するのと合致
  - バイト幅の変換が容易

  各環境のエンディアン:
  ┌──────────────────────┬────────────┐
  │ 環境                 │ エンディアン │
  ├──────────────────────┼────────────┤
  │ x86, x64 (Intel/AMD) │ リトル      │
  │ ARM (通常モード)      │ リトル      │
  │ ネットワーク (TCP/IP)  │ ビッグ      │
  │ Java (JVM)            │ ビッグ      │
  │ RISC-V               │ リトル      │
  │ MIPS                 │ 切替可能    │
  │ PowerPC              │ ビッグ      │
  └──────────────────────┴────────────┘

  ネットワークバイトオーダー = ビッグエンディアン
  ホストバイトオーダー = CPUによる（通常リトル）
```

### 7.2 エンディアン変換の実装

```python
import struct

# Python の struct モジュール
# '<' = リトルエンディアン, '>' = ビッグエンディアン, '!' = ネットワーク(BE)

value = 0x12345678

# パック（整数 → バイト列）
be_bytes = struct.pack('>I', value)  # b'\x12\x34\x56\x78'
le_bytes = struct.pack('<I', value)  # b'\x78\x56\x34\x12'

# アンパック（バイト列 → 整数）
be_value = struct.unpack('>I', be_bytes)[0]  # 0x12345678
le_value = struct.unpack('<I', le_bytes)[0]  # 0x12345678

# int.from_bytes / int.to_bytes (Python 3.2+)
n = int.from_bytes(b'\x12\x34\x56\x78', byteorder='big')    # 0x12345678
n = int.from_bytes(b'\x78\x56\x34\x12', byteorder='little') # 0x12345678

data = (0x12345678).to_bytes(4, byteorder='big')    # b'\x12\x34\x56\x78'
data = (0x12345678).to_bytes(4, byteorder='little')  # b'\x78\x56\x34\x12'

# バイトスワップ（ビット演算で実装）
def bswap32(n):
    """32ビット値のバイトスワップ"""
    return ((n & 0xFF000000) >> 24) | \
           ((n & 0x00FF0000) >>  8) | \
           ((n & 0x0000FF00) <<  8) | \
           ((n & 0x000000FF) << 24)

print(hex(bswap32(0x12345678)))  # 0x78563412

def bswap16(n):
    """16ビット値のバイトスワップ"""
    return ((n & 0xFF00) >> 8) | ((n & 0x00FF) << 8)
```

```c
// C言語でのエンディアン変換

#include <arpa/inet.h>  // POSIX

// ネットワークバイトオーダー変換関数
uint32_t net_val = htonl(0x12345678);  // Host TO Network Long
uint16_t net_s   = htons(0x1234);      // Host TO Network Short
uint32_t host_val = ntohl(net_val);    // Network TO Host Long
uint16_t host_s   = ntohs(net_s);     // Network TO Host Short

// GCCビルトイン
uint32_t swapped = __builtin_bswap32(0x12345678);  // 0x78563412
uint64_t swapped64 = __builtin_bswap64(value64);
```

```rust
// Rustでのエンディアン変換

let n: u32 = 0x12345678;

// バイトスワップ
let swapped = n.swap_bytes();  // 0x78563412

// エンディアン変換
let be_bytes = n.to_be_bytes();  // [0x12, 0x34, 0x56, 0x78]
let le_bytes = n.to_le_bytes();  // [0x78, 0x56, 0x34, 0x12]

let from_be = u32::from_be_bytes([0x12, 0x34, 0x56, 0x78]);  // 0x12345678
let from_le = u32::from_le_bytes([0x78, 0x56, 0x34, 0x12]);  // 0x12345678
```

---

## 8. 実践演習

### 演習1: 基数変換（基礎）
以下の変換を手計算で行え:
1. 10進数 `173` → 2進数 → 16進数
2. 16進数 `0xDEAD` → 10進数
3. 2進数 `1011 0110` → 10進数 → 8進数
4. 10進数 `0.6875` → 2進数
5. 8進数 `0o1777` → 16進数 → 10進数

### 演習2: ビット演算パズル（応用）
ビット演算のみを使って以下を実装せよ（算術演算禁止）:
1. 2つの整数の加算
2. 整数のセットビット数（1の数）をカウント
3. 符号を反転（2の補数）
4. 2つの整数の大小比較（分岐なし）

### 演習3: IPアドレス計算（発展）
IPv4アドレス `192.168.10.50` とサブネットマスク `255.255.255.0 (/24)` から、ビット演算でネットワークアドレスとブロードキャストアドレスを求めよ。

### 演習4: ビットフィールドの設計（実務）
ファイルパーミッションシステムを設計せよ。以下の要件を満たすこと:
- ユーザー、グループ、その他の3種類のアクセス主体
- 各アクセス主体に対してread, write, execute の権限
- 1つの整数で全権限を表現
- 権限の確認・付与・剥奪の関数を実装

### 演習5: エンディアン変換（応用）
バイト列 `[0x48, 0x65, 0x6C, 0x6C]` を:
1. ビッグエンディアンの32ビット整数として解釈
2. リトルエンディアンの32ビット整数として解釈
3. ASCII文字列として解釈

### 演習解答例

```python
# 演習2-1: ビット演算のみの加算
def add(a, b):
    """ビット演算のみで加算を実装"""
    while b:
        carry = a & b      # 繰り上がりビット
        a = a ^ b           # 繰り上がりなしの加算
        b = carry << 1      # 繰り上がりを次の桁へ
    return a

# テスト
print(add(5, 3))    # 8
print(add(100, 200)) # 300

# 動作の詳細:
# a=5 (101), b=3 (011)
# 1回目: carry=001, a=110, b=010
# 2回目: carry=010, a=100, b=100
# 3回目: carry=100, a=000, b=1000
# 4回目: carry=000, a=1000, b=0000 → 終了
# 結果: 1000 = 8 ✓


# 演習2-2: ビットカウント
def count_bits(n):
    """Brian Kernighanのアルゴリズム"""
    count = 0
    while n:
        n &= n - 1  # 最下位の1ビットをクリア
        count += 1
    return count

print(count_bits(0b1011_0110))  # 5


# 演習2-3: 符号反転（2の補数）
def negate(n):
    """~n + 1 = -n"""
    return add(~n, 1)  # ビット反転 + 1


# 演習4: ファイルパーミッション
class FilePermission:
    # ビット位置の定義
    OTHER_EXECUTE = 0  # ビット0
    OTHER_WRITE   = 1  # ビット1
    OTHER_READ    = 2  # ビット2
    GROUP_EXECUTE = 3  # ビット3
    GROUP_WRITE   = 4  # ビット4
    GROUP_READ    = 5  # ビット5
    USER_EXECUTE  = 6  # ビット6
    USER_WRITE    = 7  # ビット7
    USER_READ     = 8  # ビット8

    def __init__(self, mode=0):
        self.mode = mode

    @classmethod
    def from_octal(cls, octal_str):
        """'755' → FilePermission"""
        return cls(int(octal_str, 8))

    def has_permission(self, bit):
        """指定ビットの権限があるか確認"""
        return bool(self.mode & (1 << bit))

    def grant(self, bit):
        """権限を付与"""
        self.mode |= (1 << bit)

    def revoke(self, bit):
        """権限を剥奪"""
        self.mode &= ~(1 << bit)

    def toggle(self, bit):
        """権限を反転"""
        self.mode ^= (1 << bit)

    def __repr__(self):
        chars = ''
        for label, r, w, x in [
            ('u', self.USER_READ, self.USER_WRITE, self.USER_EXECUTE),
            ('g', self.GROUP_READ, self.GROUP_WRITE, self.GROUP_EXECUTE),
            ('o', self.OTHER_READ, self.OTHER_WRITE, self.OTHER_EXECUTE),
        ]:
            chars += 'r' if self.has_permission(r) else '-'
            chars += 'w' if self.has_permission(w) else '-'
            chars += 'x' if self.has_permission(x) else '-'
        return chars

perm = FilePermission.from_octal('755')
print(perm)  # rwxr-xr-x
print(perm.has_permission(FilePermission.USER_WRITE))  # True
print(perm.has_permission(FilePermission.OTHER_WRITE))  # False
```

---

## 9. デバッグとトラブルシューティング

### 9.1 よくある間違い

```python
# 間違い1: Cの8進数リテラルに注意
# C言語で:
# int n = 010;  // これは8！（8進数の10）、10進数の10ではない

# 間違い2: JavaScriptのビット演算は32ビット
# JavaScript:
# 0xFFFFFFFF | 0  → -1 (符号付き32ビットとして解釈)
# 正しくは: 0xFFFFFFFF >>> 0  → 4294967295

# 間違い3: Pythonの~演算子
n = 5
print(~n)     # -6 (-(n+1))
# 8ビットの範囲で反転したい場合:
print(~n & 0xFF)  # 250 (0b11111010)

# 間違い4: 符号付きの右シフト
# -1 >> 1 = -1 (算術右シフト: 符号ビットが維持される)
# Pythonでは整数は無限精度なので:
# -1 = ...1111_1111_1111 (全ビットが1)
# >> 1 = ...1111_1111_1111 (やはり全ビットが1) = -1

# 間違い5: 浮動小数点数のビット演算
# Python: int型のみビット演算可能
# float には使えない
# 3.14 & 0xFF  → TypeError

# 間違い6: オーバーフロー
# Cで: uint8_t n = 255; n + 1 = 0（ラップアラウンド）
# Pythonでは整数がオーバーフローしない（任意精度）
```

### 9.2 デバッグツール

```python
# ビット表現の可視化ヘルパー

def show_bits(n, width=8):
    """整数のビット表現を見やすく表示"""
    if n < 0:
        # 2の補数表現を表示
        n = n & ((1 << width) - 1)
    bits = format(n, f'0{width}b')
    # 4ビットずつ区切る
    grouped = ' '.join(bits[i:i+4] for i in range(0, len(bits), 4))
    print(f"Dec: {n:>{width//3+3}d}  Hex: 0x{n:0{width//4}X}  Bin: {grouped}")

show_bits(42)       # Dec:  42  Hex: 0x2A  Bin: 0010 1010
show_bits(255)      # Dec: 255  Hex: 0xFF  Bin: 1111 1111
show_bits(0, 16)    # Dec:   0  Hex: 0x0000  Bin: 0000 0000 0000 0000

def show_operation(a, b, op_name, op_func, width=8):
    """ビット演算の過程を可視化"""
    result = op_func(a, b) & ((1 << width) - 1)
    a_bits = format(a & ((1 << width) - 1), f'0{width}b')
    b_bits = format(b & ((1 << width) - 1), f'0{width}b')
    r_bits = format(result, f'0{width}b')

    print(f"  {a_bits}  ({a})")
    print(f"{op_name} {b_bits}  ({b})")
    print(f"  {'─' * width}")
    print(f"  {r_bits}  ({result})")
    print()

show_operation(0b11001010, 0b10110110, '&', lambda a, b: a & b)
#   11001010  (202)
# & 10110110  (182)
#   ────────
#   10000010  (130)

show_operation(0b11001010, 0b10110110, '|', lambda a, b: a | b)
show_operation(0b11001010, 0b10110110, '^', lambda a, b: a ^ b)
```

### 9.3 バイナリデータの調査

```bash
# ファイルの16進ダンプ
xxd file.bin | head -20
# 00000000: 504b 0304 1400 0000 0800 ...  PK..........

# hexdump（別フォーマット）
hexdump -C file.bin | head -20

# 特定のバイトオフセットから読み取り
xxd -s 0x100 -l 32 file.bin

# バイナリ比較
xxd file1.bin > /tmp/hex1.txt
xxd file2.bin > /tmp/hex2.txt
diff /tmp/hex1.txt /tmp/hex2.txt

# Pythonでバイナリファイル解析
python3 -c "
with open('file.bin', 'rb') as f:
    data = f.read(16)
    print(' '.join(f'{b:02X}' for b in data))
    # マジックナンバーの確認
    magic = {
        b'\\x89PNG': 'PNG画像',
        b'\\xff\\xd8\\xff': 'JPEG画像',
        b'PK': 'ZIP/XLSX/DOCX',
        b'\\x7fELF': 'ELF実行ファイル',
        b'GIF8': 'GIF画像',
        b'%PDF': 'PDFファイル',
    }
    for sig, name in magic.items():
        if data[:len(sig)] == sig:
            print(f'ファイル形式: {name}')
"
```

---

## FAQ

### Q1: 16進数はなぜプログラミングで多用されるのですか？
**A**: 2進数の4ビットが16進数の1桁に正確に対応するため。`0xFF` は `1111 1111` と即座に分かるが、`255` からは直感的に分からない。メモリアドレス、色コード (#FF0000)、バイト列の表現に便利。

### Q2: 3進法コンピュータは存在しましたか？
**A**: ソ連のSetun（1958年）が三進法コンピュータとして有名。理論的には3進法は情報効率が最も高い（自然対数の底eに最も近い整数が3）。しかし、実用的にはトランジスタの2状態スイッチングの信頼性が圧倒的に高く、2進法が標準となった。

### Q3: ビット演算は実務でどのくらい使いますか？
**A**: 分野による。Web開発ではほぼ使わない。システムプログラミング、ネットワーク（IPマスク）、暗号、ゲーム開発、組み込みでは頻繁に使う。「使わない」場合でも、原理を理解していることで、パフォーマンス最適化やバグ解析に役立つ。

### Q4: エンディアンの違いで実際にバグが起きますか？
**A**: 頻繁に起きる。特にネットワークプログラミングでバイト列をそのまま整数として読み込む場合や、異なるアーキテクチャ間でバイナリファイルをやり取りする場合に問題になる。必ず`htonl()/ntohl()`等の変換関数を使い、プロトコルのエンディアンを明示すべき。

### Q5: ビット演算でのパフォーマンス向上は現代のCPUでも有効ですか？
**A**: 単純な置き換え（`n * 2` → `n << 1`）は現代のコンパイラが自動最適化するので手動で行う必要はない。しかし、ビット並列性を活用したアルゴリズム（popcount、ビットボード、SIMD等）は依然として大きな効果がある。可読性を犠牲にしてまでマイクロ最適化するのは避けるべきだが、アルゴリズムレベルでのビット活用は重要。

### Q6: 量子コンピュータは2進数ですか？
**A**: 量子コンピュータは量子ビット（qubit）を使う。qbitは0と1の重ね合わせ状態を取れる点が古典ビットと異なるが、測定結果は0か1になる。量子コンピュータの出力は最終的に古典ビットに変換されるため、入出力のインタフェースは2進数である。

---

## まとめ

| 概念 | ポイント |
|------|---------|
| 2進数 | トランジスタのON/OFFに直結。ノイズ耐性が高い |
| 16進数 | 2進数の4ビット=1桁。メモリ/バイト列の表記に標準 |
| 8進数 | 2進数の3ビット=1桁。Unixパーミッション（755等）で使用 |
| 基数変換 | 除算法と減算法の2つのアプローチ。小数は乗算法で変換 |
| ビット演算 | AND, OR, XOR, NOT, シフト。フラグ管理、最適化に使用 |
| シフト演算 | 左シフト=×2^n、右シフト=÷2^n。算術/論理の違いに注意 |
| エンディアン | ビッグ=上位バイト先頭、リトル=下位バイト先頭 |
| 単位 | SI(KB=1000)とIEC(KiB=1024)の2系統がある |
| 実務応用 | IP計算、色操作、暗号、ハッシュ、ゲームAI |

---

## 次に読むべきガイド
→ [[01-character-encoding.md]] — 文字コードとUnicode

---

## 参考文献
1. Petzold, C. "Code: The Hidden Language of Computer Hardware and Software." 2nd Edition, Microsoft Press, 2022.
2. Warren, H. S. "Hacker's Delight." 2nd Edition, Addison-Wesley, 2012.
3. Bryant, R. E. & O'Hallaron, D. R. "Computer Systems: A Programmer's Perspective." Chapter 2.
4. Shannon, C. E. "A Mathematical Theory of Communication." Bell System Technical Journal, 1948.
5. Knuth, D. E. "The Art of Computer Programming, Volume 4A: Combinatorial Algorithms, Part 1." Addison-Wesley, 2011.
6. IEEE 754-2019. "IEEE Standard for Floating-Point Arithmetic."
7. RFC 791. "Internet Protocol." IETF, 1981.
