# ラムダ計算と計算モデル

> ラムダ計算はチューリングマシンと等価な計算モデルであり、関数型プログラミングの数学的基盤である。3つの構文要素（変数・抽象・適用）だけで、あらゆる計算可能な関数を表現できる。

## この章で学ぶこと

- [ ] ラムダ計算が生まれた歴史的背景と、なぜ今も重要なのかを理解する
- [ ] ラムダ計算の核心概念（α変換、β簡約、η変換）を正確に使えるようになる
- [ ] Church エンコーディングで数・ブール・データ構造を関数だけで表現できる
- [ ] Y コンビネータによる再帰の原理を理解する
- [ ] 型付きラムダ計算の体系（単純型、System F、依存型）の位置づけを把握する
- [ ] Python でラムダ計算インタプリタを実装できる
- [ ] 現代プログラミングにおけるラムダ計算の影響を具体的に説明できる


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [暗号の基礎](./04-cryptography-basics.md) の内容を理解していること

---

## 1. なぜラムダ計算が必要か

### 1.1 歴史的背景：「計算とは何か」という問い

1900年、数学者ダフィット・ヒルベルトは「数学のあらゆる問題は機械的手続きで決定できるか」という問いを提起した。これは「Entscheidungsproblem（決定問題）」と呼ばれ、20世紀前半の数学基礎論を大きく動かした。

この問いに答えるためには、まず「機械的手続き」を厳密に定義する必要があった。1936年、2人の数学者が独立にこの定義を与えた。

```
┌─────────────────────────────────────────────────────────────┐
│  1936年 — 計算の形式化の年                                    │
│                                                              │
│  アラン・チューリング（ケンブリッジ）                           │
│  → チューリングマシン                                         │
│    「テープ上のシンボルを読み書きする機械」                     │
│    命令型プログラミングの数学的基盤                             │
│                                                              │
│  アロンゾ・チャーチ（プリンストン）                             │
│  → ラムダ計算                                                 │
│    「関数の定義と適用だけによる計算体系」                       │
│    関数型プログラミングの数学的基盤                             │
│                                                              │
│  両者は等価であることが証明された                               │
│  → Church-Turing thesis                                      │
└─────────────────────────────────────────────────────────────┘
```

なぜ「等価」が重要なのか。それは「計算可能な関数」の定義がモデルによらず一意に定まることを意味するからだ。チューリングマシンで計算できる関数はラムダ計算でも計算でき、その逆も成り立つ。どのような形式化を選んでも、計算できる範囲は変わらない。これが Church-Turing thesis（チャーチ＝チューリングの提唱）である。

### 1.2 なぜチューリングマシンだけでは不十分なのか

チューリングマシンは「計算とは何か」を定義する上で極めて成功した。しかし、プログラムの性質を数学的に推論する道具としては不便な面がある。

```
チューリングマシンの特徴:
  - 状態遷移と副作用（テープへの書き込み）が本質
  - 計算の途中状態を追跡する必要がある
  - プログラムの等価性の証明が困難

ラムダ計算の特徴:
  - 関数の定義と適用だけで計算を表現
  - 式の書き換え（簡約）による計算
  - 等式推論が容易 → プログラムの正しさの証明に適する
```

現代的に言えば、チューリングマシンは「コンピュータのハードウェアが何をしているか」を記述するモデルであり、ラムダ計算は「プログラムが何を計算するか」を記述するモデルである。両者は等価だが、用途が異なる。

### 1.3 ラムダ計算を学ぶ実務的理由

「理論的に面白い」だけでなく、ラムダ計算は現代のソフトウェア開発に直接的な影響を与えている。

1. **関数型プログラミング言語の設計原理**: Haskell, OCaml, F#, Elm などは型付きラムダ計算を直接の基盤としている
2. **クロージャとラムダ式**: Python の `lambda`、JavaScript のアロー関数、Rust のクロージャはすべてラムダ計算の概念
3. **型システム**: TypeScript, Rust, Kotlin のジェネリクスは System F（多相ラムダ計算）に由来する
4. **コンパイラ最適化**: インライン展開、定数畳み込みはβ簡約の応用
5. **プログラム検証**: Coq, Agda などの定理証明系は依存型付きラムダ計算に基づく

---

## 2. 核心概念

### 2.1 ラムダ計算の構文

ラムダ計算の構文は驚くほど単純である。たった3つの構文要素しかない。

```
ラムダ項（Lambda Term）の BNF:

  M, N ::= x           （変数: variable）
          | λx.M        （ラムダ抽象: abstraction — 関数の定義）
          | M N         （適用: application — 関数の呼び出し）

  これだけで、あらゆる計算可能な関数を表現できる。
```

なぜたった3要素で十分なのか。それは、数値・ブール・データ構造といったあらゆるデータが「関数」として表現できるからだ（Church エンコーディング、後述）。

**構文の読み方の規約**:

```
1. 適用は左結合:     M N P  =  (M N) P
   理由: 関数適用は最も頻繁に起こる操作であり、
         括弧を省略できると式が読みやすくなる

2. ラムダ抽象の本体は右端まで:  λx.M N  =  λx.(M N)
   理由: 関数の本体は通常複数の項からなるため、
         暗黙にできるだけ広い範囲を取る

3. 複数のラムダ抽象は省略可能:  λx.λy.λz.M  =  λxyz.M
   理由: カリー化された関数を簡潔に書くため
```

### 2.2 自由変数と束縛変数

ラムダ計算を正しく扱うために、変数の「スコープ」の概念を厳密に定義する必要がある。

```
λx.x y  において:
  x は束縛変数（bound variable）: λx によって束縛されている
  y は自由変数（free variable）:  どのラムダ抽象にも束縛されていない

  自由変数の集合 FV を再帰的に定義:
  FV(x)     = {x}
  FV(λx.M)  = FV(M) \ {x}     （x を除去）
  FV(M N)   = FV(M) ∪ FV(N)   （合併）

  例:
  FV(λx.x)       = {}         （閉じた項: combinator）
  FV(λx.x y)     = {y}        （y は自由）
  FV((λx.x)(λy.y z)) = {z}   （z だけが自由）
```

なぜ自由変数と束縛変数の区別が必要なのか。代入（β簡約）を行うときに、変数の「衝突（capture）」を避ける必要があるからだ。これはプログラミング言語における「変数のシャドーイング」と同じ問題である。

### 2.3 α変換（Alpha Conversion）

α変換とは、束縛変数の名前を別の名前に一貫して変更する操作である。

```
α変換の規則:
  λx.M  →α  λy.M[x:=y]    （ただし y が M の自由変数でないこと）

例:
  λx.x  →α  λy.y   （恒等関数は変数名を変えても同じ関数）
  λx.λy.x y  →α  λa.λb.a b

注意: 自由変数を捕捉してはならない
  λx.λy.x  →α  λy.λy.y  ← これは不正！
  （x を y に変えると、元の x（外側で束縛）が
    内側の λy に捕捉されてしまう）
```

なぜα変換が必要なのか。数学では「関数 f(x) = x+1 と g(y) = y+1 は同じ関数」と考える。ラムダ計算でも同様に、束縛変数の名前は意味を持たない。α変換はこの原則を形式化したものである。

実装上は、de Bruijn index（変数を名前ではなく「何番目のλに束縛されているか」という数値で表す方式）を使うことで、α変換の問題を完全に回避できる。

### 2.4 β簡約（Beta Reduction）

β簡約はラムダ計算における「計算」そのものである。関数適用を実行する操作にあたる。

```
β簡約の規則:
  (λx.M) N  →β  M[x:=N]
  「λx.M という関数に N を適用すると、M の中の x を N で置き換えた結果になる」

┌──────────────────────────────────────────────────────────┐
│  β簡約の過程（ステップバイステップ）                       │
│                                                           │
│  例1: (λx.x) 5                                           │
│        ↓ β簡約: M=x, x:=5                               │
│        5                                                  │
│                                                           │
│  例2: (λx.λy.x) a b                                     │
│        ↓ β簡約: (λx.λy.x) に a を適用                    │
│        (λy.a) b                                          │
│        ↓ β簡約: (λy.a) に b を適用                        │
│        a                                                  │
│                                                           │
│  例3: (λf.λx.f (f x)) (λy.y+1) 0                       │
│        ↓ β簡約: f := (λy.y+1)                            │
│        (λx.(λy.y+1) ((λy.y+1) x)) 0                    │
│        ↓ β簡約: x := 0                                   │
│        (λy.y+1) ((λy.y+1) 0)                            │
│        ↓ 内側のβ簡約: y := 0                              │
│        (λy.y+1) (0+1)                                    │
│        ↓ 算術                                             │
│        (λy.y+1) 1                                        │
│        ↓ β簡約: y := 1                                   │
│        1+1                                                │
│        ↓ 算術                                             │
│        2                                                  │
└──────────────────────────────────────────────────────────┘
```

**代入の厳密な定義**（なぜ単純な文字列置換ではダメなのか）:

```
代入 M[x:=N] の定義:

  x[x:=N]         = N
  y[x:=N]         = y              （y ≠ x のとき）
  (M₁ M₂)[x:=N]  = (M₁[x:=N]) (M₂[x:=N])
  (λx.M)[x:=N]   = λx.M           （x は束縛されているので置換しない）
  (λy.M)[x:=N]   = λy.(M[x:=N])   （y ∉ FV(N) のとき）
  (λy.M)[x:=N]   = λz.(M[y:=z][x:=N])  （y ∈ FV(N) のとき、
                                            z は新しい変数。
                                            まずα変換してから代入）

この最後のケースが「変数捕捉の回避」であり、
単純な文字列置換では正しく処理できない理由である。
```

**変数捕捉の具体例**:

```
(λx.λy.x) y を β簡約する。

  単純置換だと:  λy.y  ← 間違い！
    （自由だった y が λy に捕捉され、恒等関数になってしまう）

  正しくは:
    1. まず α変換:  λy.x  →α  λz.x
    2. 次に β簡約:  (λx.λz.x) y  →β  λz.y  ← 正しい
    （y は自由変数のまま保たれる）
```

### 2.5 η変換（Eta Conversion）

η変換は関数の「外延的等価性」を表す。

```
η変換の規則:
  λx.(M x)  →η  M    （ただし x ∉ FV(M)）

  意味: 「任意の引数 x に対して M x を返す関数」は M と等しい

例:
  λx.(add 1) x  →η  add 1
  λx.succ x     →η  succ

  Haskell での対応:
  \x -> f x  は  f  と同じ（ポイントフリースタイル）

  Python での対応:
  lambda x: f(x)  は  f  と同じ
  # map(lambda x: str(x), lst)  →  map(str, lst)
```

なぜη変換が重要なのか。η変換は「関数は入出力の振る舞いだけで同一性が決まる」という外延性の原理を表している。これはポイントフリースタイル（変数を明示しないプログラミングスタイル）の数学的根拠であり、コードの簡潔化やリファクタリングの正当性を保証する。

### 2.6 正規形と合流性

β簡約を繰り返し適用して、これ以上簡約できない形を「β正規形（β-normal form）」と呼ぶ。

```
正規形の分類:

  β正規形:        β簡約可能な部分式（redex）が存在しない
  頭部正規形:      先頭の redex が存在しない
  弱頭部正規形:    最外の redex が存在しない（Haskell の評価戦略）

正規形を持たない項の例:
  Ω = (λx.x x)(λx.x x)
    →β (λx.x x)(λx.x x)
    →β (λx.x x)(λx.x x)
    →β ...（無限ループ）

  これはプログラムの無限ループに対応する。
```

**Church-Rosser の定理（合流性）**:

```
もし M →β* N₁ かつ M →β* N₂ ならば、
ある P が存在して N₁ →β* P かつ N₂ →β* P

     M
    / \
   /   \
  N₁    N₂
   \   /
    \ /
     P

意味: β簡約の順序に関わらず、正規形が存在すれば一意に定まる。
→ 計算結果は簡約の順番に依存しない（正規形に到達する場合）。

ただし、すべての簡約順序で正規形に到達するとは限らない。
正規順序（最左最外 redex を先に簡約する戦略）を使えば、
正規形が存在すれば必ず到達できる。
```

---

## 3. Church エンコーディング

ラムダ計算には数値もブール値も組み込みで存在しない。しかし、これらのデータ構造を「関数」として表現できる。これが Church エンコーディングである。

### 3.1 Church 数（Church Numerals）

自然数 n を「関数 f を n 回適用する高階関数」として表現する。

```
┌─────────────────────────────────────────────────────────────┐
│  Church 数の定義                                             │
│                                                              │
│  0 = λf.λx.x              f を 0 回適用 → x をそのまま返す  │
│  1 = λf.λx.f x            f を 1 回適用                     │
│  2 = λf.λx.f (f x)        f を 2 回適用                     │
│  3 = λf.λx.f (f (f x))    f を 3 回適用                     │
│  n = λf.λx.fⁿ x           f を n 回適用                     │
│                                                              │
│  なぜこの表現が自然か:                                        │
│  「3」という概念の本質は「何かを3回行うこと」                  │
│  数値の本質を「反復の回数」として捉えている                    │
│                                                              │
│  視覚化:                                                     │
│  0: x                                                        │
│  1: f(x)                                                     │
│  2: f(f(x))                                                  │
│  3: f(f(f(x)))                                               │
│     ↑ f の適用回数 = 数値の値                                │
└─────────────────────────────────────────────────────────────┘
```

**Church 数上の算術演算**:

```
後者関数（+1）:
  SUCC = λn.λf.λx.f (n f x)
  「n 回 f を適用した結果に、もう1回 f を適用する」

加算:
  ADD = λm.λn.λf.λx.m f (n f x)
  「n 回 f を適用した結果に、さらに m 回 f を適用する」

  ADD 2 3 の簡約:
  = (λm.λn.λf.λx.m f (n f x)) 2 3
  →β (λn.λf.λx.2 f (n f x)) 3
  →β λf.λx.2 f (3 f x)
  = λf.λx.2 f (f (f (f x)))
  = λf.λx.f (f (f (f (f x))))
  = 5  ✓

乗算:
  MUL = λm.λn.λf.m (n f)
  「f を n 回適用する操作を、m 回繰り返す」

  MUL 2 3 の直感:
  = λf.2 (3 f)
  = λf.2 (λx.f(f(f x)))     ← 3f = 「fを3回適用する関数」
  = λf.λx.(λx.f(f(f x)))((λx.f(f(f x))) x)
  = λf.λx.f(f(f(f(f(f x)))))
  = 6  ✓

べき乗:
  EXP = λm.λn.n m
  「m を n 回適用する」（Church 数は高階関数なので自然に表現できる）
```

Python で Church 数を実装して動作を確認する。

```python
# コード例1: Church 数の算術（Python）

# Church 数の定義
ZERO  = lambda f: lambda x: x
ONE   = lambda f: lambda x: f(x)
TWO   = lambda f: lambda x: f(f(x))
THREE = lambda f: lambda x: f(f(f(x)))

# 算術演算
SUCC = lambda n: lambda f: lambda x: f(n(f)(x))
ADD  = lambda m: lambda n: lambda f: lambda x: m(f)(n(f)(x))
MUL  = lambda m: lambda n: lambda f: m(n(f))
EXP  = lambda m: lambda n: n(m)

# Church 数を Python の整数に変換する関数
# なぜこれで変換できるか: Church 数 n は「f を n 回適用する」ので、
# f=+1, x=0 を与えれば n 回 +1 が適用されて n が得られる
to_int = lambda n: n(lambda x: x + 1)(0)

# 動作確認
print(f"SUCC(2) = {to_int(SUCC(TWO))}")          # 3
print(f"ADD(2,3) = {to_int(ADD(TWO)(THREE))}")     # 5
print(f"MUL(2,3) = {to_int(MUL(TWO)(THREE))}")     # 6
print(f"EXP(2,3) = {to_int(EXP(TWO)(THREE))}")     # 8

# 任意の自然数を Church 数に変換
def to_church(n):
    """整数 n を Church 数に変換する。
    なぜ再帰で構築するか: SUCC を n 回適用することで
    「f を n 回適用する関数」を構築する。"""
    if n == 0:
        return ZERO
    return SUCC(to_church(n - 1))

# 検証
for i in range(10):
    assert to_int(to_church(i)) == i
print("Church 数の変換テスト: 全て成功")
```

### 3.2 Church ブール値

ブール値は「2つの引数のうちどちらを選ぶか」という選択関数として表現する。

```
TRUE  = λx.λy.x    （第1引数を選択）
FALSE = λx.λy.y    （第2引数を選択）

なぜこの表現が自然か:
  if-then-else を考えると、
  IF condition THEN a ELSE b
  = condition a b
  TRUE の場合:  (λx.λy.x) a b  →  a  （then 節を選択）
  FALSE の場合: (λx.λy.y) a b  →  b  （else 節を選択）
  → ブール値自体が条件分岐の機能を持つ

論理演算:
  AND = λp.λq.p q FALSE    （p が TRUE なら q、FALSE なら FALSE）
  OR  = λp.λq.p TRUE q     （p が TRUE なら TRUE、FALSE なら q）
  NOT = λp.p FALSE TRUE    （TRUE と FALSE を入れ替え）

ゼロ判定:
  ISZERO = λn.n (λx.FALSE) TRUE
  「n が 0 なら TRUE、1以上なら FALSE」
  0 の場合: (λx.FALSE) を 0 回適用 → TRUE のまま
  n>0 の場合: (λx.FALSE) を 1 回でも適用 → FALSE
```

```python
# コード例2: Church ブール値と論理演算（Python）

TRUE  = lambda x: lambda y: x
FALSE = lambda x: lambda y: y

AND = lambda p: lambda q: p(q)(FALSE)
OR  = lambda p: lambda q: p(TRUE)(q)
NOT = lambda p: p(FALSE)(TRUE)

# Church ブール値を Python の bool に変換
to_bool = lambda b: b(True)(False)

# ゼロ判定
ISZERO = lambda n: n(lambda x: FALSE)(TRUE)

# 動作確認
print(f"AND(TRUE, TRUE)   = {to_bool(AND(TRUE)(TRUE))}")    # True
print(f"AND(TRUE, FALSE)  = {to_bool(AND(TRUE)(FALSE))}")   # False
print(f"OR(FALSE, TRUE)   = {to_bool(OR(FALSE)(TRUE))}")    # True
print(f"NOT(TRUE)         = {to_bool(NOT(TRUE))}")          # False
print(f"NOT(FALSE)        = {to_bool(NOT(FALSE))}")         # True

# Church 数との組み合わせ
ZERO = lambda f: lambda x: x
ONE  = lambda f: lambda x: f(x)
TWO  = lambda f: lambda x: f(f(x))

print(f"ISZERO(0) = {to_bool(ISZERO(ZERO))}")   # True
print(f"ISZERO(1) = {to_bool(ISZERO(ONE))}")     # False
print(f"ISZERO(2) = {to_bool(ISZERO(TWO))}")     # False
```

### 3.3 Church 対（Pair）とリスト

2つの値を1つにまとめる「対」もラムダ式で表現できる。

```
対の定義:
  PAIR  = λx.λy.λf.f x y    （x と y を保持し、選択関数 f に渡す）
  FST   = λp.p TRUE          （第1要素を取得）
  SND   = λp.p FALSE         （第2要素を取得）

  なぜこの定義が機能するか:
  PAIR a b = λf.f a b
  FST (PAIR a b) = (λf.f a b) TRUE = TRUE a b = a  ✓
  SND (PAIR a b) = (λf.f a b) FALSE = FALSE a b = b  ✓

リストの定義（Church エンコーディング）:
  NIL   = λf.λx.x              （空リスト = 0 と同じ形）
  CONS  = λh.λt.λf.λx.f h (t f x)  （リストの先頭に要素を追加）

  リスト [1, 2, 3] は:
  CONS 1 (CONS 2 (CONS 3 NIL))
  = λf.λx.f 1 (f 2 (f 3 x))

  これは right fold そのもの:
  foldr f x [1, 2, 3] = f 1 (f 2 (f 3 x))
```

なぜ Church エンコーディングが重要なのか。それは「データ構造は不要であり、関数だけで十分」ということの数学的証明だからだ。もちろん実用上はデータ構造を使うほうが効率的だが、計算の理論的な能力としては関数だけで等価なことが示される。

### 3.4 前者関数（Predecessor）— Church 数の難所

Church 数の後者関数（SUCC）は自然に定義できたが、前者関数（PRED、-1 に相当）は意外に難しい。これは Church 自身も認めた問題で、最初の解決策は Kleene によるものだった。

```
PRED の定義（Kleene の手法）:
  PRED = λn.λf.λx.n (λg.λh.h (g f)) (λu.x) (λu.u)

  なぜ複雑になるか:
  Church 数は「f を n 回適用する」という構造であり、
  「1回分だけ取り除く」という操作は本質的に困難。
  Kleene のアイデアは「対を使って状態を保持しながら数え上げる」こと。

  直感的な理解（対を使ったバージョン）:
  PRED = λn.FST (n (λp.PAIR (SND p) (SUCC (SND p))) (PAIR ZERO ZERO))

  動作:
  n=0: FST (PAIR 0 0) = 0
  n=1: (PAIR 0 0) → (PAIR 0 1) → FST = 0
  n=2: (PAIR 0 0) → (PAIR 0 1) → (PAIR 1 2) → FST = 1
  n=3: (PAIR 0 0) → (PAIR 0 1) → (PAIR 1 2) → (PAIR 2 3) → FST = 2
  → 対の第1要素は常に「n-1」を保持する
```

---

## 4. 再帰と Y コンビネータ

### 4.1 再帰の問題

ラムダ計算では関数に名前を付けることができない。では、再帰（自分自身を呼び出す関数）はどう表現するのか。

```
通常のプログラミング言語での階乗:
  fact(n) = if n == 0 then 1 else n * fact(n-1)
  → fact という名前を使って自分自身を参照している

ラムダ計算には名前がない:
  λn.if n == 0 then 1 else n * ???(n-1)
  → 自分自身をどう参照するか？
```

この問題を解決するのが不動点コンビネータ（Fixed-Point Combinator）である。

### 4.2 不動点の概念

数学で f(x) = x を満たす x を関数 f の「不動点」と呼ぶ。ラムダ計算での不動点コンビネータ Y は、任意の関数 F に対して以下を満たす。

```
Y F = F (Y F)

意味: Y F は F の不動点である。
→ Y F を展開すると F (Y F) になり、
  さらに F (F (Y F)) になり、F (F (F (Y F))) になり...
→ これが再帰の仕組みである！
```

### 4.3 Y コンビネータ

```
┌────────────────────────────────────────────────────────────┐
│  Y コンビネータ（Curry の不動点コンビネータ）                │
│                                                             │
│  Y = λf.(λx.f (x x))(λx.f (x x))                         │
│                                                             │
│  検証: Y F = ?                                              │
│  Y F = (λf.(λx.f (x x))(λx.f (x x))) F                   │
│       →β (λx.F (x x))(λx.F (x x))                        │
│       →β F ((λx.F (x x))(λx.F (x x)))                    │
│       = F (Y F)  ✓                                         │
│                                                             │
│  Y F = F (Y F) = F (F (Y F)) = F (F (F (Y F))) = ...      │
│                                                             │
│  Y コンビネータの構造分析:                                   │
│  (λx.f (x x)) は「自己適用」のパターン                      │
│  x x で自分自身のコピーを作り、f に渡す                     │
│  → 自己参照のない言語で自己参照を実現する巧妙な手法          │
└────────────────────────────────────────────────────────────┘
```

### 4.4 Y コンビネータによる階乗の定義

```
階乗関数を名前なしで定義する:

  FACT = Y (λf.λn.ISZERO n ONE (MUL n (f (PRED n))))

  展開:
  FACT 3
  = Y G 3               （G = λf.λn.ISZERO n 1 (n * f(n-1))）
  = G (Y G) 3
  = (λf.λn...)(Y G) 3
  = ISZERO 3 1 (3 * (Y G)(3-1))
  = 3 * (Y G) 2
  = 3 * G (Y G) 2
  = 3 * ISZERO 2 1 (2 * (Y G)(2-1))
  = 3 * 2 * (Y G) 1
  = 3 * 2 * G (Y G) 1
  = 3 * 2 * ISZERO 1 1 (1 * (Y G)(1-1))
  = 3 * 2 * 1 * (Y G) 0
  = 3 * 2 * 1 * G (Y G) 0
  = 3 * 2 * 1 * ISZERO 0 1 (...)
  = 3 * 2 * 1 * 1
  = 6
```

正格評価（eager evaluation）の言語では Y コンビネータがそのままでは動かない。引数が適用前にすべて評価されるため、`Y F` の展開が止まらなくなる。そこで Z コンビネータを使う。

```python
# コード例3: Z コンビネータによる再帰（Python）

# Y コンビネータは正格評価言語では無限再帰になる
# Z コンビネータ（Y の正格評価版）を使う
# Z = λf.(λx.f (λv.x x v))(λx.f (λv.x x v))
# λv.x x v は η展開で、評価を遅延させるトリック

Z = lambda f: (lambda x: f(lambda v: x(x)(v)))(lambda x: f(lambda v: x(x)(v)))

# 階乗関数（Z コンビネータ使用）
# なぜ f を引数に取るか: 再帰呼び出しの代わりに f を呼ぶ
factorial = Z(lambda f: lambda n: 1 if n == 0 else n * f(n - 1))

# フィボナッチ数列
fibonacci = Z(lambda f: lambda n: n if n <= 1 else f(n - 1) + f(n - 2))

# 動作確認
for i in range(10):
    print(f"fact({i}) = {factorial(i)}")

print()
for i in range(10):
    print(f"fib({i}) = {fibonacci(i)}")

# 出力:
# fact(0) = 1, fact(1) = 1, fact(2) = 2, ..., fact(9) = 362880
# fib(0) = 0, fib(1) = 1, fib(2) = 1, ..., fib(9) = 34
```

Haskell での対応を見てみよう。Haskell は遅延評価なので Y コンビネータがそのまま動く。

```haskell
-- コード例4: Y コンビネータと Church エンコーディング（Haskell）

-- Haskell では遅延評価なので Y コンビネータが直接書ける
-- ただし型の問題があるため、fix として標準ライブラリに用意されている

import Data.Function (fix)

-- fix f = f (fix f)  -- 定義（遅延評価なので展開が止まる）

-- 階乗
factorial :: Integer -> Integer
factorial = fix (\f n -> if n == 0 then 1 else n * f (n - 1))

-- フィボナッチ
fibonacci :: Integer -> Integer
fibonacci = fix (\f n -> if n <= 1 then n else f (n - 1) + f (n - 2))

-- Church 数を Haskell で表現
-- type Church = forall a. (a -> a) -> a -> a  -- Rank2Types が必要

-- 簡略版（Integer 特化）
type ChurchInt = (Integer -> Integer) -> Integer -> Integer

zero :: ChurchInt
zero _f x = x

one :: ChurchInt
one f x = f x

two :: ChurchInt
two f x = f (f x)

succ' :: ChurchInt -> ChurchInt
succ' n f x = f (n f x)

add :: ChurchInt -> ChurchInt -> ChurchInt
add m n f x = m f (n f x)

mul :: ChurchInt -> ChurchInt -> ChurchInt
mul m n f = m (n f)

-- Church 数を Integer に変換
toInt :: ChurchInt -> Integer
toInt n = n (+1) 0

main :: IO ()
main = do
    putStrLn $ "fact(10) = " ++ show (factorial 10)     -- 3628800
    putStrLn $ "fib(10) = " ++ show (fibonacci 10)      -- 55

    putStrLn $ "succ(2) = " ++ show (toInt (succ' two)) -- 3
    putStrLn $ "add(2,3) = " ++ show (toInt (add two (succ' two))) -- 5
    putStrLn $ "mul(2,3) = " ++ show (toInt (mul two (succ' two))) -- 6
```

---

## 5. 評価戦略

### 5.1 評価戦略の分類

β簡約を行う際、式の中に複数の redex（簡約可能な部分式）がある場合、どれを先に簡約するかによって結果の到達性や効率が異なる。

```
┌──────────────────────────────────────────────────────────────┐
│  評価戦略の分類                                                │
│                                                               │
│  ■ 正規順序（Normal Order）                                   │
│    最左最外の redex を先に簡約                                 │
│    → 正規形が存在すれば必ず到達（完全性）                      │
│    → 引数を簡約せずに代入（実引数が複数回コピーされうる）      │
│                                                               │
│  ■ 適用順序（Applicative Order）                              │
│    最左最内の redex を先に簡約                                 │
│    → 引数を先に簡約してから代入（値渡し）                      │
│    → 正規形が存在しても到達できない場合がある                  │
│    → 多くの命令型言語の評価戦略                               │
│                                                               │
│  ■ 名前呼び（Call by Name）                                   │
│    正規順序の制限版: ラムダ抽象の中は簡約しない                │
│    → Algol 60 の引数評価方式                                  │
│                                                               │
│  ■ 必要呼び（Call by Need / Lazy Evaluation）                 │
│    名前呼びにメモ化を加えた方式                                │
│    → 引数は最初に必要になった時に1回だけ評価                   │
│    → Haskell の評価戦略                                       │
│                                                               │
│  ■ 値呼び（Call by Value）                                    │
│    適用順序の制限版: 引数が値になるまで簡約してから代入         │
│    → Python, JavaScript, OCaml などの評価戦略                 │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 評価戦略の違いが結果に影響する例

```
(λx.1) ((λx.x x)(λx.x x))

■ 正規順序（最左最外を先に簡約）:
  → 1
  理由: 外側の (λx.1) を先に適用。引数は使われないので簡約不要。

■ 適用順序（最左最内を先に簡約）:
  → (λx.1) ((λx.x x)(λx.x x))
  → (λx.1) ((λx.x x)(λx.x x))
  → ...（無限ループ）
  理由: 引数 (λx.x x)(λx.x x) を先に簡約しようとするが、
        これは Ω であり正規形を持たない。
```

この例は、評価戦略の選択がプログラムの停止性に影響しうることを示している。

### 比較表1: 評価戦略の比較

| 評価戦略 | 簡約対象 | 完全性 | 効率 | 採用言語 |
|---------|---------|--------|------|---------|
| 正規順序 | 最左最外 | 正規形があれば必ず到達 | 引数が複数回コピーされる | 理論上のモデル |
| 適用順序 | 最左最内 | 正規形があっても到達しない場合あり | 引数は1回だけ評価 | 理論上のモデル |
| 名前呼び | 関数適用の引数を未評価で渡す | 正規順序と同等 | 同じ引数が複数回再評価される | Algol 60 |
| 必要呼び（遅延評価） | 必要時に評価、結果をメモ化 | 正規順序と同等 | メモ化により再評価を回避 | Haskell |
| 値呼び（正格評価） | 引数を値まで評価してから渡す | 適用順序と同等 | 不要な引数も評価される | Python, JS, OCaml |

---

## 6. 型付きラムダ計算

### 6.1 なぜ型が必要なのか

型なしラムダ計算は非常に表現力が高い。しかし、表現力が高すぎるために問題も生じる。

```
型なしラムダ計算の問題:

1. 停止しない項が書ける: Ω = (λx.x x)(λx.x x)
2. 矛盾した論理が導ける: Curry のパラドックス
3. 「意味のない」適用が書ける: TRUE 42（ブール値に数を適用）

型を導入する理由:
- 「意味のある」プログラムだけを許可する
- 停止性を保証する（ただし表現力は制限される）
- コンパイル時にエラーを検出する
```

### 6.2 単純型付きラムダ計算（Simply Typed Lambda Calculus: STLC）

```
型の構文:
  τ ::= α           （基本型: Int, Bool, ...）
       | τ₁ → τ₂    （関数型）

型付け規則:

  x : τ ∈ Γ
  ─────────── (Var)       変数の型は環境 Γ から取得
   Γ ⊢ x : τ

   Γ, x : τ₁ ⊢ M : τ₂
  ─────────────────────── (Abs)   関数の型は引数型→返り値型
   Γ ⊢ λx:τ₁.M : τ₁ → τ₂

   Γ ⊢ M : τ₁ → τ₂    Γ ⊢ N : τ₁
  ─────────────────────────────────── (App)   適用は型が一致する必要
        Γ ⊢ M N : τ₂

STLC の重要な性質:
1. 型安全性（Type Safety）: 型が付く項は「実行時エラー」を起こさない
2. 強正規化（Strong Normalization）: 型が付く項は必ず停止する
3. 決定可能性: 型推論が決定可能

強正規化の代償:
  すべての型付き項が停止する → Y コンビネータに型が付かない
  → 一般再帰が使えない → チューリング完全ではない
  → 「安全だが、すべての計算は表現できない」というトレードオフ
```

### 6.3 System F（多相ラムダ計算）

```
System F（Girard, 1972 / Reynolds, 1974）:
  型変数に対する全称量化を導入

  型の構文:
    τ ::= α | τ₁ → τ₂ | ∀α.τ

  新しい構文:
    型抽象:  Λα.M      （型パラメータの導入）
    型適用:  M [τ]     （型の具体化）

  例: 多相恒等関数
    id = Λα.λx:α.x  :  ∀α.α → α

    id [Int] 42    →  42
    id [Bool] true →  true

  Haskell との対応:
    id :: forall a. a -> a
    id x = x

    -- Haskell では型引数は暗黙に推論される
    id 42      -- a = Int と推論
    id True    -- a = Bool と推論

  Java/TypeScript との対応:
    // Java のジェネリクスは System F の具現化
    <T> T identity(T x) { return x; }

    // TypeScript
    function identity<T>(x: T): T { return x; }
```

### 6.4 Lambda Cube と依存型

```
┌───────────────────────────────────────────────────────────┐
│  Lambda Cube（Barendregt のラムダ立方体）                   │
│                                                            │
│  3つの軸で型システムの拡張を分類:                            │
│                                                            │
│  1. 多相性（∀α.τ）:    項が型に依存                        │
│     → System F, Haskell のジェネリクス                     │
│                                                            │
│  2. 型演算子（λα.τ）:  型が型に依存                        │
│     → Haskell の型クラス、型族                             │
│                                                            │
│  3. 依存型（Πx:A.B）:  型が項に依存                        │
│     → Coq, Agda, Idris                                    │
│                                                            │
│         λω ───────── λC (Calculus of Constructions)        │
│        /|           /|   ← 3つの軸すべてを持つ              │
│       / |          / |      Coq の基盤                      │
│      /  |         /  |                                      │
│    λ2 ──────── λP2   |                                      │
│     |   λω_ ─────|── λPω_                                  │
│     |  /         |  /                                       │
│     | /          | /                                        │
│     |/           |/                                         │
│    λ→ ────────── λP                                        │
│    ↑              ↑                                         │
│  STLC       依存型(LF)                                     │
│                                                            │
│  λ→  = 単純型付きラムダ計算                                │
│  λ2  = System F（多相）                                    │
│  λω  = System Fω（型演算子）                               │
│  λP  = LF（依存型）                                        │
│  λC  = CoC（全部入り）                                     │
└───────────────────────────────────────────────────────────┘
```

**依存型の具体例**:

```
依存型: 型が値に依存する

例1: 長さ付きベクトル
  Vec : Nat → Type → Type
  Vec 0 a     = Nil
  Vec (n+1) a = Cons a (Vec n a)

  -- 長さ 3 の整数ベクトル
  v : Vec 3 Int
  v = Cons 1 (Cons 2 (Cons 3 Nil))

  -- head は空でないベクトルにしか適用できない
  head : Vec (n+1) a → a   -- 型レベルで安全性を保証

例2: 長さが一致する行列の掛け算
  matmul : Mat m n → Mat n p → Mat m p
  -- m×n 行列と n×p 行列のみ掛け算可能
  -- n が一致しないとコンパイルエラー

Idris での具体コード:
  data Vect : Nat -> Type -> Type where
    Nil  : Vect Z a
    (::) : a -> Vect n a -> Vect (S n) a

  head : Vect (S n) a -> a
  head (x :: _) = x
  -- head Nil はコンパイルエラー（型が合わない）
```

### 6.5 Curry-Howard 同型対応

型付きラムダ計算の最も深遠な結果の1つは、プログラムと証明の対応関係である。

```
┌─────────────────────────────────────────────────────────────┐
│  Curry-Howard 同型対応（Curry-Howard Isomorphism）            │
│                                                              │
│  論理                      型理論/プログラミング              │
│  ─────────────────────── ──────────────────────────          │
│  命題                      型                                │
│  証明                      プログラム（項）                   │
│  含意  A → B              関数型  A → B                     │
│  連言  A ∧ B              直積型  (A, B)                     │
│  選言  A ∨ B              直和型  Either A B                 │
│  真    ⊤                  ユニット型  ()                     │
│  偽    ⊥                  空型  Void                        │
│  全称  ∀x.P(x)           多相型  forall a. f a              │
│  存在  ∃x.P(x)           存在型  exists a. f a              │
│  証明の正規化              プログラムの評価                    │
│                                                              │
│  意味: 「型 A の値を構成できる」= 「命題 A を証明できる」      │
│  → プログラミングと数学の証明は同じこと                       │
│  → 正しい型のプログラムは、正しい定理の証明                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. 実装: Python でラムダ計算インタプリタ

理論を実装に落とし込むことで理解を深める。以下はラムダ計算の構文解析、α変換、β簡約を実装した完全なインタプリタである。

```python
# コード例5: ラムダ計算インタプリタ（Python）
# 純粋なラムダ計算の構文解析とβ簡約を実装する

from __future__ import annotations
from dataclasses import dataclass
from typing import Set

# ──────────────────────────────────────────────────
# 抽象構文木（AST）の定義
# ──────────────────────────────────────────────────

@dataclass(frozen=True)
class Var:
    """変数ノード。name は変数名を保持する。"""
    name: str

    def __str__(self) -> str:
        return self.name

@dataclass(frozen=True)
class Abs:
    """ラムダ抽象ノード。param は仮引数名、body は本体。
    λparam.body を表す。"""
    param: str
    body: 'Term'

    def __str__(self) -> str:
        return f"(λ{self.param}.{self.body})"

@dataclass(frozen=True)
class App:
    """適用ノード。func を arg に適用する。
    func arg を表す。"""
    func: 'Term'
    arg: 'Term'

    def __str__(self) -> str:
        return f"({self.func} {self.arg})"

Term = Var | Abs | App

# ──────────────────────────────────────────────────
# 自由変数の計算
# ──────────────────────────────────────────────────

def free_vars(term: Term) -> Set[str]:
    """項に含まれる自由変数の集合を返す。
    なぜ必要か: β簡約で変数捕捉を避けるため。"""
    match term:
        case Var(name):
            return {name}
        case Abs(param, body):
            return free_vars(body) - {param}
        case App(func, arg):
            return free_vars(func) | free_vars(arg)

# ──────────────────────────────────────────────────
# 新しい変数名の生成（α変換用）
# ──────────────────────────────────────────────────

_counter = 0

def fresh_var(base: str = "x") -> str:
    """衝突しない新しい変数名を生成する。
    なぜカウンタを使うか: 既存の変数名と重複しない
    ことを保証するため。"""
    global _counter
    _counter += 1
    return f"{base}_{_counter}"

# ──────────────────────────────────────────────────
# 代入（capture-avoiding substitution）
# ──────────────────────────────────────────────────

def substitute(term: Term, var: str, replacement: Term) -> Term:
    """term[var := replacement] を計算する。
    変数捕捉を自動的に回避する（α変換を必要に応じて行う）。"""
    match term:
        case Var(name):
            # 変数が置換対象なら置換、そうでなければそのまま
            return replacement if name == var else term

        case Abs(param, body):
            if param == var:
                # 束縛変数と置換対象が同名→この中では置換しない
                # なぜ: ラムダ抽象が x を再束縛しているため
                return term
            elif param not in free_vars(replacement):
                # 安全に代入可能
                return Abs(param, substitute(body, var, replacement))
            else:
                # 変数捕捉が起きる→α変換してから代入
                new_param = fresh_var(param)
                new_body = substitute(body, param, Var(new_param))
                return Abs(new_param, substitute(new_body, var, replacement))

        case App(func, arg):
            return App(
                substitute(func, var, replacement),
                substitute(arg, var, replacement)
            )

# ──────────────────────────────────────────────────
# β簡約（1ステップ）
# ──────────────────────────────────────────────────

def beta_reduce_step(term: Term) -> Term | None:
    """β簡約を1ステップ実行する。正規順序（最左最外優先）。
    簡約可能な redex がなければ None を返す。"""
    match term:
        case App(Abs(param, body), arg):
            # redex を発見: (λparam.body) arg → body[param:=arg]
            return substitute(body, param, arg)

        case App(func, arg):
            # 左部分を先に簡約（正規順序）
            reduced = beta_reduce_step(func)
            if reduced is not None:
                return App(reduced, arg)
            # 左が簡約済みなら右を簡約
            reduced = beta_reduce_step(arg)
            if reduced is not None:
                return App(func, reduced)
            return None

        case Abs(param, body):
            # ラムダ抽象の内部を簡約
            reduced = beta_reduce_step(body)
            if reduced is not None:
                return Abs(param, reduced)
            return None

        case Var(_):
            return None

# ──────────────────────────────────────────────────
# 完全な簡約（正規形まで）
# ──────────────────────────────────────────────────

def normalize(term: Term, max_steps: int = 100, verbose: bool = False) -> Term:
    """β正規形まで簡約する。無限ループ防止のため最大ステップ数を設定。"""
    current = term
    for step in range(max_steps):
        if verbose:
            print(f"  Step {step}: {current}")
        next_term = beta_reduce_step(current)
        if next_term is None:
            if verbose:
                print(f"  → 正規形に到達（{step} ステップ）")
            return current
        current = next_term
    print(f"  ⚠ {max_steps} ステップで打ち切り")
    return current

# ──────────────────────────────────────────────────
# 簡易パーサー
# ──────────────────────────────────────────────────

class Parser:
    """ラムダ式の文字列を AST に変換するパーサー。
    構文: 変数は英小文字、λ は \\ で表記、適用は空白。
    例: '(\\x.x) y' → App(Abs('x', Var('x')), Var('y'))"""

    def __init__(self, text: str):
        self.text = text.replace('λ', '\\')
        self.pos = 0

    def parse(self) -> Term:
        term = self._parse_expr()
        return term

    def _skip_spaces(self):
        while self.pos < len(self.text) and self.text[self.pos] == ' ':
            self.pos += 1

    def _parse_expr(self) -> Term:
        """適用を左結合で解析する。"""
        self._skip_spaces()
        terms = []
        while self.pos < len(self.text) and self.text[self.pos] not in ')':
            terms.append(self._parse_atom())
            self._skip_spaces()
        if not terms:
            raise ValueError(f"Unexpected end of input at position {self.pos}")
        result = terms[0]
        for t in terms[1:]:
            result = App(result, t)
        return result

    def _parse_atom(self) -> Term:
        self._skip_spaces()
        if self.pos >= len(self.text):
            raise ValueError("Unexpected end of input")

        ch = self.text[self.pos]

        if ch == '(':
            self.pos += 1  # skip '('
            expr = self._parse_expr()
            self._skip_spaces()
            if self.pos < len(self.text) and self.text[self.pos] == ')':
                self.pos += 1  # skip ')'
            return expr

        if ch == '\\':
            self.pos += 1  # skip '\'
            self._skip_spaces()
            param = self._parse_var_name()
            self._skip_spaces()
            if self.pos < len(self.text) and self.text[self.pos] == '.':
                self.pos += 1  # skip '.'
            self._skip_spaces()
            body = self._parse_expr()
            return Abs(param, body)

        return Var(self._parse_var_name())

    def _parse_var_name(self) -> str:
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos].isalnum():
            self.pos += 1
        if self.pos == start:
            raise ValueError(f"Expected variable name at position {self.pos}")
        return self.text[start:self.pos]

def parse(text: str) -> Term:
    return Parser(text).parse()

# ──────────────────────────────────────────────────
# 使用例
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== ラムダ計算インタプリタ ===\n")

    # 恒等関数の適用
    expr1 = parse(r"(\x.x) y")
    print(f"入力: {expr1}")
    result1 = normalize(expr1, verbose=True)
    print(f"結果: {result1}\n")

    # 定数関数
    expr2 = parse(r"(\x.\y.x) a b")
    print(f"入力: {expr2}")
    result2 = normalize(expr2, verbose=True)
    print(f"結果: {result2}\n")

    # Church 数 2 に succ を適用
    # SUCC 2 = SUCC (λf.λx.f(f x))
    # ここでは手動で AST を構築
    succ_term = Abs("n", Abs("f", Abs("x",
        App(Var("f"), App(App(Var("n"), Var("f")), Var("x"))))))
    two = Abs("f", Abs("x", App(Var("f"), App(Var("f"), Var("x")))))
    expr3 = App(succ_term, two)
    print(f"入力: SUCC 2 = {expr3}")
    result3 = normalize(expr3, verbose=True)
    print(f"結果: {result3}\n")

    # 変数捕捉の回避テスト
    # (λx.λy.x) y は λy.y ではなく λz.y になるべき
    expr4 = App(Abs("x", Abs("y", Var("x"))), Var("y"))
    print(f"入力: {expr4}")
    print("（変数捕捉の回避テスト: 結果は λz.y の形になるべき）")
    result4 = normalize(expr4, verbose=True)
    print(f"結果: {result4}\n")
```

---

## 8. 現代のプログラミングへの影響

### 8.1 クロージャ

クロージャは「自由変数を捕捉した関数」であり、ラムダ計算の直接的な応用である。

```
ラムダ計算:
  λx.λy.x + y  において、(λx.λy.x + y) 3 = λy.3 + y
  → 返された関数 λy.3 + y は自由変数 3 を「捕捉」している

Python:
  def make_adder(x):
      return lambda y: x + y   # x はクロージャに捕捉される
  add3 = make_adder(3)
  add3(5)  # → 8

JavaScript:
  const makeAdder = x => y => x + y;  // カリー化 + クロージャ
  const add3 = makeAdder(3);
  add3(5);  // → 8

Rust:
  fn make_adder(x: i32) -> impl Fn(i32) -> i32 {
      move |y| x + y   // move でクロージャに x を移動
  }
```

なぜクロージャが重要なのか。クロージャにより「状態を持つ関数」を作れるため、オブジェクト指向の代替としてデータのカプセル化が可能になる。実際、Peter Landin は1966年に「クロージャはオブジェクトの別の形態である」と指摘した。

### 8.2 カリー化と部分適用

```
ラムダ計算ではすべての関数が1引数:
  λx.λy.x + y  は「x を受け取って、y を受け取って x+y を返す関数を返す関数」

カリー化（Currying）:
  多引数関数を1引数関数の連鎖に変換すること
  f(x, y) → f(x)(y)

  名前の由来: Haskell Curry（ただし発見者は Moses Schonfinkel）

部分適用（Partial Application）:
  カリー化された関数に一部の引数だけを適用すること
  add(x)(y) で add(3) は「3を足す関数」

Haskell（すべての関数が自動的にカリー化される）:
  add :: Int -> Int -> Int   -- これは Int -> (Int -> Int) と同じ
  add x y = x + y
  add3 = add 3              -- 部分適用: add3 :: Int -> Int

Python:
  from functools import partial
  def add(x, y): return x + y
  add3 = partial(add, 3)    # 部分適用

  # または手動カリー化
  def add_curried(x):
      return lambda y: x + y
```

### 8.3 高階関数

```
ラムダ計算では関数は第一級（first-class）:
  関数を引数に取る関数、関数を返す関数が自然に書ける

map:   (a → b) → [a] → [b]     各要素に関数を適用
filter: (a → Bool) → [a] → [a]  条件を満たす要素を選択
fold:  (b → a → b) → b → [a] → b   畳み込み

Python:
  nums = [1, 2, 3, 4, 5]
  squares = list(map(lambda x: x**2, nums))          # [1, 4, 9, 16, 25]
  evens = list(filter(lambda x: x % 2 == 0, nums))   # [2, 4]
  total = reduce(lambda acc, x: acc + x, nums, 0)    # 15

Haskell:
  squares = map (^2) [1..5]         -- [1,4,9,16,25]
  evens   = filter even [1..5]      -- [2,4]
  total   = foldl (+) 0 [1..5]      -- 15
```

### 8.4 不変性と参照透過性

```
ラムダ計算の性質:
  - 変数は束縛されたら変更不可（不変性: immutability）
  - 同じ入力に対して常に同じ出力（参照透過性: referential transparency）
  - 副作用がない（純粋性: purity）

現代プログラミングへの影響:
  React:
    - 関数コンポーネントは純粋関数
    - state は不変（useState のセッターで新しい値を生成）
    - props は読み取り専用

  Redux:
    - reducer は純粋関数: (state, action) → newState
    - state の直接変更は禁止

  Rust:
    - デフォルトで不変（let x = 5; は不変）
    - 可変にするには明示的に let mut x = 5;

  Immutable.js / Immer:
    - JavaScript でイミュータブルデータ構造を提供
```

---

## 9. トレードオフと比較分析

### 比較表2: チューリングマシン vs ラムダ計算

| 観点 | チューリングマシン | ラムダ計算 |
|------|------------------|-----------|
| 提唱者・年 | Alan Turing, 1936 | Alonzo Church, 1936 |
| 計算の表現 | 状態遷移 + テープ操作 | 関数の定義と適用 |
| 基本要素 | 状態、テープ、遷移関数 | 変数、抽象、適用 |
| 計算の実行 | テープへの読み書き | β簡約（式の書き換え） |
| 直感的類似物 | ハードウェア（CPU） | 数学の関数 |
| 影響を与えたパラダイム | 命令型プログラミング | 関数型プログラミング |
| プログラムの推論 | 困難（状態空間が膨大） | 容易（等式推論が使える） |
| 副作用 | 本質的に副作用あり | 本質的に副作用なし |
| 計算量理論 | 自然に時間・空間複雑性を定義 | 複雑性の定義がやや不自然 |
| 実装の効率性 | 実際のコンピュータに近い | 直接的な実装は非効率 |
| 停止性の判定 | 決定不能 | 決定不能 |
| 計算能力 | チューリング完全 | チューリング完全（等価） |

### 9.1 計算モデルの選択基準

```
目的に応じたモデルの選択:

■ アルゴリズムの計算量を分析したい
  → チューリングマシン（時間・空間の定義が自然）

■ プログラムの正しさを証明したい
  → ラムダ計算（等式推論と型理論が使える）

■ 並行計算をモデル化したい
  → π計算、CSP、ペトリネット

■ 確率的計算をモデル化したい
  → 確率チューリングマシン、確率ラムダ計算

■ 量子計算をモデル化したい
  → 量子チューリングマシン、量子ラムダ計算
```

### 9.2 型システムのトレードオフ

```
型システムの強さと表現力のトレードオフ:

  表現力（弱→強）    安全性（弱→強）
  ─────────────────────────────────
  型なし（Python）    実行時エラーの可能性
  単純型付き          停止性保証（非チューリング完全）
  System F            多相性あり（型推論は決定不能）
  依存型              完全な仕様記述（使いこなすのが困難）

  実用的な妥協:
  Haskell  = System F + 型クラス + 再帰（チューリング完全を維持）
  Rust     = 線形型 + 所有権（メモリ安全性を型で保証）
  TypeScript = 構造的部分型（JavaScript との互換性を維持）
```

---

## 10. アンチパターン

### アンチパターン1: 不必要なラムダ抽象（η展開の乱用）

```
× アンチパターン:
  list(map(lambda x: str(x), items))
  list(map(lambda x: len(x), items))
  sorted(items, key=lambda x: x.lower())
  list(filter(lambda x: bool(x), items))

○ 正しい書き方（η簡約を適用）:
  list(map(str, items))
  list(map(len, items))
  sorted(items, key=str.lower)
  list(filter(bool, items))

なぜアンチパターンか:
  η変換により λx.f(x) = f である。
  不必要なラムダ抽象は:
  1. コードの可読性を下げる（意図が不明瞭になる）
  2. わずかだがオーバーヘッドがある（関数呼び出しが1段増える）
  3. f の意味が直接伝わらなくなる

例外: 副作用のある関数を遅延実行する場合は意図的にラムダで包む
  # これは意図的: 評価タイミングを制御している
  button.on_click(lambda event: process(event))
  # process が副作用を持つ場合、即座に呼ばれないようにする

Haskell でも同様:
  × map (\x -> f x) xs
  ○ map f xs

  × filter (\x -> even x) xs
  ○ filter even xs
```

### アンチパターン2: 過度なポイントフリースタイル

```
× アンチパターン（過度なポイントフリー）:
  Haskell:
    average = uncurry (/) . (sum &&& genericLength)
    -- 何をしているか理解困難

  Python:
    from functools import reduce
    from operator import add, mul
    result = reduce(mul, map(add, zip(xs, ys)))
    -- 関数合成が深すぎて可読性が低い

○ 適切なバランス:
  Haskell:
    average xs = sum xs / genericLength xs
    -- 明確で読みやすい

  Python:
    result = sum(x + y for x, y in zip(xs, ys))
    -- リスト内包表記のほうが Python らしい

なぜアンチパターンか:
  ポイントフリースタイルはη変換の応用であり、
  「変数を排除して関数合成だけで表現する」手法。
  適度に使えばコードが簡潔になるが、過度に使うと:
  1. 可読性が著しく低下する
  2. デバッグが困難になる（中間値を観察できない）
  3. チームメンバーが理解できない

原則:
  - 1段階の合成は OK: map f . filter g
  - 2段階以上で可読性が下がるなら、名前を付けるべき
  - 「コードゴルフ」と「良いコード」は異なる
```

---

## 11. エッジケース分析

### エッジケース1: 自己適用と型の限界

```
自己適用: λx.x x

  この項に型を付けることができるか？
  x : A とすると、x x で x は A → B 型である必要がある
  → A = A → B
  → A は自分自身を含む型になる（無限型）
  → 単純型付きラムダ計算では型が付かない

  結果:
  - Y コンビネータ Y = λf.(λx.f(x x))(λx.f(x x)) にも型が付かない
  - 一般再帰は型なしラムダ計算の能力であり、STLC では表現不能
  - Haskell では fix を言語組み込みで提供（型システムの「穴」）

  実務への影響:
  - TypeScript: 再帰型は許可されるが自己参照の深さに制限がある
  - Haskell: 型推論は決定可能だが、一部の拡張で決定不能になりうる
  - Rust: 再帰型は Box<T> で間接参照にする必要がある

  // TypeScript で自己参照型
  type Json = string | number | boolean | null | Json[] | { [key: string]: Json };
  // → 再帰型は許可される（有限展開が可能な場合）

  // Rust で再帰型
  // × enum List { Cons(i32, List) }  // コンパイルエラー: 無限サイズ
  // ○ enum List { Cons(i32, Box<List>), Nil }  // OK: Box でヒープに間接参照
```

### エッジケース2: 評価順序とエラー伝播

```
(λx.42) (1/0) の評価:

■ 正格評価（Python, JavaScript）:
  引数 1/0 を先に評価 → ZeroDivisionError
  関数本体に到達しない

■ 遅延評価（Haskell）:
  引数は使われないので評価されない → 42
  エラーは発生しない

  Haskell:
    (\x -> 42) (error "boom")  -- → 42（error は評価されない）
    (\x -> 42) undefined       -- → 42（undefined も評価されない）

実務への影響:
  1. 遅延評価では「実行されないコード」のバグが隠蔽される
     → テストで気づかない可能性がある

  2. 正格評価では不要な計算も実行される
     → 条件分岐の工夫が必要
     Python: x if condition else y   （短絡評価）
     Python: value = expensive() if needed else default

  3. メモリリーク:
     遅延評価では「まだ評価していない式（thunk）」が蓄積しうる
     Haskell: foldl (+) 0 [1..1000000]
     → thunk のチェーンがスタックオーバーフローを引き起こす
     → foldl' を使って正格に畳み込む

  4. デバッグの困難さ:
     遅延評価では実行順序が非直感的になる
     → Debug.Trace で評価タイミングを確認する必要がある
```

### エッジケース3: 名前の衝突と変数捕捉

```
現実のプログラミングでの変数捕捉の例:

JavaScript（var のスコープ問題）:
  for (var i = 0; i < 5; i++) {
    setTimeout(function() { console.log(i); }, 100);
  }
  // 期待: 0, 1, 2, 3, 4
  // 実際: 5, 5, 5, 5, 5
  // 理由: クロージャが i を「捕捉」しているが、var はブロックスコープを持たない

  // 解決法1: let を使う（ブロックスコープ）
  for (let i = 0; i < 5; i++) { ... }

  // 解決法2: IIFE（即時実行関数式）で新しいスコープを作る
  for (var i = 0; i < 5; i++) {
    (function(j) {
      setTimeout(function() { console.log(j); }, 100);
    })(i);
  }
  // → これはまさにβ簡約: (λj.setTimeout(λ().log(j), 100))(i)

Python のクロージャの落とし穴:
  funcs = [lambda: i for i in range(5)]
  print([f() for f in funcs])
  # 期待: [0, 1, 2, 3, 4]
  # 実際: [4, 4, 4, 4, 4]
  # 理由: lambda は i を遅延参照し、ループ終了時の値を見る

  # 解決法: デフォルト引数で即座に束縛
  funcs = [lambda i=i: i for i in range(5)]
  # → これはα変換 + β簡約の応用
```

---

## 12. 演習問題

### 演習1: 基礎 — β簡約の手動実行

**問題1-1**: 以下のラムダ式をβ簡約して正規形を求めよ。

```
(a) (λx.x x) (λy.y)
(b) (λx.λy.y x) a (λz.z)
(c) (λf.λx.f (f x)) (λy.y+1) 0
```

**解答**:

```
(a) (λx.x x) (λy.y)
    →β (λy.y) (λy.y)     [x := (λy.y)]
    →β λy.y               [(λy.y) を (λy.y) に適用]

(b) (λx.λy.y x) a (λz.z)
    →β (λy.y a) (λz.z)    [x := a]
    →β (λz.z) a            [y := (λz.z)]
    →β a                    [z := a]

(c) (λf.λx.f (f x)) (λy.y+1) 0
    →β (λx.(λy.y+1) ((λy.y+1) x)) 0    [f := (λy.y+1)]
    →β (λy.y+1) ((λy.y+1) 0)             [x := 0]
    →β (λy.y+1) (0+1)                     [y := 0]
    →β (λy.y+1) 1
    →β 1+1
    →β 2
```

**問題1-2**: 以下の項の自由変数を求めよ。

```
(a) λx.x y z
(b) (λx.x) (λy.y z)
(c) λx.λy.x (λz.z y) w
```

**解答**:

```
(a) FV(λx.x y z) = {y, z}   （x は束縛、y と z は自由）
(b) FV((λx.x)(λy.y z)) = {z}  （x は左で束縛、y は右で束縛、z は自由）
(c) FV(λx.λy.x (λz.z y) w) = {w}  （x,y,z は束縛、w は自由）
```

### 演習2: 応用 — Church エンコーディングの拡張

**問題2-1**: Church 数の減算 SUB を定義し、Python で実装せよ。SUB m n は m - n を返す（m >= n の場合）。

**ヒント**: PRED を n 回適用すればよい。

**解答**:

```python
# SUB = λm.λn.n PRED m
# 「PRED を n 回 m に適用する」

ZERO  = lambda f: lambda x: x
SUCC  = lambda n: lambda f: lambda x: f(n(f)(x))

# PRED（対を使ったバージョン）
PAIR  = lambda x: lambda y: lambda f: f(x)(y)
FST   = lambda p: p(lambda x: lambda y: x)
SND   = lambda p: p(lambda x: lambda y: y)
PRED  = lambda n: FST(n(lambda p: PAIR(SND(p))(SUCC(SND(p))))(PAIR(ZERO)(ZERO)))

# 減算
SUB = lambda m: lambda n: n(PRED)(m)

to_int = lambda n: n(lambda x: x + 1)(0)

# テスト
five = SUCC(SUCC(SUCC(SUCC(SUCC(ZERO)))))
three = SUCC(SUCC(SUCC(ZERO)))

print(f"5 - 3 = {to_int(SUB(five)(three))}")   # 2
print(f"5 - 0 = {to_int(SUB(five)(ZERO))}")     # 5
print(f"3 - 3 = {to_int(SUB(three)(three))}")   # 0
```

**問題2-2**: Church ブール値を使って、2つの Church 数の等値判定 EQ を定義せよ。

**ヒント**: ISZERO(SUB m n) AND ISZERO(SUB n m) を使う。

**解答**:

```python
TRUE  = lambda x: lambda y: x
FALSE = lambda x: lambda y: y
AND   = lambda p: lambda q: p(q)(FALSE)
ISZERO = lambda n: n(lambda x: FALSE)(TRUE)

EQ = lambda m: lambda n: AND(ISZERO(SUB(m)(n)))(ISZERO(SUB(n)(m)))

to_bool = lambda b: b(True)(False)

two = SUCC(SUCC(ZERO))
three_b = SUCC(two)

print(f"2 == 2: {to_bool(EQ(two)(two))}")          # True
print(f"2 == 3: {to_bool(EQ(two)(three_b))}")       # False
print(f"3 == 3: {to_bool(EQ(three_b)(three_b))}")   # True
```

### 演習3: 発展 — インタプリタの拡張

**問題3-1**: 本章のラムダ計算インタプリタに以下の機能を追加せよ。

1. η簡約のサポート: `λx.(M x)` で `x ∉ FV(M)` のとき `M` に簡約
2. 簡約戦略の切り替え: 正規順序と適用順序を選択可能にする
3. 簡約の各ステップで使われた規則（α, β, η）を表示する

**ヒント**:

```python
# η簡約の実装
def eta_reduce_step(term: Term) -> Term | None:
    match term:
        case Abs(param, App(func, Var(name))) if name == param and param not in free_vars(func):
            return func
        case Abs(param, body):
            reduced = eta_reduce_step(body)
            return Abs(param, reduced) if reduced else None
        case App(func, arg):
            reduced = eta_reduce_step(func)
            if reduced:
                return App(reduced, arg)
            reduced = eta_reduce_step(arg)
            return App(func, reduced) if reduced else None
        case _:
            return None

# 適用順序の実装
def applicative_order_step(term: Term) -> Term | None:
    match term:
        case App(Abs(param, body), arg):
            # まず引数を正規形にする
            reduced_arg = applicative_order_step(arg)
            if reduced_arg is not None:
                return App(Abs(param, body), reduced_arg)
            # 引数が正規形ならβ簡約
            return substitute(body, param, arg)
        case App(func, arg):
            reduced = applicative_order_step(func)
            if reduced is not None:
                return App(reduced, arg)
            reduced = applicative_order_step(arg)
            return App(func, reduced) if reduced else None
        case Abs(param, body):
            reduced = applicative_order_step(body)
            return Abs(param, reduced) if reduced else None
        case _:
            return None
```

**問題3-2**: de Bruijn index を用いたラムダ計算の表現を実装せよ。名前付き表現との相互変換を行い、α変換が不要になることを確認せよ。

**ヒント**:

```
de Bruijn index:
  変数を「何番目の外側のλに束縛されているか」の数値で表す

  λx.x        → λ.0          （直近のλ）
  λx.λy.x     → λ.λ.1        （1つ外のλ）
  λx.λy.y     → λ.λ.0        （直近のλ）
  λx.λy.x y   → λ.λ.1 0
  (λx.x)(λy.y) → (λ.0)(λ.0)  （α同値な項が同じ表現になる）

  利点: λx.x と λy.y が同じ表現 λ.0 になる
  → α変換が完全に不要になる
```

---

## 13. FAQ

### FAQ 1: ラムダ計算は実際のプログラミングで使うのか？

**短い答え**: 直接使うことは少ないが、間接的に常に使っている。

**詳細な説明**: Python の `lambda x: x + 1`、JavaScript の `x => x + 1`、Rust の `|x| x + 1` はすべてラムダ計算のラムダ抽象の実装である。`map`, `filter`, `reduce` などの高階関数、クロージャ、カリー化もラムダ計算から来ている。

さらに重要なのは、ラムダ計算がプログラミング言語の設計原理を理解するための共通言語であること。型推論アルゴリズム（Hindley-Milner）、コンパイラの中間表現（CPS変換、ANF変換）、プログラム変換（インライン展開、η簡約）はすべてラムダ計算の用語で記述される。プログラミング言語の論文を読むには、ラムダ計算の知識が必須である。

### FAQ 2: Y コンビネータはなぜ面接で聞かれるのか？

**短い答え**: 計算の本質的な理解を測る良い質問だから。

**詳細な説明**: Y コンビネータは以下の理解を同時に測る。

1. **再帰の本質**: 「名前」がなくても再帰が可能であることの理解
2. **高階関数**: 関数を引数に取り、関数を返す能力
3. **不動点**: 数学的概念の理解
4. **評価戦略**: 正格評価と遅延評価の違い（Y vs Z コンビネータ）
5. **自己適用**: `(λx.x x)` という非自明な構造の理解

ただし、Y コンビネータを暗記して書けることよりも、「なぜ再帰に名前が不要なのか」「不動点とは何か」を説明できることのほうが重要である。

### FAQ 3: Church エンコーディングは効率的か？

**短い答え**: 理論的には興味深いが、実用上は非常に非効率。

**詳細な説明**: Church 数 n は「f を n 回適用する」ので、加算は O(m+n)、乗算は O(m*n) の時間がかかる。通常のバイナリ表現なら加算は O(log n) である。

Church エンコーディングの価値は効率ではなく、以下の理論的な洞察にある。

1. **計算の最小性の証明**: 数値も条件分岐もデータ構造も、関数だけで表現できることの証明
2. **言語設計への示唆**: プリミティブ型を最小限にしても表現力は変わらないことの保証
3. **型理論との接続**: Church エンコーディングの型付けは System F の重要な応用例

実用言語では当然ながらネイティブの数値型を使うべきであり、Church エンコーディングは教育・理論目的で使うものである。

### FAQ 4: 関数型プログラミングは命令型プログラミングより優れているのか？

**短い答え**: 優劣ではなく、適材適所。

**詳細な説明**: ラムダ計算とチューリングマシンは等価であり、関数型と命令型も計算能力は同じである。違いは「どちらが特定の問題をより自然に表現できるか」にある。

```
関数型が向いている場面:
  - データ変換パイプライン（ETL、ストリーム処理）
  - 並行・並列プログラミング（不変性により競合状態が起きない）
  - コンパイラ・インタプリタ（AST の変換は再帰的処理に適する）
  - 数学的モデリング（参照透過性により等式推論が可能）

命令型が向いている場面:
  - ハードウェア制御（直接的な状態操作が必要）
  - GUI プログラミング（状態の変化が本質的）
  - パフォーマンスクリティカルなコード（メモリレイアウトの制御）
  - 逐次的なアルゴリズム（グラフの BFS/DFS など）

現代のベストプラクティス:
  マルチパラダイム言語（Scala, Kotlin, Rust, Swift）を使い、
  適切な場面で適切なスタイルを選択する。
```

### FAQ 5: 依存型はなぜ普及しないのか？

**短い答え**: 型の表現力が強すぎて、プログラマの負担が大きいため。

**詳細な説明**: 依存型では「長さ n のリストの head は n > 0 のときだけ定義される」のような仕様を型で表現できる。しかし、これには以下の代償がある。

1. **型注釈の負担**: プログラマが詳細な型の証明を書く必要がある
2. **型推論の限界**: 依存型の完全な型推論は決定不能
3. **ライブラリの不足**: 依存型言語のエコシステムはまだ小さい
4. **学習コスト**: 依存型を使いこなすには数学的な訓練が必要

ただし、部分的な依存型は徐々に普及している。TypeScript のリテラル型、Rust の const generics、Haskell の DataKinds 拡張などは依存型の限定的な形態である。

---

## 14. 組合せ論理との関係

ラムダ計算の「変数の束縛」を排除したものが組合せ論理（Combinatory Logic）である。

```
基本コンビネータ:
  S = λx.λy.λz.x z (y z)   （適用を分配する）
  K = λx.λy.x               （定数関数を作る）
  I = λx.x                   （恒等関数）

  S と K だけであらゆるラムダ式を表現できる（I = S K K）

  検証: S K K x
  = K x (K x)       [S の定義: x z (y z) に x=K, y=K, z=x を代入]
  = x                [K x _ = x]
  = I x  ✓

ラムダ式から SKI への変換（括弧抽出）:
  T[x]     = x
  T[E₁ E₂] = T[E₁] T[E₂]
  T[λx.x]  = I
  T[λx.c]  = K c           （x ∉ FV(c)）
  T[λx.E₁ E₂] = S (T[λx.E₁]) (T[λx.E₂])

例: λx.λy.x を SKI に変換
  T[λx.λy.x]
  = T[λx.K x]          （λy.x = K x）
  = S (T[λx.K]) (T[λx.x])
  = S (K K) I

  検証: S (K K) I x y
  = K K x (I x) y
  = K (I x) y
  = K x y
  = x  ✓
```

なぜ組合せ論理が重要なのか。変数の束縛という概念を完全に排除できるため、コンパイラの中間表現や関数型言語の実装で使われる。実際、初期の Haskell コンパイラは SKI コンビネータへの変換を使用していた（現在はより効率的な手法が使われている）。

---

## 15. CPS 変換と継続

### 15.1 継続渡しスタイル（CPS: Continuation-Passing Style）

```
CPS 変換:
  「次に何をするか」を明示的に関数として渡すスタイル

直接スタイル:
  fact n = if n == 0 then 1 else n * fact (n-1)

CPS:
  fact_cps n k = if n == 0 then k 1
                 else fact_cps (n-1) (\v -> k (n * v))
  -- k は「結果を受け取って次に行う計算」

  fact_cps 3 id
  = fact_cps 2 (\v -> id (3 * v))
  = fact_cps 1 (\v -> (\v' -> id (3 * v')) (2 * v))
  = fact_cps 0 (\v -> (\v' -> (\v'' -> id (3 * v'')) (2 * v')) (1 * v))
  = (\v -> (\v' -> (\v'' -> id (3 * v'')) (2 * v')) (1 * v)) 1
  = (\v' -> (\v'' -> id (3 * v'')) (2 * v')) (1 * 1)
  = (\v'' -> id (3 * v'')) (2 * 1)
  = id (3 * 2)
  = 6

CPS の利点:
  1. すべての関数呼び出しが末尾呼び出しになる → スタックオーバーフロー回避
  2. 計算の制御フローを明示的に操作できる
  3. コンパイラの中間表現として優れている
```

```python
# CPS 変換の例（Python）

# 直接スタイル
def factorial_direct(n):
    if n == 0:
        return 1
    return n * factorial_direct(n - 1)

# CPS スタイル
def factorial_cps(n, k):
    """k は継続（結果を受け取る関数）。
    なぜ CPS にするか: すべての呼び出しが末尾位置になるため、
    トランポリンを使えばスタックオーバーフローを回避できる。"""
    if n == 0:
        return k(1)
    return factorial_cps(n - 1, lambda v: k(n * v))

# 使用例
print(factorial_direct(10))           # 3628800
print(factorial_cps(10, lambda x: x)) # 3628800

# CPS + トランポリン（スタックオーバーフロー回避）
def trampoline(f):
    """サンクを繰り返し呼び出して末尾再帰を模倣する。"""
    result = f
    while callable(result):
        result = result()
    return result

def factorial_trampoline(n, k=lambda x: x):
    if n == 0:
        return k(1)
    return lambda: factorial_trampoline(n - 1, lambda v: lambda: k(n * v))

print(trampoline(factorial_trampoline(1000)))  # 大きな数でもスタックオーバーフローしない
```

---

## 16. ラムダ計算の拡張と変種

### 16.1 主要な拡張

```
■ 代数的データ型付きラムダ計算
  - パターンマッチングの追加
  - Haskell, OCaml, Rust の data/enum の基盤

■ 線形ラムダ計算
  - すべての変数がちょうど1回使われる
  - リソース管理の形式化
  - Rust の所有権システムの理論的基盤

■ アフィンラムダ計算
  - すべての変数が最大1回使われる（0回でもよい）
  - Rust の move セマンティクスに対応

■ 確率的ラムダ計算
  - 確率分布を基本演算に追加
  - 確率的プログラミング言語（Stan, Pyro, Gen）の基盤

■ 量子ラムダ計算
  - 量子ビットと量子ゲートを扱う
  - 線形性が本質的（量子の複製不可能定理に対応）
```

### 16.2 線形型と Rust の所有権

```
線形ラムダ計算のルール:
  すべての変数はちょうど1回使用されなければならない

  ○ λx.x          （x を1回使用）
  × λx.x x        （x を2回使用 → 違反）
  × λx.y          （x を0回使用 → 違反）

Rust との対応:
  fn consume(s: String) {
      println!("{}", s);  // s を使用
  }  // s はここで drop される

  let s = String::from("hello");
  consume(s);      // s の所有権が移動
  // println!("{}", s);  // コンパイルエラー: s は move 済み

  → 線形型のルール: 値は1回だけ使用される
  → メモリの二重解放やダングリングポインタを型レベルで防止
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

## 17. まとめ

| 概念 | ポイント | なぜ重要か |
|------|---------|-----------|
| ラムダ抽象 (λx.M) | 関数の定義。唯一のデータ構築手段 | すべてのデータと計算を関数で表現する基盤 |
| β簡約 | (λx.M) N → M[x:=N]。計算の実行 | プログラムの実行の数学的定義 |
| α変換 | 束縛変数の名前変更。意味を保存 | 変数のスコープを正しく扱うため |
| η変換 | λx.(M x) → M。外延的等価性 | ポイントフリースタイルとリファクタリングの正当性 |
| Church エンコーディング | データを関数で表現 | 計算の最小性の証明 |
| Y コンビネータ | 名前なしで再帰を実現 | 再帰の本質的な仕組みの理解 |
| Church-Turing thesis | あらゆる計算モデルは等価 | 計算可能性の普遍的な定義 |
| 単純型付きラムダ計算 | 型安全性と強正規化 | 型システムの理論的基盤 |
| System F | 多相型（ジェネリクス）の形式化 | Java/TypeScript/Haskell のジェネリクスの基盤 |
| 依存型 | 型が値に依存 | プログラムの完全な仕様記述 |
| Curry-Howard 同型 | 型＝命題、プログラム＝証明 | プログラミングと数学の深い対応 |
| 合流性 | 簡約順序に関わらず結果は一意 | 計算の決定性の保証 |

### ラムダ計算の学習ロードマップ

```
レベル1（基礎）:
  ラムダ式の読み書き → β簡約の手動実行 → Church エンコーディング
  → 「関数だけですべてが表現できる」ことの実感

レベル2（応用）:
  Y コンビネータの理解 → 評価戦略の違い → CPS 変換
  → 「計算の制御フロー」の理解

レベル3（理論）:
  型付きラムダ計算 → Curry-Howard 同型 → 依存型
  → 「型はプログラムの仕様であり、証明である」ことの理解

レベル4（実践）:
  コンパイラの中間表現 → 型推論アルゴリズム → プログラム変換
  → 「ラムダ計算は言語実装の共通語」であることの実感
```

---

## 次に読むべきガイド


---

## 参考文献

1. Church, A. "An Unsolvable Problem of Elementary Number Theory." *American Journal of Mathematics*, 58(2):345-363, 1936.
   -- ラムダ計算の原論文。Entscheidungsproblem の否定的解決。決定問題をラムダ定義可能性の概念を用いて形式化した歴史的論文。

2. Pierce, B. C. *Types and Programming Languages*. MIT Press, 2002.
   -- 型理論の標準的教科書。単純型付きラムダ計算から System F、部分型、再帰型まで体系的に解説。実装も伴っており、理論と実践の橋渡しとして最適。

3. Barendregt, H. P. *The Lambda Calculus: Its Syntax and Semantics*. Revised edition, North-Holland, 1984.
   -- ラムダ計算の包括的な参考書。構文論、意味論、型理論の全てを網羅。研究者向けだが、定義や定理の正確な参照先として不可欠。

4. Hindley, J. R. and Seldin, J. P. *Lambda-Calculus and Combinators: An Introduction*. Cambridge University Press, 2008.
   -- ラムダ計算と組合せ論理の入門書。前提知識が少なく、学部生レベルから読み始められる。Church-Rosser の定理の証明も丁寧に扱っている。

5. Girard, J.-Y., Lafont, Y., and Taylor, P. *Proofs and Types*. Cambridge University Press, 1989.
   -- Curry-Howard 同型対応と System F の解説書。論理学と型理論の対応関係を深く理解するための必読書。オンラインで無料公開されている。

6. Rojas, R. "A Tutorial Introduction to the Lambda Calculus." 2015.
   -- ラムダ計算の簡潔な入門チュートリアル。短くまとまっており、最初の一歩として最適。オンラインで無料で入手可能。
