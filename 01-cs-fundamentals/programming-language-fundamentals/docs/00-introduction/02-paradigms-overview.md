# プログラミングパラダイム概論

> **パラダイムとは「問題をどのように分解し、解決策をどのように構造化するか」の根本的な思想体系である。**
> パラダイムを理解することは、単に文法を覚えることとは次元が異なる。
> それは「プログラマとしての世界観」を形成する行為に他ならない。

---

## この章で学ぶこと

- [ ] 主要なプログラミングパラダイム（手続き型・OOP・関数型・論理型・リアクティブ・アクター）の本質的特徴を理解する
- [ ] 各パラダイムが誕生した歴史的背景と、解決しようとした問題を把握する
- [ ] パラダイム間の関係性と相互影響を体系的に整理する
- [ ] マルチパラダイム時代における適材適所の設計判断ができるようになる
- [ ] 各パラダイムのアンチパターンを認識し、回避できるようになる

---

## 目次

1. [パラダイムとは何か -- 概念の基礎](#1-パラダイムとは何か----概念の基礎)
2. [パラダイムの歴史的系譜](#2-パラダイムの歴史的系譜)
3. [手続き型プログラミング（Procedural）](#3-手続き型プログラミングprocedural)
4. [オブジェクト指向プログラミング（OOP）](#4-オブジェクト指向プログラミングoop)
5. [関数型プログラミング（Functional）](#5-関数型プログラミングfunctional)
6. [論理型プログラミング（Logic）](#6-論理型プログラミングlogic)
7. [リアクティブプログラミングとアクターモデル](#7-リアクティブプログラミングとアクターモデル)
8. [マルチパラダイム -- 現代の主流](#8-マルチパラダイム----現代の主流)
9. [パラダイム選択の指針と設計判断](#9-パラダイム選択の指針と設計判断)
10. [アンチパターン集](#10-アンチパターン集)
11. [演習問題（3段階）](#11-演習問題3段階)
12. [FAQ -- よくある質問](#12-faq----よくある質問)
13. [まとめ](#13-まとめ)
14. [参考文献](#14-参考文献)

---

## 1. パラダイムとは何か -- 概念の基礎

### 1.1 パラダイムの定義

「パラダイム（paradigm）」という語は、科学哲学者トーマス・クーンが 1962 年の著作
*The Structure of Scientific Revolutions* で提唱した概念に由来する。
プログラミングにおけるパラダイムとは、**ソフトウェアの構造と動作をどのような
抽象化の枠組みで捉え、記述するか**という根本的な思想体系を指す。

Robert Floyd は 1978 年のチューリング賞講演「The Paradigms of Programming」で
次のように述べた。

> "A paradigm is a way of conceptualizing what it means to perform computation,
> and how tasks to be carried out on a computer should be structured and organized."

パラダイムは「プログラムをどう書くか」ではなく、**「プログラムをどう考えるか」**
を規定する。同じ問題であっても、採用するパラダイムが異なれば、
分解の仕方・データの扱い・制御の流れ・抽象化の境界が根本的に変わる。

### 1.2 パラダイムの分類体系

パラダイムは大きく2つの系統に分類される。

```
プログラミングパラダイム分類
============================================================

          ┌── 命令型 (Imperative)
          │     │
          │     ├── 手続き型 (Procedural)
          │     │     例: C, Pascal, BASIC
          │     │
          │     ├── オブジェクト指向 (OOP)
          │     │     例: Java, C#, Smalltalk
          │     │
パラダイム ─┤     └── 並行指向 (Concurrent)
          │           例: Go (goroutines)
          │
          └── 宣言型 (Declarative)
                │
                ├── 関数型 (Functional)
                │     例: Haskell, Erlang, Clojure
                │
                ├── 論理型 (Logic)
                │     例: Prolog, Datalog
                │
                └── リアクティブ (Reactive)
                      例: RxJS, ReactiveX
```

**命令型（Imperative）** は「どうやって（How）計算するか」を記述する。
プログラムは一連の命令（文）であり、計算機の状態を逐次的に変更していく。

**宣言型（Declarative）** は「何を（What）求めるか」を記述する。
プログラムは関係・制約・変換の宣言であり、実行方法は処理系に委ねる。

### 1.3 パラダイムの3つの視点

パラダイムを理解するうえで重要な3つの視点がある。

```
+-----------------+--------------------+--------------------+
| 視点            | 命令型             | 宣言型             |
+=================+====================+====================+
| 状態の扱い      | 可変 (mutable)     | 不変 (immutable)   |
|                 | 変数への再代入     | 新しい値の生成     |
+-----------------+--------------------+--------------------+
| 制御の流れ      | 明示的             | 暗黙的             |
|                 | if/for/while       | パターンマッチ     |
|                 |                    | 再帰/高階関数      |
+-----------------+--------------------+--------------------+
| 抽象化の単位    | 手続き/オブジェクト | 関数/述語/型       |
|                 | モジュール         | モナド/型クラス    |
+-----------------+--------------------+--------------------+
```

### 1.4 なぜパラダイムを学ぶのか

パラダイムを学ぶ理由は3つある。

**第一に、思考の幅が広がる。** 手続き型しか知らない開発者は、あらゆる問題を
「状態の変更と制御フロー」で解こうとする。関数型を学べば「データの変換」
という視点が加わり、より適切な解法を選べるようになる。

**第二に、言語の設計意図が理解できる。** なぜ Rust には継承がないのか、
なぜ Haskell には変数への再代入がないのか、なぜ Go にはジェネリクスが後から
追加されたのか。これらはすべてパラダイムに基づく設計判断である。

**第三に、新しい言語の習得速度が劇的に上がる。** パラダイムの本質を
理解していれば、新しい言語は「既知のパラダイムの新しい実装」として
捉えることができる。文法の暗記ではなく、概念の対応付けで学習が進む。

---

## 2. パラダイムの歴史的系譜

### 2.1 年表で見るパラダイムの進化

```
年代     出来事                                      影響
=================================================================
1936     チューリングマシン / ラムダ計算              命令型 / 関数型の理論的基盤
1957     FORTRAN 登場                                手続き型の実用化
1958     LISP 登場                                   関数型の始祖
1962     Simula 登場                                 OOP の概念が誕生
1970     Prolog 登場                                 論理型の実用化
1972     Smalltalk 登場                              純粋OOPの確立
1973     C 言語登場                                  手続き型の決定版
1978     CSP 論文 (Hoare)                            並行計算の理論
1986     Erlang 開発開始                             アクターモデルの実用化
1987     Haskell プロジェクト開始                    純粋関数型の統一
1990     Haskell 1.0                                 遅延評価 + 型クラス
1995     Java 1.0                                    OOP の大衆化
1995     JavaScript 登場                             マルチパラダイムの先駆
2003     Scala 登場                                  OOP + FP の統合
2007     Clojure 登場                                JVM 上の関数型
2010     Rust 登場                                   所有権 + 関数型要素
2011     Kotlin 登場                                 OOP + FP の実用的統合
2012     Elixir 登場                                 Erlang VM 上の現代的FP
2014     Swift 登場                                  プロトコル指向
2024--   AI支援プログラミング                        パラダイムの自動選択
```

### 2.2 二大潮流の合流

プログラミングの歴史は、命令型（チューリングマシン由来）と宣言型
（ラムダ計算由来）という二大潮流の並行進化と合流として理解できる。

```
1930s          1960s          1980s          2000s          2020s
  |              |              |              |              |
  | チューリング  |   C          |  C++         | Java 8       |
  | マシン ------+---FORTRAN ---+---Pascal ----+---C# --------+-- 命令型に
  |              |              |              |   (ラムダ式)  |   関数型要素
  |              |              |              |              |   が標準装備
  |              |              |  Haskell     | Scala        |
  | ラムダ計算 --+---LISP -----+---ML --------+---Clojure ---+-- 関数型の
  |              |              |   (型推論)    |   Elixir     |   実用化が進行
  |              |              |              |              |
  |              +-- Simula     +-- Smalltalk  +-- Kotlin     +-- マルチ
  |              |   (OOP萌芽)  |   (純粋OOP)  |   Swift      |   パラダイム
  |              |              |              |   Rust       |   が主流に
  v              v              v              v              v
```

1990年代までは「手続き型 vs OOP vs 関数型」という対立構図が強かった。
しかし 2000年代以降、マルチコアプロセッサの普及と分散システムの要請により、
**不変性・純粋関数・合成可能性**といった関数型の概念が命令型言語に
取り込まれていった。Java 8 のラムダ式、C# の LINQ、Python の
ジェネレータとリスト内包表記などがその代表例である。

### 2.3 パラダイム融合の現在地

2020年代の現在、純粋に単一のパラダイムだけで設計された主要言語は
ほぼ存在しない。Haskell でさえ IO モナドを通じて手続き的な記述を許容し、
Java でさえ Stream API と Optional で関数型スタイルを推奨している。

この融合は「パラダイムの消失」ではなく、**「パラダイムの内面化」** である。
各パラダイムの概念は言語機能として吸収され、開発者は無意識のうちに
複数のパラダイムを切り替えながらコードを書くようになった。

だからこそ、各パラダイムの本質を明確に理解しておくことが
これまで以上に重要になっている。

---

## 3. 手続き型プログラミング（Procedural）

### 3.1 基本思想

```
============================================================
 手続き型プログラミングの核心
============================================================

 思想: 「処理を手順として上から下へ順番に記述する」

 計算モデル: チューリングマシン（状態遷移機械）

 特徴:
   - 命令の逐次実行（文の列）
   - 変数への代入（状態の変更）
   - 制御構造（if, for, while, switch）
   - 手続き（関数/サブルーチン）による分割
   - スコープによる名前空間の管理

 中心的な抽象: 手続き（procedure / function）
============================================================
```

手続き型プログラミングは、最も直感的で歴史の長いパラダイムである。
プログラムを「計算機に対する一連の指示」として記述する。
人間が料理のレシピや組立説明書を書くのと同じ発想であり、
「まずこれをして、次にこれをして、条件によってはこちらをする」
という思考がそのままコードに反映される。

### 3.2 中核概念の詳細

**逐次実行（Sequential Execution）**

手続き型では、文（statement）が上から下へ順番に実行される。
これは最も基本的な制御フローであり、他のすべてのパラダイムの基盤でもある。

**状態の変更（State Mutation）**

変数に値を代入し、その値を後から変更できる。これがチューリングマシンの
テープへの書き込みに対応する。手続き型の本質は「状態の逐次的な変更」にある。

**制御構造（Control Structures）**

条件分岐（if/else, switch）と反復（for, while, do-while）により、
実行の流れを制御する。Dijkstra の構造化定理により、任意のプログラムは
逐次・選択・反復の3つの構造で記述可能であることが証明されている。

**手続きの抽出（Procedure Extraction）**

繰り返し現れる処理パターンを手続き（関数/サブルーチン）として抽出し、
名前をつけて再利用する。これは最も基本的な抽象化手法である。

### 3.3 コード例: C による手続き型プログラミング

```c
/* C: 手続き型プログラミングの典型例 */
/* ファイルからデータを読み込み、統計値を計算する */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_SIZE 1000

/* --- データ読み込み手続き --- */
int read_data(const char *filename, double data[], int max_size) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr, "Error: Cannot open %s\n", filename);
        return -1;
    }

    int count = 0;
    while (count < max_size && fscanf(fp, "%lf", &data[count]) == 1) {
        count++;  /* 状態の変更: カウンタを増加 */
    }

    fclose(fp);
    return count;
}

/* --- 平均値を計算する手続き --- */
double calculate_mean(const double data[], int n) {
    double sum = 0.0;          /* 状態の初期化 */
    for (int i = 0; i < n; i++) {
        sum += data[i];        /* 状態の変更: 合計を蓄積 */
    }
    return sum / n;
}

/* --- 標準偏差を計算する手続き --- */
double calculate_stddev(const double data[], int n, double mean) {
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = data[i] - mean;
        sum_sq += diff * diff; /* 状態の変更: 二乗和を蓄積 */
    }
    return sqrt(sum_sq / n);
}

/* --- ソート手続き（バブルソート） --- */
void sort_ascending(double data[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (data[j] > data[j + 1]) {
                /* 状態の変更: 要素の交換 */
                double temp = data[j];
                data[j] = data[j + 1];
                data[j + 1] = temp;
            }
        }
    }
}

/* --- メイン手続き --- */
int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <datafile>\n", argv[0]);
        return 1;
    }

    double data[MAX_SIZE];
    /* 手順1: データを読み込む */
    int n = read_data(argv[1], data, MAX_SIZE);
    if (n <= 0) return 1;

    /* 手順2: 統計値を計算する */
    double mean = calculate_mean(data, n);
    double stddev = calculate_stddev(data, n, mean);

    /* 手順3: ソートして中央値を求める */
    sort_ascending(data, n);
    double median = (n % 2 == 0)
        ? (data[n/2 - 1] + data[n/2]) / 2.0
        : data[n/2];

    /* 手順4: 結果を出力する */
    printf("Count:  %d\n", n);
    printf("Mean:   %.4f\n", mean);
    printf("Median: %.4f\n", median);
    printf("StdDev: %.4f\n", stddev);
    printf("Min:    %.4f\n", data[0]);
    printf("Max:    %.4f\n", data[n-1]);

    return 0;
}
```

このコードは手続き型の特徴を全て備えている。
- 逐次実行: main 関数内の手順1→2→3→4
- 状態の変更: sum, sum_sq, data[] への書き込み
- 制御構造: for ループ、if 条件分岐
- 手続きの抽出: read_data, calculate_mean, sort_ascending

### 3.4 手続き型の強みと限界

```
+-----------------------------------------------+
|          手続き型の強み                         |
+-----------------------------------------------+
| [1] 直感的: 人間の思考に近い逐次的記述          |
| [2] 高効率: ハードウェアに近い低レベル制御      |
| [3] 予測可能: ステップ実行でデバッグ容易        |
| [4] 低オーバーヘッド: 抽象化層が薄い            |
| [5] 広い適用範囲: ほぼ全てのドメインに適用可能  |
+-----------------------------------------------+

+-----------------------------------------------+
|          手続き型の限界                         |
+-----------------------------------------------+
| [1] スケーラビリティ: 大規模になるとグローバル  |
|     状態が複雑化し、理解・保守が困難に          |
| [2] 再利用性: データと処理が分離しているため    |
|     コードの再利用が難しい                      |
| [3] 並行処理: 共有可変状態は競合条件の温床      |
| [4] テスト: 副作用のある関数のテストが煩雑      |
| [5] 抽象化: 高度な抽象を表現する手段が限定的    |
+-----------------------------------------------+
```

### 3.5 手続き型が輝く場面

- **システムプログラミング**: OS カーネル、デバイスドライバ、ブートローダ
- **組み込みシステム**: メモリ・CPU リソースが限られた環境
- **スクリプト**: シェルスクリプト、バッチ処理、自動化
- **数値計算**: 科学技術計算、シミュレーション
- **プロトタイピング**: 素早くアイデアを形にする段階

---

## 4. オブジェクト指向プログラミング（OOP）

### 4.1 基本思想

```
============================================================
 OOP の核心
============================================================

 思想: 「データと振る舞いをオブジェクトとして結合し、
        オブジェクト間の相互作用としてシステムを記述する」

 起源: Simula (1962), Smalltalk (1972)

 4つの柱:
   1. カプセル化 (Encapsulation)
      — 内部状態を隠蔽し、公開インターフェースのみを提供
   2. 継承 (Inheritance)
      — 既存クラスの機能を引き継いで拡張
   3. 多態性 (Polymorphism)
      — 同じインターフェースで異なる実装を呼び出す
   4. 抽象化 (Abstraction)
      — 本質的な特徴を抽出し、不要な詳細を隠蔽

 中心的な抽象: オブジェクト（状態 + 振る舞い + アイデンティティ）
============================================================
```

OOP は 1960 年代の Simula に端を発し、1970 年代の Smalltalk で
純粋な形に昇華され、1990 年代の Java / C++ によって産業界の
標準パラダイムとなった。その核心は「現実世界のエンティティを
ソフトウェア上のオブジェクトとしてモデリングする」という発想にある。

### 4.2 4つの柱の詳解

#### 4.2.1 カプセル化（Encapsulation）

カプセル化は「情報隠蔽（Information Hiding）」とも呼ばれ、
David Parnas が 1972 年の論文 "On the Criteria To Be Used in
Decomposing Systems into Modules" で提唱した概念に基づく。

オブジェクトの内部状態（フィールド）を外部から直接アクセスできないように
隠蔽し、公開メソッド（インターフェース）を通じてのみ操作を許可する。
これにより、内部実装の変更が外部に影響を与えることなく行える。

#### 4.2.2 継承（Inheritance）

既存のクラス（親クラス/スーパークラス）の属性とメソッドを、
新しいクラス（子クラス/サブクラス）が引き継ぐ仕組み。
コードの再利用を実現するが、過度な使用は「脆い基底クラス問題
（Fragile Base Class Problem）」を引き起こす。

現代のベストプラクティスでは、**継承よりもコンポジション（合成）** が
推奨される（"Favor composition over inheritance" -- GoF, 1994）。

#### 4.2.3 多態性（Polymorphism）

同じインターフェースやメソッド名で異なる振る舞いを実現する仕組み。
多態性には以下の種類がある。

- **サブタイプ多態性**: 継承/インターフェース実装による（最も一般的）
- **パラメトリック多態性**: ジェネリクス/テンプレートによる
- **アドホック多態性**: メソッドオーバーロードによる

#### 4.2.4 抽象化（Abstraction）

問題領域の本質的な特徴だけを抽出し、実装の詳細を隠蔽する。
抽象クラスやインターフェースを用いて「何ができるか」を定義し、
「どうやるか」は具体的な実装クラスに委ねる。

### 4.3 コード例: Python による OOP

```python
"""
Python: OOP の包括的な例
-- 図形計算システム --
"""
from abc import ABC, abstractmethod
from math import pi, sqrt
from typing import Protocol, runtime_checkable


# ---- 抽象化: 共通インターフェースの定義 ----

class Shape(ABC):
    """全ての図形の抽象基底クラス"""

    @abstractmethod
    def area(self) -> float:
        """面積を計算する"""
        pass

    @abstractmethod
    def perimeter(self) -> float:
        """周長を計算する"""
        pass

    def describe(self) -> str:
        """図形の情報を文字列で返す（テンプレートメソッドパターン）"""
        return (
            f"{self.__class__.__name__}: "
            f"area={self.area():.2f}, "
            f"perimeter={self.perimeter():.2f}"
        )


# ---- プロトコル: 構造的サブタイピング (Duck Typing の型安全版) ----

@runtime_checkable
class Drawable(Protocol):
    def draw(self, canvas: str) -> None: ...


# ---- カプセル化: 内部状態の隠蔽 ----

class Circle(Shape):
    """円: カプセル化により半径の不変条件を保証"""

    def __init__(self, radius: float):
        if radius <= 0:
            raise ValueError("radius must be positive")
        self._radius = radius  # プライベート属性

    @property
    def radius(self) -> float:
        """読み取り専用プロパティ"""
        return self._radius

    def area(self) -> float:
        return pi * self._radius ** 2

    def perimeter(self) -> float:
        return 2 * pi * self._radius

    def draw(self, canvas: str) -> None:
        print(f"Drawing circle (r={self._radius}) on {canvas}")


class Rectangle(Shape):
    """長方形"""

    def __init__(self, width: float, height: float):
        if width <= 0 or height <= 0:
            raise ValueError("dimensions must be positive")
        self._width = width
        self._height = height

    def area(self) -> float:
        return self._width * self._height

    def perimeter(self) -> float:
        return 2 * (self._width + self._height)

    def draw(self, canvas: str) -> None:
        print(f"Drawing rectangle ({self._width}x{self._height}) on {canvas}")


# ---- 継承: 既存クラスの拡張 ----

class Square(Rectangle):
    """正方形: 長方形の特殊化（is-a 関係）"""

    def __init__(self, side: float):
        super().__init__(side, side)  # 親クラスのコンストラクタを再利用


class Triangle(Shape):
    """三角形: 3辺の長さで定義"""

    def __init__(self, a: float, b: float, c: float):
        if not self._is_valid(a, b, c):
            raise ValueError("Invalid triangle sides")
        self._a, self._b, self._c = a, b, c

    @staticmethod
    def _is_valid(a: float, b: float, c: float) -> bool:
        """三角不等式の検証"""
        return a + b > c and b + c > a and a + c > b

    def area(self) -> float:
        """ヘロンの公式"""
        s = self.perimeter() / 2
        return sqrt(s * (s - self._a) * (s - self._b) * (s - self._c))

    def perimeter(self) -> float:
        return self._a + self._b + self._c


# ---- 多態性: 統一的なインターフェースで操作 ----

def print_all_shapes(shapes: list[Shape]) -> None:
    """どの図形でも同じメソッドで操作できる"""
    for shape in shapes:
        print(shape.describe())


def draw_all(drawables: list[Drawable], canvas: str) -> None:
    """Drawable プロトコルを満たすオブジェクトを描画"""
    for d in drawables:
        d.draw(canvas)


# ---- コンポジション: 継承より合成 ----

class ShapeCollection:
    """図形のコレクション（コンポジションの例）"""

    def __init__(self):
        self._shapes: list[Shape] = []

    def add(self, shape: Shape) -> None:
        self._shapes.append(shape)

    def total_area(self) -> float:
        return sum(s.area() for s in self._shapes)

    def largest(self) -> Shape:
        return max(self._shapes, key=lambda s: s.area())


# ---- 使用例 ----

if __name__ == "__main__":
    shapes = [
        Circle(5),
        Rectangle(3, 4),
        Square(6),
        Triangle(3, 4, 5),
    ]

    # 多態性: 全ての図形を統一的に処理
    print_all_shapes(shapes)
    # => Circle: area=78.54, perimeter=31.42
    # => Rectangle: area=12.00, perimeter=14.00
    # => Square: area=36.00, perimeter=24.00
    # => Triangle: area=6.00, perimeter=12.00

    # コンポジション
    collection = ShapeCollection()
    for s in shapes:
        collection.add(s)
    print(f"\nTotal area: {collection.total_area():.2f}")
    print(f"Largest: {collection.largest().describe()}")
```

### 4.4 OOP の設計原則 -- SOLID

OOP を効果的に活用するための設計原則として、Robert C. Martin が
提唱した SOLID 原則が広く知られている。

```
+---+-----------------------------------+----------------------------------+
| S | 単一責任の原則                      | クラスの変更理由は1つだけ         |
|   | Single Responsibility Principle    | にすべき                         |
+---+-----------------------------------+----------------------------------+
| O | 開放閉鎖の原則                      | 拡張に開いて、修正に閉じる       |
|   | Open-Closed Principle              | 既存コードを変えずに機能追加     |
+---+-----------------------------------+----------------------------------+
| L | リスコフの置換原則                   | 子クラスは親クラスと置換可能     |
|   | Liskov Substitution Principle      | であるべき                       |
+---+-----------------------------------+----------------------------------+
| I | インターフェース分離の原則           | 使わないメソッドへの依存を       |
|   | Interface Segregation Principle    | 強制すべきではない               |
+---+-----------------------------------+----------------------------------+
| D | 依存性逆転の原則                    | 具体ではなく抽象に依存すべき     |
|   | Dependency Inversion Principle     | 上位モジュールは下位に依存しない |
+---+-----------------------------------+----------------------------------+
```

### 4.5 クラスベース vs プロトタイプベース

OOP には2つの実装方式がある。

```
クラスベース OOP (Java, C#, Python, C++)
=============================================
  クラス(設計図) → インスタンス(実体)

  class Dog {
      String name;
      void bark() { ... }
  }
  Dog fido = new Dog("Fido");

  特徴:
  - クラスが型を定義し、インスタンスがそれに従う
  - 静的な型階層
  - コンパイル時の型チェック

プロトタイプベース OOP (JavaScript)
=============================================
  プロトタイプ(原型) → クローン(複製)

  const dog = {
      name: "Fido",
      bark() { console.log("Woof!"); }
  };
  const puppy = Object.create(dog);
  puppy.name = "Rex";

  特徴:
  - オブジェクトが他のオブジェクトを直接継承
  - 動的な委譲チェーン
  - 実行時の柔軟な変更
```

### 4.6 OOP の強みと限界

```
+-----------------------------------------------+
|          OOP の強み                             |
+-----------------------------------------------+
| [1] モデリング: 現実世界の概念をそのまま表現   |
| [2] スケーラビリティ: 大規模開発での分業に適   |
| [3] 再利用性: 継承とコンポジションで再利用     |
| [4] 保守性: カプセル化で変更の影響範囲を限定   |
| [5] エコシステム: 豊富なデザインパターン       |
+-----------------------------------------------+

+-----------------------------------------------+
|          OOP の限界                             |
+-----------------------------------------------+
| [1] 過度な設計: クラス爆発、深い継承階層        |
| [2] 並行処理: 共有可変状態がレースコンディション|
|     の温床になる                                |
| [3] 表現力: 「動詞（振る舞い）」を「名詞       |
|     （オブジェクト）」に無理に押し込む問題      |
| [4] ボイラープレート: getter/setter/constructor |
|     等の定型コードが増加                        |
| [5] テスト: モックの多用が必要になりがち        |
+-----------------------------------------------+
```

### 4.7 OOP が輝く場面

- **GUIアプリケーション**: ウィジェット階層、イベントハンドリング
- **ゲーム開発**: エンティティ・コンポーネント、シーングラフ
- **エンタープライズシステム**: ドメインモデリング、業務ロジック
- **フレームワーク設計**: プラグイン機構、テンプレートメソッド
- **API設計**: リソースのモデリング、バージョニング

---

## 5. 関数型プログラミング（Functional）

### 5.1 基本思想

```
============================================================
 関数型プログラミングの核心
============================================================

 思想: 「計算を数学的関数の適用と合成として記述する」

 起源: ラムダ計算 (Church, 1936), LISP (McCarthy, 1958)

 原則:
   1. 純粋関数 (Pure Functions)
      — 同じ入力に対して常に同じ出力、副作用なし
   2. 不変性 (Immutability)
      — データを変更しない、新しいデータを生成する
   3. 第一級関数 (First-class Functions)
      — 関数を値として変数に代入、引数に渡す、戻り値にする
   4. 参照透過性 (Referential Transparency)
      — 式をその評価結果で置き換えても意味が変わらない
   5. 高階関数 (Higher-order Functions)
      — 関数を引数に取る、または関数を返す関数

 中心的な抽象: 関数（入力から出力への写像）
============================================================
```

関数型プログラミング（FP）は、数学の関数概念をプログラミングに
直接持ち込んだパラダイムである。命令型が「状態をどう変えるか」に
焦点を当てるのに対し、FP は「データをどう変換するか」に焦点を当てる。

この発想の転換は単なるスタイルの違いではない。
**不変性を前提とすることで、プログラムの推論が劇的に容易になり、
並行処理が安全に行え、テストが書きやすくなる** という実用的な利点をもたらす。

### 5.2 中核概念の詳解

#### 5.2.1 純粋関数と副作用

純粋関数（Pure Function）は以下の2つの性質を満たす関数である。

1. **決定性**: 同じ引数に対して常に同じ値を返す
2. **副作用なし**: 外部の状態を変更しない（ログ出力、DB書き込み、グローバル変数の変更などを行わない）

```
純粋関数 vs 非純粋関数
============================================================

純粋関数:
  f(x) = x * 2 + 1
  - f(3) は常に 7 を返す
  - 呼び出し回数に関係なく結果は同じ
  - テスト: assert f(3) == 7 で十分

非純粋関数:
  counter = 0
  def increment():
      global counter
      counter += 1       # 副作用: グローバル状態を変更
      return counter
  - increment() は呼び出すたびに異なる値を返す
  - テスト: 状態のリセットが必要
============================================================
```

#### 5.2.2 不変性とデータ構造

関数型では、一度作られたデータは変更されない。
「変更」が必要な場合は、変更箇所だけが異なる新しいデータを生成する。

これは非効率に見えるが、**永続データ構造（Persistent Data Structure）** を
用いることで、構造の大部分を共有し、効率的に実現される。

```
不変リストの更新（構造共有）
============================================================

元のリスト: [A] -> [B] -> [C] -> [D]

先頭に E を追加:

  新リスト: [E] -> [A] -> [B] -> [C] -> [D]
                    ^^^^^^^^^^^^^^^^^^^^^^^^^
                    元のリストをそのまま共有

元のリストは変更されていない（参照している他のコードに影響なし）
============================================================
```

#### 5.2.3 関数合成とパイプライン

小さな関数を組み合わせて大きな関数を構築する「関数合成」は、
FP の最も強力な抽象化手法である。

### 5.3 コード例: Haskell と JavaScript

```haskell
-- Haskell: 純粋関数型プログラミング
-- ==========================================

-- 純粋関数: 階乗（再帰 + パターンマッチ）
factorial :: Integer -> Integer
factorial 0 = 1
factorial n = n * factorial (n - 1)

-- 高階関数: 関数を引数に取る
applyTwice :: (a -> a) -> a -> a
applyTwice f x = f (f x)
-- applyTwice (+3) 10  =>  16
-- applyTwice (*2) 3   =>  12

-- 関数合成: (.) 演算子
-- process = sum . map (*2) . filter (>0)
-- 右から左へ: filter → map → sum
process :: [Int] -> Int
process = sum . map (*2) . filter (>0)
-- process [1, -2, 3, -4, 5] => 18

-- リスト内包表記
pythagoreanTriples :: Int -> [(Int, Int, Int)]
pythagoreanTriples n =
    [(a, b, c) | c <- [1..n],
                 b <- [1..c],
                 a <- [1..b],
                 a*a + b*b == c*c]

-- 代数的データ型 + パターンマッチ
data Tree a = Leaf | Node (Tree a) a (Tree a)

treeSum :: Tree Int -> Int
treeSum Leaf         = 0
treeSum (Node l v r) = treeSum l + v + treeSum r

-- Maybe モナド: null の安全な扱い
safeDivide :: Double -> Double -> Maybe Double
safeDivide _ 0 = Nothing
safeDivide x y = Just (x / y)

-- モナドの連鎖
calculate :: Double -> Double -> Double -> Maybe Double
calculate a b c = do
    x <- safeDivide a b    -- b が 0 なら Nothing が伝播
    y <- safeDivide x c    -- c が 0 なら Nothing が伝播
    return (x + y)
```

```javascript
// JavaScript: 関数型スタイル
// ==========================================

// --- 純粋関数 ---
const add = (a, b) => a + b;
const multiply = (a, b) => a * b;
const isPositive = n => n > 0;
const double = n => n * 2;

// --- 高階関数 ---
const applyTwice = (fn, value) => fn(fn(value));
// applyTwice(double, 3) => 12

// --- 関数合成（右から左） ---
const compose = (...fns) => x =>
    fns.reduceRight((acc, fn) => fn(acc), x);

// --- パイプライン（左から右） ---
const pipe = (...fns) => x =>
    fns.reduce((acc, fn) => fn(acc), x);

// --- 不変なデータ操作 ---
const numbers = [1, -2, 3, -4, 5];

// 命令型（状態を変更する）
let result = 0;
for (const n of numbers) {
    if (n > 0) result += n * 2;
}

// 関数型（データを変換する）
const result2 = numbers
    .filter(isPositive)
    .map(double)
    .reduce(add, 0);
// => 18

// --- カリー化 ---
const curry = fn => {
    const arity = fn.length;
    return function curried(...args) {
        if (args.length >= arity) return fn(...args);
        return (...moreArgs) => curried(...args, ...moreArgs);
    };
};

const curriedAdd = curry(add);
const add5 = curriedAdd(5);
// add5(3) => 8

// --- 不変なオブジェクト操作 ---
const user = { name: "Alice", age: 30, active: true };

// 悪い例: ミューテーション
// user.age = 31;  // 元のオブジェクトを変更してしまう

// 良い例: 新しいオブジェクトを生成
const updatedUser = { ...user, age: 31 };
// user は変更されていない
// updatedUser は { name: "Alice", age: 31, active: true }

// --- 再帰によるループの代替 ---
const sumArray = (arr) => {
    if (arr.length === 0) return 0;
    const [head, ...tail] = arr;
    return head + sumArray(tail);
};
```

### 5.4 関数型の重要パターン

```
関数型プログラミングの重要パターン
============================================================

1. Map-Filter-Reduce パイプライン
   データリスト → filter(条件) → map(変換) → reduce(集約)

   [1,-2,3,-4,5]
       |
       v  filter(x > 0)
   [1, 3, 5]
       |
       v  map(x * 2)
   [2, 6, 10]
       |
       v  reduce(+, 0)
   18

2. Maybe / Option パターン
   null を使わずに「値がない」ことを型で表現

   safeDivide(10, 0) => Nothing
   safeDivide(10, 2) => Just(5)

3. Either / Result パターン
   例外を使わずにエラーを型で表現

   parseAge("25")     => Right(25)
   parseAge("abc")    => Left("Invalid number")

4. パターンマッチ
   データの構造に基づいて処理を分岐

   match shape {
       Circle(r)        => pi * r * r,
       Rectangle(w, h)  => w * h,
       Triangle(b, h)   => b * h / 2,
   }
============================================================
```

### 5.5 関数型の強みと限界

```
+--------------------------------------------------+
|          関数型の強み                              |
+--------------------------------------------------+
| [1] テスト容易性: 純粋関数は入出力だけで検証     |
| [2] 並行安全性: 不変データ = 競合条件なし         |
| [3] 合成可能性: 小関数の組合せで大機能を構築     |
| [4] 推論容易性: 参照透過性で等式推論が可能       |
| [5] リファクタリング: 関数の差し替えが安全       |
+--------------------------------------------------+

+--------------------------------------------------+
|          関数型の限界                              |
+--------------------------------------------------+
| [1] 学習曲線: モナド、型クラス等の抽象概念        |
| [2] パフォーマンス予測: 遅延評価の挙動把握        |
| [3] 副作用の扱い: IO が本質的に煩雑              |
| [4] デバッグ: 合成された関数チェーンの追跡        |
| [5] 既存エコシステム: OOP前提のライブラリとの統合 |
+--------------------------------------------------+
```

### 5.6 関数型が輝く場面

- **データパイプライン**: ETL処理、ログ解析、データ変換
- **並行・分散処理**: MapReduce、Spark、ストリーム処理
- **コンパイラ・インタプリタ**: 構文解析、AST変換、コード生成
- **金融システム**: 取引処理、リスク計算（正確性が最優先）
- **設定・DSL**: 宣言的な設定記述、ドメイン固有言語
- **フロントエンド**: React（UI = f(state)）、Redux

---

## 6. 論理型プログラミング（Logic）

### 6.1 基本思想

```
============================================================
 論理型プログラミングの核心
============================================================

 思想: 「プログラムを論理的な命題と規則として記述し、
        問題の解を論理的推論によって導出する」

 起源: 一階述語論理、Prolog (Colmerauer, 1972)

 原則:
   1. 事実 (Facts) — 真である命題を宣言
   2. 規則 (Rules) — 事実から新たな事実を導く条件
   3. 質問 (Queries) — 推論エンジンに解の探索を依頼

 中心的な抽象: 論理的関係と単一化 (Unification)
============================================================
```

論理型は他のパラダイムと根本的に異なる。
命令型が「どう計算するか」を書き、関数型が「何を変換するか」を書くのに対し、
論理型は **「何が真であるか」** を書く。

### 6.2 コード例: Prolog

```prolog
% ==========================================
% Prolog: 家族関係の推論
% ==========================================

% --- 事実（Facts）: 真である命題を宣言 ---
parent(tom, bob).       % tom は bob の親である
parent(tom, liz).       % tom は liz の親である
parent(bob, ann).       % bob は ann の親である
parent(bob, pat).       % bob は pat の親である
parent(pat, jim).       % pat は jim の親である

male(tom).
male(bob).
male(jim).
female(liz).
female(ann).
female(pat).

% --- 規則（Rules）: 事実から新たな事実を導出 ---
father(X, Y) :- parent(X, Y), male(X).
mother(X, Y) :- parent(X, Y), female(X).

grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
grandfather(X, Z) :- grandparent(X, Z), male(X).

sibling(X, Y) :- parent(Z, X), parent(Z, Y), X \= Y.

% 推移的な祖先関係（再帰的規則）
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).

% --- 質問（Queries）: 推論エンジンに問い合わせ ---
% ?- grandparent(tom, ann).
% => true （tom -> bob -> ann）
%
% ?- grandfather(tom, X).
% => X = ann ; X = pat （tom の孫を列挙）
%
% ?- ancestor(tom, jim).
% => true （tom -> bob -> pat -> jim）
%
% ?- sibling(ann, pat).
% => true （同じ親 bob を持つ）
```

```prolog
% ==========================================
% Prolog: リスト操作
% ==========================================

% リストの長さ
list_length([], 0).
list_length([_|T], N) :- list_length(T, N1), N is N1 + 1.

% リストの連結
append([], L, L).
append([H|T1], L2, [H|T3]) :- append(T1, L2, T3).

% リストの反転
reverse([], []).
reverse([H|T], R) :- reverse(T, R1), append(R1, [H], R).

% リストのソート（挿入ソート）
insert_sorted(X, [], [X]).
insert_sorted(X, [H|T], [X,H|T]) :- X =< H.
insert_sorted(X, [H|T], [H|R]) :- X > H, insert_sorted(X, T, R).

insertion_sort([], []).
insertion_sort([H|T], Sorted) :-
    insertion_sort(T, SortedTail),
    insert_sorted(H, SortedTail, Sorted).
```

### 6.3 論理型の応用領域

- **人工知能**: エキスパートシステム、知識表現、自然言語処理
- **データベース**: Datalog、SQL（宣言的クエリは論理型の影響）
- **型検査**: 型推論アルゴリズムは単一化に基づく
- **定理証明**: Coq, Isabelle（論理型の発展形）
- **構成管理**: Ansible, Terraform（宣言的な状態定義）

---

## 7. リアクティブプログラミングとアクターモデル

### 7.1 リアクティブプログラミング

```
============================================================
 リアクティブプログラミングの核心
============================================================

 思想: 「データの変化を自動的に伝播するストリームとして
        非同期イベントを宣言的に処理する」

 起源: Functional Reactive Programming (Elliott & Hudak, 1997)

 原則:
   1. Observable（データストリーム）
   2. Operators（ストリーム変換）
   3. Subscription（結果の消費）
   4. バックプレッシャー（流量制御）

 中心的な抽象: データストリーム（時間軸上の値のシーケンス）
============================================================
```

```
リアクティブストリームの概念図
============================================================

時間 →
               click    click         click
ユーザー入力:  --[A]------[B]----------[C]------>

                 |         |             |
                 v         v             v
debounce(300ms): --------[A]-----------[B]---[C]->

                          |              |     |
                          v              v     v
filter(len>=2):  --------[AB]-----------[BC]------>

                          |              |
                          v              v
switchMap(API):  --------[結果1]--------[結果2]--->

============================================================
```

```javascript
// RxJS: リアクティブプログラミングの例
// ==========================================

import { fromEvent, interval, merge } from 'rxjs';
import {
    debounceTime, map, filter, switchMap,
    scan, takeUntil, distinctUntilChanged,
    catchError, retry
} from 'rxjs/operators';
import { of } from 'rxjs';

// --- 検索のオートコンプリート ---
const searchInput = document.getElementById('search');

const search$ = fromEvent(searchInput, 'input').pipe(
    debounceTime(300),                          // 300ms の静寂を待つ
    map(event => event.target.value.trim()),    // 値を取得
    filter(query => query.length >= 2),         // 2文字以上のみ
    distinctUntilChanged(),                     // 変化がない場合はスキップ
    switchMap(query =>                          // 最新のリクエストのみ有効
        fetch(`/api/search?q=${encodeURIComponent(query)}`)
            .then(res => res.json())
    ),
    catchError(err => {                         // エラーハンドリング
        console.error('Search error:', err);
        return of([]);
    })
);

search$.subscribe(results => {
    renderResults(results);
});

// --- カウンター: 複数ストリームの合成 ---
const increment$ = fromEvent(document.getElementById('inc'), 'click')
    .pipe(map(() => +1));
const decrement$ = fromEvent(document.getElementById('dec'), 'click')
    .pipe(map(() => -1));

const counter$ = merge(increment$, decrement$).pipe(
    scan((count, delta) => count + delta, 0)
);

counter$.subscribe(count => {
    document.getElementById('display').textContent = count;
});
```

### 7.2 アクターモデル

```
============================================================
 アクターモデルの核心
============================================================

 思想: 「並行計算をメッセージパッシングする独立した
        アクター（軽量プロセス）の集合として記述する」

 起源: Carl Hewitt (1973), Erlang (1986)

 原則:
   1. アクター = 状態 + メッセージキュー + 振る舞い
   2. 通信はメッセージパッシングのみ（共有メモリなし）
   3. 各アクターは他のアクターの内部状態に直接アクセスできない
   4. アクターは新しいアクターを生成できる
   5. アクターは受け取ったメッセージに応じて振る舞いを変更できる

 中心的な抽象: アクター（独立した計算実体）
============================================================
```

```
アクターモデルのアーキテクチャ
============================================================

  ┌─────────────────────────────────────────────────┐
  │               Supervisor (監視者)                │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
  │  │ Actor A  │  │ Actor B  │  │ Actor C  │      │
  │  │          │  │          │  │          │      │
  │  │ [状態]   │  │ [状態]   │  │ [状態]   │      │
  │  │ [メール  │  │ [メール  │  │ [メール  │      │
  │  │  ボックス]│  │  ボックス]│  │  ボックス]│      │
  │  └────┬─────┘  └─────┬────┘  └────┬─────┘      │
  │       │   メッセージ  │            │             │
  │       │←──────────────┘            │             │
  │       │                            │             │
  │       └───────────メッセージ───────→│             │
  │                                                  │
  │  障害時の戦略:                                    │
  │    one_for_one  — 障害アクターだけ再起動          │
  │    one_for_all  — 全子アクターを再起動            │
  │    rest_for_one — 障害以降のアクターを再起動      │
  └─────────────────────────────────────────────────┘

  "Let it crash" 哲学:
    エラーを防ぐのではなく、エラーからの回復を設計する
============================================================
```

```elixir
# Elixir: アクターモデル (GenServer) の例
# ==========================================

defmodule Counter do
  use GenServer

  # --- クライアント API ---

  def start_link(initial_value \\ 0) do
    GenServer.start_link(__MODULE__, initial_value, name: __MODULE__)
  end

  def increment do
    GenServer.cast(__MODULE__, :increment)
  end

  def decrement do
    GenServer.cast(__MODULE__, :decrement)
  end

  def get_value do
    GenServer.call(__MODULE__, :get_value)
  end

  # --- サーバーコールバック ---

  @impl true
  def init(initial_value) do
    {:ok, initial_value}
  end

  @impl true
  def handle_cast(:increment, count) do
    {:noreply, count + 1}     # 状態を更新して返す
  end

  @impl true
  def handle_cast(:decrement, count) do
    {:noreply, count - 1}
  end

  @impl true
  def handle_call(:get_value, _from, count) do
    {:reply, count, count}    # 現在の値を返す
  end
end

# 使用例:
# {:ok, _pid} = Counter.start_link(0)
# Counter.increment()
# Counter.increment()
# Counter.increment()
# Counter.decrement()
# Counter.get_value()  # => 2
```

### 7.3 リアクティブ/アクターの適用領域

| モデル | 適用領域 | 代表的技術 |
|--------|---------|-----------|
| リアクティブ | UI/UX、リアルタイム検索、ダッシュボード | RxJS, RxJava, Reactor |
| アクター | 分散システム、IoT、テレコム、チャット | Erlang/OTP, Akka, Orleans |
| CQRS/ES | イベント駆動アーキテクチャ、監査ログ | EventStoreDB, Axon |

---

## 8. マルチパラダイム -- 現代の主流

### 8.1 なぜマルチパラダイムが主流になったか

2000年代以降、以下の変化がマルチパラダイムへの移行を加速させた。

1. **マルチコアの普及**: 並行処理の必要性が増し、関数型の不変性が重要に
2. **分散システム**: マイクロサービスの浸透で、異なるパラダイムの混在が自然に
3. **開発者体験**: 単一パラダイムの制約が生産性を阻害する場面が増加
4. **言語設計の成熟**: 複数パラダイムを矛盾なく統合する技法が確立

### 8.2 主要言語のパラダイムマッピング

```
主要言語のパラダイム対応表
============================================================

言語          手続き型  OOP   関数型  並行   ジェネリクス  その他
─────────────────────────────────────────────────────────────
Python          ●       ●      ◐       ◐        ◐       動的型付
JavaScript      ●       ◐      ●       ◐        -       プロトタイプ
TypeScript      ●       ●      ●       ◐        ●       構造的型付
Java            ●       ●      ◐       ◐        ●       -
C#              ●       ●      ◐       ◐        ●       LINQ
Kotlin          ●       ●      ●       ●        ●       コルーチン
Swift           ●       ●      ●       ●        ●       プロトコル指向
Rust            ●       ◐      ●       ●        ●       所有権
Go              ●       ◐      ◐       ●        ●       goroutine
Scala           ●       ●      ●       ●        ●       暗黙変換
Haskell         -       -      ●       ◐        ●       遅延評価
Elixir          -       -      ●       ●        -       アクター
─────────────────────────────────────────────────────────────
● = 第一級サポート  ◐ = 部分的サポート  - = 非対応/最小限
```

### 8.3 コード例: TypeScript マルチパラダイム

```typescript
// TypeScript: マルチパラダイムの実践例
// ==========================================

// ---- 型定義（代数的データ型的アプローチ） ----

type Result<T, E = Error> =
    | { ok: true; value: T }
    | { ok: false; error: E };

interface User {
    readonly id: string;
    readonly name: string;
    readonly email: string;
    readonly role: 'admin' | 'user' | 'guest';
    readonly createdAt: Date;
}

// ---- リポジトリインターフェース（OOP: 抽象化） ----

interface UserRepository {
    findById(id: string): Promise<Result<User, string>>;
    findAll(): Promise<User[]>;
    save(user: User): Promise<Result<void, string>>;
}

// ---- 関数型: 純粋なドメインロジック ----

// バリデーション関数（純粋関数）
const validateEmail = (email: string): Result<string, string> => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email)
        ? { ok: true, value: email }
        : { ok: false, error: `Invalid email: ${email}` };
};

const validateName = (name: string): Result<string, string> =>
    name.length >= 2 && name.length <= 50
        ? { ok: true, value: name.trim() }
        : { ok: false, error: `Name must be 2-50 chars: ${name}` };

// 関数合成: バリデーションのパイプライン
const pipe = <T>(...fns: Array<(arg: T) => T>) =>
    (value: T): T => fns.reduce((acc, fn) => fn(acc), value);

// フィルタリング（関数型: 高階関数）
const filterUsers = (predicate: (u: User) => boolean) =>
    (users: User[]): User[] => users.filter(predicate);

const isAdmin = (u: User): boolean => u.role === 'admin';
const isActive = (u: User): boolean =>
    u.createdAt > new Date(Date.now() - 90 * 24 * 60 * 60 * 1000);

const getActiveAdmins = (users: User[]): User[] =>
    pipe(
        filterUsers(isAdmin),
        filterUsers(isActive),
    )(users);

// ---- OOP: サービス層（状態と依存関係の管理） ----

class UserService {
    constructor(
        private readonly repo: UserRepository,
        private readonly logger: { info: (msg: string) => void }
    ) {}

    async createUser(
        name: string,
        email: string,
        role: User['role'] = 'user'
    ): Promise<Result<User, string>> {
        // 関数型: バリデーション
        const nameResult = validateName(name);
        if (!nameResult.ok) return nameResult;

        const emailResult = validateEmail(email);
        if (!emailResult.ok) return emailResult;

        // OOP: オブジェクトの生成
        const user: User = {
            id: crypto.randomUUID(),
            name: nameResult.value,
            email: emailResult.value,
            role,
            createdAt: new Date(),
        };

        // 手続き型: 逐次的な副作用の実行
        const saveResult = await this.repo.save(user);
        if (!saveResult.ok) return saveResult;

        this.logger.info(`User created: ${user.id}`);
        return { ok: true, value: user };
    }

    async getActiveAdmins(): Promise<User[]> {
        const users = await this.repo.findAll();
        // 関数型: データ変換パイプライン
        return getActiveAdmins(users);
    }
}

// ---- 使い分けのポイント ----
// ドメインロジック（バリデーション、変換） → 関数型（純粋関数）
// 依存関係の管理（リポジトリ、ロガー）     → OOP（依存性注入）
// 副作用の実行（DB保存、ログ出力）         → 手続き型（逐次実行）
```

### 8.4 パラダイム混合のベストプラクティス

```
マルチパラダイム設計の指針
============================================================

レイヤー           推奨パラダイム       理由
────────────────────────────────────────────────────
プレゼンテーション  リアクティブ/宣言型  UI の状態管理
アプリケーション   OOP (DI + サービス)   依存関係の管理
ドメインロジック   関数型 (純粋関数)     テスト容易性・推論性
インフラ           手続き型              副作用の明示的管理
データアクセス     OOP (リポジトリ)      抽象化とカプセル化
ユーティリティ     関数型                合成可能性・再利用性
────────────────────────────────────────────────────

原則:
  [1] 純粋な部分を最大化し、副作用の部分を最小化する
  [2] 副作用は端（エッジ）に押し出す
      = Functional Core, Imperative Shell パターン
  [3] データの変換は関数型、状態の管理はOOP
  [4] 並行処理はアクターモデルまたは関数型
============================================================
```

### 8.5 Functional Core, Imperative Shell パターン

このパターンは Gary Bernhardt が提唱した設計手法で、
マルチパラダイム設計の最も重要な原則を体現している。

```
Functional Core, Imperative Shell
============================================================

  ┌─────────────────────────────────────────────┐
  │              Imperative Shell                │
  │    (副作用を含む外殻: DB, API, ファイル)      │
  │                                              │
  │    ┌─────────────────────────────────┐       │
  │    │        Functional Core          │       │
  │    │   (純粋関数によるビジネスロジック)│       │
  │    │                                 │       │
  │    │   input → validate → transform  │       │
  │    │     → calculate → format        │       │
  │    │                                 │       │
  │    │   テスト容易 / 推論容易          │       │
  │    │   並行安全 / 合成可能            │       │
  │    └─────────────────────────────────┘       │
  │                                              │
  │    Shell: DBから読む → Core → DBに書く       │
  └─────────────────────────────────────────────┘

  Core のテスト: 単純な入出力の検証（モック不要）
  Shell のテスト: 統合テスト（限定的で良い）
============================================================
```
