# 命令型プログラミング（Imperative Programming）

> 命令型プログラミングは「コンピュータに手順を指示する」最も直感的なパラダイムであり、フォン・ノイマンアーキテクチャと直接対応する。本章では、命令型プログラミングの本質から、構造化プログラミング、手続き型プログラミング、そして現代言語における命令型スタイルまでを体系的に解説する。

## この章で学ぶこと

- [ ] 命令型プログラミングの本質と歴史的経緯を説明できる
- [ ] 構造化プログラミングの三つの制御構造を正しく使い分けられる
- [ ] 状態管理と副作用の概念を理解し、安全なコードを書ける
- [ ] 手続き型プログラミングにおけるモジュール化の設計原則を適用できる
- [ ] 命令型と他のパラダイム（宣言型・関数型・OOP）の違いを比較分析できる
- [ ] 命令型特有のアンチパターンを認識し、回避策を実践できる

---

## 1. 命令型プログラミングとは

### 1.1 定義と基本概念

命令型プログラミング（Imperative Programming）は、プログラムを「命令（instruction）の列」として記述するプログラミングパラダイムである。プログラマはコンピュータに対して「何を（What）」ではなく「どのように（How）」処理を行うかを逐次的に指示する。

この「命令」という言葉は、ラテン語の "imperare"（命じる）に由来する。日常の料理レシピが「卵を割る → 塩を加える → かき混ぜる → フライパンで焼く」と手順を列挙するように、命令型プログラムは計算の各ステップを順番に記述する。

命令型プログラミングの核心は以下の三つの要素に集約される。

```
命令型プログラミングの三本柱
============================================

  1. 状態（State）
     - 変数としてメモリ上に値を保持する
     - プログラムの「現在の状況」を表す

  2. 代入（Assignment）
     - 変数の値を変更する操作
     - x = x + 1 は数学的には矛盾するが、
       命令型では「xの値を1増やす」という命令

  3. 制御フロー（Control Flow）
     - 命令の実行順序を制御する
     - 順次（逐次実行）、分岐（条件判定）、
       反復（ループ）の三構造
```

### 1.2 フォン・ノイマンアーキテクチャとの対応

命令型プログラミングがなぜ最も「自然な」プログラミングスタイルであるかを理解するには、現代のコンピュータの基本構造であるフォン・ノイマンアーキテクチャとの対応関係を知る必要がある。

```
フォン・ノイマンアーキテクチャと命令型プログラミングの対応
==========================================================

  ┌─────────────────────────────────────────────────────┐
  │            フォン・ノイマンマシン                       │
  │                                                     │
  │  ┌───────────┐    バス    ┌──────────────────┐     │
  │  │   CPU     │◄─────────►│    メモリ          │     │
  │  │           │           │                  │     │
  │  │ PC(次命令) │           │ プログラム領域    │     │
  │  │ ACC(演算)  │           │  命令1           │     │
  │  │ IR(命令)   │           │  命令2           │     │
  │  │           │           │  命令3 ...       │     │
  │  └───────────┘           │                  │     │
  │                          │ データ領域        │     │
  │                          │  変数A = 10      │     │
  │                          │  変数B = 20      │     │
  │                          │  変数C = ?       │     │
  │                          └──────────────────┘     │
  └─────────────────────────────────────────────────────┘

  対応関係:
  ┌─────────────────┬──────────────────────────────┐
  │ ハードウェア      │ 命令型プログラミング            │
  ├─────────────────┼──────────────────────────────┤
  │ メモリセル       │ 変数                          │
  │ ストア命令       │ 代入文（x = 10）              │
  │ ロード命令       │ 変数参照（print(x)）          │
  │ プログラムカウンタ │ 現在の実行位置               │
  │ 条件分岐命令     │ if-else文                     │
  │ ジャンプ命令     │ goto / ループ                  │
  │ サブルーチン呼出  │ 関数呼び出し                   │
  └─────────────────┴──────────────────────────────┘
```

1945 年にジョン・フォン・ノイマンが提唱したこのアーキテクチャでは、プログラムとデータが同一のメモリに格納され、CPU がメモリから命令を一つずつ取り出して（フェッチ）、解読し（デコード）、実行する（エクスキュート）。命令型プログラミングは、このフェッチ・デコード・エクスキュートサイクルを抽象化したものにほかならない。

### 1.3 命令型の基本例：合計値の計算

最もシンプルな命令型プログラムの例として、リスト内の数値の合計を計算するコードを見てみよう。

**Python による命令型スタイル:**

```python
def sum_imperative(numbers):
    """
    命令型スタイルでリストの合計値を計算する。
    状態変数 total を用い、ループで逐次更新していく。
    """
    total = 0                    # 状態の初期化
    index = 0                    # ループカウンタの初期化
    while index < len(numbers):  # 反復（終了条件の判定）
        total = total + numbers[index]  # 状態の更新（代入）
        index = index + 1               # カウンタの更新
    return total                 # 最終状態を返す

# 実行例
data = [10, 20, 30, 40, 50]
result = sum_imperative(data)
print(f"合計: {result}")  # 合計: 150
```

**C 言語による同等のコード:**

```c
#include <stdio.h>

int sum_imperative(int numbers[], int length) {
    int total = 0;                  /* 状態の初期化 */
    int index = 0;                  /* ループカウンタの初期化 */
    while (index < length) {        /* 反復 */
        total = total + numbers[index];  /* 状態の更新 */
        index = index + 1;              /* カウンタの更新 */
    }
    return total;                   /* 最終状態を返す */
}

int main(void) {
    int data[] = {10, 20, 30, 40, 50};
    int length = sizeof(data) / sizeof(data[0]);
    int result = sum_imperative(data, length);
    printf("合計: %d\n", result);   /* 合計: 150 */
    return 0;
}
```

このコードでは、変数 `total` が「現在の合計値」という状態を保持し、ループの各反復で代入文によって状態が更新される。プログラムの各時点における `total` の値は以下のように遷移する。

```
状態遷移の追跡（numbers = [10, 20, 30, 40, 50]）
=================================================

  繰り返し   index   numbers[index]   total（更新前）   total（更新後）
  ─────────────────────────────────────────────────────────────────
  初期状態     0         -                -                0
  1回目       0        10                0               10
  2回目       1        20               10               30
  3回目       2        30               30               60
  4回目       3        40               60              100
  5回目       4        50              100              150
  ─────────────────────────────────────────────────────────────────
  終了        5         -              150        → return 150
```

### 1.4 命令型 vs 宣言型：根本的な思考の違い

命令型プログラミングを理解する上で、対照的なパラダイムである宣言型プログラミング（Declarative Programming）との比較が有効である。

```
命令型と宣言型の根本的な違い
========================================

  ┌──────────────────────────────────────────────────────┐
  │  命令型（Imperative）: 「どうやって」計算するかを記述    │
  │                                                      │
  │  手順書のようなもの:                                    │
  │  Step 1: 空のリストを用意する                           │
  │  Step 2: 元のリストの各要素を見る                       │
  │  Step 3: 条件を満たすなら新リストに追加する              │
  │  Step 4: 全要素を処理したら終了                         │
  └──────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────┐
  │  宣言型（Declarative）: 「何が」欲しいかを記述           │
  │                                                      │
  │  注文書のようなもの:                                    │
  │  「価格が100以上の商品名の一覧をください」               │
  │  → 具体的な手順はシステム側が決定する                    │
  └──────────────────────────────────────────────────────┘
```

具体的なコードで比較する。

**命令型（Python）:**

```python
# 命令型: 「どうやって」フィルタリングするかを逐一指示する
items = [
    {"name": "ノートPC", "price": 150000},
    {"name": "マウス", "price": 3000},
    {"name": "キーボード", "price": 12000},
    {"name": "モニター", "price": 45000},
    {"name": "USBケーブル", "price": 800},
]

expensive_items = []                  # 空リストを用意
for item in items:                    # 各要素を順に処理
    if item["price"] >= 10000:        # 条件を判定
        expensive_items.append(item["name"])  # 結果に追加

print(expensive_items)
# ['ノートPC', 'キーボード', 'モニター']
```

**宣言型（SQL）:**

```sql
-- 宣言型: 「何が」欲しいかだけを記述する
SELECT name
FROM items
WHERE price >= 10000;

-- 実行エンジンがインデックスの使用、テーブルスキャン等を自動判断する
```

**宣言型（Python リスト内包表記）:**

```python
# 宣言型寄りのスタイル
expensive_items = [item["name"] for item in items if item["price"] >= 10000]
```

| 比較項目 | 命令型 | 宣言型 |
|---------|--------|--------|
| 記述内容 | 手順（How） | 結果（What） |
| 抽象度 | 低い（機械に近い） | 高い（人間に近い） |
| 状態管理 | 明示的に管理する | 処理系が管理する |
| 副作用 | 頻繁に発生する | 最小限に抑える |
| 最適化 | 開発者が行う | 処理系が行う |
| 学習曲線 | 直感的で取り組みやすい | 発想の転換が必要 |
| デバッグ | ステップ実行で追跡しやすい | 内部動作が見えにくい |
| 代表言語 | C, Pascal, BASIC | SQL, HTML, Haskell |
| 適用領域 | システムプログラミング、制御処理 | データ問い合わせ、UI定義 |

---

## 2. 命令型プログラミングの歴史

### 2.1 機械語とアセンブリ言語の時代（1940年代〜1950年代前半）

命令型プログラミングの歴史は、コンピュータそのものの歴史と重なる。最初期のプログラミングは機械語（マシンコード）で行われた。

1940 年代、ENIAC（1945年）や EDVAC（1949年）といった初期のコンピュータでは、プログラムは配線の接続変更やスイッチの設定で行われていた。フォン・ノイマンが提唱したストアドプログラム方式の導入により、プログラムをデータと同様にメモリに格納できるようになった。

```
機械語プログラムの例（仮想的な8ビットCPU）
==========================================

  アドレス   機械語        意味
  ─────────────────────────────────────────
  0000:     0001 0001    LOAD  R1, [addr1]    ; メモリからR1にロード
  0001:     0001 0010    LOAD  R2, [addr2]    ; メモリからR2にロード
  0010:     0010 0001    ADD   R1, R2         ; R1 = R1 + R2
  0011:     0011 0001    STORE R1, [addr3]    ; R1をメモリに格納
  0100:     1111 0000    HALT                 ; 停止

  → 0と1の羅列で、人間には極めて読みにくい
```

1950 年代前半にアセンブリ言語が登場し、機械語命令にニーモニック（人間が読める記号名）が付けられた。これにより可読性は向上したが、依然としてハードウェアに密着した低水準の命令型プログラミングであった。

```nasm
; x86アセンブリ言語の例：2つの数値の加算
section .data
    num1 dd 10          ; 32ビット整数 10
    num2 dd 20          ; 32ビット整数 20
    result dd 0         ; 結果格納用

section .text
    global _start

_start:
    mov eax, [num1]     ; num1の値をEAXレジスタにロード
    add eax, [num2]     ; num2の値をEAXに加算
    mov [result], eax   ; 結果をメモリに格納

    ; プログラム終了（Linux システムコール）
    mov eax, 1          ; sys_exit
    xor ebx, ebx        ; 終了コード 0
    int 0x80            ; カーネル呼び出し
```

### 2.2 FORTRAN の登場（1957年）

1957 年、IBM のジョン・バッカス（John Backus）率いるチームが FORTRAN（FORmula TRANslation）を発表した。これは世界初の実用的な高水準プログラミング言語であり、命令型プログラミングの歴史における画期的な転換点であった。

FORTRAN は数学の公式をほぼそのままプログラムとして記述できるように設計された。科学技術計算の分野で爆発的に普及し、アセンブリ言語で書いた場合の 20 分の 1 程度の労力でプログラムを書けるようになった。

```fortran
C     FORTRAN 77 による数値積分（台形公式）
C     命令型プログラミングの典型例
      PROGRAM TRAPEZOID
      IMPLICIT NONE
      INTEGER N, I
      REAL A, B, H, SUM, X

C     積分区間と分割数の設定
      A = 0.0
      B = 1.0
      N = 1000

C     刻み幅の計算
      H = (B - A) / N

C     台形公式による数値積分
      SUM = 0.0
      DO 10 I = 1, N - 1
          X = A + I * H
          SUM = SUM + X * X
   10 CONTINUE

      SUM = H * (A*A/2.0 + SUM + B*B/2.0)

      WRITE(*,*) 'Integral of x^2 from 0 to 1 =', SUM
C     理論値: 1/3 = 0.333333...
      STOP
      END
```

### 2.3 ALGOL と構造化への萌芽（1958年〜1960年代）

1958 年に ALGOL 58、1960 年に ALGOL 60 が国際委員会により策定された。ALGOL（ALGOrithmic Language）は、後続のほぼ全ての命令型言語に影響を与えた極めて重要な言語である。

ALGOL 60 が導入した革新的な概念は多い。

- ブロック構造（begin ... end）
- レキシカルスコープ（変数の有効範囲の入れ子構造）
- 再帰呼び出し
- BNF（バッカス・ナウア記法）による構文の形式的定義
- 値呼び出しと名前呼び出し

ALGOL は学術界で広く受け入れられたが、商用コンピュータでの普及には至らなかった。しかし、その設計思想は Pascal、C、Java といった後続言語に大きな影響を与えた。

### 2.4 C 言語の誕生と普及（1972年）

1972 年、ベル研究所のデニス・リッチー（Dennis Ritchie）が C 言語を開発した。C は、ケン・トンプソン（Ken Thompson）が開発した B 言語の後継であり、UNIX オペレーティングシステムを記述するために設計された。

C 言語の特徴は、高水準言語の抽象化能力とアセンブリ言語に近い低水準操作の両方を兼ね備えた点にある。ポインタによるメモリの直接操作、構造体によるデータの組織化、関数によるモジュール化など、命令型プログラミングの本質的な機能を過不足なく提供した。

```c
/*
 * C言語による動的配列の実装
 * 命令型プログラミングの特徴が凝縮された例:
 *   - 明示的なメモリ管理（malloc/realloc/free）
 *   - ポインタによる間接参照
 *   - 構造体による状態のカプセル化
 *   - 手続き（関数）による操作の定義
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int *data;       /* 要素を格納する配列へのポインタ */
    int size;        /* 現在の要素数 */
    int capacity;    /* 確保済みの容量 */
} DynamicArray;

/* 初期化 */
DynamicArray *array_create(int initial_capacity) {
    DynamicArray *arr = malloc(sizeof(DynamicArray));
    if (arr == NULL) return NULL;
    arr->data = malloc(sizeof(int) * initial_capacity);
    if (arr->data == NULL) {
        free(arr);
        return NULL;
    }
    arr->size = 0;
    arr->capacity = initial_capacity;
    return arr;
}

/* 要素の追加 */
int array_push(DynamicArray *arr, int value) {
    if (arr->size >= arr->capacity) {
        int new_capacity = arr->capacity * 2;
        int *new_data = realloc(arr->data, sizeof(int) * new_capacity);
        if (new_data == NULL) return -1;
        arr->data = new_data;
        arr->capacity = new_capacity;
    }
    arr->data[arr->size] = value;
    arr->size++;
    return 0;
}

/* 要素の取得 */
int array_get(DynamicArray *arr, int index) {
    if (index < 0 || index >= arr->size) {
        fprintf(stderr, "Index out of bounds: %d\n", index);
        exit(1);
    }
    return arr->data[index];
}

/* 解放 */
void array_destroy(DynamicArray *arr) {
    if (arr != NULL) {
        free(arr->data);
        free(arr);
    }
}

int main(void) {
    DynamicArray *arr = array_create(4);

    for (int i = 0; i < 10; i++) {
        array_push(arr, i * 10);
    }

    printf("Size: %d, Capacity: %d\n", arr->size, arr->capacity);
    /* Size: 10, Capacity: 16 */

    for (int i = 0; i < arr->size; i++) {
        printf("[%d] = %d\n", i, array_get(arr, i));
    }

    array_destroy(arr);
    return 0;
}
```

### 2.5 命令型言語の系譜

```
命令型プログラミング言語の系譜
══════════════════════════════════════════════════════════════

  1940s   機械語 ─────┐
                      │
  1950s   アセンブリ ──┤
          FORTRAN(1957)├──── 科学技術計算の基盤
          COBOL(1959) ├──── ビジネスデータ処理
                      │
  1960s   ALGOL 60 ───┤──── 全ての構造化言語の祖先
          BASIC(1964) ├──── 教育用、初心者向け
          PL/I(1964)  │
                      │
  1970s   Pascal(1970)├──── 教育・構造化プログラミング
          C(1972)     ├──── システムプログラミング
          ↓           │
  1980s   C++(1983)   ├──── C + オブジェクト指向
          Ada(1983)   │
          Perl(1987)  ├──── テキスト処理・スクリプト
                      │
  1990s   Python(1991)├──── マルチパラダイム
          Java(1995)  ├──── OOP + VM
          PHP(1995)   ├──── Web開発
          Ruby(1995)  │
          JavaScript(1995)
                      │
  2000s   C#(2000)    ├──── .NET プラットフォーム
          Go(2009)    ├──── 並行処理重視
                      │
  2010s   Rust(2010)  ├──── 安全性 + 性能
          Kotlin(2011)├──── JVM上のモダン言語
          Swift(2014) ├──── Apple プラットフォーム
          TypeScript(2012)
                      │
  2020s   各言語がマルチパラダイム化
          命令型 + 関数型 + OOP の融合が主流に
```

---

## 3. 制御構造（Control Structures）

### 3.1 構造化定理

1966 年、コラド・ベーム（Corrado Bohm）とジュゼッペ・ヤコピーニ（Giuseppe Jacopini）は、任意のプログラムの制御フローが以下の三つの基本構造の組み合わせだけで表現できることを数学的に証明した。これが「構造化定理」（Structured Program Theorem）である。

```
構造化定理：三つの基本制御構造
═══════════════════════════════════════════════════

  1. 順次（Sequence）          2. 選択（Selection）
  ┌─────────┐                ┌─────────┐
  │ 文1     │                │ 条件P?  │
  └────┬────┘                └──┬───┬──┘
       │                    Yes │   │ No
  ┌────▼────┐            ┌─────▼┐ ┌▼─────┐
  │ 文2     │            │ 文A  │ │ 文B  │
  └────┬────┘            └──┬───┘ └──┬───┘
       │                    │        │
  ┌────▼────┐               └───┬────┘
  │ 文3     │                   │
  └─────────┘                   ▼

  3. 反復（Iteration）
       │
  ┌────▼────┐
  │ 条件P?  │◄──────┐
  └──┬───┬──┘       │
  Yes│   │No    ┌───┴───┐
     │   │      │ 文S   │
     │   ▼      └───────┘
     │
     └──────►┌───────┐
             │ 文S   │──────┐
             └───────┘      │
                            │
             ┌──────────────┘
             │
             ▼（条件Pへ戻る）

  構造化定理の意味:
  goto文がなくても、この3構造の組み合わせだけで
  あらゆるアルゴリズムを表現できる。
```

### 3.2 順次構造（Sequence）

順次構造は最も基本的な制御構造であり、文（statement）を上から下へ順番に実行する。代入文、関数呼び出し、入出力操作など、ほぼ全ての文は順次構造に従う。

```python
# 順次構造の例：円の面積と円周を計算する
import math

radius = 5.0                                # 文1: 半径の設定
area = math.pi * radius ** 2                # 文2: 面積の計算
circumference = 2 * math.pi * radius        # 文3: 円周の計算

print(f"半径: {radius}")                     # 文4: 結果の出力
print(f"面積: {area:.2f}")                   # 文5
print(f"円周: {circumference:.2f}")          # 文6

# 出力:
# 半径: 5.0
# 面積: 78.54
# 円周: 31.42
```

順次構造では文の順序が結果に影響することが重要である。

```python
# 順序の違いが結果に影響する例
x = 10
y = x + 5    # y = 15
x = 20       # xが変更されても、yはすでに計算済み
print(y)     # 15（20 + 5 = 25 ではない）

# 対比：宣言型のスプレッドシートでは
# セルB1 = A1 + 5 とすると、A1を変更するとB1も自動更新される
# → これが命令型と宣言型の根本的な違い
```

### 3.3 選択構造（Selection）

選択構造は、条件式の真偽に基づいて実行する文を選び分ける制御構造である。

**基本的な if-else 文（Python）:**

```python
def classify_temperature(celsius):
    """
    温度を分類する選択構造の例。
    複数の条件を elif で連鎖させるパターン。
    """
    if celsius >= 35:
        category = "猛暑"
        advice = "外出を控え、こまめに水分を補給すること"
    elif celsius >= 30:
        category = "真夏日"
        advice = "熱中症に注意すること"
    elif celsius >= 25:
        category = "夏日"
        advice = "適度な水分補給を心がけること"
    elif celsius >= 15:
        category = "快適"
        advice = "過ごしやすい気温"
    elif celsius >= 5:
        category = "肌寒い"
        advice = "上着を用意すること"
    elif celsius >= 0:
        category = "寒い"
        advice = "防寒対策をしっかり行うこと"
    else:
        category = "厳寒"
        advice = "凍結に注意すること"

    return category, advice

# 実行例
for temp in [38, 28, 10, -3]:
    cat, adv = classify_temperature(temp)
    print(f"{temp}°C → {cat}: {adv}")

# 出力:
# 38°C → 猛暑: 外出を控え、こまめに水分を補給すること
# 28°C → 夏日: 適度な水分補給を心がけること
# 10°C → 肌寒い: 上着を用意すること
# -3°C → 厳寒: 凍結に注意すること
```

**switch-case 文（C 言語）:**

```c
#include <stdio.h>

/*
 * switch文による多分岐の例。
 * 曜日の番号から名前を返す。
 * break を忘れるとフォールスルーが発生するため注意が必要。
 */
const char *day_name(int day) {
    switch (day) {
        case 0: return "日曜日";
        case 1: return "月曜日";
        case 2: return "火曜日";
        case 3: return "水曜日";
        case 4: return "木曜日";
        case 5: return "金曜日";
        case 6: return "土曜日";
        default: return "不明";
    }
}

/* フォールスルーを意図的に使う例 */
void classify_day(int day) {
    switch (day) {
        case 0:
        case 6:
            printf("%s は休日\n", day_name(day));
            break;
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
            printf("%s は平日\n", day_name(day));
            break;
        default:
            printf("無効な曜日番号: %d\n", day);
            break;
    }
}

int main(void) {
    for (int i = 0; i <= 7; i++) {
        classify_day(i);
    }
    return 0;
}
/*
 * 出力:
 * 日曜日 は休日
 * 月曜日 は平日
 * 火曜日 は平日
 * 水曜日 は平日
 * 木曜日 は平日
 * 金曜日 は平日
 * 土曜日 は休日
 * 無効な曜日番号: 7
 */
```

### 3.4 反復構造（Iteration）

反復構造は、条件が満たされている間（あるいは満たされるまで）、文を繰り返し実行する制御構造である。

**while ループ（前判定反復）:**

```python
def binary_search(sorted_list, target):
    """
    二分探索：while ループによる反復構造の典型例。
    ソート済みリストから目的の要素を効率的に探索する。

    - 前判定ループ: 条件を先に判定してからループ本体を実行
    - 状態変数 low, high を更新しながら探索範囲を半減させる
    """
    low = 0
    high = len(sorted_list) - 1

    while low <= high:                       # 前判定: 探索範囲が存在する間
        mid = (low + high) // 2              # 中央のインデックスを計算
        if sorted_list[mid] == target:       # 選択: 見つかった場合
            return mid
        elif sorted_list[mid] < target:      # 選択: 右半分を探索
            low = mid + 1                    # 状態の更新
        else:                                # 選択: 左半分を探索
            high = mid - 1                   # 状態の更新

    return -1  # 見つからなかった場合

# 実行例
data = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
print(binary_search(data, 23))   # 5（インデックス5に存在）
print(binary_search(data, 50))   # -1（存在しない）
```

**for ループ（回数指定反復）:**

```python
def selection_sort(arr):
    """
    選択ソート：二重ループの典型例。
    外側ループ: 確定する位置を順に処理
    内側ループ: 未ソート部分から最小値を探索
    """
    n = len(arr)
    for i in range(n - 1):              # 外側ループ: n-1回繰り返す
        min_index = i
        for j in range(i + 1, n):       # 内側ループ: 未ソート部分を走査
            if arr[j] < arr[min_index]:
                min_index = j
        # 最小値を先頭と交換（代入による状態変更）
        arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr

# 実行例
data = [64, 25, 12, 22, 11]
print(f"ソート前: {data}")
result = selection_sort(data)
print(f"ソート後: {result}")

# ソート前: [64, 25, 12, 22, 11]
# ソート後: [11, 12, 22, 25, 64]
```

**do-while ループ（後判定反復）:**

```c
#include <stdio.h>

/*
 * do-while ループの例：ユーザ入力の検証
 * 「少なくとも1回は実行する」場合に適する。
 * Python には do-while 構文がないため、C言語で示す。
 */
int main(void) {
    int number;

    do {
        printf("1から100の整数を入力してください: ");
        scanf("%d", &number);

        if (number < 1 || number > 100) {
            printf("範囲外です。もう一度入力してください。\n");
        }
    } while (number < 1 || number > 100);  /* 後判定 */

    printf("入力された値: %d\n", number);
    return 0;
}
```

### 3.5 制御構造の組み合わせ

実際のプログラムでは、三つの制御構造を自由に組み合わせてアルゴリズムを構築する。

```python
def find_primes_sieve(limit):
    """
    エラトステネスの篩：三つの制御構造を組み合わせた例。

    - 順次: 初期化、結果の収集
    - 選択: 素数かどうかの判定
    - 反復: 篩の処理、倍数の除去
    """
    # 順次: 初期化
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False

    # 反復（外側）: 2からsqrt(limit)まで
    i = 2
    while i * i <= limit:
        # 選択: iが素数として残っている場合のみ処理
        if is_prime[i]:
            # 反復（内側）: iの倍数を除去
            j = i * i
            while j <= limit:
                is_prime[j] = False  # 代入: 状態の更新
                j += i
        i += 1

    # 順次: 結果の収集
    primes = []
    for num in range(2, limit + 1):
        if is_prime[num]:            # 選択
            primes.append(num)

    return primes

# 実行例
result = find_primes_sieve(50)
print(f"50以下の素数: {result}")
# 50以下の素数: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
print(f"個数: {len(result)}")
# 個数: 15
```

---

## 4. 状態管理と副作用

### 4.1 状態（State）とは

命令型プログラミングにおける「状態」とは、プログラムの実行中にメモリ上に保持される全ての変数の値の集合を指す。プログラムの各時点における状態は、それまでに実行された全ての代入文の結果として決まる。

```
状態の概念図
════════════════════════════════════════════

  プログラム:              状態の遷移:
  ─────────               ──────────
  x = 10                  S0: { }
      ↓                        ↓
  y = 20                  S1: { x=10 }
      ↓                        ↓
  z = x + y               S2: { x=10, y=20 }
      ↓                        ↓
  x = z * 2               S3: { x=10, y=20, z=30 }
      ↓                        ↓
  (終了)                   S4: { x=60, y=20, z=30 }

  ポイント:
  - 各文の実行は「状態→状態」の変換（状態遷移）
  - 同じ変数 x が異なる時点で異なる値を持つ
  - プログラムの「意味」は状態遷移の列として理解できる
```

### 4.2 副作用（Side Effect）

副作用とは、関数やサブルーチンの実行が、返り値を返す以外に外部の状態を変更することを指す。具体的には以下のような操作が副作用に該当する。

1. グローバル変数の変更
2. 引数として渡されたオブジェクトの変更（破壊的操作）
3. ファイルへの書き込み
4. 画面への出力
5. ネットワーク通信
6. データベースの更新

```python
# === 副作用のある関数 ===

# 例1: グローバル変数の変更
counter = 0

def increment():
    """呼び出すたびにグローバル変数 counter を変更する（副作用）"""
    global counter
    counter += 1
    return counter

print(increment())  # 1
print(increment())  # 2
print(increment())  # 3
# → 同じ引数（なし）で呼んでも、毎回異なる結果を返す


# 例2: 引数の破壊的変更
def remove_negatives(numbers):
    """元のリストを直接変更する（副作用）"""
    i = 0
    while i < len(numbers):
        if numbers[i] < 0:
            numbers.pop(i)     # 元のリストを変更
        else:
            i += 1
    return numbers

data = [3, -1, 4, -1, 5, -9, 2, 6]
result = remove_negatives(data)
print(result)   # [3, 4, 5, 2, 6]
print(data)     # [3, 4, 5, 2, 6] ← 元のリストも変わっている！


# === 副作用のない関数（純粋関数） ===

def remove_negatives_pure(numbers):
    """新しいリストを作成して返す（副作用なし）"""
    result = []
    for num in numbers:
        if num >= 0:
            result.append(num)
    return result

data = [3, -1, 4, -1, 5, -9, 2, 6]
result = remove_negatives_pure(data)
print(result)   # [3, 4, 5, 2, 6]
print(data)     # [3, -1, 4, -1, 5, -9, 2, 6] ← 元のリストは不変
```

### 4.3 参照透過性（Referential Transparency）

参照透過性は、式を評価した結果の値で式そのものを置き換えても、プログラムの振る舞いが変わらない性質を指す。副作用のない純粋関数は参照透過性を持つ。

```python
# 参照透過な関数
def add(a, b):
    return a + b

# add(3, 4) はいつでも 7 に置き換えられる
x = add(3, 4) + add(3, 4)   # x = 7 + 7 = 14
y = 7 + 7                    # y = 14
# x と y は常に等しい → 参照透過

# 参照透過でない関数
import random

def random_add(a, b):
    return a + b + random.randint(0, 10)

# random_add(3, 4) の結果は呼び出すたびに異なる
# → 値で置き換えると意味が変わる → 参照透過でない
```

### 4.4 可変状態がもたらす問題

命令型プログラミングの根幹である「可変状態」は、プログラムの複雑さの主要な源泉でもある。

```python
# === 可変状態の問題：エイリアシング ===

def process_orders(orders, vip_orders):
    """
    注文リストを処理する。
    問題: orders と vip_orders が同じオブジェクトを参照していた場合、
    意図しない副作用が発生する。
    """
    # 一般注文に送料を加算
    for order in orders:
        order["total"] += 500

    # VIP注文は送料無料
    for order in vip_orders:
        order["total"] -= 500    # 送料分を減算

    return orders, vip_orders

# 問題のあるケース: 同じ辞書オブジェクトが両方のリストに存在
shared_order = {"item": "Book", "total": 3000}
orders = [shared_order]
vip_orders = [shared_order]  # 同じオブジェクトへの参照！

process_orders(orders, vip_orders)
print(shared_order)
# {"item": "Book", "total": 3000}
# → +500して-500されたので元に戻るが、
#   これは偶然であり、処理順序が変わると結果も変わる

# === 解決策: 防御的コピー ===

import copy

def process_orders_safe(orders, vip_orders):
    """防御的コピーにより、元のデータを保護する"""
    orders_copy = copy.deepcopy(orders)
    vip_copy = copy.deepcopy(vip_orders)

    for order in orders_copy:
        order["total"] += 500

    for order in vip_copy:
        order["total"] -= 500

    return orders_copy, vip_copy
```

### 4.5 メモリモデルと変数のライフサイクル

命令型プログラミングでは、変数はメモリ上の領域に対応する。変数のライフサイクル（生成・使用・解放）を正しく管理することが重要である。

```
メモリレイアウト（典型的なCプログラム）
═══════════════════════════════════════════════

  高アドレス
  ┌──────────────────────┐
  │     コマンドライン引数    │
  │     環境変数            │
  ├──────────────────────┤
  │                      │
  │     スタック領域  ↓     │ ← ローカル変数、関数引数、戻りアドレス
  │                      │    関数呼び出しのたびに伸長（LIFO）
  │     (...空き領域...)   │
  │                      │
  │     ヒープ領域   ↑     │ ← malloc/freeで動的に管理
  │                      │    プログラマの責任で解放
  ├──────────────────────┤
  │     BSS セグメント     │ ← 未初期化のグローバル変数（0で初期化）
  ├──────────────────────┤
  │     データセグメント    │ ← 初期化済みグローバル変数、静的変数
  ├──────────────────────┤
  │     テキストセグメント  │ ← プログラムのコード（読み取り専用）
  └──────────────────────┘
  低アドレス

  変数の種類と格納場所:
  ┌──────────────┬─────────────┬─────────────────────┐
  │ 変数の種類    │ 格納場所     │ ライフサイクル        │
  ├──────────────┼─────────────┼─────────────────────┤
  │ ローカル変数  │ スタック     │ 関数の実行中のみ      │
  │ グローバル変数 │ データ/BSS  │ プログラム全体        │
  │ 動的確保     │ ヒープ      │ malloc〜freeまで      │
  │ 静的変数     │ データ      │ プログラム全体        │
  └──────────────┴─────────────┴─────────────────────┘
```

```c
#include <stdio.h>
#include <stdlib.h>

int global_var = 100;           /* データセグメント */
int uninitialized_global;       /* BSSセグメント（0に初期化） */

void demonstrate_memory(void) {
    int local_var = 42;                       /* スタック */
    static int static_var = 0;                /* データセグメント */
    int *heap_var = malloc(sizeof(int));       /* ヒープ */

    static_var++;  /* 関数呼び出しをまたいで値が保持される */
    *heap_var = static_var * 10;

    printf("local:  %d (addr: %p)\n", local_var, (void *)&local_var);
    printf("static: %d (addr: %p)\n", static_var, (void *)&static_var);
    printf("heap:   %d (addr: %p)\n", *heap_var, (void *)heap_var);
    printf("global: %d (addr: %p)\n", global_var, (void *)&global_var);

    free(heap_var);  /* ヒープ領域の解放 */
}

int main(void) {
    printf("=== 1回目の呼び出し ===\n");
    demonstrate_memory();
    printf("\n=== 2回目の呼び出し ===\n");
    demonstrate_memory();
    return 0;
}
/*
 * 出力例:
 * === 1回目の呼び出し ===
 * local:  42 (addr: 0x7ffc...)
 * static: 1  (addr: 0x601...)
 * heap:   10 (addr: 0x1a3...)
 * global: 100 (addr: 0x601...)
 *
 * === 2回目の呼び出し ===
 * local:  42 (addr: 0x7ffc...)    ← localは毎回初期化される
 * static: 2  (addr: 0x601...)     ← staticは値が保持される
 * heap:   20 (addr: 0x1a3...)
 * global: 100 (addr: 0x601...)
 */
```

---

## 5. 構造化プログラミング

### 5.1 goto 有害論の背景

1960 年代後半、ソフトウェアの規模と複雑さが急速に増大し、「ソフトウェア危機（Software Crisis）」と呼ばれる状況が生じた。プロジェクトの遅延、予算超過、バグの多発が常態化していた。

この危機の原因の一つとして、当時のプログラムで多用されていた goto 文による非構造的な制御フローが指摘された。goto 文を無秩序に使用したプログラムは、制御の流れが絡み合い、理解・保守が極めて困難なコード、いわゆる「スパゲッティコード」を生み出していた。

```
goto 文によるスパゲッティコードの例（擬似コード）
═══════════════════════════════════════════════════

  10: READ X
  20: IF X < 0 GOTO 80
  30: Y = X * 2
  40: IF Y > 100 GOTO 70
  50: PRINT Y
  60: GOTO 10
  70: PRINT "TOO LARGE"
  75: GOTO 10
  80: IF X = -1 GOTO 100
  90: PRINT "NEGATIVE"
  95: GOTO 10
  100: END

  制御フローの図:
  10 → 20 → 30 → 40 → 50 → 60 → 10（ループ）
         ↓              ↓
        80 → 90 → 95   70 → 75
         ↓    ↑           ↑
        100   10          10

  → 行ったり来たりで制御の流れが複雑に絡み合う
  → コードの一部を変更すると、予期しない場所に影響が波及する
```

### 5.2 ダイクストラの提唱

1968 年、オランダの計算機科学者エドガー・ダイクストラ（Edsger W. Dijkstra）は、ACM（Association for Computing Machinery）の機関誌 Communications of the ACM に「Go To Statement Considered Harmful」と題する書簡を発表した。この短い論文（実際には編集者への手紙）は、プログラミングの歴史に大きな影響を与えた。

ダイクストラの主張の核心は以下の点にある。

1. **goto 文の過度な使用はプログラムの品質を低下させる**：goto 文が多用されると、プログラムの実行状態を追跡することが困難になり、プログラムの正しさを推論することが極めて難しくなる。

2. **構造化された制御フローが必要である**：順次・選択・反復の三つの制御構造のみを用いることで、プログラムの構造が明確になり、正しさの検証が容易になる。

3. **プログラムの静的構造と動的振る舞いの対応**：goto 文を排除することで、プログラムのソースコード（静的構造）から実行時の振る舞い（動的振る舞い）を容易に推論できるようになる。

### 5.3 構造化プログラミングの原則

ダイクストラの提唱に基づき、構造化プログラミングは以下の原則としてまとめられた。

```
構造化プログラミングの原則
═════════════════════════════════════════

  1. 単一入口・単一出口（Single Entry, Single Exit）
     - 各制御構造ブロックは一つの入口と一つの出口を持つ
     - 途中からの飛び込みや飛び出しを禁止

  2. 三つの基本制御構造のみを使用
     - 順次（Sequence）
     - 選択（Selection）: if-then-else
     - 反復（Iteration）: while-do

  3. トップダウン設計
     - 問題を大きな単位から小さな単位へ分解
     - 各レベルで詳細を隠蔽（段階的詳細化）

  4. 段階的詳細化（Stepwise Refinement）
     - Niklaus Wirth（1971）が体系化
     - 抽象的な記述から具体的な実装へ段階的に展開
```

### 5.4 goto 文のある BASIC コードと構造化コードの比較

**BASIC（goto 文を使用）:**

```basic
10 REM フィボナッチ数列（goto使用版）
20 LET A = 0
30 LET B = 1
40 LET N = 0
50 PRINT A
60 LET N = N + 1
70 IF N >= 10 THEN GOTO 120
80 LET T = A + B
90 LET A = B
100 LET B = T
110 GOTO 50
120 END
```

**Python（構造化プログラミング）:**

```python
def fibonacci_sequence(count):
    """
    構造化されたフィボナッチ数列の生成。
    - goto文なし、while ループで反復を表現
    - 関数として切り出されており、再利用可能
    - 変数のスコープが関数内に限定されている
    """
    a, b = 0, 1
    result = []
    n = 0

    while n < count:       # 反復: 指定回数まで繰り返す
        result.append(a)   # 順次: 結果リストに追加
        a, b = b, a + b    # 順次: 次の項を計算
        n += 1             # 順次: カウンタを更新

    return result

# 実行例
fib = fibonacci_sequence(10)
print(fib)
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

### 5.5 構造化プログラミングの影響と限界

構造化プログラミングは、ソフトウェア開発に革命的な変化をもたらした。

**影響:**

1. **可読性の飛躍的向上**: プログラムの制御フローが予測可能になった
2. **保守性の改善**: コードの一部を変更しても他の部分への影響が限定的になった
3. **検証可能性**: プログラムの正しさを論理的に推論しやすくなった
4. **言語設計への影響**: Pascal（1970年）は構造化プログラミングを体現する教育用言語として設計された

**限界:**

構造化プログラミングは goto 文の排除と制御構造の整理には成功したが、データの組織化については十分な指針を提供しなかった。プログラムが大規模になると、データ構造とそれに対する操作の管理が新たな課題となった。この課題に対する解答の一つがオブジェクト指向プログラミングである。

また、厳密な「単一出口」の原則は、現代の実務では緩和されている。例外処理（try-catch）、早期リターン（guard clause）、break 文などは goto 文ではないが厳密には単一出口の原則に反する。しかし、これらは適切に使用すれば可読性を向上させることが広く認められている。

```python
# 早期リターン（ガード節）の例
# 厳密な単一出口には反するが、可読性が高い

def calculate_discount(customer, amount):
    """ガード節による早期リターンの例"""
    # ガード節: 異常ケースを先に処理して早期に返す
    if customer is None:
        return 0
    if amount <= 0:
        return 0
    if not customer.get("is_active", False):
        return 0

    # 正常ケースの処理（ネストが浅くなる）
    base_rate = 0.05
    if customer.get("is_vip", False):
        base_rate = 0.15
    elif customer.get("years", 0) >= 3:
        base_rate = 0.10

    return amount * base_rate
```

---

## 6. 手続き型プログラミング

### 6.1 手続き型の基本概念

手続き型プログラミング（Procedural Programming）は、命令型プログラミングのサブパラダイムであり、プログラムを「手続き（プロシージャ）」あるいは「関数」の集まりとして組織化する手法である。

命令型プログラミングが「文の列」としてプログラムを構成するのに対し、手続き型プログラミングは関連する文のまとまりに名前を付け、再利用可能な単位（手続き）として切り出す。

```
命令型プログラミングと手続き型プログラミングの関係
═══════════════════════════════════════════════════

  ┌──────────────────────────────────────────────┐
  │         命令型プログラミング                     │
  │  （状態 + 代入 + 制御フロー）                    │
  │                                              │
  │  ┌──────────────────────────────────────┐    │
  │  │   構造化プログラミング                   │    │
  │  │   （goto排除、3制御構造）                │    │
  │  │                                      │    │
  │  │  ┌──────────────────────────────┐   │    │
  │  │  │  手続き型プログラミング         │   │    │
  │  │  │  （関数/手続きによるモジュール化）│   │    │
  │  │  └──────────────────────────────┘   │    │
  │  │                                      │    │
  │  │  ┌──────────────────────────────┐   │    │
  │  │  │  オブジェクト指向プログラミング  │   │    │
  │  │  │  （データ+操作のカプセル化）     │   │    │
  │  │  └──────────────────────────────┘   │    │
  │  └──────────────────────────────────────┘    │
  └──────────────────────────────────────────────┘

  ポイント: 手続き型とOOPは、どちらも構造化の一形態
```

### 6.2 関数（手続き）の設計原則

良い手続き（関数）を設計するための原則を以下に示す。

```
関数設計の五原則
═════════════════════════════════════════

  1. 単一責任の原則（Single Responsibility）
     - 一つの関数は一つのことだけを行う
     - 「何をする関数か」を一文で説明できること

  2. 適切な抽象化レベル
     - 一つの関数内の処理は同じ抽象度であること
     - 高水準の処理と低水準の詳細を混在させない

  3. 明示的な入出力
     - 引数から入力を受け、返り値で結果を返す
     - グローバル変数への依存を最小化する

  4. 適切な粒度
     - 長すぎず短すぎない（目安: 20〜30行以内）
     - 画面に収まる程度が望ましい

  5. 意味のある命名
     - 動詞+名詞の形式が基本（例: calculate_tax）
     - 処理内容が名前から推測できること
```

### 6.3 手続き型プログラミングの実践例

以下に、手続き型プログラミングの原則に従った完全な実装例を示す。学生の成績管理システムを段階的に設計する。

```python
"""
学生成績管理システム（手続き型プログラミングの実践例）

設計方針:
- 関数によるモジュール化（各関数は単一責任）
- データは辞書のリストとして管理
- グローバル変数は使用しない
- 各関数は入力（引数）と出力（返り値）が明確
"""


# ===== データ操作関数 =====

def create_student(name, student_id):
    """学生データを作成する"""
    return {
        "name": name,
        "id": student_id,
        "scores": {}
    }


def add_score(student, subject, score):
    """学生に科目の成績を追加する（新しい辞書を返す）"""
    new_scores = dict(student["scores"])
    new_scores[subject] = score
    return {
        "name": student["name"],
        "id": student["id"],
        "scores": new_scores
    }


def calculate_average(student):
    """学生の平均点を計算する"""
    scores = student["scores"]
    if not scores:
        return 0.0
    total = 0
    count = 0
    for score in scores.values():
        total += score
        count += 1
    return total / count


def determine_grade(average):
    """平均点から評定を判定する"""
    if average >= 90:
        return "A"
    elif average >= 80:
        return "B"
    elif average >= 70:
        return "C"
    elif average >= 60:
        return "D"
    else:
        return "F"


# ===== 集計関数 =====

def find_top_students(students, n=3):
    """上位n名の学生を取得する"""
    averages = []
    for student in students:
        avg = calculate_average(student)
        averages.append((student, avg))

    # 平均点で降順ソート（選択ソートで実装）
    for i in range(len(averages)):
        max_idx = i
        for j in range(i + 1, len(averages)):
            if averages[j][1] > averages[max_idx][1]:
                max_idx = j
        averages[i], averages[max_idx] = averages[max_idx], averages[i]

    result = []
    for i in range(min(n, len(averages))):
        result.append(averages[i])
    return result


def calculate_subject_stats(students, subject):
    """特定科目の統計情報（平均、最高、最低）を計算する"""
    scores = []
    for student in students:
        if subject in student["scores"]:
            scores.append(student["scores"][subject])

    if not scores:
        return {"subject": subject, "count": 0, "avg": 0, "max": 0, "min": 0}

    total = 0
    max_score = scores[0]
    min_score = scores[0]
    for score in scores:
        total += score
        if score > max_score:
            max_score = score
        if score < min_score:
            min_score = score

    return {
        "subject": subject,
        "count": len(scores),
        "avg": total / len(scores),
        "max": max_score,
        "min": min_score,
    }


# ===== 表示関数 =====

def format_student_report(student):
    """学生の成績レポートを文字列として整形する"""
    avg = calculate_average(student)
    grade = determine_grade(avg)

    lines = []
    lines.append(f"学生名: {student['name']} (ID: {student['id']})")
    lines.append("-" * 40)

    for subject, score in student["scores"].items():
        lines.append(f"  {subject:12s}: {score:5.1f}")

    lines.append("-" * 40)
    lines.append(f"  {'平均点':12s}: {avg:5.1f}")
    lines.append(f"  {'評定':12s}:     {grade}")

    return "\n".join(lines)


def format_ranking(top_students):
    """ランキングを文字列として整形する"""
    lines = []
    lines.append("=== 成績上位者 ===")
    for rank, (student, avg) in enumerate(top_students, 1):
        grade = determine_grade(avg)
        lines.append(f"  {rank}位: {student['name']} "
                      f"(平均: {avg:.1f}, 評定: {grade})")
    return "\n".join(lines)


# ===== メイン処理 =====

def main():
    """メイン処理: データの生成、処理、表示を順に実行する"""

    # データの準備
    students = [
        create_student("佐藤太郎", "S001"),
        create_student("鈴木花子", "S002"),
        create_student("田中一郎", "S003"),
        create_student("高橋美咲", "S004"),
        create_student("伊藤健太", "S005"),
    ]

    # 成績データの設定
    score_data = [
        ("S001", [("数学", 85), ("英語", 72), ("物理", 90)]),
        ("S002", [("数学", 92), ("英語", 88), ("物理", 76)]),
        ("S003", [("数学", 68), ("英語", 55), ("物理", 73)]),
        ("S004", [("数学", 95), ("英語", 91), ("物理", 88)]),
        ("S005", [("数学", 78), ("英語", 82), ("物理", 65)]),
    ]

    # 成績の登録
    for sid, scores in score_data:
        for i in range(len(students)):
            if students[i]["id"] == sid:
                for subject, score in scores:
                    students[i] = add_score(students[i], subject, score)

    # 個別レポートの表示
    print("=" * 40)
    print("      個別成績レポート")
    print("=" * 40)
    for student in students:
        print()
        print(format_student_report(student))

    # ランキングの表示
    print()
    top = find_top_students(students, 3)
    print(format_ranking(top))

    # 科目別統計の表示
    print()
    print("=== 科目別統計 ===")
    for subject in ["数学", "英語", "物理"]:
        stats = calculate_subject_stats(students, subject)
        print(f"  {stats['subject']}: "
              f"平均={stats['avg']:.1f}, "
              f"最高={stats['max']:.0f}, "
              f"最低={stats['min']:.0f}")


# エントリポイント
if __name__ == "__main__":
    main()
```

実行結果:

```
========================================
      個別成績レポート
========================================

学生名: 佐藤太郎 (ID: S001)
----------------------------------------
  数学          :  85.0
  英語          :  72.0
  物理          :  90.0
----------------------------------------
  平均点         :  82.3
  評定          :     B

学生名: 鈴木花子 (ID: S002)
----------------------------------------
  数学          :  92.0
  英語          :  88.0
  物理          :  76.0
----------------------------------------
  平均点         :  85.3
  評定          :     B

...（以下略）

=== 成績上位者 ===
  1位: 高橋美咲 (平均: 91.3, 評定: A)
  2位: 鈴木花子 (平均: 85.3, 評定: B)
  3位: 佐藤太郎 (平均: 82.3, 評定: B)

=== 科目別統計 ===
  数学: 平均=83.6, 最高=95, 最低=68
  英語: 平均=77.6, 最高=91, 最低=55
  物理: 平均=78.4, 最高=90, 最低=65
```

### 6.4 モジュール化と情報隠蔽

手続き型プログラミングでは、関連する関数とデータをモジュール（ファイル）にまとめることで、プログラムの規模に対応する。C 言語では、ヘッダファイル（.h）によるインターフェースの公開と、ソースファイル（.c）による実装の隠蔽が基本的なモジュール化手法である。

```c
/* === stack.h（インターフェース: 公開される宣言） === */
#ifndef STACK_H
#define STACK_H

#define STACK_MAX_SIZE 100

typedef struct {
    int data[STACK_MAX_SIZE];
    int top;
} Stack;

/* 公開関数の宣言（利用者はこれだけ知ればよい） */
void stack_init(Stack *s);
int  stack_push(Stack *s, int value);
int  stack_pop(Stack *s, int *value);
int  stack_peek(const Stack *s, int *value);
int  stack_is_empty(const Stack *s);
int  stack_size(const Stack *s);

#endif /* STACK_H */
```

```c
/* === stack.c（実装: 内部の詳細） === */
#include "stack.h"
#include <stdio.h>

void stack_init(Stack *s) {
    s->top = -1;
}

int stack_push(Stack *s, int value) {
    if (s->top >= STACK_MAX_SIZE - 1) {
        fprintf(stderr, "Stack overflow\n");
        return -1;
    }
    s->top++;
    s->data[s->top] = value;
    return 0;
}

int stack_pop(Stack *s, int *value) {
    if (s->top < 0) {
        fprintf(stderr, "Stack underflow\n");
        return -1;
    }
    *value = s->data[s->top];
    s->top--;
    return 0;
}

int stack_peek(const Stack *s, int *value) {
    if (s->top < 0) {
        return -1;
    }
    *value = s->data[s->top];
    return 0;
}

int stack_is_empty(const Stack *s) {
    return s->top < 0;
}

int stack_size(const Stack *s) {
    return s->top + 1;
}
```

```c
/* === main.c（利用コード） === */
#include <stdio.h>
#include "stack.h"

int main(void) {
    Stack s;
    stack_init(&s);

    /* 値をプッシュ */
    stack_push(&s, 10);
    stack_push(&s, 20);
    stack_push(&s, 30);

    printf("スタックサイズ: %d\n", stack_size(&s));  /* 3 */

    /* 値をポップ */
    int value;
    while (!stack_is_empty(&s)) {
        stack_pop(&s, &value);
        printf("ポップ: %d\n", value);
    }
    /* ポップ: 30, ポップ: 20, ポップ: 10 */

    return 0;
}
```

### 6.5 手続き型プログラミングの長所と短所

| 観点 | 長所 | 短所 |
|------|------|------|
| 学習コスト | 直感的で初学者が取り組みやすい | 大規模になると設計知識が必要 |
| 実行効率 | ハードウェアに近く高速 | 最適化は開発者の責任 |
| コード構造 | 関数による明確なモジュール化 | データと操作が分離しがち |
| 再利用性 | 関数単位での再利用が容易 | データ構造と関数の組み合わせが固定的 |
| 保守性 | 小〜中規模では高い | 大規模ではデータの整合性維持が困難 |
| テスト | 関数単位のテストが容易 | 副作用のある関数のテストは難しい |
| 並行性 | 逐次処理の表現が自然 | 共有状態の並行アクセスが問題になる |

