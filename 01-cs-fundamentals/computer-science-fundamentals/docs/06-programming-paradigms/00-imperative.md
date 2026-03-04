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

---

## 7. 命令型と他のパラダイムの比較

### 7.1 命令型 vs 関数型プログラミング

命令型プログラミングと関数型プログラミング（Functional Programming）は、計算に対する根本的に異なるアプローチを取る。

```
命令型 vs 関数型の本質的な違い
═══════════════════════════════════════════════

  命令型: 「状態を変化させる命令の列」
  ─────────────────────────────────────
  プログラム = 状態機械（State Machine）

  状態S0 → [命令1] → 状態S1 → [命令2] → 状態S2 → ...

  → 「時間の経過」に沿ってプログラムが進む
  → 同じ命令でも、実行時の状態によって結果が異なる

  関数型: 「関数の合成による変換」
  ─────────────────────────────────────
  プログラム = 関数の合成（Function Composition）

  入力 → f → g → h → 出力

  → 「データの変換」としてプログラムを記述
  → 同じ入力に対して常に同じ出力（参照透過性）
```

具体的なコードで比較する。リスト内の偶数を二乗して合計する処理を、命令型と関数型の両方で実装する。

```python
# === 命令型スタイル ===

def sum_of_even_squares_imperative(numbers):
    """
    命令型: 状態変数 total を逐次更新する
    """
    total = 0                         # 状態の初期化
    for num in numbers:               # 反復
        if num % 2 == 0:              # 選択
            total += num ** 2         # 状態の更新（代入）
    return total


# === 関数型スタイル ===

def sum_of_even_squares_functional(numbers):
    """
    関数型: 関数の合成でデータを変換する
    中間状態を持たない
    """
    from functools import reduce
    return reduce(
        lambda acc, x: acc + x,       # 集約関数
        map(
            lambda x: x ** 2,         # 変換関数
            filter(
                lambda x: x % 2 == 0, # フィルタ関数
                numbers
            )
        ),
        0                             # 初期値
    )


# === Python的な関数型スタイル（リスト内包表記） ===

def sum_of_even_squares_pythonic(numbers):
    """
    Pythonらしい宣言的スタイル
    """
    return sum(x ** 2 for x in numbers if x % 2 == 0)


# 実行例（いずれも同じ結果）
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(sum_of_even_squares_imperative(data))    # 220
print(sum_of_even_squares_functional(data))    # 220
print(sum_of_even_squares_pythonic(data))      # 220

# 2^2 + 4^2 + 6^2 + 8^2 + 10^2 = 4 + 16 + 36 + 64 + 100 = 220
```

| 比較項目 | 命令型 | 関数型 |
|---------|--------|--------|
| 中心概念 | 状態の変化 | 値の変換 |
| 変数 | 可変（mutable） | 不変（immutable）が基本 |
| ループ | for / while | 再帰 / 高階関数（map, filter, reduce） |
| 副作用 | 一般的 | 避ける（純粋関数を志向） |
| 並行処理 | ロック等による同期が必要 | 共有状態がないため安全 |
| デバッグ | ステップ実行で追跡 | 各関数の入出力を検証 |
| 計算モデル | チューリングマシン | ラムダ計算 |
| 代表言語 | C, Pascal, Go | Haskell, Erlang, Clojure |

### 7.2 命令型 vs オブジェクト指向プログラミング

オブジェクト指向プログラミング（Object-Oriented Programming, OOP）は、命令型プログラミングの拡張として位置づけられることが多い。OOP はデータ（状態）とそれに対する操作（メソッド）をオブジェクトとしてまとめ、カプセル化する。

**手続き型アプローチ（データと操作が分離）:**

```python
# 手続き型: データは辞書、操作は独立した関数

def create_bank_account(owner, balance=0):
    """口座データを作成する"""
    return {"owner": owner, "balance": balance, "history": []}

def deposit(account, amount):
    """入金する"""
    if amount <= 0:
        print("入金額は正の数でなければならない")
        return account
    new_balance = account["balance"] + amount
    new_history = list(account["history"])
    new_history.append(f"入金: +{amount}")
    return {
        "owner": account["owner"],
        "balance": new_balance,
        "history": new_history,
    }

def withdraw(account, amount):
    """出金する"""
    if amount <= 0:
        print("出金額は正の数でなければならない")
        return account
    if amount > account["balance"]:
        print("残高不足")
        return account
    new_balance = account["balance"] - amount
    new_history = list(account["history"])
    new_history.append(f"出金: -{amount}")
    return {
        "owner": account["owner"],
        "balance": new_balance,
        "history": new_history,
    }

def get_balance_info(account):
    """残高情報を取得する"""
    return f"{account['owner']}の残高: {account['balance']}円"

# 利用例
acc = create_bank_account("佐藤太郎", 10000)
acc = deposit(acc, 5000)
acc = withdraw(acc, 3000)
print(get_balance_info(acc))  # 佐藤太郎の残高: 12000円
```

**オブジェクト指向アプローチ（データと操作を統合）:**

```python
class BankAccount:
    """
    OOP: データ（残高、履歴）と操作（入出金）を
    一つのクラスにカプセル化する
    """

    def __init__(self, owner, balance=0):
        self._owner = owner         # プライベート属性
        self._balance = balance
        self._history = []

    def deposit(self, amount):
        """入金する"""
        if amount <= 0:
            raise ValueError("入金額は正の数でなければならない")
        self._balance += amount
        self._history.append(f"入金: +{amount}")

    def withdraw(self, amount):
        """出金する"""
        if amount <= 0:
            raise ValueError("出金額は正の数でなければならない")
        if amount > self._balance:
            raise ValueError("残高不足")
        self._balance -= amount
        self._history.append(f"出金: -{amount}")

    @property
    def balance(self):
        return self._balance

    def __str__(self):
        return f"{self._owner}の残高: {self._balance}円"


# 利用例
acc = BankAccount("佐藤太郎", 10000)
acc.deposit(5000)
acc.withdraw(3000)
print(acc)  # 佐藤太郎の残高: 12000円
```

| 比較項目 | 手続き型 | オブジェクト指向 |
|---------|---------|--------------|
| データと操作 | 分離している | 統合（カプセル化）されている |
| 状態管理 | 関数外から直接アクセス可能 | アクセス制御により保護される |
| 再利用 | 関数の再利用 | クラスの継承・委譲による再利用 |
| 多態性 | 関数ポインタ等で実現 | 言語レベルでサポート |
| 設計の単位 | 関数 | オブジェクト（クラス） |
| 適する規模 | 小〜中規模 | 中〜大規模 |
| 複雑性管理 | 関数の階層化 | オブジェクト間の関係設計 |

### 7.3 パラダイム選択の指針

```
パラダイム選択のデシジョンツリー
═══════════════════════════════════════════════════

  問題の性質を分析
      │
      ├── ハードウェア制御・組み込み → 命令型（C）
      │
      ├── スクリプト・自動化 → 手続き型（Python, Shell）
      │
      ├── データ変換・並行処理 → 関数型（Haskell, Elixir）
      │
      ├── 複雑なドメインモデル → OOP（Java, C#）
      │
      ├── データ問い合わせ → 宣言型（SQL）
      │
      ├── 論理推論・制約充足 → 論理型（Prolog）
      │
      └── 複合的な要件 → マルチパラダイム（Python, Rust, TS）

  現実のプロジェクトでは:
  - 一つのパラダイムに固執するのではなく、
    問題の各部分に最適なパラダイムを適用する
  - ほとんどの現代言語はマルチパラダイムをサポートする
```

---

## 8. 現代言語での命令型スタイル

### 8.1 Python における命令型と宣言型の融合

Python はマルチパラダイム言語の代表格であり、命令型スタイルと関数型・宣言型スタイルを自然に組み合わせることができる。

```python
"""
現代的なPythonにおけるパラダイムの融合例:
ログファイルを解析してエラー統計を生成する
"""
from collections import defaultdict
from datetime import datetime


# === 命令型スタイル ===

def analyze_logs_imperative(log_lines):
    """命令型: 明示的なループと状態管理"""
    error_counts = {}
    error_times = []

    for line in log_lines:
        # 各行を解析
        parts = line.strip().split(" ", 3)
        if len(parts) < 4:
            continue

        timestamp_str = parts[0] + " " + parts[1]
        level = parts[2].strip("[]")
        message = parts[3]

        if level == "ERROR":
            # エラーカウントの更新
            if message in error_counts:
                error_counts[message] += 1
            else:
                error_counts[message] = 1

            # タイムスタンプの記録
            try:
                ts = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                error_times.append(ts)
            except ValueError:
                pass

    # 最も多いエラーを見つける
    most_common_error = None
    max_count = 0
    for error, count in error_counts.items():
        if count > max_count:
            max_count = count
            most_common_error = error

    return {
        "total_errors": len(error_times),
        "unique_errors": len(error_counts),
        "most_common": most_common_error,
        "most_common_count": max_count,
        "error_counts": error_counts,
    }


# === 宣言型/関数型スタイル ===

def analyze_logs_declarative(log_lines):
    """宣言型: 高階関数とジェネレータの活用"""

    def parse_line(line):
        """行をパースして構造化データに変換"""
        parts = line.strip().split(" ", 3)
        if len(parts) < 4:
            return None
        return {
            "timestamp": parts[0] + " " + parts[1],
            "level": parts[2].strip("[]"),
            "message": parts[3],
        }

    # データ変換パイプライン
    parsed = (parse_line(line) for line in log_lines)           # 遅延評価
    valid = (entry for entry in parsed if entry is not None)    # フィルタ
    errors = [entry for entry in valid if entry["level"] == "ERROR"]

    # 集計（defaultdictで簡潔に）
    error_counts = defaultdict(int)
    for error in errors:
        error_counts[error["message"]] += 1

    most_common = max(error_counts.items(), key=lambda x: x[1],
                      default=(None, 0))

    return {
        "total_errors": len(errors),
        "unique_errors": len(error_counts),
        "most_common": most_common[0],
        "most_common_count": most_common[1],
        "error_counts": dict(error_counts),
    }


# テスト用データ
sample_logs = [
    "2024-01-15 10:23:45 [INFO] Application started",
    "2024-01-15 10:24:01 [ERROR] Database connection failed",
    "2024-01-15 10:24:15 [ERROR] Database connection failed",
    "2024-01-15 10:25:30 [WARNING] High memory usage",
    "2024-01-15 10:26:00 [ERROR] File not found: config.yml",
    "2024-01-15 10:27:12 [INFO] Retry successful",
    "2024-01-15 10:28:45 [ERROR] Database connection failed",
    "2024-01-15 10:30:00 [ERROR] Timeout waiting for response",
]

# 両方のアプローチで同じ結果が得られる
result1 = analyze_logs_imperative(sample_logs)
result2 = analyze_logs_declarative(sample_logs)

print(f"総エラー数: {result1['total_errors']}")        # 5
print(f"一意のエラー数: {result1['unique_errors']}")     # 3
print(f"最多エラー: {result1['most_common']}")           # Database connection failed
print(f"最多エラー回数: {result1['most_common_count']}")  # 3
```

### 8.2 Rust における命令型と関数型の融合

Rust は、命令型の制御フローと関数型のイテレータ・パターンマッチを高いレベルで融合した言語である。所有権システムにより、命令型プログラミングの最大の問題点である「可変状態の安全性」をコンパイル時に保証する。

```rust
// Rust における命令型スタイルと関数型スタイルの比較

/// 命令型スタイル: 明示的なループと可変変数
fn word_frequency_imperative(text: &str) -> Vec<(String, usize)> {
    use std::collections::HashMap;

    let mut counts: HashMap<String, usize> = HashMap::new();

    // 命令型ループ
    for word in text.split_whitespace() {
        let word_lower = word.to_lowercase();
        // パターンマッチで安全にカウントを更新
        let count = counts.entry(word_lower).or_insert(0);
        *count += 1;
    }

    // 結果をベクタに変換してソート
    let mut result: Vec<(String, usize)> = counts.into_iter().collect();
    result.sort_by(|a, b| b.1.cmp(&a.1));
    result
}

/// 関数型スタイル: イテレータチェーン
fn word_frequency_functional(text: &str) -> Vec<(String, usize)> {
    use std::collections::HashMap;

    let counts: HashMap<String, usize> = text
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .fold(HashMap::new(), |mut acc, word| {
            *acc.entry(word).or_insert(0) += 1;
            acc
        });

    let mut result: Vec<_> = counts.into_iter().collect();
    result.sort_by(|a, b| b.1.cmp(&a.1));
    result
}

fn main() {
    let text = "the quick brown fox jumps over the lazy dog the fox";

    let result = word_frequency_imperative(text);
    for (word, count) in &result {
        println!("{}: {}", word, count);
    }
    // the: 3
    // fox: 2
    // quick: 1
    // brown: 1
    // ...
}
```

### 8.3 Go における命令型の徹底

Go は意図的にシンプルな命令型スタイルを採用している言語である。ジェネリクス（Go 1.18 で追加）以前は、map, filter, reduce などの高階関数を言語レベルで提供しておらず、命令型ループを中心とした設計思想を貫いていた。

```go
package main

import (
    "fmt"
    "math"
    "sort"
    "strings"
)

// Point は二次元平面上の点を表す
type Point struct {
    X, Y float64
}

// Distance は2点間のユークリッド距離を計算する
func Distance(p1, p2 Point) float64 {
    dx := p1.X - p2.X
    dy := p1.Y - p2.Y
    return math.Sqrt(dx*dx + dy*dy)
}

// FindClosestPair は点の集合から最も近い2点の組を見つける
// 命令型スタイル: 明示的な二重ループ
func FindClosestPair(points []Point) (Point, Point, float64) {
    if len(points) < 2 {
        return Point{}, Point{}, -1
    }

    minDist := math.Inf(1)
    var closest1, closest2 Point

    // 命令型の二重ループで全ての組み合わせを検査
    for i := 0; i < len(points); i++ {
        for j := i + 1; j < len(points); j++ {
            dist := Distance(points[i], points[j])
            if dist < minDist {
                minDist = dist
                closest1 = points[i]
                closest2 = points[j]
            }
        }
    }

    return closest1, closest2, minDist
}

// GroupByQuadrant は点を象限ごとに分類する
func GroupByQuadrant(points []Point) map[string][]Point {
    groups := map[string][]Point{
        "第1象限": {},
        "第2象限": {},
        "第3象限": {},
        "第4象限": {},
        "軸上":    {},
    }

    for _, p := range points {
        switch {
        case p.X > 0 && p.Y > 0:
            groups["第1象限"] = append(groups["第1象限"], p)
        case p.X < 0 && p.Y > 0:
            groups["第2象限"] = append(groups["第2象限"], p)
        case p.X < 0 && p.Y < 0:
            groups["第3象限"] = append(groups["第3象限"], p)
        case p.X > 0 && p.Y < 0:
            groups["第4象限"] = append(groups["第4象限"], p)
        default:
            groups["軸上"] = append(groups["軸上"], p)
        }
    }

    return groups
}

func main() {
    points := []Point{
        {1.0, 2.0}, {3.0, 4.0}, {1.5, 2.5},
        {-1.0, 3.0}, {-2.0, -1.0}, {4.0, -2.0},
        {0.0, 5.0},
    }

    // 最近点対の探索
    p1, p2, dist := FindClosestPair(points)
    fmt.Printf("最近点対: (%.1f, %.1f) と (%.1f, %.1f)\n", p1.X, p1.Y, p2.X, p2.Y)
    fmt.Printf("距離: %.4f\n", dist)

    // 象限ごとの分類
    groups := GroupByQuadrant(points)
    for quadrant, pts := range groups {
        if len(pts) > 0 {
            strs := make([]string, len(pts))
            for i, p := range pts {
                strs[i] = fmt.Sprintf("(%.1f, %.1f)", p.X, p.Y)
            }
            fmt.Printf("%s: %s\n", quadrant, strings.Join(strs, ", "))
        }
    }

    // ソート（命令型: sort.Sliceで比較関数を指定）
    sort.Slice(points, func(i, j int) bool {
        d1 := Distance(points[i], Point{0, 0})
        d2 := Distance(points[j], Point{0, 0})
        return d1 < d2
    })

    fmt.Println("\n原点からの距離でソート:")
    for _, p := range points {
        d := Distance(p, Point{0, 0})
        fmt.Printf("  (%.1f, %.1f) -> 距離 %.4f\n", p.X, p.Y, d)
    }
}
```

### 8.4 Java における命令型とストリーム API の対比

Java 8 以降、ストリーム API の導入により、従来の命令型ループに代わる宣言型スタイルが利用可能になった。

```java
import java.util.*;
import java.util.stream.*;

public class ImperativeVsStreams {

    record Employee(String name, String department, int salary) {}

    /**
     * 命令型スタイル: 部門ごとの平均給与を計算する
     */
    static Map<String, Double> avgSalaryImperative(List<Employee> employees) {
        // 部門ごとの合計と人数を集計
        Map<String, Integer> totalByDept = new HashMap<>();
        Map<String, Integer> countByDept = new HashMap<>();

        for (Employee emp : employees) {
            String dept = emp.department();
            // 合計の更新
            if (totalByDept.containsKey(dept)) {
                totalByDept.put(dept, totalByDept.get(dept) + emp.salary());
            } else {
                totalByDept.put(dept, emp.salary());
            }
            // 人数の更新
            if (countByDept.containsKey(dept)) {
                countByDept.put(dept, countByDept.get(dept) + 1);
            } else {
                countByDept.put(dept, 1);
            }
        }

        // 平均の計算
        Map<String, Double> result = new HashMap<>();
        for (String dept : totalByDept.keySet()) {
            double avg = (double) totalByDept.get(dept) / countByDept.get(dept);
            result.put(dept, avg);
        }
        return result;
    }

    /**
     * ストリームAPI（宣言型スタイル）: 同じ処理をパイプラインで記述
     */
    static Map<String, Double> avgSalaryStreams(List<Employee> employees) {
        return employees.stream()
            .collect(Collectors.groupingBy(
                Employee::department,
                Collectors.averagingInt(Employee::salary)
            ));
    }

    public static void main(String[] args) {
        List<Employee> employees = List.of(
            new Employee("佐藤", "開発", 600000),
            new Employee("鈴木", "開発", 550000),
            new Employee("田中", "営業", 500000),
            new Employee("高橋", "営業", 480000),
            new Employee("伊藤", "人事", 520000),
            new Employee("渡辺", "開発", 700000),
            new Employee("山本", "営業", 530000)
        );

        // 命令型
        Map<String, Double> result1 = avgSalaryImperative(employees);
        System.out.println("命令型: " + result1);

        // ストリームAPI
        Map<String, Double> result2 = avgSalaryStreams(employees);
        System.out.println("ストリーム: " + result2);

        // どちらも同じ結果:
        // {開発=616666.67, 営業=503333.33, 人事=520000.0}
    }
}
```

---

## 9. アンチパターン

### 9.1 アンチパターン1: グローバル状態の乱用

グローバル変数に過度に依存したプログラムは、予測困難な振る舞いと保守の困難さをもたらす。

```python
# ===== アンチパターン: グローバル状態の乱用 =====

# グローバル変数で全ての状態を管理
current_user = None
cart_items = []
total_price = 0
discount_rate = 0
is_logged_in = False
error_message = ""

def login(username, password):
    global current_user, is_logged_in, error_message
    # 認証処理（省略）
    current_user = username
    is_logged_in = True
    error_message = ""

def add_to_cart(item, price):
    global cart_items, total_price, error_message
    if not is_logged_in:          # 別のグローバル変数に依存
        error_message = "ログインしてください"
        return
    cart_items.append({"item": item, "price": price})
    total_price += price          # グローバル変数を直接変更

def apply_discount(rate):
    global discount_rate, total_price
    discount_rate = rate
    total_price = total_price * (1 - rate)  # 適用済みかの判定がない

def checkout():
    global cart_items, total_price, error_message
    if not is_logged_in:
        error_message = "ログインしてください"
        return
    if total_price <= 0:
        error_message = "カートが空です"
        return
    # 注文処理...
    print(f"{current_user}の注文: {total_price}円")
    cart_items = []
    total_price = 0

# 問題点:
# 1. apply_discount() を2回呼ぶと割引が二重適用される
# 2. どの関数がどのグローバル変数を変更するか追跡が困難
# 3. テスト時にグローバル状態の初期化が必要
# 4. 並行実行すると状態が競合する


# ===== 改善版: 状態をオブジェクトに局所化 =====

class ShoppingSession:
    """状態をクラス内に閉じ込め、操作を制御する"""

    def __init__(self):
        self._user = None
        self._cart = []
        self._discount_applied = False

    def login(self, username, password):
        # 認証処理
        self._user = username

    @property
    def is_logged_in(self):
        return self._user is not None

    def add_to_cart(self, item, price):
        if not self.is_logged_in:
            raise RuntimeError("ログインしてください")
        self._cart.append({"item": item, "price": price})

    def apply_discount(self, rate):
        if self._discount_applied:
            raise RuntimeError("割引は一度のみ適用可能です")
        self._discount_applied = True
        self._discount_rate = rate

    @property
    def total(self):
        subtotal = sum(item["price"] for item in self._cart)
        if self._discount_applied:
            subtotal *= (1 - self._discount_rate)
        return subtotal

    def checkout(self):
        if not self.is_logged_in:
            raise RuntimeError("ログインしてください")
        if not self._cart:
            raise RuntimeError("カートが空です")
        order_total = self.total
        print(f"{self._user}の注文: {order_total}円")
        self._cart = []
        self._discount_applied = False
        return order_total
```

### 9.2 アンチパターン2: スパゲッティコードと深いネスト

複雑な条件分岐が深くネストすると、コードの可読性と保守性が著しく低下する。

```python
# ===== アンチパターン: 深いネストのスパゲッティコード =====

def process_order_bad(order):
    """深いネストの悪い例"""
    if order is not None:
        if order.get("status") == "pending":
            if order.get("items"):
                total = 0
                for item in order["items"]:
                    if item.get("price") is not None:
                        if item["price"] > 0:
                            if item.get("quantity", 0) > 0:
                                subtotal = item["price"] * item["quantity"]
                                if order.get("discount"):
                                    if order["discount"] > 0:
                                        if order["discount"] <= 0.5:
                                            subtotal *= (1 - order["discount"])
                                        else:
                                            print("割引率が大きすぎる")
                                            return None
                                total += subtotal
                            else:
                                print("数量が不正")
                                return None
                        else:
                            print("価格が不正")
                            return None
                    else:
                        print("価格が未設定")
                        return None
                return total
            else:
                print("商品がない")
                return None
        else:
            print("ステータスが不正")
            return None
    else:
        print("注文がない")
        return None


# ===== 改善版: ガード節とヘルパー関数で平坦化 =====

def validate_order(order):
    """注文の基本検証"""
    if order is None:
        return False, "注文がない"
    if order.get("status") != "pending":
        return False, "ステータスが不正"
    if not order.get("items"):
        return False, "商品がない"
    return True, ""


def validate_item(item):
    """商品アイテムの検証"""
    if item.get("price") is None:
        return False, "価格が未設定"
    if item["price"] <= 0:
        return False, "価格が不正"
    if item.get("quantity", 0) <= 0:
        return False, "数量が不正"
    return True, ""


def validate_discount(discount):
    """割引率の検証"""
    if discount is None or discount <= 0:
        return 1.0  # 割引なし
    if discount > 0.5:
        raise ValueError("割引率が大きすぎる")
    return 1.0 - discount


def calculate_item_subtotal(item, discount_rate):
    """アイテムの小計を計算する"""
    return item["price"] * item["quantity"] * discount_rate


def process_order_good(order):
    """改善版: ガード節とヘルパー関数で平坦化"""
    # ガード節: 異常ケースを早期に排除
    is_valid, error = validate_order(order)
    if not is_valid:
        print(error)
        return None

    # 割引率の検証
    try:
        discount_multiplier = validate_discount(order.get("discount"))
    except ValueError as e:
        print(str(e))
        return None

    # メイン処理: 合計計算
    total = 0
    for item in order["items"]:
        is_valid, error = validate_item(item)
        if not is_valid:
            print(error)
            return None
        total += calculate_item_subtotal(item, discount_multiplier)

    return total
```

### 9.3 アンチパターン3: マジックナンバーとハードコーディング

コード中に意味の不明な数値リテラルや文字列リテラルが散在するパターンである。

```python
# ===== アンチパターン: マジックナンバー =====

def calculate_shipping_bad(weight, distance):
    """マジックナンバーだらけの悪い例"""
    if weight < 2.0:
        base = 500
    elif weight < 10.0:
        base = 1200
    elif weight < 30.0:
        base = 2500
    else:
        base = 5000

    if distance > 500:
        base *= 1.5
    if distance > 1000:
        base *= 1.2

    if base > 10000:
        base = 10000

    return int(base * 1.1)  # 何の1.1？


# ===== 改善版: 定数に名前を付ける =====

# 重量区分の閾値（kg）
WEIGHT_LIGHT = 2.0
WEIGHT_MEDIUM = 10.0
WEIGHT_HEAVY = 30.0

# 重量区分ごとの基本料金（円）
BASE_RATE_LIGHT = 500
BASE_RATE_MEDIUM = 1200
BASE_RATE_HEAVY = 2500
BASE_RATE_EXTRA_HEAVY = 5000

# 距離加算の閾値（km）
LONG_DISTANCE_THRESHOLD = 500
VERY_LONG_DISTANCE_THRESHOLD = 1000

# 距離加算の乗率
LONG_DISTANCE_MULTIPLIER = 1.5
VERY_LONG_DISTANCE_MULTIPLIER = 1.2

# 送料上限（円）
MAX_SHIPPING_COST = 10000

# 消費税率
TAX_RATE = 0.1

def calculate_shipping_good(weight, distance):
    """定数名で意味が明確な改善例"""
    # 重量に基づく基本料金の決定
    if weight < WEIGHT_LIGHT:
        base = BASE_RATE_LIGHT
    elif weight < WEIGHT_MEDIUM:
        base = BASE_RATE_MEDIUM
    elif weight < WEIGHT_HEAVY:
        base = BASE_RATE_HEAVY
    else:
        base = BASE_RATE_EXTRA_HEAVY

    # 距離による加算
    if distance > VERY_LONG_DISTANCE_THRESHOLD:
        base *= LONG_DISTANCE_MULTIPLIER * VERY_LONG_DISTANCE_MULTIPLIER
    elif distance > LONG_DISTANCE_THRESHOLD:
        base *= LONG_DISTANCE_MULTIPLIER

    # 上限の適用
    base = min(base, MAX_SHIPPING_COST)

    # 税込み価格
    return int(base * (1 + TAX_RATE))
```

### 9.4 アンチパターン4: 関数の肥大化と責務の混在

一つの関数があまりにも多くのことを行い、数百行に膨れ上がるパターンである。単一責任の原則に違反し、テスト・保守・再利用が困難になる。

```python
# ===== アンチパターン: 全部入り関数 =====

def process_everything(filepath):
    """
    一つの関数でファイル読み込み、バリデーション、
    計算、整形、ファイル出力を全て行う。
    テストも再利用も困難。
    """
    # ファイル読み込み（30行）
    # バリデーション（40行）
    # データ変換（50行）
    # 計算処理（60行）
    # 結果の整形（30行）
    # ファイル出力（20行）
    # 合計: 230行以上の巨大関数
    pass


# ===== 改善版: 責務ごとに分割 =====

def read_data(filepath):
    """ファイルからデータを読み込む"""
    pass

def validate_data(raw_data):
    """データの妥当性を検証する"""
    pass

def transform_data(validated_data):
    """データを処理用の形式に変換する"""
    pass

def calculate_results(transformed_data):
    """計算処理を行う"""
    pass

def format_report(results):
    """結果をレポート形式に整形する"""
    pass

def write_report(report, output_path):
    """レポートをファイルに出力する"""
    pass

def process_pipeline(input_path, output_path):
    """
    パイプライン: 各関数を順に呼び出す。
    各ステップは独立してテスト可能。
    """
    raw = read_data(input_path)
    validated = validate_data(raw)
    transformed = transform_data(validated)
    results = calculate_results(transformed)
    report = format_report(results)
    write_report(report, output_path)
```

---

## 10. 演習問題

### 10.1 基礎演習

**演習 1: 基本制御構造の練習**

以下の仕様に従って、Python で関数を実装せよ。

```
仕様: FizzBuzz の拡張版
=========================

入力: 正の整数 n
出力: 1 から n までの各数について以下のルールで文字列のリストを返す

  - 3 の倍数 → "Fizz"
  - 5 の倍数 → "Buzz"
  - 7 の倍数 → "Whizz"
  - 3 と 5 の倍数 → "FizzBuzz"
  - 3 と 7 の倍数 → "FizzWhizz"
  - 5 と 7 の倍数 → "BuzzWhizz"
  - 3 と 5 と 7 の倍数 → "FizzBuzzWhizz"
  - それ以外 → 数値の文字列表現

使用する制御構造: for ループ、if-elif-else
```

**模範解答:**

```python
def fizzbuzz_extended(n):
    """
    FizzBuzz拡張版: 3, 5, 7 の倍数を判定する。
    ビルダーパターンで文字列を組み立てる手法を用いる。
    """
    result = []

    for i in range(1, n + 1):
        output = ""

        if i % 3 == 0:
            output += "Fizz"
        if i % 5 == 0:
            output += "Buzz"
        if i % 7 == 0:
            output += "Whizz"

        if output == "":
            output = str(i)

        result.append(output)

    return result


# テスト
output = fizzbuzz_extended(105)
for i, val in enumerate(output, 1):
    if val != str(i):  # 数値でないもの（変換が起きたもの）のみ表示
        print(f"{i:3d}: {val}")

# 出力例:
#   3: Fizz
#   5: Buzz
#   6: Fizz
#   7: Whizz
#   9: Fizz
#  10: Buzz
#  12: Fizz
#  14: Whizz
#  15: FizzBuzz
#  21: FizzWhizz
#  35: BuzzWhizz
# 105: FizzBuzzWhizz
```

**演習 2: 配列操作の練習**

以下の仕様に従って、C 言語で関数を実装せよ。

```
仕様: 配列の回転
================

入力: 整数配列 arr、配列の長さ n、回転数 k
処理: 配列を右に k 回転させる
  例: [1,2,3,4,5] を k=2 で回転 → [4,5,1,2,3]

制約:
  - 追加の配列を使用せず、O(1) の追加メモリで実装すること
  - ヒント: 反転（reverse）操作を3回適用する
```

**模範解答:**

```c
#include <stdio.h>

/*
 * 配列の指定範囲を反転する補助関数
 * 命令型の特徴: インデックス操作と値の交換
 */
void reverse(int arr[], int start, int end) {
    while (start < end) {
        int temp = arr[start];
        arr[start] = arr[end];
        arr[end] = temp;
        start++;
        end--;
    }
}

/*
 * 配列を右に k 回転させる
 * アルゴリズム:
 *   1. 全体を反転
 *   2. 先頭 k 要素を反転
 *   3. 残り n-k 要素を反転
 *
 * 例: [1,2,3,4,5], k=2
 *   全体反転: [5,4,3,2,1]
 *   先頭2反転: [4,5,3,2,1]
 *   残り3反転: [4,5,1,2,3]
 */
void rotate_right(int arr[], int n, int k) {
    if (n <= 1) return;

    k = k % n;  /* k が n 以上の場合に対応 */
    if (k == 0) return;

    reverse(arr, 0, n - 1);      /* 全体を反転 */
    reverse(arr, 0, k - 1);      /* 先頭 k 要素を反転 */
    reverse(arr, k, n - 1);      /* 残りを反転 */
}

void print_array(int arr[], int n) {
    printf("[");
    for (int i = 0; i < n; i++) {
        if (i > 0) printf(", ");
        printf("%d", arr[i]);
    }
    printf("]\n");
}

int main(void) {
    int arr[] = {1, 2, 3, 4, 5};
    int n = 5;

    printf("回転前: ");
    print_array(arr, n);

    rotate_right(arr, n, 2);

    printf("回転後: ");
    print_array(arr, n);

    return 0;
}
/* 出力:
 * 回転前: [1, 2, 3, 4, 5]
 * 回転後: [4, 5, 1, 2, 3]
 */
```

### 10.2 応用演習

**演習 3: 状態機械（ステートマシン）の実装**

以下の仕様に従って、文字列から数値を解析する有限状態機械を実装せよ。

```
仕様: 簡易数値パーサ
=====================

入力: 文字列（例: "  -123.456  "）
出力: 解析結果（浮動小数点数または整数）

状態遷移図:
  [開始] --(空白)--> [開始]
  [開始] --(+/-)--> [符号]
  [開始] --(数字)--> [整数部]
  [符号] --(数字)--> [整数部]
  [整数部] --(数字)--> [整数部]
  [整数部] --(.)--> [小数点]
  [整数部] --(空白)--> [末尾空白]
  [小数点] --(数字)--> [小数部]
  [小数部] --(数字)--> [小数部]
  [小数部] --(空白)--> [末尾空白]
  [末尾空白] --(空白)--> [末尾空白]
  上記以外の遷移 --> [エラー]
```

**模範解答:**

```python
def parse_number(text):
    """
    有限状態機械（FSM）による数値パーサ。

    命令型プログラミングの典型的応用: 状態遷移を
    変数と条件分岐で明示的に管理する。
    """
    # 状態の定義
    STATE_START = "start"
    STATE_SIGN = "sign"
    STATE_INTEGER = "integer"
    STATE_DOT = "dot"
    STATE_DECIMAL = "decimal"
    STATE_TRAILING = "trailing"
    STATE_ERROR = "error"

    state = STATE_START
    result_str = ""

    for ch in text:
        if state == STATE_START:
            if ch == ' ':
                pass  # 先頭の空白をスキップ
            elif ch in ('+', '-'):
                result_str += ch
                state = STATE_SIGN
            elif ch.isdigit():
                result_str += ch
                state = STATE_INTEGER
            else:
                state = STATE_ERROR

        elif state == STATE_SIGN:
            if ch.isdigit():
                result_str += ch
                state = STATE_INTEGER
            else:
                state = STATE_ERROR

        elif state == STATE_INTEGER:
            if ch.isdigit():
                result_str += ch
            elif ch == '.':
                result_str += ch
                state = STATE_DOT
            elif ch == ' ':
                state = STATE_TRAILING
            else:
                state = STATE_ERROR

        elif state == STATE_DOT:
            if ch.isdigit():
                result_str += ch
                state = STATE_DECIMAL
            else:
                state = STATE_ERROR

        elif state == STATE_DECIMAL:
            if ch.isdigit():
                result_str += ch
            elif ch == ' ':
                state = STATE_TRAILING
            else:
                state = STATE_ERROR

        elif state == STATE_TRAILING:
            if ch == ' ':
                pass  # 末尾の空白をスキップ
            else:
                state = STATE_ERROR

        if state == STATE_ERROR:
            return None, f"不正な文字 '{ch}' を検出"

    # 終了状態の検証
    if state in (STATE_INTEGER, STATE_DECIMAL, STATE_TRAILING):
        if '.' in result_str:
            return float(result_str), "浮動小数点数"
        else:
            return int(result_str), "整数"
    elif state == STATE_START:
        return None, "空の入力"
    else:
        return None, f"不完全な入力（状態: {state}）"


# テスト
test_cases = [
    "  42  ",
    "  -123.456  ",
    "+7.0",
    "  100  ",
    "  12.34.56  ",  # エラー
    "  abc  ",       # エラー
    "  ",            # エラー
    "  +  ",         # エラー
]

for text in test_cases:
    value, description = parse_number(text)
    print(f"'{text}' -> {value} ({description})")

# 出力:
# '  42  ' -> 42 (整数)
# '  -123.456  ' -> -123.456 (浮動小数点数)
# '+7.0' -> 7.0 (浮動小数点数)
# '  100  ' -> 100 (整数)
# '  12.34.56  ' -> None (不正な文字 '.' を検出)
# '  abc  ' -> None (不正な文字 'a' を検出)
# '  ' -> None (空の入力)
# '  +  ' -> None (不完全な入力（状態: sign）)
```

### 10.3 発展演習

**演習 4: 命令型から関数型へのリファクタリング**

以下の命令型コードを、関数型スタイル（map, filter, reduce / リスト内包表記）にリファクタリングせよ。振る舞いは完全に同一でなければならない。

```python
# ===== リファクタリング対象の命令型コード =====

def analyze_text_imperative(text):
    """テキストを解析して統計情報を返す（命令型）"""
    words = text.lower().split()

    # 1. ストップワードの除去
    stop_words = {"the", "a", "an", "is", "are", "was", "were",
                  "in", "on", "at", "to", "for", "of", "and",
                  "or", "but", "not", "with", "by"}
    filtered = []
    for word in words:
        clean = ""
        for ch in word:
            if ch.isalnum():
                clean += ch
        if clean and clean not in stop_words:
            filtered.append(clean)

    # 2. 単語の長さ別にグループ化
    groups = {}
    for word in filtered:
        length = len(word)
        if length not in groups:
            groups[length] = []
        groups[length].append(word)

    # 3. 各グループの要約
    summary = {}
    for length, word_list in groups.items():
        unique = []
        seen = set()
        for w in word_list:
            if w not in seen:
                unique.append(w)
                seen.add(w)
        summary[length] = {
            "count": len(word_list),
            "unique": len(unique),
            "words": unique
        }

    return summary
```

**模範解答（関数型スタイル）:**

```python
from collections import Counter
from itertools import groupby

def analyze_text_functional(text):
    """テキストを解析して統計情報を返す（関数型）"""
    stop_words = frozenset({
        "the", "a", "an", "is", "are", "was", "were",
        "in", "on", "at", "to", "for", "of", "and",
        "or", "but", "not", "with", "by"
    })

    # パイプライン: 分割 → クリーニング → フィルタ
    clean_word = lambda w: "".join(ch for ch in w if ch.isalnum())
    words = [
        cleaned for w in text.lower().split()
        if (cleaned := clean_word(w)) and cleaned not in stop_words
    ]

    # グループ化と要約を一度に生成
    sorted_words = sorted(words, key=len)
    summary = {
        length: {
            "count": len(group_words := list(group)),
            "unique": len(unique := list(dict.fromkeys(group_words))),
            "words": unique,
        }
        for length, group in groupby(sorted_words, key=len)
    }

    return summary


# テスト: 両方の関数が同じ結果を返すことを確認
sample = """
The quick brown fox jumps over the lazy dog.
A fox is not a dog, but the fox and the dog are friends.
"""

result_imp = analyze_text_imperative(sample)
result_fun = analyze_text_functional(sample)

for length in sorted(result_imp.keys()):
    r = result_imp[length]
    print(f"  長さ{length}: {r['count']}語 ({r['unique']}種類) {r['words']}")
```

---

## 11. FAQ（よくある質問）

### Q1. 命令型プログラミングは古いパラダイムで、今後は関数型に置き換えられるのか

命令型プログラミングが関数型に完全に置き換えられるということは考えにくい。理由は以下の通りである。

**命令型が今後も重要であり続ける理由:**

1. **ハードウェアとの対応**: 現代のコンピュータはフォン・ノイマンアーキテクチャに基づいており、命令型プログラミングはこのアーキテクチャと直接対応する。OS、デバイスドライバ、組み込みシステムなどの低水準プログラミングでは命令型が不可欠である。

2. **性能要件**: 性能が重要な場面では、メモリ配置やキャッシュの利用効率を明示的に制御する必要がある。これは命令型プログラミングの領域である。

3. **直感性**: 初学者にとって「手順を順番に書く」という命令型の考え方は最も直感的であり、プログラミング教育の入り口として適している。

**現実的なトレンド:**

現代の主流言語（Python, JavaScript, Rust, Kotlin, Swift など）はマルチパラダイムであり、命令型と関数型の要素を融合している。「命令型か関数型か」という二者択一ではなく、問題に応じて適切なスタイルを選択する能力が重要となっている。

### Q2. goto 文は絶対に使うべきではないのか

ダイクストラの「goto 有害論」は「goto 文の無秩序な使用」を批判したものであり、「goto 文の全面禁止」を主張したものではない。現代のプログラミングにおいても、goto 文が合理的に使用されるケースがある。

**goto 文が許容される場面:**

1. **エラー処理のクリーンアップ（C 言語）**: C 言語には例外処理機構がないため、リソースの解放処理を共通化する目的で goto 文が使用される。Linux カーネルのコーディングスタイルガイドでもこの用法は推奨されている。

```c
int process_file(const char *filename) {
    FILE *fp = NULL;
    char *buffer = NULL;
    int result = -1;

    fp = fopen(filename, "r");
    if (fp == NULL) goto cleanup;

    buffer = malloc(1024);
    if (buffer == NULL) goto cleanup;

    /* 処理 ... */
    result = 0;

cleanup:
    free(buffer);       /* NULLでもfreeは安全 */
    if (fp) fclose(fp);
    return result;
}
```

2. **多重ループからの脱出**: 一部の言語では、ネストしたループを一度に抜けるために goto 文やラベル付き break が使用される。

**多くの現代言語がgotoを排除した理由:**

Python, Java, JavaScript, Ruby, Go（limited goto）など多くの現代言語は、例外処理（try-catch）、ラベル付き break/continue、defer 文などの構造化された代替手段を提供しており、goto 文の必要性を大幅に低減している。

### Q3. 命令型プログラミングで並行処理を安全に行うにはどうすればよいか

命令型プログラミングにおける並行処理の最大の課題は「共有可変状態（Shared Mutable State）」である。複数のスレッドやプロセスが同じ変数を同時に読み書きすると、データ競合（Race Condition）が発生する。

**主な解決策:**

1. **排他制御（Mutex / Lock）**: 共有リソースへのアクセスを逐次化する。
2. **メッセージパッシング**: 共有状態を持たず、メッセージの送受信でデータをやりとりする（Go の goroutine + channel）。
3. **不変データの活用**: 関数型プログラミングのアプローチを取り入れ、データを不変にする。
4. **所有権システム**: Rust のように、コンパイル時にデータ競合を検出する。

```python
import threading

# === 問題のあるコード: データ競合 ===

counter = 0

def increment_unsafe():
    global counter
    for _ in range(100000):
        counter += 1  # この操作はアトミックではない
        # 読み取り → 加算 → 書き込み の3ステップであり、
        # 他のスレッドが途中で割り込む可能性がある

# === 解決策: ロックによる排他制御 ===

counter_safe = 0
lock = threading.Lock()

def increment_safe():
    global counter_safe
    for _ in range(100000):
        with lock:                    # ロックを取得
            counter_safe += 1         # この間、他のスレッドは待機
                                      # ロックを自動解放

# テスト
threads = [threading.Thread(target=increment_safe) for _ in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"期待値: 400000, 実際: {counter_safe}")
# ロック使用: 常に 400000
```

### Q4. 命令型プログラミングにおける「良いコード」とは何か

命令型プログラミングにおいて「良いコード」の基準は以下の通りである。

1. **可読性**: コードを読む人が意図を理解できること。変数名・関数名が適切で、処理の流れが追えること。

2. **予測可能性**: 関数の振る舞いが入力と名前から予測できること。副作用が最小限に抑えられていること。

3. **テスト容易性**: 関数が独立しており、単体テストが書きやすいこと。外部依存が注入可能であること。

4. **変更容易性**: 要件変更に対して、影響範囲が限定されること。モジュール間の結合が疎であること。

5. **単純さ**: 必要以上に複雑な構造を避けること。「動くコード」と「美しいコード」のバランスを取ること。

### Q5. 命令型プログラミングを学ぶ最適な順序は何か

以下の順序で学ぶことを推奨する。

```
学習ロードマップ
═══════════════════════════════════════

  Phase 1: 基礎（1-2ヶ月）
  ├── 変数と代入
  ├── 基本データ型（整数、浮動小数点、文字列、真偽値）
  ├── 順次・選択・反復の3制御構造
  ├── 関数の定義と呼び出し
  └── 推奨言語: Python

  Phase 2: データ構造と中級制御（2-3ヶ月）
  ├── 配列、リスト、辞書
  ├── ネストした制御構造
  ├── 再帰
  ├── ファイル入出力
  ├── エラー処理
  └── 推奨言語: Python + C 入門

  Phase 3: 設計原則（2-3ヶ月）
  ├── 関数設計（単一責任、引数と返り値）
  ├── モジュール化
  ├── 状態管理のベストプラクティス
  ├── テスト駆動開発の基礎
  └── 推奨言語: Python

  Phase 4: 他パラダイムとの融合（3ヶ月〜）
  ├── オブジェクト指向プログラミング
  ├── 関数型プログラミングの要素
  ├── 並行プログラミング
  ├── 低水準プログラミング（C/Rust）
  └── 推奨: マルチパラダイム言語を深く学ぶ
```

### Q6. 命令型コードのデバッグで最も重要な技法は何か

命令型コードのデバッグで最も有効な技法は「状態の追跡（State Tracking）」である。命令型プログラムは状態遷移の連鎖であるため、各時点の状態（変数の値の集合）を把握することが問題発見の鍵となる。

**具体的な技法:**

1. **ステップ実行（Step Execution）**: デバッガを使い、一文ずつ実行して変数の変化を観察する。
2. **ウォッチポイント**: 特定の変数が変更された時点でプログラムを停止させる。
3. **アサーション**: プログラム中に `assert` 文を埋め込み、前提条件の違反を早期に検出する。
4. **ロギング**: 状態遷移の履歴をログに記録し、事後解析に利用する。
5. **不変条件の検証**: ループの各反復で成立すべき条件（ループ不変条件）を明示し、検証する。

```python
def binary_search_debug(sorted_list, target):
    """デバッグ技法を組み込んだ二分探索"""
    low = 0
    high = len(sorted_list) - 1
    iteration = 0

    while low <= high:
        iteration += 1

        # ループ不変条件の検証
        assert 0 <= low <= high < len(sorted_list), \
            f"不変条件違反: low={low}, high={high}, len={len(sorted_list)}"

        mid = (low + high) // 2

        # 状態のログ出力
        print(f"  反復{iteration}: low={low}, mid={mid}, high={high}, "
              f"arr[mid]={sorted_list[mid]}, target={target}")

        if sorted_list[mid] == target:
            print(f"  → 発見! インデックス={mid}")
            return mid
        elif sorted_list[mid] < target:
            low = mid + 1
            print(f"  → 右半分を探索 (low={low})")
        else:
            high = mid - 1
            print(f"  → 左半分を探索 (high={high})")

    print(f"  → 見つからなかった ({iteration}回の反復)")
    return -1
```

---

## 12. まとめ・参考文献

### 12.1 まとめ

本章で学んだ命令型プログラミングの重要概念を以下に整理する。

| 概念 | ポイント |
|------|---------|
| 命令型の本質 | 「どうやるか（How）」を逐次的に指示する。状態・代入・制御フローが三本柱 |
| フォン・ノイマン対応 | 変数=メモリ、代入=ストア命令、制御フロー=ジャンプ命令。ハードウェアと直接対応する |
| 歴史的経緯 | 機械語→アセンブリ→FORTRAN→ALGOL→C。各段階で抽象度が向上した |
| 制御構造 | 順次・選択・反復の三構造のみで任意のアルゴリズムを表現可能（構造化定理） |
| 状態管理 | 可変状態は命令型の根幹であり、同時に複雑さの主要な源泉でもある |
| 副作用 | グローバル変数の変更、I/O操作など。管理しないとバグの温床になる |
| 構造化プログラミング | Dijkstraによるgoto排除。コードの可読性と保守性を劇的に向上させた |
| 手続き型 | 関数によるモジュール化。単一責任・明示的入出力・適切な粒度が原則 |
| パラダイム比較 | 命令型は状態変化、関数型は値変換、OOPはデータと操作の統合 |
| 現代のトレンド | マルチパラダイム。命令型と関数型の融合が主流 |
| アンチパターン | グローバル状態乱用、深いネスト、マジックナンバー、関数の肥大化 |

```
命令型プログラミングの位置づけ（最終整理）
═══════════════════════════════════════════════════

  プログラミングパラダイムの全体像:

  ┌───────────────────────────────────────────┐
  │           プログラミングパラダイム            │
  │                                           │
  │  ┌─────────────┐   ┌─────────────┐      │
  │  │  命令型       │   │  宣言型       │      │
  │  │  (How)       │   │  (What)      │      │
  │  │              │   │              │      │
  │  │ ・手続き型    │   │ ・関数型      │      │
  │  │ ・OOP        │   │ ・論理型      │      │
  │  │ ・構造化     │   │ ・データフロー  │      │
  │  └─────────────┘   └─────────────┘      │
  │                                           │
  │  現代: マルチパラダイム                       │
  │  Python, Rust, Kotlin, TypeScript ...      │
  │  → 命令型と宣言型の要素を自在に組み合わせる    │
  └───────────────────────────────────────────┘
```

### 12.2 参考文献

1. Dijkstra, E. W. "Go To Statement Considered Harmful." *Communications of the ACM*, Vol. 11, No. 3, pp. 147-148, March 1968.
   - 構造化プログラミングの出発点となった歴史的論文。goto 文の無秩序な使用がプログラムの品質を低下させることを指摘した。

2. Bohm, C. & Jacopini, G. "Flow Diagrams, Turing Machines and Languages with Only Two Formation Rules." *Communications of the ACM*, Vol. 9, No. 5, pp. 366-371, May 1966.
   - 構造化定理（任意のプログラムが順次・選択・反復の三構造で表現可能であること）を数学的に証明した論文。

3. Wirth, N. "Program Development by Stepwise Refinement." *Communications of the ACM*, Vol. 14, No. 4, pp. 221-227, April 1971.
   - 段階的詳細化（Stepwise Refinement）の概念を体系化した論文。トップダウン設計の基盤となった。

4. Kernighan, B. W. & Ritchie, D. M. *The C Programming Language*, 2nd Edition. Prentice Hall, 1988.
   - C 言語の標準的な教科書であり、命令型・手続き型プログラミングの模範的なスタイルを示している。通称 K&R。

5. Knuth, D. E. "Structured Programming with go to Statements." *Computing Surveys*, Vol. 6, No. 4, pp. 261-301, December 1974.
   - goto 有害論に対する均衡のとれた分析を提供し、goto 文の合理的な使用場面についても論じた。

6. Abelson, H. & Sussman, G. J. *Structure and Interpretation of Computer Programs*, 2nd Edition. MIT Press, 1996.
   - 命令型・関数型を含むプログラミングの基礎概念を深く掘り下げた MIT の名著。通称 SICP。

7. Van Roy, P. & Haridi, S. *Concepts, Techniques, and Models of Computer Programming*. MIT Press, 2004.
   - 命令型を含む複数のプログラミングパラダイムを統一的な枠組みで解説した包括的教科書。

---

## 次に読むべきガイド

- [[01-object-oriented.md]] -- オブジェクト指向プログラミング（命令型の拡張としての OOP）
- [[02-functional.md]] -- 関数型プログラミング（命令型と対照的なパラダイム）
- [[03-declarative.md]] -- 宣言型プログラミング（SQL, HTML 等の「何を」記述するアプローチ）

