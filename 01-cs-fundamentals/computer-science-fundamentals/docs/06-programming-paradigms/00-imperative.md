# 命令型プログラミング

> 命令型プログラミングは「コンピュータに手順を指示する」最も直感的なパラダイムであり、フォン・ノイマンアーキテクチャと直接対応する。

## この章で学ぶこと

- [ ] 命令型プログラミングの特徴を説明できる
- [ ] 構造化プログラミングの原則を理解する
- [ ] 手続き型とオブジェクト指向の違いを説明できる

---

## 1. 命令型プログラミングの本質

```
命令型: 「何をするか」ではなく「どうやるか」を記述

  特徴:
  1. 状態（変数）を持つ
  2. 代入文で状態を変更する
  3. 制御フロー（if, for, while）で実行順序を制御
  4. フォン・ノイマンアーキテクチャと直接対応
     → 変数 = メモリ、代入 = ストア命令

  命令型 vs 宣言型:
  ────────────────────────────────
  命令型（How）:
    result = []
    for item in items:
        if item.price > 100:
            result.append(item.name)

  宣言型（What）:
    result = [item.name for item in items if item.price > 100]

  SQL（宣言型の代表）:
    SELECT name FROM items WHERE price > 100;
    → 「どうやって」探すかはDBエンジンが決める
```

---

## 2. 構造化プログラミング

```
構造化プログラミング（Dijkstra, 1968）:

  「Go To Statement Considered Harmful」
  → goto文を排除し、3つの制御構造のみで構築

  1. 順次（Sequence）: 上から下へ順に実行
  2. 選択（Selection）: if-else
  3. 反復（Iteration）: while, for

  構造化定理:
  「任意のフローチャートはこの3構造の組み合わせで表現できる」

  影響:
  - goto文の廃止 → コードの読みやすさが劇的に向上
  - 関数/サブルーチンによるモジュール化
  - C, Pascal, Java, Python ... 全ての現代言語の基盤
```

---

## 3. 手続き型プログラミング

```python
# 手続き型: 関数（手続き）の集まりとしてプログラムを構築

# ❌ グローバル状態に依存（スパゲッティコード）
total = 0
def add_item(price):
    global total
    total += price

# ✅ 純粋な関数（入力→出力が明確）
def calculate_total(prices):
    return sum(prices)

def apply_discount(total, rate):
    return total * (1 - rate)

def calculate_tax(amount, tax_rate):
    return amount * (1 + tax_rate)

# パイプライン的に組み合わせ
prices = [100, 200, 300]
total = calculate_total(prices)
discounted = apply_discount(total, 0.1)
final = calculate_tax(discounted, 0.08)
```

---

## 4. パラダイムの選択

```
各パラダイムの使い分け:

  ┌──────────────┬──────────────────┬─────────────────┐
  │ パラダイム    │ 得意な場面        │ 代表的言語       │
  ├──────────────┼──────────────────┼─────────────────┤
  │ 手続き型     │ スクリプト、ツール │ C, Shell, Python │
  │ OOP         │ 大規模アプリ      │ Java, C#, Python │
  │ 関数型      │ データ変換、並行   │ Haskell, Elixir │
  │ 論理型      │ 知識表現、推論    │ Prolog           │
  │ マルチパラダイム│ 柔軟な設計     │ Python, Rust, TS │
  └──────────────┴──────────────────┴─────────────────┘

  現代のトレンド:
  → マルチパラダイム（複数のパラダイムを組み合わせ）
  → Python: 手続き型 + OOP + 関数型（lambda, map, filter）
  → Rust: 命令型 + 関数型（所有権 + パターンマッチ + イテレータ）
  → TypeScript: OOP + 関数型（型推論 + ジェネリクス）
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| 命令型 | 手順を記述。変数、代入、制御フロー |
| 構造化 | 順次・選択・反復の3構造。gotoを排除 |
| 手続き型 | 関数でモジュール化。入力→出力を明確に |
| マルチパラダイム | 現代の主流。状況に応じて使い分け |

---

## 次に読むべきガイド
→ [[01-object-oriented.md]] — オブジェクト指向プログラミング

---

## 参考文献
1. Dijkstra, E. W. "Go To Statement Considered Harmful." CACM, 1968.
2. Böhm, C. & Jacopini, G. "Flow Diagrams, Turing Machines and Languages." CACM, 1966.
