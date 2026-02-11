# 関数型プログラミング

> 関数型プログラミングは「副作用のない純粋関数の合成」であり、並行処理とテスタビリティに優れる。

## この章で学ぶこと

- [ ] 純粋関数と副作用の概念を説明できる
- [ ] map/filter/reduceを使いこなせる
- [ ] 不変性のメリットを理解する

---

## 1. 関数型の核心概念

### 1.1 純粋関数

```python
# 純粋関数: 同じ入力 → 同じ出力、副作用なし
def add(a, b):
    return a + b  # ✅ 純粋: 入力のみに依存、外部を変更しない

# 不純な関数:
total = 0
def add_to_total(x):
    global total
    total += x     # ❌ 副作用: 外部状態を変更
    return total   # ❌ 外部状態に依存

# 純粋関数の利点:
# 1. テストが容易（モック不要）
# 2. 並行処理に安全（共有状態なし）
# 3. キャッシュ可能（メモ化）
# 4. 推論が容易（参照透過性）
```

### 1.2 高階関数

```python
# 高階関数: 関数を引数に取るか、関数を返す関数

# map: 各要素に関数を適用
names = ["alice", "bob", "charlie"]
upper = list(map(str.upper, names))  # ['ALICE', 'BOB', 'CHARLIE']

# filter: 条件を満たす要素を抽出
nums = [1, 2, 3, 4, 5, 6]
evens = list(filter(lambda x: x % 2 == 0, nums))  # [2, 4, 6]

# reduce: 畳み込み（累積演算）
from functools import reduce
total = reduce(lambda acc, x: acc + x, nums, 0)  # 21

# Python的に書くならリスト内包表記:
upper = [name.upper() for name in names]
evens = [x for x in nums if x % 2 == 0]
total = sum(nums)

# 関数合成
def compose(f, g):
    return lambda x: f(g(x))

double = lambda x: x * 2
add_one = lambda x: x + 1
double_then_add = compose(add_one, double)
print(double_then_add(5))  # 11
```

### 1.3 不変性（Immutability）

```python
# 不変データ構造のメリット:
# 1. スレッドセーフ（ロック不要）
# 2. 予測可能（どこかで変更されない）
# 3. 変更履歴の追跡が容易

# ❌ ミュータブル:
cart = [{"item": "A", "qty": 1}]
cart[0]["qty"] = 2  # 元のデータが変更される！

# ✅ イミュータブル:
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)  # frozen=True で不変
class CartItem:
    item: str
    qty: int

# 更新は新しいオブジェクトを生成
item = CartItem("A", 1)
updated = CartItem(item.item, 2)  # 元のitemは変更されない

# JavaScript/React での不変性:
# const newState = { ...state, count: state.count + 1 };
# → Reactの状態管理は不変性が前提
```

---

## 2. 関数型パターン

```python
# パイプライン（データの変換チェーン）
def pipeline(data, *functions):
    for func in functions:
        data = func(data)
    return data

result = pipeline(
    [1, 2, 3, 4, 5],
    lambda xs: [x * 2 for x in xs],      # 全て2倍
    lambda xs: [x for x in xs if x > 4],  # 4より大きい
    sum                                    # 合計
)
# [1,2,3,4,5] → [2,4,6,8,10] → [6,8,10] → 24

# カリー化
from functools import partial
def multiply(x, y):
    return x * y

double = partial(multiply, 2)
triple = partial(multiply, 3)
print(double(5))  # 10
print(triple(5))  # 15
```

---

## 3. 実務での関数型

```
関数型が活躍する場面:

  1. データ変換パイプライン（ETL, ストリーム処理）
  2. React/Reduxの状態管理
  3. 並行/並列処理（Erlang/Elixir）
  4. テスト駆動開発（純粋関数はテスト容易）
  5. 機械学習のデータ前処理

  関数型を取り入れた命令型言語:
  - Python: lambda, map, filter, functools
  - JavaScript: Array.map/filter/reduce, スプレッド構文
  - Java 8+: Stream API, Optional, ラムダ式
  - Rust: イテレータ, パターンマッチ, Option/Result
  - TypeScript: 型推論, ジェネリクス, readonly
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| 純粋関数 | 副作用なし。テスト容易、並行安全 |
| 高階関数 | map/filter/reduce。データ変換の基本 |
| 不変性 | データを変更せず新しいデータを作る |
| 関数合成 | 小さな関数を組み合わせて複雑な処理を構築 |

---

## 次に読むべきガイド
→ [[03-concurrent.md]] — 並行・並列プログラミング

---

## 参考文献
1. Hutton, G. "Programming in Haskell." Cambridge, 2016.
2. Chiusano, P. & Bjarnason, R. "Functional Programming in Scala." Manning, 2014.
