# データ構造 - 完全初心者ガイド

## 目次

1. [概要](#概要)
2. [リスト（List）](#リストlist)
3. [タプル（Tuple）](#タプルtuple)
4. [辞書（Dictionary）](#辞書dictionary)
5. [セット（Set）](#セットset)
6. [内包表記](#内包表記)
7. [演習問題](#演習問題)
8. [次のステップ](#次のステップ)

---

## 概要

### 何を学ぶか

- リスト（可変、順序あり）
- タプル（不変、順序あり）
- 辞書（キーと値のペア）
- セット（重複なし、順序なし）
- 内包表記（簡潔な記法）

### 学習時間：2〜3時間

---

## リスト（List）

### 基本操作

```python
# リストの作成
fruits = ["りんご", "バナナ", "オレンジ"]

# インデックスアクセス
print(fruits[0])   # りんご
print(fruits[-1])  # オレンジ

# スライス
print(fruits[0:2])  # ['りんご', 'バナナ']

# 長さ
print(len(fruits))  # 3
```

### リストの変更

```python
fruits = ["りんご", "バナナ", "オレンジ"]

# 追加
fruits.append("ぶどう")
fruits.insert(1, "いちご")  # 位置1に挿入

# 削除
fruits.remove("バナナ")     # 値で削除
del fruits[0]              # インデックスで削除
last = fruits.pop()        # 最後の要素を削除して返す

# 変更
fruits[0] = "メロン"

# ソート
numbers = [3, 1, 4, 1, 5]
numbers.sort()             # [1, 1, 3, 4, 5]
numbers.reverse()          # [5, 4, 3, 1, 1]
```

### リストのメソッド

```python
numbers = [1, 2, 3, 2, 4]

numbers.count(2)      # 2の個数 → 2
numbers.index(3)      # 3の位置 → 2
numbers.extend([5, 6])  # リストを結合
numbers.clear()       # すべて削除
```

---

## タプル（Tuple）

### 特徴：イミュータブル（不変）

```python
# タプルの作成
point = (10, 20)
colors = ("red", "green", "blue")

# アクセス
print(point[0])   # 10

# ❌ 変更不可
# point[0] = 15  # TypeError

# アンパック
x, y = point
print(x, y)  # 10 20

# 1要素のタプル（カンマ必須）
single = (42,)
```

### タプルの用途

```python
# 複数の戻り値
def get_user():
    return "太郎", 25, "東京"

name, age, city = get_user()

# 辞書のキー（リストは不可）
locations = {
    (0, 0): "原点",
    (1, 0): "右",
    (0, 1): "上"
}
```

---

## 辞書（Dictionary）

### 基本操作

```python
# 辞書の作成
user = {
    "name": "太郎",
    "age": 25,
    "city": "東京"
}

# アクセス
print(user["name"])        # 太郎
print(user.get("age"))     # 25
print(user.get("email", "なし"))  # デフォルト値

# 追加・変更
user["email"] = "taro@example.com"
user["age"] = 26

# 削除
del user["city"]
email = user.pop("email")  # 削除して返す
```

### 辞書のメソッド

```python
user = {"name": "太郎", "age": 25}

# キー、値、ペアの取得
print(user.keys())    # dict_keys(['name', 'age'])
print(user.values())  # dict_values(['太郎', 25])
print(user.items())   # dict_items([('name', '太郎'), ('age', 25)])

# ループ
for key, value in user.items():
    print(f"{key}: {value}")

# 存在確認
if "name" in user:
    print("nameキーが存在します")
```

---

## セット（Set）

### 特徴：重複なし、順序なし

```python
# セットの作成
numbers = {1, 2, 3, 2, 1}  # 重複は削除される
print(numbers)  # {1, 2, 3}

# 追加・削除
numbers.add(4)
numbers.remove(1)  # ない場合はエラー
numbers.discard(10)  # ない場合も安全

# 集合演算
a = {1, 2, 3}
b = {3, 4, 5}

print(a | b)  # 和集合: {1, 2, 3, 4, 5}
print(a & b)  # 積集合: {3}
print(a - b)  # 差集合: {1, 2}
print(a ^ b)  # 対称差: {1, 2, 4, 5}
```

---

## 内包表記

### リスト内包表記

```python
# 従来の方法
squares = []
for i in range(10):
    squares.append(i ** 2)

# 内包表記（簡潔）
squares = [i ** 2 for i in range(10)]

# 条件付き
evens = [i for i in range(10) if i % 2 == 0]
# [0, 2, 4, 6, 8]
```

### 辞書内包表記

```python
# 辞書の作成
squares = {i: i ** 2 for i in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

### セット内包表記

```python
unique_lengths = {len(word) for word in ["apple", "banana", "pear"]}
# {4, 5, 6}
```

---

## 演習問題

### 問題1：リストの操作

```python
# 1〜100の偶数のリストを作成
evens = [i for i in range(1, 101) if i % 2 == 0]

# 合計と平均
total = sum(evens)
average = total / len(evens)
print(f"合計: {total}, 平均: {average}")
```

### 問題2：辞書の操作

```python
# 学生の成績管理
students = {
    "太郎": {"math": 80, "english": 75},
    "花子": {"math": 90, "english": 85},
    "次郎": {"math": 70, "english": 80}
}

# 各学生の平均点
for name, scores in students.items():
    avg = sum(scores.values()) / len(scores)
    print(f"{name}の平均点: {avg}")
```

---

## 次のステップ

**次のガイド**：[06-modules-packages.md](./06-modules-packages.md) - モジュールとパッケージ
