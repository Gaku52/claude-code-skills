# Python基本文法 - 完全初心者ガイド

## 目次

1. [概要](#概要)
2. [前提知識](#前提知識)
3. [変数と代入](#変数と代入)
4. [データ型](#データ型)
5. [演算子](#演算子)
6. [文字列操作](#文字列操作)
7. [型変換](#型変換)
8. [コメント](#コメント)
9. [よくある間違い](#よくある間違い)
10. [演習問題](#演習問題)
11. [次のステップ](#次のステップ)

---

## 概要

### 何を学ぶか

- 変数の宣言と代入
- 基本的なデータ型（int、float、str、bool）
- 算術演算子、比較演算子、論理演算子
- 文字列の操作とフォーマット
- 型変換の方法

### 学習時間の目安

- 読了：30〜40分
- 演習含む：2〜3時間

---

## 前提知識

- [01-python-intro.md](./01-python-intro.md) を完了していること

---

## 変数と代入

### 変数とは

**変数**は、データを格納する箱のようなものです。

```python
# 変数への代入
name = "太郎"
age = 25
height = 175.5

print(name)    # 太郎
print(age)     # 25
print(height)  # 175.5
```

### 変数名のルール

```python
# ✅ 正しい変数名
user_name = "太郎"
userName = "太郎"
user1 = "太郎"
_private = "プライベート"

# ❌ 間違った変数名
1user = "太郎"       # 数字から始まる
user-name = "太郎"   # ハイフン使用
class = "A組"        # 予約語
```

**命名規則**：
- 英数字とアンダースコア `_` のみ
- 数字から始めない
- 予約語は使わない
- **慣習**：`snake_case`（小文字 + アンダースコア）

### 複数の変数への代入

```python
# 同時に代入
x, y, z = 10, 20, 30

# 同じ値を代入
a = b = c = 0

# 値の交換
x, y = y, x
```

---

## データ型

### 1. 整数型（int）

```python
age = 25
year = 2024
negative = -10

print(type(age))  # <class 'int'>
```

### 2. 浮動小数点型（float）

```python
height = 175.5
pi = 3.14159
temperature = -5.0

print(type(height))  # <class 'float'>
```

### 3. 文字列型（str）

```python
name = "太郎"
message = 'こんにちは'
multiline = """複数行の
文字列です"""

print(type(name))  # <class 'str'>
```

### 4. 真偽値型（bool）

```python
is_student = True
is_adult = False

print(type(is_student))  # <class 'bool'>
```

### 型の確認

```python
x = 10
print(type(x))        # <class 'int'>
print(isinstance(x, int))  # True
```

---

## 演算子

### 1. 算術演算子

```python
a = 10
b = 3

# 四則演算
print(a + b)   # 13  加算
print(a - b)   # 7   減算
print(a * b)   # 30  乗算
print(a / b)   # 3.333... 除算

# その他
print(a // b)  # 3   整数除算（切り捨て）
print(a % b)   # 1   剰余（余り）
print(a ** b)  # 1000 べき乗
```

### 2. 比較演算子

```python
x = 10
y = 20

print(x == y)  # False  等しい
print(x != y)  # True   等しくない
print(x > y)   # False  大きい
print(x < y)   # True   小さい
print(x >= y)  # False  以上
print(x <= y)  # True   以下
```

### 3. 論理演算子

```python
age = 25
has_license = True

# and（かつ）
print(age >= 18 and has_license)  # True

# or（または）
print(age < 18 or has_license)    # True

# not（否定）
print(not has_license)            # False
```

### 4. 代入演算子

```python
x = 10

x += 5   # x = x + 5  → 15
x -= 3   # x = x - 3  → 12
x *= 2   # x = x * 2  → 24
x /= 4   # x = x / 4  → 6.0
```

---

## 文字列操作

### 文字列の結合

```python
first_name = "太郎"
last_name = "山田"

# + 演算子
full_name = last_name + " " + first_name
print(full_name)  # 山田 太郎

# * 演算子（繰り返し）
print("=" * 20)  # ====================
```

### 文字列フォーマット

```python
name = "太郎"
age = 25

# f-string（推奨）
message = f"私の名前は{name}で、{age}歳です。"
print(message)

# format()
message = "私の名前は{}で、{}歳です。".format(name, age)

# %演算子（古い方法）
message = "私の名前は%sで、%d歳です。" % (name, age)
```

### 文字列のインデックスとスライス

```python
text = "Hello, World!"

# インデックス
print(text[0])      # H
print(text[-1])     # !

# スライス
print(text[0:5])    # Hello
print(text[7:])     # World!
print(text[:5])     # Hello
print(text[::2])    # Hlo ol!（2文字ずつ）
```

### 文字列のメソッド

```python
text = "  Hello, World!  "

print(text.upper())      # "  HELLO, WORLD!  "
print(text.lower())      # "  hello, world!  "
print(text.strip())      # "Hello, World!"
print(text.replace("World", "Python"))  # "  Hello, Python!  "
print(text.split(","))   # ['  Hello', ' World!  ']
print(len(text))         # 17
```

---

## 型変換

### 明示的な型変換

```python
# 文字列 → 整数
age_str = "25"
age_int = int(age_str)
print(type(age_int))  # <class 'int'>

# 文字列 → 浮動小数点
height_str = "175.5"
height_float = float(height_str)

# 整数 → 文字列
num = 100
num_str = str(num)

# 数値 → 真偽値
print(bool(0))    # False
print(bool(1))    # True
print(bool(""))   # False（空文字列）
print(bool("a"))  # True
```

### よくあるエラー

```python
# ❌ エラー：int()に無効な文字列
# age = int("二十五")  # ValueError

# ✅ 正しい
age = int("25")

# ❌ エラー：文字列と数値の結合
# message = "私は" + 25 + "歳です"  # TypeError

# ✅ 正しい
message = "私は" + str(25) + "歳です"
# または
message = f"私は{25}歳です"
```

---

## コメント

### 単一行コメント

```python
# これはコメントです
print("Hello")  # 行末コメント
```

### 複数行コメント

```python
"""
これは複数行の
コメントです
"""

'''
シングルクォートでも
書けます
'''
```

### Docstring（ドキュメント文字列）

```python
def greet(name):
    """
    挨拶を表示する関数

    Args:
        name (str): 名前
    """
    print(f"こんにちは、{name}さん！")
```

---

## よくある間違い

### 間違い1：型の不一致

```python
# ❌ エラー
age = "25"
next_year = age + 1  # TypeError

# ✅ 修正
age = int("25")
next_year = age + 1
```

### 間違い2：変数名のタイプミス

```python
user_name = "太郎"
print(username)  # NameError
```

### 間違い3：整数除算と通常除算の混同

```python
# Python 3の場合
print(10 / 3)   # 3.333...（float）
print(10 // 3)  # 3（int）
```

---

## 演習問題

### 問題1：BMI計算機

```python
# BMI = 体重(kg) ÷ 身長(m)²

weight = float(input("体重(kg): "))
height = float(input("身長(cm): "))

height_m = height / 100  # cmをmに変換
bmi = weight / (height_m ** 2)

print(f"あなたのBMIは{bmi:.1f}です")
```

### 問題2：温度変換

```python
# 摂氏 → 華氏: F = C × 9/5 + 32

celsius = float(input("摂氏温度: "))
fahrenheit = celsius * 9/5 + 32

print(f"{celsius}°C = {fahrenheit:.1f}°F")
```

---

## 次のステップ

### このガイドで学んだこと

- ✅ 変数と代入
- ✅ 基本データ型
- ✅ 演算子
- ✅ 文字列操作
- ✅ 型変換

### 次に学ぶべきガイド

1. **[03-control-flow.md](./03-control-flow.md)** - 条件分岐とループ

---

**次のガイド**：[03-control-flow.md](./03-control-flow.md)

**前のガイド**：[01-python-intro.md](./01-python-intro.md)

**親ガイド**：[Python Development - SKILL.md](../../SKILL.md)
