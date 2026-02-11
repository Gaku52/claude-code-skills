# モジュールとパッケージ - 完全初心者ガイド

## 目次

1. [概要](#概要)
2. [モジュールとは](#モジュールとは)
3. [標準ライブラリ](#標準ライブラリ)
4. [pipとパッケージ管理](#pipとパッケージ管理)
5. [仮想環境](#仮想環境)
6. [演習問題](#演習問題)
7. [次のステップ](#次のステップ)

---

## 概要

### 何を学ぶか

- モジュールのインポート
- 標準ライブラリの活用
- pipでのパッケージインストール
- 仮想環境の作成と使用

### 学習時間：1〜2時間

---

## モジュールとは

### モジュールのインポート

```python
# モジュール全体をインポート
import math
print(math.pi)       # 3.141592...
print(math.sqrt(16)) # 4.0

# 特定の関数をインポート
from math import pi, sqrt
print(pi)
print(sqrt(16))

# 別名をつける
import math as m
print(m.pi)

# すべてをインポート（非推奨）
from math import *
```

### 自作モジュール

```python
# mymodule.py
def greet(name):
    return f"こんにちは、{name}さん！"

PI = 3.14159

# main.py
import mymodule
print(mymodule.greet("太郎"))
print(mymodule.PI)
```

---

## 標準ライブラリ

### よく使う標準ライブラリ

#### os（OS操作）

```python
import os

# 現在のディレクトリ
print(os.getcwd())

# ディレクトリ内のファイル一覧
print(os.listdir("."))

# パス結合
path = os.path.join("folder", "file.txt")
```

#### datetime（日付・時刻）

```python
from datetime import datetime, timedelta

# 現在時刻
now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

# 日付の計算
tomorrow = now + timedelta(days=1)
print(tomorrow)
```

#### random（乱数）

```python
import random

# ランダムな整数
print(random.randint(1, 10))

# リストからランダムに選択
fruits = ["りんご", "バナナ", "オレンジ"]
print(random.choice(fruits))

# シャッフル
random.shuffle(fruits)
```

#### json（JSON操作）

```python
import json

# 辞書 → JSON文字列
data = {"name": "太郎", "age": 25}
json_str = json.dumps(data, ensure_ascii=False)

# JSON文字列 → 辞書
data2 = json.loads(json_str)
```

---

## pipとパッケージ管理

### pipの基本コマンド

```bash
# パッケージのインストール
pip install requests

# 特定バージョンをインストール
pip install requests==2.28.0

# 複数パッケージをインストール
pip install requests pandas numpy

# インストール済みパッケージの確認
pip list

# パッケージのアンインストール
pip uninstall requests

# パッケージ情報の表示
pip show requests

# アップグレード
pip install --upgrade requests
```

### requirements.txt

```bash
# インストール済みパッケージを保存
pip freeze > requirements.txt

# requirements.txtからインストール
pip install -r requirements.txt
```

**requirements.txtの例**：
```
requests==2.28.0
pandas==1.5.0
numpy==1.23.0
```

---

## 仮想環境

### なぜ仮想環境が必要か

- プロジェクトごとに異なるバージョンのパッケージを使える
- システム全体に影響を与えない
- requirements.txtで環境を再現できる

### venvの使い方

```bash
# 仮想環境の作成
python -m venv myenv

# 仮想環境の有効化
# Windows
myenv\Scripts\activate

# macOS/Linux
source myenv/bin/activate

# パッケージのインストール（仮想環境内）
pip install requests

# 仮想環境の無効化
deactivate
```

### プロジェクト構造の例

```
my_project/
├── myenv/              # 仮想環境（.gitignoreに追加）
├── src/
│   └── main.py
├── requirements.txt
└── README.md
```

---

## 演習問題

### 問題1：ファイル操作

```python
import os

# 現在のディレクトリ内の.pyファイルを一覧表示
for file in os.listdir("."):
    if file.endswith(".py"):
        print(file)
```

### 問題2：日付計算

```python
from datetime import datetime, timedelta

# 100日後の日付を計算
today = datetime.now()
future = today + timedelta(days=100)
print(f"100日後: {future.strftime('%Y年%m月%d日')}")
```

### 問題3：JSONファイルの読み書き

```python
import json

# データの準備
users = [
    {"name": "太郎", "age": 25},
    {"name": "花子", "age": 30}
]

# JSONファイルに保存
with open("users.json", "w", encoding="utf-8") as f:
    json.dump(users, f, ensure_ascii=False, indent=2)

# JSONファイルから読み込み
with open("users.json", "r", encoding="utf-8") as f:
    loaded_users = json.load(f)
    print(loaded_users)
```

---

## 次のステップ

### このガイドで学んだこと

- ✅ モジュールのインポート
- ✅ 標準ライブラリの活用
- ✅ pipでのパッケージ管理
- ✅ 仮想環境の作成

### おめでとうございます！

Python基礎ガイドを完了しました。

### 次に学ぶべきこと

1. **オブジェクト指向プログラミング**
   - クラスとオブジェクト
   - 継承、カプセル化

2. **ファイルI/O**
   - ファイルの読み書き
   - CSVファイルの操作

3. **エラー処理**
   - try-except
   - カスタム例外

4. **実践プロジェクト**
   - Webスクレイピング
   - データ分析
   - Web APIの利用

### 関連リソース

- [Python公式ドキュメント](https://docs.python.org/ja/3/)
- [Real Python](https://realpython.com/)
- [PyPI](https://pypi.org/)

---

**前のガイド**：[05-data-structures.md](./05-data-structures.md)

**親ガイド**：[Python Development - SKILL.md](../../SKILL.md)
