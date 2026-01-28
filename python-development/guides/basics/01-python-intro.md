# Python入門 - 完全初心者ガイド

## 目次

1. [概要](#概要)
2. [前提知識](#前提知識)
3. [Pythonとは何か](#pythonとは何か)
4. [なぜPythonを学ぶのか](#なぜpythonを学ぶのか)
5. [Pythonのインストール](#pythonのインストール)
6. [開発環境のセットアップ](#開発環境のセットアップ)
7. [最初のPythonプログラム](#最初のpythonプログラム)
8. [REPL（対話モード）の使い方](#repl対話モードの使い方)
9. [よくあるトラブルと解決策](#よくあるトラブルと解決策)
10. [演習問題](#演習問題)
11. [次のステップ](#次のステップ)

---

## 概要

### 何を学ぶか

このガイドでは、以下を学びます：
- Pythonの基本概念と特徴
- Pythonが人気の理由と活用分野
- Pythonのインストール方法
- 開発環境（VS Code + Python拡張機能）のセットアップ
- 最初のPythonプログラムの実行
- REPL（対話モード）の活用

### なぜ重要か

**Python**は、世界で最も人気のあるプログラミング言語の一つです（TIOBE Index 2024年1位）。Pythonを学ぶことで：
- **初心者に優しい**：シンプルで読みやすい文法
- **幅広い用途**：Web開発、データ分析、AI/ML、自動化など
- **豊富なライブラリ**：130万以上のパッケージ（PyPI）
- **キャリアの選択肢**：データサイエンティスト、バックエンドエンジニア、AIエンジニアなど

### 学習時間の目安

- このガイドの読了：30〜40分
- 環境構築を含めた完全理解：1〜2時間

---

## 前提知識

### 必要な知識

**なし**。このガイドは、プログラミング完全初心者を対象としています。

### 推奨環境

- **OS**：Windows 10/11、macOS 10.15以降、またはLinux
- **メモリ**：最低4GB（8GB以上推奨）
- **ストレージ**：最低2GBの空き容量

---

## Pythonとは何か

### 公式定義

Python公式サイトでは、Pythonを次のように定義しています：

> "Python is a programming language that lets you work quickly and integrate systems more effectively."
> （Pythonは、迅速に作業し、システムをより効果的に統合できるプログラミング言語です）

### より詳しい説明

Pythonは、**1991年にGuido van Rossumによって開発された汎用プログラミング言語**です。

#### 1. インタープリタ言語

Pythonは**インタープリタ言語**です。コンパイル不要で、書いたコードをすぐに実行できます。

```python
# このコードを書いて、すぐに実行できる
print("Hello, World!")
```

**コンパイル言語（C、Java等）との違い**：
- **コンパイル言語**：コード → コンパイル → 実行ファイル → 実行
- **インタープリタ言語**：コード → 実行（直接実行）

#### 2. 動的型付け言語

Pythonは**動的型付け**です。変数の型を明示的に宣言する必要がありません。

```python
# 型を宣言しない（自動的に判定される）
name = "太郎"      # 文字列
age = 25          # 整数
height = 175.5    # 浮動小数点数
is_student = True # 真偽値
```

**静的型付け言語（TypeScript、Java等）との違い**：
```typescript
// TypeScript（静的型付け）
let name: string = "太郎";
let age: number = 25;
```

```python
# Python（動的型付け）
name = "太郎"
age = 25
```

#### 3. オブジェクト指向言語

Pythonは**オブジェクト指向プログラミング**をサポートしています。

```python
class Dog:
    def __init__(self, name):
        self.name = name

    def bark(self):
        print(f"{self.name}: ワンワン！")

# オブジェクトを作成
my_dog = Dog("ポチ")
my_dog.bark()  # 出力: ポチ: ワンワン！
```

#### 4. バッテリー同梱（Batteries Included）

Pythonは**標準ライブラリが豊富**です。追加インストール不要で、多くの機能が使えます。

```python
# ファイル操作
import os
print(os.getcwd())  # 現在のディレクトリを表示

# 日付・時刻
import datetime
print(datetime.datetime.now())  # 現在時刻を表示

# HTTP通信
import urllib.request
response = urllib.request.urlopen('https://example.com')
```

---

## なぜPythonを学ぶのか

### 1. 読みやすく、書きやすい

Pythonは**可読性**を重視して設計されています。

```python
# Pythonのコード（英語のように読める）
if age >= 20:
    print("成人です")
else:
    print("未成年です")
```

```java
// Java（比較）
if (age >= 20) {
    System.out.println("成人です");
} else {
    System.out.println("未成年です");
}
```

**Pythonの特徴**：
- インデント（字下げ）でブロックを表現
- セミコロン不要
- 中括弧 `{}` 不要

### 2. 幅広い分野で活用

Pythonは、以下の分野で広く使われています：

#### Web開発
- **Django**：Instagram、Pinterest、Disqusで使用
- **Flask**：Uber、Redditで使用
- **FastAPI**：最速のPythonフレームワーク

#### データサイエンス・機械学習
- **NumPy、Pandas**：データ分析
- **Matplotlib、Seaborn**：可視化
- **scikit-learn**：機械学習
- **TensorFlow、PyTorch**：深層学習

#### 自動化・スクリプト
- **ファイル操作**：大量ファイルの一括処理
- **Webスクレイピング**：BeautifulSoup、Scrapy
- **タスク自動化**：日常業務の効率化

#### その他
- **ゲーム開発**：Pygame
- **デスクトップアプリ**：Tkinter、PyQt
- **科学計算**：SciPy、SymPy

### 3. 豊富なライブラリ・フレームワーク

**PyPI（Python Package Index）**には、130万以上のパッケージが公開されています。

```bash
# pipで簡単にインストール
pip install requests      # HTTP通信
pip install pandas        # データ分析
pip install django        # Webフレームワーク
pip install opencv-python # 画像処理
```

### 4. コミュニティが活発

- **Stack Overflow**：1,900万件以上のPython関連質問
- **GitHub**：Python関連リポジトリ数は全言語中2位
- **Python Conference（PyCon）**：世界中で開催

### 5. 求人需要が高い

2024年現在、Python関連の求人は増加傾向にあります：
- **データサイエンティスト**
- **機械学習エンジニア**
- **バックエンドエンジニア**
- **DevOpsエンジニア**

---

## Pythonのインストール

### Pythonのバージョン

2024年1月現在、Pythonには以下のバージョンがあります：
- **Python 3.12.x**：最新安定版（推奨）
- **Python 3.11.x**：安定版
- **Python 2.7.x**：**非推奨**（2020年にサポート終了）

**注意**：必ず **Python 3.x** をインストールしてください。

### インストール手順

#### Windows

1. **公式サイトにアクセス**
   - https://www.python.org/downloads/

2. **最新版をダウンロード**
   - 「Download Python 3.12.x」をクリック

3. **インストーラーを実行**
   - ✅ **重要**：「Add Python to PATH」にチェックを入れる
   - 「Install Now」をクリック

4. **インストール確認**
```bash
# コマンドプロンプトで確認
python --version
# 出力例: Python 3.12.0

pip --version
# 出力例: pip 23.3.1
```

#### macOS

**方法1：公式インストーラー（推奨）**

1. https://www.python.org/downloads/ からダウンロード
2. `.pkg`ファイルを実行
3. デフォルト設定のままインストール

**方法2：Homebrew**

```bash
# Homebrewがインストールされている場合
brew install python@3.12

# 確認
python3 --version
pip3 --version
```

#### Linux（Ubuntu/Debian）

```bash
# システムのパッケージを更新
sudo apt update

# Pythonをインストール
sudo apt install python3 python3-pip

# 確認
python3 --version
pip3 --version
```

---

## 開発環境のセットアップ

### 推奨エディタ：VS Code

**Visual Studio Code（VS Code）**は、Pythonに最適な無料エディタです。

#### 1. VS Codeのインストール

1. https://code.visualstudio.com/ にアクセス
2. ダウンロードしてインストール

#### 2. Python拡張機能のインストール

1. VS Codeを起動
2. 左サイドバーの「拡張機能」アイコンをクリック
3. 「Python」で検索
4. **Microsoft公式のPython拡張機能**をインストール

#### 3. プロジェクトフォルダの作成

```bash
# ホームディレクトリに移動
cd ~

# Pythonプロジェクト用フォルダを作成
mkdir python-learning
cd python-learning

# VS Codeで開く
code .
```

---

## 最初のPythonプログラム

### Hello, World!

1. **ファイルを作成**
   - VS Codeで `hello.py` という名前のファイルを作成

2. **コードを書く**
```python
# hello.py
print("Hello, World!")
```

3. **実行**

**ターミナルで実行**：
```bash
python hello.py
# または（macOS/Linux）
python3 hello.py
```

**出力**：
```
Hello, World!
```

**VS Codeで実行**：
- ファイルを開いた状態で右上の「▷」ボタンをクリック
- または `F5` キー

### もう少し複雑な例

```python
# greeting.py
name = input("あなたの名前を入力してください: ")
age = input("年齢を入力してください: ")

print(f"こんにちは、{name}さん！")
print(f"あなたは{age}歳ですね。")

# 年齢を数値に変換
age_number = int(age)
if age_number >= 20:
    print("成人ですね！")
else:
    years_left = 20 - age_number
    print(f"成人まであと{years_left}年です。")
```

**実行例**：
```
$ python greeting.py
あなたの名前を入力してください: 太郎
年齢を入力してください: 18
こんにちは、太郎さん！
あなたは18歳ですね。
成人まであと2年です。
```

---

## REPL（対話モード）の使い方

### REPLとは

**REPL**（Read-Eval-Print Loop）は、Pythonを対話的に実行できるモードです。

- **R**ead：入力を読み取る
- **E**val：評価（実行）する
- **P**rint：結果を表示
- **L**oop：繰り返す

### REPLの起動

```bash
# ターミナルで
python
# または
python3
```

**出力**：
```python
Python 3.12.0 (main, Oct  2 2023, 14:00:00)
[Clang 15.0.0 (clang-1500.0.40.1)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

### REPLで実験

```python
>>> 2 + 3
5

>>> name = "太郎"
>>> print(f"こんにちは、{name}さん！")
こんにちは、太郎さん！

>>> numbers = [1, 2, 3, 4, 5]
>>> sum(numbers)
15

>>> # 複数行のコード
>>> for i in range(3):
...     print(f"カウント: {i}")
...
カウント: 0
カウント: 1
カウント: 2

>>> # 終了
>>> exit()
```

**REPLの利点**：
- **即座に実験できる**：コードを書いて即座に結果を確認
- **学習に最適**：新しい機能を試すのに便利
- **計算機として使える**：簡単な計算に便利

---

## よくあるトラブルと解決策

### トラブル1：`python: command not found`

**症状**：
```bash
$ python --version
python: command not found
```

**原因**：
- Pythonがインストールされていない
- PATHが通っていない

**解決策**：

**Windows**：
1. Pythonを再インストール
2. 「Add Python to PATH」にチェックを入れる

**macOS/Linux**：
```bash
# python3を使う
python3 --version

# またはエイリアスを設定
echo 'alias python=python3' >> ~/.bashrc
source ~/.bashrc
```

### トラブル2：`ModuleNotFoundError`

**症状**：
```python
>>> import requests
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'requests'
```

**原因**：
- パッケージがインストールされていない

**解決策**：
```bash
pip install requests
# または
pip3 install requests
```

### トラブル3：インデントエラー

**症状**：
```python
if age >= 20:
print("成人です")  # インデントがない
```

**エラー**：
```
IndentationError: expected an indented block
```

**解決策**：
```python
if age >= 20:
    print("成人です")  # 4スペースのインデント
```

**Pythonのインデントルール**：
- タブまたは4スペース（4スペース推奨）
- 混在させない（タブとスペースを混ぜない）

### トラブル4：文字エンコーディング問題

**症状**：
日本語が文字化けする

**解決策**：
```python
# ファイルの先頭に追加
# -*- coding: utf-8 -*-

print("こんにちは")
```

---

## 演習問題

### 問題1：自己紹介プログラム

**難易度**：初級

**課題**：
以下の情報を表示するプログラムを作成してください。
- 自分の名前
- 年齢
- 好きな食べ物

**解答例**：
```python
# self_intro.py
name = "山田太郎"
age = 25
favorite_food = "カレー"

print("【自己紹介】")
print(f"名前: {name}")
print(f"年齢: {age}歳")
print(f"好きな食べ物: {favorite_food}")
```

### 問題2：簡単な計算機

**難易度**：初級〜中級

**課題**：
2つの数値を入力して、四則演算の結果を表示するプログラムを作成してください。

**解答例**：
```python
# calculator.py
print("=== 簡単な計算機 ===")

# 入力を受け取る
num1 = float(input("1つ目の数値を入力: "))
num2 = float(input("2つ目の数値を入力: "))

# 計算
addition = num1 + num2
subtraction = num1 - num2
multiplication = num1 * num2
division = num1 / num2 if num2 != 0 else "エラー（0で割れません）"

# 結果を表示
print(f"\n【計算結果】")
print(f"{num1} + {num2} = {addition}")
print(f"{num1} - {num2} = {subtraction}")
print(f"{num1} × {num2} = {multiplication}")
print(f"{num1} ÷ {num2} = {division}")
```

---

## 次のステップ

### このガイドで学んだこと

- ✅ Pythonの基本概念と特徴
- ✅ Pythonのインストール方法
- ✅ 開発環境（VS Code）のセットアップ
- ✅ 最初のPythonプログラムの実行
- ✅ REPL（対話モード）の活用

### 次に学ぶべきガイド

1. **[02-basic-syntax.md](./02-basic-syntax.md)**
   - 変数と型
   - 演算子
   - 文字列操作

2. **[03-control-flow.md](./03-control-flow.md)**
   - if文（条件分岐）
   - for/whileループ

### 関連リソース

#### 公式ドキュメント
- [Python.org](https://www.python.org/)
- [Python日本語ドキュメント](https://docs.python.org/ja/3/)
- [Python チュートリアル](https://docs.python.org/ja/3/tutorial/index.html)

#### 推奨書籍
- 「Python 1年生」（翔泳社）：完全初心者向け
- 「退屈なことはPythonにやらせよう」（オライリー）：自動化入門

#### オンラインリソース
- [PyPI（Python Package Index）](https://pypi.org/)
- [Real Python](https://realpython.com/)：実践的チュートリアル

---

**次のガイド**：[02-basic-syntax.md](./02-basic-syntax.md)

**前のガイド**：なし（これが最初のガイドです）

**親ガイド**：[Python Development - SKILL.md](../../SKILL.md)
