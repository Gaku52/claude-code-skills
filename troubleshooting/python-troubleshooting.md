# Python トラブルシューティング

## 目次

1. [概要](#概要)
2. [インストール・環境構築エラー](#インストール環境構築エラー)
3. [構文・インデントエラー](#構文インデントエラー)
4. [型・値エラー](#型値エラー)
5. [モジュール・インポートエラー](#モジュールインポートエラー)
6. [ファイル・パスエラー](#ファイルパスエラー)
7. [Web フレームワークエラー](#webフレームワークエラー)
8. [データベースエラー](#データベースエラー)

---

## 概要

このガイドは、Python開発で頻繁に遭遇するエラーと解決策をまとめたトラブルシューティングデータベースです。

**収録エラー数:** 25個

**対象バージョン:** Python 3.9 ~ 3.12

---

## インストール・環境構築エラー

### ❌ エラー1: python: command not found

```
bash: python: command not found
```

**原因:**
- Pythonがインストールされていない
- `python`コマンドではなく`python3`を使用する必要がある

**解決策:**

```bash
# macOS/Linux: python3を使用
python3 --version

# エイリアスを設定（~/.zshrc または ~/.bashrc）
alias python=python3
alias pip=pip3

# または pyenv で管理（推奨）
curl https://pyenv.run | bash

# Python 3.11 をインストール
pyenv install 3.11.0
pyenv global 3.11.0

# 確認
python --version
```

**Windows:**

```bash
# Python公式サイトからインストーラーをダウンロード
# https://www.python.org/downloads/

# インストール後
python --version
pip --version
```

---

### ❌ エラー2: ModuleNotFoundError: No module named 'pip'

```
ModuleNotFoundError: No module named 'pip'
```

**原因:**
- pipがインストールされていない

**解決策:**

```bash
# macOS/Linux
python3 -m ensurepip --upgrade

# または get-pip.py を使用
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py

# 確認
pip3 --version
```

---

### ❌ エラー3: error: externally-managed-environment

```
error: externally-managed-environment

× This environment is externally managed
```

**原因:**
- Python 3.11以降、システムのPythonに直接パッケージをインストールできない

**解決策:**

```bash
# ✅ 仮想環境を作成（推奨）
python3 -m venv venv

# 仮想環境を有効化
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

# パッケージをインストール
pip install requests

# 仮想環境を無効化
deactivate
```

**または pipx を使用（CLIツール）:**

```bash
# pipxをインストール
python3 -m pip install --user pipx
python3 -m pipx ensurepath

# CLIツールをインストール
pipx install black
pipx install ruff
```

---

### ❌ エラー4: pip install fails with permission denied

```
ERROR: Could not install packages due to an OSError: [Errno 13] Permission denied
```

**原因:**
- システムディレクトリへの書き込み権限がない

**解決策:**

```bash
# ❌ sudoは使わない（非推奨）
# sudo pip install package

# ✅ ユーザーディレクトリにインストール
pip install --user package

# ✅ 仮想環境を使用（推奨）
python3 -m venv venv
source venv/bin/activate
pip install package
```

---

### ❌ エラー5: ImportError: cannot import name 'X' from partially initialized module

```
ImportError: cannot import name 'calculate' from partially initialized module 'utils' (most likely due to a circular import)
```

**原因:**
- 循環インポート

**間違った例:**

```python
# ❌ utils.py
from main import app

def calculate():
    return 42

# ❌ main.py
from utils import calculate

app = "MyApp"
```

**解決策:**

```python
# ✅ utils.py
def calculate():
    return 42

# ✅ main.py
from utils import calculate

app = "MyApp"

# ✅ または関数内でインポート
def some_function():
    from utils import calculate
    return calculate()
```

---

## 構文・インデントエラー

### ❌ エラー6: IndentationError: expected an indented block

```
IndentationError: expected an indented block
```

**原因:**
- インデントが不足している
- タブとスペースが混在している

**間違った例:**

```python
# ❌ インデント不足
def greet():
print("Hello")  # エラー

# ❌ タブとスペースの混在
def calculate():
    x = 1  # スペース4つ
	y = 2  # タブ（エラー）
    return x + y
```

**解決策:**

```python
# ✅ 正しいインデント（スペース4つ）
def greet():
    print("Hello")

# ✅ if文のインデント
if True:
    print("True")
else:
    print("False")

# ✅ ネストしたインデント
def calculate():
    x = 1
    if x > 0:
        return x * 2
    return 0
```

**VS Code設定（.editorconfig）:**

```ini
[*.py]
indent_style = space
indent_size = 4
```

---

### ❌ エラー7: SyntaxError: invalid syntax

```
SyntaxError: invalid syntax
```

**原因:**
- 括弧の閉じ忘れ
- コロンの忘れ
- Python 2とPython 3の混在

**間違った例:**

```python
# ❌ コロン忘れ
if x > 0
    print("Positive")

# ❌ 括弧の閉じ忘れ
result = calculate(1, 2

# ❌ Python 2のprint文（Python 3では関数）
print "Hello"  # Python 2
```

**解決策:**

```python
# ✅ コロンを追加
if x > 0:
    print("Positive")

# ✅ 括弧を閉じる
result = calculate(1, 2)

# ✅ Python 3のprint関数
print("Hello")

# ✅ 複数行の括弧
result = calculate(
    arg1=1,
    arg2=2,
    arg3=3
)
```

---

### ❌ エラー8: SyntaxError: f-string expression part cannot include a backslash

```
SyntaxError: f-string expression part cannot include a backslash
```

**原因:**
- f-string内でバックスラッシュを使用している

**間違った例:**

```python
# ❌ f-string内でバックスラッシュ
name = "John"
print(f"Hello {name.replace('o', '\n')}")  # エラー
```

**解決策:**

```python
# ✅ 変数に代入
name = "John"
newline = '\n'
print(f"Hello {name.replace('o', newline)}")

# ✅ または関数を使用
def replace_with_newline(text):
    return text.replace('o', '\n')

print(f"Hello {replace_with_newline(name)}")

# ✅ または通常の文字列フォーマット
print("Hello {}".format(name.replace('o', '\n')))
```

---

## 型・値エラー

### ❌ エラー9: TypeError: 'NoneType' object is not iterable

```
TypeError: 'NoneType' object is not iterable
```

**原因:**
- `None`をイテレート（ループ）しようとしている

**間違った例:**

```python
# ❌ 関数がNoneを返している
def get_users():
    pass  # Noneを返す

for user in get_users():  # エラー
    print(user)
```

**解決策:**

```python
# ✅ 空リストを返す
def get_users():
    return []  # または実際のデータ

for user in get_users():
    print(user)

# ✅ Noneチェック
users = get_users()
if users is not None:
    for user in users:
        print(user)

# ✅ デフォルト値を使用
users = get_users() or []
for user in users:
    print(user)
```

---

### ❌ エラー10: TypeError: unsupported operand type(s) for +: 'int' and 'str'

```
TypeError: unsupported operand type(s) for +: 'int' and 'str'
```

**原因:**
- 異なる型同士の演算

**間違った例:**

```python
# ❌ intとstrの加算
age = 25
message = "Age: " + age  # エラー

# ❌ リストとintの加算
numbers = [1, 2, 3]
result = numbers + 4  # エラー
```

**解決策:**

```python
# ✅ 型変換
age = 25
message = "Age: " + str(age)

# ✅ f-string（推奨）
message = f"Age: {age}"

# ✅ リストにappend
numbers = [1, 2, 3]
numbers.append(4)

# ✅ または連結
numbers = [1, 2, 3]
result = numbers + [4]
```

---

### ❌ エラー11: ValueError: invalid literal for int() with base 10

```
ValueError: invalid literal for int() with base 10: 'abc'
```

**原因:**
- 数値に変換できない文字列をintに変換しようとしている

**解決策:**

```python
# ❌ 数値でない文字列
number = int("abc")  # エラー

# ✅ エラーハンドリング
def safe_int(value, default=0):
    try:
        return int(value)
    except ValueError:
        return default

number = safe_int("abc", 0)  # 0

# ✅ 数値チェック
def parse_int(value):
    if value.isdigit():
        return int(value)
    else:
        raise ValueError(f"Cannot convert '{value}' to int")

# ✅ 正規表現で検証
import re

def is_numeric(value):
    return re.match(r'^-?\d+$', value) is not None

if is_numeric("123"):
    number = int("123")
```

---

### ❌ エラー12: KeyError: 'key_name'

```
KeyError: 'email'
```

**原因:**
- 辞書に存在しないキーにアクセスしている

**間違った例:**

```python
# ❌ キーが存在しない
user = {"name": "John"}
email = user["email"]  # エラー
```

**解決策:**

```python
# ✅ get() メソッドを使用
user = {"name": "John"}
email = user.get("email")  # None
email = user.get("email", "default@example.com")  # デフォルト値

# ✅ キーの存在確認
if "email" in user:
    email = user["email"]
else:
    email = None

# ✅ try-except
try:
    email = user["email"]
except KeyError:
    email = None

# ✅ defaultdict を使用
from collections import defaultdict

user = defaultdict(lambda: None)
user["name"] = "John"
email = user["email"]  # None
```

---

### ❌ エラー13: IndexError: list index out of range

```
IndexError: list index out of range
```

**原因:**
- リストの範囲外のインデックスにアクセスしている

**解決策:**

```python
# ❌ 範囲外アクセス
numbers = [1, 2, 3]
fourth = numbers[3]  # エラー（インデックスは0-2）

# ✅ 範囲チェック
numbers = [1, 2, 3]
if len(numbers) > 3:
    fourth = numbers[3]
else:
    fourth = None

# ✅ try-except
try:
    fourth = numbers[3]
except IndexError:
    fourth = None

# ✅ get() 関数を自作
def safe_get(lst, index, default=None):
    try:
        return lst[index]
    except IndexError:
        return default

fourth = safe_get(numbers, 3)  # None
```

---

## モジュール・インポートエラー

### ❌ エラー14: ModuleNotFoundError: No module named 'requests'

```
ModuleNotFoundError: No module named 'requests'
```

**原因:**
- パッケージがインストールされていない

**解決策:**

```bash
# パッケージをインストール
pip install requests

# requirements.txtから一括インストール
pip install -r requirements.txt

# 仮想環境が有効化されているか確認
which python
which pip

# インストール済みパッケージ確認
pip list
pip show requests
```

**requirements.txt:**

```
requests==2.31.0
flask==3.0.0
sqlalchemy==2.0.23
```

---

### ❌ エラー15: ImportError: attempted relative import with no known parent package

```
ImportError: attempted relative import with no known parent package
```

**原因:**
- パッケージとして実行されていない状態で相対インポートを使用

**間違った例:**

```python
# ❌ utils.py をスクリプトとして実行
# python utils.py

# utils.py
from .config import settings  # エラー
```

**解決策:**

```python
# ✅ プロジェクト構造
# myproject/
#   __init__.py
#   main.py
#   utils.py
#   config.py

# utils.py（絶対インポート）
from myproject.config import settings

# または実行方法を変更
python -m myproject.utils  # モジュールとして実行

# main.py
from .utils import helper_function  # 相対インポート可能
from .config import settings
```

---

### ❌ エラー16: ImportError: cannot import name 'X' from 'Y'

```
ImportError: cannot import name 'Flask' from 'flask'
```

**原因:**
- ファイル名がモジュール名と同じ
- パッケージのバージョンが古い

**解決策:**

```bash
# ファイル名を確認
ls flask.py  # ← これがあるとflaskモジュールと競合

# ファイル名を変更
mv flask.py app.py

# パッケージを更新
pip install --upgrade flask

# キャッシュをクリア
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

**コード例:**

```python
# ✅ 正しいインポート
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/health')
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True)
```

---

## ファイル・パスエラー

### ❌ エラー17: FileNotFoundError: [Errno 2] No such file or directory

```
FileNotFoundError: [Errno 2] No such file or directory: 'data.txt'
```

**原因:**
- ファイルが存在しない
- 相対パスが間違っている

**解決策:**

```python
# ❌ 相対パス（実行場所に依存）
with open('data.txt') as f:
    content = f.read()

# ✅ 絶対パス
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, 'data.txt')

with open(file_path) as f:
    content = f.read()

# ✅ pathlib を使用（推奨）
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / 'data.txt'

if file_path.exists():
    content = file_path.read_text()
else:
    print(f"File not found: {file_path}")

# ✅ try-except
try:
    with open('data.txt') as f:
        content = f.read()
except FileNotFoundError:
    print("File does not exist")
    content = ""
```

---

### ❌ エラー18: PermissionError: [Errno 13] Permission denied

```
PermissionError: [Errno 13] Permission denied: 'file.txt'
```

**原因:**
- ファイルの読み書き権限がない

**解決策:**

```bash
# ファイル権限を確認
ls -la file.txt

# 権限を変更
chmod 644 file.txt

# 所有者を変更
sudo chown $USER:$USER file.txt
```

**Python コード:**

```python
# ✅ 権限チェック
import os

file_path = 'file.txt'

if os.access(file_path, os.R_OK):
    with open(file_path) as f:
        content = f.read()
else:
    print(f"No read permission for {file_path}")

# ✅ try-except
try:
    with open(file_path, 'w') as f:
        f.write("Hello")
except PermissionError:
    print(f"Cannot write to {file_path}")
```

---

### ❌ エラー19: UnicodeDecodeError: 'utf-8' codec can't decode byte

```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x82 in position 0: invalid start byte
```

**原因:**
- ファイルのエンコーディングがUTF-8でない

**解決策:**

```python
# ❌ エンコーディング指定なし
with open('file.txt') as f:
    content = f.read()  # エラー

# ✅ エンコーディングを指定
with open('file.txt', encoding='shift-jis') as f:
    content = f.read()

# ✅ エラーを無視
with open('file.txt', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# ✅ chardet でエンコーディング検出
import chardet

with open('file.txt', 'rb') as f:
    raw_data = f.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']

with open('file.txt', encoding=encoding) as f:
    content = f.read()
```

---

## Web フレームワークエラー

### ❌ エラー20: Flask - RuntimeError: Working outside of application context

```
RuntimeError: Working outside of application context.
This typically means that you attempted to use functionality that needed
to interface with the current application object in some way.
```

**原因:**
- アプリケーションコンテキスト外でFlaskの機能を使用

**解決策:**

```python
# ❌ アプリケーションコンテキスト外
from flask import Flask, current_app

app = Flask(__name__)

print(current_app.config)  # エラー

# ✅ アプリケーションコンテキスト内
from flask import Flask, current_app

app = Flask(__name__)

with app.app_context():
    print(current_app.config)

# ✅ デコレーター付き関数内
@app.route('/config')
def get_config():
    return current_app.config  # OK
```

---

### ❌ エラー21: FastAPI - 422 Unprocessable Entity (Validation Error)

```
{
  "detail": [
    {
      "loc": ["body", "email"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**原因:**
- リクエストボディのバリデーションエラー

**解決策:**

```python
# ✅ Pydanticモデルを定義
from fastapi import FastAPI
from pydantic import BaseModel, EmailStr, Field

app = FastAPI()

class UserCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    age: int = Field(..., ge=0, le=150)

@app.post('/users')
async def create_user(user: UserCreate):
    return {"message": f"User {user.name} created"}

# ✅ Optionalフィールド
from typing import Optional

class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    age: Optional[int] = None

@app.put('/users/{user_id}')
async def update_user(user_id: int, user: UserUpdate):
    return {"message": f"User {user_id} updated"}
```

**テストリクエスト:**

```bash
# ✅ 正しいリクエスト
curl -X POST http://localhost:8000/users \
  -H "Content-Type: application/json" \
  -d '{"name": "John", "email": "john@example.com", "age": 25}'

# ❌ バリデーションエラー（emailがない）
curl -X POST http://localhost:8000/users \
  -H "Content-Type: application/json" \
  -d '{"name": "John", "age": 25}'
```

---

### ❌ エラー22: Django - ImproperlyConfigured: Requested setting DATABASES

```
django.core.exceptions.ImproperlyConfigured: Requested setting DATABASES, but settings are not configured.
```

**原因:**
- Djangoの設定が読み込まれていない

**解決策:**

```python
# ✅ manage.py経由で実行
python manage.py runserver

# ✅ スクリプトでDjango設定を読み込み
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
django.setup()

# この後にDjangoのモデルなどを使用可能
from myapp.models import User
users = User.objects.all()
```

**settings.py:**

```python
# ✅ 環境変数から設定を読み込む
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')

DEBUG = os.getenv('DEBUG', 'False') == 'True'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('DB_NAME', 'mydb'),
        'USER': os.getenv('DB_USER', 'postgres'),
        'PASSWORD': os.getenv('DB_PASSWORD', 'password'),
        'HOST': os.getenv('DB_HOST', 'localhost'),
        'PORT': os.getenv('DB_PORT', '5432'),
    }
}
```

---

## データベースエラー

### ❌ エラー23: SQLAlchemy - OperationalError: (sqlite3.OperationalError) no such table

```
sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) no such table: users
```

**原因:**
- テーブルが作成されていない

**解決策:**

```python
# ✅ SQLAlchemyでテーブル作成
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)

# データベース接続
engine = create_engine('sqlite:///app.db')

# テーブル作成
Base.metadata.create_all(engine)

# セッション作成
Session = sessionmaker(bind=engine)
session = Session()

# ユーザー作成
new_user = User(name="John", email="john@example.com")
session.add(new_user)
session.commit()
```

**Alembicでマイグレーション:**

```bash
# Alembicをインストール
pip install alembic

# 初期化
alembic init alembic

# マイグレーションファイル作成
alembic revision --autogenerate -m "Create users table"

# マイグレーション実行
alembic upgrade head
```

---

### ❌ エラー24: psycopg2.OperationalError: could not connect to server

```
psycopg2.OperationalError: could not connect to server: Connection refused
    Is the server running on host "localhost" (127.0.0.1) and accepting
    TCP/IP connections on port 5432?
```

**原因:**
- PostgreSQLサーバーが起動していない

**解決策:**

```bash
# PostgreSQL起動（macOS Homebrew）
brew services start postgresql

# Linux
sudo systemctl start postgresql

# Docker
docker run -d \
  --name postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=mydb \
  -p 5432:5432 \
  postgres:15

# 接続確認
psql -h localhost -U postgres -d mydb
```

**Python コード:**

```python
# ✅ 接続エラーハンドリング
import psycopg2
from psycopg2 import OperationalError

def create_connection():
    try:
        connection = psycopg2.connect(
            host="localhost",
            port=5432,
            database="mydb",
            user="postgres",
            password="password"
        )
        print("PostgreSQL connection successful")
        return connection
    except OperationalError as e:
        print(f"The error '{e}' occurred")
        return None

connection = create_connection()
```

**SQLAlchemy:**

```python
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

DATABASE_URL = "postgresql://postgres:password@localhost:5432/mydb"

try:
    engine = create_engine(DATABASE_URL)
    connection = engine.connect()
    print("Database connected successfully")
except OperationalError as e:
    print(f"Database connection failed: {e}")
```

---

### ❌ エラー25: pymongo.errors.ServerSelectionTimeoutError

```
pymongo.errors.ServerSelectionTimeoutError: localhost:27017: [Errno 61] Connection refused
```

**原因:**
- MongoDBサーバーが起動していない

**解決策:**

```bash
# MongoDB起動（macOS Homebrew）
brew services start mongodb-community

# Linux
sudo systemctl start mongod

# Docker
docker run -d \
  --name mongodb \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=password \
  -p 27017:27017 \
  mongo:6
```

**Python コード:**

```python
# ✅ MongoDBクライアント
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

MONGODB_URI = "mongodb://admin:password@localhost:27017/"

try:
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    # 接続テスト
    client.server_info()
    print("MongoDB connected successfully")

    db = client['mydb']
    collection = db['users']

    # データ挿入
    user = {"name": "John", "email": "john@example.com"}
    result = collection.insert_one(user)
    print(f"User inserted with ID: {result.inserted_id}")

except ServerSelectionTimeoutError as e:
    print(f"MongoDB connection failed: {e}")
finally:
    client.close()
```

---

## まとめ

### このガイドで学んだこと

- Python開発における25の頻出エラー
- 各エラーの原因と解決策
- ベストプラクティス

### エラー解決の基本手順

1. **エラーメッセージを読む** - 最後の行だけでなくトレースバック全体を確認
2. **トレースバックを確認** - エラーが発生したファイル・行番号
3. **公式ドキュメントを確認** - [Python Docs](https://docs.python.org/3/)
4. **このガイドで検索** - よくあるエラーはここに記載
5. **仮想環境を確認** - `which python`, `which pip`でパスを確認

### デバッグツール

```python
# pdb（Python Debugger）
import pdb; pdb.set_trace()

# breakpoint()（Python 3.7+）
breakpoint()

# ロギング
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug("This is a debug message")
logger.info("This is an info message")
logger.error("This is an error message")
```

### さらに学ぶ

- **[Python公式ドキュメント](https://docs.python.org/3/)**
- **[FastAPI公式ドキュメント](https://fastapi.tiangolo.com/)**
- **[Django公式ドキュメント](https://docs.djangoproject.com/)**

---

**関連ガイド:**
- [Python Development - 基礎ガイド](../python-development/SKILL.md)
- [Backend Development - バックエンド開発](../backend-development/SKILL.md)

**親ガイド:** [トラブルシューティングDB](./README.md)
