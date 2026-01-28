# 環境変数と設定管理 - 完全初心者ガイド

## 目次

1. [概要](#概要)
2. [環境変数とは](#環境変数とは)
3. [なぜ環境変数が必要か](#なぜ環境変数が必要か)
4. [.envファイル](#envファイル)
5. [python-dotenvの使い方](#python-dotenvの使い方)
6. [設定管理のベストプラクティス](#設定管理のベストプラクティス)
7. [実装例](#実装例)
8. [次のステップ](#次のステップ)

---

## 概要

### 何を学ぶか

- 環境変数の基本概念
- .envファイルの使い方
- python-dotenvによる環境変数管理
- 本番環境と開発環境の設定分離
- セキュアな設定管理

### なぜ重要か

環境変数を適切に管理することで、機密情報をコードに含めず、異なる環境（開発・ステージング・本番）で同じコードを使い回せます。

### 学習時間：40〜50分

---

## 環境変数とは

### 定義

**環境変数**は、オペレーティングシステムが提供する動的な値のことです。プログラムが実行時に参照できます。

### 確認方法

```bash
# macOS/Linux
echo $PATH
echo $HOME

# Windows
echo %PATH%
echo %USERPROFILE%
```

### Pythonで環境変数を読む

```python
import os

# 環境変数を取得
path = os.environ["PATH"]
print(path)

# 存在しない場合のデフォルト値
db_host = os.environ.get("DB_HOST", "localhost")
print(db_host)  # 環境変数がなければ "localhost"
```

---

## なぜ環境変数が必要か

### 1. 機密情報の保護

**❌ 悪い例：コードに直接書く**

```python
# ❌ GitHubに公開される！
DATABASE_URL = "postgresql://admin:password123@db.example.com/mydb"
SECRET_KEY = "super-secret-key-12345"
```

**✅ 良い例：環境変数を使う**

```python
import os

DATABASE_URL = os.environ["DATABASE_URL"]
SECRET_KEY = os.environ["SECRET_KEY"]
```

### 2. 環境ごとの設定切り替え

```python
# 開発環境
DATABASE_URL = "sqlite:///./dev.db"
DEBUG = True

# 本番環境
DATABASE_URL = "postgresql://user:pass@prod-server/db"
DEBUG = False
```

環境変数を使えば、コードを変えずに設定を切り替えられます。

### 3. デプロイの柔軟性

```bash
# ローカル開発
export DATABASE_URL="sqlite:///./test.db"
python app.py

# 本番環境（Heroku、AWS等）
# 管理画面で環境変数を設定するだけ
```

---

## .envファイル

### .envファイルとは

**.env**は、環境変数を定義するファイルです。

```bash
# .env
DATABASE_URL=postgresql://user:password@localhost/mydb
SECRET_KEY=your-secret-key-here
DEBUG=True
PORT=8000
```

### .envファイルのメリット

- ローカル開発が簡単
- チームメンバーと設定を共有しやすい（.env.exampleを使う）
- Gitに含めない（.gitignoreに追加）

### .gitignoreに追加

```bash
# .gitignore
.env
.env.local
.env.*.local
```

### .env.exampleを作る

```bash
# .env.example（Gitにコミット可能）
DATABASE_URL=postgresql://user:password@localhost/dbname
SECRET_KEY=your-secret-key
DEBUG=True
PORT=8000
```

チームメンバーは`.env.example`をコピーして自分の`.env`を作ります：

```bash
cp .env.example .env
# .envを編集して自分の設定を入れる
```

---

## python-dotenvの使い方

### インストール

```bash
pip install python-dotenv
```

### 基本的な使い方

```python
from dotenv import load_dotenv
import os

# .envファイルを読み込む
load_dotenv()

# 環境変数を取得
database_url = os.environ["DATABASE_URL"]
secret_key = os.environ["SECRET_KEY"]
debug = os.environ.get("DEBUG", "False") == "True"

print(f"Database: {database_url}")
print(f"Debug mode: {debug}")
```

### 特定のファイルを指定

```python
from dotenv import load_dotenv

# 本番用の設定
load_dotenv(".env.production")

# テスト用の設定
load_dotenv(".env.test")
```

### 既存の環境変数を上書きしない

```python
# 既に設定されている環境変数は上書きしない
load_dotenv(override=False)
```

---

## 設定管理のベストプラクティス

### 1. Pydantic Settingsを使う

```python
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    database_url: str = Field(..., alias="DATABASE_URL")
    secret_key: str = Field(..., alias="SECRET_KEY")
    debug: bool = Field(False, alias="DEBUG")
    port: int = Field(8000, alias="PORT")

    class Config:
        env_file = ".env"
        case_sensitive = False

# 使用例
settings = Settings()
print(settings.database_url)
print(settings.debug)
```

**利点**：
- 型チェック
- バリデーション
- デフォルト値
- 自動的に.envを読み込む

### 2. 環境ごとの設定ファイル

```python
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    secret_key: str
    debug: bool = False

    class Config:
        # 環境に応じて.envファイルを切り替え
        env = os.environ.get("ENV", "development")
        env_file = f".env.{env}"

# 使用例
# ENV=production python app.py  → .env.production を読む
# ENV=test python app.py        → .env.test を読む
```

### 3. 必須の環境変数をチェック

```python
import os
import sys

REQUIRED_ENV_VARS = ["DATABASE_URL", "SECRET_KEY"]

def check_env_vars():
    missing = [var for var in REQUIRED_ENV_VARS if var not in os.environ]
    if missing:
        print(f"エラー: 以下の環境変数が設定されていません: {', '.join(missing)}")
        sys.exit(1)

# アプリケーション起動時にチェック
check_env_vars()
```

---

## 実装例

### FastAPIでの環境変数管理

```python
from fastapi import FastAPI
from pydantic_settings import BaseSettings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 設定クラス
class Settings(BaseSettings):
    app_name: str = "My API"
    database_url: str
    secret_key: str
    debug: bool = False
    cors_origins: list[str] = ["http://localhost:3000"]

    class Config:
        env_file = ".env"

settings = Settings()

# FastAPIアプリケーション
app = FastAPI(
    title=settings.app_name,
    debug=settings.debug
)

# データベース接続
engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine)

@app.get("/")
def read_root():
    return {
        "app_name": settings.app_name,
        "debug": settings.debug
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}
```

### .envファイル（ローカル開発）

```bash
# .env
APP_NAME="My API (Development)"
DATABASE_URL=sqlite:///./dev.db
SECRET_KEY=dev-secret-key-change-in-production
DEBUG=True
CORS_ORIGINS=["http://localhost:3000", "http://localhost:5173"]
```

### 環境別の設定例

#### 開発環境（.env.development）

```bash
APP_NAME="My API (Dev)"
DATABASE_URL=sqlite:///./dev.db
DEBUG=True
LOG_LEVEL=DEBUG
```

#### ステージング環境（.env.staging）

```bash
APP_NAME="My API (Staging)"
DATABASE_URL=postgresql://user:pass@staging-db.example.com/db
DEBUG=False
LOG_LEVEL=INFO
```

#### 本番環境（.env.production）

```bash
APP_NAME="My API"
DATABASE_URL=postgresql://user:pass@prod-db.example.com/db
DEBUG=False
LOG_LEVEL=WARNING
SENTRY_DSN=https://xxx@sentry.io/xxx
```

---

## セキュリティのベストプラクティス

### 1. .envをGitに含めない

```bash
# .gitignore
.env
.env.local
.env.*.local
*.env
```

### 2. .env.exampleを用意

```bash
# .env.example
DATABASE_URL=postgresql://user:password@localhost/dbname
SECRET_KEY=change-this-to-a-random-secret-key
DEBUG=False
PORT=8000
```

### 3. 本番環境では環境変数を直接設定

```bash
# Heroku
heroku config:set SECRET_KEY=xxx

# AWS
# Systems Manager Parameter Store や Secrets Manager を使用

# Docker
docker run -e DATABASE_URL=xxx -e SECRET_KEY=yyy myapp
```

### 4. シークレットキーの生成

```python
import secrets

# 安全なシークレットキーを生成
secret_key = secrets.token_urlsafe(32)
print(secret_key)
# 例：A3TvL9XK2pR8qN5mZ7wY1jC6uH4bS0eD
```

---

## よくある間違い

### ❌ 間違い1：環境変数の型変換を忘れる

```python
# ❌ 文字列として取得される
DEBUG = os.environ.get("DEBUG", "False")
if DEBUG:  # "False"も真と判定される！
    print("Debug mode")
```

**✅ 正しい方法**：

```python
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
# または
DEBUG = os.environ.get("DEBUG", "False") == "True"
```

### ❌ 間違い2：.envをGitにコミット

```bash
# ❌ 絶対にやってはいけない
git add .env
git commit -m "Add config"
```

もしコミットしてしまった場合：

```bash
# 履歴から削除
git rm --cached .env
git commit -m "Remove .env from git"

# シークレットキーを再生成して環境変数を更新
```

---

## 演習問題

### 問題：設定管理システムを作る

以下の要件を満たす設定管理を実装してください：

1. データベースURL、シークレットキー、デバッグモードを環境変数から読む
2. 必須の環境変数がない場合はエラーを出す
3. Pydantic Settingsを使って型安全にする

**解答例**：

```python
from pydantic_settings import BaseSettings
from pydantic import Field, validator
import sys

class Settings(BaseSettings):
    database_url: str = Field(..., description="データベースURL")
    secret_key: str = Field(..., min_length=32, description="シークレットキー")
    debug: bool = Field(False, description="デバッグモード")
    port: int = Field(8000, ge=1, le=65535, description="ポート番号")

    @validator("secret_key")
    def validate_secret_key(cls, v):
        if v == "change-me" or len(v) < 32:
            raise ValueError("シークレットキーは32文字以上の安全な値を設定してください")
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False

# 設定を読み込む
try:
    settings = Settings()
    print("✅ 設定の読み込みに成功しました")
    print(f"Database: {settings.database_url}")
    print(f"Debug: {settings.debug}")
    print(f"Port: {settings.port}")
except Exception as e:
    print(f"❌ エラー: {e}")
    sys.exit(1)
```

---

## 次のステップ

### このガイドで学んだこと

- ✅ 環境変数の基本
- ✅ .envファイルの使い方
- ✅ python-dotenvによる環境変数管理
- ✅ Pydantic Settingsでの型安全な設定管理
- ✅ 環境別の設定分離

### 次に学ぶべきガイド

**次のガイド**：[07-simple-api-tutorial.md](./07-simple-api-tutorial.md) - 総合演習：シンプルなAPI構築

---

**前のガイド**：[05-authentication-basics.md](./05-authentication-basics.md)

**親ガイド**：[Backend Development - SKILL.md](../../SKILL.md)
