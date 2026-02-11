# REST API入門 - 完全初心者ガイド

## 目次

1. [概要](#概要)
2. [REST APIとは](#rest-apiとは)
3. [RESTの6原則](#restの6原則)
4. [リソース設計](#リソース設計)
5. [エンドポイント設計](#エンドポイント設計)
6. [実装例](#実装例)
7. [次のステップ](#次のステップ)

---

## 概要

### 何を学ぶか

- REST APIの基本概念
- RESTful設計の原則
- エンドポイントの命名規則
- CRUD操作の実装

### 学習時間：40〜50分

---

## REST APIとは

### 定義

**REST（Representational State Transfer）**は、Web APIの設計アーキテクチャです。

**特徴**：
- ステートレス（状態を持たない）
- リソース指向
- HTTPメソッドを活用
- JSON形式でデータをやり取り

---

## RESTの6原則

### 1. クライアント/サーバー分離

```
クライアント ←→ サーバー
（独立）       （独立）
```

### 2. ステートレス

サーバーはクライアントの状態を保持しません。

```python
# ❌ ステートフル（セッションを使う）
@app.get("/cart")
def get_cart(session: Session):
    return session.cart

# ✅ ステートレス（トークンで認証）
@app.get("/cart")
def get_cart(token: str = Header(...)):
    user = verify_token(token)
    return db.query(Cart).filter(Cart.user_id == user.id).all()
```

### 3. キャッシュ可能

レスポンスにキャッシュ情報を含めます。

```python
from fastapi import Response

@app.get("/users/{user_id}")
def get_user(user_id: int, response: Response):
    response.headers["Cache-Control"] = "max-age=3600"  # 1時間キャッシュ
    return {"id": user_id, "name": "太郎"}
```

### 4. 統一インターフェース

一貫したAPIパターンを使います。

### 5. 階層化システム

```
クライアント → ロードバランサー → APIサーバー → データベース
```

### 6. コードオンデマンド（オプション）

JavaScriptなどをクライアントに送信できます。

---

## リソース設計

### リソースとは

API で操作する「もの」です。

**例**：
- ユーザー（users）
- 記事（posts）
- コメント（comments）

### リソースの表現

```
/users          - ユーザー一覧
/users/1        - ID=1のユーザー
/users/1/posts  - ID=1のユーザーの記事一覧
```

---

## エンドポイント設計

### 基本パターン

| メソッド | エンドポイント | 説明 |
|---------|---------------|------|
| GET | /users | ユーザー一覧取得 |
| GET | /users/:id | 特定ユーザー取得 |
| POST | /users | ユーザー作成 |
| PUT | /users/:id | ユーザー更新（全体） |
| PATCH | /users/:id | ユーザー更新（一部） |
| DELETE | /users/:id | ユーザー削除 |

### 命名規則

```python
# ✅ 良い例（複数形、小文字、ハイフン）
GET /users
GET /blog-posts
GET /user-profiles

# ❌ 悪い例
GET /getUsers        # 動詞を使わない
GET /Users           # 小文字を使う
GET /user            # 複数形を使う
```

### ネストしたリソース

```python
# ユーザーの記事一覧
GET /users/1/posts

# 記事のコメント一覧
GET /posts/1/comments

# ❌ 深すぎるネスト（3階層まで）
GET /users/1/posts/1/comments/1/likes  # 避ける
```

---

## 実装例

### FastAPIでのREST API実装

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

# モデル定義
class User(BaseModel):
    id: int
    name: str
    email: str

# ダミーデータベース
users_db: List[User] = [
    User(id=1, name="太郎", email="taro@example.com"),
    User(id=2, name="花子", email="hanako@example.com")
]

# GET /users - 一覧取得
@app.get("/users", response_model=List[User])
async def get_users():
    return users_db

# GET /users/:id - 詳細取得
@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int):
    user = next((u for u in users_db if u.id == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="ユーザーが見つかりません")
    return user

# POST /users - 作成
@app.post("/users", response_model=User, status_code=201)
async def create_user(name: str, email: str):
    new_id = max([u.id for u in users_db], default=0) + 1
    new_user = User(id=new_id, name=name, email=email)
    users_db.append(new_user)
    return new_user

# PUT /users/:id - 更新
@app.put("/users/{user_id}", response_model=User)
async def update_user(user_id: int, name: str, email: str):
    user = next((u for u in users_db if u.id == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="ユーザーが見つかりません")
    user.name = name
    user.email = email
    return user

# DELETE /users/:id - 削除
@app.delete("/users/{user_id}")
async def delete_user(user_id: int):
    global users_db
    users_db = [u for u in users_db if u.id != user_id]
    return {"message": "削除しました"}
```

### クエリパラメータ

```python
# GET /users?limit=10&offset=0
@app.get("/users")
async def get_users(limit: int = 10, offset: int = 0):
    return users_db[offset:offset+limit]

# GET /users?name=太郎
@app.get("/users")
async def search_users(name: str = None):
    if name:
        return [u for u in users_db if name in u.name]
    return users_db
```

---

## 次のステップ

**次のガイド**：[04-database-intro.md](./04-database-intro.md) - データベース基礎
