# HTTP基礎 - 完全初心者ガイド

## 目次

1. [概要](#概要)
2. [HTTPとは](#httpとは)
3. [HTTPメソッド](#httpメソッド)
4. [ステータスコード](#ステータスコード)
5. [リクエストとレスポンス](#リクエストとレスポンス)
6. [ヘッダー](#ヘッダー)
7. [演習問題](#演習問題)
8. [次のステップ](#次のステップ)

---

## 概要

### 何を学ぶか

- HTTPプロトコルの基礎
- HTTPメソッド（GET、POST、PUT、DELETE）
- ステータスコード
- リクエスト/レスポンスの構造

### 学習時間：30〜40分

---

## HTTPとは

### 定義

**HTTP（HyperText Transfer Protocol）**は、WebブラウザとWebサーバー間で情報をやり取りするためのプロトコル（通信規約）です。

```
クライアント                      サーバー
(ブラウザ)                      (Webサーバー)
    │                               │
    │  HTTPリクエスト                │
    │  GET /api/users/1              │
    │───────────────────────────────>│
    │                               │
    │  HTTPレスポンス                │
    │  200 OK                        │
    │  {"id": 1, "name": "太郎"}     │
    │<───────────────────────────────│
```

---

## HTTPメソッド

### 主要な4つのメソッド（CRUD）

| メソッド | 意味 | CRUD | 例 |
|---------|------|------|-----|
| **GET** | 取得 | Read | ユーザー情報を取得 |
| **POST** | 作成 | Create | 新規ユーザーを作成 |
| **PUT/PATCH** | 更新 | Update | ユーザー情報を更新 |
| **DELETE** | 削除 | Delete | ユーザーを削除 |

### 実例

```python
from fastapi import FastAPI

app = FastAPI()

# GET - 取得
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    return {"id": user_id, "name": "太郎"}

# POST - 作成
@app.post("/users")
async def create_user(name: str, email: str):
    return {"id": 1, "name": name, "email": email}

# PUT - 更新（全体）
@app.put("/users/{user_id}")
async def update_user(user_id: int, name: str, email: str):
    return {"id": user_id, "name": name, "email": email}

# PATCH - 更新（一部）
@app.patch("/users/{user_id}")
async def patch_user(user_id: int, name: str = None):
    return {"id": user_id, "name": name}

# DELETE - 削除
@app.delete("/users/{user_id}")
async def delete_user(user_id: int):
    return {"message": "削除しました"}
```

---

## ステータスコード

### 主要なステータスコード

#### 2xx - 成功

| コード | 意味 | 使用例 |
|--------|------|--------|
| **200** | OK | 正常に処理完了 |
| **201** | Created | リソース作成成功 |
| **204** | No Content | 成功（レスポンスボディなし） |

#### 4xx - クライアントエラー

| コード | 意味 | 使用例 |
|--------|------|--------|
| **400** | Bad Request | 不正なリクエスト |
| **401** | Unauthorized | 認証が必要 |
| **403** | Forbidden | アクセス権限がない |
| **404** | Not Found | リソースが見つからない |

#### 5xx - サーバーエラー

| コード | 意味 | 使用例 |
|--------|------|--------|
| **500** | Internal Server Error | サーバー内部エラー |
| **502** | Bad Gateway | ゲートウェイエラー |
| **503** | Service Unavailable | サービス利用不可 |

### 実装例

```python
from fastapi import HTTPException

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="ユーザーが見つかりません")

    return user

@app.post("/users")
async def create_user(name: str, email: str):
    if not email:
        raise HTTPException(status_code=400, detail="メールアドレスは必須です")

    user = User(name=name, email=email)
    db.add(user)
    db.commit()

    return user, 201  # 201 Created
```

---

## リクエストとレスポンス

### HTTPリクエストの構造

```http
POST /api/users HTTP/1.1
Host: example.com
Content-Type: application/json
Authorization: Bearer eyJhbGc...

{
  "name": "太郎",
  "email": "taro@example.com"
}
```

**構成要素**：
1. **リクエストライン**：メソッド、パス、HTTPバージョン
2. **ヘッダー**：メタデータ
3. **ボディ**：送信するデータ

### HTTPレスポンスの構造

```http
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 52

{
  "id": 1,
  "name": "太郎",
  "email": "taro@example.com"
}
```

**構成要素**：
1. **ステータスライン**：HTTPバージョン、ステータスコード
2. **ヘッダー**：メタデータ
3. **ボディ**：返すデータ

---

## ヘッダー

### よく使うヘッダー

#### リクエストヘッダー

```http
Content-Type: application/json
Authorization: Bearer token123
Accept: application/json
User-Agent: Mozilla/5.0...
```

#### レスポンスヘッダー

```http
Content-Type: application/json
Content-Length: 123
Cache-Control: no-cache
Set-Cookie: session=abc123
```

### 実装例

```python
from fastapi import Header

@app.get("/protected")
async def protected_route(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "認証が必要です")

    token = authorization.replace("Bearer ", "")
    user = verify_token(token)

    return {"user": user}
```

---

## 演習問題

### 問題：curlでAPIを呼ぶ

```bash
# GET
curl http://localhost:8000/users/1

# POST
curl -X POST http://localhost:8000/users \
  -H "Content-Type: application/json" \
  -d '{"name": "太郎", "email": "taro@example.com"}'

# DELETE
curl -X DELETE http://localhost:8000/users/1
```

---

## 次のステップ

**次のガイド**：[03-rest-api-intro.md](./03-rest-api-intro.md) - REST API設計
