# API設計

> RESTful API・GraphQL・gRPC の設計原則を理解し、一貫性のある直感的なインターフェースを構築するための命名規則・バージョニング・エラーハンドリング・ページネーションの実践手法を解説する

## この章で学ぶこと

1. **RESTful API 設計原則** — リソース指向設計、HTTP メソッドの適切な使い分け、ステータスコード戦略
2. **API の品質要素** — バージョニング、ページネーション、エラーレスポンス、レート制限の標準パターン
3. **API スタイルの比較** — REST vs GraphQL vs gRPC の特性と選定基準

---

## 1. RESTful API 設計原則

### 1.1 リソース設計

```
URL 設計の原則

  GOOD: 名詞（リソース）ベース
    GET    /users              ← ユーザー一覧
    GET    /users/123          ← 特定ユーザー
    POST   /users              ← ユーザー作成
    PUT    /users/123          ← ユーザー更新（全体）
    PATCH  /users/123          ← ユーザー更新（部分）
    DELETE /users/123          ← ユーザー削除

  GOOD: ネストしたリソース
    GET    /users/123/orders         ← ユーザー123の注文一覧
    POST   /users/123/orders         ← ユーザー123の注文作成
    GET    /users/123/orders/456     ← 特定の注文

  BAD: 動詞ベース
    POST   /createUser               ← RPC スタイル
    GET    /getUserById?id=123       ← RPC スタイル
    POST   /deleteUser/123           ← HTTP メソッドと矛盾
```

### 1.2 HTTP メソッドとステータスコード

```
HTTP メソッドの意味

  GET     - 取得 (べき等、安全)
  POST    - 作成 (非べき等)
  PUT     - 全体更新 (べき等)
  PATCH   - 部分更新 (べき等)
  DELETE  - 削除 (べき等)

レスポンスステータスコード

  2xx 成功
  ├── 200 OK              - 取得・更新成功
  ├── 201 Created         - 作成成功 (+ Location ヘッダー)
  └── 204 No Content      - 削除成功 (レスポンスボディなし)

  4xx クライアントエラー
  ├── 400 Bad Request     - リクエストが不正
  ├── 401 Unauthorized    - 認証失敗
  ├── 403 Forbidden       - 認可失敗 (権限なし)
  ├── 404 Not Found       - リソースなし
  ├── 409 Conflict        - 競合 (重複作成など)
  ├── 422 Unprocessable   - バリデーションエラー
  └── 429 Too Many Req    - レート制限超過

  5xx サーバーエラー
  ├── 500 Internal Error  - サーバー内部エラー
  ├── 502 Bad Gateway     - 上流サービスエラー
  └── 503 Service Unavail - メンテナンス中
```

---

## 2. エラーレスポンス設計

```python
# 統一エラーレスポンス形式
from flask import Flask, jsonify
from dataclasses import dataclass

@dataclass
class APIError:
    code: str
    message: str
    details: list = None

    def to_response(self, status_code: int):
        body = {
            "error": {
                "code": self.code,
                "message": self.message,
            }
        }
        if self.details:
            body["error"]["details"] = self.details
        return jsonify(body), status_code

# エラーハンドラー
@app.errorhandler(422)
def validation_error(e):
    return APIError(
        code="VALIDATION_ERROR",
        message="入力データに問題があります",
        details=[
            {"field": "email", "message": "メールアドレスの形式が不正です"},
            {"field": "age", "message": "年齢は0以上の整数で指定してください"},
        ]
    ).to_response(422)

# レスポンス例
# {
#   "error": {
#     "code": "VALIDATION_ERROR",
#     "message": "入力データに問題があります",
#     "details": [
#       {"field": "email", "message": "メールアドレスの形式が不正です"},
#       {"field": "age", "message": "年齢は0以上の整数で指定してください"}
#     ]
#   }
# }
```

---

## 3. ページネーション

### 3.1 カーソルベース vs オフセットベース

```
【オフセットベース】
  GET /users?page=3&per_page=20

  レスポンス:
  {
    "data": [...],
    "pagination": {
      "page": 3,
      "per_page": 20,
      "total": 150,
      "total_pages": 8
    }
  }

  メリット: シンプル、任意ページへのジャンプ
  デメリット: 大量データで性能劣化 (OFFSET N)

【カーソルベース】
  GET /users?cursor=eyJpZCI6MTAwfQ&limit=20

  レスポンス:
  {
    "data": [...],
    "pagination": {
      "next_cursor": "eyJpZCI6MTIwfQ",
      "has_next": true
    }
  }

  メリット: 大量データでも高速、一貫性
  デメリット: 任意ページへのジャンプ不可
```

### 3.2 実装

```python
# カーソルベースページネーション (FastAPI)
from fastapi import FastAPI, Query
import base64, json

app = FastAPI()

@app.get("/api/v1/users")
async def list_users(
    cursor: str = Query(None, description="ページネーションカーソル"),
    limit: int = Query(20, ge=1, le=100, description="取得件数"),
):
    # カーソルのデコード
    if cursor:
        decoded = json.loads(base64.b64decode(cursor))
        last_id = decoded['id']
        query = "SELECT * FROM users WHERE id > %s ORDER BY id LIMIT %s"
        users = db.execute(query, (last_id, limit + 1))
    else:
        query = "SELECT * FROM users ORDER BY id LIMIT %s"
        users = db.execute(query, (limit + 1,))

    users = list(users)
    has_next = len(users) > limit
    if has_next:
        users = users[:limit]

    # 次のカーソル生成
    next_cursor = None
    if has_next and users:
        next_cursor = base64.b64encode(
            json.dumps({"id": users[-1].id}).encode()
        ).decode()

    return {
        "data": [user.to_dict() for user in users],
        "pagination": {
            "next_cursor": next_cursor,
            "has_next": has_next,
            "limit": limit,
        }
    }
```

---

## 4. バージョニング

```python
# URL パスベース（最も一般的）
# GET /api/v1/users
# GET /api/v2/users

from fastapi import APIRouter

v1_router = APIRouter(prefix="/api/v1")
v2_router = APIRouter(prefix="/api/v2")

@v1_router.get("/users/{user_id}")
async def get_user_v1(user_id: int):
    """v1: name フィールドを返す"""
    user = await get_user(user_id)
    return {"id": user.id, "name": user.name, "email": user.email}

@v2_router.get("/users/{user_id}")
async def get_user_v2(user_id: int):
    """v2: first_name / last_name に分離"""
    user = await get_user(user_id)
    return {
        "id": user.id,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "email": user.email,
    }
```

---

## 5. OpenAPI (Swagger) ドキュメント

```python
# FastAPI による自動ドキュメント生成
from fastapi import FastAPI, Query, Path, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(
    title="注文管理API",
    version="1.0.0",
    description="ECサイトの注文管理を行うRESTful API",
)

class OrderCreate(BaseModel):
    """注文作成リクエスト"""
    user_id: str = Field(..., description="ユーザーID", example="user-123")
    items: list = Field(..., description="注文アイテム", min_length=1)

class OrderResponse(BaseModel):
    """注文レスポンス"""
    id: str = Field(..., description="注文ID")
    status: str = Field(..., description="注文ステータス")
    total_amount: int = Field(..., description="合計金額（円）")

@app.post("/api/v1/orders",
          response_model=OrderResponse,
          status_code=201,
          summary="注文作成",
          tags=["Orders"])
async def create_order(order: OrderCreate):
    """新しい注文を作成する。

    - 注文アイテムは1件以上必須
    - 在庫がない場合は 409 Conflict を返す
    """
    result = await order_service.create(order)
    return result
```

---

## 6. API スタイル比較

| 特性 | REST | GraphQL | gRPC |
|------|------|---------|------|
| **プロトコル** | HTTP/1.1, HTTP/2 | HTTP (通常POST) | HTTP/2 |
| **データ形式** | JSON | JSON | Protocol Buffers |
| **型安全性** | OpenAPI で付加 | スキーマ内蔵 | .proto ファイル |
| **オーバーフェッチ** | 発生しやすい | クライアントが必要なフィールドを指定 | 定義済みメッセージ |
| **N+1 問題** | エンドポイント設計に依存 | DataLoader で解決 | ストリーミングで軽減 |
| **キャッシュ** | HTTP キャッシュが自然 | 困難（POST のみ） | 独自実装 |
| **学習コスト** | 低 | 中 | 高 |
| **最適用途** | 公開API、CRUD中心 | 複雑なデータグラフ | マイクロサービス間通信 |

| 判断基準 | REST | GraphQL | gRPC |
|---------|------|---------|------|
| 公開 API | 最適 | 良い | 不向き |
| モバイルアプリ | 良い | 最適 | 可能 |
| マイクロサービス間 | 良い | 可能 | 最適 |
| リアルタイム | WebSocket 追加 | Subscription | 双方向ストリーミング |
| ファイルアップロード | multipart/form-data | 不向き | ストリーミング |

---

## 7. アンチパターン

### アンチパターン 1: HTTP メソッドの誤用

```
BAD:
  POST /users/123/delete     ← DELETE を使うべき
  GET  /users/create?name=A  ← GET は副作用なし
  POST /users/123            ← 更新なら PUT/PATCH

GOOD:
  DELETE /users/123
  POST   /users   (Body: {"name": "A"})
  PATCH  /users/123 (Body: {"name": "B"})
```

### アンチパターン 2: レスポンス形式の不統一

```json
// BAD: エンドポイントごとに形式が異なる
// GET /users     → [{"id": 1, "name": "Alice"}]
// GET /orders    → {"results": [{"id": 1}], "count": 10}
// GET /products  → {"data": {"items": [...]}}

// GOOD: 統一された Envelope 形式
// GET /users
{
  "data": [{"id": 1, "name": "Alice"}],
  "pagination": {"next_cursor": "...", "has_next": true}
}
// GET /orders
{
  "data": [{"id": 1, "status": "placed"}],
  "pagination": {"next_cursor": "...", "has_next": false}
}
```

---

## 8. FAQ

### Q1. API のバージョニングはいつ行うべき？

**A.** 後方互換性が壊れる変更が必要な場合にのみバージョンを上げる。フィールドの追加は後方互換なのでバージョン不要。フィールドの削除・名前変更・型の変更は非互換なのでバージョンアップ。旧バージョンは廃止日を設定し、6-12ヶ月の移行期間を設ける。Sunset ヘッダーで廃止予定日を通知する。

### Q2. ページネーションはオフセットとカーソルのどちらを使うべき？

**A.** データ量と要件で判断する。データが少なく（<10万件）ページジャンプが必要ならオフセット。大量データ・リアルタイム更新があるならカーソル。SNS のタイムライン、ログ検索などは常にカーソルベース。管理画面の一覧表示はオフセットで十分な場合が多い。

### Q3. REST と GraphQL を同一プロジェクトで併用してよいか？

**A.** 併用は合理的な選択。公開 API は REST（キャッシュ、シンプルさ）、フロントエンド向け BFF は GraphQL（柔軟なデータ取得）という使い分けが一般的。ただし、チームの学習コストと運用コストを考慮し、小規模チームでは片方に統一する方が効率的。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| リソース設計 | 名詞ベースの URL、HTTP メソッドで操作を表現 |
| ステータスコード | 2xx/4xx/5xx を正確に使い分け |
| エラーレスポンス | 統一形式（code, message, details）|
| ページネーション | 小規模はオフセット、大規模はカーソル |
| バージョニング | URL パスベースが最も明確 |
| ドキュメント | OpenAPI (Swagger) で自動生成 |
| API スタイル選定 | 公開API=REST、BFF=GraphQL、内部通信=gRPC |

---

## 次に読むべきガイド

- [コードレビューチェックリスト](./04-code-review-checklist.md) — API コードのレビュー観点
- [テスト原則](../01-practices/04-testing-principles.md) — API テストの設計
- [レートリミッター設計](../../system-design-guide/docs/03-case-studies/03-rate-limiter.md) — API レート制限の実装

---

## 参考文献

1. **RESTful Web APIs** — Leonard Richardson & Mike Amundsen (O'Reilly, 2013) — REST 設計の包括的ガイド
2. **API Design Patterns** — JJ Geewax (Manning, 2021) — API 設計パターンのカタログ
3. **Google API Design Guide** — https://cloud.google.com/apis/design — Google の API 設計基準
4. **Microsoft REST API Guidelines** — https://github.com/microsoft/api-guidelines
