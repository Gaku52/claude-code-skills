# API設計

> RESTful API・GraphQL・gRPC の設計原則を理解し、一貫性のある直感的なインターフェースを構築するための命名規則・バージョニング・エラーハンドリング・ページネーション・セキュリティ・テストの実践手法を解説する

---

## 前提知識

| トピック | 内容 | 参照先 |
|---------|------|--------|
| HTTPプロトコルの基礎 | メソッド、ステータスコード、ヘッダー | [../../04-web-and-network/](../../04-web-and-network/) |
| クリーンコードの基本原則 | 命名規則・関数設計 | [00-naming-conventions.md](../00-principles/00-naming-conventions.md) |
| エラーハンドリング | 例外処理の基本パターン | [03-error-handling.md](../01-practices/03-error-handling.md) |
| テスト原則 | テストピラミッド・テスト設計 | [04-testing-principles.md](../01-practices/04-testing-principles.md) |
| 関数型エラーハンドリング | Result/Either型 | [02-functional-principles.md](./02-functional-principles.md) |

---

## この章で学ぶこと

1. **RESTful API 設計原則** — リソース指向設計、HTTP メソッドの適切な使い分け、ステータスコード戦略を理解し適用できる
2. **API の品質要素** — バージョニング、ページネーション、エラーレスポンス、レート制限の標準パターンを実装できる
3. **API スタイルの比較と選定** — REST vs GraphQL vs gRPC の特性と、プロジェクト要件に応じた選定判断ができる
4. **セキュリティと認証** — OAuth 2.0 / JWT / API Key の認証パターンを理解し、セキュアな API を設計できる
5. **API テストとドキュメント** — OpenAPI による自動ドキュメント生成と、契約テストによる品質保証を実践できる

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

```
リソース設計の判断フロー:

  1. リソースを名詞で命名する（複数形）
     /users, /orders, /products

  2. 関係性をネストで表現する（2階層まで推奨）
     /users/123/orders
     NG: /users/123/orders/456/items/789/reviews（深すぎる）
     OK: /orders/456/items  or  /reviews?item_id=789

  3. リソースにならない操作は「動作リソース」として扱う
     POST /orders/456/cancel    ← 注文キャンセル（動作）
     POST /users/123/activate   ← ユーザー有効化

  4. 検索・フィルタはクエリパラメータ
     GET /products?category=electronics&min_price=1000&sort=price_asc

  5. バルク操作
     POST /users/bulk-create     ← 一括作成
     PATCH /orders/bulk-update   ← 一括更新
```

### 1.2 HTTP メソッドとステータスコード

```
HTTP メソッドの意味と安全性

  メソッド    意味       べき等    安全    リクエストボディ
  ─────────────────────────────────────────────────────
  GET       取得        YES      YES     なし
  HEAD      ヘッダ取得  YES      YES     なし
  POST      作成        NO       NO      あり
  PUT       全体更新    YES      NO      あり
  PATCH     部分更新    YES      NO      あり
  DELETE    削除        YES      NO      通常なし
  OPTIONS   仕様確認    YES      YES     なし

  安全: サーバーの状態を変更しない
  べき等: 同じリクエストを何度送っても結果が同じ
    例: DELETE /users/123 を2回送っても、
        1回目: 削除成功(200)
        2回目: 既に存在しない(404) ← 状態は同じ
```

```
レスポンスステータスコード

  2xx 成功
  ├── 200 OK              - 取得・更新成功
  ├── 201 Created         - 作成成功 (+ Location ヘッダー)
  ├── 202 Accepted        - 非同期処理を受け付けた
  └── 204 No Content      - 削除成功 (レスポンスボディなし)

  3xx リダイレクト
  ├── 301 Moved Permanently - リソースが恒久的に移動
  └── 304 Not Modified      - キャッシュ有効（ETag一致）

  4xx クライアントエラー
  ├── 400 Bad Request     - リクエストが不正
  ├── 401 Unauthorized    - 認証失敗
  ├── 403 Forbidden       - 認可失敗 (権限なし)
  ├── 404 Not Found       - リソースなし
  ├── 405 Method Not Allowed - 許可されていないメソッド
  ├── 409 Conflict        - 競合 (重複作成など)
  ├── 422 Unprocessable   - バリデーションエラー
  └── 429 Too Many Req    - レート制限超過

  5xx サーバーエラー
  ├── 500 Internal Error  - サーバー内部エラー
  ├── 502 Bad Gateway     - 上流サービスエラー
  └── 503 Service Unavail - メンテナンス中
```

### 1.3 ステータスコード選択のフローチャート

```
リクエスト処理のステータスコード判断:

  リクエスト受信
    │
    ├── 認証は通ったか？
    │   └── NO → 401 Unauthorized
    │
    ├── 認可は通ったか？
    │   └── NO → 403 Forbidden
    │
    ├── リクエスト形式は正しいか？
    │   └── NO → 400 Bad Request
    │
    ├── リソースは存在するか？
    │   └── NO → 404 Not Found
    │
    ├── バリデーションは通ったか？
    │   └── NO → 422 Unprocessable Entity
    │
    ├── ビジネスルール上の競合はないか？
    │   └── YES(競合あり) → 409 Conflict
    │
    ├── 処理は成功したか？
    │   ├── 作成 → 201 Created
    │   ├── 削除 → 204 No Content
    │   ├── 非同期 → 202 Accepted
    │   └── その他 → 200 OK
    │
    └── サーバーエラー → 500 Internal Server Error
```

---

## 2. エラーレスポンス設計

### 2.1 統一エラーフォーマット

```python
# 統一エラーレスポンス形式（RFC 7807 Problem Details準拠）
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

class ProblemDetail(BaseModel):
    """RFC 7807 Problem Details for HTTP APIs"""
    type: str              # エラータイプのURI
    title: str             # 人間が読めるエラータイトル
    status: int            # HTTPステータスコード
    detail: Optional[str]  # エラーの詳細説明
    instance: Optional[str]  # エラーが発生したリクエストURI
    errors: Optional[list[dict]] = None  # バリデーションエラー詳細

app = FastAPI()

# エラーハンドラー
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ProblemDetail(
            type=f"https://api.example.com/errors/{exc.detail.get('code', 'unknown')}",
            title=exc.detail.get("title", "Error"),
            status=exc.status_code,
            detail=exc.detail.get("detail"),
            instance=str(request.url),
            errors=exc.detail.get("errors"),
        ).model_dump(exclude_none=True),
    )
```

### 2.2 エラーレスポンスの実例

```python
# バリデーションエラー (422)
{
    "type": "https://api.example.com/errors/validation_error",
    "title": "Validation Error",
    "status": 422,
    "detail": "入力データに問題があります",
    "instance": "/api/v1/users",
    "errors": [
        {"field": "email", "message": "メールアドレスの形式が不正です", "code": "invalid_format"},
        {"field": "age", "message": "年齢は0以上の整数で指定してください", "code": "out_of_range"}
    ]
}

# 認証エラー (401)
{
    "type": "https://api.example.com/errors/authentication_required",
    "title": "Authentication Required",
    "status": 401,
    "detail": "このリソースへのアクセスには認証が必要です"
}

# レート制限 (429)
{
    "type": "https://api.example.com/errors/rate_limit_exceeded",
    "title": "Rate Limit Exceeded",
    "status": 429,
    "detail": "リクエスト上限に達しました。60秒後に再試行してください",
    "retry_after": 60
}

# ビジネスルール違反 (409)
{
    "type": "https://api.example.com/errors/insufficient_stock",
    "title": "Insufficient Stock",
    "status": 409,
    "detail": "商品「MacBook Pro」の在庫が不足しています（要求: 5, 在庫: 2）"
}
```

### 2.3 エラーコード体系の設計

```
エラーコード命名規則:

  {ドメイン}_{カテゴリ}_{詳細}

  例:
    AUTH_TOKEN_EXPIRED         - 認証トークン期限切れ
    AUTH_INVALID_CREDENTIALS   - 認証情報不正
    USER_NOT_FOUND            - ユーザー未発見
    USER_EMAIL_DUPLICATE      - メールアドレス重複
    ORDER_INSUFFICIENT_STOCK  - 在庫不足
    ORDER_ALREADY_CANCELLED   - 既にキャンセル済み
    PAYMENT_CARD_DECLINED     - カード決済拒否
    RATE_LIMIT_EXCEEDED       - レート制限超過

  利点:
  ├── クライアントがエラーの種類をプログラムで判別可能
  ├── エラー辞書の自動生成が可能
  ├── i18n（多言語対応）のキーとして使用可能
  └── ログ検索やアラート設定の条件として使用可能
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

```
ページネーション方式の選定基準:

  要件                          推奨方式
  ────────────────────────────────────────
  管理画面の一覧 (<10万件)      オフセット
  SNSタイムライン               カーソル
  検索結果 (ページジャンプ必要)  オフセット
  リアルタイムフィード          カーソル
  データエクスポート            カーソル
  ログ検索                      カーソル
  ECサイト商品一覧              ハイブリッド*

  *ハイブリッド: 最初の数ページはオフセット、
   深いページはカーソルに切り替え
```

### 3.2 カーソルベースの実装

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

### 3.3 複合ソートのカーソル

```python
# 複合条件でのカーソルベースページネーション
# 例: created_at DESC, id DESC でソート

@app.get("/api/v1/orders")
async def list_orders(
    cursor: str = Query(None),
    limit: int = Query(20, ge=1, le=100),
    status: str = Query(None, description="ステータスフィルタ"),
):
    if cursor:
        decoded = json.loads(base64.b64decode(cursor))
        # 複合カーソル: 同じ created_at の場合に id で一意に特定
        query = """
            SELECT * FROM orders
            WHERE (created_at, id) < (%s, %s)
            {status_filter}
            ORDER BY created_at DESC, id DESC
            LIMIT %s
        """
        params = [decoded["created_at"], decoded["id"]]
    else:
        query = """
            SELECT * FROM orders
            {status_filter}
            ORDER BY created_at DESC, id DESC
            LIMIT %s
        """
        params = []

    # ステータスフィルタの動的追加
    if status:
        status_filter = "AND status = %s"
        params.append(status)
    else:
        status_filter = ""

    query = query.replace("{status_filter}", status_filter)
    params.append(limit + 1)
    orders = db.execute(query, params)

    orders = list(orders)
    has_next = len(orders) > limit
    if has_next:
        orders = orders[:limit]

    next_cursor = None
    if has_next and orders:
        last = orders[-1]
        next_cursor = base64.b64encode(json.dumps({
            "created_at": last.created_at.isoformat(),
            "id": last.id,
        }).encode()).decode()

    return {
        "data": [o.to_dict() for o in orders],
        "pagination": {"next_cursor": next_cursor, "has_next": has_next},
    }
```

---

## 4. バージョニング

### 4.1 バージョニング戦略の比較

```
バージョニング方式の比較:

  方式              例                          メリット              デメリット
  ──────────────────────────────────────────────────────────────────────────────
  URLパス           /api/v1/users               最も明確、ルーティング容易  URL変更
  クエリパラメータ   /api/users?version=1        URLパスはクリーン        見落とされやすい
  ヘッダー          Accept: application/vnd.    URL変更なし             検証が難しい
                    api.v1+json
  コンテンツ        Accept: application/        柔軟性が高い            実装が複雑
  ネゴシエーション   vnd.api+json; version=1

  推奨: URLパスベース（最も広く採用、開発者が直感的に理解可能）
```

### 4.2 バージョニングの実装

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

### 4.3 バージョン廃止ポリシー

```
API バージョン廃止のライフサイクル:

  v1 リリース ──── v2 リリース ──── v1 非推奨通知 ──── v1 廃止
      │               │                │                │
      │               │                │ Sunset ヘッダー │
      │               │                │ 追加            │
      └───── 運用 ─────┴── 移行期間 ────┴── 廃止 ────────┘
                       (6-12ヶ月)

  廃止通知の実装:
  ┌──────────────────────────────────────────────────┐
  │ HTTP/1.1 200 OK                                   │
  │ Sunset: Sat, 01 Mar 2026 00:00:00 GMT             │
  │ Deprecation: true                                  │
  │ Link: <https://api.example.com/docs/migration>;    │
  │       rel="deprecation"                            │
  └──────────────────────────────────────────────────┘

  クライアント対応:
  1. Sunset ヘッダーを監視し、廃止前にマイグレーション
  2. 非推奨 API の使用をログ/アラートで検出
  3. SDK のバージョンアップで自動対応
```

---

## 5. 認証と認可

### 5.1 認証パターンの比較

```
認証方式の比較:

  方式          セキュリティ    実装コスト    最適用途
  ──────────────────────────────────────────────────────
  API Key       低             低           内部API、サーバー間通信
  Basic Auth    低             最低          開発環境、内部ツール
  Bearer Token  中             中           モバイルアプリ、SPA
  (JWT)
  OAuth 2.0     高             高           サードパーティ連携
  mTLS          最高           最高          マイクロサービス間
```

### 5.2 JWT 認証の実装

```python
# FastAPI + JWT 認証
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta

app = FastAPI()
security = HTTPBearer()

SECRET_KEY = "your-secret-key"  # 実際は環境変数から取得
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(user_id: str, roles: list[str]) -> str:
    """アクセストークン生成"""
    payload = {
        "sub": user_id,
        "roles": roles,
        "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """トークン検証ミドルウェア"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail={
            "code": "AUTH_TOKEN_EXPIRED",
            "title": "Token Expired",
            "detail": "認証トークンの有効期限が切れています",
        })
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail={
            "code": "AUTH_INVALID_TOKEN",
            "title": "Invalid Token",
            "detail": "無効な認証トークンです",
        })

def require_role(required_role: str):
    """ロールベース認可"""
    def role_checker(token: dict = Depends(verify_token)):
        if required_role not in token.get("roles", []):
            raise HTTPException(status_code=403, detail={
                "code": "AUTH_INSUFFICIENT_PERMISSIONS",
                "title": "Forbidden",
                "detail": f"この操作には '{required_role}' ロールが必要です",
            })
        return token
    return role_checker

# 使用例
@app.get("/api/v1/users")
async def list_users(token: dict = Depends(verify_token)):
    """認証必須エンドポイント"""
    return {"users": [...]}

@app.delete("/api/v1/users/{user_id}")
async def delete_user(user_id: str, token: dict = Depends(require_role("admin"))):
    """admin ロール必須"""
    pass
```

### 5.3 API セキュリティチェックリスト

```
API セキュリティの必須項目:

  認証・認可
  ├── [x] 全エンドポイントに認証を設定（公開APIは明示的に除外）
  ├── [x] トークンの有効期限を短く設定（アクセス: 15-30分, リフレッシュ: 7-30日）
  ├── [x] ロールベースアクセス制御（RBAC）を実装
  └── [x] 認可チェックはリソース単位（自分のデータのみ操作可能）

  入力検証
  ├── [x] 全入力をサーバーサイドでバリデーション
  ├── [x] SQL インジェクション対策（パラメータバインド）
  ├── [x] XSS 対策（出力エスケープ、Content-Type 明示）
  └── [x] パスパラメータのバリデーション（../traversal 防止）

  通信
  ├── [x] HTTPS 強制（HSTS ヘッダー）
  ├── [x] CORS 設定（許可オリジンを明示）
  └── [x] レスポンスに不要な情報を含めない（Server ヘッダー除去）

  レート制限
  ├── [x] エンドポイント別のレート制限
  ├── [x] 429 レスポンスに Retry-After ヘッダー
  └── [x] 認証試行の回数制限（ブルートフォース対策）
```

---

## 6. OpenAPI (Swagger) ドキュメント

### 6.1 自動ドキュメント生成

```python
# FastAPI による自動ドキュメント生成
from fastapi import FastAPI, Query, Path, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

app = FastAPI(
    title="注文管理API",
    version="1.0.0",
    description="ECサイトの注文管理を行うRESTful API",
)

class OrderStatus(str, Enum):
    pending = "pending"
    confirmed = "confirmed"
    shipped = "shipped"
    delivered = "delivered"
    cancelled = "cancelled"

class OrderItemCreate(BaseModel):
    """注文アイテム"""
    product_id: str = Field(..., description="商品ID", example="prod-456")
    quantity: int = Field(..., ge=1, le=100, description="数量", example=2)

class OrderCreate(BaseModel):
    """注文作成リクエスト"""
    user_id: str = Field(..., description="ユーザーID", example="user-123")
    items: list[OrderItemCreate] = Field(
        ..., description="注文アイテム", min_length=1
    )
    note: Optional[str] = Field(None, max_length=500, description="備考")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "user-123",
                    "items": [
                        {"product_id": "prod-456", "quantity": 2},
                        {"product_id": "prod-789", "quantity": 1},
                    ],
                    "note": "配達は午前中にお願いします",
                }
            ]
        }
    }

class OrderResponse(BaseModel):
    """注文レスポンス"""
    id: str = Field(..., description="注文ID")
    status: OrderStatus = Field(..., description="注文ステータス")
    total_amount: int = Field(..., description="合計金額（円）")
    created_at: str = Field(..., description="作成日時 (ISO 8601)")

@app.post(
    "/api/v1/orders",
    response_model=OrderResponse,
    status_code=201,
    summary="注文作成",
    tags=["Orders"],
    responses={
        409: {"description": "在庫不足"},
        422: {"description": "バリデーションエラー"},
    },
)
async def create_order(order: OrderCreate):
    """新しい注文を作成する。

    - 注文アイテムは1件以上必須
    - 在庫がない場合は 409 Conflict を返す
    - 作成成功時は 201 Created + Location ヘッダーを返す
    """
    result = await order_service.create(order)
    return result
```

### 6.2 契約テスト（Contract Testing）

```python
# OpenAPI スキーマに基づく契約テスト
import pytest
from fastapi.testclient import TestClient
from jsonschema import validate

client = TestClient(app)

class TestOrderAPI:
    """注文API の契約テスト"""

    def test_create_order_returns_201(self):
        """正常系: 注文作成は 201 を返す"""
        response = client.post("/api/v1/orders", json={
            "user_id": "user-123",
            "items": [{"product_id": "prod-456", "quantity": 2}],
        })
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert "status" in data
        assert data["status"] == "pending"

    def test_create_order_with_empty_items_returns_422(self):
        """異常系: 空のアイテムは 422 を返す"""
        response = client.post("/api/v1/orders", json={
            "user_id": "user-123",
            "items": [],
        })
        assert response.status_code == 422

    def test_create_order_without_auth_returns_401(self):
        """異常系: 認証なしは 401 を返す"""
        response = client.post("/api/v1/orders", json={
            "user_id": "user-123",
            "items": [{"product_id": "prod-456", "quantity": 2}],
        }, headers={})  # Authorization ヘッダーなし
        assert response.status_code == 401

    def test_list_orders_pagination(self):
        """ページネーション: next_cursor が返される"""
        response = client.get("/api/v1/orders?limit=2")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "pagination" in data
        assert "has_next" in data["pagination"]

    def test_response_matches_schema(self):
        """レスポンスが OpenAPI スキーマに準拠している"""
        expected_schema = {
            "type": "object",
            "required": ["id", "status", "total_amount", "created_at"],
            "properties": {
                "id": {"type": "string"},
                "status": {"type": "string", "enum": ["pending", "confirmed", "shipped", "delivered", "cancelled"]},
                "total_amount": {"type": "integer"},
                "created_at": {"type": "string"},
            },
        }
        response = client.post("/api/v1/orders", json={
            "user_id": "user-123",
            "items": [{"product_id": "prod-456", "quantity": 1}],
        })
        validate(response.json(), expected_schema)
```

---

## 7. レート制限

### 7.1 レート制限の設計

```
レート制限のアルゴリズム比較:

  アルゴリズム      特徴                  メリット            デメリット
  ──────────────────────────────────────────────────────────────────
  固定窓           時間窓ごとにカウント    実装簡単            窓の境界でバースト
  スライディング窓  連続した時間窓         均一な制限          メモリ使用量大
  トークンバケツ    一定速度でトークン充填  バースト許容        パラメータ調整難
  リーキーバケツ    一定速度で処理         安定した出力        バースト不可

  推奨: トークンバケツ（バースト対応 + 実装の容易さのバランス）
```

```python
# レート制限の実装例 (FastAPI + Redis)
import redis
from fastapi import Request, HTTPException
import time

redis_client = redis.Redis(host="localhost", port=6379, db=0)

class RateLimiter:
    """トークンバケツ方式のレート制限"""

    def __init__(self, rate: int, per: int):
        """
        rate: 許可するリクエスト数
        per: 時間窓（秒）
        """
        self.rate = rate
        self.per = per

    async def check(self, key: str) -> tuple[bool, dict]:
        """レート制限チェック"""
        now = time.time()
        pipe = redis_client.pipeline()

        # スライディングウィンドウ
        window_start = now - self.per
        pipe.zremrangebyscore(key, 0, window_start)  # 古いエントリを削除
        pipe.zadd(key, {str(now): now})               # 現在のリクエストを追加
        pipe.zcard(key)                                # ウィンドウ内のリクエスト数
        pipe.expire(key, self.per)                     # TTL設定
        _, _, count, _ = pipe.execute()

        remaining = max(0, self.rate - count)
        headers = {
            "X-RateLimit-Limit": str(self.rate),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(now + self.per)),
        }

        if count > self.rate:
            headers["Retry-After"] = str(self.per)
            return False, headers
        return True, headers

# ミドルウェアとして適用
rate_limiter = RateLimiter(rate=100, per=60)  # 60秒に100リクエスト

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # クライアント識別（APIキー or IP）
    client_id = request.headers.get("X-API-Key") or request.client.host
    key = f"rate_limit:{client_id}"

    allowed, headers = await rate_limiter.check(key)

    if not allowed:
        return JSONResponse(
            status_code=429,
            content={
                "type": "https://api.example.com/errors/rate_limit_exceeded",
                "title": "Rate Limit Exceeded",
                "status": 429,
                "detail": f"{rate_limiter.per}秒間に{rate_limiter.rate}リクエストまで",
            },
            headers=headers,
        )

    response = await call_next(request)
    for k, v in headers.items():
        response.headers[k] = v
    return response
```

---

## 8. API スタイル比較

### 8.1 REST vs GraphQL vs gRPC

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

### 8.2 GraphQL の実装例

```typescript
// GraphQL スキーマ定義
const typeDefs = `
  type User {
    id: ID!
    name: String!
    email: String!
    orders(first: Int, after: String): OrderConnection!
  }

  type Order {
    id: ID!
    status: OrderStatus!
    totalAmount: Int!
    items: [OrderItem!]!
    createdAt: String!
  }

  type OrderItem {
    product: Product!
    quantity: Int!
    price: Int!
  }

  type Product {
    id: ID!
    name: String!
    price: Int!
  }

  type OrderConnection {
    edges: [OrderEdge!]!
    pageInfo: PageInfo!
  }

  type OrderEdge {
    cursor: String!
    node: Order!
  }

  type PageInfo {
    hasNextPage: Boolean!
    endCursor: String
  }

  enum OrderStatus {
    PENDING
    CONFIRMED
    SHIPPED
    DELIVERED
    CANCELLED
  }

  type Query {
    user(id: ID!): User
    orders(userId: ID!, first: Int, after: String): OrderConnection!
  }

  type Mutation {
    createOrder(input: CreateOrderInput!): Order!
    cancelOrder(id: ID!): Order!
  }

  input CreateOrderInput {
    userId: ID!
    items: [OrderItemInput!]!
  }

  input OrderItemInput {
    productId: ID!
    quantity: Int!
  }
`;

// リゾルバ（N+1問題をDataLoaderで解決）
import DataLoader from "dataloader";

const productLoader = new DataLoader<string, Product>(async (ids) => {
  const products = await db.products.findMany({ where: { id: { in: ids as string[] } } });
  const productMap = new Map(products.map(p => [p.id, p]));
  return ids.map(id => productMap.get(id)!);
});

const resolvers = {
  OrderItem: {
    product: (item: OrderItem) => productLoader.load(item.productId),
  },
};
```

### 8.3 gRPC の定義例

```protobuf
// order_service.proto
syntax = "proto3";

package order.v1;

service OrderService {
  // 単項RPC: 注文作成
  rpc CreateOrder(CreateOrderRequest) returns (CreateOrderResponse);

  // 単項RPC: 注文取得
  rpc GetOrder(GetOrderRequest) returns (Order);

  // サーバーストリーミング: 注文ステータスの監視
  rpc WatchOrderStatus(WatchOrderStatusRequest) returns (stream OrderStatusUpdate);

  // クライアントストリーミング: バルク注文作成
  rpc BulkCreateOrders(stream CreateOrderRequest) returns (BulkCreateOrdersResponse);
}

message CreateOrderRequest {
  string user_id = 1;
  repeated OrderItem items = 2;
  string note = 3;
}

message OrderItem {
  string product_id = 1;
  int32 quantity = 2;
}

message Order {
  string id = 1;
  string user_id = 2;
  OrderStatus status = 3;
  int32 total_amount = 4;
  repeated OrderItem items = 5;
  google.protobuf.Timestamp created_at = 6;
}

enum OrderStatus {
  ORDER_STATUS_UNSPECIFIED = 0;
  ORDER_STATUS_PENDING = 1;
  ORDER_STATUS_CONFIRMED = 2;
  ORDER_STATUS_SHIPPED = 3;
  ORDER_STATUS_DELIVERED = 4;
  ORDER_STATUS_CANCELLED = 5;
}
```

---

## 9. API 設計のベストプラクティス

### 9.1 命名規則

```
API 命名規則:

  URL パス:
    ├── 小文字のみ使用
    ├── 単語の区切りはハイフン（kebab-case）
    ├── リソース名は複数形
    └── 末尾のスラッシュなし

    GOOD: /api/v1/order-items
    BAD:  /api/v1/orderItems
    BAD:  /api/v1/order_items/

  クエリパラメータ:
    ├── snake_case を推奨
    └── 一般的なパラメータは統一

    GOOD: ?sort_by=created_at&order=desc
    BAD:  ?sortBy=createdAt&order=DESC

  レスポンスボディ:
    ├── camelCase (JavaScript クライアント向け)
    ├── または snake_case (Python/Ruby クライアント向け)
    └── プロジェクト内で統一

  JSON フィールド命名:
    ├── boolean: is_, has_, can_ プレフィックス
    │   "is_active": true, "has_orders": false
    ├── 日時: ISO 8601 形式 + _at サフィックス
    │   "created_at": "2025-03-15T10:30:00Z"
    └── ID: {リソース}_id
        "user_id": "usr-123", "order_id": "ord-456"
```

### 9.2 HATEOAS（API のセルフドキュメント性）

```python
# HATEOAS: レスポンスにナビゲーション用のリンクを含める
{
    "data": {
        "id": "ord-123",
        "status": "pending",
        "total_amount": 3500,
        "_links": {
            "self": {"href": "/api/v1/orders/ord-123"},
            "cancel": {"href": "/api/v1/orders/ord-123/cancel", "method": "POST"},
            "items": {"href": "/api/v1/orders/ord-123/items"},
            "user": {"href": "/api/v1/users/usr-456"},
        }
    }
}

# ステータスによって利用可能なアクションが変わる
# status: "shipped" の場合 → cancel リンクは含まれない
# status: "delivered" の場合 → return リンクが追加される
```

---

## 10. アンチパターン

### 10.1 アンチパターン：HTTP メソッドの誤用

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

**問題点**: HTTP メソッドのセマンティクスを無視すると、キャッシュ、ブラウザの戻るボタン、HTTPクライアントの自動リトライなどが正しく動作しない。GET リクエストで副作用があると、クローラーやプリフェッチにより意図しないデータ変更が起こりうる。

### 10.2 アンチパターン：レスポンス形式の不統一

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

**問題点**: クライアント SDK の自動生成が困難になる。フロントエンドのレスポンスパース処理がエンドポイントごとに異なり、バグの温床となる。

### 10.3 アンチパターン：過度にネストした URL

```
BAD:
  GET /companies/123/departments/456/teams/789/members/012/tasks

  問題:
  - URL が長すぎて可読性が低い
  - 中間リソースの ID が全て必要
  - キャッシュの粒度が粗くなる

GOOD:
  GET /tasks?team_id=789
  GET /teams/789/members
  GET /members/012/tasks

  原則: ネストは最大2階層まで
  3階層以上が必要な場合はクエリパラメータでフィルタリング
```

**問題点**: 深いネストはクライアントに不要な親リソースの ID を強制する。大規模 API では URL の表現力よりも、クエリパラメータによるフィルタリングの柔軟性が重要。

### 10.4 アンチパターン：内部構造の露出

```python
# BAD: DB のカラム名やテーブル構造がそのままレスポンスに漏れる
{
    "user_tbl_id": 123,           # テーブル名が露出
    "usr_pwd_hash": "abc123...",  # パスワードハッシュが露出
    "created_ts": 1710489600,     # 内部タイムスタンプ形式
    "is_del_flg": 0,              # 内部フラグ
}

# GOOD: API 用の DTO に変換してレスポンス
{
    "id": "usr-123",
    "name": "Alice",
    "email": "alice@example.com",
    "created_at": "2025-03-15T10:00:00Z",
}
```

**問題点**: 内部構造の露出はセキュリティリスク（攻撃者にスキーマ情報を与える）であり、DB スキーマの変更が API の破壊的変更に直結する。

---

## 11. 演習問題

### 演習1（基礎）: REST API エンドポイント設計

**課題**: オンライン書店の API を設計せよ。以下のリソースと操作をカバーすること。

```
リソース: 書籍（books）、著者（authors）、レビュー（reviews）、ユーザー（users）

操作:
  1. 書籍の CRUD
  2. 著者による書籍の検索
  3. 書籍へのレビュー投稿・取得
  4. ユーザーの注文履歴取得
  5. 書籍の在庫確認
```

**期待される出力**:

```
エンドポイント一覧（メソッド、URL、説明、ステータスコード）
```

**模範解答**:

```
書籍:
  GET    /api/v1/books                    200  書籍一覧（ページネーション）
  GET    /api/v1/books?author_id=123      200  著者で絞り込み
  GET    /api/v1/books/{id}               200  書籍詳細
  POST   /api/v1/books                    201  書籍登録（admin）
  PATCH  /api/v1/books/{id}               200  書籍更新（admin）
  DELETE /api/v1/books/{id}               204  書籍削除（admin）
  GET    /api/v1/books/{id}/stock         200  在庫確認

著者:
  GET    /api/v1/authors                  200  著者一覧
  GET    /api/v1/authors/{id}             200  著者詳細
  GET    /api/v1/authors/{id}/books       200  著者の書籍一覧

レビュー:
  GET    /api/v1/books/{id}/reviews       200  書籍のレビュー一覧
  POST   /api/v1/books/{id}/reviews       201  レビュー投稿（認証必須）
  PATCH  /api/v1/reviews/{id}             200  レビュー更新（自分のみ）
  DELETE /api/v1/reviews/{id}             204  レビュー削除（自分 or admin）

ユーザー:
  GET    /api/v1/users/me                 200  自分の情報
  GET    /api/v1/users/me/orders          200  注文履歴
  GET    /api/v1/users/me/reviews         200  自分のレビュー一覧
```

---

### 演習2（応用）: エラーレスポンスの設計

**課題**: 以下のエラーシナリオに対して、RFC 7807 準拠のエラーレスポンスを設計せよ。

```
シナリオ:
  1. 書籍の在庫不足で注文失敗
  2. 同じ書籍に対する重複レビュー
  3. 認証トークン期限切れ
  4. リクエストボディのバリデーションエラー（複数フィールド）
```

**期待される出力**:

```json
// 各シナリオごとの JSON レスポンス（status, type, title, detail, errors）
```

**模範解答**:

```json
// 1. 在庫不足 (409 Conflict)
{
  "type": "https://api.bookstore.com/errors/insufficient_stock",
  "title": "Insufficient Stock",
  "status": 409,
  "detail": "書籍「Effective Java」の在庫が不足しています（要求: 3, 在庫: 1）",
  "instance": "/api/v1/orders"
}

// 2. 重複レビュー (409 Conflict)
{
  "type": "https://api.bookstore.com/errors/duplicate_review",
  "title": "Duplicate Review",
  "status": 409,
  "detail": "この書籍に対するレビューは既に投稿済みです",
  "instance": "/api/v1/books/book-123/reviews"
}

// 3. トークン期限切れ (401 Unauthorized)
{
  "type": "https://api.bookstore.com/errors/token_expired",
  "title": "Token Expired",
  "status": 401,
  "detail": "認証トークンの有効期限が切れています。再ログインしてください"
}

// 4. バリデーションエラー (422 Unprocessable Entity)
{
  "type": "https://api.bookstore.com/errors/validation_error",
  "title": "Validation Error",
  "status": 422,
  "detail": "入力データに2件の問題があります",
  "instance": "/api/v1/books",
  "errors": [
    {"field": "title", "message": "タイトルは1〜200文字で入力してください", "code": "string_too_short"},
    {"field": "price", "message": "価格は0以上の整数で入力してください", "code": "value_error"}
  ]
}
```

---

### 演習3（発展）: API バージョン移行戦略の設計

**課題**: 以下のシナリオで v1 → v2 の移行戦略を設計せよ。

```
変更内容:
  v1: GET /api/v1/users/{id} → {"id": 1, "name": "Alice Smith", "email": "..."}
  v2: GET /api/v2/users/{id} → {"id": 1, "first_name": "Alice", "last_name": "Smith", "email": "..."}

条件:
  - v1 のクライアントは50以上存在
  - モバイルアプリは即時更新できない
  - 移行期間は6ヶ月
```

**期待される出力**:

```
1. v2 のコード実装
2. v1 の廃止スケジュール
3. クライアントへの通知方法
4. v1 の互換性維持レイヤー
```

**模範解答**:

```python
# 1. 内部は v2 のデータモデルに統一
class UserModel:
    id: int
    first_name: str
    last_name: str
    email: str

# 2. v1 互換レイヤー（アダプター）
@v1_router.get("/users/{user_id}")
async def get_user_v1(user_id: int, response: Response):
    user = await user_service.get(user_id)

    # 廃止通知ヘッダー
    response.headers["Sunset"] = "Sat, 01 Sep 2026 00:00:00 GMT"
    response.headers["Deprecation"] = "true"
    response.headers["Link"] = (
        '<https://api.example.com/docs/v1-to-v2-migration>; rel="deprecation"'
    )

    # v1 形式に変換
    return {
        "id": user.id,
        "name": f"{user.first_name} {user.last_name}",  # 後方互換
        "email": user.email,
    }

# 3. v2 エンドポイント
@v2_router.get("/users/{user_id}")
async def get_user_v2(user_id: int):
    user = await user_service.get(user_id)
    return {
        "id": user.id,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "email": user.email,
    }

# 4. 廃止スケジュール
"""
Month 1: v2 リリース + v1 Deprecation ヘッダー追加
Month 2: クライアントにメール通知 + ドキュメントに移行ガイド掲載
Month 3: v1 の使用状況をモニタリング（アクセスログ分析）
Month 4: 使用率の高いクライアントに個別通知
Month 5: v1 レスポンスに warning フィールド追加
Month 6: v1 を 410 Gone にして廃止
"""

# 5. v1 使用状況のモニタリング
@app.middleware("http")
async def track_api_version(request: Request, call_next):
    if request.url.path.startswith("/api/v1/"):
        client_id = request.headers.get("X-API-Key", "unknown")
        logger.warning(f"Deprecated v1 API access: {client_id} -> {request.url.path}")
        # メトリクスに記録
        metrics.increment("api.v1.deprecated_access", tags={"client": client_id})
    return await call_next(request)
```

---

## 12. FAQ

### Q1. API のバージョニングはいつ行うべき？

**A.** 後方互換性が壊れる変更が必要な場合にのみバージョンを上げる。フィールドの追加は後方互換なのでバージョン不要。フィールドの削除・名前変更・型の変更は非互換なのでバージョンアップ。旧バージョンは廃止日を設定し、6-12ヶ月の移行期間を設ける。Sunset ヘッダーで廃止予定日を通知する。

具体的な判断基準:
- **バージョン不要（後方互換）**: フィールド追加、新エンドポイント追加、オプショナルパラメータ追加
- **バージョン必要（非互換）**: フィールド削除/名前変更、型変更、必須パラメータ追加、レスポンス構造の変更、エラーコードの変更

### Q2. ページネーションはオフセットとカーソルのどちらを使うべき？

**A.** データ量と要件で判断する。データが少なく（<10万件）ページジャンプが必要ならオフセット。大量データ・リアルタイム更新があるならカーソル。SNS のタイムライン、ログ検索などは常にカーソルベース。管理画面の一覧表示はオフセットで十分な場合が多い。

カーソルの注意点: ソート条件が変わるとカーソルが無効になるため、カーソルにソート条件を含めるか、ソート変更時は先頭から再取得する設計にする。

### Q3. REST と GraphQL を同一プロジェクトで併用してよいか？

**A.** 併用は合理的な選択。公開 API は REST（キャッシュ、シンプルさ）、フロントエンド向け BFF は GraphQL（柔軟なデータ取得）という使い分けが一般的。ただし、チームの学習コストと運用コストを考慮し、小規模チームでは片方に統一する方が効率的。

### Q4. API のレスポンスに null を含めるべきか、フィールドを省略すべきか？

**A.** 「null を含める」方が安全。フィールドの省略は「データがない」と「フィールドが存在しない」の区別がつかない。ただし、PATCH リクエストのボディでは「送信されたフィールドのみ更新」のため、省略とnullに意味の違いがある。

```json
// 推奨: null を明示
{"name": "Alice", "nickname": null, "avatar_url": null}

// 非推奨: フィールド省略（nickname があるのかないのか不明）
{"name": "Alice"}
```

### Q5. API レスポンスの日時フォーマットは？

**A.** ISO 8601 形式（UTC）を標準とする。`"2025-03-15T10:30:00Z"` のように、タイムゾーンは常に Z（UTC）で返す。クライアント側でローカルタイムゾーンに変換する。Unix タイムスタンプは避ける（可読性が低く、精度の解釈が曖昧）。

### Q6. 大容量ファイルのアップロード API はどう設計するか？

**A.** 以下の3段階アプローチを推奨する:

1. **アップロードURL取得**: `POST /api/v1/uploads` → 署名付きURL（presigned URL）を返す
2. **直接アップロード**: クライアントが S3/GCS に直接アップロード（API サーバーを経由しない）
3. **アップロード完了通知**: `POST /api/v1/uploads/{id}/complete` → サーバーがメタデータを保存

これにより API サーバーの帯域を消費せず、大容量ファイル（GB単位）にも対応できる。

---

## 13. まとめ

| 項目 | ポイント |
|------|---------|
| リソース設計 | 名詞ベースの URL、HTTP メソッドで操作を表現。ネスト2階層まで |
| ステータスコード | 2xx/4xx/5xx を正確に使い分け。フローチャートで判断 |
| エラーレスポンス | RFC 7807 準拠（type, title, status, detail, errors） |
| ページネーション | 小規模はオフセット、大規模はカーソル |
| バージョニング | URL パスベース、Sunset ヘッダーで廃止通知 |
| 認証・認可 | JWT + RBAC、セキュリティチェックリスト準拠 |
| レート制限 | トークンバケツ、X-RateLimit-* ヘッダー |
| ドキュメント | OpenAPI (Swagger) で自動生成 + 契約テスト |
| API スタイル選定 | 公開API=REST、BFF=GraphQL、内部通信=gRPC |
| 命名規則 | URL=kebab-case、クエリ=snake_case、JSON=統一 |

```
API 設計の品質チェックフロー:

  設計完了
    │
    ├── リソースとURLは直感的か？
    ├── ステータスコードは正確か？
    ├── エラーレスポンスは統一されているか？
    ├── ページネーションは適切か？
    ├── 認証・認可は全エンドポイントに設定されているか？
    ├── レート制限は設定されているか？
    ├── OpenAPI ドキュメントは最新か？
    ├── 契約テストはパスするか？
    └── 後方互換性は維持されているか？
```

---

## 次に読むべきガイド

- [04-code-review-checklist.md](./04-code-review-checklist.md) — コードレビューチェックリスト（API コードのレビュー観点）
- [../01-practices/04-testing-principles.md](../01-practices/04-testing-principles.md) — テスト原則（API テストの設計）
- [02-functional-principles.md](./02-functional-principles.md) — 関数型プログラミング原則（Result型によるAPIエラーハンドリング）
- [../../system-design-guide/docs/03-case-studies/03-rate-limiter.md](../../system-design-guide/docs/03-case-studies/03-rate-limiter.md) — レートリミッター設計の詳細
- [../../system-design-guide/docs/01-components/](../../system-design-guide/docs/01-components/) — システム設計のコンポーネント（ロードバランサー、キャッシュ）
- [../../design-patterns-guide/docs/04-architectural/](../../design-patterns-guide/docs/04-architectural/) — アーキテクチャパターン（BFF、API Gateway）
- [../../04-web-and-network/](../../04-web-and-network/) — Web/ネットワーク基礎（HTTP、TLS、DNS）

---

## 参考文献

1. **RESTful Web APIs** — Leonard Richardson & Mike Amundsen (O'Reilly, 2013) — REST 設計の包括的ガイド
2. **API Design Patterns** — JJ Geewax (Manning, 2021) — API 設計パターンのカタログ
3. **Google API Design Guide** — https://cloud.google.com/apis/design — Google の API 設計基準
4. **Microsoft REST API Guidelines** — https://github.com/microsoft/api-guidelines — Microsoft の REST API ガイドライン
5. **RFC 7807: Problem Details for HTTP APIs** — https://www.rfc-editor.org/rfc/rfc7807 — エラーレスポンスの標準
6. **Stripe API Reference** — https://stripe.com/docs/api — 優れた API 設計の実例
7. **GitHub REST API** — https://docs.github.com/en/rest — 大規模 REST API の実例
8. **GraphQL Official Documentation** — https://graphql.org/learn/ — GraphQL の公式ドキュメント
9. **gRPC Official Documentation** — https://grpc.io/docs/ — gRPC の公式ドキュメント
10. **OWASP API Security Top 10** — https://owasp.org/www-project-api-security/ — API セキュリティのベストプラクティス
