# 認証の基礎 - 完全初心者ガイド

## 目次

1. [概要](#概要)
2. [認証と認可の違い](#認証と認可の違い)
3. [認証方式](#認証方式)
4. [JWTトークン](#jwtトークン)
5. [パスワードのハッシュ化](#パスワードのハッシュ化)
6. [実装例](#実装例)
7. [セキュリティのベストプラクティス](#セキュリティのベストプラクティス)
8. [次のステップ](#次のステップ)

---

## 概要

### 何を学ぶか

- 認証（Authentication）と認可（Authorization）
- パスワード認証とJWT
- セキュアなパスワード管理
- FastAPIでの認証実装

### 学習時間：1〜2時間

---

## 認証と認可の違い

### 認証（Authentication）

**「あなたは誰ですか？」**

ユーザーが本人であることを確認するプロセス。

```python
# ログイン（認証）
@app.post("/login")
def login(email: str, password: str):
    user = authenticate(email, password)  # 本人確認
    return {"token": create_token(user.id)}
```

### 認可（Authorization）

**「あなたは何ができますか？」**

ユーザーが特定のリソースにアクセスできるかを確認するプロセス。

```python
# 管理者専用エンドポイント（認可）
@app.get("/admin/users")
def get_all_users(current_user: User = Depends(get_current_user)):
    if not current_user.is_admin:  # 権限確認
        raise HTTPException(403, "管理者権限が必要です")
    return db.query(User).all()
```

### 比較表

| 項目 | 認証 | 認可 |
|------|------|------|
| **質問** | あなたは誰？ | 何ができる？ |
| **タイミング** | ログイン時 | リソースアクセス時 |
| **例** | ユーザー名とパスワード | 管理者権限チェック |

---

## 認証方式

### 1. セッションベース認証

```python
# サーバー側でセッションを保持
sessions = {}

@app.post("/login")
def login(email: str, password: str):
    user = authenticate(email, password)
    session_id = generate_session_id()
    sessions[session_id] = user.id  # サーバーに保存
    return {"session_id": session_id}

@app.get("/profile")
def get_profile(session_id: str):
    user_id = sessions.get(session_id)
    if not user_id:
        raise HTTPException(401, "認証が必要です")
    return get_user(user_id)
```

**欠点**：
- サーバーがセッション情報を保持（ステートフル）
- スケーリングが困難

### 2. トークンベース認証（JWT）

```python
# トークンをクライアントに渡す（ステートレス）
@app.post("/login")
def login(email: str, password: str):
    user = authenticate(email, password)
    token = create_jwt_token(user.id)  # JWT生成
    return {"token": token}

@app.get("/profile")
def get_profile(token: str = Header(...)):
    user_id = decode_jwt_token(token)  # JWT検証
    return get_user(user_id)
```

**利点**：
- ステートレス（サーバーは状態を持たない）
- スケーラブル
- マイクロサービスに適している

---

## JWTトークン

### JWTの構造

```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoxLCJleHAiOjE2MzI4NTc2MDB9.abc123...
│                                        │                                    │
└── Header                               └── Payload                          └── Signature
```

#### 1. Header（ヘッダー）

```json
{
  "alg": "HS256",
  "typ": "JWT"
}
```

#### 2. Payload（ペイロード）

```json
{
  "user_id": 1,
  "email": "taro@example.com",
  "exp": 1632857600
}
```

#### 3. Signature（署名）

```
HMACSHA256(
  base64UrlEncode(header) + "." + base64UrlEncode(payload),
  secret
)
```

### JWTの生成と検証

```python
import jwt
from datetime import datetime, timedelta

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

def create_jwt_token(user_id: int) -> str:
    """JWTトークンを生成"""
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(hours=24)  # 24時間有効
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token

def decode_jwt_token(token: str) -> int:
    """JWTトークンを検証してuser_idを返す"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["user_id"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "トークンの有効期限が切れています")
    except jwt.JWTError:
        raise HTTPException(401, "無効なトークンです")
```

---

## パスワードのハッシュ化

### なぜハッシュ化が必要か

**❌ 平文保存**：データベースが漏洩したら全ユーザーのパスワードが露出

**✅ ハッシュ化**：元に戻せない形式で保存

### bcryptによるハッシュ化

```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """パスワードをハッシュ化"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """パスワードを検証"""
    return pwd_context.verify(plain_password, hashed_password)

# 使用例
hashed = hash_password("mypassword123")
# $2b$12$N9qo8uLOickgx2ZMRZoMye...

is_valid = verify_password("mypassword123", hashed)
# True
```

### ソルト（Salt）

**ソルト**は、同じパスワードでも異なるハッシュを生成するためのランダムな文字列です。

```python
# bcryptは自動的にソルトを追加
hash1 = hash_password("password123")  # $2b$12$abc...
hash2 = hash_password("password123")  # $2b$12$xyz...（異なる）
```

---

## 実装例

### 完全な認証システム

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta

app = FastAPI()
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = "your-secret-key-change-this"
ALGORITHM = "HS256"

# モデル
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)

# スキーマ
class UserRegister(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# パスワード関連
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

# JWT関連
def create_access_token(user_id: int) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> int:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["user_id"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "トークンの有効期限が切れています")
    except jwt.JWTError:
        raise HTTPException(401, "無効なトークンです")

# 依存関数
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    token = credentials.credentials
    user_id = decode_token(token)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "ユーザーが見つかりません")
    return user

# エンドポイント

@app.post("/register")
def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """新規登録"""
    # メールの重複チェック
    existing = db.query(User).filter(User.email == user_data.email).first()
    if existing:
        raise HTTPException(400, "このメールアドレスは既に登録されています")

    # ユーザー作成
    user = User(
        email=user_data.email,
        hashed_password=hash_password(user_data.password)
    )
    db.add(user)
    db.commit()
    return {"message": "登録完了"}

@app.post("/login", response_model=Token)
def login(email: str, password: str, db: Session = Depends(get_db)):
    """ログイン"""
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(401, "メールアドレスまたはパスワードが間違っています")

    token = create_access_token(user.id)
    return {"access_token": token, "token_type": "bearer"}

@app.get("/me")
def get_me(current_user: User = Depends(get_current_user)):
    """ログインユーザー情報取得"""
    return {
        "id": current_user.id,
        "email": current_user.email
    }

@app.get("/protected")
def protected_route(current_user: User = Depends(get_current_user)):
    """認証が必要なエンドポイント"""
    return {"message": f"こんにちは、{current_user.email}さん！"}
```

---

## セキュリティのベストプラクティス

### 1. パスワードポリシー

```python
import re

def validate_password(password: str) -> bool:
    """
    パスワードのバリデーション
    - 8文字以上
    - 大文字、小文字、数字を含む
    """
    if len(password) < 8:
        return False
    if not re.search(r"[A-Z]", password):
        return False
    if not re.search(r"[a-z]", password):
        return False
    if not re.search(r"\d", password):
        return False
    return True
```

### 2. レート制限

```python
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

@app.post("/login")
@limiter.limit("5/minute")  # 1分間に5回まで
async def login(email: str, password: str):
    ...
```

### 3. HTTPS必須

```python
# 本番環境ではHTTPSを必須にする
if not request.url.scheme == "https":
    raise HTTPException(400, "HTTPSが必要です")
```

### 4. トークンのリフレッシュ

```python
def create_tokens(user_id: int):
    access_token = create_token(user_id, expires_delta=timedelta(minutes=15))
    refresh_token = create_token(user_id, expires_delta=timedelta(days=7))
    return access_token, refresh_token

@app.post("/refresh")
def refresh(refresh_token: str):
    user_id = decode_token(refresh_token)
    new_access_token = create_token(user_id, timedelta(minutes=15))
    return {"access_token": new_access_token}
```

---

## 次のステップ

### このガイドで学んだこと

- ✅ 認証と認可の違い
- ✅ JWTトークン
- ✅ パスワードのハッシュ化
- ✅ FastAPIでの認証実装

### 次に学ぶべきガイド

**次のガイド**：[06-environment-variables.md](./06-environment-variables.md) - 環境変数と設定管理

---

**前のガイド**：[04-database-intro.md](./04-database-intro.md)

**親ガイド**：[Backend Development - SKILL.md](../../SKILL.md)
