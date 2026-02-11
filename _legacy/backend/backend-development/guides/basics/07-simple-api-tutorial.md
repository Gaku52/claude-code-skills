# シンプルなAPI構築 - 総合演習

## 目次

1. [概要](#概要)
2. [プロジェクトのゴール](#プロジェクトのゴール)
3. [プロジェクトセットアップ](#プロジェクトセットアップ)
4. [データベース設計](#データベース設計)
5. [モデルとスキーマ](#モデルとスキーマ)
6. [認証システム](#認証システム)
7. [CRUD API実装](#crud-api実装)
8. [テストとデバッグ](#テストとデバッグ)
9. [まとめ](#まとめ)

---

## 概要

### 何を学ぶか

このチュートリアルでは、これまで学んだ全ての概念を統合して、**タスク管理API**を実装します。

### 実装する機能

- ✅ ユーザー登録・ログイン
- ✅ JWT認証
- ✅ タスクのCRUD操作
- ✅ 環境変数管理
- ✅ エラーハンドリング

### 学習時間：1〜2時間

---

## プロジェクトのゴール

### 完成するAPI

```
POST   /register          - ユーザー登録
POST   /login             - ログイン
GET    /me                - ログインユーザー情報
GET    /tasks             - タスク一覧
POST   /tasks             - タスク作成
GET    /tasks/:id         - タスク詳細
PUT    /tasks/:id         - タスク更新
DELETE /tasks/:id         - タスク削除
```

---

## プロジェクトセットアップ

### 1. ディレクトリ構成

```
task-api/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPIアプリケーション
│   ├── database.py       # データベース接続
│   ├── models.py         # SQLAlchemyモデル
│   ├── schemas.py        # Pydanticスキーマ
│   ├── auth.py           # 認証ロジック
│   └── config.py         # 設定管理
├── .env                  # 環境変数
├── .env.example          # 環境変数のテンプレート
├── .gitignore
├── requirements.txt
└── README.md
```

### 2. 必要なパッケージをインストール

```bash
# プロジェクトディレクトリ作成
mkdir task-api
cd task-api

# 仮想環境作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# パッケージインストール
pip install fastapi uvicorn sqlalchemy python-dotenv passlib[bcrypt] python-jose[cryptography]
```

### 3. requirements.txt

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
python-dotenv==1.0.0
passlib[bcrypt]==1.7.4
python-jose[cryptography]==3.3.0
pydantic-settings==2.1.0
```

### 4. .env.example

```bash
# .env.example
DATABASE_URL=sqlite:///./task.db
SECRET_KEY=your-secret-key-here-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### 5. .envファイル作成

```bash
cp .env.example .env
# .envを編集してシークレットキーを設定
```

---

## データベース設計

### ER図

```
users テーブル
┌────┬───────┬──────────┬──────────┐
│ id │ email │ password │ name     │
└────┴───────┴──────────┴──────────┘
       │
       │ 1:多
       │
       ↓
tasks テーブル
┌────┬────────┬─────────┬───────────┬─────────┐
│ id │ title  │ done    │ user_id   │ created │
└────┴────────┴─────────┴───────────┴─────────┘
```

---

## モデルとスキーマ

### app/config.py

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    class Config:
        env_file = ".env"

settings = Settings()
```

### app/database.py

```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .config import settings

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False}  # SQLiteのみ
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### app/models.py

```python
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    name = Column(String, nullable=False)

    tasks = relationship("Task", back_populates="owner")

class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(String, nullable=True)
    done = Column(Boolean, default=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="tasks")
```

### app/schemas.py

```python
from pydantic import BaseModel, EmailStr
from datetime import datetime

# ユーザー関連
class UserBase(BaseModel):
    email: EmailStr
    name: str

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int

    class Config:
        from_attributes = True

# トークン関連
class Token(BaseModel):
    access_token: str
    token_type: str

# タスク関連
class TaskBase(BaseModel):
    title: str
    description: str | None = None
    done: bool = False

class TaskCreate(TaskBase):
    pass

class TaskUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    done: bool | None = None

class TaskResponse(TaskBase):
    id: int
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True
```

---

## 認証システム

### app/auth.py

```python
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from .config import settings
from .database import get_db
from .models import User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(user_id: int) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {
        "user_id": user_id,
        "exp": expire
    }
    token = jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)
    return token

def decode_token(token: str) -> int:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        user_id: int = payload.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=401, detail="無効なトークンです")
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="無効なトークンです")

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    token = credentials.credentials
    user_id = decode_token(token)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="ユーザーが見つかりません")
    return user
```

---

## CRUD API実装

### app/main.py

```python
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from .database import engine, get_db, Base
from .models import User, Task
from .schemas import (
    UserCreate, UserResponse, Token,
    TaskCreate, TaskUpdate, TaskResponse
)
from .auth import (
    hash_password, verify_password,
    create_access_token, get_current_user
)

# テーブル作成
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Task API", version="1.0.0")

# ========================================
# 認証エンドポイント
# ========================================

@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """新規ユーザー登録"""
    # メール重複チェック
    existing = db.query(User).filter(User.email == user_data.email).first()
    if existing:
        raise HTTPException(400, "このメールアドレスは既に登録されています")

    # ユーザー作成
    user = User(
        email=user_data.email,
        name=user_data.name,
        hashed_password=hash_password(user_data.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@app.post("/login", response_model=Token)
def login(email: str, password: str, db: Session = Depends(get_db)):
    """ログイン"""
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(401, "メールアドレスまたはパスワードが間違っています")

    token = create_access_token(user.id)
    return {"access_token": token, "token_type": "bearer"}

@app.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """ログインユーザー情報取得"""
    return current_user

# ========================================
# タスクエンドポイント
# ========================================

@app.get("/tasks", response_model=List[TaskResponse])
def get_tasks(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """タスク一覧取得"""
    tasks = db.query(Task)\
        .filter(Task.user_id == current_user.id)\
        .offset(skip)\
        .limit(limit)\
        .all()
    return tasks

@app.post("/tasks", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
def create_task(
    task_data: TaskCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """タスク作成"""
    task = Task(**task_data.dict(), user_id=current_user.id)
    db.add(task)
    db.commit()
    db.refresh(task)
    return task

@app.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task(
    task_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """タスク詳細取得"""
    task = db.query(Task).filter(
        Task.id == task_id,
        Task.user_id == current_user.id
    ).first()
    if not task:
        raise HTTPException(404, "タスクが見つかりません")
    return task

@app.put("/tasks/{task_id}", response_model=TaskResponse)
def update_task(
    task_id: int,
    task_data: TaskUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """タスク更新"""
    task = db.query(Task).filter(
        Task.id == task_id,
        Task.user_id == current_user.id
    ).first()
    if not task:
        raise HTTPException(404, "タスクが見つかりません")

    # 更新
    update_data = task_data.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(task, key, value)

    db.commit()
    db.refresh(task)
    return task

@app.delete("/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_task(
    task_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """タスク削除"""
    task = db.query(Task).filter(
        Task.id == task_id,
        Task.user_id == current_user.id
    ).first()
    if not task:
        raise HTTPException(404, "タスクが見つかりません")

    db.delete(task)
    db.commit()
    return None
```

---

## テストとデバッグ

### サーバー起動

```bash
uvicorn app.main:app --reload
```

### curlでテスト

```bash
# 1. ユーザー登録
curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "name": "テストユーザー",
    "password": "password123"
  }'

# 2. ログイン
curl -X POST http://localhost:8000/login \
  -d "email=test@example.com&password=password123"
# レスポンス: {"access_token":"xxx","token_type":"bearer"}

# トークンを環境変数に保存
export TOKEN="取得したトークン"

# 3. ログインユーザー情報取得
curl -X GET http://localhost:8000/me \
  -H "Authorization: Bearer $TOKEN"

# 4. タスク作成
curl -X POST http://localhost:8000/tasks \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "APIの実装",
    "description": "FastAPIでREST APIを実装する",
    "done": false
  }'

# 5. タスク一覧取得
curl -X GET http://localhost:8000/tasks \
  -H "Authorization: Bearer $TOKEN"

# 6. タスク更新
curl -X PUT http://localhost:8000/tasks/1 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"done": true}'

# 7. タスク削除
curl -X DELETE http://localhost:8000/tasks/1 \
  -H "Authorization: Bearer $TOKEN"
```

### Swagger UIでテスト

ブラウザで http://localhost:8000/docs を開くと、自動生成されたAPIドキュメントが表示されます。

1. `/register`でユーザー登録
2. `/login`でトークン取得
3. 右上の「Authorize」ボタンをクリック
4. `Bearer <token>`を入力
5. 各エンドポイントをテスト

---

## よくあるエラーと解決方法

### エラー1：401 Unauthorized

**原因**：トークンが無効または期限切れ

**解決**：
```bash
# 再ログインしてトークンを取得
curl -X POST http://localhost:8000/login \
  -d "email=test@example.com&password=password123"
```

### エラー2：404 タスクが見つかりません

**原因**：他のユーザーのタスクにアクセスしようとしている

**解決**：自分のタスクIDを確認

```bash
curl -X GET http://localhost:8000/tasks \
  -H "Authorization: Bearer $TOKEN"
```

### エラー3：500 Internal Server Error

**原因**：データベース接続エラー

**解決**：
```python
# database.pyでエラーログを確認
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 拡張アイデア

### 1. ページネーション

```python
@app.get("/tasks")
def get_tasks(
    page: int = 1,
    per_page: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    skip = (page - 1) * per_page
    tasks = db.query(Task)\
        .filter(Task.user_id == current_user.id)\
        .offset(skip)\
        .limit(per_page)\
        .all()

    total = db.query(Task).filter(Task.user_id == current_user.id).count()

    return {
        "tasks": tasks,
        "page": page,
        "per_page": per_page,
        "total": total,
        "pages": (total + per_page - 1) // per_page
    }
```

### 2. フィルタリング

```python
@app.get("/tasks")
def get_tasks(
    done: bool | None = None,
    search: str | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = db.query(Task).filter(Task.user_id == current_user.id)

    if done is not None:
        query = query.filter(Task.done == done)

    if search:
        query = query.filter(Task.title.contains(search))

    return query.all()
```

### 3. ソート

```python
from sqlalchemy import desc, asc

@app.get("/tasks")
def get_tasks(
    sort_by: str = "created_at",
    order: str = "desc",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = db.query(Task).filter(Task.user_id == current_user.id)

    if order == "desc":
        query = query.order_by(desc(getattr(Task, sort_by)))
    else:
        query = query.order_by(asc(getattr(Task, sort_by)))

    return query.all()
```

---

## まとめ

### このチュートリアルで学んだこと

- ✅ FastAPIプロジェクトの構成
- ✅ データベース設計とORM
- ✅ JWT認証の実装
- ✅ CRUD APIの実装
- ✅ 環境変数管理
- ✅ エラーハンドリング
- ✅ APIのテスト方法

### 次のステップ

1. **テストコード追加**：pytestでユニットテスト
2. **デプロイ**：Heroku、Render、AWS等にデプロイ
3. **フロントエンド連携**：React等と連携
4. **機能拡張**：タグ、優先度、期限等を追加

---

**前のガイド**：[06-environment-variables.md](./06-environment-variables.md)

**親ガイド**：[Backend Development - SKILL.md](../../SKILL.md)

**おめでとうございます！** バックエンド開発の基礎を全て学びました。
