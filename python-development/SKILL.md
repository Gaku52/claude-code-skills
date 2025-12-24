---
name: python-development
description: Python開発ガイド。FastAPI、Django、Flask、型ヒント、非同期処理、データ処理など、Pythonアプリケーション開発のベストプラクティス。
---

# Python Development Skill

## 📋 目次

1. [概要](#概要)
2. [いつ使うか](#いつ使うか)
3. [FastAPI](#fastapi)
4. [Django](#django)
5. [型ヒント](#型ヒント)
6. [非同期処理](#非同期処理)
7. [実践例](#実践例)
8. [Agent連携](#agent連携)

---

## 概要

このSkillは、Python開発をカバーします：

- **FastAPI** - モダンAPI フレームワーク
- **Django** - フルスタックWebフレームワーク
- **型ヒント** - 型安全性向上
- **非同期処理** - async/await
- **データ処理** - Pandas, NumPy
- **テスト** - Pytest

---

## いつ使うか

### 🎯 必須のタイミング

- [ ] 新規Python プロジェクト作成時
- [ ] API開発時（FastAPI）
- [ ] Webアプリ開発時（Django）
- [ ] データ処理スクリプト作成時

---

## FastAPI

### プロジェクトセットアップ

```bash
# 仮想環境作成
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate

# パッケージインストール
pip install fastapi uvicorn sqlalchemy pydantic
```

### 基本的なAPI

```python
# main.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    description: str | None = None

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}

@app.post("/items/")
async def create_item(item: Item):
    return item

# 起動: uvicorn main:app --reload
```

### データベース統合（SQLAlchemy）

```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./app.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# models.py
from sqlalchemy import Column, Integer, String
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)

# schemas.py
from pydantic import BaseModel, EmailStr

class UserBase(BaseModel):
    name: str
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int

    class Config:
        from_attributes = True

# crud.py
from sqlalchemy.orm import Session
import models, schemas

def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()

def create_user(db: Session, user: schemas.UserCreate):
    db_user = models.User(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# main.py
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
import crud, models, schemas
from database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users/", response_model=list[schemas.UserResponse])
async def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users

@app.post("/users/", response_model=schemas.UserResponse)
async def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    return crud.create_user(db=db, user=user)
```

### バリデーション

```python
from pydantic import BaseModel, validator, EmailStr, Field

class User(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    age: int = Field(..., ge=0, le=120)

    @validator('name')
    def name_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v

    @validator('age')
    def age_must_be_adult(cls, v):
        if v < 18:
            raise ValueError('Must be 18 or older')
        return v
```

### 認証（JWT）

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # ユーザー取得処理
    return username

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # ユーザー認証処理
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(current_user: str = Depends(get_current_user)):
    return {"username": current_user}
```

---

## Django

### プロジェクト作成

```bash
# インストール
pip install django

# プロジェクト作成
django-admin startproject myproject
cd myproject

# アプリ作成
python manage.py startapp users

# マイグレーション
python manage.py makemigrations
python manage.py migrate

# スーパーユーザー作成
python manage.py createsuperuser

# サーバー起動
python manage.py runserver
```

### モデル

```python
# users/models.py
from django.db import models

class User(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

class Post(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='posts')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.title
```

### ビュー

```python
# users/views.py
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from .models import User

def user_list(request):
    users = User.objects.all()
    return JsonResponse({'users': list(users.values())})

def user_detail(request, pk):
    user = get_object_or_404(User, pk=pk)
    return JsonResponse({
        'id': user.id,
        'name': user.name,
        'email': user.email
    })
```

### Django REST Framework

```bash
pip install djangorestframework
```

```python
# users/serializers.py
from rest_framework import serializers
from .models import User

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'name', 'email', 'created_at']

# users/views.py
from rest_framework import viewsets
from .models import User
from .serializers import UserSerializer

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer

# urls.py
from rest_framework.routers import DefaultRouter
from users.views import UserViewSet

router = DefaultRouter()
router.register(r'users', UserViewSet)

urlpatterns = router.urls
```

---

## 型ヒント

### 基本的な型ヒント

```python
# 基本型
def greet(name: str) -> str:
    return f"Hello, {name}"

# リスト
def process_numbers(numbers: list[int]) -> list[int]:
    return [n * 2 for n in numbers]

# 辞書
def get_user_info(user_id: int) -> dict[str, str]:
    return {"id": str(user_id), "name": "John"}

# Optional
from typing import Optional

def find_user(user_id: int) -> Optional[dict]:
    # ユーザーが見つからない場合はNone
    return None

# Union（Python 3.10+は | 記法）
def process_value(value: int | str) -> str:
    return str(value)
```

### 高度な型ヒント

```python
from typing import TypedDict, Callable

# TypedDict
class UserDict(TypedDict):
    id: int
    name: str
    email: str

def create_user() -> UserDict:
    return {"id": 1, "name": "John", "email": "john@example.com"}

# Callable
def apply_function(func: Callable[[int], int], value: int) -> int:
    return func(value)

# ジェネリクス
from typing import TypeVar, Generic

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self) -> None:
        self.items: list[T] = []

    def push(self, item: T) -> None:
        self.items.append(item)

    def pop(self) -> T:
        return self.items.pop()

stack: Stack[int] = Stack()
stack.push(1)
```

---

## 非同期処理

### async/await

```python
import asyncio
import aiohttp

# 非同期関数
async def fetch_data(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

# 並列実行
async def fetch_all(urls: list[str]) -> list[dict]:
    tasks = [fetch_data(url) for url in urls]
    return await asyncio.gather(*tasks)

# 実行
asyncio.run(fetch_all(["https://api.example.com/1", "https://api.example.com/2"]))
```

### FastAPIでの非同期

```python
from fastapi import FastAPI
import httpx

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com/users/{user_id}")
        return response.json()
```

---

## 実践例

### Example 1: FastAPI CRUD

```python
# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class User(BaseModel):
    id: int
    name: str
    email: str

class UserCreate(BaseModel):
    name: str
    email: str

users_db: dict[int, User] = {}
next_id = 1

@app.get("/users/", response_model=list[User])
async def get_users():
    return list(users_db.values())

@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int):
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    return users_db[user_id]

@app.post("/users/", response_model=User, status_code=201)
async def create_user(user: UserCreate):
    global next_id
    new_user = User(id=next_id, **user.dict())
    users_db[next_id] = new_user
    next_id += 1
    return new_user

@app.delete("/users/{user_id}", status_code=204)
async def delete_user(user_id: int):
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    del users_db[user_id]
```

---

## Agent連携

### 📖 Agentへの指示例

**FastAPI CRUD作成**
```
FastAPIで/api/postsのCRUD APIを作成してください。
Pydantic BaseModelでバリデーションを含めてください。
```

**Django モデル作成**
```
Djangoで以下のモデルを作成してください：
- User（name, email）
- Post（title, content, author）
マイグレーションファイルも生成してください。
```

---

## まとめ

### Pythonのベストプラクティス

1. **型ヒント** - 型安全性向上
2. **FastAPI** - モダンAPI開発
3. **非同期処理** - パフォーマンス向上
4. **Pydantic** - データバリデーション

---

_Last updated: 2025-12-24_
