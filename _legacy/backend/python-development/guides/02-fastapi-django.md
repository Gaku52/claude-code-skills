# 🚀 FastAPI & Django 開発ガイド

> **目的**: FastAPI と Django を使った高品質な Web アプリケーション開発の実践的な手法を習得する

## 📚 目次

1. [FastAPI 基礎](#fastapi-基礎)
2. [FastAPI + SQLAlchemy](#fastapi--sqlalchemy)
3. [FastAPI 認証・認可](#fastapi-認証認可)
4. [Django 基礎](#django-基礎)
5. [Django REST Framework](#django-rest-framework)
6. [パフォーマンス最適化](#パフォーマンス最適化)

---

## FastAPI 基礎

### プロジェクトセットアップ

```bash
# プロジェクト作成
mkdir myapi && cd myapi
poetry init

# 依存関係追加
poetry add fastapi uvicorn[standard] pydantic pydantic-settings
poetry add --group dev pytest httpx ruff mypy

# ディレクトリ構造
mkdir -p src/myapi/{api,models,schemas,services}
touch src/myapi/__init__.py
```

**ディレクトリ構造**:
```
myapi/
├── src/
│   └── myapi/
│       ├── __init__.py
│       ├── main.py           # アプリケーションエントリーポイント
│       ├── config.py         # 設定管理
│       ├── api/              # API エンドポイント
│       │   ├── __init__.py
│       │   ├── deps.py       # 依存性注入
│       │   └── v1/
│       │       ├── __init__.py
│       │       ├── users.py
│       │       └── posts.py
│       ├── models/           # データベースモデル
│       │   ├── __init__.py
│       │   └── user.py
│       ├── schemas/          # Pydantic スキーマ
│       │   ├── __init__.py
│       │   └── user.py
│       └── services/         # ビジネスロジック
│           ├── __init__.py
│           └── user.py
├── tests/
├── pyproject.toml
└── .env
```

### 基本的な API エンドポイント

**src/myapi/main.py**:
```python
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

app = FastAPI(
    title="My API",
    description="FastAPI Example",
    version="1.0.0",
)


class User(BaseModel):
    id: int
    name: str = Field(..., min_length=1, max_length=100)
    email: str


class UserCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: str


# In-memory storage (デモ用)
users_db: dict[int, User] = {}
next_id = 1


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/users", response_model=list[User])
async def list_users():
    """全ユーザー一覧を取得"""
    return list(users_db.values())


@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int):
    """特定ユーザーを取得"""
    if user_id not in users_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return users_db[user_id]


@app.post("/users", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate):
    """ユーザーを作成"""
    global next_id
    new_user = User(id=next_id, **user.model_dump())
    users_db[next_id] = new_user
    next_id += 1
    return new_user


@app.put("/users/{user_id}", response_model=User)
async def update_user(user_id: int, user: UserCreate):
    """ユーザーを更新"""
    if user_id not in users_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    updated_user = User(id=user_id, **user.model_dump())
    users_db[user_id] = updated_user
    return updated_user


@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: int):
    """ユーザーを削除"""
    if user_id not in users_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    del users_db[user_id]
```

**実行**:
```bash
# 開発サーバー起動
poetry run uvicorn src.myapi.main:app --reload

# ドキュメント確認
# http://localhost:8000/docs (Swagger UI)
# http://localhost:8000/redoc (ReDoc)
```

### 環境変数管理

**src/myapi/config.py**:
```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # アプリケーション設定
    app_name: str = "My API"
    debug: bool = False

    # データベース設定
    database_url: str = "postgresql://user:password@localhost/dbname"

    # セキュリティ設定
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # CORS設定
    cors_origins: list[str] = ["http://localhost:3000"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
```

**.env**:
```bash
SECRET_KEY="your-secret-key-here-change-in-production"
DATABASE_URL="postgresql://user:password@localhost/mydb"
DEBUG=true
CORS_ORIGINS=["http://localhost:3000","http://localhost:8000"]
```

**main.py に CORS 追加**:
```python
from fastapi.middleware.cors import CORSMiddleware
from src.myapi.config import settings

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## FastAPI + SQLAlchemy

### セットアップ

```bash
# 依存関係追加
poetry add sqlalchemy alembic psycopg2-binary
```

### データベースモデル

**src/myapi/database.py**:
```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from src.myapi.config import settings

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """依存性注入用のDB セッション"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

**src/myapi/models/user.py**:
```python
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func

from src.myapi.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
```

### Pydantic スキーマ

**src/myapi/schemas/user.py**:
```python
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field


class UserBase(BaseModel):
    email: EmailStr
    name: str = Field(..., min_length=1, max_length=100)


class UserCreate(UserBase):
    password: str = Field(..., min_length=8)


class UserUpdate(BaseModel):
    email: EmailStr | None = None
    name: str | None = Field(None, min_length=1, max_length=100)
    password: str | None = Field(None, min_length=8)


class UserInDB(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime | None

    model_config = {"from_attributes": True}


class User(UserInDB):
    """API レスポンス用（hashed_password を含まない）"""
    pass
```

### CRUD 操作

**src/myapi/services/user.py**:
```python
from sqlalchemy.orm import Session
from passlib.context import CryptContext

from src.myapi.models.user import User as UserModel
from src.myapi.schemas.user import UserCreate, UserUpdate


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_user(db: Session, user_id: int) -> UserModel | None:
    return db.query(UserModel).filter(UserModel.id == user_id).first()


def get_user_by_email(db: Session, email: str) -> UserModel | None:
    return db.query(UserModel).filter(UserModel.email == email).first()


def get_users(db: Session, skip: int = 0, limit: int = 100) -> list[UserModel]:
    return db.query(UserModel).offset(skip).limit(limit).all()


def create_user(db: Session, user: UserCreate) -> UserModel:
    hashed_password = get_password_hash(user.password)
    db_user = UserModel(
        email=user.email,
        name=user.name,
        hashed_password=hashed_password,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def update_user(db: Session, user_id: int, user: UserUpdate) -> UserModel | None:
    db_user = get_user(db, user_id)
    if not db_user:
        return None

    update_data = user.model_dump(exclude_unset=True)
    if "password" in update_data:
        update_data["hashed_password"] = get_password_hash(update_data.pop("password"))

    for field, value in update_data.items():
        setattr(db_user, field, value)

    db.commit()
    db.refresh(db_user)
    return db_user


def delete_user(db: Session, user_id: int) -> bool:
    db_user = get_user(db, user_id)
    if not db_user:
        return False
    db.delete(db_user)
    db.commit()
    return True
```

### API エンドポイント

**src/myapi/api/v1/users.py**:
```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.myapi.database import get_db
from src.myapi.schemas.user import User, UserCreate, UserUpdate
from src.myapi.services import user as user_service

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/", response_model=list[User])
def list_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """全ユーザー一覧を取得"""
    users = user_service.get_users(db, skip=skip, limit=limit)
    return users


@router.get("/{user_id}", response_model=User)
def get_user(user_id: int, db: Session = Depends(get_db)):
    """特定ユーザーを取得"""
    user = user_service.get_user(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user


@router.post("/", response_model=User, status_code=status.HTTP_201_CREATED)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """ユーザーを作成"""
    db_user = user_service.get_user_by_email(db, user.email)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    return user_service.create_user(db, user)


@router.put("/{user_id}", response_model=User)
def update_user(user_id: int, user: UserUpdate, db: Session = Depends(get_db)):
    """ユーザーを更新"""
    db_user = user_service.update_user(db, user_id, user)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return db_user


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int, db: Session = Depends(get_db)):
    """ユーザーを削除"""
    success = user_service.delete_user(db, user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
```

**src/myapi/main.py** (更新):
```python
from fastapi import FastAPI
from src.myapi.api.v1 import users

app = FastAPI(title="My API", version="1.0.0")

# ルーター登録
app.include_router(users.router, prefix="/api/v1")
```

### Alembic マイグレーション

```bash
# 初期化
poetry run alembic init alembic

# マイグレーション作成
poetry run alembic revision --autogenerate -m "Create users table"

# マイグレーション適用
poetry run alembic upgrade head

# ロールバック
poetry run alembic downgrade -1
```

**alembic/env.py** (設定):
```python
from src.myapi.database import Base
from src.myapi.models.user import User  # すべてのモデルをインポート
from src.myapi.config import settings

# target_metadata を設定
target_metadata = Base.metadata

# database_url を設定
config.set_main_option("sqlalchemy.url", settings.database_url)
```

---

## FastAPI 認証・認可

### JWT トークン認証

```bash
# 依存関係追加
poetry add python-jose[cryptography] passlib[bcrypt]
```

**src/myapi/auth.py**:
```python
from datetime import datetime, timedelta
from typing import Any

from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from src.myapi.config import settings
from src.myapi.database import get_db
from src.myapi.services import user as user_service

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """アクセストークン作成"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """現在のユーザーを取得（認証必須）"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = user_service.get_user(db, user_id=user_id)
    if user is None:
        raise credentials_exception

    return user


def get_current_active_user(current_user = Depends(get_current_user)):
    """アクティブなユーザーを取得"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user
```

**src/myapi/api/v1/auth.py**:
```python
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel

from src.myapi.database import get_db
from src.myapi.services.user import verify_password, get_user_by_email
from src.myapi.auth import create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


class Token(BaseModel):
    access_token: str
    token_type: str


@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """ログイン（トークン取得）"""
    user = get_user_by_email(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user.id})
    return {"access_token": access_token, "token_type": "bearer"}
```

**使用例（認証が必要なエンドポイント）**:
```python
from fastapi import APIRouter, Depends
from src.myapi.auth import get_current_active_user
from src.myapi.schemas.user import User

router = APIRouter()


@router.get("/me", response_model=User)
def read_users_me(current_user: User = Depends(get_current_active_user)):
    """現在のユーザー情報を取得"""
    return current_user
```

---

## Django 基礎

### プロジェクトセットアップ

```bash
# プロジェクト作成
mkdir myproject && cd myproject
poetry init
poetry add django psycopg2-binary
poetry add --group dev pytest pytest-django

# Django プロジェクト作成
poetry run django-admin startproject config .
poetry run python manage.py startapp users
```

**ディレクトリ構造**:
```
myproject/
├── config/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── users/
│   ├── migrations/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── views.py
│   └── urls.py
├── manage.py
└── pyproject.toml
```

### 環境変数管理

```bash
# 依存関係追加
poetry add django-environ
```

**config/settings.py**:
```python
import environ

env = environ.Env(
    DEBUG=(bool, False)
)

# .env ファイル読み込み
environ.Env.read_env()

SECRET_KEY = env('SECRET_KEY')
DEBUG = env('DEBUG')

DATABASES = {
    'default': env.db()  # DATABASE_URL から自動解析
}

ALLOWED_HOSTS = env.list('ALLOWED_HOSTS', default=[])
```

**.env**:
```bash
SECRET_KEY="your-secret-key-here"
DEBUG=True
DATABASE_URL="postgresql://user:password@localhost/mydb"
ALLOWED_HOSTS=localhost,127.0.0.1
```

### モデル定義

**users/models.py**:
```python
from django.db import models
from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    """カスタムユーザーモデル"""
    email = models.EmailField(unique=True)
    bio = models.TextField(blank=True)
    avatar = models.ImageField(upload_to='avatars/', blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

    def __str__(self):
        return self.email


class Post(models.Model):
    """投稿モデル"""
    title = models.CharField(max_length=200)
    content = models.TextField()
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='posts')
    published = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at']),
        ]

    def __str__(self):
        return self.title
```

**config/settings.py** (AUTH_USER_MODEL 設定):
```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'users',  # アプリを追加
]

AUTH_USER_MODEL = 'users.User'
```

### マイグレーション

```bash
# マイグレーション作成
poetry run python manage.py makemigrations

# マイグレーション適用
poetry run python manage.py migrate

# スーパーユーザー作成
poetry run python manage.py createsuperuser
```

### Admin サイト

**users/admin.py**:
```python
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User, Post


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ['email', 'username', 'is_staff', 'created_at']
    list_filter = ['is_staff', 'is_active', 'created_at']
    search_fields = ['email', 'username']
    ordering = ['-created_at']

    fieldsets = BaseUserAdmin.fieldsets + (
        ('Additional Info', {'fields': ('bio', 'avatar')}),
    )


@admin.register(Post)
class PostAdmin(admin.ModelAdmin):
    list_display = ['title', 'author', 'published', 'created_at']
    list_filter = ['published', 'created_at']
    search_fields = ['title', 'content']
    raw_id_fields = ['author']
    date_hierarchy = 'created_at'
```

---

## Django REST Framework

### セットアップ

```bash
# 依存関係追加
poetry add djangorestframework djangorestframework-simplejwt
```

**config/settings.py**:
```python
INSTALLED_APPS = [
    # ...
    'rest_framework',
    'users',
]

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticatedOrReadOnly',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 10,
}
```

### シリアライザー

**users/serializers.py**:
```python
from rest_framework import serializers
from .models import User, Post


class UserSerializer(serializers.ModelSerializer):
    posts_count = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = ['id', 'email', 'username', 'bio', 'avatar', 'posts_count', 'created_at']
        read_only_fields = ['id', 'created_at']

    def get_posts_count(self, obj):
        return obj.posts.count()


class UserCreateSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=8)

    class Meta:
        model = User
        fields = ['email', 'username', 'password']

    def create(self, validated_data):
        user = User.objects.create_user(**validated_data)
        return user


class PostSerializer(serializers.ModelSerializer):
    author = UserSerializer(read_only=True)

    class Meta:
        model = Post
        fields = ['id', 'title', 'content', 'author', 'published', 'created_at', 'updated_at']
        read_only_fields = ['id', 'author', 'created_at', 'updated_at']


class PostCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Post
        fields = ['title', 'content', 'published']
```

### ViewSet

**users/views.py**:
```python
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from .models import User, Post
from .serializers import (
    UserSerializer,
    UserCreateSerializer,
    PostSerializer,
    PostCreateSerializer,
)


class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer

    def get_serializer_class(self):
        if self.action == 'create':
            return UserCreateSerializer
        return UserSerializer

    def get_permissions(self):
        if self.action == 'create':
            return [AllowAny()]
        return [IsAuthenticated()]

    @action(detail=False, methods=['get'])
    def me(self, request):
        """現在のユーザー情報を取得"""
        serializer = self.get_serializer(request.user)
        return Response(serializer.data)


class PostViewSet(viewsets.ModelViewSet):
    queryset = Post.objects.select_related('author').all()

    def get_serializer_class(self):
        if self.action in ['create', 'update', 'partial_update']:
            return PostCreateSerializer
        return PostSerializer

    def get_queryset(self):
        queryset = super().get_queryset()
        if not self.request.user.is_authenticated:
            # 未認証ユーザーには公開済みのみ
            queryset = queryset.filter(published=True)
        return queryset

    def perform_create(self, serializer):
        serializer.save(author=self.request.user)

    @action(detail=False, methods=['get'])
    def my_posts(self, request):
        """自分の投稿一覧を取得"""
        posts = self.queryset.filter(author=request.user)
        page = self.paginate_queryset(posts)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = self.get_serializer(posts, many=True)
        return Response(serializer.data)
```

### URL 設定

**users/urls.py**:
```python
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import UserViewSet, PostViewSet

router = DefaultRouter()
router.register('users', UserViewSet)
router.register('posts', PostViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
```

**config/urls.py**:
```python
from django.contrib import admin
from django.urls import path, include
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/', include('users.urls')),
    path('api/v1/auth/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/v1/auth/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
]
```

### テスト

**users/tests.py**:
```python
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from .models import User, Post


class UserAPITestCase(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            email='test@example.com',
            username='testuser',
            password='testpass123'
        )

    def test_create_user(self):
        """ユーザー作成テスト"""
        url = reverse('user-list')
        data = {
            'email': 'newuser@example.com',
            'username': 'newuser',
            'password': 'newpass123'
        }
        response = self.client.post(url, data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(User.objects.count(), 2)

    def test_get_me(self):
        """自分の情報取得テスト"""
        self.client.force_authenticate(user=self.user)
        url = reverse('user-me')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['email'], 'test@example.com')


class PostAPITestCase(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            email='test@example.com',
            username='testuser',
            password='testpass123'
        )
        self.post = Post.objects.create(
            title='Test Post',
            content='Test Content',
            author=self.user,
            published=True
        )

    def test_list_posts(self):
        """投稿一覧取得テスト"""
        url = reverse('post-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['results']), 1)

    def test_create_post_authenticated(self):
        """認証済みユーザーの投稿作成テスト"""
        self.client.force_authenticate(user=self.user)
        url = reverse('post-list')
        data = {
            'title': 'New Post',
            'content': 'New Content',
            'published': False
        }
        response = self.client.post(url, data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(Post.objects.count(), 2)
```

```bash
# テスト実行
poetry run python manage.py test
```

---

## パフォーマンス最適化

### FastAPI 最適化

**非同期処理**:
```python
from fastapi import FastAPI
import httpx

app = FastAPI()


@app.get("/fetch")
async def fetch_data():
    """非同期で外部 API を呼び出し"""
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
        return response.json()


@app.get("/fetch-multiple")
async def fetch_multiple():
    """複数の API を並列で呼び出し"""
    async with httpx.AsyncClient() as client:
        tasks = [
            client.get("https://api.example.com/data1"),
            client.get("https://api.example.com/data2"),
            client.get("https://api.example.com/data3"),
        ]
        responses = await asyncio.gather(*tasks)
        return [r.json() for r in responses]
```

**キャッシュ**:
```python
from functools import lru_cache
from fastapi import Depends

@lru_cache()
def get_settings():
    """設定をキャッシュ"""
    return Settings()


@app.get("/config")
def get_config(settings: Settings = Depends(get_settings)):
    return {"app_name": settings.app_name}
```

**バックグラウンドタスク**:
```python
from fastapi import BackgroundTasks

def send_email(email: str, message: str):
    """メール送信（時間がかかる処理）"""
    print(f"Sending email to {email}: {message}")


@app.post("/send-notification")
async def send_notification(email: str, background_tasks: BackgroundTasks):
    """通知送信（バックグラウンドで実行）"""
    background_tasks.add_task(send_email, email, "Welcome!")
    return {"message": "Notification sent"}
```

### Django 最適化

**N+1 問題解決**:
```python
# ❌ N+1 問題
posts = Post.objects.all()
for post in posts:
    print(post.author.email)  # 各投稿ごとに DB クエリ

# ✅ select_related（1対1、ForeignKey）
posts = Post.objects.select_related('author').all()
for post in posts:
    print(post.author.email)  # 1回の JOIN で取得

# ✅ prefetch_related（ManyToMany、逆参照）
users = User.objects.prefetch_related('posts').all()
for user in users:
    print(user.posts.count())  # 2回のクエリで取得
```

**インデックス追加**:
```python
class Post(models.Model):
    # ...

    class Meta:
        indexes = [
            models.Index(fields=['-created_at']),
            models.Index(fields=['author', 'published']),
        ]
```

**クエリ最適化**:
```python
# ❌ 全フィールド取得
posts = Post.objects.all()

# ✅ 必要なフィールドのみ
posts = Post.objects.only('id', 'title', 'created_at')

# ✅ 除外
posts = Post.objects.defer('content')
```

**キャッシュ**:
```bash
# Redis インストール
poetry add django-redis
```

**config/settings.py**:
```python
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}
```

**使用例**:
```python
from django.core.cache import cache

def get_posts():
    """投稿一覧をキャッシュ"""
    posts = cache.get('posts')
    if posts is None:
        posts = list(Post.objects.select_related('author').all())
        cache.set('posts', posts, 60 * 15)  # 15分キャッシュ
    return posts
```

---

## まとめ

### FastAPI チェックリスト

**基本**:
- [ ] Pydantic スキーマで型安全性確保
- [ ] 環境変数管理（pydantic-settings）
- [ ] CORS 設定

**データベース**:
- [ ] SQLAlchemy でモデル定義
- [ ] Alembic でマイグレーション管理
- [ ] 依存性注入でセッション管理

**認証**:
- [ ] JWT トークン認証
- [ ] パスワードハッシュ化（bcrypt）
- [ ] 依存性注入で認証チェック

**最適化**:
- [ ] 非同期処理（async/await）
- [ ] バックグラウンドタスク
- [ ] キャッシュ活用

### Django チェックリスト

**基本**:
- [ ] カスタムユーザーモデル
- [ ] 環境変数管理（django-environ）
- [ ] Admin サイト活用

**REST API**:
- [ ] Django REST Framework
- [ ] Serializer で検証
- [ ] ViewSet で CRUD 実装

**認証**:
- [ ] JWT 認証（simplejwt）
- [ ] パーミッション設定

**最適化**:
- [ ] N+1 問題解決（select_related/prefetch_related）
- [ ] インデックス追加
- [ ] Redis キャッシュ

---

## 次のステップ

1. **03-data-processing.md**: データ処理・自動化ガイド

---

*型安全で高性能な Web アプリケーションを構築しましょう。*
