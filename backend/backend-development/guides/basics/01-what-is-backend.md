# バックエンド開発とは - 完全初心者ガイド

## 目次

1. [概要](#概要)
2. [バックエンドとは何か](#バックエンドとは何か)
3. [フロントエンドとの違い](#フロントエンドとの違い)
4. [バックエンドの役割](#バックエンドの役割)
5. [バックエンド技術スタック](#バックエンド技術スタック)
6. [なぜバックエンドを学ぶのか](#なぜバックエンドを学ぶのか)
7. [次のステップ](#次のステップ)

---

## 概要

### 何を学ぶか

- バックエンド開発の基本概念
- フロントエンドとバックエンドの違い
- バックエンドの責務と役割
- 主要な技術スタック

### 学習時間：30〜40分

---

## バックエンドとは何か

### 定義

**バックエンド**とは、ユーザーの目に見えない「サーバー側」のシステムのことです。

```
┌─────────────┐      HTTP        ┌─────────────┐
│             │  ──────────────→ │             │
│ フロント    │                  │ バックエンド │
│ エンド      │  ←────────────── │             │
│ (ブラウザ)  │      JSON        │  (サーバー) │
└─────────────┘                  └─────────────┘
                                        │
                                        ↓
                                 ┌─────────────┐
                                 │ データベース │
                                 └─────────────┘
```

### 具体例

**Twitterを例に**：

- **フロントエンド**：画面、ボタン、入力フォーム
- **バックエンド**：
  - ユーザー認証（ログイン）
  - ツイートの保存
  - タイムラインの生成
  - 画像のアップロード
  - 通知の送信

---

## フロントエンドとの違い

### フロントエンド

```javascript
// フロントエンド（React）
function Tweet() {
  const [text, setText] = useState('');

  const handleSubmit = async () => {
    await fetch('/api/tweets', {
      method: 'POST',
      body: JSON.stringify({ text })
    });
  };

  return (
    <form onSubmit={handleSubmit}>
      <textarea value={text} onChange={e => setText(e.target.value)} />
      <button>ツイート</button>
    </form>
  );
}
```

### バックエンド

```python
# バックエンド（FastAPI）
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

app = FastAPI()

@app.post("/api/tweets")
async def create_tweet(
    text: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # データベースに保存
    tweet = Tweet(text=text, user_id=user.id)
    db.add(tweet)
    db.commit()
    return {"id": tweet.id, "text": tweet.text}
```

### 比較表

| 項目 | フロントエンド | バックエンド |
|------|--------------|------------|
| **実行場所** | ブラウザ | サーバー |
| **言語** | HTML、CSS、JavaScript | Python、Node.js、Go、Java等 |
| **主な役割** | UI/UX、ユーザーインタラクション | ビジネスロジック、データ処理 |
| **データ** | 表示のみ | 保存・更新・削除 |
| **セキュリティ** | クライアント側（信頼できない） | サーバー側（信頼できる） |

---

## バックエンドの役割

### 1. API提供

フロントエンドが使うAPIを提供します。

```python
# GET /api/users/:id - ユーザー情報取得
@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    return {"id": user_id, "name": "太郎", "email": "taro@example.com"}

# POST /api/users - ユーザー作成
@app.post("/api/users")
async def create_user(name: str, email: str):
    # データベースに保存
    return {"id": 1, "name": name, "email": email}
```

### 2. データベース操作

データの永続化を担当します。

```python
# ユーザーの取得
user = db.query(User).filter(User.id == user_id).first()

# ユーザーの作成
new_user = User(name="太郎", email="taro@example.com")
db.add(new_user)
db.commit()

# ユーザーの更新
user.name = "次郎"
db.commit()

# ユーザーの削除
db.delete(user)
db.commit()
```

### 3. 認証・認可

誰がアクセスしているか、何ができるかを管理します。

```python
# 認証（Authentication）
@app.post("/login")
async def login(email: str, password: str):
    user = authenticate_user(email, password)
    token = create_access_token(user.id)
    return {"token": token}

# 認可（Authorization）
@app.get("/admin/users")
async def get_all_users(current_user: User = Depends(get_current_admin_user)):
    # 管理者のみアクセス可能
    return db.query(User).all()
```

### 4. ビジネスロジック

アプリケーション固有のルールを実装します。

```python
# 例：購入処理
@app.post("/purchase")
async def purchase(product_id: int, user: User = Depends(get_current_user)):
    # 在庫確認
    product = db.query(Product).filter(Product.id == product_id).first()
    if product.stock <= 0:
        raise HTTPException(400, "在庫切れです")

    # ポイント確認
    if user.points < product.price:
        raise HTTPException(400, "ポイントが不足しています")

    # 購入処理
    user.points -= product.price
    product.stock -= 1
    order = Order(user_id=user.id, product_id=product.id)
    db.add(order)
    db.commit()

    return {"message": "購入完了"}
```

---

## バックエンド技術スタック

### プログラミング言語

| 言語 | フレームワーク | 特徴 |
|------|--------------|------|
| **Python** | Django、FastAPI、Flask | 学習しやすい、AI/ML統合 |
| **JavaScript/TypeScript** | Node.js、Express、NestJS | フロントと同じ言語 |
| **Go** | Gin、Echo | 高速、並行処理に強い |
| **Java** | Spring Boot | 大規模エンタープライズ |
| **Ruby** | Ruby on Rails | 開発速度が速い |

### データベース

**リレーショナル（SQL）**：
- PostgreSQL
- MySQL
- SQLite

**NoSQL**：
- MongoDB（ドキュメント）
- Redis（キャッシュ）
- Elasticsearch（検索）

### インフラ

- **クラウド**：AWS、GCP、Azure
- **コンテナ**：Docker、Kubernetes
- **CI/CD**：GitHub Actions、CircleCI

---

## なぜバックエンドを学ぶのか

### 1. 需要が高い

2024年現在、バックエンドエンジニアの求人は多数：
- Web API開発
- マイクロサービス
- データ基盤構築
- DevOps

### 2. フルスタック開発が可能

フロントエンド + バックエンド = **フルスタックエンジニア**

### 3. スケーラブルなシステム構築

- 大量のユーザーを支えるシステム
- 高可用性（ダウンしない）
- パフォーマンス最適化

### 4. 多様なキャリアパス

- バックエンドエンジニア
- インフラエンジニア
- アーキテクト
- SRE（Site Reliability Engineer）

---

## 次のステップ

### このガイドで学んだこと

- ✅ バックエンドの基本概念
- ✅ フロントエンドとの違い
- ✅ バックエンドの役割
- ✅ 技術スタック

### 次に学ぶべきガイド

1. **[02-http-basics.md](./02-http-basics.md)** - HTTP、リクエスト/レスポンス
2. **[03-rest-api-intro.md](./03-rest-api-intro.md)** - REST APIの基本

---

**次のガイド**：[02-http-basics.md](./02-http-basics.md)

**親ガイド**：[Backend Development - SKILL.md](../../SKILL.md)
