# URL 短縮サービス設計

> TinyURL / bit.ly のような URL 短縮サービスをゼロから設計する。ハッシュ生成、リダイレクト最適化、スケーラビリティの観点でシステム設計面接に対応できる力を身につける。

---

## この章で学ぶこと

1. **要件定義とスケール見積もり** — 機能要件・非機能要件の整理と、トラフィック・ストレージの概算
2. **コア設計** — 短縮 URL の生成アルゴリズム、データモデル、リダイレクトフロー
3. **スケーラビリティ** — キャッシュ戦略、データベース分割、高可用性設計

---

## 1. 要件定義

### 1.1 機能要件

- 長い URL を短い URL に変換する（例: `https://example.com/very/long/path` → `https://short.ly/abc123`）
- 短縮 URL にアクセスすると元の URL にリダイレクトする
- カスタムエイリアスを指定できる（オプション）
- URL の有効期限を設定できる（オプション）
- アクセス統計を確認できる（オプション）

### 1.2 非機能要件

- **高可用性**: リダイレクトは 99.99% の可用性が必要
- **低遅延**: リダイレクトは 100ms 以内
- **スケール**: 読み取り:書き込み = 100:1 の読み取り重視

### 1.3 スケール見積もり

```
前提:
  - 1日あたり新規 URL 作成: 100M (1億)
  - 読み取り/書き込み比: 100:1
  - 1日あたりリダイレクト: 10B (100億)
  - URL 保持期間: 5年

書き込み QPS:
  100M / 86400 ≈ 1,160 QPS (ピーク時 ≈ 2,300 QPS)

読み取り QPS:
  10B / 86400 ≈ 115,740 QPS (ピーク時 ≈ 230,000 QPS)

ストレージ:
  5年間の総 URL 数: 100M * 365 * 5 = 182.5B
  1レコード ≈ 500 bytes
  合計: 182.5B * 500 = 91.25 TB
```

---

## 2. コア設計

### 2.1 高レベルアーキテクチャ

```
+--------+     +--------+     +--------+     +----------+
| クライ  | --> | ロード  | --> | API    | --> | 短縮URL   |
| アント  |     | バランサ |     | サーバー|     | 生成     |
+--------+     +--------+     +--------+     +----------+
                                   |               |
                                   v               v
                              +--------+     +----------+
                              | キャッシュ|     | データ    |
                              | (Redis) |     | ベース    |
                              +--------+     +----------+
```

### 2.2 短縮キー生成アルゴリズム

```python
# コード例 1: Base62 エンコーディングによる短縮キー生成
import string

CHARSET = string.digits + string.ascii_lowercase + string.ascii_uppercase
BASE = len(CHARSET)  # 62

def encode_base62(num: int) -> str:
    """整数を Base62 文字列に変換する"""
    if num == 0:
        return CHARSET[0]
    result = []
    while num > 0:
        result.append(CHARSET[num % BASE])
        num //= BASE
    return ''.join(reversed(result))

def decode_base62(s: str) -> int:
    """Base62 文字列を整数に戻す"""
    num = 0
    for char in s:
        num = num * BASE + CHARSET.index(char)
    return num

# 7文字のBase62で表現可能な範囲
# 62^7 = 3,521,614,606,208 (約3.5兆通り)
print(encode_base62(123456789))  # => "8M0kX"
```

### 2.3 ハッシュ方式 vs カウンター方式

| 方式 | 仕組み | 長所 | 短所 |
|------|--------|------|------|
| MD5/SHA256 ハッシュ | URL をハッシュ化し先頭7文字を使う | 分散環境でも衝突少 | 衝突処理が必要、同一URL→同一キー |
| カウンター + Base62 | 自動採番の整数を Base62 変換 | 衝突なし、予測可能な長さ | 分散カウンターが必要 |
| 事前生成 (KGS) | キーを事前生成しプールから取り出す | 衝突なし、高速 | キー管理サービスが必要 |

### 2.4 Key Generation Service (KGS)

```python
# コード例 2: 事前生成キープールによる短縮URL生成
import redis
import random
import string

class KeyGenerationService:
    """
    短縮URLキーを事前生成し、Redis のキューに格納する。
    APIサーバーはキューからキーを取り出すだけで高速。
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.Redis.from_url(redis_url)
        self.key_length = 7
        self.unused_key = "kgs:unused_keys"
        self.used_key = "kgs:used_keys"

    def generate_keys(self, count: int = 100_000):
        """キーを一括生成してプールに追加する"""
        charset = string.ascii_letters + string.digits
        pipe = self.redis.pipeline()
        for _ in range(count):
            key = ''.join(random.choices(charset, k=self.key_length))
            pipe.sadd(self.unused_key, key)
        pipe.execute()
        print(f"Generated {count} keys. Pool size: {self.pool_size()}")

    def get_key(self) -> str:
        """未使用キーをプールから取得する（アトミック操作）"""
        key = self.redis.spop(self.unused_key)
        if key is None:
            raise RuntimeError("Key pool exhausted! Run generate_keys()")
        self.redis.sadd(self.used_key, key)
        return key.decode()

    def pool_size(self) -> int:
        return self.redis.scard(self.unused_key)
```

---

## 3. データモデルとリダイレクト

### 3.1 データベーススキーマ

```sql
-- コード例 3: URL マッピングテーブル
CREATE TABLE url_mappings (
    id            BIGINT PRIMARY KEY AUTO_INCREMENT,
    short_key     VARCHAR(7) NOT NULL UNIQUE,
    original_url  TEXT NOT NULL,
    user_id       BIGINT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at    TIMESTAMP NULL,
    click_count   BIGINT DEFAULT 0,

    INDEX idx_short_key (short_key),
    INDEX idx_expires_at (expires_at)
) ENGINE=InnoDB;

-- アクセスログ（分析用）
CREATE TABLE click_logs (
    id            BIGINT PRIMARY KEY AUTO_INCREMENT,
    short_key     VARCHAR(7) NOT NULL,
    clicked_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address    VARCHAR(45),
    user_agent    TEXT,
    referer       TEXT,
    country_code  VARCHAR(2),

    INDEX idx_short_key_time (short_key, clicked_at)
) ENGINE=InnoDB;
```

### 3.2 リダイレクトフロー

```
クライアント                API サーバー              Redis            DB
    |                          |                      |               |
    |  GET /abc123             |                      |               |
    |------------------------->|                      |               |
    |                          |  GET abc123          |               |
    |                          |--------------------->|               |
    |                          |                      |               |
    |                          |  キャッシュヒット?     |               |
    |                          |<---------------------|               |
    |                          |                      |               |
    |                    [ヒット]                      |               |
    |  301/302 Redirect        |                      |               |
    |<-------------------------|                      |               |
    |                          |                      |               |
    |                    [ミス]                        |               |
    |                          |  SELECT ... WHERE    |               |
    |                          |  short_key='abc123'  |               |
    |                          |------------------------------------->|
    |                          |                      |               |
    |                          |  original_url        |               |
    |                          |<-------------------------------------|
    |                          |                      |               |
    |                          |  SET abc123 url      |               |
    |                          |--------------------->|               |
    |                          |                      |               |
    |  301/302 Redirect        |                      |               |
    |<-------------------------|                      |               |
```

### 3.3 リダイレクト実装

```python
# コード例 4: FastAPI によるリダイレクト API
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, HttpUrl
import redis
import databases

app = FastAPI()
cache = redis.Redis(host="localhost", port=6379, db=0)
db = databases.Database("mysql://user:pass@localhost/url_shortener")

CACHE_TTL = 3600 * 24  # 24時間

class ShortenRequest(BaseModel):
    url: HttpUrl
    custom_alias: str | None = None

@app.post("/api/shorten")
async def shorten_url(request: ShortenRequest):
    kgs = KeyGenerationService()
    short_key = request.custom_alias or kgs.get_key()

    # DB に保存
    await db.execute(
        "INSERT INTO url_mappings (short_key, original_url) VALUES (:key, :url)",
        {"key": short_key, "url": str(request.url)}
    )

    # キャッシュにも保存
    cache.setex(f"url:{short_key}", CACHE_TTL, str(request.url))

    return {"short_url": f"https://short.ly/{short_key}"}

@app.get("/{short_key}")
async def redirect(short_key: str):
    # 1. キャッシュを確認
    cached = cache.get(f"url:{short_key}")
    if cached:
        return RedirectResponse(
            url=cached.decode(),
            status_code=301  # 永続リダイレクト (キャッシュ可能)
        )

    # 2. DB を確認
    row = await db.fetch_one(
        "SELECT original_url FROM url_mappings WHERE short_key = :key",
        {"key": short_key}
    )
    if not row:
        raise HTTPException(status_code=404, detail="URL not found")

    # 3. キャッシュに保存
    cache.setex(f"url:{short_key}", CACHE_TTL, row["original_url"])

    return RedirectResponse(url=row["original_url"], status_code=301)
```

---

## 4. スケーラビリティ設計

### 4.1 キャッシュ戦略

```
                  読み取りリクエスト 115K QPS
                         |
                         v
                  +--------------+
                  | CDN / Edge   |  <-- 最もホットなURLをエッジキャッシュ
                  | Cache        |      (80%のリクエストをここで処理)
                  +--------------+
                         |  キャッシュミス (20%)
                         v
                  +--------------+
                  | Redis        |  <-- アプリレベルキャッシュ
                  | Cluster      |      (残り19%をここで処理)
                  +--------------+
                         |  キャッシュミス (1%)
                         v
                  +--------------+
                  | DB Replica   |  <-- リードレプリカ
                  | (Read)       |      (最終的なフォールバック)
                  +--------------+
```

### 4.2 データベースシャーディング

```python
# コード例 5: 短縮キーに基づくハッシュシャーディング
class ShardRouter:
    """短縮キーのハッシュ値に基づいてシャードを決定する"""

    def __init__(self, num_shards: int = 16):
        self.num_shards = num_shards
        self.shard_connections = {
            i: f"mysql://user:pass@shard-{i}.db.internal/urls"
            for i in range(num_shards)
        }

    def get_shard(self, short_key: str) -> int:
        """一貫性ハッシュでシャード番号を決定する"""
        hash_value = hash(short_key)
        return hash_value % self.num_shards

    def get_connection(self, short_key: str) -> str:
        shard_id = self.get_shard(short_key)
        return self.shard_connections[shard_id]
```

---

## 5. 301 vs 302 リダイレクト

| 観点 | 301 (Moved Permanently) | 302 (Found / Temporary) |
|------|------------------------|------------------------|
| ブラウザキャッシュ | キャッシュされる | キャッシュされない |
| サーバー負荷 | 低い（2回目以降はサーバーに来ない） | 高い（毎回サーバーに来る） |
| アクセス統計 | 正確に取れない | 毎回記録可能 |
| SEO | リンクジュースが転送先に移る | 元 URL にリンクジュースが残る |
| 推奨用途 | 統計不要で高速性重視 | アクセス分析が必要な場合 |

---

## 6. アンチパターン

### アンチパターン 1: 「全リクエストを DB に直接アクセス」

```
[誤り] キャッシュなしで全リダイレクトをDBに問い合わせる

  115,000 QPS → DB直接アクセス → DB が即座にダウン

[正解] 多層キャッシュ戦略を採用する
  - 80-20 の法則: 20%のURLが80%のトラフィックを受ける
  - Redis キャッシュで99%のリクエストを処理可能
  - CDN エッジキャッシュでさらに負荷を分散
```

### アンチパターン 2: 「短縮キーを連番で生成」

```
[誤り] auto_increment の連番をそのまま短縮キーにする

問題点:
  - 予測可能: /1, /2, /3 ... で全URLを列挙可能
  - セキュリティリスク: 非公開URLが推測される
  - 見た目が悪い: 小さな数字は短すぎる（/1 は1文字）

[正解] ランダムまたは事前生成キー (KGS) を使用する
  - 予測不可能な7文字のランダムキー
  - 必要に応じて Base62 エンコーディング
  - KGS で衝突チェックを事前に完了
```

---

## 7. FAQ

### Q1: 短縮キーは何文字が最適ですか？

**A:** 6〜7 文字が一般的です。Base62（英大小文字 + 数字）の場合:

- 6文字: 62^6 = 56.8B（568億通り）
- 7文字: 62^7 = 3.52T（3.5兆通り）

5年間で 182.5B URL を想定する場合、7文字あれば十分な余裕があります。

### Q2: 同じ URL が複数回短縮された場合、同じキーを返すべきですか？

**A:** 設計判断に依存しますが、一般的には異なるキーを返します。

- **異なるキー**: ユーザーごとに別々のアクセス統計が取れる。実装がシンプル
- **同じキー**: ストレージ効率が良い。ただし URL → key の逆引きインデックスが必要で、複雑になる

### Q3: 期限切れ URL の削除はどう行いますか？

**A:** 以下の戦略を組み合わせます。

1. **Lazy Deletion**: リダイレクト時に期限切れを確認し、404 を返す（即座にユーザー体験に反映）
2. **Background Cleanup**: Cron ジョブで期限切れレコードを定期的に削除（DB 容量回復）
3. **TTL in Cache**: Redis キャッシュには TTL を設定し、自動的に期限切れ

---

## 8. まとめ

| 設計要素 | 選択 | 理由 |
|----------|------|------|
| キー生成 | KGS (事前生成) | 衝突なし、高速 |
| キー長 | 7文字 (Base62) | 3.5兆通りで十分 |
| リダイレクト | 301 or 302 | 要件に応じて選択 |
| キャッシュ | Redis + CDN | 読み取り QPS に対応 |
| データベース | NoSQL or シャーディング RDB | スケーラビリティ |
| 可用性 | マルチ AZ + レプリカ | 99.99% SLA |

---

## 次に読むべきガイド

- [チャットシステム設計](./01-chat-system.md) — リアルタイム通信のシステム設計
- [通知システム設計](./02-notification-system.md) — 大規模通知配信の設計
- [データベース設計の基礎](../01-fundamentals/02-database.md) — シャーディング・レプリケーションの詳細

---

## 参考文献

1. Xu, A. (2020). "System Design Interview: An Insider's Guide." *Chapter 8: Design a URL Shortener*. https://www.systemdesigninterview.com/
2. Kleppmann, M. (2017). "Designing Data-Intensive Applications." *O'Reilly Media*. Chapter 5-6: Replication and Partitioning. https://dataintensive.net/
3. Nishtala, R. et al. (2013). "Scaling Memcache at Facebook." *10th USENIX Symposium on Networked Systems Design and Implementation (NSDI '13)*. https://www.usenix.org/conference/nsdi13/technical-sessions/presentation/nishtala
