# URL 短縮サービス設計

> TinyURL / bit.ly のような URL 短縮サービスをゼロから設計する。ハッシュ生成、リダイレクト最適化、スケーラビリティの観点でシステム設計面接に対応できる力を身につける。

---

## この章で学ぶこと

1. **要件定義とスケール見積もり** — 機能要件・非機能要件の整理と、トラフィック・ストレージの概算
2. **コア設計** — 短縮 URL の生成アルゴリズム、データモデル、リダイレクトフロー
3. **スケーラビリティ** — キャッシュ戦略、データベース分割、高可用性設計
4. **運用設計** — 監視、アラート、障害対応、容量計画
5. **セキュリティ** — 悪意ある URL の検出、レート制限、不正利用防止

---

## 前提知識

| トピック | 必要レベル | 参考ガイド |
|---------|-----------|-----------|
| データベース基礎 | RDB のインデックス設計、NoSQL の基本 | [データベース](../01-components/01-database.md) |
| キャッシュ設計 | Redis の基本操作、キャッシュ戦略 | [キャッシュ](../01-components/00-cache.md) |
| ロードバランサー | L4/L7 の違い、分散アルゴリズム | [ロードバランサー](../01-components/03-load-balancer.md) |
| ハッシュとエンコーディング | Base62, MD5, SHA-256 の概念 | [アルゴリズム基礎](../../../01-cs-fundamentals/algorithms-guide/docs/00-foundations/00-complexity.md) |
| Python / FastAPI | Web API 開発の基礎 | [Python ガイド](../../../02-programming/python-guide/docs/00-basics/00-introduction.md) |

---

## 背景

### URL 短縮サービスとは

```
長い URL:
  https://www.example.com/products/electronics/smartphones/iphone-15-pro?color=blue&storage=256gb&ref=campaign-2026-winter

短縮 URL:
  https://short.ly/abc123

用途:
  1. SNS（Twitter/X の文字数制限）での URL 共有
  2. QR コードの最適化（短い URL はより小さな QR コードになる）
  3. アクセス解析（クリック数、地域、デバイス等の追跡）
  4. ブランディング（カスタムドメイン: yourbrand.link/event）
  5. 印刷物での URL 掲載（短い方が読みやすい）
```

### なぜシステム設計面接で頻出なのか

```
URL 短縮サービスは以下の要素を網羅的にテストできる:

  1. スケール見積もり: トラフィック・ストレージの概算
  2. ハッシュ/エンコーディング: アルゴリズムの選択と設計
  3. データベース設計: スキーマ、インデックス、シャーディング
  4. キャッシュ戦略: 多層キャッシュ、キャッシュ無効化
  5. 可用性設計: レプリカ、フェイルオーバー
  6. トレードオフ判断: 301 vs 302、一貫性 vs 可用性

  → シンプルな要件の中に、多くの設計判断が詰まっている
```

---

## 1. 要件定義

### 1.1 機能要件

```
=== 必須機能 (Must Have) ===
- 長い URL を短い URL に変換する
  例: https://example.com/very/long/path → https://short.ly/abc123
- 短縮 URL にアクセスすると元の URL にリダイレクトする
- API でプログラマティックに短縮 URL を作成できる

=== 重要機能 (Should Have) ===
- カスタムエイリアスを指定できる（例: short.ly/my-event）
- URL の有効期限を設定できる
- アクセス統計を確認できる（クリック数、地域、デバイス）

=== 追加機能 (Nice to Have) ===
- ユーザー認証と URL 管理ダッシュボード
- カスタムドメインのサポート
- QR コード自動生成
- URL のプレビューページ（リダイレクト前に行き先を表示）
```

### 1.2 非機能要件

```
=== パフォーマンス ===
- リダイレクトのレイテンシ: P99 < 100ms
- URL 作成のレイテンシ: P99 < 500ms

=== 可用性 ===
- リダイレクト: 99.99% (年間ダウンタイム < 53分)
- URL 作成: 99.9% (年間ダウンタイム < 8.8時間)

=== スケーラビリティ ===
- 読み取り:書き込み = 100:1 の読み取り重視
- 水平スケール可能な設計

=== セキュリティ ===
- 悪意あるリダイレクト先（フィッシング等）の検出・ブロック
- レート制限: IP あたり 100 URL/時間
- 短縮キーの予測不可能性
```

### 1.3 スケール見積もり

```python
# === スケール見積もり計算 ===

# 前提
daily_url_creation = 100_000_000    # 1日1億 URL 作成
read_write_ratio = 100              # 読み取り:書き込み = 100:1
daily_redirects = daily_url_creation * read_write_ratio  # 100億リダイレクト/日
retention_years = 5                 # 5年間保持

# QPS (Queries Per Second)
write_qps = daily_url_creation / 86400           # ≈ 1,160 QPS
write_qps_peak = write_qps * 2                   # ≈ 2,320 QPS (ピーク)
read_qps = daily_redirects / 86400               # ≈ 115,740 QPS
read_qps_peak = read_qps * 2                     # ≈ 231,480 QPS (ピーク)

# ストレージ
total_urls = daily_url_creation * 365 * retention_years  # 182.5B (1,825億)
bytes_per_record = 500                                    # URL + メタデータ
total_storage_tb = total_urls * bytes_per_record / (1024**4)  # ≈ 91.25 TB

# 帯域幅
avg_redirect_size_bytes = 500           # リダイレクトレスポンスのサイズ
bandwidth_mbps = (read_qps * avg_redirect_size_bytes * 8) / (1024**2)  # ≈ 442 Mbps

# キャッシュ (80-20 の法則)
daily_unique_urls = daily_redirects * 0.2         # 20% がユニーク
cache_memory_gb = (daily_unique_urls * bytes_per_record) / (1024**3)  # ≈ 931 GB

print(f"""
=== スケール見積もり結果 ===
書き込み QPS:     {write_qps:,.0f} (ピーク: {write_qps_peak:,.0f})
読み取り QPS:     {read_qps:,.0f} (ピーク: {read_qps_peak:,.0f})
5年間の総URL数:   {total_urls / 1e9:.1f}B ({total_urls / 1e9:.1f}億)
ストレージ:       {total_storage_tb:.1f} TB
帯域幅:          {bandwidth_mbps:.0f} Mbps
キャッシュ必要量: {cache_memory_gb:.0f} GB
""")
```

### 1.4 Back-of-the-envelope 計算のフレームワーク

```
=== システム設計面接での概算テンプレート ===

Step 1: 書き込み QPS
  {日次書き込み数} ÷ 86,400 = 平均 QPS
  平均 QPS × 2 = ピーク QPS

Step 2: 読み取り QPS
  書き込み QPS × R/W 比率

Step 3: ストレージ
  {日次書き込み数} × {保持日数} × {1レコードのサイズ}

Step 4: 帯域幅
  QPS × {レスポンスサイズ}

Step 5: キャッシュ
  80-20 の法則を適用
  → {日次読み取り数} × 20% × {1レコードのサイズ}

重要: 正確な数値より桁数が合っていることが重要
  1,160 QPS と 1,200 QPS は同じ。
  1,160 QPS と 11,600 QPS は全く違う。
```

---

## 2. コア設計

### 2.1 高レベルアーキテクチャ

```
                    URL 短縮サービス全体構成

  ┌──────────┐
  │ Client   │
  │ (Browser/│
  │  Mobile) │
  └────┬─────┘
       │
       v
  ┌──────────┐     ┌──────────────────────────────────────────┐
  │ CDN /    │     │            Application Layer               │
  │ Edge     │     │  ┌────────────┐    ┌────────────────┐     │
  │ Cache    │ ──> │  │ Load       │ -> │ API Servers    │     │
  └──────────┘     │  │ Balancer   │    │ (Write: 短縮)  │     │
                   │  │ (Nginx/ALB)│    │ (Read: リダイ  │     │
                   │  └────────────┘    │  レクト)        │     │
                   │                    └───────┬────────┘     │
                   └────────────────────────────┼──────────────┘
                                                │
                              ┌─────────────────┼─────────────────┐
                              │                 │                  │
                              v                 v                  v
                     ┌──────────┐      ┌──────────┐      ┌──────────┐
                     │ Redis    │      │ DB Master│      │ Key Gen  │
                     │ Cluster  │      │ (Write)  │      │ Service  │
                     │ (Cache)  │      └────┬─────┘      │ (KGS)   │
                     └──────────┘           │            └──────────┘
                                     ┌──────┴──────┐
                                     v             v
                               ┌──────────┐  ┌──────────┐
                               │ DB       │  │ DB       │
                               │ Replica1 │  │ Replica2 │
                               │ (Read)   │  │ (Read)   │
                               └──────────┘  └──────────┘
```

### 2.2 短縮キー生成アルゴリズム

```python
# === 方式1: Base62 エンコーディング ===
import string

CHARSET = string.digits + string.ascii_lowercase + string.ascii_uppercase
BASE = len(CHARSET)  # 62


def encode_base62(num: int) -> str:
    """整数を Base62 文字列に変換する

    62進法でエンコード:
    0-9 → '0'-'9'
    10-35 → 'a'-'z'
    36-61 → 'A'-'Z'
    """
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


# 7文字の Base62 で表現可能な範囲
# 62^7 = 3,521,614,606,208 (約3.5兆通り)
assert encode_base62(123456789) == "8M0kX"
assert decode_base62("8M0kX") == 123456789


# === 方式2: MD5 ハッシュの先頭7文字 ===
import hashlib


def generate_hash_key(url: str, attempt: int = 0) -> str:
    """URL の MD5 ハッシュから短縮キーを生成

    衝突時は attempt をインクリメントして再試行
    """
    input_str = url if attempt == 0 else f"{url}:{attempt}"
    hash_hex = hashlib.md5(input_str.encode()).hexdigest()

    # 16進数を整数に変換し、Base62 エンコード
    hash_int = int(hash_hex[:12], 16)  # 先頭12文字を使用
    key = encode_base62(hash_int)
    return key[:7]  # 7文字に切り詰め


# 例
print(generate_hash_key("https://example.com/very/long/path"))  # => "aBcD123"


# === 方式3: Snowflake ID + Base62 ===
import time
import threading


class SnowflakeIDGenerator:
    """Twitter Snowflake 方式の分散 ID 生成

    64bit の ID:
    - 1bit: 符号（常に0）
    - 41bit: タイムスタンプ（ミリ秒）→ 約69年
    - 10bit: マシン ID（1024台）
    - 12bit: シーケンス番号（4096/ms/マシン）

    特徴:
    - 時系列順のID生成
    - 分散環境で衝突なし
    - 高スループット（1マシンで409万ID/秒）
    """

    def __init__(self, machine_id: int):
        self._machine_id = machine_id & 0x3FF  # 10bit
        self._sequence = 0
        self._last_timestamp = 0
        self._lock = threading.Lock()
        self._epoch = 1704067200000  # 2024-01-01 00:00:00 UTC

    def next_id(self) -> int:
        with self._lock:
            timestamp = int(time.time() * 1000) - self._epoch

            if timestamp == self._last_timestamp:
                self._sequence = (self._sequence + 1) & 0xFFF
                if self._sequence == 0:
                    # 同一ミリ秒でシーケンス上限 → 次のミリ秒まで待機
                    while timestamp <= self._last_timestamp:
                        timestamp = int(time.time() * 1000) - self._epoch
            else:
                self._sequence = 0

            self._last_timestamp = timestamp

            return (
                (timestamp << 22) |
                (self._machine_id << 12) |
                self._sequence
            )


# Snowflake ID → Base62 短縮キー
generator = SnowflakeIDGenerator(machine_id=1)
snowflake_id = generator.next_id()
short_key = encode_base62(snowflake_id)[:7]
```

### 2.3 ハッシュ方式 vs カウンター方式 vs KGS

| 方式 | 仕組み | 長所 | 短所 | 適用場面 |
|------|--------|------|------|---------|
| MD5/SHA256 ハッシュ | URL をハッシュ化し先頭7文字を使う | 同一 URL → 同一キー可能 | 衝突処理が必要 | デデュプリケーション重視 |
| カウンター + Base62 | 自動採番の整数を Base62 変換 | 衝突なし、予測可能な長さ | 分散カウンターが必要 | シンプルな実装 |
| Snowflake + Base62 | 分散 ID を Base62 変換 | 衝突なし、分散対応 | 8-10文字になりがち | 大規模分散システム |
| 事前生成 (KGS) | キーを事前生成しプールから取り出す | 衝突なし、最高速 | キー管理サービスが必要 | 超高スループット |

### 2.4 Key Generation Service (KGS)

```python
# infrastructure/key_generation_service.py
import redis
import random
import string
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class KeyGenerationService:
    """短縮URLキーを事前生成し、Redis のキューに格納する。

    設計:
    1. バックグラウンドで大量のキーを事前生成
    2. Redis Set に格納（未使用キープール）
    3. API サーバーは spop でアトミックにキーを取得
    4. キー残量が閾値を下回ったら自動補充

    ┌────────────┐     ┌─────────────────┐     ┌──────────┐
    │ Key        │ --> │ Redis           │ --> │ API      │
    │ Generator  │     │ unused_keys Set │     │ Server   │
    │ (Background│     │ (100万キー)      │     │ (spop)   │
    └────────────┘     └─────────────────┘     └──────────┘
    """

    UNUSED_KEY = "kgs:unused_keys"
    USED_KEY = "kgs:used_keys"
    MIN_POOL_SIZE = 100_000
    BATCH_GENERATE_SIZE = 500_000

    def __init__(
        self,
        redis_client: redis.Redis,
        key_length: int = 7,
    ):
        self._redis = redis_client
        self._key_length = key_length

    def generate_keys(self, count: int = BATCH_GENERATE_SIZE) -> int:
        """キーを一括生成してプールに追加する

        Returns:
            実際に追加されたキー数
        """
        charset = string.ascii_letters + string.digits
        generated = 0
        pipe = self._redis.pipeline()

        for _ in range(count):
            key = ''.join(random.choices(charset, k=self._key_length))
            pipe.sadd(self.UNUSED_KEY, key)
            generated += 1

            # パイプラインのバッファが大きくなりすぎないように
            if generated % 10_000 == 0:
                pipe.execute()
                pipe = self._redis.pipeline()

        pipe.execute()
        pool_size = self.pool_size()
        logger.info(
            f"キー生成完了: {generated}件追加, "
            f"プールサイズ: {pool_size}"
        )
        return generated

    def get_key(self) -> str:
        """未使用キーをプールから取得する（アトミック操作）

        Raises:
            RuntimeError: キープールが枯渇した場合
        """
        key = self._redis.spop(self.UNUSED_KEY)
        if key is None:
            raise RuntimeError(
                "キープール枯渇! generate_keys() を実行してください"
            )
        # 使用済みセットに追加（重複チェック用）
        self._redis.sadd(self.USED_KEY, key)
        return key.decode()

    def return_key(self, key: str) -> None:
        """キーをプールに返却する（URL 作成が失敗した場合）"""
        self._redis.srem(self.USED_KEY, key)
        self._redis.sadd(self.UNUSED_KEY, key)

    def pool_size(self) -> int:
        """未使用キーの残数を取得"""
        return self._redis.scard(self.UNUSED_KEY)

    def ensure_pool_size(self) -> None:
        """プールサイズが閾値を下回ったら自動補充"""
        current = self.pool_size()
        if current < self.MIN_POOL_SIZE:
            logger.warning(
                f"キープール残量低下: {current}, 補充開始"
            )
            self.generate_keys(self.BATCH_GENERATE_SIZE)

    def is_key_used(self, key: str) -> bool:
        """キーが使用済みか確認（カスタムエイリアスの重複チェック用）"""
        return self._redis.sismember(self.USED_KEY, key)
```

---

## 3. データモデルとリダイレクト

### 3.1 データベーススキーマ

```sql
-- URL マッピングテーブル（メインテーブル）
CREATE TABLE url_mappings (
    id            BIGINT PRIMARY KEY AUTO_INCREMENT,
    short_key     VARCHAR(7) NOT NULL UNIQUE,
    original_url  TEXT NOT NULL,
    user_id       BIGINT NULL,
    custom_alias  BOOLEAN DEFAULT FALSE,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at    TIMESTAMP NULL,
    click_count   BIGINT DEFAULT 0,
    is_active     BOOLEAN DEFAULT TRUE,

    -- インデックス
    INDEX idx_short_key (short_key),             -- リダイレクト時のルックアップ
    INDEX idx_user_id (user_id),                 -- ユーザーの URL 一覧
    INDEX idx_expires_at (expires_at),           -- 期限切れクリーンアップ
    INDEX idx_created_at (created_at)            -- 最新順の一覧
) ENGINE=InnoDB;

-- アクセスログ（分析用、別テーブル/別 DB）
CREATE TABLE click_events (
    id            BIGINT PRIMARY KEY AUTO_INCREMENT,
    short_key     VARCHAR(7) NOT NULL,
    clicked_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address    VARCHAR(45),                    -- IPv6 対応
    user_agent    VARCHAR(500),
    referer       VARCHAR(2048),
    country_code  CHAR(2),
    device_type   VARCHAR(20),                    -- mobile / desktop / tablet
    os            VARCHAR(50),
    browser       VARCHAR(50),

    -- パーティション: 月単位（大量データの管理用）
    INDEX idx_short_key_time (short_key, clicked_at)
) ENGINE=InnoDB
PARTITION BY RANGE (UNIX_TIMESTAMP(clicked_at)) (
    PARTITION p2026_01 VALUES LESS THAN (UNIX_TIMESTAMP('2026-02-01')),
    PARTITION p2026_02 VALUES LESS THAN (UNIX_TIMESTAMP('2026-03-01')),
    PARTITION p_future VALUES LESS THAN MAXVALUE
);

-- ユーザーテーブル（認証機能がある場合）
CREATE TABLE users (
    id            BIGINT PRIMARY KEY AUTO_INCREMENT,
    email         VARCHAR(255) NOT NULL UNIQUE,
    api_key       VARCHAR(64) NOT NULL UNIQUE,
    plan          ENUM('free', 'pro', 'enterprise') DEFAULT 'free',
    rate_limit    INT DEFAULT 100,                -- 1時間あたりの URL 作成上限
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_api_key (api_key)
) ENGINE=InnoDB;
```

### 3.2 NoSQL 代替案（DynamoDB）

```python
# DynamoDB スキーマ設計

"""
DynamoDB テーブル設計（NoSQL 代替案）

テーブル: url_mappings
  パーティションキー: short_key (String)
  属性:
    - original_url (String)
    - user_id (String, optional)
    - created_at (Number, Unix timestamp)
    - expires_at (Number, Unix timestamp, optional)
    - click_count (Number)
    - is_active (Boolean)

  GSI: user_id-index
    パーティションキー: user_id
    ソートキー: created_at

メリット:
  - short_key でのルックアップが O(1) → 超低レイテンシ
  - 水平スケールが自動（シャーディング不要）
  - TTL 機能で期限切れ URL を自動削除

デメリット:
  - 複雑なクエリが困難（SQL の柔軟性がない）
  - コストが読み書きキャパシティに依存
"""

import boto3
from datetime import datetime, timezone, timedelta

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('url_mappings')


def create_short_url(short_key: str, original_url: str, ttl_days: int = 0):
    """URL マッピングを作成"""
    item = {
        'short_key': short_key,
        'original_url': original_url,
        'created_at': int(datetime.now(timezone.utc).timestamp()),
        'click_count': 0,
        'is_active': True,
    }
    if ttl_days > 0:
        expires_at = datetime.now(timezone.utc) + timedelta(days=ttl_days)
        item['expires_at'] = int(expires_at.timestamp())
        item['ttl'] = int(expires_at.timestamp())  # DynamoDB TTL

    table.put_item(
        Item=item,
        ConditionExpression='attribute_not_exists(short_key)',  # 重複防止
    )


def get_original_url(short_key: str) -> str | None:
    """短縮キーから元 URL を取得"""
    response = table.get_item(
        Key={'short_key': short_key},
        ProjectionExpression='original_url, is_active, expires_at',
    )
    item = response.get('Item')
    if not item or not item.get('is_active', True):
        return None

    # 期限切れチェック
    expires_at = item.get('expires_at')
    if expires_at and expires_at < int(datetime.now(timezone.utc).timestamp()):
        return None

    return item['original_url']


def increment_click_count(short_key: str):
    """クリックカウントをアトミックにインクリメント"""
    table.update_item(
        Key={'short_key': short_key},
        UpdateExpression='ADD click_count :inc',
        ExpressionAttributeValues={':inc': 1},
    )
```

### 3.3 リダイレクトフロー

```
クライアント                API サーバー              Redis            DB
    |                          |                      |               |
    |  GET /abc123             |                      |               |
    |------------------------->|                      |               |
    |                          |                      |               |
    |                          |  GET url:abc123      |               |
    |                          |--------------------->|               |
    |                          |                      |               |
    |                    [キャッシュヒット]              |               |
    |  301/302 Redirect        |                      |               |
    |<-------------------------|                      |               |
    |                          |                      |               |
    |                    [キャッシュミス]                |               |
    |                          |  SELECT original_url |               |
    |                          |  WHERE short_key=    |               |
    |                          |  'abc123'            |               |
    |                          |------------------------------------->|
    |                          |                      |               |
    |                          |  original_url        |               |
    |                          |<-------------------------------------|
    |                          |                      |               |
    |                          |  SET url:abc123      |               |
    |                          |  EX 86400            |               |
    |                          |--------------------->|               |
    |                          |                      |               |
    |  301/302 Redirect        |                      |               |
    |<-------------------------|                      |               |
    |                          |                      |               |
    |          [非同期: クリックイベント記録]             |               |
    |                          |---> [Message Queue]  |               |
    |                          |     → Click Logger   |               |
```

### 3.4 API 実装

```python
# application/api/url_shortener_api.py
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, HttpUrl, Field
from typing import Optional
import redis
import databases
import logging
import asyncio
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

app = FastAPI(title="URL Shortener API")
cache = redis.Redis(host="redis-host", port=6379, db=0, decode_responses=True)
db = databases.Database("mysql://user:pass@db-host/url_shortener")

CACHE_TTL = 86400  # 24時間
BASE_URL = "https://short.ly"


class ShortenRequest(BaseModel):
    """URL 短縮リクエスト"""
    url: HttpUrl
    custom_alias: Optional[str] = Field(
        None, min_length=4, max_length=30, pattern=r'^[a-zA-Z0-9_-]+$'
    )
    expires_in_days: Optional[int] = Field(None, ge=1, le=3650)


class ShortenResponse(BaseModel):
    """URL 短縮レスポンス"""
    short_url: str
    short_key: str
    original_url: str
    expires_at: Optional[str] = None


class UrlStatsResponse(BaseModel):
    """URL 統計レスポンス"""
    short_key: str
    original_url: str
    click_count: int
    created_at: str
    expires_at: Optional[str] = None


# === URL 短縮 API ===

@app.post("/api/v1/shorten", response_model=ShortenResponse)
async def shorten_url(
    request: ShortenRequest,
    api_key: str = Header(..., alias="X-API-Key"),
):
    """長い URL を短縮する"""
    # 1. API キーの認証（省略: 実際は認証ミドルウェアで処理）
    user = await authenticate_api_key(api_key)

    # 2. レート制限チェック
    await check_rate_limit(user.id)

    # 3. URL の安全性チェック（フィッシング等）
    if not await is_safe_url(str(request.url)):
        raise HTTPException(
            status_code=400,
            detail="URL が安全でない可能性があります"
        )

    # 4. 短縮キーの取得
    if request.custom_alias:
        short_key = request.custom_alias
        # カスタムエイリアスの重複チェック
        existing = await db.fetch_one(
            "SELECT id FROM url_mappings WHERE short_key = :key",
            {"key": short_key}
        )
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"エイリアス '{short_key}' は既に使用されています"
            )
    else:
        kgs = KeyGenerationService(redis_client=cache)
        short_key = kgs.get_key()

    # 5. 有効期限の計算
    expires_at = None
    if request.expires_in_days:
        expires_at = datetime.now(timezone.utc) + timedelta(
            days=request.expires_in_days
        )

    # 6. DB に保存
    try:
        await db.execute(
            """
            INSERT INTO url_mappings
                (short_key, original_url, user_id, custom_alias, expires_at)
            VALUES (:key, :url, :user_id, :custom, :expires)
            """,
            {
                "key": short_key,
                "url": str(request.url),
                "user_id": user.id,
                "custom": request.custom_alias is not None,
                "expires": expires_at,
            }
        )
    except Exception as e:
        # DB 保存失敗時はキーをプールに返却
        if not request.custom_alias:
            kgs.return_key(short_key)
        raise HTTPException(status_code=500, detail="URL の作成に失敗しました")

    # 7. キャッシュにも保存
    cache.setex(f"url:{short_key}", CACHE_TTL, str(request.url))

    return ShortenResponse(
        short_url=f"{BASE_URL}/{short_key}",
        short_key=short_key,
        original_url=str(request.url),
        expires_at=expires_at.isoformat() if expires_at else None,
    )


# === リダイレクト API ===

@app.get("/{short_key}")
async def redirect(short_key: str, request: Request):
    """短縮 URL から元 URL へリダイレクトする"""
    # 1. キャッシュを確認
    cached_url = cache.get(f"url:{short_key}")
    if cached_url:
        # 非同期でクリックイベントを記録（リダイレクトをブロックしない）
        asyncio.create_task(
            record_click_event(short_key, request)
        )
        return RedirectResponse(
            url=cached_url,
            status_code=301,  # 永続リダイレクト
        )

    # 2. DB を確認
    row = await db.fetch_one(
        """
        SELECT original_url, expires_at, is_active
        FROM url_mappings
        WHERE short_key = :key
        """,
        {"key": short_key}
    )

    if not row:
        raise HTTPException(status_code=404, detail="URL が見つかりません")

    if not row["is_active"]:
        raise HTTPException(status_code=410, detail="URL は無効化されています")

    # 3. 期限切れチェック
    if row["expires_at"] and row["expires_at"] < datetime.now(timezone.utc):
        raise HTTPException(status_code=410, detail="URL の有効期限が切れています")

    original_url = row["original_url"]

    # 4. キャッシュに保存
    cache.setex(f"url:{short_key}", CACHE_TTL, original_url)

    # 5. 非同期でクリックイベントを記録
    asyncio.create_task(record_click_event(short_key, request))

    return RedirectResponse(url=original_url, status_code=301)


# === 統計 API ===

@app.get("/api/v1/stats/{short_key}", response_model=UrlStatsResponse)
async def get_stats(
    short_key: str,
    api_key: str = Header(..., alias="X-API-Key"),
):
    """URL のアクセス統計を取得する"""
    user = await authenticate_api_key(api_key)

    row = await db.fetch_one(
        """
        SELECT short_key, original_url, click_count, created_at, expires_at
        FROM url_mappings
        WHERE short_key = :key AND user_id = :user_id
        """,
        {"key": short_key, "user_id": user.id}
    )
    if not row:
        raise HTTPException(status_code=404, detail="URL が見つかりません")

    return UrlStatsResponse(
        short_key=row["short_key"],
        original_url=row["original_url"],
        click_count=row["click_count"],
        created_at=row["created_at"].isoformat(),
        expires_at=row["expires_at"].isoformat() if row["expires_at"] else None,
    )


# === ヘルパー関数 ===

async def record_click_event(short_key: str, request: Request) -> None:
    """クリックイベントを非同期で記録する

    注: 本番環境ではメッセージキュー（Kafka 等）に
    発行して別サービスで処理するのが推奨
    """
    try:
        await db.execute(
            """
            INSERT INTO click_events
                (short_key, ip_address, user_agent, referer)
            VALUES (:key, :ip, :ua, :referer)
            """,
            {
                "key": short_key,
                "ip": request.client.host,
                "ua": request.headers.get("User-Agent", "")[:500],
                "referer": request.headers.get("Referer", "")[:2048],
            }
        )
        # クリックカウントのインクリメント
        await db.execute(
            "UPDATE url_mappings SET click_count = click_count + 1 WHERE short_key = :key",
            {"key": short_key}
        )
    except Exception as e:
        # クリック記録の失敗はリダイレクトに影響させない
        logger.error(f"クリックイベント記録失敗: {short_key}, {e}")


async def is_safe_url(url: str) -> bool:
    """URL の安全性チェック（Google Safe Browsing API 等）"""
    # 実装省略: 実際は外部 API で確認
    dangerous_patterns = [
        "phishing.example.com",
        "malware.example.com",
    ]
    return not any(pattern in url for pattern in dangerous_patterns)


async def check_rate_limit(user_id: int) -> None:
    """レート制限チェック"""
    key = f"rate_limit:{user_id}"
    current = cache.incr(key)
    if current == 1:
        cache.expire(key, 3600)  # 1時間のウィンドウ
    if current > 100:  # 1時間あたり100件
        raise HTTPException(
            status_code=429,
            detail="レート制限を超過しました。1時間後に再試行してください"
        )


async def authenticate_api_key(api_key: str):
    """API キーの認証"""
    row = await db.fetch_one(
        "SELECT id, email, plan, rate_limit FROM users WHERE api_key = :key",
        {"key": api_key}
    )
    if not row:
        raise HTTPException(status_code=401, detail="無効な API キーです")
    return row
```

---

## 4. スケーラビリティ設計

### 4.1 キャッシュ戦略

```
                  読み取りリクエスト 115K QPS
                         |
                         v
                  +--------------+
                  | CDN / Edge   |  ← 最もホットな URL をエッジキャッシュ
                  | Cache        |    (80% のリクエストをここで処理)
                  | (CloudFront/ |    TTL: 5分 (頻繁にアクセスされる URL)
                  |  Fastly)     |
                  +--------------+
                         |  キャッシュミス (20%)
                         v
                  +--------------+
                  | Redis        |  ← アプリレベルキャッシュ
                  | Cluster      |    (残り 19% をここで処理)
                  | (6ノード)     |    TTL: 24時間
                  +--------------+
                         |  キャッシュミス (1%)
                         v
                  +--------------+
                  | DB Replica   |  ← リードレプリカ
                  | (Read)       |    (最終的なフォールバック)
                  +--------------+

  実効 DB アクセス: 115,000 × 0.01 = 1,150 QPS
  → DB への負荷を 99% 削減
```

### 4.2 Redis Cluster 設計

```python
# infrastructure/cache/redis_cache.py
import redis
from redis.sentinel import Sentinel
import json
import logging

logger = logging.getLogger(__name__)


class UrlCache:
    """URL キャッシュ（Redis Cluster / Sentinel 対応）

    設計判断:
    - Redis Cluster: 水平分散（16384 スロット）
    - Sentinel: 高可用性（マスター障害時の自動フェイルオーバー）
    - TTL: 24時間（ホットな URL は自然にキャッシュに残る）
    """

    DEFAULT_TTL = 86400  # 24時間

    def __init__(self, redis_client: redis.Redis):
        self._redis = redis_client

    def get_url(self, short_key: str) -> str | None:
        """キャッシュから URL を取得"""
        result = self._redis.get(f"url:{short_key}")
        if result:
            logger.debug(f"キャッシュヒット: {short_key}")
            return result
        logger.debug(f"キャッシュミス: {short_key}")
        return None

    def set_url(
        self, short_key: str, original_url: str, ttl: int = DEFAULT_TTL
    ) -> None:
        """URL をキャッシュに保存"""
        self._redis.setex(f"url:{short_key}", ttl, original_url)

    def delete_url(self, short_key: str) -> None:
        """キャッシュから URL を削除（URL 無効化時）"""
        self._redis.delete(f"url:{short_key}")

    def get_cache_stats(self) -> dict:
        """キャッシュの統計情報"""
        info = self._redis.info("stats")
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        return {
            "hits": hits,
            "misses": misses,
            "hit_rate": f"{(hits / total * 100):.1f}%" if total > 0 else "N/A",
            "total_keys": self._redis.dbsize(),
        }
```

### 4.3 データベースシャーディング

```python
# infrastructure/database/shard_router.py
import hashlib


class ConsistentHashShardRouter:
    """一貫性ハッシュに基づくシャードルーティング

    設計判断:
    - short_key のハッシュでシャードを決定
    - 一貫性ハッシュリングでシャード追加/削除時の影響を最小化
    - 仮想ノードで負荷の均等分散を実現

    ┌────────────────────────────────────────────┐
    │  一貫性ハッシュリング                         │
    │                                              │
    │       Shard0-v0                              │
    │      /          \                            │
    │   Shard2-v2   Shard1-v0                      │
    │     |              |                         │
    │   Shard1-v1   Shard0-v1                      │
    │      \          /                            │
    │       Shard2-v0                              │
    │                                              │
    │  short_key のハッシュ位置から                   │
    │  時計回りで最初のノードがシャード               │
    └────────────────────────────────────────────┘
    """

    def __init__(
        self,
        shard_configs: dict[int, str],
        virtual_nodes: int = 150,
    ):
        self._shard_configs = shard_configs
        self._virtual_nodes = virtual_nodes
        self._ring: dict[int, int] = {}
        self._sorted_keys: list[int] = []
        self._build_ring()

    def _build_ring(self) -> None:
        """ハッシュリングを構築"""
        for shard_id in self._shard_configs:
            for i in range(self._virtual_nodes):
                key = f"shard-{shard_id}-vnode-{i}"
                hash_val = self._hash(key)
                self._ring[hash_val] = shard_id
        self._sorted_keys = sorted(self._ring.keys())

    def get_shard(self, short_key: str) -> int:
        """短縮キーからシャード番号を決定"""
        hash_val = self._hash(short_key)
        # 二分探索で最初の仮想ノードを見つける
        for ring_key in self._sorted_keys:
            if hash_val <= ring_key:
                return self._ring[ring_key]
        # リングの末尾を超えた場合は先頭に戻る
        return self._ring[self._sorted_keys[0]]

    def get_connection_string(self, short_key: str) -> str:
        """シャードの接続文字列を取得"""
        shard_id = self.get_shard(short_key)
        return self._shard_configs[shard_id]

    @staticmethod
    def _hash(key: str) -> int:
        """キーのハッシュ値を計算"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)


# 使用例
router = ConsistentHashShardRouter(
    shard_configs={
        0: "mysql://user:pass@shard-0.db/urls",
        1: "mysql://user:pass@shard-1.db/urls",
        2: "mysql://user:pass@shard-2.db/urls",
        3: "mysql://user:pass@shard-3.db/urls",
    },
    virtual_nodes=150,
)

# 短縮キーからシャードを決定
shard_id = router.get_shard("abc123")
conn_str = router.get_connection_string("abc123")
```

### 4.4 高可用性設計

```
=== マルチ AZ デプロイメント ===

  Region: ap-northeast-1 (東京)

  AZ-a                    AZ-c                    AZ-d
  ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
  │ API Server   │        │ API Server   │        │ API Server   │
  │ × 3 instances│        │ × 3 instances│        │ × 3 instances│
  │              │        │              │        │              │
  │ Redis Master │        │ Redis Replica│        │ Redis Replica│
  │              │        │              │        │              │
  │ DB Master    │        │ DB Replica   │        │ DB Replica   │
  │              │        │              │        │              │
  │ KGS Primary  │        │              │        │ KGS Standby  │
  └──────────────┘        └──────────────┘        └──────────────┘
         ↑                       ↑                       ↑
         └───────────────────────┼───────────────────────┘
                                 │
                          [ALB / NLB]
                                 │
                          [Route 53]
                          (ヘルスチェック + フェイルオーバー)

  障害パターンと対応:
  1. API Server 障害 → ALB がヘルスチェックで検出、トラフィック迂回
  2. Redis Master 障害 → Sentinel がフェイルオーバー、Replica を昇格
  3. DB Master 障害 → 手動 or 自動フェイルオーバー、Replica を昇格
  4. AZ 全体障害 → Route 53 が他 AZ にルーティング
```

---

## 5. 301 vs 302 リダイレクト

### 5.1 詳細比較

| 観点 | 301 (Moved Permanently) | 302 (Found / Temporary) |
|------|------------------------|------------------------|
| ブラウザキャッシュ | キャッシュされる | キャッシュされない |
| サーバー負荷 | 低い（2回目以降はサーバーに来ない） | 高い（毎回サーバーに来る） |
| アクセス統計 | 正確に取れない（ブラウザがキャッシュ） | 毎回記録可能 |
| SEO | リンクジュースが転送先に移る | 元 URL にリンクジュースが残る |
| リダイレクト先の変更 | ブラウザキャッシュが残り反映遅延 | 即座に反映 |
| 推奨用途 | 統計不要で高速性重視 | アクセス分析が必要な場合 |

### 5.2 使い分けの判断基準

```python
def determine_redirect_status(url_mapping: dict) -> int:
    """リダイレクトのステータスコードを判断する

    判断基準:
    1. アクセス統計が有効 → 302（毎回サーバーを経由させる）
    2. カスタムエイリアス → 302（リダイレクト先の変更可能性）
    3. それ以外 → 301（パフォーマンス最優先）
    """
    if url_mapping.get('analytics_enabled'):
        return 302  # 毎回サーバーを経由 → 統計取得可能
    if url_mapping.get('custom_alias'):
        return 302  # リダイレクト先が変更される可能性
    if url_mapping.get('expires_at'):
        return 302  # 有効期限あり → 期限切れの検知が必要
    return 301      # 永続リダイレクト → 最高のパフォーマンス
```

---

## 6. テスト

### 6.1 API のテスト

```python
# tests/test_url_shortener_api.py
import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch


class TestShortenURL:
    """URL 短縮 API のテスト"""

    @pytest.fixture
    def mock_kgs(self):
        kgs = AsyncMock()
        kgs.get_key.return_value = "abc1234"
        return kgs

    @pytest.mark.asyncio
    async def test_URL短縮_正常(self, client: AsyncClient, mock_kgs):
        """有効な URL を短縮できる"""
        response = await client.post(
            "/api/v1/shorten",
            json={"url": "https://example.com/very/long/path"},
            headers={"X-API-Key": "valid-api-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "short_url" in data
        assert data["short_url"].startswith("https://short.ly/")
        assert len(data["short_key"]) == 7

    @pytest.mark.asyncio
    async def test_カスタムエイリアス(self, client: AsyncClient):
        """カスタムエイリアスで短縮 URL を作成できる"""
        response = await client.post(
            "/api/v1/shorten",
            json={
                "url": "https://example.com/event",
                "custom_alias": "my-event",
            },
            headers={"X-API-Key": "valid-api-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["short_key"] == "my-event"

    @pytest.mark.asyncio
    async def test_カスタムエイリアス重複(self, client: AsyncClient):
        """既存のエイリアスと重複する場合は 409 エラー"""
        # 1回目: 成功
        await client.post(
            "/api/v1/shorten",
            json={
                "url": "https://example.com/1",
                "custom_alias": "duplicate",
            },
            headers={"X-API-Key": "valid-api-key"},
        )

        # 2回目: 409 Conflict
        response = await client.post(
            "/api/v1/shorten",
            json={
                "url": "https://example.com/2",
                "custom_alias": "duplicate",
            },
            headers={"X-API-Key": "valid-api-key"},
        )
        assert response.status_code == 409

    @pytest.mark.asyncio
    async def test_不正なURL(self, client: AsyncClient):
        """不正な URL は 422 エラー"""
        response = await client.post(
            "/api/v1/shorten",
            json={"url": "not-a-valid-url"},
            headers={"X-API-Key": "valid-api-key"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_レート制限(self, client: AsyncClient):
        """レート制限を超えると 429 エラー"""
        for _ in range(101):  # 100 + 1 回
            response = await client.post(
                "/api/v1/shorten",
                json={"url": "https://example.com/test"},
                headers={"X-API-Key": "valid-api-key"},
            )

        assert response.status_code == 429


class TestRedirect:
    """リダイレクト API のテスト"""

    @pytest.mark.asyncio
    async def test_リダイレクト_正常(self, client: AsyncClient):
        """有効な短縮 URL でリダイレクトされる"""
        # URL を作成
        create_resp = await client.post(
            "/api/v1/shorten",
            json={"url": "https://example.com/target"},
            headers={"X-API-Key": "valid-api-key"},
        )
        short_key = create_resp.json()["short_key"]

        # リダイレクト
        response = await client.get(
            f"/{short_key}",
            follow_redirects=False,
        )
        assert response.status_code in (301, 302)
        assert response.headers["location"] == "https://example.com/target"

    @pytest.mark.asyncio
    async def test_存在しないキー(self, client: AsyncClient):
        """存在しない短縮キーは 404 エラー"""
        response = await client.get("/nonexistent", follow_redirects=False)
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_期限切れURL(self, client: AsyncClient):
        """期限切れの URL は 410 エラー"""
        # 期限切れ URL を直接 DB に作成（テスト用）
        # ... 省略
        response = await client.get("/expired-key", follow_redirects=False)
        assert response.status_code == 410


class TestKeyGenerationService:
    """KGS のテスト"""

    def test_キー生成(self):
        """指定数のキーが生成される"""
        kgs = KeyGenerationService(redis_client=fake_redis)
        generated = kgs.generate_keys(count=1000)
        assert generated == 1000
        assert kgs.pool_size() >= 1000

    def test_キー取得はアトミック(self):
        """同じキーが2回取得されない"""
        kgs = KeyGenerationService(redis_client=fake_redis)
        kgs.generate_keys(count=100)

        keys = set()
        for _ in range(100):
            key = kgs.get_key()
            assert key not in keys, f"重複キー: {key}"
            keys.add(key)

    def test_プール枯渇時はエラー(self):
        """プールが空の場合は RuntimeError"""
        kgs = KeyGenerationService(redis_client=fake_redis)
        # プールは空
        with pytest.raises(RuntimeError, match="キープール枯渇"):
            kgs.get_key()

    def test_キー返却(self):
        """失敗時にキーをプールに返却できる"""
        kgs = KeyGenerationService(redis_client=fake_redis)
        kgs.generate_keys(count=10)
        initial_size = kgs.pool_size()

        key = kgs.get_key()
        assert kgs.pool_size() == initial_size - 1

        kgs.return_key(key)
        assert kgs.pool_size() == initial_size
```

### 6.2 Base62 エンコーディングのテスト

```python
# tests/test_base62.py
import pytest


class TestBase62:
    """Base62 エンコーディングのテスト"""

    def test_エンコード_デコードの往復(self):
        """エンコード → デコードで元の値に戻る"""
        for num in [0, 1, 61, 62, 100, 123456789, 2**40]:
            encoded = encode_base62(num)
            decoded = decode_base62(encoded)
            assert decoded == num, f"num={num}, encoded={encoded}"

    def test_キー長の検証(self):
        """7文字以内で十分な範囲をカバーできる"""
        # 62^7 = 3,521,614,606,208
        max_7char = 62**7 - 1
        encoded = encode_base62(max_7char)
        assert len(encoded) == 7

    def test_ゼロ(self):
        """0 のエンコード"""
        assert encode_base62(0) == "0"

    def test_全文字が使われる(self):
        """Base62 の全文字（0-9, a-z, A-Z）が使われる"""
        chars_used = set()
        for i in range(62):
            chars_used.add(encode_base62(i))
        assert len(chars_used) == 62
```

---

## 7. 比較表

### 7.1 データベース選択

| 特性 | MySQL (RDB) | DynamoDB (NoSQL) | Cassandra |
|-----|:-----------:|:----------------:|:---------:|
| ルックアップ速度 | 速い（インデックス） | 非常に速い（ハッシュキー） | 速い |
| 水平スケール | シャーディングが必要 | 自動 | 自動 |
| スキーマ柔軟性 | 固定（ALTER必要） | 柔軟 | 柔軟 |
| トランザクション | 完全サポート | 制限的 | 制限的 |
| 運用コスト | 中（自前管理） | 低（マネージド） | 高（運用複雑） |
| 適用条件 | 中小規模、SQL が必要 | 大規模、シンプルなクエリ | 超大規模、書き込み重視 |

### 7.2 キー生成方式

| 方式 | 速度 | 衝突 | 分散対応 | 予測可能性 | 推奨場面 |
|-----|:----:|:----:|:-------:|:---------:|---------|
| Base62 カウンター | 高 | なし | 困難 | 高（連番） | 小規模・単一サーバー |
| MD5 ハッシュ | 中 | あり | 容易 | 低 | 同一URL同一キーが必要 |
| Snowflake + Base62 | 高 | なし | 容易 | 中 | 大規模分散 |
| KGS | 最高 | なし | 容易 | 低 | 超高スループット |

### 7.3 キャッシュ戦略

| 層 | 技術 | ヒット率 | TTL | 用途 |
|---|------|:-------:|:---:|------|
| L1: CDN Edge | CloudFront / Fastly | 80% | 5分 | ホット URL の高速配信 |
| L2: App Cache | Redis Cluster | 19% | 24時間 | アプリレベルの汎用キャッシュ |
| L3: DB Replica | MySQL Replica | 1% | N/A | キャッシュミスのフォールバック |

---

## 8. アンチパターン

### アンチパターン 1: 全リクエストを DB に直接アクセス

```
WHY: URL 短縮サービスは読み取りが 100:1 で圧倒的に多い。
     キャッシュなしでは DB が 115,000 QPS を処理できない。

NG:
  115,000 QPS → DB 直接アクセス → DB が即座にダウン

OK: 多層キャッシュ戦略を採用する
  80-20 の法則: 20% の URL が 80% のトラフィックを受ける
  CDN + Redis で 99% のリクエストを DB 到達前に処理
  実効 DB アクセス: 115,000 × 0.01 = 1,150 QPS
```

### アンチパターン 2: 短縮キーを連番で生成

```
WHY: 連番は予測可能であり、セキュリティリスクがある。
     全 URL を /1, /2, /3 ... で列挙可能。

NG:
  auto_increment の連番をそのまま短縮キーにする
  → 非公開 URL が推測される
  → 競合他社が全 URL をスクレイピング可能

OK: ランダムまたは事前生成キー (KGS) を使用する
  → 予測不可能な 7 文字のランダムキー
  → KGS で衝突チェックを事前に完了
```

### アンチパターン 3: リダイレクトでアクセス統計を同期記録

```
WHY: リダイレクト API のクリティカルパスに DB 書き込みを入れると、
     レイテンシが増加し、DB 書き込み障害がリダイレクトを阻害する。

NG:
  @app.get("/{short_key}")
  async def redirect(short_key):
      url = await get_url(short_key)
      await db.execute("INSERT INTO click_events ...")  # 同期書き込み
      await db.execute("UPDATE url_mappings SET click_count += 1")  # 同期更新
      return RedirectResponse(url)
  # → リダイレクトのレイテンシが DB 書き込みに依存

OK: 非同期でクリックイベントを記録
  @app.get("/{short_key}")
  async def redirect(short_key):
      url = await get_url(short_key)
      asyncio.create_task(record_click(short_key))  # 非同期
      return RedirectResponse(url)
  # → リダイレクトは即座に完了。統計はバックグラウンドで記録
```

### アンチパターン 4: 単一障害点の放置

```
WHY: KGS や Redis が単一インスタンスの場合、
     その障害で全サービスが停止する。

NG:
  [API Server] → [KGS (1台)] → 障害! → 全 URL 作成不能
  [API Server] → [Redis (1台)] → 障害! → 全リダイレクト遅延

OK: 各コンポーネントを冗長化
  KGS: Primary + Standby (自動フェイルオーバー)
  Redis: Cluster (6ノード) + Sentinel
  DB: Master + 2 Replica (異なる AZ)
  API: 複数インスタンス + ヘルスチェック
```

### アンチパターン 5: URL の安全性チェックなし

```
WHY: 短縮 URL はフィッシングやマルウェア配布に悪用される。
     安全性チェックなしでは、自サービスが攻撃の踏み台になる。

NG:
  def shorten(url):
      return create_short_url(url)  # 何でも受け入れる

OK: 多層の安全性チェック
  1. URL 作成時: Google Safe Browsing API でスキャン
  2. 定期スキャン: 既存 URL を定期的に再チェック
  3. ユーザー報告: 悪意ある URL の報告機能
  4. リダイレクト前: プレビューページの提供（オプション）
```

---

## 9. 演習問題

### 演習1: 基本 -- Base62 エンコーディングの実装（30分）

**課題**: Base62 エンコーディングの完全な実装

要件:
1. `encode_base62(num: int) -> str` と `decode_base62(s: str) -> int` を実装
2. 0 から 62^7-1 までの範囲で正しく動作すること
3. 負の数は ValueError を送出すること
4. property-based testing で encode/decode の往復が正しいことを検証

**期待する出力**:
```python
assert encode_base62(0) == "0"
assert encode_base62(61) == "Z"
assert encode_base62(62) == "10"
assert encode_base62(123456789) == "8M0kX"
assert decode_base62("8M0kX") == 123456789

# 往復テスト
for i in range(10000):
    assert decode_base62(encode_base62(i)) == i
```

### 演習2: 応用 -- KGS のスレッドセーフ実装（60分）

**課題**: マルチスレッド環境で安全に動作する Key Generation Service を実装

要件:
1. Redis の spop を使用してアトミックにキーを取得
2. プールサイズ監視と自動補充機能
3. キー返却機能（URL 作成失敗時）
4. 100 並行スレッドでの動作テスト

**期待する出力**:
```python
# 並行テスト
import concurrent.futures

kgs = KeyGenerationService(redis_client)
kgs.generate_keys(count=1000)

all_keys = []
with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
    futures = [executor.submit(kgs.get_key) for _ in range(1000)]
    for future in concurrent.futures.as_completed(futures):
        all_keys.append(future.result())

# 全キーがユニーク
assert len(set(all_keys)) == 1000
```

### 演習3: 発展 -- 完全なシステム設計ドキュメント（90分）

**課題**: 以下の追加要件を含む URL 短縮サービスの設計書を作成

追加要件:
1. マルチリージョン対応（東京 + バージニア）
2. カスタムドメインのサポート（`yourbrand.link/event`）
3. A/B テスト機能（同一短縮 URL で複数のリダイレクト先をランダムに振り分け）
4. 不正利用検出（短時間に大量のリダイレクトが発生した場合にアラート）

成果物:
- 高レベルアーキテクチャ図
- データモデル（テーブル設計）
- API 設計（エンドポイント一覧）
- 障害シナリオと対応策
- コスト見積もり（AWS ベース）

**期待する成果物の構成**:
```
1. 要件定義（機能/非機能）
2. アーキテクチャ図（マルチリージョン構成）
3. データモデル（カスタムドメイン、A/B テスト対応）
4. API 設計（RESTful + Rate Limiting）
5. 障害対応マトリクス
6. コスト概算
```

---

## 10. FAQ

### Q1: 短縮キーは何文字が最適ですか？

**A:** 6-7文字が一般的。Base62（英大小文字 + 数字）の場合:

```
文字数と表現可能な範囲:
  5文字: 62^5 =     916,132,832 (約9億)
  6文字: 62^6 =  56,800,235,584 (約568億)
  7文字: 62^7 = 3,521,614,606,208 (約3.5兆)

推奨:
  - 5年間で 182.5B URL を想定 → 7文字で十分
  - カスタムエイリアスは 4-30文字を許容
  - 短すぎるとブルートフォースで列挙される危険

面接では: 「7文字 Base62 で 3.5兆通り。
  5年間の 1,825億 URL に対して約20倍の余裕がある」と回答
```

### Q2: 同じ URL が複数回短縮された場合、同じキーを返すべきですか？

**A:** 設計判断に依存するが、一般的には異なるキーを返す。

```
方式1: 毎回異なるキーを返す（推奨）
  メリット:
  - ユーザーごとに独立したアクセス統計
  - 実装がシンプル（重複チェック不要）
  - 有効期限をユーザーごとに設定可能
  デメリット:
  - 同一 URL に対して複数キーが存在（ストレージ効率低下）

方式2: 同じキーを返す（デデュプリケーション）
  メリット:
  - ストレージ効率が良い
  - CDN キャッシュの効率向上
  デメリット:
  - original_url → short_key の逆引きインデックスが必要
  - ユーザーごとの統計が取れない
  - 有効期限の管理が複雑

面接では:
  「bit.ly は同じユーザーが同じ URL を短縮すると同じキーを返す。
   異なるユーザーなら異なるキーを返す」と回答
```

### Q3: 期限切れ URL の削除はどう行いますか？

**A:** 3つの戦略を組み合わせる。

```python
# 戦略1: Lazy Deletion（リダイレクト時に確認）
@app.get("/{short_key}")
async def redirect(short_key):
    url_data = await get_url_data(short_key)
    if url_data['expires_at'] < datetime.now():
        raise HTTPException(status_code=410, detail="期限切れ")
    return RedirectResponse(url_data['original_url'])

# 戦略2: Background Cleanup（定期バッチ）
# Cron: 毎日 3:00 AM に実行
async def cleanup_expired_urls():
    deleted = await db.execute(
        "DELETE FROM url_mappings WHERE expires_at < NOW() LIMIT 10000"
    )
    # Redis キャッシュからも削除
    # ...

# 戦略3: TTL in Cache
# Redis の TTL で自動期限切れ
cache.setex(f"url:{short_key}", ttl_seconds, original_url)

# DynamoDB の TTL（NoSQL の場合）
# expires_at カラムを TTL として設定 → 自動削除
```

### Q4: URL 短縮サービスの面接での進め方は？

**A:** 以下のフレームワークで 35-40分で回答する。

```
Step 1: 要件確認 (3-5分)
  - 機能要件: "URL 短縮とリダイレクトが必須。統計は Must Have ですか？"
  - 非機能要件: "想定するスケールは？ read:write 比率は？"
  - スケール見積もり: QPS、ストレージ、帯域幅の概算

Step 2: 高レベル設計 (10-15分)
  - API 設計: POST /api/shorten, GET /:short_key
  - アーキテクチャ図: Client → LB → API → Cache → DB
  - データモデル: url_mappings テーブル

Step 3: 深掘り (15-20分)
  面接官の関心に応じて深掘り:
  - キー生成: Base62 vs Hash vs KGS
  - キャッシュ戦略: 多層キャッシュ、TTL
  - DB 分割: シャーディング戦略
  - 可用性: レプリカ、フェイルオーバー

Step 4: まとめ (3-5分)
  - トレードオフの説明
  - さらなる改善点の提案
```

### Q5: read:write 比率が逆（write 重視）の場合はどうする？

**A:** データベースとキー生成の設計が変わる。

```
Write 重視の場合の変更点:

1. DB: Write-optimized DB (Cassandra, ScyllaDB)
   - LSM-tree ベースで書き込みに最適化
   - MySQL の B-tree より書き込みが高速

2. キー生成: KGS の強化
   - プールサイズを拡大（1000万キー以上）
   - 複数 KGS インスタンスで並列取得
   - キー生成をバックグラウンドで常時実行

3. 非同期書き込み:
   - URL 作成リクエスト → メッセージキュー → 非同期 DB 書き込み
   - レスポンスはキュー投入後に即座に返す

4. バッチ書き込み:
   - 書き込みをバッファリングしてバッチ INSERT
   - レイテンシとスループットのトレードオフ
```

### Q6: マルチリージョンでの整合性はどう担保する？

**A:** 結果整合性を許容し、リージョン間の非同期レプリケーションを採用する。

```
設計:
  Tokyo Region  ←── async replication ──→  Virginia Region

  1. 書き込み: 最寄りのリージョンで受け付け
  2. レプリケーション: 非同期で他リージョンに伝播（数百ms〜数秒の遅延）
  3. 読み取り: 最寄りのリージョンのレプリカから応答

  衝突の可能性:
  - 同一カスタムエイリアスが異なるリージョンで同時に作成される
  - 解決: カスタムエイリアスは「グローバル一意性チェック」を同期的に実行
         ランダムキーは衝突の確率が極めて低いため非同期で問題なし

  DNS ルーティング:
  Route 53 のレイテンシベースルーティングで最寄りリージョンに誘導
```

### Q7: コスト見積もりは？

**A:** AWS ベースの概算（月額）。

```
=== 月額コスト概算（100M URL/日の場合）===

API サーバー:
  EC2 c6g.2xlarge × 12台 (4AZ × 3台)
  → $12 × 0.268/hr × 730hr = $2,348/月

ロードバランサー:
  ALB × 2 (internal + external)
  → $16.43 × 2 + トラフィック ≈ $500/月

Redis (ElastiCache):
  r6g.2xlarge × 6ノード (Cluster)
  → $0.452/hr × 6 × 730hr = $1,980/月

RDB (Aurora MySQL):
  db.r6g.4xlarge × 3 (Master + 2 Replica)
  → $1.12/hr × 3 × 730hr = $2,453/月
  ストレージ: 91TB × $0.10 = $9,100/月

CDN (CloudFront):
  10B req/日 × 30日 × $0.009/10K req ≈ $2,700/月

合計: 約 $19,000/月 (約280万円/月)

コスト最適化のポイント:
  - リザーブドインスタンスで 30-40% 削減
  - Spot インスタンスをバッチ処理に活用
  - S3 + Athena でクリックログをコスト効率よく保存
```

---

## まとめ

| 設計要素 | 選択 | 理由 |
|----------|------|------|
| キー生成 | KGS (事前生成) | 衝突なし、高速、分散対応 |
| キー長 | 7文字 (Base62) | 3.5兆通りで5年間十分 |
| リダイレクト | 301（統計不要時）/ 302（統計必要時） | 要件に応じて選択 |
| キャッシュ | CDN + Redis Cluster | 読み取り QPS に対応、99% の DB 負荷削減 |
| データベース | MySQL (Aurora) + シャーディング | 信頼性と拡張性のバランス |
| 可用性 | マルチ AZ + レプリカ + フェイルオーバー | 99.99% SLA |
| セキュリティ | URL スキャン + レート制限 | 悪用防止 |
| 統計収集 | 非同期（メッセージキュー経由） | リダイレクトのレイテンシに影響しない |

---

## 次に読むべきガイド

- [チャットシステム設計](./01-chat-system.md) — リアルタイム通信のシステム設計
- [通知システム設計](./02-notification-system.md) — 大規模通知配信の設計
- [レート制限設計](./03-rate-limiter.md) — API レート制限の詳細設計
- [検索エンジン設計](./04-search-engine.md) — 全文検索システムの設計
- [データベース](../01-components/01-database.md) — シャーディング・レプリケーションの詳細
- [キャッシュ](../01-components/00-cache.md) — キャッシュ戦略の体系的解説
- [イベント駆動アーキテクチャ](../02-architecture/03-event-driven.md) — 非同期処理の設計パターン

---

## 参考文献

1. **System Design Interview: An Insider's Guide** — Alex Xu (2020) — Chapter 8: Design a URL Shortener
2. **Designing Data-Intensive Applications** — Martin Kleppmann (O'Reilly, 2017) — Chapter 5-6: Replication and Partitioning
3. **Scaling Memcache at Facebook** — Nishtala, R. et al. (NSDI '13, 2013) — 大規模キャッシュシステムの設計
4. **Consistent Hashing and Random Trees** — Karger, D. et al. (STOC '97, 1997) — 一貫性ハッシュの理論的基盤
5. **Twitter Snowflake** — https://blog.twitter.com/engineering/en_us/a/2010/announcing-snowflake — 分散 ID 生成の実践
6. **bit.ly Architecture** — https://highscalability.com/bitly-lessons-learned-building-a-distributed-system-that-han/ — 実運用の知見
