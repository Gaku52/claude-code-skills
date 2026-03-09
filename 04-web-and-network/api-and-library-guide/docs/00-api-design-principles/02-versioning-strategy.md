# バージョニング戦略

> APIのバージョニングは後方互換性と進化のバランスを追求する技術的判断の集合体である。破壊的変更の定義、URI/ヘッダー/クエリパラメータベースの戦略、セマンティックバージョニング、非推奨化プロセスを深く理解し、長期運用に耐えるAPIを設計する。

## この章で学ぶこと

- [ ] 破壊的変更と非破壊的変更を明確に区別できるようになる
- [ ] 3つのバージョニング方式（URI、ヘッダー、クエリパラメータ）の特性と選択基準を把握する
- [ ] セマンティックバージョニングの原則をAPIに適用する方法を理解する
- [ ] 非推奨化と移行のプロセスを設計・実行できるようになる
- [ ] バージョンレス設計の思想と実践手法を習得する
- [ ] 破壊的変更の影響を最小化するための戦略的アプローチを身につける

## 前提知識

- API First設計の概念 → 参照: [API First設計](./00-api-first-design.md)
- API命名規則の基本 → 参照: [命名規則と慣例](./01-naming-and-conventions.md)
- セマンティックバージョニング（SemVer）の基本知識

---

## 1. なぜAPIバージョニングが必要なのか

ソフトウェアは常に進化する。ビジネス要件の変化、技術的負債の解消、セキュリティ対応など、APIを変更する理由は尽きない。しかし、APIはプロバイダーとコンシューマーの間の「契約」であり、一方的な変更はシステム全体を破壊しかねない。

```
APIバージョニングが解決する根本的課題:

  問題:
  ┌──────────────┐     契約（API仕様）     ┌──────────────┐
  │  API Provider │◄──────────────────────►│  API Consumer │
  │  （サーバー） │                         │ （クライアント）│
  └──────────────┘                         └──────────────┘
       │                                        │
       │ 仕様を変更したい                        │ 既存動作を維持したい
       │ （機能追加/修正/改善）                   │ （安定稼働が最優先）
       │                                        │
       └─────────── 利害の対立 ──────────────────┘

  解決策: バージョニング
  ┌──────────────┐     v1（旧契約）        ┌──────────────┐
  │  API Provider │◄──────────────────────►│  旧クライアント │
  │              │     v2（新契約）        ┌──────────────┐
  │              │◄──────────────────────►│  新クライアント │
  └──────────────┘                         └──────────────┘
```

### 1.1 バージョニングなしのリスク

バージョニングを設けない場合に発生し得る問題を整理する。

| リスク分類 | 具体例 | 影響度 |
|-----------|--------|--------|
| 機能破壊 | フィールド名変更でクライアントのパースが失敗 | 致命的 |
| データ喪失 | レスポンス構造の変更でデータのマッピングが不正確に | 重大 |
| 信頼喪失 | 予告なき仕様変更でパートナー企業の信頼を失う | 長期的 |
| 運用コスト | 緊急ロールバックやホットフィックスの頻発 | 中〜高 |
| 法的リスク | SLA違反による契約上の問題 | 状況依存 |

### 1.2 バージョニング導入のタイミング

APIを公開する時点でバージョニング戦略を決定しておくことが望ましい。後からバージョニングを追加するのは、事実上「最初の破壊的変更」を引き起こすためである。

```
推奨されるタイミング:

  ✓ API設計フェーズ        → バージョニング方式の決定
  ✓ 初回リリース（GA）     → v1 としてリリース
  ✓ 破壊的変更が必要な時   → v2 をリリースし、v1 と並行運用
  ✗ 内部APIのみの段階      → 厳密なバージョニングは過剰な場合がある
  ✗ プロトタイプ/α版       → 頻繁な破壊的変更は許容される
```

---

## 2. 破壊的変更の定義と分類

### 2.1 変更の3分類

APIの変更は、クライアントへの影響度に応じて3つに分類される。

```
分類マトリクス:

  影響度
  高 │  ┌─────────────────────────────────────┐
     │  │  破壊的変更（Breaking Changes）      │
     │  │  バージョンアップが必須               │
     │  │  例: フィールド削除、型変更           │
     │  └─────────────────────────────────────┘
  中 │  ┌─────────────────────────────────────┐
     │  │  グレーゾーン（Gray Area）            │
     │  │  クライアント実装に依存               │
     │  │  例: デフォルト値変更、順序変更       │
     │  └─────────────────────────────────────┘
  低 │  ┌─────────────────────────────────────┐
     │  │  非破壊的変更（Non-Breaking Changes） │
     │  │  バージョンアップ不要                 │
     │  │  例: フィールド追加、エンドポイント追加│
     │  └─────────────────────────────────────┘
     └──────────────────────────────────────────
```

### 2.2 非破壊的変更（バージョンアップ不要）

以下の変更はクライアントの動作を壊さないため、バージョンアップなしで適用できる。

```
非破壊的変更の詳細リスト:

  レスポンス関連:
    ✓ レスポンスにフィールドを追加
    ✓ レスポンスヘッダーの追加
    ✓ エラーメッセージの文言改善（コードが変わらない場合）
    ✓ レスポンス速度の改善

  リクエスト関連:
    ✓ オプショナルなリクエストパラメータの追加
    ✓ オプショナルなリクエストヘッダーの追加

  エンドポイント関連:
    ✓ 新しいエンドポイントの追加
    ✓ 新しいHTTPメソッドのサポート追加
    ✓ 新しいリソースタイプの追加

  ドキュメント関連:
    ✓ APIドキュメントの改善
    ✓ 使用例の追加

  前提条件:
    → クライアントは「Robustness Principle（堅牢性の原則）」に従い、
      未知のフィールドを無視する実装であること
    → "Be conservative in what you send, be liberal in what you accept"
      （Postel's Law / RFC 761）
```

### 2.3 破壊的変更（バージョンアップ必須）

以下の変更はクライアントの動作を壊す可能性が高く、新しいバージョンとしてリリースすべきである。

```python
# コード例1: 破壊的変更の具体例

# === フィールド削除 ===
# v1レスポンス
{
    "user": {
        "id": 123,
        "name": "田中太郎",
        "email": "tanaka@example.com",
        "phone": "03-1234-5678"    # v2で削除 → 破壊的
    }
}

# v2レスポンス（phoneフィールドが消失）
{
    "user": {
        "id": 123,
        "name": "田中太郎",
        "email": "tanaka@example.com"
        # phone が存在しない → クライアントのuser.phoneがnull/undefinedに
    }
}

# === 型変更 ===
# v1: idがinteger
{"id": 123, "name": "田中太郎"}

# v2: idがstring（破壊的変更）
{"id": "usr_123", "name": "田中太郎"}
# → クライアントがidを数値として処理している場合に障害発生

# === 必須パラメータの追加 ===
# v1: POST /api/v1/users
# Body: {"name": "田中太郎"}  ← nameだけでOK

# v2: POST /api/v2/users
# Body: {"name": "田中太郎", "email": "tanaka@example.com"}
# ← emailが必須に → 旧クライアントのリクエストが400エラーに

# === エンドポイントの変更 ===
# v1: GET /api/v1/users/{id}/orders
# v2: GET /api/v2/customers/{id}/purchases  ← パスもリソース名も変更
```

### 2.4 グレーゾーン変更

クライアント実装の品質に依存する変更群。保守的に判断するならば、破壊的変更として扱うのが安全である。

```python
# コード例2: グレーゾーン変更の具体例と判断基準

# === デフォルト値の変更 ===
# v1: GET /api/v1/users → ソート順がデフォルトで created_at ASC
# v2: GET /api/v1/users → ソート順がデフォルトで updated_at DESC
# 判断: クライアントがデフォルトのソート順に依存している場合は破壊的

# === ページネーションのデフォルトサイズ変更 ===
# v1: デフォルトで1ページ100件
# v2: デフォルトで1ページ20件
# 判断: 全件取得前提のクライアントにとっては破壊的

# === レスポンスフィールドの並び順変更 ===
# v1: {"id": 1, "name": "foo", "email": "bar"}
# v2: {"name": "foo", "email": "bar", "id": 1}
# 判断: JSON仕様上は順序不定だが、位置ベースのパーサーには影響する

# === エラーコードの追加 ===
# v1: 認証エラーは常に 401
# v2: 認証エラーを 401（認証なし）と 403（権限不足）に分離
# 判断: クライアントが401のみをハンドリングしている場合は影響あり

# === Null許容性の変更 ===
# v1: "address"フィールドは常にオブジェクト
# v2: "address"フィールドがnullになり得る
# 判断: null チェックなしのクライアントでNullPointerException等が発生

# === 判断フローチャート ===
# 1. クライアントのコード変更が必要か？
#    → Yes: 破壊的変更として扱う
#    → No:  次のチェックへ
# 2. クライアントの動作が変わるか？
#    → Yes: グレーゾーン（保守的に判断）
#    → No:  非破壊的変更
# 3. 影響を受けるクライアントの割合は？
#    → 多い: 破壊的変更として扱う
#    → 少ない: ケースバイケースで判断
```

### 2.5 破壊的変更の影響分析テンプレート

破壊的変更を検討する際は、以下のテンプレートで影響を事前分析することを推奨する。

| 分析項目 | 内容 |
|---------|------|
| 変更内容 | 具体的に何を変更するか |
| 変更理由 | なぜこの変更が必要か |
| 影響範囲 | 影響を受けるエンドポイント一覧 |
| クライアント影響 | どのクライアントが影響を受けるか |
| 移行コスト | クライアント側の修正に必要な工数 |
| 代替案 | 非破壊的な方法で同じ目的を達成できないか |
| ロールバック計画 | 問題発生時の復旧手順 |
| スケジュール | 告知→並行運用→旧バージョン終了の時系列 |

---

## 3. バージョニング方式の詳細比較

APIバージョニングには主要な3つの方式と、いくつかの派生形がある。それぞれの方式を実装例とともに詳しく解説する。

### 3.1 URI パスバージョニング

最も広く採用されているバージョニング方式。バージョン番号をURLパスに含める。

```
URI パスバージョニングのパターン:

  標準パターン:
    https://api.example.com/v1/users
    https://api.example.com/v2/users

  サブドメインパターン:
    https://v1.api.example.com/users
    https://v2.api.example.com/users

  パスプレフィックスパターン:
    https://api.example.com/api/v1/users
    https://api.example.com/api/v2/users

  リクエストフロー:
  ┌──────────┐    GET /v1/users    ┌───────────┐    route    ┌───────────┐
  │ クライアント│──────────────────►│ APIゲート  │──────────►│ v1ハンドラ │
  └──────────┘                    │ ウェイ     │           └───────────┘
                                  │           │
  ┌──────────┐    GET /v2/users    │           │    route    ┌───────────┐
  │ クライアント│──────────────────►│           │──────────►│ v2ハンドラ │
  └──────────┘                    └───────────┘           └───────────┘
```

```python
# コード例3: URI パスバージョニングの実装（Python / Flask）

from flask import Flask, jsonify, request
from functools import wraps

app = Flask(__name__)

# --- バージョンルーティングの基本実装 ---

# v1: ユーザーリソース
@app.route('/api/v1/users', methods=['GET'])
def get_users_v1():
    """v1ではシンプルなユーザー情報を返す"""
    users = fetch_users_from_db()
    return jsonify({
        "users": [
            {
                "id": u.id,
                "name": u.name,
                "email": u.email
            }
            for u in users
        ]
    })

# v2: ユーザーリソース（拡張版）
@app.route('/api/v2/users', methods=['GET'])
def get_users_v2():
    """v2ではページネーション付きのレスポンスを返す"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)

    pagination = fetch_users_paginated(page, per_page)

    return jsonify({
        "data": [
            {
                "id": u.id,
                "full_name": u.name,        # name → full_name に変更
                "email": u.email,
                "profile": {                  # v2で追加されたネスト構造
                    "avatar_url": u.avatar_url,
                    "bio": u.bio,
                    "created_at": u.created_at.isoformat()
                }
            }
            for u in pagination.items
        ],
        "pagination": {                       # v2で追加されたメタ情報
            "current_page": pagination.page,
            "total_pages": pagination.pages,
            "total_items": pagination.total,
            "per_page": pagination.per_page
        }
    })

# --- Blueprintによるバージョン分離（推奨パターン） ---

from flask import Blueprint

# バージョンごとにBlueprintを作成
v1_bp = Blueprint('v1', __name__, url_prefix='/api/v1')
v2_bp = Blueprint('v2', __name__, url_prefix='/api/v2')

@v1_bp.route('/users', methods=['GET'])
def v1_get_users():
    return jsonify({"version": "v1", "users": []})

@v2_bp.route('/users', methods=['GET'])
def v2_get_users():
    return jsonify({"version": "v2", "data": [], "pagination": {}})

# アプリケーションに登録
app.register_blueprint(v1_bp)
app.register_blueprint(v2_bp)

# --- バージョン非推奨ミドルウェア ---

DEPRECATED_VERSIONS = {'v1'}
SUNSET_DATES = {'v1': 'Sat, 01 Jan 2026 00:00:00 GMT'}

@app.before_request
def add_deprecation_headers():
    """非推奨バージョンへのリクエストに警告ヘッダーを付与"""
    path = request.path
    for version in DEPRECATED_VERSIONS:
        if f'/api/{version}/' in path:
            # RFC 8594 Sunset Header
            from flask import g
            g.deprecated_version = version

@app.after_request
def inject_sunset_header(response):
    """レスポンスにSunsetヘッダーを挿入"""
    from flask import g
    version = getattr(g, 'deprecated_version', None)
    if version:
        response.headers['Deprecation'] = 'true'
        response.headers['Sunset'] = SUNSET_DATES.get(version, '')
        response.headers['Link'] = (
            f'<https://api.example.com/api/v2/docs>; '
            f'rel="successor-version"'
        )
    return response
```

### 3.2 ヘッダーバージョニング

HTTPヘッダーを使ってバージョンを指定する方式。RESTの原則であるコンテンツネゴシエーションを活用する。

```
ヘッダーバージョニングのバリエーション:

  ① Accept ヘッダー（メディアタイプ）:
    Accept: application/vnd.example.v1+json
    Accept: application/vnd.example.v2+json

  ② カスタムヘッダー:
    X-API-Version: 1
    X-API-Version: 2

  ③ Accept ヘッダー（パラメータ付き）:
    Accept: application/json; version=1
    Accept: application/json; version=2

  リクエストフロー:
  ┌──────────┐   GET /users              ┌───────────┐
  │ クライアント│  Accept: ...vnd.v1+json  │           │   ┌───────────┐
  │          │──────────────────────────►│ APIゲート  │──►│ v1シリアラ │
  └──────────┘                          │ ウェイ     │   │ イザ       │
                                        │           │   └───────────┘
  ┌──────────┐   GET /users              │ (ヘッダー  │
  │ クライアント│  Accept: ...vnd.v2+json  │  で分岐)   │   ┌───────────┐
  │          │──────────────────────────►│           │──►│ v2シリアラ │
  └──────────┘                          └───────────┘   │ イザ       │
                                                        └───────────┘

  注意: URLは全バージョンで同一（/users）
```

```python
# コード例4: ヘッダーバージョニングの実装（Python / Flask）

from flask import Flask, jsonify, request, abort
from functools import wraps

app = Flask(__name__)

# --- メディアタイプパーサー ---

def parse_api_version(accept_header: str) -> int:
    """
    Acceptヘッダーからバージョンを抽出する。

    対応フォーマット:
      - application/vnd.example.v1+json → 1
      - application/vnd.example.v2+json → 2
      - application/json; version=1     → 1
      - application/json                → デフォルト（最新安定版）
    """
    if not accept_header:
        return get_default_version()

    # vnd フォーマットの解析
    import re
    vnd_match = re.search(
        r'application/vnd\.example\.v(\d+)\+json',
        accept_header
    )
    if vnd_match:
        return int(vnd_match.group(1))

    # パラメータフォーマットの解析
    param_match = re.search(
        r'application/json;\s*version=(\d+)',
        accept_header
    )
    if param_match:
        return int(param_match.group(1))

    return get_default_version()

def get_default_version() -> int:
    """バージョン未指定時のデフォルトバージョンを返す"""
    return 2  # 最新安定版

# --- バージョンデコレータ ---

def api_version(version: int):
    """指定バージョンでのみアクセス可能にするデコレータ"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            requested_version = parse_api_version(
                request.headers.get('Accept', '')
            )
            if requested_version != version:
                return None  # このハンドラではない

            response = f(*args, **kwargs)

            # Content-Typeにバージョン情報を含める
            if hasattr(response, 'headers'):
                response.headers['Content-Type'] = (
                    f'application/vnd.example.v{version}+json'
                )
            return response
        return wrapper
    return decorator

# --- マルチバージョンディスパッチャー ---

class VersionedEndpoint:
    """複数バージョンのハンドラを管理するディスパッチャー"""

    def __init__(self):
        self.handlers = {}

    def version(self, v: int):
        """バージョンごとのハンドラを登録するデコレータ"""
        def decorator(f):
            self.handlers[v] = f
            return f
        return decorator

    def dispatch(self):
        """リクエストのバージョンに応じて適切なハンドラを呼び出す"""
        requested_version = parse_api_version(
            request.headers.get('Accept', '')
        )
        handler = self.handlers.get(requested_version)
        if handler is None:
            abort(406, description=(
                f"API version {requested_version} is not supported. "
                f"Supported versions: {list(self.handlers.keys())}"
            ))
        return handler()

# 使用例
users_endpoint = VersionedEndpoint()

@users_endpoint.version(1)
def get_users_v1():
    return jsonify({"users": [{"id": 1, "name": "田中太郎"}]})

@users_endpoint.version(2)
def get_users_v2():
    return jsonify({
        "data": [{"id": 1, "full_name": "田中太郎", "profile": {}}],
        "pagination": {"page": 1, "total": 1}
    })

@app.route('/api/users', methods=['GET'])
def users():
    return users_endpoint.dispatch()
```

### 3.3 クエリパラメータバージョニング

URLのクエリパラメータでバージョンを指定する方式。

```
クエリパラメータバージョニングのパターン:

  基本形:
    GET /api/users?version=1
    GET /api/users?version=2
    GET /api/users?v=2

  リクエストフロー:
  ┌──────────┐  GET /users?version=1   ┌───────────┐   ┌──────────┐
  │ クライアント│─────────────────────►│ ルーター    │──►│ v1ロジック │
  └──────────┘                        │           │   └──────────┘
                                      │ ?version  │
  ┌──────────┐  GET /users?version=2   │ を解析    │   ┌──────────┐
  │ クライアント│─────────────────────►│           │──►│ v2ロジック │
  └──────────┘                        └───────────┘   └──────────┘

  省略時の挙動（3パターン）:
    A) 最新版をデフォルト → GET /users → v2 が返る
    B) 最古版をデフォルト → GET /users → v1 が返る（安全寄り）
    C) エラーを返す      → GET /users → 400 Bad Request

    推奨: パターンA（最新安定版をデフォルト）
```

### 3.4 3方式の総合比較

| 比較項目 | URI パス | ヘッダー | クエリパラメータ |
|---------|---------|---------|---------------|
| 可読性 | 非常に高い | 低い | 高い |
| ブラウザテスト | 容易 | 困難（curl等が必要） | 容易 |
| キャッシュ | バージョン別に自然分離 | Varyヘッダーが必要 | パラメータ含むキーが必要 |
| CDNルーティング | 容易 | CDN依存 | 可能だが設定複雑 |
| REST原則準拠 | やや違反（同一リソースに複数URI） | 準拠（コンテンツネゴシエーション） | やや違反 |
| ハイパーメディア | バージョンがリンクに含まれる | リンクはバージョン非依存 | パラメータがリンクに混入 |
| 実装の容易さ | 非常に容易 | やや複雑 | 容易 |
| API Gateway対応 | 全ゲートウェイ対応 | 主要ゲートウェイ対応 | 全ゲートウェイ対応 |
| 主な採用例 | GitHub, Twitter, Google | Stripe (カスタムヘッダー) | Amazon, Netflix (一部) |
| 推奨度 | 高（最も一般的） | 中〜高（純粋なREST） | 中（手軽だが非推奨傾向） |

### 3.5 ハイブリッドアプローチ

実際のプロダクションでは、複数の方式を組み合わせるケースもある。

```
ハイブリッドの例:

  ① URI + ヘッダーの併用:
    → メジャーバージョンはURIで管理: /api/v1/, /api/v2/
    → マイナーバージョンはヘッダーで管理: X-API-Minor-Version: 3
    → 適用例: メジャー変更は稀、マイナー改善は頻繁なAPI

  ② URI + 日付ベースの併用:
    → メジャーバージョンはURI: /api/v2/
    → 日付バージョンはヘッダー: API-Version: 2024-01-15
    → 適用例: Stripe風の細やかなバージョン管理

  ③ APIゲートウェイでのバージョン統合:
    ┌─────────────────────────────────────────┐
    │              API Gateway                 │
    │                                          │
    │  /v1/* ─────────────► Backend v1         │
    │  /v2/* ─────────────► Backend v2         │
    │  Accept:vnd.v3+json ► Backend v3 (beta)  │
    │                                          │
    │  ルール: URIバージョンを優先、            │
    │         なければヘッダーを確認             │
    └─────────────────────────────────────────┘
```

---

## 4. セマンティックバージョニングとAPIバージョン

### 4.1 セマンティックバージョニング（SemVer）の基本

セマンティックバージョニング（Semantic Versioning / SemVer）は、バージョン番号に意味を持たせる規則体系である。

```
SemVer形式: MAJOR.MINOR.PATCH

  MAJOR（メジャー）: 後方互換性のない変更
    1.0.0 → 2.0.0
    例: フィールド削除、型変更、エンドポイント再設計

  MINOR（マイナー）: 後方互換性のある機能追加
    1.0.0 → 1.1.0
    例: 新しいエンドポイント追加、オプショナルパラメータ追加

  PATCH（パッチ）: 後方互換性のあるバグ修正
    1.0.0 → 1.0.1
    例: レスポンスのバグ修正、ドキュメント修正

  プレリリース:
    2.0.0-alpha.1  → アルファ版
    2.0.0-beta.1   → ベータ版
    2.0.0-rc.1     → リリース候補

  バージョン番号の比較順序:
    1.0.0-alpha < 1.0.0-alpha.1 < 1.0.0-beta
    < 1.0.0-beta.2 < 1.0.0-rc.1 < 1.0.0
```

### 4.2 SemVerとAPIバージョンのマッピング

APIのURIバージョニングでは通常メジャーバージョンのみを公開するが、内部的にはSemVerで管理することが望ましい。

```
マッピング戦略:

  内部バージョン       公開APIバージョン     URIパス
  ─────────────────  ─────────────────    ──────────
  1.0.0              v1                    /api/v1/
  1.1.0              v1                    /api/v1/  （新機能追加）
  1.2.0              v1                    /api/v1/  （さらに機能追加）
  1.2.1              v1                    /api/v1/  （バグ修正）
  2.0.0              v2                    /api/v2/  （破壊的変更）
  2.1.0              v2                    /api/v2/  （新機能追加）

  ポイント:
  → URIのバージョンはメジャーバージョンのみ
  → マイナー/パッチはURIに影響しない
  → 内部バージョンはCHANGELOGやドキュメントで管理
  → レスポンスヘッダーで詳細バージョンを通知可能:
     X-API-Version: 2.1.0
```

### 4.3 CHANGELOG管理

```markdown
# コード例5: CHANGELOG.mdの構成例

# Changelog

All notable changes to the Example API will be documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/).

## [2.1.0] - 2025-06-15

### Added
- `GET /api/v2/users/{id}/preferences` エンドポイントを追加
- ユーザーレスポンスに `timezone` フィールドを追加（オプショナル）
- バッチ取得 `POST /api/v2/users/batch` をサポート

### Changed
- `GET /api/v2/users` のデフォルトページサイズを50から20に変更

### Deprecated
- `GET /api/v2/users/{id}/settings` は v2.3.0 で削除予定
  代替: `GET /api/v2/users/{id}/preferences`

## [2.0.0] - 2025-01-10 [BREAKING]

### Breaking Changes
- ユーザーレスポンスの `name` フィールドを `full_name` に変更
- `GET /api/v1/users` のフラットレスポンスを
  `data` + `pagination` のラッパー構造に変更
- 認証方式を API Key から OAuth 2.0 に変更

### Added
- ユーザープロフィール情報（avatar_url, bio）を追加
- ページネーションメタ情報をレスポンスに含める

### Migration Guide
- 詳細は [v1→v2 移行ガイド](./migration/v1-to-v2.md) を参照

## [1.2.1] - 2024-11-20

### Fixed
- `GET /api/v1/users` でメールアドレスが null の場合に
  500 エラーが返る問題を修正

## [1.2.0] - 2024-10-01

### Added
- ユーザー検索 `GET /api/v1/users?search=keyword` をサポート
- レスポンスに `created_at` フィールドを追加
```

---

## 5. バージョン移行プロセスの設計

### 5.1 非推奨化（Deprecation）の完全フロー

APIバージョンの廃止は、段階的かつ透明性のあるプロセスで進めなければならない。

```
非推奨化タイムライン（標準プラン）:

  T-12ヶ月  T-6ヶ月   T-3ヶ月   T-1ヶ月    T（廃止日）
  ────────────────────────────────────────────────────►
  │         │         │         │          │
  │         │         │         │          └─ 410 Gone を返す
  │         │         │         │             移行先へのリダイレクト情報
  │         │         │         │
  │         │         │         └─ 最終警告メール送信
  │         │         │            レート制限を厳格化
  │         │         │            ダッシュボードで警告表示
  │         │         │
  │         │         └─ Deprecation ヘッダー付与開始
  │         │            主要クライアントへの個別通知
  │         │            移行ガイドの公開
  │         │
  │         └─ 新バージョン GA リリース
  │            並行運用開始
  │            ドキュメントで非推奨を明記
  │
  └─ 新バージョンベータ公開
     移行計画の策定開始
     パートナーへの事前告知

  エンタープライズプラン: 上記の2倍の期間（24ヶ月並行運用）
```

### 5.2 HTTP ヘッダーによる非推奨通知

RFC 8594（Sunset Header）と関連するヘッダーを活用して、プログラマティックに非推奨を通知する。

```http
HTTP/1.1 200 OK
Content-Type: application/json
Deprecation: true
Sunset: Sat, 01 Jul 2026 00:00:00 GMT
Link: <https://api.example.com/v2/docs>; rel="successor-version"
Link: <https://api.example.com/v1-to-v2-migration>; rel="deprecation"
X-API-Version: 1.2.1
X-API-Warn: "This API version is deprecated. Please migrate to v2."

{
  "data": { ... },
  "_deprecation_notice": {
    "message": "API v1 は 2026年7月1日に廃止されます。v2への移行をお願いします。",
    "migration_guide": "https://api.example.com/v1-to-v2-migration",
    "sunset_date": "2026-07-01T00:00:00Z"
  }
}
```

### 5.3 使用状況のモニタリング

旧バージョンの使用状況を継続的に監視し、廃止判断の根拠とする。

```
モニタリングダッシュボード（概念図）:

  API v1 利用状況
  ═══════════════════════════════════════════

  日次リクエスト数:
  1月 ████████████████████████████████ 320K
  2月 ██████████████████████████████   300K
  3月 ████████████████████████         240K
  4月 ██████████████████               180K  ← 移行ガイド公開
  5月 ████████████                     120K
  6月 ████████                          80K  ← 最終警告
  7月 ███                               30K  ← 廃止予定月

  ユニーククライアント数:
  1月: 45社  → 7月: 3社（個別対応で移行支援）

  監視すべきメトリクス:
  - リクエスト数（日次/週次/月次）
  - ユニーククライアント数
  - エラー率の変化
  - レスポンスタイムの変化
  - 新バージョン(v2)の採用率
```

### 5.4 移行ガイドの構成

移行ガイドはクライアント開発者が最も参照するドキュメントであり、以下の要素を含むべきである。

```
移行ガイドの必須セクション:

  1. 変更概要サマリー
     → 何がなぜ変わったのかを簡潔に説明

  2. 変更点の詳細一覧
     → 各フィールド/エンドポイントの変更をテーブル形式で記載

  3. 新旧マッピング表
     → v1のフィールドがv2のどのフィールドに対応するかを明示

  4. コード例（Before / After）
     → 主要言語ごとの移行コードサンプル

  5. FAQ
     → 移行時によくある質問と回答

  6. スケジュール
     → 廃止日、マイルストーン

  7. サポート情報
     → 問い合わせ先、Slackチャンネル、メーリングリスト
```

### 5.5 新旧フィールドマッピングの設計

バージョン移行において最も重要なのは、新旧フィールドの対応関係を明確にすることである。

| v1 フィールド | v2 フィールド | 変更種別 | 備考 |
|-------------|-------------|---------|------|
| `name` | `full_name` | リネーム | フィールド名変更のみ、値は同一 |
| `email` | `email` | 変更なし | そのまま移行可能 |
| `id` (integer) | `id` (string) | 型変更 | `"usr_"` プレフィックス付きに変更 |
| `phone` | 削除 | 削除 | `profile.phone_number` に移動 |
| (なし) | `profile` | 新規追加 | ネストされたオブジェクト |
| (なし) | `profile.avatar_url` | 新規追加 | - |
| (なし) | `profile.bio` | 新規追加 | - |
| (なし) | `profile.phone_number` | 移動 | v1 の `phone` が移動 |
| (なし) | `profile.created_at` | 新規追加 | ISO 8601 形式 |

---

## 6. バージョンレス設計（Evolvable API）

### 6.1 バージョンレスAPIの思想

バージョンレス設計とは、明示的なバージョン番号を使わずにAPIを進化させるアプローチである。「バージョンを作らなくて済むようにAPI設計を工夫する」という考え方が根底にある。

```
バージョンレス設計の原則:

  ┌──────────────────────────────────────────────────┐
  │         バージョンレスAPIの4つの柱               │
  │                                                  │
  │  ① Additive Changes Only（追加のみ）            │
  │     → フィールドの追加は許可、削除/変更は禁止    │
  │                                                  │
  │  ② Robustness Principle（堅牢性原則）            │
  │     → クライアントは未知のフィールドを無視する   │
  │                                                  │
  │  ③ Optional by Default（デフォルトでオプショナル）│
  │     → 新しいフィールドは常にオプショナル          │
  │                                                  │
  │  ④ Explicit Contract（明示的契約）               │
  │     → 何が保証されるかを明確にドキュメント化     │
  └──────────────────────────────────────────────────┘
```

### 6.2 日付ベースバージョニング（Stripe方式）

Stripeが採用する日付ベースバージョニングは、バージョンレスと明示的バージョニングのハイブリッドとして高い評価を受けている。

```python
# コード例6: 日付ベースバージョニングの実装概念

from datetime import date
from flask import Flask, jsonify, request

app = Flask(__name__)

# バージョン定義: 日付 → 変更内容のマッピング
VERSION_CHANGES = {
    '2024-01-15': {
        'description': 'Initial GA release',
        'changes': []
    },
    '2024-06-01': {
        'description': 'Add profile field to user response',
        'changes': [
            {
                'type': 'field_added',
                'endpoint': '/users',
                'field': 'profile',
                'default': None
            }
        ]
    },
    '2025-01-10': {
        'description': 'Rename name to full_name',
        'changes': [
            {
                'type': 'field_renamed',
                'endpoint': '/users',
                'old_field': 'name',
                'new_field': 'full_name'
            }
        ]
    },
    '2025-06-15': {
        'description': 'Change id from integer to string',
        'changes': [
            {
                'type': 'field_type_changed',
                'endpoint': '/users',
                'field': 'id',
                'old_type': 'integer',
                'new_type': 'string',
                'transform': lambda v: f'usr_{v}'
            }
        ]
    }
}

SUPPORTED_VERSIONS = sorted(VERSION_CHANGES.keys())
DEFAULT_VERSION = SUPPORTED_VERSIONS[-1]  # 最新

def get_requested_version() -> str:
    """リクエストからAPIバージョンを取得する"""
    # Stripe-Version ヘッダーを確認
    version = request.headers.get('Stripe-Version',
              request.headers.get('API-Version'))

    if version and version in VERSION_CHANGES:
        return version

    # アカウントのデフォルトバージョンを使用
    # （実際のStripeではアカウント作成時のバージョンが固定される）
    return DEFAULT_VERSION

def transform_response(data: dict, endpoint: str,
                       requested_version: str) -> dict:
    """
    リクエストされたバージョンに合わせてレスポンスを変換する。
    最新のデータ構造を基準に、古いバージョン向けに変換を適用する。
    """
    result = data.copy()

    # 新しいバージョンの変更を逆順に適用（巻き戻し）
    for version_date in reversed(SUPPORTED_VERSIONS):
        if version_date <= requested_version:
            break  # リクエストされたバージョンまで到達

        changes = VERSION_CHANGES[version_date]['changes']
        for change in changes:
            if change.get('endpoint') != endpoint:
                continue

            if change['type'] == 'field_renamed':
                # 新名→旧名に巻き戻す
                new_field = change['new_field']
                old_field = change['old_field']
                if new_field in result:
                    result[old_field] = result.pop(new_field)

            elif change['type'] == 'field_added':
                # 追加されたフィールドを除去
                field = change['field']
                result.pop(field, None)

            elif change['type'] == 'field_type_changed':
                # 型変更を巻き戻す（string → integer）
                field = change['field']
                if field in result and isinstance(result[field], str):
                    result[field] = int(
                        result[field].replace('usr_', '')
                    )

    return result

@app.route('/api/users/<user_id>', methods=['GET'])
def get_user(user_id):
    """日付ベースバージョニングによるユーザー取得"""
    requested_version = get_requested_version()

    # 内部では常に最新のデータ構造を使用
    user_data = {
        'id': f'usr_{user_id}',
        'full_name': '田中太郎',
        'email': 'tanaka@example.com',
        'profile': {
            'avatar_url': 'https://example.com/avatar.jpg',
            'bio': 'ソフトウェアエンジニア'
        }
    }

    # リクエストされたバージョンに合わせて変換
    response_data = transform_response(
        user_data, '/users', requested_version
    )

    response = jsonify(response_data)
    response.headers['API-Version'] = requested_version
    return response

# --- 使用例 ---
# 最新バージョン（2025-06-15）:
#   curl -H "API-Version: 2025-06-15" https://api.example.com/api/users/123
#   → {"id": "usr_123", "full_name": "田中太郎", ...}

# 古いバージョン（2024-01-15）:
#   curl -H "API-Version: 2024-01-15" https://api.example.com/api/users/123
#   → {"id": 123, "name": "田中太郎", ...}  # 旧構造で返る
```

### 6.3 フィールド選択によるバージョンレス化

GraphQLのようにクライアントが必要なフィールドを選択するアプローチ。REST APIにおいてもフィールド選択パラメータを導入することで、破壊的変更の影響を軽減できる。

```python
# コード例7: フィールド選択の実装

from flask import Flask, jsonify, request

app = Flask(__name__)

def filter_fields(data: dict, fields: list) -> dict:
    """指定されたフィールドのみを含むレスポンスを構築する"""
    if not fields:
        return data  # fieldsが未指定なら全フィールドを返す

    result = {}
    for field in fields:
        # ドット記法によるネストフィールドのサポート
        # 例: "profile.avatar_url"
        parts = field.split('.')
        source = data
        target = result
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                # 最後のパート: 値をコピー
                if part in source:
                    target[part] = source[part]
            else:
                # 中間パート: ネスト構造を構築
                if part in source and isinstance(source[part], dict):
                    if part not in target:
                        target[part] = {}
                    source = source[part]
                    target = target[part]
                else:
                    break
    return result

@app.route('/api/users', methods=['GET'])
def get_users():
    """
    フィールド選択パラメータの使用例:
      GET /api/users?fields=id,full_name,email
      GET /api/users?fields=id,profile.avatar_url
    """
    fields_param = request.args.get('fields', '')
    fields = [f.strip() for f in fields_param.split(',')
              if f.strip()] if fields_param else []

    users = [
        {
            'id': 'usr_1',
            'full_name': '田中太郎',
            'email': 'tanaka@example.com',
            'profile': {
                'avatar_url': 'https://example.com/avatar1.jpg',
                'bio': 'エンジニア',
                'phone_number': '03-1234-5678'
            }
        },
        {
            'id': 'usr_2',
            'full_name': '佐藤花子',
            'email': 'sato@example.com',
            'profile': {
                'avatar_url': 'https://example.com/avatar2.jpg',
                'bio': 'デザイナー',
                'phone_number': '03-9876-5432'
            }
        }
    ]

    filtered_users = [filter_fields(u, fields) for u in users]
    return jsonify({'data': filtered_users})

# --- リクエスト例と結果 ---

# 全フィールド:
# GET /api/users
# → {"data": [{"id": "usr_1", "full_name": "...", ...}]}

# 選択フィールド:
# GET /api/users?fields=id,full_name
# → {"data": [{"id": "usr_1", "full_name": "田中太郎"},
#              {"id": "usr_2", "full_name": "佐藤花子"}]}

# ネストフィールド:
# GET /api/users?fields=id,profile.avatar_url
# → {"data": [{"id": "usr_1", "profile": {"avatar_url": "..."}},
#              {"id": "usr_2", "profile": {"avatar_url": "..."}}]}
```

### 6.4 Feature Flagsによる段階的公開

新機能をフィーチャーフラグで制御することで、同一バージョン内で段階的に機能を公開する手法。

```
Feature Flagsの段階的展開:

  Phase 1: Internal Testing
  ┌──────────────────────────────────────────┐
  │ Feature: enhanced_user_profile           │
  │ Enabled: internal_testers only (0.1%)    │
  │ Status: Alpha                            │
  └──────────────────────────────────────────┘

  Phase 2: Beta Partners
  ┌──────────────────────────────────────────┐
  │ Feature: enhanced_user_profile           │
  │ Enabled: beta_partners + internal (5%)   │
  │ Status: Beta                             │
  └──────────────────────────────────────────┘

  Phase 3: Gradual Rollout
  ┌──────────────────────────────────────────┐
  │ Feature: enhanced_user_profile           │
  │ Enabled: 25% → 50% → 75% → 100%        │
  │ Status: GA                               │
  └──────────────────────────────────────────┘

  Phase 4: Default On
  ┌──────────────────────────────────────────┐
  │ Feature: enhanced_user_profile           │
  │ Enabled: 100% (default)                  │
  │ Status: Standard                         │
  └──────────────────────────────────────────┘

  リクエストでのフラグ指定:
    GET /api/users?include=enhanced_profile
    GET /api/users?features=new_pagination,enhanced_profile
```

---

## 7. APIゲートウェイにおけるバージョン管理

### 7.1 ゲートウェイパターン

大規模なAPI運用では、APIゲートウェイがバージョンルーティングの中心的な役割を果たす。

```
APIゲートウェイのバージョンルーティング:

  ┌─────────────┐
  │  クライアント  │
  └──────┬──────┘
         │
         ▼
  ┌──────────────────────────────────────────────────┐
  │              API Gateway (Kong / AWS API GW)      │
  │                                                    │
  │  ┌────────────────────────────────────────────┐   │
  │  │          バージョンルーティング              │   │
  │  │                                            │   │
  │  │  /v1/*  ──────────►  Backend Service v1    │   │
  │  │                      (legacy, maintenance) │   │
  │  │                                            │   │
  │  │  /v2/*  ──────────►  Backend Service v2    │   │
  │  │                      (current, active)     │   │
  │  │                                            │   │
  │  │  /v3-beta/* ──────►  Backend Service v3    │   │
  │  │                      (preview, unstable)   │   │
  │  └────────────────────────────────────────────┘   │
  │                                                    │
  │  追加機能:                                         │
  │  ├─ レート制限（バージョン別に設定可能）            │
  │  ├─ 認証・認可                                     │
  │  ├─ リクエスト/レスポンス変換                      │
  │  ├─ キャッシュ（バージョン別）                     │
  │  ├─ アクセスログ（バージョン別メトリクス）         │
  │  └─ 非推奨ヘッダーの自動付与                       │
  └──────────────────────────────────────────────────┘
```

### 7.2 リクエスト/レスポンス変換パターン

ゲートウェイでリクエストやレスポンスを変換することで、バックエンドは最新バージョンのみを実装し、旧バージョンとの互換性をゲートウェイ層で吸収するパターンがある。

```
変換パターンのアーキテクチャ:

  クライアント(v1)        API Gateway           バックエンド(v2のみ)
  ─────────────        ──────────────         ──────────────────
       │                     │                       │
       │  GET /v1/users      │                       │
       │────────────────────►│                       │
       │                     │  リクエスト変換         │
       │                     │  /v1/users → /v2/users│
       │                     │  パラメータマッピング   │
       │                     │──────────────────────►│
       │                     │                       │
       │                     │◄──────────────────────│
       │                     │  v2レスポンス          │
       │                     │                       │
       │                     │  レスポンス変換         │
       │                     │  full_name → name     │
       │                     │  profile削除           │
       │◄────────────────────│                       │
       │  v1レスポンス        │                       │

  利点:
  ✓ バックエンドは最新バージョンのみ実装すればよい
  ✓ 旧バージョンのメンテナンスコストがゲートウェイ設定に集約される
  ✓ バックエンドのコードが単純に保てる

  欠点:
  ✗ ゲートウェイの変換ルールが複雑化する
  ✗ 変換によるパフォーマンスオーバーヘッド
  ✗ 変換ルールのテスト/デバッグが困難
```

---

## 8. マイクロサービスにおけるバージョニング

### 8.1 サービス間APIのバージョニング

マイクロサービスアーキテクチャでは、外部向けAPIだけでなくサービス間の内部APIもバージョニングの対象となる。

```
マイクロサービス間のバージョニング考慮事項:

  ┌───────────┐     v2      ┌───────────┐     v1      ┌───────────┐
  │ Order      │────────────►│ User       │────────────►│ Billing    │
  │ Service    │             │ Service    │             │ Service    │
  │ (v2依存)   │             │ (v1,v2公開)│             │ (v1公開)   │
  └───────────┘             └───────────┘             └───────────┘
       │                         │
       │ v1                      │ v1
       ▼                         ▼
  ┌───────────┐             ┌───────────┐
  │ Inventory  │             │ Notification│
  │ Service    │             │ Service     │
  │ (v1公開)   │             │ (v1公開)    │
  └───────────┘             └───────────┘

  戦略:
  ① Consumer-Driven Contracts（CDC）
     → 各コンシューマーが期待する契約をテストで定義
     → プロバイダーは全コンシューマーの契約を満たすことを検証
     → ツール: Pact, Spring Cloud Contract

  ② Tolerant Reader パターン
     → コンシューマーは必要なフィールドのみを読み取る
     → 未知のフィールドは無視する
     → レスポンス構造の追加的変更に強い

  ③ Schema Registry
     → Avro/Protobuf等のスキーマをレジストリで一元管理
     → スキーマの互換性を自動チェック
     → ツール: Confluent Schema Registry
```

### 8.2 イベント駆動アーキテクチャにおけるバージョニング

非同期通信（イベント/メッセージ）のバージョニングも重要な課題である。

```python
# コード例8: イベントスキーマのバージョニング

import json
from datetime import datetime
from typing import Any

class VersionedEvent:
    """バージョン付きイベントの基底クラス"""

    def __init__(self, event_type: str, version: int,
                 payload: dict):
        self.metadata = {
            'event_id': generate_uuid(),
            'event_type': event_type,
            'version': version,
            'timestamp': datetime.utcnow().isoformat(),
            'source': 'user-service'
        }
        self.payload = payload

    def to_dict(self) -> dict:
        return {
            'metadata': self.metadata,
            'payload': self.payload
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# --- イベント定義: ユーザー作成 ---

class UserCreatedEventV1(VersionedEvent):
    """v1: シンプルなユーザー作成イベント"""

    def __init__(self, user_id: int, name: str, email: str):
        super().__init__(
            event_type='user.created',
            version=1,
            payload={
                'user_id': user_id,
                'name': name,
                'email': email
            }
        )


class UserCreatedEventV2(VersionedEvent):
    """v2: プロフィール情報を含むユーザー作成イベント"""

    def __init__(self, user_id: str, full_name: str,
                 email: str, profile: dict):
        super().__init__(
            event_type='user.created',
            version=2,
            payload={
                'user_id': user_id,      # string型に変更
                'full_name': full_name,   # name → full_name
                'email': email,
                'profile': profile        # 新規追加
            }
        )


# --- イベントコンシューマー: バージョン対応 ---

class UserEventConsumer:
    """複数バージョンのイベントを処理するコンシューマー"""

    def handle(self, event_json: str) -> None:
        event = json.loads(event_json)
        version = event['metadata']['version']
        payload = event['payload']

        handler = getattr(self, f'_handle_v{version}', None)
        if handler is None:
            # 未知のバージョン: ログを記録して処理をスキップ
            log_warning(
                f"Unknown event version: {version}, "
                f"event_type: {event['metadata']['event_type']}"
            )
            return

        handler(payload)

    def _handle_v1(self, payload: dict) -> None:
        """v1イベントの処理"""
        user_id = payload['user_id']
        name = payload['name']
        email = payload['email']
        # v1の処理ロジック
        create_user_record(user_id, name, email)

    def _handle_v2(self, payload: dict) -> None:
        """v2イベントの処理"""
        user_id = payload['user_id']
        full_name = payload['full_name']
        email = payload['email']
        profile = payload.get('profile', {})
        # v2の処理ロジック
        create_user_record_v2(user_id, full_name, email, profile)


# --- Upcastingパターン ---

class EventUpcaster:
    """古いバージョンのイベントを最新バージョンに変換する"""

    @staticmethod
    def upcast(event: dict) -> dict:
        version = event['metadata']['version']
        payload = event['payload']

        # v1 → v2 への変換
        if version == 1:
            payload = {
                'user_id': str(payload['user_id']),
                'full_name': payload['name'],
                'email': payload['email'],
                'profile': {}  # デフォルト値で補完
            }
            event['metadata']['version'] = 2
            event['payload'] = payload

        return event
```

---

## 9. テスト戦略

### 9.1 バージョン互換性テスト

APIの各バージョンが正しく動作することを保証するためのテスト戦略を設計する。

```python
# コード例9: バージョン互換性テストの実装（pytest）

import pytest
import json
from app import create_app

@pytest.fixture
def client():
    app = create_app(testing=True)
    with app.test_client() as client:
        yield client

class TestUserEndpointV1:
    """v1 ユーザーエンドポイントのテスト"""

    def test_get_users_v1_response_structure(self, client):
        """v1のレスポンス構造が正しいことを検証"""
        response = client.get('/api/v1/users')
        data = json.loads(response.data)

        assert response.status_code == 200
        assert 'users' in data
        assert isinstance(data['users'], list)

        if data['users']:
            user = data['users'][0]
            # v1では 'name' フィールドが存在する（'full_name'ではない）
            assert 'name' in user
            assert 'full_name' not in user
            # v1では 'id' が integer
            assert isinstance(user['id'], int)
            # v1では 'profile' は存在しない
            assert 'profile' not in user

    def test_get_users_v1_deprecation_headers(self, client):
        """v1が非推奨の場合、適切なヘッダーが返ることを検証"""
        response = client.get('/api/v1/users')

        assert response.headers.get('Deprecation') == 'true'
        assert 'Sunset' in response.headers
        assert 'Link' in response.headers

class TestUserEndpointV2:
    """v2 ユーザーエンドポイントのテスト"""

    def test_get_users_v2_response_structure(self, client):
        """v2のレスポンス構造が正しいことを検証"""
        response = client.get('/api/v2/users')
        data = json.loads(response.data)

        assert response.status_code == 200
        assert 'data' in data
        assert 'pagination' in data
        assert isinstance(data['data'], list)

        if data['data']:
            user = data['data'][0]
            # v2では 'full_name' フィールドが存在する
            assert 'full_name' in user
            assert 'name' not in user
            # v2では 'id' が string
            assert isinstance(user['id'], str)
            # v2では 'profile' が存在する
            assert 'profile' in user

    def test_get_users_v2_pagination(self, client):
        """v2のページネーションが正しく動作することを検証"""
        response = client.get('/api/v2/users?page=1&per_page=10')
        data = json.loads(response.data)

        pagination = data['pagination']
        assert 'current_page' in pagination
        assert 'total_pages' in pagination
        assert 'total_items' in pagination
        assert pagination['current_page'] == 1

class TestVersionCompatibility:
    """バージョン間の互換性テスト"""

    def test_v1_and_v2_same_data(self, client):
        """v1とv2が同じデータソースから返すことを検証"""
        v1_response = client.get('/api/v1/users')
        v2_response = client.get('/api/v2/users')

        v1_data = json.loads(v1_response.data)
        v2_data = json.loads(v2_response.data)

        # ユーザー数が同じであること
        assert len(v1_data['users']) == len(v2_data['data'])

    def test_v1_name_maps_to_v2_full_name(self, client):
        """v1のnameがv2のfull_nameに対応することを検証"""
        v1_response = client.get('/api/v1/users/1')
        v2_response = client.get('/api/v2/users/1')

        v1_user = json.loads(v1_response.data)
        v2_user = json.loads(v2_response.data)

        assert v1_user['name'] == v2_user['data']['full_name']

class TestDateBasedVersioning:
    """日付ベースバージョニングのテスト"""

    def test_old_version_returns_old_structure(self, client):
        """古いバージョンを指定すると旧構造が返ることを検証"""
        response = client.get(
            '/api/users/123',
            headers={'API-Version': '2024-01-15'}
        )
        data = json.loads(response.data)

        # 旧構造: idがinteger, nameフィールド
        assert isinstance(data['id'], int)
        assert 'name' in data
        assert 'full_name' not in data

    def test_new_version_returns_new_structure(self, client):
        """新しいバージョンを指定すると新構造が返ることを検証"""
        response = client.get(
            '/api/users/123',
            headers={'API-Version': '2025-06-15'}
        )
        data = json.loads(response.data)

        # 新構造: idがstring, full_nameフィールド
        assert isinstance(data['id'], str)
        assert 'full_name' in data
        assert 'name' not in data

    def test_version_header_in_response(self, client):
        """レスポンスにバージョン情報が含まれることを検証"""
        response = client.get(
            '/api/users/123',
            headers={'API-Version': '2024-06-01'}
        )

        assert response.headers.get('API-Version') == '2024-06-01'
```

### 9.2 Contract Testing（契約テスト）

Consumer-Driven Contract（CDC）テストにより、プロバイダーとコンシューマー間の契約を自動検証する。

```
Contract Testingのフロー:

  ┌──────────────┐                      ┌──────────────┐
  │  Consumer     │                      │  Provider     │
  │  (Order Svc)  │                      │  (User Svc)   │
  └──────┬───────┘                      └──────┬───────┘
         │                                     │
         │  1. コンシューマーが契約を定義         │
         │  ┌─────────────────────────┐        │
         │  │ "GET /users/1 を呼ぶと   │        │
         │  │  id, full_name, email    │        │
         │  │  を含むJSONが返る"       │        │
         │  └───────────┬─────────────┘        │
         │              │                       │
         │              │  2. 契約をブローカーに公開
         │              ▼                       │
         │     ┌─────────────────┐              │
         │     │  Pact Broker     │              │
         │     │  (契約の保管庫)   │              │
         │     └────────┬────────┘              │
         │              │                       │
         │              │  3. プロバイダーが契約を検証
         │              └──────────────────────►│
         │                                      │
         │                  4. 検証結果をブローカーに報告
         │              ┌───────────────────────│
         │              ▼                       │
         │     ┌─────────────────┐              │
         │     │  Pact Broker     │              │
         │     │  ✓ 検証成功      │              │
         │     │  or              │              │
         │     │  ✗ 検証失敗      │              │
         │     └─────────────────┘              │
```

---

## 10. アンチパターン

### 10.1 アンチパターン1: バージョン番号のインフレーション

頻繁にメジャーバージョンを上げてしまい、クライアントが追従できなくなるパターン。

```
アンチパターン: バージョン爆発

  悪い例:
  ┌──────────────────────────────────────────────────────┐
  │  2024年1月:  /api/v1/users  ← 初期リリース          │
  │  2024年3月:  /api/v2/users  ← 小さな変更でv2に      │
  │  2024年5月:  /api/v3/users  ← また小さな変更でv3に  │
  │  2024年7月:  /api/v4/users  ← レスポンス追加でv4に  │
  │  2024年9月:  /api/v5/users  ← パフォーマンス改善でv5│
  │  2024年11月: /api/v6/users  ← 新フィールド追加でv6  │
  │                                                      │
  │  結果:                                                │
  │  ✗ クライアントが6バージョンのうちどれを使えばよいか  │
  │    判断できない                                       │
  │  ✗ 各バージョンのメンテナンスコストが膨大             │
  │  ✗ ドキュメントがバージョンごとに分散                 │
  │  ✗ 開発チームが旧バージョンの動作を把握しきれない     │
  └──────────────────────────────────────────────────────┘

  正しいアプローチ:
  ┌──────────────────────────────────────────────────────┐
  │  2024年1月:  /api/v1/users  ← 初期リリース          │
  │  2024年3月:  /api/v1/users  ← 非破壊的変更（v1維持）│
  │  2024年5月:  /api/v1/users  ← 非破壊的変更（v1維持）│
  │  2024年7月:  /api/v1/users  ← 非破壊的変更（v1維持）│
  │  2025年1月:  /api/v2/users  ← 蓄積した破壊的変更を  │
  │                                まとめてv2にリリース  │
  │                                                      │
  │  原則:                                                │
  │  ✓ 破壊的変更をバッチにまとめてバージョンアップ       │
  │  ✓ 非破壊的変更は現行バージョンに追加                 │
  │  ✓ メジャーバージョンは1〜2年に1回程度が目安          │
  │  ✓ 同時並行運用は最大2〜3バージョンに抑える          │
  └──────────────────────────────────────────────────────┘
```

**問題の本質**: 非破壊的変更を破壊的変更と誤認してバージョンを上げている。変更が非破壊的であれば、新しいバージョンは不要である。「追加」「オプショナル化」は非破壊的変更であり、バージョンアップの理由にならない。

**対策**:
- 破壊的変更の明確な定義を策定し、チーム全体で共有する
- バージョンアップの承認プロセスを設ける（アーキテクチャレビュー等）
- 非破壊的変更の手法（フィールド追加、Feature Flags等）を積極活用する

### 10.2 アンチパターン2: バージョン固定の放置（ゾンビバージョン）

旧バージョンを非推奨にも廃止にもせず、永続的に運用し続けてしまうパターン。

```
アンチパターン: ゾンビバージョン

  悪い例:
  ┌──────────────────────────────────────────────────────┐
  │  /api/v1/users  ← 2020年リリース。まだ稼働中         │
  │  /api/v2/users  ← 2022年リリース。まだ稼働中         │
  │  /api/v3/users  ← 2024年リリース。最新               │
  │                                                      │
  │  v1の状態:                                            │
  │  ✗ セキュリティパッチが当たっていない                 │
  │  ✗ 旧ライブラリに依存（EOLのフレームワーク上で動作） │
  │  ✗ 担当者が退職済みで、コードの理解者がいない        │
  │  ✗ テストが壊れたまま放置されている                  │
  │  ✗ しかし「誰かが使っているかもしれない」ので停止不可│
  └──────────────────────────────────────────────────────┘

  根本原因:
  ① 非推奨化プロセスが定義されていない
  ② 利用状況のモニタリングが行われていない
  ③ 「停止するとクレームが来る」という恐怖
  ④ 廃止のための予算/工数が確保されていない

  正しいアプローチ:
  ┌──────────────────────────────────────────────────────┐
  │  リリース時点で廃止計画を策定する:                    │
  │                                                      │
  │  v1 リリース時:                                      │
  │   → 「v2リリース後12ヶ月で廃止」をドキュメントに明記 │
  │   → SLAに並行運用期間を記載                          │
  │                                                      │
  │  v2 リリース時:                                      │
  │   → v1の非推奨化を開始                               │
  │   → Deprecation/Sunsetヘッダーを付与                 │
  │   → 利用状況モニタリングを開始                       │
  │                                                      │
  │  v1 廃止日:                                          │
  │   → 410 Gone を返す                                  │
  │   → リダイレクト情報を含める                         │
  │   → 旧コードを完全に削除                             │
  └──────────────────────────────────────────────────────┘
```

**問題の本質**: 旧バージョンの廃止は技術的な判断だけでなく、ビジネス判断とプロセス設計の問題である。明確な廃止ポリシーがないと、旧バージョンは永遠に残り続ける。

**対策**:
- APIライフサイクルポリシーを策定し、SLAの一部として公開する
- 旧バージョンの利用状況を定期的にレビューする（月次等）
- 廃止判断の閾値を定める（例: 月間リクエストが全体の1%未満になったら廃止検討）
- 廃止工数をスプリント計画に組み込む

---

## 11. エッジケース分析

### 11.1 エッジケース1: 複数バージョンにまたがるトランザクション

クライアントが複数のエンドポイントを組み合わせて1つのトランザクションを構成している場合、一部のエンドポイントだけがバージョンアップすると整合性の問題が発生する。

```
エッジケース: 部分的バージョンアップの罠

  シナリオ:
  クライアントの処理フロー（注文作成）:
    1. POST /api/v2/orders        ← v2に移行済み
    2. GET  /api/v1/users/{id}    ← v1のまま
    3. POST /api/v2/payments      ← v2に移行済み

  問題:
  ┌──────────────────────────────────────────────────┐
  │  v2のordersは user_id を string ("usr_123") で    │
  │  受け取る仕様に変更された。                       │
  │                                                    │
  │  しかし v1 の users は id を integer (123) で       │
  │  返すため、クライアントが v1 から取得した id を     │
  │  v2 の orders に渡すと型不一致エラーが発生する。   │
  │                                                    │
  │  1. GET /api/v1/users/123 → {"id": 123, ...}      │
  │  2. POST /api/v2/orders                             │
  │     Body: {"user_id": 123}  ← integer!             │
  │     → 400 Bad Request: user_id must be string      │
  └──────────────────────────────────────────────────┘

  対策:
  ① バージョン整合性ポリシーの策定:
     → 同一クライアントは全エンドポイントで同じバージョンを使用
     → ゲートウェイでバージョン混在を検出・警告

  ② 型変換の互換レイヤー:
     → v2のordersがinteger型のuser_idも受け付ける
     → 内部で自動変換: 123 → "usr_123"
     → ただし、これは技術的負債になりやすい

  ③ クロスバージョン互換テスト:
     → v1とv2の組み合わせをCIで自動テスト
     → 互換性マトリクスを維持
```

### 11.2 エッジケース2: キャッシュとバージョンの不整合

CDNやブラウザキャッシュにバージョン間で不整合なデータが残るケース。

```
エッジケース: キャッシュ汚染

  シナリオ:
  ┌────────┐  GET /users  ┌─────┐  GET /v1/users  ┌──────┐
  │クライアント│──────────────►│ CDN │────────────────►│Backend│
  └────────┘              └─────┘                └──────┘

  問題の発生パターン:

  1. 12:00 - クライアントAが /api/v1/users をリクエスト
     → CDNがv1レスポンスをキャッシュ（TTL: 1時間）

  2. 12:30 - APIプロバイダーがv1を廃止し、/api/v1/ を削除
     → /api/v1/ へのリクエストは 410 Gone を返すようになる

  3. 12:45 - クライアントBが /api/v1/users をリクエスト
     → CDNがキャッシュからv1レスポンスを返す（古いデータ）
     → クライアントBはv1がまだ生きていると認識

  4. 13:00 - CDNキャッシュが期限切れ
     → 以降はバックエンドの 410 Gone が返る
     → クライアントBが突然エラーに遭遇

  対策:
  ① バージョン別のキャッシュキー設定:
     Cache-Control: public, max-age=3600
     Vary: Accept, API-Version

  ② 廃止前のキャッシュパージ:
     → CDN上の旧バージョンキャッシュを廃止と同時にパージ
     → CloudFront: Invalidation, Fastly: Purge API

  ③ 段階的なTTL短縮:
     → 廃止3ヶ月前: TTLを1時間→10分に短縮
     → 廃止1ヶ月前: TTLを10分→1分に短縮
     → 廃止当日: キャッシュ無効（no-cache）

  ④ Surrogate-Keyによる選択的パージ:
     Surrogate-Key: api-v1 users-list
     → 廃止時に "api-v1" タグのキャッシュを一括パージ
```

### 11.3 エッジケース3: モバイルアプリとバージョン強制

モバイルアプリケーションではユーザーがアプリを更新しない限り古いAPIバージョンが呼ばれ続ける。ウェブアプリと異なり、クライアントのバージョンをサーバー側で制御できない。

```
モバイルアプリのバージョニング課題:

  ウェブアプリ:
    サーバーデプロイ → 全ユーザーが即座に新バージョンを利用
    → バージョン移行が容易

  モバイルアプリ:
    ストア公開 → ユーザーが更新しない限り旧バージョンが残存
    → 6ヶ月後でも旧バージョンのインストールが30%以上

  対策:
  ① アプリ内強制アップデート:
     → APIレスポンスで最小サポートバージョンを通知
     → 旧アプリは更新ダイアログを表示
     → ただし、UXへの影響が大きいため慎重に判断

  ② APIバージョンの長期サポート:
     → モバイルアプリ向けAPIは最低18〜24ヶ月サポート
     → デスクトップ/ウェブ向けより長い並行運用期間が必要

  ③ クライアントバージョン別メトリクス:
     → User-Agentやカスタムヘッダーでアプリバージョンを識別
     → 古いアプリバージョンの利用率が5%未満になったら廃止検討
```

---

## 12. 演習問題

### 12.1 基礎演習: バージョニング方式の選定

以下のシナリオそれぞれに最適なバージョニング方式を選定し、その理由を述べよ。

**シナリオA**: 社内のマイクロサービス間で使用する内部API。サービスは全てKubernetes上で稼働し、Istioサービスメッシュで接続されている。変更頻度は月1〜2回。

**シナリオB**: 金融機関向けのオープンバンキングAPI。PSD2規制に準拠する必要があり、外部のフィンテック企業50社以上が利用する。契約上のSLAが厳格。

**シナリオC**: スタートアップの初期プロダクト。APIコンシューマーは自社のモバイルアプリのみ。2週間スプリントで頻繁にAPIが変更される。ユーザー数は1000人未満。

```
解答のヒント:

  考慮すべき観点:
  ├─ コンシューマーの数と種類（内部/外部）
  ├─ 変更頻度
  ├─ 規制要件の有無
  ├─ SLAの厳格さ
  ├─ 開発チームの規模とスキル
  ├─ 運用インフラの成熟度
  └─ 将来の拡張予定

  各シナリオの期待される解答方向:
  A: 内部APIのためバージョンレスまたは軽量なバージョニング
     Consumer-Driven Contracts + Tolerant Reader が有効
  B: 厳格なURIバージョニング + 長期並行運用
     SemVerによる内部管理 + CHANGELOGの公開
  C: バージョンレスまたはクエリパラメータ方式
     モバイルアプリのみなので柔軟に対応可能
     ただし将来の外部公開を見据えてURIバージョニングも検討
```

### 12.2 応用演習: 移行計画の策定

以下のv1 APIをv2に移行する計画を策定せよ。

```
v1 API仕様:
  POST /api/v1/products
  Request:
    {
      "name": "ノートPC",
      "price": 98000,         # integer（円単位）
      "category": "electronics",
      "tags": "laptop,portable" # カンマ区切り文字列
    }

  Response:
    {
      "id": 1,
      "name": "ノートPC",
      "price": 98000,
      "category": "electronics",
      "tags": "laptop,portable",
      "created": "2024-01-15"  # YYYY-MM-DD形式
    }

v2 API仕様（変更予定）:
  POST /api/v2/products
  Request:
    {
      "name": "ノートPC",
      "price": {               # オブジェクトに変更
        "amount": 98000,
        "currency": "JPY"
      },
      "category_id": "cat_electronics",  # IDベースに変更
      "tags": ["laptop", "portable"]     # 配列に変更
    }

  Response:
    {
      "id": "prod_1",         # string型、プレフィックス付き
      "name": "ノートPC",
      "price": {
        "amount": 98000,
        "currency": "JPY"
      },
      "category": {
        "id": "cat_electronics",
        "name": "Electronics"
      },
      "tags": ["laptop", "portable"],
      "created_at": "2024-01-15T00:00:00Z"  # ISO 8601形式
    }
```

**課題**:
1. 全ての破壊的変更をリストアップし、影響度を分析せよ
2. 12ヶ月の移行タイムラインを作成せよ
3. 移行ガイドの主要セクション（新旧マッピング表、コード例）を作成せよ
4. v1からv2への自動変換関数を設計せよ

```
解答のヒント:

  破壊的変更の一覧:
  ├─ price: integer → object (型変更)
  ├─ category: string → object (構造変更)
  ├─ category → category_id (リクエスト側のフィールド名変更)
  ├─ tags: string → array (型変更)
  ├─ id: integer → string (型変更 + プレフィックス追加)
  ├─ created → created_at (フィールド名変更)
  └─ created: YYYY-MM-DD → ISO 8601 (フォーマット変更)

  タイムライン例:
  月1-2:  v2の設計・レビュー・実装
  月3:    v2ベータ公開、内部テスト
  月4:    v2 GA公開、並行運用開始、移行ガイド公開
  月5-8:  v1に非推奨ヘッダー付与、主要クライアントへの個別通知
  月9-10: v1のレート制限を段階的に厳格化
  月11:   最終警告の送信
  月12:   v1の廃止（410 Gone）
```

### 12.3 発展演習: バージョニングフレームワークの設計

以下の要件を満たすバージョニングフレームワークを設計せよ（コードまたは擬似コードで記述）。

**要件**:
1. URI パスバージョニングとヘッダーバージョニングの両方をサポート
2. バージョンごとのリクエスト/レスポンス変換機能
3. 非推奨バージョンへのアクセスに自動でSunsetヘッダーを付与
4. バージョン別のアクセスメトリクスの収集
5. 未サポートバージョンへのリクエストに適切なエラーレスポンスを返す

```python
# 解答の骨格（発展させること）:

from abc import ABC, abstractmethod
from typing import Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class VersionConfig:
    """バージョンの設定情報"""
    version: str                         # "v1", "v2", "2024-01-15" 等
    status: str                          # "active", "deprecated", "sunset"
    release_date: datetime               # リリース日
    sunset_date: Optional[datetime]      # 廃止予定日（Noneなら未定）
    successor: Optional[str]             # 後継バージョン（Noneなら最新）
    transformers: dict = field(          # エンドポイント別の変換関数
        default_factory=dict
    )

class VersioningMiddleware:
    """バージョニングミドルウェアの基底クラス"""

    def __init__(self):
        self.versions: dict[str, VersionConfig] = {}
        self.metrics: dict[str, int] = {}

    def register_version(self, config: VersionConfig) -> None:
        """バージョンを登録する"""
        self.versions[config.version] = config

    def resolve_version(self, request) -> str:
        """リクエストからバージョンを解決する"""
        # 1. URIパスをチェック
        version = self._extract_from_path(request.path)
        if version:
            return version

        # 2. ヘッダーをチェック
        version = self._extract_from_header(request.headers)
        if version:
            return version

        # 3. デフォルトバージョンを返す
        return self._get_default_version()

    def process_request(self, request):
        """リクエストを処理し、適切なバージョンにルーティング"""
        version = self.resolve_version(request)

        # メトリクス収集
        self._record_metrics(version)

        # バージョンの状態チェック
        config = self.versions.get(version)
        if config is None:
            return self._unsupported_version_response(version)

        if config.status == 'sunset':
            return self._gone_response(version, config)

        # リクエスト変換（必要な場合）
        transformed_request = self._transform_request(
            request, version
        )

        return transformed_request, config

    def process_response(self, response, version: str):
        """レスポンスにバージョン関連ヘッダーを付与"""
        config = self.versions[version]

        # レスポンス変換
        transformed = self._transform_response(
            response, version
        )

        # 非推奨ヘッダーの付与
        if config.status == 'deprecated':
            transformed.headers['Deprecation'] = 'true'
            if config.sunset_date:
                transformed.headers['Sunset'] = (
                    config.sunset_date.strftime(
                        '%a, %d %b %Y %H:%M:%S GMT'
                    )
                )
            if config.successor:
                transformed.headers['Link'] = (
                    f'</{config.successor}/docs>; '
                    f'rel="successor-version"'
                )

        # バージョン情報ヘッダー
        transformed.headers['X-API-Version'] = version

        return transformed

    # --- 以下、各種ヘルパーメソッドを実装 ---
    # _extract_from_path, _extract_from_header,
    # _get_default_version, _record_metrics,
    # _unsupported_version_response, _gone_response,
    # _transform_request, _transform_response
```

**発展課題**:
- 上記の骨格コードを完全に実装せよ
- ユニットテストを作成し、全てのパターン（アクティブ/非推奨/廃止済みバージョン）をカバーせよ
- OpenAPI仕様からバージョン間の差分を自動検出する機能を追加せよ

---

## 13. 実践的なバージョニングポリシーテンプレート

組織やプロジェクトで使用できるバージョニングポリシーのテンプレートを以下に示す。

```
=================================================================
           [プロジェクト名] API バージョニングポリシー
                     Version 1.0 / 2025-01-01
=================================================================

1. バージョニング方式
   本APIはURIパスバージョニングを採用する。
   形式: /api/v{MAJOR}/
   例: /api/v1/, /api/v2/

2. バージョン番号の管理
   - 公開バージョン: メジャーバージョンのみ（v1, v2, v3...）
   - 内部バージョン: セマンティックバージョニング（MAJOR.MINOR.PATCH）
   - 内部バージョンはX-API-Versionヘッダーで通知

3. 破壊的変更の定義
   以下の変更を破壊的変更とする:
   a) レスポンスフィールドの削除
   b) フィールドの型変更
   c) 必須パラメータの追加
   d) エンドポイントのURL変更/削除
   e) ステータスコードの意味変更
   f) 認証/認可方式の変更
   g) フィールド名の変更

4. 非破壊的変更の定義
   以下の変更はバージョンアップなしで適用する:
   a) オプショナルフィールドの追加
   b) 新規エンドポイントの追加
   c) エラーメッセージの文言改善
   d) パフォーマンスの改善

5. ライフサイクルポリシー
   a) 新バージョンリリース後、旧バージョンは最低12ヶ月間
      並行運用する
   b) エンタープライズ契約のお客様には最低24ヶ月間の
      並行運用を保証する
   c) 並行運用期間はSLAに明記する

6. 非推奨化プロセス
   a) 新バージョンGA後、旧バージョンにDeprecationヘッダーを付与
   b) 廃止6ヶ月前: ドキュメント・メール・ダッシュボードで告知
   c) 廃止3ヶ月前: 主要クライアントへの個別通知
   d) 廃止1ヶ月前: 最終警告、レート制限の段階的厳格化
   e) 廃止日: 410 Gone を返す

7. 移行支援
   a) 移行ガイド（変更点一覧、新旧マッピング、コード例）を公開
   b) サンドボックス環境での事前検証を提供
   c) テクニカルサポートによる移行支援を提供

8. 同時運用バージョン数
   最大3バージョンを同時運用する（current, deprecated, sunset予告）。
   それ以上古いバージョンは廃止する。

9. 緊急時の例外
   セキュリティ脆弱性への対応など緊急性の高い場合は、
   上記プロセスを短縮して変更を適用する場合がある。
   その場合も可能な限り事前に通知する。

10. ポリシーの更新
    本ポリシーは年1回レビューし、必要に応じて更新する。
    ポリシーの変更自体も6ヶ月前に告知する。
=================================================================
```

---

## 14. FAQ（よくある質問）

### FAQ 1: URIバージョニングとヘッダーバージョニング、どちらを選ぶべきか？

**回答**: 大多数のケースではURIパスバージョニング（/api/v1/）を推奨する。理由は以下の通り。

- 最も直感的で理解しやすい。開発者がURLを見ただけでバージョンがわかる
- ブラウザやcurlでの手動テストが容易
- CDNやロードバランサーでのルーティング設定が単純
- ドキュメントでの説明がしやすい
- 業界で最も広く採用されており、学習コストが低い

ただし、以下のケースではヘッダーバージョニングを検討する価値がある。
- REST原則への厳密な準拠が求められる場合
- 同一リソースに対して複数の表現を提供する必要がある場合（コンテンツネゴシエーション）
- 日付ベースの細やかなバージョン管理が必要な場合（Stripe方式）

### FAQ 2: 最初のリリースは v0 と v1 のどちらにすべきか？

**回答**: v1 からスタートすることを推奨する。

v0 はSemVerにおいて「初期開発段階であり、いつでも破壊的変更が起こり得る」という意味を持つ。パブリックAPIでv0を使うと、「このAPIは不安定で信頼できない」というメッセージになりかねない。

ただし、以下の場合はv0を使うことも妥当である。
- 明確にプレビュー/ベータ版として提供する場合
- 内部APIで、安定性よりも柔軟性を優先する場合
- フィードバック収集が主目的で、本番利用を想定していない場合

### FAQ 3: バージョンアップ時に旧バージョンのバグ修正はどこまで行うべきか？

**回答**: セキュリティ修正は必ず行い、機能的なバグ修正はポリシーに応じて判断する。

推奨する対応レベル:

| 修正種別 | 旧バージョン対応 | 理由 |
|---------|----------------|------|
| セキュリティ脆弱性 | 必須 | 利用者の安全を守る義務がある |
| データ整合性バグ | 推奨 | データ破損はビジネスに直結する |
| 機能バグ（重大） | ケースバイケース | 影響度と移行スケジュールによる |
| 機能バグ（軽微） | 新バージョンでのみ修正 | 移行インセンティブにもなる |
| UX改善 | 新バージョンでのみ対応 | 旧バージョンへの投資を最小化 |

### FAQ 4: GraphQLにバージョニングは必要か？

**回答**: GraphQLは本質的にバージョンレスな設計を志向しているが、完全にバージョニング不要ということではない。

GraphQLが持つバージョンレスの仕組み:
- クライアントが必要なフィールドを選択するため、フィールド追加が非破壊的
- `@deprecated` ディレクティブによるフィールド単位の非推奨化
- スキーマの段階的進化が容易

ただし、以下の場合はバージョニングの検討が必要:
- スキーマの根本的な再設計が必要な場合
- 型の変更（String → Int等）が必要な場合
- クエリの構文やセマンティクスを変更する場合

### FAQ 5: APIバージョニングとマイクロサービスのバージョニングは同じか？

**回答**: 密接に関連するが、異なる関心事である。

- **APIバージョニング**: 外部に公開するインターフェースの契約管理。コンシューマーへの影響を制御する
- **サービスバージョニング**: 内部のデプロイメント単位の管理。Blue-Greenデプロイ、カナリアリリース等のデプロイ戦略と関連する

1つのサービスが複数のAPIバージョンを提供することもあれば、APIバージョンの変更なしにサービスが何度もデプロイされることもある。両者を混同しないことが重要である。

---

## 15. まとめ

| 概念 | ポイント |
|------|---------|
| URLバージョニング | パスに/v1/を含める方式、最もシンプルで広く採用 |
| ヘッダーバージョニング | Accept headerで指定、URLをクリーンに保つ |
| 破壊的変更管理 | 非推奨通知→移行期間→旧バージョン廃止のライフサイクル |
| 後方互換性 | フィールド追加はOK、削除・型変更は破壊的変更 |

### この章のキーポイント

1. **バージョニング戦略は早期に決定** — 後からの変更は困難
2. **破壊的変更は計画的に管理** — 非推奨通知と移行期間を設ける
3. **後方互換性を常に意識** — クライアントへの影響を最小化する

### 15.1 要点の整理

| 概念 | ポイント |
|------|---------|
| 破壊的変更 | フィールド削除、型変更、必須パラメータ追加、エンドポイント変更 |
| URIバージョニング | /api/v1/ が最も一般的で推奨。メジャーバージョンのみ公開 |
| ヘッダーバージョニング | REST原則に準拠。Stripe風の日付ベースに適する |
| クエリパラメータ | 実装が簡単だが長期運用では非推奨傾向 |
| セマンティックバージョニング | MAJOR.MINOR.PATCH で内部管理。公開はMAJORのみ |
| 非推奨化 | 6ヶ月前告知 → 12ヶ月並行運用 → 410 Gone で終了 |
| バージョンレス | Additive only + Feature Flags + Robustness Principle |
| Contract Testing | Consumer-Driven Contracts で互換性を自動検証 |
| APIゲートウェイ | バージョンルーティングとレスポンス変換の中核 |
| ポリシー | リリース時点で廃止計画を策定。ゾンビバージョンを防ぐ |

### 15.2 バージョニング方式の選定フローチャート

```
バージョニング方式の選定フロー:

  START
    │
    ├─ 外部公開APIか？
    │   ├─ Yes → コンシューマー数は？
    │   │        ├─ 多い（10社以上）→ URIバージョニング（推奨）
    │   │        ├─ 少ない（1〜9社）→ URIまたはヘッダー
    │   │        └─ 自社アプリのみ → 状況に応じて選択
    │   │
    │   └─ No（内部API）→ バージョンレスまたは軽量バージョニング
    │                      Consumer-Driven Contracts を検討
    │
    ├─ 変更頻度は？
    │   ├─ 高い → バージョンレス + Feature Flags
    │   ├─ 中程度 → URIバージョニング
    │   └─ 低い → URIバージョニング（最もシンプル）
    │
    ├─ 規制要件があるか？
    │   ├─ Yes → URIバージョニング + 厳格なポリシー
    │   └─ No → 柔軟に選択可能
    │
    └─ 既存システムとの互換性は？
        ├─ CDN/プロキシ経由 → URIバージョニング（ルーティング容易）
        ├─ APIゲートウェイ → いずれの方式も対応可能
        └─ 直接接続 → 制約なし
```

### 15.3 チェックリスト

APIバージョニング戦略を策定する際に確認すべき項目。

- [ ] バージョニング方式を決定したか（URI/ヘッダー/クエリパラメータ）
- [ ] 破壊的変更の定義をドキュメント化したか
- [ ] 非推奨化プロセスを策定したか（タイムライン、通知方法）
- [ ] 並行運用期間を決定したか（最低12ヶ月を推奨）
- [ ] CHANGELOGの運用ルールを定めたか
- [ ] バージョン互換性テストをCIに組み込んだか
- [ ] 利用状況モニタリングの仕組みを構築したか
- [ ] 移行ガイドのテンプレートを用意したか
- [ ] APIゲートウェイのバージョンルーティングを設定したか
- [ ] バージョニングポリシーを公開したか
- [ ] エンタープライズ向けの長期サポートを検討したか
- [ ] モバイルアプリ特有のバージョン管理を考慮したか

---

## まとめ

このガイドでは以下を学びました:

- APIバージョニングが必要となる理由と、破壊的変更の定義・分類方法
- URIバージョニング、ヘッダーバージョニング、クエリパラメータ方式の比較と適切な選定基準
- セマンティックバージョニングを活用した内部バージョン管理とCHANGELOG運用の実践
- 非推奨化プロセスの設計（告知→並行運用→廃止）とバージョンレス設計（Evolvable API）の考え方
- APIゲートウェイやマイクロサービス環境でのバージョンルーティングとContract Testingによる互換性自動検証

---

## 次に読むべきガイド
→ [ページネーションとフィルタリング](./03-pagination-and-filtering.md)

---

## 参考文献

1. Stripe. "API Versioning." stripe.com/docs/api/versioning, 2024. -- 日付ベースバージョニングの代表的実装。全APIコンシューマーが特定の日付バージョンに固定され、明示的にアップグレードしない限り旧挙動が維持される仕組みを詳細に解説している。

2. RFC 8594. "The Sunset HTTP Header Field." IETF, 2019. -- APIエンドポイントの廃止予定日をHTTPヘッダーで通知するための標準仕様。Deprecationヘッダーと組み合わせることで、プログラマティックな非推奨化通知を実現する。

3. Fielding, Roy Thomas. "Architectural Styles and the Design of Network-based Software Architectures." Doctoral dissertation, University of California, Irvine, 2000. -- RESTアーキテクチャスタイルの原典。コンテンツネゴシエーションの概念がヘッダーバージョニングの理論的基盤となっている。

4. Preston-Werner, Tom. "Semantic Versioning 2.0.0." semver.org, 2013. -- セマンティックバージョニングの仕様書。MAJOR.MINOR.PATCHの各番号が持つ意味と、バージョン番号の比較ルールを定義している。APIバージョニングの内部管理に広く適用される。

5. Pact Foundation. "Consumer-Driven Contract Testing." docs.pact.io, 2024. -- マイクロサービス間のAPI互換性を自動検証するためのContract Testingフレームワーク。コンシューマーが定義した契約をプロバイダーが満たすことを継続的に検証する手法を詳述している。

6. Google Cloud. "API Design Guide - Versioning." cloud.google.com/apis/design, 2024. -- Google CloudのAPI設計ガイドにおけるバージョニングセクション。メジャーバージョンのURIパス組み込み、マイナーバージョンの内部管理、互換性維持のルールをGoogleの大規模APIエコシステムの視点から解説している。

