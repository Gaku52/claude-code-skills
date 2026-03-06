# バージョニング戦略

> APIのバージョニングは後方互換性と進化のバランスを追求する技術的判断の集合体である。破壊的変更の定義、URI/ヘッダー/クエリパラメータベースの戦略、セマンティックバージョニング、非推奨化プロセスを深く理解し、長期運用に耐えるAPIを設計する。

## この章で学ぶこと

- [ ] 破壊的変更と非破壊的変更を明確に区別できるようになる
- [ ] 3つのバージョニング方式（URI、ヘッダー、クエリパラメータ）の特性と選択基準を把握する
- [ ] セマンティックバージョニングの原則をAPIに適用する方法を理解する
- [ ] 非推奨化と移行のプロセスを設計・実行できるようになる
- [ ] バージョンレス設計の思想と実践手法を習得する
- [ ] 破壊的変更の影響を最小化するための戦略的アプローチを身につける

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

