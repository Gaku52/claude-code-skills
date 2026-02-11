# セキュリティ原則

> 最小権限、多層防御、ゼロトラスト、セキュアバイデフォルトなど、堅牢なシステム設計の土台となるセキュリティ原則を解説する。

## この章で学ぶこと

1. **最小権限の原則**を適用して攻撃対象面を最小化する方法を理解する
2. **多層防御とゼロトラスト**の設計思想を実装に落とし込む手法を習得する
3. **セキュアバイデフォルト**の考え方でシステムを安全に初期化する技術を身につける

---

## 1. 最小権限の原則（Principle of Least Privilege）

ユーザー、プロセス、システムに対して、その業務を遂行するために必要最小限の権限のみを付与する原則。

```
最小権限の適用イメージ:

  過剰な権限付与:                   最小権限の適用:
  +-------------------+             +-------------------+
  | Admin権限         |             | read:products     |
  | - 全DB読み書き    |             | write:cart        |
  | - ユーザー管理    |    ==>      | read:own_orders   |
  | - サーバー設定    |             |                   |
  | - ログ閲覧       |             | (ECサイトの一般    |
  +-------------------+             |  ユーザーに必要な  |
  (全権限を付与)                    |  権限のみ)        |
                                    +-------------------+
```

```python
# コード例1: 最小権限を実現するRBAC
from enum import Enum, auto
from typing import Set, Dict
from functools import wraps

class Permission(Enum):
    READ_PRODUCTS = auto()
    WRITE_PRODUCTS = auto()
    READ_ORDERS = auto()
    WRITE_ORDERS = auto()
    MANAGE_USERS = auto()
    VIEW_ANALYTICS = auto()
    ADMIN_SETTINGS = auto()

class Role(Enum):
    VIEWER = auto()
    EDITOR = auto()
    ORDER_MANAGER = auto()
    ADMIN = auto()

# 各ロールに最小限の権限を割り当てる
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.VIEWER: {
        Permission.READ_PRODUCTS,
    },
    Role.EDITOR: {
        Permission.READ_PRODUCTS,
        Permission.WRITE_PRODUCTS,
    },
    Role.ORDER_MANAGER: {
        Permission.READ_PRODUCTS,
        Permission.READ_ORDERS,
        Permission.WRITE_ORDERS,
    },
    Role.ADMIN: {
        Permission.READ_PRODUCTS,
        Permission.WRITE_PRODUCTS,
        Permission.READ_ORDERS,
        Permission.WRITE_ORDERS,
        Permission.MANAGE_USERS,
        Permission.VIEW_ANALYTICS,
        Permission.ADMIN_SETTINGS,
    },
}

def require_permission(permission: Permission):
    """権限チェックデコレータ"""
    def decorator(func):
        @wraps(func)
        def wrapper(user, *args, **kwargs):
            user_permissions = ROLE_PERMISSIONS.get(user.role, set())
            if permission not in user_permissions:
                raise PermissionError(
                    f"権限不足: {user.role.name} には "
                    f"{permission.name} がありません"
                )
            return func(user, *args, **kwargs)
        return wrapper
    return decorator

@require_permission(Permission.WRITE_PRODUCTS)
def update_product(user, product_id: int, data: dict):
    """商品情報を更新する（EDITOR以上の権限が必要）"""
    # 商品更新ロジック
    pass
```

### IAMポリシーでの最小権限

```python
# コード例2: AWS IAMポリシーの最小権限設計
import json

def create_minimal_iam_policy(bucket_name: str, prefix: str) -> str:
    """特定のS3バケット・プレフィックスにのみアクセスできるポリシー"""
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AllowListBucketWithPrefix",
                "Effect": "Allow",
                "Action": ["s3:ListBucket"],
                "Resource": f"arn:aws:s3:::{bucket_name}",
                "Condition": {
                    "StringLike": {
                        "s3:prefix": [f"{prefix}/*"]
                    }
                }
            },
            {
                "Sid": "AllowReadWriteWithPrefix",
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    # DeleteObject は意図的に含めない
                ],
                "Resource": f"arn:aws:s3:::{bucket_name}/{prefix}/*"
            },
            {
                # 明示的な拒否で安全ネットを張る
                "Sid": "DenyDeleteActions",
                "Effect": "Deny",
                "Action": [
                    "s3:DeleteObject",
                    "s3:DeleteBucket",
                ],
                "Resource": "*"
            }
        ]
    }
    return json.dumps(policy, indent=2)

print(create_minimal_iam_policy("my-app-data", "uploads/user-123"))
```

---

## 2. 多層防御（Defense in Depth）

単一の防御機構に依存せず、複数の独立した防御層を重ねる戦略。

```
多層防御モデル:

  攻撃者 -->  [WAF]  -->  [ネットワーク]  -->  [ホスト]  -->  [アプリ]  -->  [データ]
               |            FW/IDS             OS強化        入力検証       暗号化
               |            セグメント         パッチ管理     認証/認可      アクセス制御
               v            VPN               アンチウイルス  セッション管理  DLP
           ブロック
           or
           通過

  各層が独立して動作 → 1つの層が突破されても次の層が防御
```

```python
# コード例3: 多層防御の実装パターン
from typing import List, Callable, Optional
from dataclasses import dataclass
import re

@dataclass
class SecurityCheckResult:
    passed: bool
    layer: str
    message: str

class DefenseInDepth:
    """多層防御チェーンの実装"""

    def __init__(self):
        self.layers: List[tuple] = []  # (layer_name, check_func)

    def add_layer(self, name: str, check: Callable) -> 'DefenseInDepth':
        self.layers.append((name, check))
        return self  # メソッドチェーン対応

    def validate(self, request: dict) -> List[SecurityCheckResult]:
        """全防御層を順に通過させる"""
        results = []
        for layer_name, check_func in self.layers:
            try:
                passed, msg = check_func(request)
                results.append(SecurityCheckResult(passed, layer_name, msg))
                if not passed:
                    # 失敗してもログは記録し続ける（全層の状態を把握）
                    break
            except Exception as e:
                results.append(SecurityCheckResult(
                    False, layer_name, f"Error: {e}"
                ))
                break
        return results

# 各防御層の定義
def rate_limit_check(request: dict) -> tuple:
    """Layer 1: レートリミット"""
    ip = request.get("ip", "unknown")
    # 実際にはRedis等でカウントを管理
    count = get_request_count(ip)
    if count > 100:
        return False, f"Rate limit exceeded for {ip}"
    return True, "Rate limit OK"

def waf_check(request: dict) -> tuple:
    """Layer 2: WAF（SQLインジェクションパターン検出）"""
    body = str(request.get("body", ""))
    dangerous_patterns = [
        r"('|\")\s*(OR|AND)\s+.*=",
        r";\s*(DROP|DELETE|UPDATE|INSERT)",
        r"UNION\s+SELECT",
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, body, re.IGNORECASE):
            return False, f"WAF blocked: suspicious pattern detected"
    return True, "WAF check passed"

def input_validation(request: dict) -> tuple:
    """Layer 3: 入力値検証"""
    body = request.get("body", {})
    if isinstance(body, dict):
        for key, value in body.items():
            if isinstance(value, str) and len(value) > 10000:
                return False, f"Input too long: {key}"
    return True, "Input validation passed"

# 防御チェーンの構築
defense = (DefenseInDepth()
    .add_layer("RateLimit", rate_limit_check)
    .add_layer("WAF", waf_check)
    .add_layer("InputValidation", input_validation))
```

---

## 3. ゼロトラスト（Zero Trust）

「決して信頼せず、常に検証する」という原則。ネットワーク境界の内外を問わず、すべてのアクセスを検証する。

```
従来型 (境界防御):            ゼロトラスト:

+---外部---+---内部---+       +--+  +--+  +--+  +--+
|          | FW |     |       |検| ->|検| ->|検| ->|検|
| 攻撃者   | -- | 自由|       |証|  |証|  |証|  |証|
|          |    | 移動|       +--+  +--+  +--+  +--+
+----------+----+-----+       毎回  毎回  毎回  毎回
  FWを突破すると内部は無防備     すべてのアクセスを検証
```

### ゼロトラストの5つの柱

| 柱 | 説明 | 実装例 |
|----|------|--------|
| ID検証 | すべてのユーザー・デバイスを認証 | MFA、証明書認証 |
| デバイス検証 | デバイスの正当性とセキュリティ状態を検証 | MDM、デバイス証明書 |
| ネットワーク | マイクロセグメンテーションによる通信制御 | サービスメッシュ、mTLS |
| アプリケーション | アプリレベルでのアクセス制御 | OAuth2、RBAC |
| データ | データの分類と暗号化 | 暗号化、DLP、ラベリング |

```python
# コード例4: ゼロトラストリクエスト検証
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import jwt

@dataclass
class ZeroTrustContext:
    """ゼロトラスト検証に必要なコンテキスト"""
    user_id: str
    device_id: str
    device_trust_level: str    # managed/unmanaged/unknown
    network_location: str      # corporate/vpn/public
    mfa_verified: bool
    last_auth_time: datetime
    client_cert_valid: bool
    risk_score: float          # 0.0 - 1.0

class ZeroTrustPolicyEngine:
    """ゼロトラストポリシーエンジン"""

    MAX_SESSION_AGE = timedelta(hours=1)
    MAX_RISK_SCORE = 0.7

    def evaluate(self, ctx: ZeroTrustContext,
                 resource: str, action: str) -> dict:
        """アクセスリクエストを全方位的に評価する"""
        checks = {
            "identity_verified": self._check_identity(ctx),
            "device_compliant": self._check_device(ctx),
            "network_allowed": self._check_network(ctx, resource),
            "mfa_satisfied": self._check_mfa(ctx, resource),
            "session_fresh": self._check_session_age(ctx),
            "risk_acceptable": self._check_risk(ctx),
        }

        all_passed = all(checks.values())
        return {
            "allowed": all_passed,
            "checks": checks,
            "action_required": self._get_remediation(checks) if not all_passed else None,
        }

    def _check_identity(self, ctx: ZeroTrustContext) -> bool:
        return ctx.user_id is not None and ctx.user_id != ""

    def _check_device(self, ctx: ZeroTrustContext) -> bool:
        return ctx.device_trust_level in ("managed",)

    def _check_network(self, ctx: ZeroTrustContext, resource: str) -> bool:
        # 機密リソースはVPN/社内ネットワークからのみ
        sensitive_resources = {"admin_panel", "financial_data"}
        if resource in sensitive_resources:
            return ctx.network_location in ("corporate", "vpn")
        return True

    def _check_mfa(self, ctx: ZeroTrustContext, resource: str) -> bool:
        sensitive_resources = {"admin_panel", "financial_data", "user_data"}
        if resource in sensitive_resources:
            return ctx.mfa_verified
        return True

    def _check_session_age(self, ctx: ZeroTrustContext) -> bool:
        return (datetime.now() - ctx.last_auth_time) < self.MAX_SESSION_AGE

    def _check_risk(self, ctx: ZeroTrustContext) -> bool:
        return ctx.risk_score < self.MAX_RISK_SCORE

    def _get_remediation(self, checks: dict) -> str:
        if not checks["mfa_satisfied"]:
            return "MFA認証が必要です"
        if not checks["device_compliant"]:
            return "管理対象デバイスからアクセスしてください"
        if not checks["session_fresh"]:
            return "セッションが期限切れです。再認証してください"
        return "アクセスが拒否されました。管理者に連絡してください"
```

---

## 4. セキュアバイデフォルト（Secure by Default）

システムの初期状態を安全な構成にする原則。ユーザーが明示的に緩和しない限り、最も安全な設定が適用される。

```python
# コード例5: セキュアバイデフォルトの設定クラス
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class SecureServerConfig:
    """セキュアバイデフォルトなサーバー設定"""

    # --- デフォルトで安全な設定 ---
    # TLS設定
    tls_min_version: str = "TLSv1.3"
    tls_ciphers: List[str] = field(default_factory=lambda: [
        "TLS_AES_256_GCM_SHA384",
        "TLS_CHACHA20_POLY1305_SHA256",
    ])
    hsts_enabled: bool = True
    hsts_max_age: int = 31536000  # 1年

    # セキュリティヘッダー（デフォルトで有効）
    content_security_policy: str = "default-src 'self'"
    x_frame_options: str = "DENY"
    x_content_type_options: str = "nosniff"
    referrer_policy: str = "strict-origin-when-cross-origin"

    # Cookie設定（デフォルトで安全）
    cookie_secure: bool = True
    cookie_httponly: bool = True
    cookie_samesite: str = "Strict"

    # CORS（デフォルトで制限）
    cors_allowed_origins: List[str] = field(default_factory=list)
    cors_allow_credentials: bool = False

    # レートリミット（デフォルトで有効）
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60

    # ログ（デフォルトで有効）
    access_log_enabled: bool = True
    error_log_enabled: bool = True
    log_sensitive_data: bool = False  # 機密データのログは無効

    # デバッグ（デフォルトで無効）
    debug_mode: bool = False
    expose_stack_traces: bool = False
    expose_server_version: bool = False

    def to_headers(self) -> dict:
        """HTTPレスポンスヘッダーを生成する"""
        headers = {
            "X-Content-Type-Options": self.x_content_type_options,
            "X-Frame-Options": self.x_frame_options,
            "Referrer-Policy": self.referrer_policy,
            "Content-Security-Policy": self.content_security_policy,
        }
        if self.hsts_enabled:
            headers["Strict-Transport-Security"] = (
                f"max-age={self.hsts_max_age}; includeSubDomains; preload"
            )
        return headers

# デフォルト設定は安全
config = SecureServerConfig()
print(config.debug_mode)         # False（安全）
print(config.cookie_secure)      # True（安全）
print(config.tls_min_version)    # TLSv1.3（安全）
```

---

## 5. その他の重要なセキュリティ原則

### 原則一覧

| 原則 | 説明 | 例 |
|------|------|-----|
| フェイルセーフ | 障害時は安全な状態に遷移する | アクセス判定失敗時はデフォルト拒否 |
| 完全仲介 | すべてのアクセスを検証する | ミドルウェアでの一貫した認可チェック |
| 経済性 | 設計はシンプルに保つ | 複雑な権限モデルより単純なRBAC |
| オープン設計 | セキュリティは秘密に依存しない | 暗号アルゴリズムの公開性 |
| 権限分離 | 重要操作には複数の承認を要求 | 4-eyes principle、MFA |
| 攻撃対象面の最小化 | 不要な機能・ポートを無効化 | 未使用APIエンドポイントの削除 |

```
フェイルセーフのフロー:

  リクエスト --> [認可チェック]
                    |
              +-----+-----+
              |           |
           成功        失敗/エラー
              |           |
         アクセス許可   アクセス拒否（安全側に倒す）
                         + ログ記録
                         + アラート送信
```

---

## アンチパターン

### アンチパターン1: 過剰な権限付与（God Mode）

「面倒だから」「動かないから」という理由で管理者権限やワイルドカード権限を付与するパターン。特にIAMで `*` リソースに `*` アクションを許可するのは最も危険な設定である。

```json
{
  "Effect": "Allow",
  "Action": "*",
  "Resource": "*"
}
```

この設定は絶対に避け、具体的なアクション・リソースを指定すること。

### アンチパターン2: セキュリティ by オブスキュリティ

隠蔽のみに依存するセキュリティ。「管理画面のURLを推測しにくくすれば大丈夫」「ソースコードは非公開だから安全」という考えは危険である。URLの推測困難性は補助的な対策にはなるが、適切な認証・認可が前提である。

---

## FAQ

### Q1: 最小権限を徹底すると開発効率が下がりませんか?

初期設定コストは上がるが、インシデント対応コストを考えると長期的には効率的である。また、IaCで権限テンプレートを管理し、開発環境ではやや緩い権限を使いつつ、本番環境で厳格に適用するアプローチが現実的である。

### Q2: ゼロトラストは社内ネットワークでも必要ですか?

必要である。近年のセキュリティインシデントの多くは内部ネットワークからの横展開（Lateral Movement）によるものであり、VPN接続後に社内リソースへ自由にアクセスできる状態は危険である。マイクロセグメンテーションとmTLSの導入が推奨される。

### Q3: セキュアバイデフォルトとユーザビリティのバランスはどう取りますか?

安全な初期設定を提供しつつ、必要に応じてユーザーが明示的に緩和できる設計にする。例えば、CSPのデフォルトは厳格に設定し、特定のCDNが必要な場合にのみホワイトリストに追加する。「opt-out」モデルを採用し、緩和時には警告を表示する。

---

## まとめ

| 原則 | 核心 | 実装のポイント |
|------|------|---------------|
| 最小権限 | 必要最小限の権限のみ付与 | RBAC、IAMポリシーの精密化 |
| 多層防御 | 複数の独立した防御層を重ねる | WAF+FW+入力検証+暗号化 |
| ゼロトラスト | すべてのアクセスを検証する | mTLS、マイクロセグメンテーション |
| セキュアバイデフォルト | 初期状態を安全に | 安全なデフォルト値、opt-outモデル |
| フェイルセーフ | 障害時は安全な状態に | デフォルト拒否、エラー時のアクセス遮断 |

---

## 次に読むべきガイド

- [../01-web-security/00-owasp-top10.md](../01-web-security/00-owasp-top10.md) -- セキュリティ原則を具体的脆弱性対策に適用
- [../04-application-security/00-secure-coding.md](../04-application-security/00-secure-coding.md) -- コーディングレベルでの原則適用
- [../05-cloud-security/00-cloud-security-basics.md](../05-cloud-security/00-cloud-security-basics.md) -- クラウド環境での原則適用

---

## 参考文献

1. Saltzer, J.H. & Schroeder, M.D., "The Protection of Information in Computer Systems" -- Proceedings of the IEEE, 1975
2. NIST SP 800-207, "Zero Trust Architecture" -- https://csrc.nist.gov/publications/detail/sp/800-207/final
3. Google BeyondCorp -- https://cloud.google.com/beyondcorp
4. OWASP Security Design Principles -- https://owasp.org/www-project-developer-guide/
