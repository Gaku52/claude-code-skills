# セキュリティ原則

> 最小権限、多層防御、ゼロトラスト、セキュアバイデフォルトなど、堅牢なシステム設計の土台となるセキュリティ原則を解説する。これらの原則は1975年にSaltzerとSchroederが発表した「The Protection of Information in Computer Systems」に端を発し、50年経った現在でもすべてのセキュアシステム設計の基盤である。

## 前提知識

- ネットワークの基礎（TCP/IP、HTTP/HTTPS）
- 基本的な認証・認可の概念
- [暗号化の基礎](./01-cryptography-basics.md) の知識があると理解が深まる

## この章で学ぶこと

1. **最小権限の原則**を適用して攻撃対象面を最小化する方法を理解する
2. **多層防御とゼロトラスト**の設計思想を実装に落とし込む手法を習得する
3. **セキュアバイデフォルト**の考え方でシステムを安全に初期化する技術を身につける
4. **Saltzer & Schroederの8原則**の全体像と相互関係を把握する
5. 各原則の**実装パターンとアンチパターン**を理解し、実プロジェクトに適用できるようになる

---

## 1. Saltzer & Schroederの8原則 — セキュリティ設計の起源

1975年にJerome SaltzerとMichael Schroederが発表したセキュリティ設計原則は、現代のあらゆるセキュリティフレームワークの基盤である。まずこの8原則の全体像を理解する。

```
Saltzer & Schroeder の 8 原則（1975年）:

+================================================================+
|  原則                        | 現代における適用                    |
|================================================================|
|  1. Economy of Mechanism     | KISS原則、シンプルな設計           |
|     (メカニズムの経済性)      |                                   |
|  2. Fail-Safe Defaults       | デフォルト拒否、ホワイトリスト     |
|     (フェイルセーフデフォルト) |                                   |
|  3. Complete Mediation        | ゼロトラスト、毎回検証            |
|     (完全仲介)                |                                   |
|  4. Open Design              | 公開暗号アルゴリズム              |
|     (オープン設計)            |                                   |
|  5. Separation of Privilege   | MFA、4-eyes principle            |
|     (権限分離)                |                                   |
|  6. Least Privilege           | RBAC、IAM最小権限                |
|     (最小権限)                |                                   |
|  7. Least Common Mechanism    | プロセス分離、サンドボックス      |
|     (最少共有メカニズム)      |                                   |
|  8. Psychological Acceptability| UX設計、SSO                      |
|     (心理的受容性)            |                                   |
+================================================================+

これらの原則は独立ではなく相互に関連する:

  最小権限 ──→ 権限分離 ──→ 完全仲介
      │                         │
      ▼                         ▼
  フェイルセーフ ←── メカニズムの経済性
      │
      ▼
  心理的受容性 ←── オープン設計
```

### なぜ50年前の原則が今も有効なのか

セキュリティの脅威は変化するが、防御の「構造」は変わらない。これらの原則は特定の技術ではなく「設計哲学」であるため、クラウドネイティブ、マイクロサービス、ゼロトラストといった現代のアーキテクチャにもそのまま適用できる。NISTのSP 800-160（Systems Security Engineering）やOWASPのSecurity Design Principlesも、この8原則を現代的に再解釈したものである。

---

## 2. 最小権限の原則（Principle of Least Privilege）

ユーザー、プロセス、システムに対して、その業務を遂行するために必要最小限の権限のみを付与する原則。この原則は攻撃対象面（Attack Surface）を最小化し、侵害が発生した場合の被害（Blast Radius）を限定する。

### なぜ最小権限が重要か — 内部メカニズム

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

  攻撃対象面（Attack Surface）の比較:

  過剰な権限:                         最小権限:
  +---------------------------------+  +--------+
  |                                 |  |        |
  |   攻撃者が侵害した場合          |  | 限定的 |
  |   全システムが危険に             |  | な被害 |
  |                                 |  |        |
  +---------------------------------+  +--------+
  Blast Radius = 全体                  Blast Radius = 最小
```

最小権限が重要な理由は3つある。

1. **被害の限定（Blast Radius Reduction）**: 侵害が発生しても、そのアカウントが持つ権限の範囲内でしか被害が広がらない
2. **横展開の防止（Lateral Movement Prevention）**: 攻撃者が1つのアカウントを奪取しても、他のリソースへのアクセスが制限される
3. **監査の簡素化（Audit Simplification）**: 各アカウントの権限が明確であるため、異常なアクセスパターンを検出しやすい

### コード例1: RBACによる最小権限の実装

```python
# コード例1: 最小権限を実現するRBAC（Role-Based Access Control）
from enum import Enum, auto
from typing import Set, Dict, Optional, List
from functools import wraps
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class Permission(Enum):
    READ_PRODUCTS = auto()
    WRITE_PRODUCTS = auto()
    READ_ORDERS = auto()
    WRITE_ORDERS = auto()
    MANAGE_USERS = auto()
    VIEW_ANALYTICS = auto()
    ADMIN_SETTINGS = auto()
    DELETE_DATA = auto()
    EXPORT_DATA = auto()

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
        # 注: DELETE_DATA と EXPORT_DATA は ADMIN にも
        # デフォルトでは付与しない（個別承認が必要）
    },
}

class User:
    def __init__(self, user_id: str, role: Role,
                 temporary_permissions: Optional[Set[Permission]] = None,
                 temporary_expiry: Optional[datetime] = None):
        self.user_id = user_id
        self.role = role
        self.temporary_permissions = temporary_permissions or set()
        self.temporary_expiry = temporary_expiry

    def get_effective_permissions(self) -> Set[Permission]:
        """有効な権限を計算する（ロール権限 + 一時権限）"""
        perms = ROLE_PERMISSIONS.get(self.role, set()).copy()
        # 一時権限が期限内なら追加
        if (self.temporary_permissions and self.temporary_expiry
                and datetime.now() < self.temporary_expiry):
            perms |= self.temporary_permissions
        return perms

def require_permission(permission: Permission):
    """権限チェックデコレータ — 完全仲介の原則も体現"""
    def decorator(func):
        @wraps(func)
        def wrapper(user: User, *args, **kwargs):
            effective_perms = user.get_effective_permissions()
            if permission not in effective_perms:
                logger.warning(
                    "権限不足: user=%s role=%s required=%s",
                    user.user_id, user.role.name, permission.name
                )
                raise PermissionError(
                    f"権限不足: {user.role.name} には "
                    f"{permission.name} がありません"
                )
            logger.info(
                "アクセス許可: user=%s action=%s",
                user.user_id, permission.name
            )
            return func(user, *args, **kwargs)
        return wrapper
    return decorator

@require_permission(Permission.WRITE_PRODUCTS)
def update_product(user: User, product_id: int, data: dict):
    """商品情報を更新する（EDITOR以上の権限が必要）"""
    # 商品更新ロジック
    pass

@require_permission(Permission.DELETE_DATA)
def delete_user_data(user: User, target_user_id: str):
    """ユーザーデータを削除する（個別承認された一時権限が必要）"""
    # データ削除ロジック — ADMINでもデフォルトでは実行不可
    pass

# 使用例: 一時的な権限昇格（Just-In-Time Access）
admin = User("admin-001", Role.ADMIN)
# GDPR削除リクエスト対応のため、一時的にDELETE_DATA権限を付与
admin.temporary_permissions = {Permission.DELETE_DATA}
admin.temporary_expiry = datetime.now() + timedelta(hours=1)
# 1時間後に自動的に権限が無効化される
```

### コード例2: AWS IAMポリシーでの最小権限設計

```python
# コード例2: AWS IAMポリシーの最小権限設計
import json
from typing import List, Optional

def create_minimal_iam_policy(
    bucket_name: str,
    prefix: str,
    allow_delete: bool = False,
    condition_ip_range: Optional[str] = None
) -> str:
    """
    特定のS3バケット・プレフィックスにのみアクセスできるポリシーを生成。

    設計方針:
    1. Resource は具体的に指定（ワイルドカード禁止）
    2. Action は必要最小限（Get/Put のみ、Delete はオプション）
    3. Condition で追加の制約を付与（IPレンジ、MFA等）
    4. 明示的 Deny で安全ネットを張る
    """
    statements = [
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
            ],
            "Resource": f"arn:aws:s3:::{bucket_name}/{prefix}/*"
        },
    ]

    # 削除権限は明示的に要求された場合のみ付与
    if allow_delete:
        statements.append({
            "Sid": "AllowDeleteWithPrefix",
            "Effect": "Allow",
            "Action": ["s3:DeleteObject"],
            "Resource": f"arn:aws:s3:::{bucket_name}/{prefix}/*",
            "Condition": {
                "Bool": {"aws:MultiFactorAuthPresent": "true"}
            }
        })

    # IPレンジ制約の追加
    if condition_ip_range:
        for stmt in statements:
            if stmt["Effect"] == "Allow":
                stmt.setdefault("Condition", {})
                stmt["Condition"]["IpAddress"] = {
                    "aws:SourceIp": condition_ip_range
                }

    # 明示的な拒否で安全ネットを張る
    statements.append({
        "Sid": "DenyDangerousActions",
        "Effect": "Deny",
        "Action": [
            "s3:DeleteBucket",
            "s3:PutBucketPolicy",
            "s3:PutBucketAcl",
            "s3:PutObjectAcl",
        ],
        "Resource": "*"
    })

    policy = {
        "Version": "2012-10-17",
        "Statement": statements
    }
    return json.dumps(policy, indent=2)

# 使用例
print(create_minimal_iam_policy(
    "my-app-data",
    "uploads/user-123",
    allow_delete=False,
    condition_ip_range="10.0.0.0/8"
))
```

### Just-In-Time（JIT）アクセスの実装

最小権限の実践において重要なのが、必要な時にのみ権限を付与し、不要になったら即座に剥奪する「Just-In-Time Access」のアプローチである。

```python
# コード例3: JIT（Just-In-Time）アクセスマネージャー
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable
import threading
import logging

logger = logging.getLogger(__name__)

@dataclass
class AccessGrant:
    """一時的なアクセス許可"""
    user_id: str
    permission: str
    resource: str
    granted_at: datetime
    expires_at: datetime
    reason: str
    approved_by: str
    revoked: bool = False

class JITAccessManager:
    """Just-In-Time アクセス管理 — 最小権限の動的実装"""

    MAX_GRANT_DURATION = timedelta(hours=8)

    def __init__(self):
        self._grants: Dict[str, AccessGrant] = {}
        self._cleanup_timer: Optional[threading.Timer] = None
        self._start_cleanup_loop()

    def request_access(
        self,
        user_id: str,
        permission: str,
        resource: str,
        duration: timedelta,
        reason: str,
        approved_by: str,
    ) -> AccessGrant:
        """一時的なアクセスを要求する"""
        if duration > self.MAX_GRANT_DURATION:
            raise ValueError(
                f"最大許可時間は {self.MAX_GRANT_DURATION} です。"
                "長期アクセスが必要な場合はロール変更を申請してください。"
            )

        now = datetime.now()
        grant = AccessGrant(
            user_id=user_id,
            permission=permission,
            resource=resource,
            granted_at=now,
            expires_at=now + duration,
            reason=reason,
            approved_by=approved_by,
        )

        grant_id = f"{user_id}:{permission}:{resource}"
        self._grants[grant_id] = grant

        logger.info(
            "JIT access granted: user=%s perm=%s resource=%s "
            "expires=%s reason=%s approved_by=%s",
            user_id, permission, resource,
            grant.expires_at.isoformat(), reason, approved_by,
        )

        return grant

    def check_access(self, user_id: str, permission: str,
                     resource: str) -> bool:
        """アクセス権限を確認する"""
        grant_id = f"{user_id}:{permission}:{resource}"
        grant = self._grants.get(grant_id)

        if grant is None:
            return False
        if grant.revoked:
            return False
        if datetime.now() > grant.expires_at:
            grant.revoked = True
            logger.info("JIT access expired: user=%s perm=%s",
                       user_id, permission)
            return False

        return True

    def revoke_access(self, user_id: str, permission: str,
                      resource: str, reason: str = "manual_revoke"):
        """アクセスを即座に取り消す"""
        grant_id = f"{user_id}:{permission}:{resource}"
        grant = self._grants.get(grant_id)
        if grant and not grant.revoked:
            grant.revoked = True
            logger.info(
                "JIT access revoked: user=%s perm=%s reason=%s",
                user_id, permission, reason,
            )

    def _cleanup_expired(self):
        """期限切れのグラントをクリーンアップする"""
        now = datetime.now()
        expired = [
            gid for gid, g in self._grants.items()
            if now > g.expires_at or g.revoked
        ]
        for gid in expired:
            del self._grants[gid]

    def _start_cleanup_loop(self):
        """定期的なクリーンアップループ"""
        self._cleanup_expired()
        self._cleanup_timer = threading.Timer(60.0, self._start_cleanup_loop)
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()
```

### Kubernetes における最小権限

```yaml
# コード例4: Kubernetes RBAC — 最小権限の適用
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: production
  name: app-deployer
rules:
  # Deployment の読み取りと更新のみ許可
  - apiGroups: ["apps"]
    resources: ["deployments"]
    verbs: ["get", "list", "watch", "update", "patch"]
    # delete は含めない（誤削除防止）
  # Pod のログ閲覧のみ許可
  - apiGroups: [""]
    resources: ["pods", "pods/log"]
    verbs: ["get", "list"]
    # exec は含めない（本番コンテナ内操作を防止）
  # ConfigMap の読み取りのみ
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get", "list"]
    # Secrets へのアクセスは別途承認が必要
---
# NG: 過剰な権限付与（クラスタ管理者権限）
# apiVersion: rbac.authorization.k8s.io/v1
# kind: ClusterRoleBinding
# metadata:
#   name: dev-cluster-admin
# roleRef:
#   apiGroup: rbac.authorization.k8s.io
#   kind: ClusterRole
#   name: cluster-admin  # ← 絶対に開発者に付与しない
```

---

## 3. 多層防御（Defense in Depth）

単一の防御機構に依存せず、複数の独立した防御層を重ねる戦略。軍事用語の「縦深防御」に由来し、中世の城の設計（堀→城壁→内壁→天守閣）と同じ発想である。

### なぜ多層防御が必要か

セキュリティにおいて「完璧な防御層」は存在しない。どんなに優れたWAFにもバイパス手法が存在し、どんなに堅牢なファイアウォールにも設定ミスの可能性がある。多層防御の核心は「1つの層が破られても、次の層が機能する」という冗長性にある。

```
多層防御モデル — 城の比喩:

  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │  [堀]      [城壁]     [内壁]     [天守閣]   [金庫]  │
  │   │         │          │          │          │      │
  │   ▼         ▼          ▼          ▼          ▼      │
  │  WAF/     FW/IDS    ホスト強化  アプリ層   データ層  │
  │  CDN      ネットワーク OS/パッチ  認証/認可  暗号化  │
  │  DDoS防御 セグメント  アンチウイルス 入力検証 DLP    │
  │  Bot検知  VPN        HIDS       セッション  バックアップ│
  │           mTLS       CIS        CSRF防止   監査ログ │
  │                      Benchmark                      │
  │                                                     │
  │  攻撃者 ──→ 層1突破 ──→ 層2で検知 ──→ 層3で遮断     │
  │                                                     │
  │  各層が独立して動作 → 1つの層が突破されても          │
  │  次の層が防御 + 検知 + アラート送信                   │
  └─────────────────────────────────────────────────────┘

具体的な防御層の構成:

  Layer 1: ペリメター防御
    +-- WAF (OWASP Core Rule Set)
    +-- DDoS Mitigation (CloudFlare, AWS Shield)
    +-- Bot Detection
    +-- GeoIP Blocking

  Layer 2: ネットワーク防御
    +-- Firewall (Security Groups, NACLs)
    +-- IDS/IPS (Suricata, AWS GuardDuty)
    +-- Network Segmentation (VPC, Subnet)
    +-- mTLS (サービス間通信)

  Layer 3: ホスト防御
    +-- OS Hardening (CIS Benchmark)
    +-- Patch Management (自動パッチ適用)
    +-- Host-based IDS (OSSEC, Falco)
    +-- Immutable Infrastructure (コンテナ)

  Layer 4: アプリケーション防御
    +-- Input Validation (サーバーサイド)
    +-- Authentication (MFA, OAuth2)
    +-- Authorization (RBAC, ABAC)
    +-- Session Management

  Layer 5: データ防御
    +-- Encryption at Rest (AES-256)
    +-- Encryption in Transit (TLS 1.3)
    +-- Data Loss Prevention (DLP)
    +-- Backup & Recovery
```

### コード例5: 多層防御チェーンの実装

```python
# コード例5: 多層防御の実装パターン — Chain of Responsibilityパターン
from typing import List, Callable, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import re
import hashlib
import logging

logger = logging.getLogger(__name__)

@dataclass
class SecurityCheckResult:
    passed: bool
    layer: str
    message: str
    severity: str = "info"  # info, warning, critical
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SecurityRequest:
    """セキュリティ検証対象のリクエスト"""
    ip: str
    method: str
    path: str
    headers: Dict[str, str]
    body: Any
    user_agent: str
    timestamp: datetime = field(default_factory=datetime.now)

class SecurityLayer(ABC):
    """防御層の抽象基底クラス"""

    @abstractmethod
    def check(self, request: SecurityRequest) -> SecurityCheckResult:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

class RateLimitLayer(SecurityLayer):
    """Layer 1: レートリミット"""
    name = "RateLimit"

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._counters: Dict[str, List[datetime]] = {}

    def check(self, request: SecurityRequest) -> SecurityCheckResult:
        ip = request.ip
        now = datetime.now()
        # ウィンドウ外のリクエストを除去
        self._counters.setdefault(ip, [])
        self._counters[ip] = [
            t for t in self._counters[ip]
            if (now - t).total_seconds() < self.window_seconds
        ]
        self._counters[ip].append(now)

        count = len(self._counters[ip])
        if count > self.max_requests:
            return SecurityCheckResult(
                False, self.name,
                f"Rate limit exceeded: {count}/{self.max_requests}",
                severity="warning",
                details={"ip": ip, "count": count}
            )
        return SecurityCheckResult(True, self.name, "Rate limit OK")

class WAFLayer(SecurityLayer):
    """Layer 2: WAF（Web Application Firewall）"""
    name = "WAF"

    DANGEROUS_PATTERNS = [
        (r"('|\")\s*(OR|AND)\s+.*=", "SQL Injection"),
        (r";\s*(DROP|DELETE|UPDATE|INSERT)", "SQL Injection"),
        (r"UNION\s+SELECT", "SQL Injection"),
        (r"<script[^>]*>", "XSS"),
        (r"javascript:", "XSS"),
        (r"on\w+\s*=", "XSS Event Handler"),
        (r"\.\./\.\./", "Path Traversal"),
        (r"/etc/passwd", "Path Traversal"),
        (r"\$\{.*\}", "Template Injection"),
        (r"{{.*}}", "Template Injection"),
    ]

    def check(self, request: SecurityRequest) -> SecurityCheckResult:
        body = str(request.body or "")
        path = request.path
        check_target = f"{path} {body}"

        for pattern, attack_type in self.DANGEROUS_PATTERNS:
            if re.search(pattern, check_target, re.IGNORECASE):
                return SecurityCheckResult(
                    False, self.name,
                    f"WAF blocked: {attack_type} detected",
                    severity="critical",
                    details={"attack_type": attack_type, "path": path}
                )
        return SecurityCheckResult(True, self.name, "WAF check passed")

class InputValidationLayer(SecurityLayer):
    """Layer 3: 入力値検証"""
    name = "InputValidation"

    MAX_FIELD_LENGTH = 10000
    MAX_BODY_SIZE = 1_000_000  # 1MB

    def check(self, request: SecurityRequest) -> SecurityCheckResult:
        body = request.body
        if isinstance(body, (str, bytes)):
            if len(body) > self.MAX_BODY_SIZE:
                return SecurityCheckResult(
                    False, self.name,
                    f"Body too large: {len(body)} bytes",
                    severity="warning"
                )
        elif isinstance(body, dict):
            for key, value in body.items():
                if isinstance(value, str) and len(value) > self.MAX_FIELD_LENGTH:
                    return SecurityCheckResult(
                        False, self.name,
                        f"Input too long: {key} ({len(value)} chars)",
                        severity="warning"
                    )
        return SecurityCheckResult(True, self.name, "Input validation passed")

class DefenseInDepth:
    """多層防御チェーンのオーケストレーター"""

    def __init__(self):
        self.layers: List[SecurityLayer] = []

    def add_layer(self, layer: SecurityLayer) -> 'DefenseInDepth':
        self.layers.append(layer)
        return self  # メソッドチェーン対応

    def validate(self, request: SecurityRequest) -> List[SecurityCheckResult]:
        """全防御層を順に通過させる"""
        results = []
        for layer in self.layers:
            try:
                result = layer.check(request)
                results.append(result)
                if not result.passed:
                    logger.warning(
                        "Security check failed: layer=%s msg=%s severity=%s",
                        result.layer, result.message, result.severity
                    )
                    # 失敗しても全層の状態を把握するため、
                    # critical以外は検証を続行するオプションもある
                    if result.severity == "critical":
                        break
            except Exception as e:
                result = SecurityCheckResult(
                    False, layer.name, f"Error: {e}", severity="critical"
                )
                results.append(result)
                logger.error("Security layer error: %s %s", layer.name, e)
                break
        return results

    def is_allowed(self, request: SecurityRequest) -> bool:
        """リクエストが全層を通過したかを判定"""
        results = self.validate(request)
        return all(r.passed for r in results)

# 防御チェーンの構築と使用例
defense = (DefenseInDepth()
    .add_layer(RateLimitLayer(max_requests=100))
    .add_layer(WAFLayer())
    .add_layer(InputValidationLayer()))

# テスト
test_request = SecurityRequest(
    ip="192.168.1.100",
    method="POST",
    path="/api/search",
    headers={"Content-Type": "application/json"},
    body={"query": "normal search term"},
    user_agent="Mozilla/5.0"
)
print(defense.is_allowed(test_request))  # True

malicious_request = SecurityRequest(
    ip="192.168.1.100",
    method="POST",
    path="/api/search",
    headers={"Content-Type": "application/json"},
    body={"query": "' OR 1=1 --"},
    user_agent="sqlmap/1.0"
)
print(defense.is_allowed(malicious_request))  # False
```

### 多層防御における独立性の重要性

多層防御の効果を最大化するには、各層が**独立して動作**することが重要である。

```
独立性の原則:

  NG: 層同士が依存している
  +--------+     +--------+     +--------+
  | WAF    | --> | 認証   | --> | 認可   |
  | (WAFが |     | (WAFの |     |        |
  |  通せば |     |  結果に |     |        |
  |  認証   |     |  依存)  |     |        |
  |  スキップ)|   |        |     |        |
  +--------+     +--------+     +--------+
  → WAFをバイパスすると全層が無効化

  OK: 各層が独立して判定
  +--------+     +--------+     +--------+
  | WAF    | --> | 認証   | --> | 認可   |
  | (独立   |     | (独立   |     | (独立  |
  |  判定)  |     |  判定)  |     |  判定) |
  +--------+     +--------+     +--------+
  → 1層が突破されても他の層は正常に機能
```

---

## 4. ゼロトラスト（Zero Trust）

「決して信頼せず、常に検証する（Never Trust, Always Verify）」という原則。NIST SP 800-207で正式に定義されたアーキテクチャモデルで、ネットワーク境界の内外を問わず、すべてのアクセスを検証する。

### 従来型境界防御との比較

```
従来型 (境界防御 / Castle-and-Moat):

  +---外部---+---内部---+
  |          | FW |     |
  | 攻撃者   | -- | 自由|    問題点:
  |          |    | 移動|    1. FWを突破すると内部は無防備
  +----------+----+-----+    2. VPN接続後の横展開が容易
   FWを突破すると内部は無防備  3. 内部脅威に対して脆弱
                              4. クラウド/リモートワークに不適合

ゼロトラスト:

  +--+  +--+  +--+  +--+
  |検| ->|検| ->|検| ->|検|    利点:
  |証|  |証|  |証|  |証|    1. すべてのアクセスを毎回検証
  +--+  +--+  +--+  +--+    2. 横展開を効果的に防止
  毎回  毎回  毎回  毎回     3. 内部脅威にも対応
  検証  検証  検証  検証     4. クラウドネイティブに適合

NIST SP 800-207 ゼロトラストの3つの基本原則:
  1. リソースへのアクセスは、セッション単位で許可する
  2. リソースへのアクセスは、動的ポリシーで決定する
  3. すべての通信は、ネットワーク位置に関係なく保護する
```

### ゼロトラストの7つの柱（CISA Zero Trust Maturity Model）

```
+================================================================+
|         CISA ゼロトラスト成熟度モデルの7つの柱                     |
|================================================================|
|                                                                |
|  1. Identity (アイデンティティ)                                  |
|     +-- 強力な認証 (MFA, FIDO2)                                |
|     +-- 継続的な身元確認                                        |
|     +-- リスクベースの認証強度調整                               |
|                                                                |
|  2. Devices (デバイス)                                          |
|     +-- デバイスの健全性検証 (MDM, EDR)                         |
|     +-- デバイス証明書によるデバイス認証                         |
|     +-- BYOD vs 管理デバイスのポリシー分離                      |
|                                                                |
|  3. Networks (ネットワーク)                                      |
|     +-- マイクロセグメンテーション                               |
|     +-- 暗号化通信 (mTLS)                                      |
|     +-- ネットワーク位置に依存しないアクセス制御                 |
|                                                                |
|  4. Applications & Workloads (アプリケーション)                  |
|     +-- アプリレベルの認証・認可                                 |
|     +-- API ゲートウェイでの制御                                |
|     +-- サービスメッシュ (Istio, Linkerd)                       |
|                                                                |
|  5. Data (データ)                                               |
|     +-- データ分類とラベリング                                  |
|     +-- 暗号化 (保存時・転送時)                                 |
|     +-- DLP (Data Loss Prevention)                             |
|                                                                |
|  6. Visibility & Analytics (可視性と分析)                        |
|     +-- 統合ログ収集 (SIEM)                                    |
|     +-- 行動分析 (UEBA)                                        |
|     +-- リアルタイムリスクスコアリング                           |
|                                                                |
|  7. Automation & Orchestration (自動化)                          |
|     +-- SOAR (Security Orchestration, Automation, Response)    |
|     +-- 自動封じ込め                                           |
|     +-- Policy as Code                                         |
+================================================================+
```

| 柱 | 説明 | 実装例 | 成熟度指標 |
|----|------|--------|-----------|
| Identity | すべてのユーザー・サービスを認証 | MFA、FIDO2、証明書認証 | パスワードレス認証率 |
| Devices | デバイスの正当性とセキュリティ状態を検証 | MDM、EDR、デバイス証明書 | 管理デバイス率 |
| Networks | マイクロセグメンテーションによる通信制御 | サービスメッシュ、mTLS | mTLS カバレッジ率 |
| Applications | アプリレベルでのアクセス制御 | OAuth2、RBAC、API Gateway | API認証カバレッジ |
| Data | データの分類と暗号化 | 暗号化、DLP、ラベリング | 暗号化率 |
| Visibility | 統合ログ収集と分析 | SIEM、UEBA | ログカバレッジ率 |
| Automation | 自動検知・自動対応 | SOAR、Policy as Code | 自動対応率 |

### コード例6: ゼロトラストポリシーエンジンの実装

```python
# コード例6: ゼロトラストリクエスト検証 — NIST SP 800-207準拠
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class TrustLevel(Enum):
    """信頼レベル — 認証強度とコンテキストに基づく"""
    UNTRUSTED = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class DeviceTrustLevel(Enum):
    UNKNOWN = "unknown"
    UNMANAGED = "unmanaged"
    MANAGED = "managed"
    COMPLIANT = "compliant"  # managedかつポリシー準拠

class NetworkLocation(Enum):
    PUBLIC = "public"
    VPN = "vpn"
    CORPORATE = "corporate"

@dataclass
class ZeroTrustContext:
    """ゼロトラスト検証に必要なコンテキスト"""
    user_id: str
    device_id: str
    device_trust_level: DeviceTrustLevel
    network_location: NetworkLocation
    mfa_verified: bool
    mfa_method: Optional[str] = None  # totp, fido2, push
    last_auth_time: datetime = field(default_factory=datetime.now)
    client_cert_valid: bool = False
    risk_score: float = 0.0          # 0.0 - 1.0
    geo_location: Optional[str] = None
    previous_geo_location: Optional[str] = None
    login_hour: Optional[int] = None
    failed_attempts: int = 0

@dataclass
class PolicyDecision:
    """ポリシーエンジンの判定結果"""
    allowed: bool
    trust_level: TrustLevel
    checks: Dict[str, bool]
    action_required: Optional[str] = None
    reason: Optional[str] = None
    conditions: List[str] = field(default_factory=list)

class ZeroTrustPolicyEngine:
    """ゼロトラストポリシーエンジン — 動的・継続的検証"""

    # リソースごとの要求信頼レベル
    RESOURCE_TRUST_REQUIREMENTS: Dict[str, TrustLevel] = {
        "public_api": TrustLevel.LOW,
        "user_profile": TrustLevel.MEDIUM,
        "admin_panel": TrustLevel.CRITICAL,
        "financial_data": TrustLevel.CRITICAL,
        "user_data": TrustLevel.HIGH,
        "analytics": TrustLevel.MEDIUM,
    }

    MAX_SESSION_AGE = timedelta(hours=1)
    MAX_RISK_SCORE = 0.7
    IMPOSSIBLE_TRAVEL_THRESHOLD_KM = 500  # 1時間以内

    def evaluate(self, ctx: ZeroTrustContext,
                 resource: str, action: str) -> PolicyDecision:
        """アクセスリクエストを全方位的に評価する"""
        checks = {
            "identity_verified": self._check_identity(ctx),
            "device_compliant": self._check_device(ctx),
            "network_allowed": self._check_network(ctx, resource),
            "mfa_satisfied": self._check_mfa(ctx, resource),
            "session_fresh": self._check_session_age(ctx),
            "risk_acceptable": self._check_risk(ctx),
            "no_impossible_travel": self._check_impossible_travel(ctx),
            "within_business_hours": self._check_business_hours(ctx, resource),
        }

        # 信頼レベルの計算
        trust_level = self._calculate_trust_level(ctx, checks)

        # リソースの要求レベルとの比較
        required_level = self.RESOURCE_TRUST_REQUIREMENTS.get(
            resource, TrustLevel.MEDIUM
        )

        all_passed = (
            all(checks.values())
            and trust_level.value >= required_level.value
        )

        decision = PolicyDecision(
            allowed=all_passed,
            trust_level=trust_level,
            checks=checks,
        )

        if not all_passed:
            decision.action_required = self._get_remediation(
                checks, trust_level, required_level
            )
            decision.reason = self._get_denial_reason(checks)

        # 監査ログの出力
        logger.info(
            "ZeroTrust decision: user=%s resource=%s action=%s "
            "allowed=%s trust=%s required=%s",
            ctx.user_id, resource, action,
            all_passed, trust_level.name, required_level.name,
        )

        return decision

    def _check_identity(self, ctx: ZeroTrustContext) -> bool:
        return bool(ctx.user_id)

    def _check_device(self, ctx: ZeroTrustContext) -> bool:
        return ctx.device_trust_level in (
            DeviceTrustLevel.MANAGED,
            DeviceTrustLevel.COMPLIANT,
        )

    def _check_network(self, ctx: ZeroTrustContext, resource: str) -> bool:
        sensitive_resources = {"admin_panel", "financial_data"}
        if resource in sensitive_resources:
            return ctx.network_location in (
                NetworkLocation.CORPORATE,
                NetworkLocation.VPN,
            )
        return True  # ゼロトラストではネットワーク位置は補助的

    def _check_mfa(self, ctx: ZeroTrustContext, resource: str) -> bool:
        sensitive_resources = {"admin_panel", "financial_data", "user_data"}
        if resource in sensitive_resources:
            if not ctx.mfa_verified:
                return False
            # 機密リソースではFIDO2を要求
            if resource == "financial_data":
                return ctx.mfa_method == "fido2"
        return True

    def _check_session_age(self, ctx: ZeroTrustContext) -> bool:
        return (datetime.now() - ctx.last_auth_time) < self.MAX_SESSION_AGE

    def _check_risk(self, ctx: ZeroTrustContext) -> bool:
        return ctx.risk_score < self.MAX_RISK_SCORE

    def _check_impossible_travel(self, ctx: ZeroTrustContext) -> bool:
        """不可能な移動（Impossible Travel）の検出"""
        if ctx.geo_location and ctx.previous_geo_location:
            if ctx.geo_location != ctx.previous_geo_location:
                # 実際にはジオコーディングで距離計算
                # ここでは異なる国の場合のみフラグ
                return False
        return True

    def _check_business_hours(self, ctx: ZeroTrustContext,
                              resource: str) -> bool:
        """営業時間外のアクセスチェック"""
        sensitive = {"admin_panel", "financial_data"}
        if resource in sensitive and ctx.login_hour is not None:
            return 6 <= ctx.login_hour <= 22
        return True

    def _calculate_trust_level(self, ctx: ZeroTrustContext,
                               checks: Dict[str, bool]) -> TrustLevel:
        """コンテキストに基づく信頼レベルの動的計算"""
        score = 0
        if checks["identity_verified"]:
            score += 1
        if checks["device_compliant"]:
            score += 1
        if checks["mfa_satisfied"]:
            score += 1
        if ctx.client_cert_valid:
            score += 1
        if ctx.risk_score < 0.3:
            score += 1

        if score >= 5:
            return TrustLevel.CRITICAL
        elif score >= 4:
            return TrustLevel.HIGH
        elif score >= 3:
            return TrustLevel.MEDIUM
        elif score >= 1:
            return TrustLevel.LOW
        return TrustLevel.UNTRUSTED

    def _get_remediation(self, checks: Dict[str, bool],
                         current: TrustLevel,
                         required: TrustLevel) -> str:
        if not checks["mfa_satisfied"]:
            return "MFA認証が必要です"
        if not checks["device_compliant"]:
            return "管理対象デバイスからアクセスしてください"
        if not checks["session_fresh"]:
            return "セッションが期限切れです。再認証してください"
        if not checks["no_impossible_travel"]:
            return "異常なアクセスパターンが検出されました。本人確認が必要です"
        if current.value < required.value:
            return f"このリソースには {required.name} レベルの信頼が必要です"
        return "アクセスが拒否されました。管理者に連絡してください"

    def _get_denial_reason(self, checks: Dict[str, bool]) -> str:
        failed = [k for k, v in checks.items() if not v]
        return f"Failed checks: {', '.join(failed)}"
```

### Google BeyondCorpの実装事例

Google BeyondCorpは、ゼロトラストアーキテクチャの最も有名な実装事例である。

```
Google BeyondCorp アーキテクチャ:

  +---------------------------------------------------------+
  |  従来: VPN → 社内ネットワーク → リソースに自由アクセス     |
  +---------------------------------------------------------+
                          ↓
  +---------------------------------------------------------+
  |  BeyondCorp:                                             |
  |                                                          |
  |  ユーザー                                                |
  |    │                                                    |
  |    ▼                                                    |
  |  [Access Proxy]                                         |
  |    │  ← ユーザー認証 (SSO + MFA)                        |
  |    │  ← デバイス認証 (デバイス証明書)                     |
  |    │  ← デバイスインベントリ確認                          |
  |    │  ← アクセスポリシー評価                              |
  |    │  ← リスクスコア確認                                 |
  |    ▼                                                    |
  |  [リソース]  ← VPNは不要、全通信がインターネット経由      |
  +---------------------------------------------------------+

  核心: ネットワーク位置ではなく、ユーザーとデバイスの
        信頼性に基づいてアクセスを制御する
```

---

## 5. セキュアバイデフォルト（Secure by Default）

システムの初期状態を安全な構成にする原則。ユーザーが明示的に緩和しない限り、最も安全な設定が適用される。この原則は「心理的受容性」の原則と組み合わせることで、安全性とユーザビリティを両立する。

### なぜセキュアバイデフォルトが重要か

統計的に、多くのセキュリティインシデントは「デフォルト設定のまま運用した」ことが原因で発生する。デフォルトパスワード、デフォルトで有効なデバッグモード、デフォルトで無効な暗号化など、安全でないデフォルト設定は攻撃者にとっての「低い果実（Low-Hanging Fruit）」である。

```
セキュアバイデフォルトの考え方:

  安全でないデフォルト（NG）:           セキュアバイデフォルト（OK）:
  +-----------------------------+      +-----------------------------+
  | debug_mode: true            |      | debug_mode: false           |
  | tls_enabled: false          |      | tls_enabled: true           |
  | cors_origin: "*"            |      | cors_origin: [] (空)        |
  | cookie_secure: false        |      | cookie_secure: true         |
  | log_sensitive: true         |      | log_sensitive: false        |
  | admin_password: "admin"     |      | admin_password: (未設定→   |
  |                             |      |   初回起動時に強制変更)      |
  +-----------------------------+      +-----------------------------+

  原則:
  - セキュリティ設定は「オプトアウト」モデル
    (デフォルト安全、必要に応じて緩和)
  - 緩和時には警告を表示し、理由をログに記録
  - 本番環境での緩和は追加承認を要求
```

### コード例7: セキュアバイデフォルトの設定クラス

```python
# コード例7: セキュアバイデフォルトの設定クラス — 完全版
from dataclasses import dataclass, field
from typing import List, Optional, Set
import warnings
import logging

logger = logging.getLogger(__name__)

@dataclass
class SecureServerConfig:
    """セキュアバイデフォルトなサーバー設定

    設計方針:
    1. すべてのデフォルト値は「最も安全な設定」
    2. 緩和する場合は明示的なメソッド呼び出しが必要
    3. 緩和時には警告ログを出力
    4. 本番環境では一部の緩和を禁止
    """

    # --- デフォルトで安全な設定 ---
    # TLS設定
    tls_min_version: str = "TLSv1.3"
    tls_ciphers: List[str] = field(default_factory=lambda: [
        "TLS_AES_256_GCM_SHA384",
        "TLS_CHACHA20_POLY1305_SHA256",
    ])
    hsts_enabled: bool = True
    hsts_max_age: int = 31536000  # 1年
    hsts_include_subdomains: bool = True
    hsts_preload: bool = True

    # セキュリティヘッダー（デフォルトで有効）
    content_security_policy: str = "default-src 'self'"
    x_frame_options: str = "DENY"
    x_content_type_options: str = "nosniff"
    referrer_policy: str = "strict-origin-when-cross-origin"
    permissions_policy: str = "camera=(), microphone=(), geolocation=()"

    # Cookie設定（デフォルトで安全）
    cookie_secure: bool = True
    cookie_httponly: bool = True
    cookie_samesite: str = "Strict"
    cookie_max_age: int = 3600  # 1時間（短め）

    # CORS（デフォルトで制限）
    cors_allowed_origins: List[str] = field(default_factory=list)
    cors_allow_credentials: bool = False
    cors_max_age: int = 600  # 10分

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

    # 環境
    _environment: str = "production"

    def relax_for_development(self):
        """開発環境用の緩和設定（本番では呼び出し禁止）"""
        if self._environment == "production":
            raise RuntimeError(
                "本番環境でセキュリティ設定を緩和することはできません"
            )
        logger.warning(
            "セキュリティ設定を開発環境向けに緩和しています。"
            "本番環境では絶対に使用しないでください。"
        )
        self.tls_min_version = "TLSv1.2"
        self.cors_allowed_origins = ["http://localhost:3000"]
        self.cookie_secure = False  # HTTPでの開発用
        self.debug_mode = True

    def add_cors_origin(self, origin: str):
        """CORS許可オリジンを追加する（監査ログ付き）"""
        if origin == "*":
            raise ValueError(
                "CORS origin に '*' は許可されません。"
                "具体的なオリジンを指定してください。"
            )
        logger.info("CORS origin added: %s", origin)
        self.cors_allowed_origins.append(origin)

    def relax_csp(self, directive: str, value: str):
        """CSPを緩和する（警告付き）"""
        if "'unsafe-inline'" in value or "'unsafe-eval'" in value:
            warnings.warn(
                f"CSP の {directive} に unsafe ディレクティブを追加しています。"
                "XSS のリスクが増大します。",
                SecurityWarning,
            )
        logger.warning("CSP relaxed: %s = %s", directive, value)
        self.content_security_policy = (
            f"{self.content_security_policy}; {directive} {value}"
        )

    def to_headers(self) -> dict:
        """HTTPレスポンスヘッダーを生成する"""
        headers = {
            "X-Content-Type-Options": self.x_content_type_options,
            "X-Frame-Options": self.x_frame_options,
            "Referrer-Policy": self.referrer_policy,
            "Content-Security-Policy": self.content_security_policy,
            "Permissions-Policy": self.permissions_policy,
        }
        if self.hsts_enabled:
            hsts_value = f"max-age={self.hsts_max_age}"
            if self.hsts_include_subdomains:
                hsts_value += "; includeSubDomains"
            if self.hsts_preload:
                hsts_value += "; preload"
            headers["Strict-Transport-Security"] = hsts_value
        if not self.expose_server_version:
            headers["Server"] = ""  # サーバーバージョンを隠蔽
        return headers

    def validate(self) -> List[str]:
        """設定の安全性を検証し、問題点を返す"""
        issues = []
        if self.debug_mode and self._environment == "production":
            issues.append("CRITICAL: 本番環境でデバッグモードが有効です")
        if not self.hsts_enabled:
            issues.append("WARNING: HSTSが無効です")
        if not self.cookie_secure:
            issues.append("WARNING: Cookieのsecureフラグが無効です")
        if self.cors_allowed_origins == ["*"]:
            issues.append("CRITICAL: CORSが全オリジンに開放されています")
        if self.expose_stack_traces:
            issues.append("WARNING: スタックトレースが公開されています")
        if self.log_sensitive_data:
            issues.append("WARNING: 機密データのログが有効です")
        if self.tls_min_version < "TLSv1.2":
            issues.append("CRITICAL: TLS 1.2未満は安全ではありません")
        return issues

class SecurityWarning(UserWarning):
    """セキュリティに関する警告"""
    pass

# デフォルト設定は安全
config = SecureServerConfig()
print(config.debug_mode)         # False（安全）
print(config.cookie_secure)      # True（安全）
print(config.tls_min_version)    # TLSv1.3（安全）
print(config.validate())         # []（問題なし）
```

---

## 6. その他の重要なセキュリティ原則

### 6.1 フェイルセーフデフォルト（Fail-Safe Defaults）

障害や例外が発生した場合に、システムが安全な状態に遷移する原則。

```
フェイルセーフのフロー:

  リクエスト --> [認可チェック]
                    |
              +-----+-----+
              |           |
           成功        失敗/エラー/タイムアウト
              |           |
         アクセス許可   アクセス拒否（安全側に倒す）
                         + ログ記録
                         + アラート送信
                         + 管理者通知

  実装の核心:
  - try/catchのcatchブロックは「許可」ではなく「拒否」を返す
  - タイムアウト時はアクセス拒否（応答がない = 承認されていない）
  - デフォルトのswitch/caseは「拒否」
  - ホワイトリスト方式（許可リストにないものは拒否）
```

```python
# フェイルセーフの実装例
def check_authorization(user_id: str, resource: str) -> bool:
    """フェイルセーフな認可チェック"""
    try:
        # 認可サービスに問い合わせ
        result = auth_service.check(user_id, resource, timeout=5)
        if result.status == "ALLOW":
            return True
        # DENY や UNKNOWN は全て拒否
        return False
    except TimeoutError:
        # タイムアウト = 安全側に倒す = 拒否
        logger.warning("Auth check timeout for user=%s", user_id)
        return False
    except Exception as e:
        # 予期しないエラー = 安全側に倒す = 拒否
        logger.error("Auth check error: %s", e)
        return False
    # ここに到達することはないが、万が一の場合も拒否
    # return False
```

### 6.2 完全仲介（Complete Mediation）

すべてのアクセスを検証する原則。キャッシュされた認可結果に依存せず、毎回チェックを行う。

### 6.3 メカニズムの経済性（Economy of Mechanism）

セキュリティメカニズムの設計はシンプルに保つ。コードが複雑になるほどバグが増え、バグが増えるほど脆弱性が増える。

### 6.4 オープン設計（Open Design）

セキュリティは秘密に依存しない。暗号アルゴリズムは公開され、公開レビューを受けたものを使用する（Kerckhoffs' principle）。独自暗号、独自プロトコルは避ける。

### 6.5 権限分離（Separation of Privilege）

重要な操作には複数の独立した承認を要求する。4-eyes principle（二人以上の承認）やMFAがこの原則の実装である。

### 6.6 最少共有メカニズム（Least Common Mechanism）

共有リソースやメカニズムを最小化する。プロセス分離、サンドボックス、コンテナ化がこの原則の実装である。

### 6.7 心理的受容性（Psychological Acceptability）

セキュリティメカニズムはユーザーの業務を妨げないように設計する。過度に煩雑なセキュリティは回避行動を誘発し、結果的にセキュリティを低下させる。

### 原則の総合比較表

| 原則 | 説明 | 実装例 | 違反するとどうなるか |
|------|------|--------|-------------------|
| 最小権限 | 必要最小限の権限のみ付与 | RBAC、IAM | 侵害時の被害拡大 |
| 多層防御 | 複数の独立した防御層 | WAF+FW+入力検証 | 単一障害点が致命的に |
| ゼロトラスト | すべてのアクセスを検証 | mTLS、マイクロセグメンテーション | 内部からの横展開 |
| セキュアバイデフォルト | 初期状態を安全に | 安全なデフォルト値 | デフォルト設定の悪用 |
| フェイルセーフ | 障害時は安全な状態に | デフォルト拒否 | エラー時の不正アクセス |
| 完全仲介 | 毎回アクセスを検証 | ミドルウェアでの認可 | キャッシュ汚染攻撃 |
| 経済性 | 設計はシンプルに | 単純なRBAC | 複雑さによる脆弱性 |
| オープン設計 | 秘密に依存しない | 公開暗号アルゴリズム | 秘密の漏洩で崩壊 |
| 権限分離 | 複数の承認を要求 | MFA、4-eyes | 単一の権限奪取で侵害 |
| 最少共有 | 共有リソースを最小化 | プロセス分離 | 共有リソース経由の攻撃 |
| 心理的受容性 | ユーザビリティとの両立 | SSO、パスワードレス | セキュリティの回避行動 |

---

## 7. エッジケースと注意事項

### エッジケース1: 最小権限と緊急時対応の矛盾

最小権限を厳格に適用すると、緊急時（インシデント発生時）に必要な権限がなく対応が遅れる場合がある。これに対処するために「Break Glass」（ガラスを割る）手順を用意する。

```python
# Break Glass（緊急時権限昇格）の実装
class BreakGlassAccess:
    """緊急時の権限昇格メカニズム"""

    def activate(self, user_id: str, reason: str,
                 incident_id: str) -> str:
        """緊急権限を有効化する"""
        # 1. 複数チャネルでアラート送信
        self.alert_security_team(user_id, reason, incident_id)
        self.alert_management(user_id, reason, incident_id)

        # 2. 全操作を詳細にログ記録
        self.enable_detailed_audit_log(user_id)

        # 3. 時間制限付きの権限付与（最大2時間）
        token = self.grant_emergency_access(
            user_id,
            duration=timedelta(hours=2),
            incident_id=incident_id,
        )

        # 4. 自動失効タイマーの設定
        self.schedule_auto_revoke(user_id, token)

        return token
```

### エッジケース2: ゼロトラストとレガシーシステムの共存

全システムを一度にゼロトラストに移行することは現実的でない。レガシーシステムにはプロキシやゲートウェイを配置してゼロトラスト層を追加する。

### エッジケース3: セキュアバイデフォルトと後方互換性

既存のクライアントがセキュアでない設定に依存している場合、デフォルト値の変更は破壊的変更になる。段階的な移行（まず警告→次バージョンで変更）が必要である。

### エッジケース4: 多層防御のコストとパフォーマンス

防御層を増やすほどレイテンシが増加する。各層のパフォーマンスインパクトを計測し、リスクとパフォーマンスのバランスを取る。

---

## 8. アンチパターン

### アンチパターン1: 過剰な権限付与（God Mode）

「面倒だから」「動かないから」という理由で管理者権限やワイルドカード権限を付与するパターン。特にIAMで `*` リソースに `*` アクションを許可するのは最も危険な設定である。

```json
// NG: God Mode ポリシー
{
  "Effect": "Allow",
  "Action": "*",
  "Resource": "*"
}
```

この設定は絶対に避け、具体的なアクション・リソースを指定すること。AWS IAM Access Analyzerを使用して、過剰な権限を検出・修正する。

### アンチパターン2: セキュリティ by オブスキュリティ（Security by Obscurity）

隠蔽のみに依存するセキュリティ。「管理画面のURLを推測しにくくすれば大丈夫」「ソースコードは非公開だから安全」という考えは危険である。

```
NG パターン:
  - 管理画面: /admin-secret-xyz123  ← URLの隠蔽だけで認証なし
  - APIキー: ソースコードにハードコード ← 非公開リポジトリだから安全?
  - 独自暗号: 自作の暗号化アルゴリズム ← 公開レビューを受けていない

OK パターン:
  - 管理画面: /admin + 強力な認証(MFA) + IP制限 + 監査ログ
  - APIキー: 環境変数 or Secrets Manager + ローテーション
  - 暗号化: AES-256-GCM + AWS KMS管理鍵
```

隠蔽は**補助的な対策**にはなるが、適切な認証・認可・暗号化が前提である。

### アンチパターン3: セキュリティの事後対応（Bolt-on Security）

開発の最後にセキュリティを「追加」するアプローチ。設計段階からセキュリティを組み込む（Security by Design / Shift Left）べきである。

```
NG: ウォーターフォールでの事後対応
  要件 → 設計 → 実装 → テスト → [セキュリティテスト] → リリース
                                        ↑
                                   ここで初めてセキュリティを考慮
                                   → 設計の根本的な問題は修正困難

OK: Shift Leftアプローチ
  [脅威モデリング] → [セキュア設計] → [SAST] → [DAST] → リリース
   ↑                  ↑                ↑         ↑
   設計段階から         セキュリティ     自動      ランタイム
   セキュリティを       アーキテクチャ   スキャン  テスト
   考慮                レビュー
```

---

## 9. 演習問題

### 演習1: 最小権限のIAMポリシー設計（難易度: 基礎）

以下の要件を満たすAWS IAMポリシーをJSON形式で記述せよ。

**要件:**
- S3バケット `data-lake-prod` の `reports/` プレフィックスに対してのみ読み取りを許可
- 同バケットの `uploads/` プレフィックスに対してのみ書き込みを許可
- 削除は一切禁止
- MFA認証済みの場合のみアクセス可能

**ヒント:** Condition要素を使用してMFA制約を追加する

### 演習2: 多層防御の設計（難易度: 応用）

以下のWebアプリケーションに対して、多層防御の設計を行え。

**システム構成:**
- フロントエンド: React SPA
- バックエンド: Python FastAPI
- データベース: PostgreSQL
- インフラ: AWS ECS on Fargate

**タスク:**
1. 最低5層の防御層を定義する
2. 各層で使用するツール/技術を選定する
3. 各層が独立して機能することを確認する方法を述べる
4. 1つの層が突破された場合のシナリオと対応を記述する

### 演習3: ゼロトラストポリシーの実装（難易度: 発展）

以下のシナリオに対して、ゼロトラストポリシーを実装せよ。

**シナリオ:**
- 従業員がカフェのWi-FiからBYOD端末で社内の顧客データにアクセスしようとしている
- 従業員はパスワード認証済みだがMFA未実施
- リスクスコアは0.5（中程度）
- 前回のログインは東京からで、現在のIPアドレスは大阪

**タスク:**
1. このアクセスリクエストを許可すべきか、拒否すべきかを判断する
2. 判断の根拠を各チェック項目ごとに説明する
3. アクセスを許可するために必要な追加条件を列挙する
4. 上記のZeroTrustPolicyEngineクラスを使って実装し、テストコードを書く

---

## 10. まとめ

| 原則 | 核心 | 実装のポイント | 参照規格 |
|------|------|---------------|---------|
| 最小権限 | 必要最小限の権限のみ付与 | RBAC、IAMポリシーの精密化、JIT Access | NIST AC-6 |
| 多層防御 | 複数の独立した防御層を重ねる | WAF+FW+入力検証+暗号化、各層の独立性 | NIST SC-3 |
| ゼロトラスト | すべてのアクセスを検証する | mTLS、マイクロセグメンテーション、継続的検証 | NIST SP 800-207 |
| セキュアバイデフォルト | 初期状態を安全に | 安全なデフォルト値、opt-outモデル、設定検証 | CWE-1188 |
| フェイルセーフ | 障害時は安全な状態に | デフォルト拒否、エラー時のアクセス遮断 | NIST AC-7 |
| 権限分離 | 重要操作には複数の承認 | MFA、4-eyes principle、Break Glass | NIST AC-5 |
| オープン設計 | セキュリティは秘密に依存しない | 公開暗号、公開レビュー | Kerckhoffs' principle |

### 学習の次のステップ

1. 各原則を自身のプロジェクトに適用し、現状とのギャップ分析を行う
2. ゼロトラストの段階的導入計画を策定する
3. 多層防御の各層について、テストとモニタリングを実装する

---

## 次に読むべきガイド

- [../01-web-security/00-owasp-top10.md](../01-web-security/00-owasp-top10.md) -- セキュリティ原則を具体的脆弱性対策に適用
- [../04-application-security/00-secure-coding.md](../04-application-security/00-secure-coding.md) -- コーディングレベルでの原則適用
- [../05-cloud-security/00-cloud-security-basics.md](../05-cloud-security/00-cloud-security-basics.md) -- クラウド環境での原則適用
- [../06-operations/02-compliance.md](../06-operations/02-compliance.md) -- セキュリティ原則をコンプライアンス要件にマッピング

---

## FAQ

### Q1: 最小権限を徹底すると開発効率が下がりませんか?

初期設定コストは上がるが、インシデント対応コストを考えると長期的には効率的である。また、IaCで権限テンプレートを管理し、開発環境ではやや緩い権限を使いつつ、本番環境で厳格に適用するアプローチが現実的である。AWS IAM Access Analyzerを使えば、実際に使用されている権限のみに絞り込むことができる。

### Q2: ゼロトラストは社内ネットワークでも必要ですか?

必要である。近年のセキュリティインシデントの多くは内部ネットワークからの横展開（Lateral Movement）によるものであり、VPN接続後に社内リソースへ自由にアクセスできる状態は危険である。マイクロセグメンテーションとmTLSの導入が推奨される。SolarWinds事件（2020年）やColonial Pipeline事件（2021年）は、いずれも内部ネットワークの信頼に起因するインシデントであった。

### Q3: セキュアバイデフォルトとユーザビリティのバランスはどう取りますか?

安全な初期設定を提供しつつ、必要に応じてユーザーが明示的に緩和できる設計にする。例えば、CSPのデフォルトは厳格に設定し、特定のCDNが必要な場合にのみホワイトリストに追加する。「opt-out」モデルを採用し、緩和時には警告を表示する。セキュリティの「心理的受容性」の原則に従い、ユーザーの自然なワークフローを妨げない形でセキュリティを組み込むことが重要である。

### Q4: 多層防御ではどの層を優先すべきですか?

投資対効果（ROI）の観点からは、以下の優先順位が推奨される: (1) アプリケーション層（入力検証、認証・認可） (2) データ層（暗号化） (3) ネットワーク層（ファイアウォール、セグメンテーション） (4) ペリメター層（WAF） (5) ホスト層（OS強化）。アプリケーション層の防御は最も費用対効果が高く、多くの脆弱性を根本から防止できる。

### Q5: ゼロトラストを段階的に導入するにはどうすればよいですか?

以下の段階的アプローチが推奨される: Phase 1: アイデンティティ基盤の強化（SSO + MFA）、Phase 2: デバイス管理の導入（MDM + EDR）、Phase 3: マイクロセグメンテーションの実装、Phase 4: 継続的検証の自動化（UEBA + リスクスコアリング）。各フェーズで3-6ヶ月を見込む。

### Q6: Break Glass手順はどの程度頻繁に使用されるべきですか?

Break Glassは「最後の手段」であり、月に1回以上使用される場合は、通常の権限設定が不適切である可能性がある。使用頻度を追跡し、頻繁に使用されるケースについては、通常のアクセスパスを整備すべきである。全使用について事後レビューを実施し、プロセスの改善に活かす。

---

## 参考文献

1. Saltzer, J.H. & Schroeder, M.D., "The Protection of Information in Computer Systems" -- Proceedings of the IEEE, 1975
2. NIST SP 800-207, "Zero Trust Architecture" -- https://csrc.nist.gov/publications/detail/sp/800-207/final
3. Google BeyondCorp -- https://cloud.google.com/beyondcorp
4. OWASP Security Design Principles -- https://owasp.org/www-project-developer-guide/
5. CISA Zero Trust Maturity Model -- https://www.cisa.gov/zero-trust-maturity-model
6. NIST SP 800-160, "Systems Security Engineering" -- https://csrc.nist.gov/publications/detail/sp/800-160/vol-1/final
7. AWS Well-Architected Framework - Security Pillar -- https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/
8. CIS Controls v8 -- https://www.cisecurity.org/controls
