# セキュリティ概要

> 情報セキュリティの根幹であるCIA三原則から、追加属性、脅威分類、リスク評価、主要フレームワーク、多層防御、セキュリティ組織文化まで体系的に解説する。セキュリティの全体像を俯瞰し、以降の各章で深掘りする内容の基盤を構築する。

## この章で学ぶこと

1. **CIA三原則**（機密性・完全性・可用性）の意味と相互関係、トレードオフを理解する
2. **脅威の分類とリスク評価**の基本プロセスおよび定量的・定性的手法を把握する
3. **主要セキュリティフレームワーク**（NIST CSF, ISO 27001, CIS Controls 等）の特徴と使い分けを知る
4. **多層防御とセキュリティ開発ライフサイクル**の設計思想を身につける

## 前提知識

- プログラミングの基礎（Python の基本構文が読めること）
- ネットワークの基礎（HTTP、TCP/IP の概要）
- [../01-web-security/00-owasp-top10.md](../01-web-security/00-owasp-top10.md) -- OWASP Top 10（本章を読了後に推奨）
- [../../authentication-and-authorization/docs/00-fundamentals/](../../authentication-and-authorization/docs/00-fundamentals/) -- 認証・認可の基礎（関連知識）

---

## 1. 情報セキュリティとは何か

情報セキュリティとは、情報資産を**脅威**から保護し、事業継続性を確保するための活動全般を指す。技術的対策だけでなく、組織・人・プロセスを含む包括的な取り組みである。

### 1.1 情報セキュリティの3つの柱

```
+------------------------------------------------------------------+
|                  情報セキュリティの3つの柱                          |
|                                                                  |
|  +------------------+  +------------------+  +------------------+|
|  |  技術的対策       |  |  組織的対策       |  |  人的対策        ||
|  |                  |  |                  |  |                  ||
|  | - 暗号化         |  | - ポリシー策定    |  | - セキュリティ   ||
|  | - アクセス制御    |  | - リスク管理      |  |   意識教育      ||
|  | - IDS/IPS        |  | - インシデント     |  | - 訓練・演習    ||
|  | - FW/WAF         |  |   対応手順       |  | - 内部規程遵守  ||
|  | - 脆弱性管理     |  | - 事業継続計画    |  | - ソーシャル     ||
|  | - ログ監視       |  | - コンプライアンス |  |   エンジニアリ  ||
|  |                  |  |                  |  |   ング対策      ||
|  +------------------+  +------------------+  +------------------+|
|                                                                  |
|  すべてが連携して初めて効果的なセキュリティが実現する                  |
+------------------------------------------------------------------+
```

### WHY: なぜ情報セキュリティが重要なのか

情報セキュリティは単なる技術的課題ではなく、ビジネスそのものである。以下の理由から、組織にとって不可欠な要素となっている。

1. **法的義務**: GDPR、個人情報保護法、PCI DSS 等の規制に違反すると、巨額の罰金が科される
2. **ビジネスリスク**: データ漏洩1件あたりの平均コストは約445万ドル（IBM 2023年調査）
3. **信頼の喪失**: セキュリティインシデントは顧客信頼の失墜に直結し、回復に数年を要する
4. **サプライチェーンリスク**: 自社だけでなく取引先・パートナーにも影響が波及する

```python
# コード例1: セキュリティインシデントのコスト試算モデル
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class IncidentSeverity(Enum):
    """インシデントの深刻度レベル"""
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"
    CRITICAL = "重大"

@dataclass
class SecurityIncident:
    """セキュリティインシデントのモデル"""
    name: str
    severity: IncidentSeverity
    affected_records: int
    detection_days: int  # 検出までの日数
    containment_days: int  # 封じ込めまでの日数

    @property
    def estimated_cost(self) -> float:
        """
        インシデントコストの概算（IBM Cost of Data Breach 2023 ベース）
        レコード単価 * 影響レコード数 + 検出遅延ペナルティ
        """
        cost_per_record = {
            IncidentSeverity.LOW: 50,
            IncidentSeverity.MEDIUM: 120,
            IncidentSeverity.HIGH: 180,
            IncidentSeverity.CRITICAL: 250,
        }
        base_cost = cost_per_record[self.severity] * self.affected_records
        # 検出に200日以上かかると追加コスト30%
        delay_penalty = 1.3 if self.detection_days > 200 else 1.0
        return base_cost * delay_penalty

    @property
    def total_lifecycle_days(self) -> int:
        return self.detection_days + self.containment_days

    def summary(self) -> str:
        return (
            f"インシデント: {self.name}\n"
            f"  深刻度: {self.severity.value}\n"
            f"  影響レコード: {self.affected_records:,}\n"
            f"  ライフサイクル: {self.total_lifecycle_days}日 "
            f"(検出{self.detection_days}日 + 封じ込め{self.containment_days}日)\n"
            f"  推定コスト: ${self.estimated_cost:,.0f}"
        )

# 使用例
breach = SecurityIncident(
    name="顧客情報データベース漏洩",
    severity=IncidentSeverity.HIGH,
    affected_records=50000,
    detection_days=250,
    containment_days=60,
)
print(breach.summary())
# => インシデント: 顧客情報データベース漏洩
# =>   深刻度: 高
# =>   影響レコード: 50,000
# =>   ライフサイクル: 310日 (検出250日 + 封じ込め60日)
# =>   推定コスト: $11,700,000
```

---

## 2. CIA三原則

情報セキュリティの最も基本的な原則は **CIA triad** と呼ばれる3つの要素で構成される。すべてのセキュリティ活動はこの三原則の達成を目指している。

```
                    Confidentiality
                    (機密性)
                       /\
                      /  \
                     / 情報 \
                    / セキュ \
                   /  リティ  \
                  /   の核心   \
                 /______________\
          Integrity            Availability
          (完全性)              (可用性)

CIA三原則: すべてのセキュリティ活動の基盤
- 3要素のバランスが重要
- ビジネス要件に応じて優先度を調整
- 1つでも欠けると全体が崩壊する
```

### 2.1 機密性（Confidentiality）

許可された者だけが情報にアクセスできる状態を保つこと。不正なアクセスからデータを守ることが核心である。

**なぜ重要か**: 機密性が破られると、個人情報漏洩、企業秘密の流出、競争優位性の喪失が発生する。GDPR違反の場合、全世界年間売上の4%または2,000万ユーロの罰金が科される可能性がある。

```python
# コード例2: ロールベースアクセス制御（RBAC）による機密性の確保
import hashlib
import hmac
import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps

# ログ設定（監査証跡として重要）
logger = logging.getLogger("access_control")
logging.basicConfig(level=logging.INFO)

class Permission(Enum):
    """システムの権限定義"""
    READ = auto()
    WRITE = auto()
    DELETE = auto()
    MANAGE_USERS = auto()
    VIEW_AUDIT_LOG = auto()
    EXPORT_DATA = auto()

class DataClassification(Enum):
    """データ分類（機密レベル）"""
    PUBLIC = "公開"
    INTERNAL = "社内限定"
    CONFIDENTIAL = "機密"
    RESTRICTED = "極秘"

@dataclass
class User:
    """ユーザーモデル"""
    user_id: str
    name: str
    role: str
    department: str
    clearance_level: DataClassification = DataClassification.INTERNAL

@dataclass
class AccessControlSystem:
    """ロールベースアクセス制御（RBAC）の実装"""

    role_permissions: Dict[str, Set[Permission]] = field(default_factory=dict)
    data_classifications: Dict[str, DataClassification] = field(default_factory=dict)

    def __post_init__(self):
        # デフォルトのロール定義
        self.role_permissions = {
            "admin": {
                Permission.READ, Permission.WRITE, Permission.DELETE,
                Permission.MANAGE_USERS, Permission.VIEW_AUDIT_LOG,
                Permission.EXPORT_DATA,
            },
            "editor": {
                Permission.READ, Permission.WRITE,
            },
            "viewer": {
                Permission.READ,
            },
            "auditor": {
                Permission.READ, Permission.VIEW_AUDIT_LOG,
            },
        }

    def check_permission(self, user: User, action: Permission,
                         resource: str) -> bool:
        """ユーザーのロールとデータ分類に基づいてアクセスを判定"""
        # ロールに基づく権限チェック
        allowed_permissions = self.role_permissions.get(user.role, set())
        if action not in allowed_permissions:
            self._log_access_denied(user, action, resource, "権限不足")
            return False

        # データ分類に基づくアクセスチェック
        resource_classification = self.data_classifications.get(
            resource, DataClassification.INTERNAL
        )
        if not self._check_clearance(user.clearance_level, resource_classification):
            self._log_access_denied(user, action, resource, "クリアランス不足")
            return False

        self._log_access_granted(user, action, resource)
        return True

    def _check_clearance(self, user_level: DataClassification,
                         resource_level: DataClassification) -> bool:
        """ユーザーのクリアランスレベルがリソースの分類以上かチェック"""
        hierarchy = [
            DataClassification.PUBLIC,
            DataClassification.INTERNAL,
            DataClassification.CONFIDENTIAL,
            DataClassification.RESTRICTED,
        ]
        return hierarchy.index(user_level) >= hierarchy.index(resource_level)

    def _log_access_denied(self, user: User, action: Permission,
                           resource: str, reason: str) -> None:
        logger.warning(
            f"ACCESS DENIED: user={user.user_id}, action={action.name}, "
            f"resource={resource}, reason={reason}"
        )

    def _log_access_granted(self, user: User, action: Permission,
                            resource: str) -> None:
        logger.info(
            f"ACCESS GRANTED: user={user.user_id}, action={action.name}, "
            f"resource={resource}"
        )

# 使用例
acl = AccessControlSystem()
acl.data_classifications["customer_data"] = DataClassification.CONFIDENTIAL
acl.data_classifications["public_catalog"] = DataClassification.PUBLIC

admin_user = User("u001", "Admin Taro", "admin", "IT",
                   DataClassification.RESTRICTED)
viewer_user = User("u002", "Viewer Hanako", "viewer", "Marketing",
                    DataClassification.INTERNAL)

# admin は customer_data にアクセスできる
print(acl.check_permission(admin_user, Permission.READ, "customer_data"))
# => True

# viewer は customer_data にアクセスできない（クリアランス不足）
print(acl.check_permission(viewer_user, Permission.READ, "customer_data"))
# => False
```

### 2.2 完全性（Integrity）

情報が不正に改ざんされていないことを保証すること。データが正確で信頼できる状態を維持することが核心である。

**なぜ重要か**: 完全性が破られると、金融取引の改ざん、医療記録の変更、ソフトウェアの改竄（サプライチェーン攻撃）等が発生する。SolarWinds事件（2020年）では、ビルドシステムにバックドアが仕込まれ、18,000以上の組織が影響を受けた。

```python
# コード例3: ハッシュチェーンによるデータ完全性の検証
import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class IntegrityRecord:
    """改ざん検知可能なデータレコード"""
    data: dict
    timestamp: float = field(default_factory=time.time)
    previous_hash: str = ""
    record_hash: str = ""

    def __post_init__(self):
        if not self.record_hash:
            self.record_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """レコードのハッシュを計算（チェーン構造）"""
        content = json.dumps({
            "data": self.data,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def verify(self) -> bool:
        """レコードの完全性を検証"""
        expected = self._compute_hash()
        return hmac.compare_digest(self.record_hash, expected)

class IntegrityChain:
    """ハッシュチェーンによる完全性保証（簡易ブロックチェーン構造）"""

    def __init__(self):
        self.records: List[IntegrityRecord] = []

    def add_record(self, data: dict) -> IntegrityRecord:
        """新しいレコードをチェーンに追加"""
        previous_hash = (
            self.records[-1].record_hash if self.records else "GENESIS"
        )
        record = IntegrityRecord(
            data=data,
            previous_hash=previous_hash,
        )
        self.records.append(record)
        return record

    def verify_chain(self) -> dict:
        """チェーン全体の完全性を検証"""
        results = {"valid": True, "total": len(self.records), "errors": []}

        for i, record in enumerate(self.records):
            # 個別レコードのハッシュ検証
            if not record.verify():
                results["valid"] = False
                results["errors"].append(f"Record {i}: ハッシュ不一致（改ざん検出）")

            # チェーンの連続性検証
            if i > 0:
                expected_prev = self.records[i - 1].record_hash
                if record.previous_hash != expected_prev:
                    results["valid"] = False
                    results["errors"].append(
                        f"Record {i}: チェーン断裂（前レコード改ざん検出）"
                    )

        return results

# 使用例
chain = IntegrityChain()
chain.add_record({"transaction": "入金", "amount": 10000, "user": "alice"})
chain.add_record({"transaction": "出金", "amount": 3000, "user": "alice"})
chain.add_record({"transaction": "送金", "amount": 2000, "user": "bob"})

# 完全性検証
result = chain.verify_chain()
print(f"チェーン検証: {'OK' if result['valid'] else 'NG'}")
print(f"レコード数: {result['total']}")
# => チェーン検証: OK
# => レコード数: 3

# 改ざんテスト
chain.records[1].data["amount"] = 99999  # 改ざん!
result = chain.verify_chain()
print(f"改ざん後の検証: {'OK' if result['valid'] else 'NG'}")
print(f"エラー: {result['errors']}")
# => 改ざん後の検証: NG
# => エラー: ['Record 1: ハッシュ不一致（改ざん検出）']
```

### 2.3 可用性（Availability）

必要なときに情報やサービスが利用可能であること。サービスの継続性と信頼性を保つことが核心である。

**なぜ重要か**: 可用性が破られると、サービス停止による売上損失、SLA違反による違約金、ブランドイメージの毀損が発生する。AWSの大規模障害（2021年12月）では、Netflix、Disney+、Ring 等の多くのサービスが影響を受け、数時間のダウンタイムが発生した。

```python
# コード例4: 可用性監視とフェイルオーバー制御
import time
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

class ServiceStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class HealthCheckResult:
    """ヘルスチェック結果"""
    endpoint: str
    status: ServiceStatus
    latency_ms: float
    timestamp: float
    error: Optional[str] = None

@dataclass
class AvailabilityMonitor:
    """サービスの可用性を監視しフェイルオーバーを制御するシステム"""

    endpoints: List[str]
    timeout_seconds: int = 5
    max_failures: int = 3  # フェイルオーバー閾値
    check_interval: int = 30  # ヘルスチェック間隔（秒）
    failure_counts: Dict[str, int] = field(default_factory=dict)
    history: List[HealthCheckResult] = field(default_factory=list)

    def __post_init__(self):
        for ep in self.endpoints:
            self.failure_counts[ep] = 0

    def check_health(self, endpoint: str) -> HealthCheckResult:
        """
        エンドポイントの稼働状態を確認する。
        実際にはHTTPリクエストを送信するが、ここではシミュレーション。
        """
        start = time.time()
        try:
            # 実運用では requests.get(f"{endpoint}/health", timeout=self.timeout_seconds)
            # ここではシミュレーション
            latency = (time.time() - start) * 1000
            result = HealthCheckResult(
                endpoint=endpoint,
                status=ServiceStatus.HEALTHY,
                latency_ms=round(latency, 2),
                timestamp=time.time(),
            )
        except Exception as e:
            result = HealthCheckResult(
                endpoint=endpoint,
                status=ServiceStatus.UNHEALTHY,
                latency_ms=0,
                timestamp=time.time(),
                error=str(e),
            )

        # 失敗カウントの更新
        if result.status == ServiceStatus.UNHEALTHY:
            self.failure_counts[endpoint] = (
                self.failure_counts.get(endpoint, 0) + 1
            )
        else:
            self.failure_counts[endpoint] = 0

        self.history.append(result)
        return result

    def should_failover(self, endpoint: str) -> bool:
        """フェイルオーバーが必要かどうかを判定"""
        return self.failure_counts.get(endpoint, 0) >= self.max_failures

    def select_healthy_endpoint(self) -> Optional[str]:
        """最も健全なエンドポイントを選択"""
        for ep in self.endpoints:
            if not self.should_failover(ep):
                return ep
        return None  # 全エンドポイント障害

    def calculate_uptime(self, endpoint: str) -> float:
        """稼働率を計算"""
        ep_history = [h for h in self.history if h.endpoint == endpoint]
        if not ep_history:
            return 0.0
        healthy = sum(1 for h in ep_history
                      if h.status == ServiceStatus.HEALTHY)
        return (healthy / len(ep_history)) * 100

# 使用例
monitor = AvailabilityMonitor(
    endpoints=[
        "https://primary.example.com",
        "https://secondary.example.com",
        "https://tertiary.example.com",
    ],
    max_failures=3,
)

# ヘルスチェック実行（シミュレーション）
result = monitor.check_health("https://primary.example.com")
print(f"Status: {result.status.value}, Latency: {result.latency_ms}ms")

# フェイルオーバー判定
selected = monitor.select_healthy_endpoint()
print(f"Selected endpoint: {selected}")
```

### 2.4 CIA三原則の相互関係とトレードオフ

CIA三原則は相互に影響し合い、場合によってはトレードオフが発生する。

```
CIA三原則のトレードオフ:

  機密性 ←─────→ 可用性
   強い暗号化        迅速なアクセス
   厳格な認証        利便性
   アクセス制限       サービス継続

      \               /
       \             /
        \           /
         完全性
         データの正確性
         改ざん検知
         監査証跡

トレードオフの例:
- 暗号化を強化（機密性↑）→ パフォーマンス低下（可用性↓）
- アクセス制限を緩和（可用性↑）→ 不正アクセスリスク増（機密性↓）
- 監査ログ大量保存（完全性↑）→ ストレージ圧迫（可用性↓）
```

| 原則 | 脅威例 | 対策例 | 失敗時の影響 | 業界別重点 |
|------|--------|--------|-------------|----------|
| 機密性 | 不正アクセス、盗聴、内部犯行 | 暗号化、アクセス制御、DLP | 情報漏洩、プライバシー侵害、規制違反 | 金融、医療、政府 |
| 完全性 | データ改ざん、MITM、マルウェア | ハッシュ、デジタル署名、バージョン管理 | 不正取引、信頼性喪失、法的問題 | 金融、製造、ヘルスケア |
| 可用性 | DDoS、障害、災害、ランサムウェア | 冗長構成、CDN、BCP/DR、バックアップ | サービス停止、売上損失、SLA違反 | EC、SaaS、インフラ |

### 業界別CIA優先度

| 業界 | 最優先 | 理由 |
|------|--------|------|
| 金融（銀行） | 完全性 | 取引データの正確性が生命線 |
| 医療 | 可用性 | 緊急時にシステムが使えないと人命に関わる |
| 軍事/政府 | 機密性 | 国家機密の保護が最優先 |
| EC/小売 | 可用性 | サービス停止が直接の売上損失 |
| 個人情報取扱 | 機密性 | GDPRや個人情報保護法への準拠 |
| 製造/IoT | 完全性 | 制御データの改ざんが物理的被害に直結 |

---

## 3. 追加のセキュリティ属性

CIA三原則に加えて、ISO 27001等で定義される以下の属性も重要視される。これらは「拡張CIA」とも呼ばれる。

```
+-------------------------------------------------------------------+
|                  拡張セキュリティ属性                                |
|                                                                   |
|  +--------------+  +--------------+  +-----------+  +-----------+ |
|  | 真正性       |  | 責任追跡性    |  | 否認防止   |  | 信頼性    | |
|  | Authenticity |  | Account-     |  | Non-      |  | Relia-    | |
|  |              |  | ability      |  | repudia-  |  | bility    | |
|  |              |  |              |  | tion      |  |           | |
|  | デジタル署名  |  | 監査ログ     |  | タイム     |  | テスト    | |
|  | PKI          |  | アクセスログ  |  | スタンプ   |  | 冗長設計  | |
|  | 証明書       |  | SIEM         |  | 電子署名   |  | 品質管理  | |
|  +--------------+  +--------------+  +-----------+  +-----------+ |
|                                                                   |
|  +--------------+  +--------------+                               |
|  | プライバシー  |  | 安全性       |                               |
|  | Privacy      |  | Safety       |                               |
|  |              |  |              |                               |
|  | データ最小化  |  | フェイル     |                               |
|  | 匿名化       |  | セーフ       |                               |
|  | 同意管理     |  | 物理安全     |                               |
|  +--------------+  +--------------+                               |
+-------------------------------------------------------------------+
```

| 属性 | 説明 | 実現手段 | 実例 |
|------|------|---------|------|
| 真正性（Authenticity） | 情報の送信元が本物であること | デジタル署名、PKI、証明書検証 | メールのDKIM署名 |
| 責任追跡性（Accountability） | 行為者を特定・追跡できること | 監査ログ、アクセスログ、SIEM | AWS CloudTrail |
| 否認防止（Non-repudiation） | 行為を後から否定できないこと | タイムスタンプ、電子署名 | 電子契約サービス |
| 信頼性（Reliability） | システムが一貫して期待通りに動作 | テスト、冗長設計、品質管理 | SLA 99.99% |
| プライバシー | 個人情報の適切な取扱い | データ最小化、匿名化、同意管理 | GDPR準拠 |
| 安全性（Safety） | 物理的な安全の確保 | フェイルセーフ設計、安全装置 | 産業制御システム |

```python
# コード例5: 監査ログによる責任追跡性の実装
import json
import time
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field, asdict

@dataclass
class AuditLogEntry:
    """改ざん検知機能付き監査ログエントリ"""
    event_id: str
    timestamp: str
    actor: str            # 操作者
    action: str           # 操作内容
    resource: str         # 対象リソース
    result: str           # 成功/失敗
    source_ip: str        # 接続元IP
    details: Dict[str, Any] = field(default_factory=dict)
    integrity_hash: str = ""

    def __post_init__(self):
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """ログエントリの改ざん検知用ハッシュを計算"""
        content = json.dumps({
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "result": self.result,
            "source_ip": self.source_ip,
            "details": self.details,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """ログエントリの改ざんを検証"""
        import hmac as _hmac
        return _hmac.compare_digest(
            self.integrity_hash, self._compute_hash()
        )

class AuditLogger:
    """監査ログ管理システム"""

    def __init__(self):
        self.logs: list = []
        self._counter = 0

    def log(self, actor: str, action: str, resource: str,
            result: str, source_ip: str,
            details: Optional[Dict[str, Any]] = None) -> AuditLogEntry:
        """監査ログを記録"""
        self._counter += 1
        entry = AuditLogEntry(
            event_id=f"EVT-{self._counter:06d}",
            timestamp=datetime.utcnow().isoformat() + "Z",
            actor=actor,
            action=action,
            resource=resource,
            result=result,
            source_ip=source_ip,
            details=details or {},
        )
        self.logs.append(entry)
        return entry

    def query(self, actor: Optional[str] = None,
              action: Optional[str] = None) -> list:
        """監査ログの検索"""
        results = self.logs
        if actor:
            results = [e for e in results if e.actor == actor]
        if action:
            results = [e for e in results if e.action == action]
        return results

    def verify_all(self) -> Dict[str, Any]:
        """全ログの完全性を検証"""
        total = len(self.logs)
        valid = sum(1 for e in self.logs if e.verify_integrity())
        return {
            "total": total,
            "valid": valid,
            "tampered": total - valid,
            "integrity": "OK" if valid == total else "COMPROMISED",
        }

# 使用例
audit = AuditLogger()
audit.log("admin", "LOGIN", "/auth", "SUCCESS", "192.168.1.100")
audit.log("admin", "DELETE", "/users/42", "SUCCESS", "192.168.1.100",
          {"target_user": "user42", "reason": "規約違反"})
audit.log("user01", "READ", "/reports/financial", "DENIED", "10.0.0.50",
          {"reason": "権限不足"})

# ログの検索
admin_actions = audit.query(actor="admin")
print(f"admin の操作数: {len(admin_actions)}")

# 完全性検証
integrity = audit.verify_all()
print(f"ログ完全性: {integrity['integrity']}")
# => ログ完全性: OK
```

---

## 4. 脅威の分類

セキュリティ脅威を体系的に分類し理解することは、効果的な対策を立てるための前提である。

### 4.1 脅威の発生源による分類

```
+-------------------------------------------------------------------+
|                    脅威の分類体系                                    |
|                                                                   |
|  +--- 外部脅威 ---+   +--- 内部脅威 ---+   +--- 環境脅威 ---+     |
|  |                |   |                |   |                |     |
|  | - APT          |   | - 内部犯行     |   | - 自然災害     |     |
|  | - ランサムウェア |   | - 過失         |   | - 停電         |     |
|  | - フィッシング  |   | - ソーシャル    |   | - 火災         |     |
|  | - DDoS         |   |   エンジニア   |   | - パンデミック  |     |
|  | - ゼロデイ     |   |   リング被害   |   | - 水害         |     |
|  | - サプライ     |   | - BYOD        |   |                |     |
|  |   チェーン     |   | - シャドーIT   |   |                |     |
|  +----------------+   +----------------+   +----------------+     |
|                                                                   |
|  対策の方向性:                                                     |
|  外部 → FW、IDS/IPS、脅威インテリジェンス                           |
|  内部 → アクセス制御、DLP、教育訓練                                 |
|  環境 → BCP/DR、冗長化、バックアップ                                |
+-------------------------------------------------------------------+
```

### 4.2 STRIDE脅威モデル

STRIDE はMicrosoftが開発した脅威分類フレームワークで、6つのカテゴリで脅威を整理する。

| カテゴリ | 英語名 | 脅かされるCIA属性 | 攻撃例 | 対策 |
|---------|--------|-----------------|--------|------|
| S: なりすまし | Spoofing | 真正性 | フィッシング、セッションハイジャック | MFA、証明書認証 |
| T: 改ざん | Tampering | 完全性 | SQLインジェクション、MITM | ハッシュ、署名 |
| R: 否認 | Repudiation | 否認防止 | ログの消去 | 監査ログ、署名 |
| I: 情報漏洩 | Information Disclosure | 機密性 | ディレクトリリスティング | 暗号化、ACL |
| D: サービス妨害 | Denial of Service | 可用性 | DDoS、リソース枯渇 | CDN、レートリミット |
| E: 権限昇格 | Elevation of Privilege | 認可 | バッファオーバーフロー | 最小権限、サンドボックス |

詳細は [01-threat-modeling.md](./01-threat-modeling.md) で解説する。

---

## 5. リスク評価

リスク評価は「何を守るべきか」「どの脅威が最も危険か」を定量的・定性的に判断するプロセスである。

### 5.1 リスクの計算式

```
リスク = 脅威 × 脆弱性 × 影響度

  脅威（Threat）: 攻撃の発生可能性
  脆弱性（Vulnerability）: 弱点の存在
  影響度（Impact）: 攻撃成功時の被害

  リスク = 高 × 高 × 高 → 最優先で対処
  リスク = 高 × 低 × 高 → 脆弱性緩和が効果的
  リスク = 低 × 高 × 低 → 監視で十分
```

```python
# コード例6: リスクマトリクスの生成と可視化
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class RiskItem:
    """リスク評価対象"""
    name: str
    threat_level: int      # 1-5: 脅威レベル
    vulnerability: int     # 1-5: 脆弱性レベル
    impact: int            # 1-5: 影響度

    @property
    def risk_score(self) -> float:
        """リスクスコア（正規化: 0-100）"""
        raw = self.threat_level * self.vulnerability * self.impact
        return round((raw / 125) * 100, 1)  # 125 = 5*5*5

    @property
    def risk_level(self) -> str:
        score = self.risk_score
        if score >= 60:
            return "Critical"
        elif score >= 40:
            return "High"
        elif score >= 20:
            return "Medium"
        return "Low"

    @property
    def recommended_action(self) -> str:
        level = self.risk_level
        actions = {
            "Critical": "即時対応: 緩和策の即座の適用",
            "High": "優先対応: 30日以内に対策を実施",
            "Medium": "計画対応: 次のスプリントで対策を計画",
            "Low": "受容/監視: リスクを受容し定期的に再評価",
        }
        return actions[level]

def generate_risk_matrix() -> str:
    """5x5のリスクマトリクスを文字列として生成する"""
    impact_labels = ["軽微", "  小", "  中", "  大", "甚大"]
    likelihood_labels = ["  稀", "  低", "  中", "  高", "極高"]

    lines = []
    lines.append("影響度→  |  軽微  |   小   |   中   |   大   |  甚大  |")
    lines.append("発生可能性↓" + "-" * 50)

    for l_idx in range(5, 0, -1):
        row_label = likelihood_labels[l_idx - 1]
        cells = []
        for i_idx in range(1, 6):
            score = l_idx * i_idx
            if score >= 20:
                level = "C"  # Critical
            elif score >= 12:
                level = "H"  # High
            elif score >= 6:
                level = "M"  # Medium
            else:
                level = "L"  # Low
            cells.append(f"{score:2d}({level})")
        lines.append(f"  {row_label}   | {'  |  '.join(cells)}  |")

    return "\n".join(lines)

# リスクマトリクスの表示
print(generate_risk_matrix())

# 具体的なリスク評価
risks = [
    RiskItem("SQLインジェクション", 4, 5, 5),
    RiskItem("DDoS攻撃", 3, 3, 4),
    RiskItem("内部者による情報漏洩", 2, 4, 5),
    RiskItem("フィッシング", 5, 3, 3),
]

print("\n--- リスク評価結果 ---")
for risk in sorted(risks, key=lambda r: r.risk_score, reverse=True):
    print(f"{risk.name}: スコア={risk.risk_score}, "
          f"レベル={risk.risk_level}, "
          f"推奨={risk.recommended_action}")
```

### 5.2 定量的リスク評価 vs 定性的リスク評価

| 特性 | 定量的評価 | 定性的評価 |
|------|----------|----------|
| 表現方法 | 金額（ALE = SLE x ARO） | レベル（高/中/低） |
| 精度 | 高い（データに基づく） | 主観的（経験に基づく） |
| コスト | 高い（データ収集が必要） | 低い（専門家判断で可能） |
| 適用場面 | 経営判断、保険、投資対効果分析 | 初期評価、小規模プロジェクト |
| 代表的手法 | ALE分析、モンテカルロ法 | リスクマトリクス、DREAD |

---

## 6. セキュリティフレームワーク

### 6.1 主要フレームワーク比較

| フレームワーク | 発行元 | 特徴 | 対象 | 認証 | コスト |
|---------------|--------|------|------|------|-------|
| NIST CSF | 米国NIST | 5つの機能、柔軟性が高い | 全業種 | なし | 無料 |
| ISO 27001 | ISO/IEC | ISMS構築要求事項、国際標準 | 全業種（特に国際取引） | あり | 認証取得に費用要 |
| CIS Controls | CIS | 優先順位付き具体的対策18項目 | IT運用チーム | なし | 無料 |
| SOC 2 | AICPA | Trust Services Criteria | クラウドサービス提供者 | あり | 監査費用要 |
| PCI DSS | PCI SSC | カード情報保護12要件 | 決済関連事業者 | あり | 準拠評価費用要 |
| NIST SP 800-53 | 米国NIST | 包括的セキュリティ管理策 | 米国政府機関 | なし | 無料 |

### 6.2 NIST サイバーセキュリティフレームワーク 2.0

```
NIST CSF 2.0 の6つの機能（Governが追加）:

                    +----------+
                    | ガバナンス |
                    | Govern   |
                    +----+-----+
                         |
+----------+    +--------v-+    +----------+
|  特定    | -> |  防御    | -> |  検知    |
| Identify |    | Protect  |    | Detect   |
+----------+    +----------+    +----------+
                                      |
+----------+    +----------+          |
|  復旧    | <- |  対応    | <--------+
| Recover  |    | Respond  |
+----------+    +----------+

各機能は独立しておらず、ガバナンスが全体を統括する
```

| 機能 | 目的 | 主な活動 | 成果物例 |
|------|------|---------|---------|
| ガバナンス（Govern） | 組織の方針と戦略 | リスクマネジメント戦略、役割定義 | セキュリティポリシー |
| 特定（Identify） | 資産・リスクの把握 | 資産管理、リスク評価、環境分析 | 資産台帳、リスク登録簿 |
| 防御（Protect） | 脅威からの保護 | アクセス制御、暗号化、教育訓練 | アクセス制御基準 |
| 検知（Detect） | 異常の検出 | 監視、ログ分析、侵入検知 | SIEM設定、アラートルール |
| 対応（Respond） | インシデント対応 | 分析、封じ込め、通知、報告 | インシデント対応手順 |
| 復旧（Recover） | 事業の復旧 | 復旧計画実行、改善活動 | DR計画、復旧手順 |

---

## 7. セキュリティの実践的アプローチ

### 7.1 多層防御（Defense in Depth）

単一の防御に依存せず、複数の独立した防御層を重ねて攻撃の成功率を下げる戦略。城の防御に例えると、堀・城壁・門番・兵士・天守閣の多層構造に相当する。

```
多層防御モデル（外側から内側へ）:

+------------------------------------------------------------------+
|  L1: 物理セキュリティ（入退室管理、施錠、監視カメラ）                |
|  +------------------------------------------------------------+  |
|  |  L2: ネットワーク（FW、IDS/IPS、セグメンテーション、VPN）     |  |
|  |  +------------------------------------------------------+  |  |
|  |  |  L3: ホスト（OS強化、AV/EDR、パッチ管理、HIDS）        |  |  |
|  |  |  +--------------------------------------------------+|  |  |
|  |  |  |  L4: アプリ（入力検証、認証、認可、暗号化、WAF）    ||  |  |
|  |  |  |  +----------------------------------------------+||  |  |
|  |  |  |  |  L5: データ（暗号化、分類、DLP、バックアップ）  |||  |  |
|  |  |  |  +----------------------------------------------+||  |  |
|  |  |  +--------------------------------------------------+|  |  |
|  |  +------------------------------------------------------+  |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+

攻撃者は全ての層を突破しなければデータに到達できない
各層は独立して動作し、1つが突破されても残りが防御する
```

### 7.2 セキュリティ開発ライフサイクル（SDL）

```
SDL の各フェーズ:

  要件定義        設計           実装           テスト         運用
  +----------+  +----------+  +----------+  +----------+  +----------+
  | セキュリ |  | 脅威モデ  |  | セキュア  |  | 脆弱性   |  | 監視     |
  | ティ要件 |->| リング   |->| コーディ |->| スキャン  |->| パッチ   |
  | 定義     |  | レビュー  |  | ング     |  | ペンテスト|  | 管理     |
  +----------+  +----------+  +----------+  +----------+  +----------+
       |             |             |             |             |
  リスク評価    アーキテクチャ  SAST/SCA      DAST/IAST    インシデント
  コンプライ   セキュリティ   コードレビュー  レッドチーム   対応
  アンス確認   パターン       依存性管理     バグバウンティ  BCP/DR
```

### 7.3 ゼロトラストアーキテクチャ

「社内ネットワーク = 安全」という前提を捨て、すべてのアクセスを検証するアプローチ。

```
従来の境界防御:                    ゼロトラスト:

  +--- 外部(危険) ---+              +--+    +--+    +--+
  |                  |              |検|    |検|    |検|
  |  FW  +-- 内部 --|              |証| -> |証| -> |証|
  |  ==  | (安全)   |              +--+    +--+    +--+
  |      +----------|              毎回     毎回    毎回
  +------------------+              検証    検証    検証

  FWが破られると内部は無防備        すべてのアクセスで認証・認可
```

詳細は [02-security-principles.md](./02-security-principles.md) で解説する。

---

## 8. セキュリティメトリクス

セキュリティの効果を測定するための指標。「測定できないものは管理できない」の原則に基づく。

```python
# コード例7: セキュリティメトリクスダッシュボード
from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime, timedelta

@dataclass
class SecurityMetrics:
    """セキュリティKPIの収集と分析"""

    # 脆弱性管理メトリクス
    open_vulnerabilities: Dict[str, int] = field(default_factory=dict)
    # パッチ管理メトリクス
    patch_compliance_rate: float = 0.0
    # インシデントメトリクス
    incidents: List[dict] = field(default_factory=list)
    # 教育メトリクス
    training_completion_rate: float = 0.0

    def __post_init__(self):
        self.open_vulnerabilities = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }

    def add_incident(self, severity: str, detected_at: datetime,
                     resolved_at: datetime, category: str) -> None:
        """インシデントを記録"""
        self.incidents.append({
            "severity": severity,
            "detected_at": detected_at,
            "resolved_at": resolved_at,
            "category": category,
            "mttr_hours": (resolved_at - detected_at).total_seconds() / 3600,
        })

    @property
    def mean_time_to_respond(self) -> float:
        """平均対応時間（MTTR）を計算"""
        if not self.incidents:
            return 0.0
        total_hours = sum(i["mttr_hours"] for i in self.incidents)
        return round(total_hours / len(self.incidents), 2)

    @property
    def vulnerability_density(self) -> int:
        """脆弱性の総数"""
        return sum(self.open_vulnerabilities.values())

    def generate_report(self) -> str:
        """セキュリティメトリクスレポートを生成"""
        lines = [
            "=== セキュリティメトリクスレポート ===",
            f"日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "--- 脆弱性管理 ---",
            f"  未解決脆弱性: {self.vulnerability_density}件",
        ]
        for severity, count in self.open_vulnerabilities.items():
            lines.append(f"    {severity}: {count}件")
        lines.extend([
            "",
            "--- パッチ管理 ---",
            f"  パッチ適用率: {self.patch_compliance_rate:.1f}%",
            "",
            "--- インシデント対応 ---",
            f"  インシデント数: {len(self.incidents)}件",
            f"  平均対応時間(MTTR): {self.mean_time_to_respond}時間",
            "",
            "--- セキュリティ教育 ---",
            f"  研修完了率: {self.training_completion_rate:.1f}%",
        ])
        return "\n".join(lines)

# 使用例
metrics = SecurityMetrics()
metrics.open_vulnerabilities = {
    "critical": 2, "high": 8, "medium": 24, "low": 45
}
metrics.patch_compliance_rate = 94.5
metrics.training_completion_rate = 87.3

now = datetime.now()
metrics.add_incident(
    "high", now - timedelta(hours=12), now - timedelta(hours=4), "マルウェア"
)
metrics.add_incident(
    "medium", now - timedelta(hours=6), now - timedelta(hours=2), "不正アクセス"
)

print(metrics.generate_report())
```

---

## アンチパターン

### アンチパターン1: セキュリティ後付け（Bolt-on Security）

開発完了後にセキュリティを追加しようとするパターン。設計段階からセキュリティを組み込む「Security by Design」が正しいアプローチである。

```python
# NG: 後からセキュリティを追加
class UserServiceBad:
    def create_user(self, username: str, password: str) -> None:
        # パスワードを平文で保存してしまう
        self.db.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, password)
        )

# OK: 最初からセキュリティを組み込む
import bcrypt

class UserServiceGood:
    def create_user(self, username: str, password: str) -> None:
        # Step 1: パスワードポリシーの検証
        self._validate_password_policy(password)
        # Step 2: bcryptでハッシュ化（ソルト自動生成）
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12))
        # Step 3: パラメータ化クエリで安全に保存
        self.db.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, hashed.decode())
        )
        # Step 4: 監査ログの記録
        self.audit_log.record("CREATE_USER", username)

    def _validate_password_policy(self, password: str) -> None:
        """パスワードポリシーの検証"""
        if len(password) < 12:
            raise ValueError("パスワードは12文字以上必要です")
        if not any(c.isupper() for c in password):
            raise ValueError("大文字を含める必要があります")
        if not any(c.isdigit() for c in password):
            raise ValueError("数字を含める必要があります")
        if not any(c in "!@#$%^&*()-_+=[]{}|;:,.<>?" for c in password):
            raise ValueError("特殊文字を含める必要があります")
```

### アンチパターン2: 単一防御線への依存（Single Point of Security）

ファイアウォールだけ、暗号化だけ、WAFだけに頼るパターン。1つの防御が突破されると全てが失われる。

```python
# NG: ファイアウォールだけに依存
class InsecureApp:
    """FWが守ってくれるから大丈夫、という危険な思考"""
    def handle_request(self, user_input: str) -> str:
        # 入力バリデーションなし
        # SQLパラメータ化なし
        # 認証チェックなし
        query = f"SELECT * FROM data WHERE name = '{user_input}'"
        return self.db.execute(query)

# OK: 多層防御の実装
class SecureApp:
    """複数の防御層を実装"""
    def handle_request(self, user_input: str, auth_token: str) -> str:
        # Layer 1: 認証チェック
        user = self.auth.verify_token(auth_token)
        if not user:
            raise AuthenticationError("認証が必要です")

        # Layer 2: 認可チェック
        if not self.acl.check_permission(user, "read", "data"):
            raise AuthorizationError("権限がありません")

        # Layer 3: 入力バリデーション
        if not self.validator.is_safe(user_input):
            raise ValidationError("不正な入力です")

        # Layer 4: パラメータ化クエリ
        result = self.db.execute(
            "SELECT * FROM data WHERE name = ?", (user_input,)
        )

        # Layer 5: 出力フィルタリング
        return self.sanitize_output(result)
```

### アンチパターン3: セキュリティ・スルー・オブスキュリティ

秘密であることだけに依存するセキュリティ。カスタムの暗号アルゴリズム、推測しにくいURL、ソースコードの非公開だけに頼る考え方。

```python
# NG: 秘密のURLだけで保護
@app.route("/admin-secret-panel-xyz123")
def admin_panel_insecure():
    # URLが秘密だから大丈夫...は間違い
    return render_template("admin.html")

# OK: 適切な認証・認可 + 秘密のURL（補助的対策）
@app.route("/admin")
@require_authentication
@require_role("admin")
@require_mfa
def admin_panel_secure():
    return render_template("admin.html")
```

---

## 実践演習

### 演習1（基礎）: CIA三原則の分析

以下のセキュリティインシデントについて、CIA三原則のどれが脅かされているか分類し、適切な対策を提案してください。

1. ランサムウェアがファイルサーバーを暗号化し、業務データにアクセスできなくなった
2. 社員が顧客リストを個人のUSBドライブにコピーして持ち出した
3. 攻撃者がWebサイトの価格表を改ざんし、商品が100円で購入できるようになった
4. DDoS攻撃によりECサイトが3時間ダウンした
5. 開発者がGitHubの公開リポジトリにAPIキーをコミットした

<details>
<summary>模範解答</summary>

1. **ランサムウェアによるファイル暗号化**
   - 脅かされるCIA: **可用性**（データにアクセスできない）+ **機密性**（データが窃取される可能性）
   - 対策:
     - 定期的なオフラインバックアップ（3-2-1ルール）
     - EDR/EPPの導入
     - ネットワークセグメンテーション
     - ユーザー教育（フィッシング対策）
     - インシデント対応手順の策定

2. **顧客リストの持ち出し**
   - 脅かされるCIA: **機密性**（許可されていない者がデータにアクセス）
   - 対策:
     - DLP（Data Loss Prevention）の導入
     - USBポートの無効化/制限
     - ファイルアクセスの監査ログ
     - データ分類ポリシーの策定
     - 退職時のアクセス権限の即時無効化

3. **価格表の改ざん**
   - 脅かされるCIA: **完全性**（データが不正に変更された）
   - 対策:
     - Webアプリケーションの入力バリデーション
     - WAFの導入
     - データベースの変更監査ログ
     - 重要データの変更にはダブルチェック（4-eyes principle）
     - バージョン管理とロールバック機能

4. **DDoS攻撃によるダウン**
   - 脅かされるCIA: **可用性**（サービスが利用できない）
   - 対策:
     - CDNの利用（CloudFlare、AWS Shield等）
     - レートリミットの設定
     - オートスケーリングの設定
     - DDoS対策サービスの導入
     - 冗長構成（マルチリージョン）

5. **APIキーの公開**
   - 脅かされるCIA: **機密性**（秘密情報が公開された）
   - 対策:
     - pre-commitフックによるシークレット検出（git-secrets、truffleHog）
     - 環境変数やシークレット管理サービスの使用（AWS Secrets Manager等）
     - APIキーの即時ローテーション
     - GitHubのシークレットスキャン機能の有効化
     - .gitignoreの適切な設定
</details>

### 演習2（応用）: リスク評価ツールの実装

以下の要件を満たすリスク評価ツールを実装してください。

1. リスク項目を登録できる（名前、脅威レベル、脆弱性レベル、影響度、各1-5）
2. リスクスコアを自動計算する
3. リスクレベル（Critical/High/Medium/Low）を判定する
4. 優先順位順にソートしてレポートを出力する
5. 対策ステータス（未対策/対策中/対策済み）を管理できる

<details>
<summary>模範解答</summary>

```python
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
from datetime import datetime

class MitigationStatus(Enum):
    NOT_STARTED = "未対策"
    IN_PROGRESS = "対策中"
    COMPLETED = "対策済み"

@dataclass
class RiskEntry:
    """リスク評価エントリ"""
    name: str
    description: str
    threat_level: int         # 1-5
    vulnerability_level: int  # 1-5
    impact_level: int         # 1-5
    owner: str = ""
    mitigation_status: MitigationStatus = MitigationStatus.NOT_STARTED
    mitigation_plan: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        for name_field, value in [
            ("threat_level", self.threat_level),
            ("vulnerability_level", self.vulnerability_level),
            ("impact_level", self.impact_level),
        ]:
            if not 1 <= value <= 5:
                raise ValueError(f"{name_field}は1-5の範囲で指定: {value}")

    @property
    def risk_score(self) -> float:
        """リスクスコア（0-100に正規化）"""
        raw = self.threat_level * self.vulnerability_level * self.impact_level
        return round((raw / 125) * 100, 1)

    @property
    def risk_level(self) -> str:
        score = self.risk_score
        if score >= 60:
            return "Critical"
        elif score >= 40:
            return "High"
        elif score >= 20:
            return "Medium"
        return "Low"

class RiskAssessmentTool:
    """リスク評価ツール"""

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.risks: List[RiskEntry] = []

    def add_risk(self, risk: RiskEntry) -> None:
        """リスクを登録"""
        self.risks.append(risk)

    def update_status(self, name: str, status: MitigationStatus,
                      plan: str = "") -> bool:
        """対策ステータスを更新"""
        for risk in self.risks:
            if risk.name == name:
                risk.mitigation_status = status
                if plan:
                    risk.mitigation_plan = plan
                return True
        return False

    def get_sorted_risks(self) -> List[RiskEntry]:
        """リスクスコア順にソート"""
        return sorted(self.risks, key=lambda r: r.risk_score, reverse=True)

    def generate_report(self) -> str:
        """リスク評価レポートを生成"""
        lines = [
            f"=== リスク評価レポート: {self.project_name} ===",
            f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"登録リスク数: {len(self.risks)}",
            "",
            "--- リスク一覧（優先度順）---",
        ]

        for i, risk in enumerate(self.get_sorted_risks(), 1):
            lines.extend([
                f"\n{i}. {risk.name} [{risk.risk_level}]",
                f"   スコア: {risk.risk_score}/100",
                f"   脅威: {risk.threat_level}/5, "
                f"脆弱性: {risk.vulnerability_level}/5, "
                f"影響: {risk.impact_level}/5",
                f"   ステータス: {risk.mitigation_status.value}",
                f"   対策計画: {risk.mitigation_plan or '未設定'}",
            ])

        # サマリー
        by_level = {}
        for risk in self.risks:
            by_level[risk.risk_level] = by_level.get(risk.risk_level, 0) + 1

        lines.extend([
            "\n--- サマリー ---",
            f"  Critical: {by_level.get('Critical', 0)}件",
            f"  High: {by_level.get('High', 0)}件",
            f"  Medium: {by_level.get('Medium', 0)}件",
            f"  Low: {by_level.get('Low', 0)}件",
        ])

        return "\n".join(lines)

# 使用例
tool = RiskAssessmentTool("ECサイトv2.0")

tool.add_risk(RiskEntry(
    name="SQLインジェクション",
    description="検索機能の入力バリデーション不備",
    threat_level=4, vulnerability_level=5, impact_level=5,
    owner="開発チーム",
))
tool.add_risk(RiskEntry(
    name="DDoS攻撃",
    description="CDN未導入によるサービス停止リスク",
    threat_level=3, vulnerability_level=3, impact_level=4,
    owner="インフラチーム",
))
tool.add_risk(RiskEntry(
    name="弱いパスワードポリシー",
    description="6文字以上のみの要件",
    threat_level=4, vulnerability_level=4, impact_level=3,
    owner="セキュリティチーム",
))

tool.update_status("SQLインジェクション", MitigationStatus.IN_PROGRESS,
                   "ORMへの移行とパラメータ化クエリの強制")

print(tool.generate_report())
```
</details>

### 演習3（発展）: セキュリティフレームワーク適用計画

あなたの組織がクラウドベースのSaaSサービスを提供するスタートアップ（社員50名）だとします。顧客から「SOC 2 Type II 監査レポート」の提出を求められました。以下を策定してください。

1. SOC 2の5つのTrust Services Criteria（TSC）と自社の現状ギャップを分析する
2. 各TSCに対する具体的な対策を最低3つ提案する
3. 実装の優先順位とタイムライン（6ヶ月計画）を作成する

<details>
<summary>模範解答</summary>

**1. Trust Services Criteria と現状ギャップ分析**

| TSC | 基準 | 想定される現状 | ギャップ |
|-----|------|-------------|---------|
| Security（セキュリティ） | 不正アクセスからの保護 | 基本的なFW、パスワード認証あり | MFA未導入、IDSなし、脆弱性管理なし |
| Availability（可用性） | サービスの安定稼働 | 単一リージョン、手動復旧 | DR計画なし、SLA未定義、冗長化不足 |
| Processing Integrity（処理の完全性） | データ処理の正確性 | 基本的なバリデーションあり | 監査証跡不足、変更管理プロセスなし |
| Confidentiality（機密性） | 機密データの保護 | HTTPS通信 | データ分類なし、保存時暗号化なし、DLPなし |
| Privacy（プライバシー） | 個人情報の適切な管理 | プライバシーポリシーあり | データマッピング不足、同意管理不備 |

**2. 各TSCの対策（各3つ以上）**

Security:
- MFAの全社導入（Google Workspace + Okta）
- 脆弱性スキャナーの導入（Snyk + Trivy）
- セキュリティ意識教育プログラムの実施（四半期ごと）
- WAFの導入（AWS WAF）

Availability:
- マルチAZ構成への移行
- DR計画の策定とテスト
- SLA定義（99.9%目標）とステータスページ
- 自動スケーリングの設定

Processing Integrity:
- CI/CDパイプラインでの自動テスト
- データベース変更の監査ログ
- 変更管理プロセス（RFC）の導入

Confidentiality:
- データ分類ポリシーの策定と実施
- 保存時暗号化（AWS KMS）
- DLP の導入（Google Workspace DLP）

Privacy:
- データマッピングとRoPA作成
- プライバシー影響評価（PIA）の実施
- 同意管理プラットフォームの導入

**3. 6ヶ月タイムライン**

```
Month 1-2: 基盤整備
  - セキュリティポリシー策定
  - データ分類ポリシー策定
  - MFA導入（最優先）
  - 脆弱性スキャナー導入

Month 3-4: 技術的対策
  - 保存時暗号化の実装
  - WAF導入
  - マルチAZ移行
  - 監査ログの整備

Month 5-6: プロセスと監査準備
  - 変更管理プロセス導入
  - セキュリティ教育プログラム実施
  - DR計画策定とテスト
  - SOC 2 Type I の事前評価（Readiness Assessment）
```

この6ヶ月計画完了後、SOC 2 Type I 監査を受け、更に6-12ヶ月の運用実績を積んでType II 監査に臨む流れが標準的である。
</details>

---

## FAQ

### Q1: CIA三原則の中で最も重要なのはどれですか?

業種やシステムによって優先度は異なる。金融系では完全性（取引データの正確性）、医療系では可用性（緊急時のシステム利用）、個人情報を扱うシステムでは機密性が重視される傾向がある。重要なのは、自組織のコンテキストに応じてバランスを取ることであり、1つの要素だけに偏らないことである。

### Q2: セキュリティフレームワークはどれを選べばよいですか?

組織の規模、業種、規制要件によって異なる。以下が一般的な選択フローである:
- **まず NIST CSF** で全体像を把握（無料、柔軟）
- **国際取引がある場合**: ISO 27001 認証の取得を検討
- **クレジットカード処理**: PCI DSS 準拠は必須
- **SaaS事業者**: SOC 2 認証が顧客から求められることが多い
- **IT運用チーム向け**: CIS Controls で具体的な対策リストを参照

### Q3: リスク評価はどの頻度で行うべきですか?

最低でも年1回の定期的な見直しに加え、以下のタイミングでも実施すべきである:
- システムの大規模変更時
- 新規サービス・機能のリリース前
- 新しい脅威や脆弱性の発見時
- セキュリティインシデント発生後
- 組織の合併・買収時
- 法規制の変更時

### Q4: セキュリティ対策のROI（投資対効果）はどう計算しますか?

定量的には ALE（年間損失予測額）の削減額と対策コストを比較する。

```
ROI = (対策前ALE - 対策後ALE - 年間対策コスト) / 年間対策コスト x 100%

ALE = SLE(単一損失予測額) x ARO(年間発生率)

例: SQLインジェクション対策
  SLE: $500,000（データ漏洩時の損害額）
  ARO: 0.3（年間30%の確率で発生）
  対策前ALE: $150,000
  対策後ALE: $15,000（ARO を 0.03 に低減）
  年間対策コスト: $30,000
  ROI = (150,000 - 15,000 - 30,000) / 30,000 x 100% = 350%
```

### Q5: 小規模組織でもセキュリティフレームワークは必要ですか?

規模に関わらず、体系的なアプローチは必要である。ただし、小規模組織では以下のように簡略化できる:
- CIS Controls の Implementation Group 1（IG1: 基本的な56のセーフガード）から始める
- NIST CSF の Core を参考に、最低限の資産管理とリスク評価を実施
- 全てを完璧にする必要はなく、リスクベースで優先順位をつけて段階的に実施

---

## まとめ

| 項目 | 要点 |
|------|------|
| CIA三原則 | 機密性・完全性・可用性のバランスが情報セキュリティの基盤。業界により優先度は異なる |
| 拡張属性 | 真正性・責任追跡性・否認防止・信頼性も含めた包括的なセキュリティ確保が必要 |
| 脅威分類 | STRIDE等のフレームワークで体系的に脅威を特定・分類する |
| リスク評価 | 脅威 x 脆弱性 x 影響度でリスクを定量化し、対策の優先順位を決定する |
| フレームワーク | NIST CSF, ISO 27001, CIS Controls 等を組織の要件に応じて選択する |
| 多層防御 | 複数の独立した防御層を重ねて、単一障害点を排除する |
| SDL | 開発ライフサイクルの全フェーズにセキュリティ活動を統合する |
| メトリクス | MTTR、脆弱性密度等の指標でセキュリティの効果を継続的に測定する |

---

## 次に読むべきガイド

- [01-threat-modeling.md](./01-threat-modeling.md) -- STRIDE、DREAD、アタックツリーを用いた脅威モデリングの詳細
- [02-security-principles.md](./02-security-principles.md) -- 最小権限、ゼロトラスト、セキュアバイデフォルト等の設計原則
- [../01-web-security/00-owasp-top10.md](../01-web-security/00-owasp-top10.md) -- Webセキュリティの具体的な脆弱性と対策
- [../../authentication-and-authorization/docs/00-fundamentals/](../../authentication-and-authorization/docs/00-fundamentals/) -- 認証・認可の基礎（CIA三原則の機密性に直結）
- [../02-cryptography/](../02-cryptography/) -- 暗号技術の詳細（CIA三原則の機密性・完全性の実装手段）

---

## 参考文献

1. NIST Cybersecurity Framework 2.0 -- https://www.nist.gov/cyberframework
2. ISO/IEC 27001:2022 Information security management systems -- https://www.iso.org/standard/27001
3. CIS Critical Security Controls v8 -- https://www.cisecurity.org/controls
4. OWASP Foundation -- https://owasp.org/
5. IBM Cost of a Data Breach Report 2023 -- https://www.ibm.com/reports/data-breach
6. NIST SP 800-30 Guide for Conducting Risk Assessments -- https://csrc.nist.gov/publications/detail/sp/800-30/rev-1/final
7. Saltzer, J.H. & Schroeder, M.D., "The Protection of Information in Computer Systems" -- Proceedings of the IEEE, 1975
