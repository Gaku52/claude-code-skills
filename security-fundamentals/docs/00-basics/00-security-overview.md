# セキュリティ概要

> 情報セキュリティの根幹であるCIA三原則から脅威モデリング、リスク評価、主要フレームワークまでを体系的に解説する。

## この章で学ぶこと

1. **CIA三原則**（機密性・完全性・可用性）の意味と相互関係を理解する
2. **脅威モデリングとリスク評価**の基本プロセスを把握する
3. **主要セキュリティフレームワーク**（NIST CSF, ISO 27001 等）の特徴と使い分けを知る

---

## 1. 情報セキュリティとは何か

情報セキュリティとは、情報資産を**脅威**から保護し、事業継続性を確保するための活動全般を指す。技術的対策だけでなく、組織・人・プロセスを含む包括的な取り組みである。

```
+---------------------------------------------+
|           情報セキュリティの3要素              |
|                                             |
|   技術的対策   組織的対策   人的対策          |
|   (暗号化等)   (ポリシー)   (教育訓練)       |
+---------------------------------------------+
```

---

## 2. CIA三原則

情報セキュリティの最も基本的な原則は **CIA triad** と呼ばれる3つの要素で構成される。

```
              Confidentiality
              (機密性)
                 /\
                /  \
               /    \
              /  CIA  \
             /  Triad  \
            /          \
           /____________\
    Integrity          Availability
    (完全性)            (可用性)
```

### 2.1 機密性（Confidentiality）

許可された者だけが情報にアクセスできる状態を保つこと。

```python
# コード例1: アクセス制御による機密性の確保
import hashlib
import hmac

class AccessControl:
    """ロールベースアクセス制御(RBAC)の基本実装"""

    def __init__(self):
        self.permissions = {
            "admin": ["read", "write", "delete", "manage_users"],
            "editor": ["read", "write"],
            "viewer": ["read"],
        }

    def check_permission(self, role: str, action: str) -> bool:
        """ユーザーのロールに基づいてアクションの許可を判定する"""
        allowed = self.permissions.get(role, [])
        if action in allowed:
            return True
        # 監査ログに拒否を記録
        log_access_denied(role, action)
        return False

    def encrypt_sensitive_data(self, data: bytes, key: bytes) -> bytes:
        """機密データを暗号化して保存する"""
        from cryptography.fernet import Fernet
        f = Fernet(key)
        return f.encrypt(data)
```

### 2.2 完全性（Integrity）

情報が不正に改ざんされていないことを保証すること。

```python
# コード例2: ハッシュによるデータ完全性の検証
import hashlib
import json

class IntegrityChecker:
    """データの完全性をSHA-256ハッシュで検証する"""

    @staticmethod
    def compute_hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def verify_integrity(data: bytes, expected_hash: str) -> bool:
        actual_hash = hashlib.sha256(data).hexdigest()
        # タイミング攻撃を防ぐためhmac.compare_digestを使用
        return hmac.compare_digest(actual_hash, expected_hash)

# 使用例
original_data = b'{"user_id": 1, "balance": 10000}'
stored_hash = IntegrityChecker.compute_hash(original_data)

# 改ざん検出
tampered_data = b'{"user_id": 1, "balance": 99999}'
print(IntegrityChecker.verify_integrity(tampered_data, stored_hash))
# => False（改ざんを検出）
```

### 2.3 可用性（Availability）

必要なときに情報やサービスが利用可能であること。

```python
# コード例3: ヘルスチェックと冗長性による可用性確保
import time
from typing import List

class HealthMonitor:
    """サービスの可用性を監視する"""

    def __init__(self, endpoints: List[str], timeout: int = 5):
        self.endpoints = endpoints
        self.timeout = timeout
        self.status = {ep: "unknown" for ep in endpoints}

    def check_health(self, endpoint: str) -> dict:
        """エンドポイントの稼働状態を確認する"""
        import requests
        try:
            start = time.time()
            response = requests.get(
                f"{endpoint}/health",
                timeout=self.timeout
            )
            latency = time.time() - start
            return {
                "status": "healthy" if response.status_code == 200 else "degraded",
                "latency_ms": round(latency * 1000, 2),
                "status_code": response.status_code,
            }
        except requests.exceptions.RequestException as e:
            return {"status": "unhealthy", "error": str(e)}

    def failover(self, primary: str, secondary: str) -> str:
        """プライマリが停止した場合にセカンダリへフェイルオーバーする"""
        primary_health = self.check_health(primary)
        if primary_health["status"] == "healthy":
            return primary
        # フェイルオーバー実行とアラート送信
        alert(f"Failover triggered: {primary} -> {secondary}")
        return secondary
```

### CIA三原則の相互関係

| 原則 | 脅威例 | 対策例 | 失敗時の影響 |
|------|--------|--------|-------------|
| 機密性 | 不正アクセス、盗聴 | 暗号化、アクセス制御 | 情報漏洩、プライバシー侵害 |
| 完全性 | データ改ざん、MITM | ハッシュ、デジタル署名 | 不正取引、信頼性喪失 |
| 可用性 | DDoS、障害 | 冗長構成、CDN | サービス停止、売上損失 |

---

## 3. 追加のセキュリティ属性

CIA三原則に加えて、以下の属性も重要視される。

```
+-------------------------------------------------------------------+
|                  拡張セキュリティ属性                                |
|                                                                   |
|  +----------+  +----------+  +----------+  +----------+          |
|  |真正性    |  |責任追跡性|  |否認防止  |  |信頼性    |          |
|  |Authen-   |  |Account-  |  |Non-      |  |Relia-    |          |
|  |ticity    |  |ability   |  |repudia-  |  |bility    |          |
|  |          |  |          |  |tion      |  |          |          |
|  +----------+  +----------+  +----------+  +----------+          |
+-------------------------------------------------------------------+
```

| 属性 | 説明 | 実現手段 |
|------|------|---------|
| 真正性（Authenticity） | 情報の送信元が本物であることの確認 | デジタル署名、PKI |
| 責任追跡性（Accountability） | 行為者を特定・追跡できること | 監査ログ、アクセスログ |
| 否認防止（Non-repudiation） | 行為を後から否定できないこと | タイムスタンプ、電子署名 |
| 信頼性（Reliability） | システムが一貫して期待通りに動作すること | テスト、冗長設計 |

---

## 4. 脅威モデリングの概要

脅威モデリングとは、システムに対する潜在的な脅威を体系的に特定・分析するプロセスである。

```python
# コード例4: 簡易的な脅威モデリングツール
from dataclasses import dataclass, field
from enum import Enum
from typing import List

class ThreatCategory(Enum):
    SPOOFING = "なりすまし"
    TAMPERING = "改ざん"
    REPUDIATION = "否認"
    INFORMATION_DISCLOSURE = "情報漏洩"
    DENIAL_OF_SERVICE = "サービス妨害"
    ELEVATION_OF_PRIVILEGE = "権限昇格"

@dataclass
class Threat:
    name: str
    category: ThreatCategory
    description: str
    likelihood: int  # 1-5
    impact: int      # 1-5
    mitigations: List[str] = field(default_factory=list)

    @property
    def risk_score(self) -> int:
        """リスクスコア = 発生可能性 x 影響度"""
        return self.likelihood * self.impact

    @property
    def risk_level(self) -> str:
        score = self.risk_score
        if score >= 20:
            return "Critical"
        elif score >= 12:
            return "High"
        elif score >= 6:
            return "Medium"
        return "Low"

# 脅威の定義例
sql_injection = Threat(
    name="SQLインジェクション",
    category=ThreatCategory.TAMPERING,
    description="ユーザー入力を介してSQL文を注入し、データベースを不正操作する",
    likelihood=4,
    impact=5,
    mitigations=[
        "パラメータ化クエリの使用",
        "入力バリデーション",
        "WAFの導入",
    ],
)
print(f"Risk: {sql_injection.risk_level} (Score: {sql_injection.risk_score})")
# => Risk: Critical (Score: 20)
```

---

## 5. リスク評価

リスク評価は「何を守るべきか」「どの脅威が最も危険か」を定量的・定性的に判断するプロセスである。

```python
# コード例5: リスクマトリクスの生成
def generate_risk_matrix():
    """5x5のリスクマトリクスを生成する"""
    impact_labels = ["軽微", "小", "中", "大", "甚大"]
    likelihood_labels = ["稀", "低", "中", "高", "極高"]

    matrix = []
    for l_idx, likelihood in enumerate(likelihood_labels, 1):
        row = []
        for i_idx, impact in enumerate(impact_labels, 1):
            score = l_idx * i_idx
            if score >= 20:
                level = "CRITICAL"
            elif score >= 12:
                level = "HIGH"
            elif score >= 6:
                level = "MEDIUM"
            else:
                level = "LOW"
            row.append(f"{score:2d}({level[:1]})")
        matrix.append((likelihood, row))

    # 表示
    print("       | " + " | ".join(f"{i:>7s}" for i in impact_labels))
    print("-" * 60)
    for label, row in reversed(matrix):
        print(f"{label:>5s}  | " + " | ".join(f"{v:>7s}" for v in row))

generate_risk_matrix()
```

出力イメージ:

```
       |    軽微 |      小 |      中 |      大 |    甚大
------------------------------------------------------------
  極高  |  5(L) | 10(M) | 15(H) | 20(C) | 25(C)
    高  |  4(L) |  8(M) | 12(H) | 16(H) | 20(C)
    中  |  3(L) |  6(M) |  9(M) | 12(H) | 15(H)
    低  |  2(L) |  4(L) |  6(M) |  8(M) | 10(M)
    稀  |  1(L) |  2(L) |  3(L) |  4(L) |  5(L)
```

---

## 6. セキュリティフレームワーク

### 6.1 主要フレームワーク比較

| フレームワーク | 発行元 | 特徴 | 対象 |
|---------------|--------|------|------|
| NIST CSF | 米国NIST | 5つの機能（特定・防御・検知・対応・復旧）、柔軟性が高い | 全業種 |
| ISO 27001 | ISO/IEC | 認証取得可能、ISMS構築のための要求事項 | 全業種（特に国際取引） |
| CIS Controls | CIS | 優先順位付きの具体的対策リスト18項目 | IT運用チーム |
| SOC 2 | AICPA | SaaS企業向け、Trust Services Criteria | クラウドサービス提供者 |
| PCI DSS | PCI SSC | クレジットカード情報の保護要件 | 決済関連事業者 |

### 6.2 NIST サイバーセキュリティフレームワーク

```
+----------+    +----------+    +----------+
|  特定    | -> |  防御    | -> |  検知    |
| Identify |    | Protect  |    | Detect   |
+----------+    +----------+    +----------+
                                      |
+----------+    +----------+          |
|  復旧    | <- |  対応    | <--------+
| Recover  |    | Respond  |
+----------+    +----------+
```

各機能の概要:

| 機能 | 目的 | 主な活動 |
|------|------|---------|
| 特定（Identify） | 資産・リスクの把握 | 資産管理、リスク評価、ガバナンス |
| 防御（Protect） | 脅威からの保護 | アクセス制御、暗号化、教育訓練 |
| 検知（Detect） | 異常の検出 | 監視、ログ分析、侵入検知 |
| 対応（Respond） | インシデント対応 | 分析、封じ込め、通知 |
| 復旧（Recover） | 事業の復旧 | 復旧計画、改善活動、広報 |

---

## 7. セキュリティの実践的アプローチ

### 7.1 多層防御（Defense in Depth）

```
+--------------------------------------------------+
|  物理セキュリティ（入退室管理、施錠）               |
|  +--------------------------------------------+  |
|  |  ネットワーク（FW、IDS/IPS、セグメンテーション）|  |
|  |  +--------------------------------------+  |  |
|  |  |  ホスト（OS強化、AV、パッチ管理）      |  |  |
|  |  |  +--------------------------------+  |  |  |
|  |  |  |  アプリ（入力検証、認証、暗号化） |  |  |  |
|  |  |  |  +----------------------------+|  |  |  |
|  |  |  |  |  データ（暗号化、分類、DLP） ||  |  |  |
|  |  |  |  +----------------------------+|  |  |  |
|  |  |  +--------------------------------+  |  |  |
|  |  +--------------------------------------+  |  |
|  +--------------------------------------------+  |
+--------------------------------------------------+
```

---

## アンチパターン

### アンチパターン1: セキュリティ後付け（Bolt-on Security）

開発完了後にセキュリティを追加しようとするパターン。設計段階からセキュリティを組み込む「Security by Design」が正しいアプローチである。

```python
# 悪い例: 後からセキュリティを追加
class UserService:
    def create_user(self, username, password):
        # パスワードを平文で保存してしまう
        db.execute("INSERT INTO users VALUES (?, ?)", (username, password))

# 良い例: 最初からセキュリティを組み込む
class UserService:
    def create_user(self, username, password):
        # パスワードポリシーの検証
        self._validate_password_policy(password)
        # bcryptでハッシュ化してから保存
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        db.execute("INSERT INTO users VALUES (?, ?)", (username, hashed))
```

### アンチパターン2: 単一防御線への依存（Single Point of Security）

ファイアウォールだけ、または暗号化だけに頼るパターン。1つの防御が突破されると全てが失われる。多層防御を徹底すること。

---

## FAQ

### Q1: CIA三原則の中で最も重要なのはどれですか?

業種やシステムによって優先度は異なる。金融系では完全性、医療系では可用性、個人情報を扱うシステムでは機密性が重視される傾向がある。重要なのは、自組織のコンテキストに応じてバランスを取ることである。

### Q2: セキュリティフレームワークはどれを選べばよいですか?

組織の規模、業種、規制要件によって異なる。まずNIST CSFで全体像を把握し、必要に応じてISO 27001の認証取得やPCI DSS準拠を検討するのが一般的なアプローチである。

### Q3: リスク評価はどの頻度で行うべきですか?

最低でも年1回の定期的な見直しに加え、システム変更時、新規脅威の発見時、インシデント発生後にも実施すべきである。継続的なリスク評価が理想的だが、現実的にはイベントドリブンで補完する。

---

## まとめ

| 項目 | 要点 |
|------|------|
| CIA三原則 | 機密性・完全性・可用性のバランスが情報セキュリティの基盤 |
| 脅威モデリング | 体系的に脅威を特定し、リスクを定量化するプロセス |
| リスク評価 | 発生可能性と影響度からリスクレベルを判定し、対策の優先順位を決定 |
| フレームワーク | NIST CSF, ISO 27001, CIS Controls 等を組織の要件に応じて選択 |
| 多層防御 | 複数の防御層を重ねて、単一障害点を排除する |

---

## 次に読むべきガイド

- [01-threat-modeling.md](./01-threat-modeling.md) -- 脅威モデリングの詳細な手法と実践手順
- [02-security-principles.md](./02-security-principles.md) -- 最小権限、ゼロトラストなどの設計原則
- [../01-web-security/00-owasp-top10.md](../01-web-security/00-owasp-top10.md) -- Webセキュリティの具体的な脆弱性と対策

---

## 参考文献

1. NIST Cybersecurity Framework 2.0 -- https://www.nist.gov/cyberframework
2. ISO/IEC 27001:2022 Information security management systems -- https://www.iso.org/standard/27001
3. CIS Critical Security Controls v8 -- https://www.cisecurity.org/controls
4. OWASP Foundation -- https://owasp.org/
