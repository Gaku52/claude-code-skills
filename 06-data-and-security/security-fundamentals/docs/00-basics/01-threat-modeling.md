# 脅威モデリング

> システムに潜む脅威を体系的に洗い出すためのSTRIDE、DREAD、アタックツリー、データフロー図を用いた実践的手法を解説する。

## この章で学ぶこと

1. **STRIDE** モデルを使った脅威の分類と特定方法を理解する
2. **DREAD** スコアリングによるリスクの定量的評価手法を習得する
3. **アタックツリーとデータフロー図** を用いた脅威分析の実践手順を身につける

---

## 1. 脅威モデリングとは

脅威モデリングとは、設計段階でシステムに対する潜在的な攻撃を体系的に特定・評価し、適切な対策を計画するプロセスである。開発の早期段階で実施することで、修正コストを大幅に削減できる。

```
脅威モデリングの基本プロセス:

+----------+    +----------+    +----------+    +----------+
| 1.対象の  | -> | 2.脅威の  | -> | 3.リスク  | -> | 4.対策の  |
| 分解     |    | 特定     |    | 評価     |    | 決定     |
| (DFD作成) |    | (STRIDE) |    | (DREAD)  |    | (緩和策) |
+----------+    +----------+    +----------+    +----------+
      |                                               |
      +-----------------------------------------------+
                     反復的に改善
```

---

## 2. STRIDE モデル

MicrosoftがSDLで開発した脅威分類フレームワーク。各カテゴリは情報セキュリティの属性に対応する。

### 2.1 STRIDEの6カテゴリ

| カテゴリ | 英語名 | 脅かされる属性 | 攻撃例 |
|---------|--------|---------------|--------|
| S: なりすまし | Spoofing | 真正性 | 偽のログインページ、セッションハイジャック |
| T: 改ざん | Tampering | 完全性 | SQLインジェクション、パラメータ改ざん |
| R: 否認 | Repudiation | 否認防止 | ログの消去、証跡のない操作 |
| I: 情報漏洩 | Information Disclosure | 機密性 | ディレクトリリスティング、エラーメッセージ |
| D: サービス妨害 | Denial of Service | 可用性 | DDoS、リソース枯渇攻撃 |
| E: 権限昇格 | Elevation of Privilege | 認可 | 水平/垂直権限昇格、バッファオーバーフロー |

```
STRIDE と セキュリティ属性の対応:

  S ──> 真正性 (Authentication)
  T ──> 完全性 (Integrity)
  R ──> 否認防止 (Non-repudiation)
  I ──> 機密性 (Confidentiality)
  D ──> 可用性 (Availability)
  E ──> 認可 (Authorization)
```

### 2.2 STRIDE分析の実装

```python
# コード例1: STRIDE分析ツール
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

class StrideCategory(Enum):
    SPOOFING = "S: なりすまし"
    TAMPERING = "T: 改ざん"
    REPUDIATION = "R: 否認"
    INFO_DISCLOSURE = "I: 情報漏洩"
    DENIAL_OF_SERVICE = "D: サービス妨害"
    ELEVATION_OF_PRIVILEGE = "E: 権限昇格"

@dataclass
class StrideThreat:
    category: StrideCategory
    target_component: str
    description: str
    attack_vector: str
    mitigations: List[str] = field(default_factory=list)
    dread_score: Optional[int] = None

class StrideAnalyzer:
    """STRIDE分析を実行するクラス"""

    def __init__(self, system_name: str):
        self.system_name = system_name
        self.threats: List[StrideThreat] = []

    def add_threat(self, threat: StrideThreat) -> None:
        self.threats.append(threat)

    def analyze_component(self, component: str) -> List[StrideThreat]:
        """特定コンポーネントに対するSTRIDE全カテゴリの脅威を検討"""
        prompts = {
            StrideCategory.SPOOFING:
                f"{component}に対してなりすましは可能か?",
            StrideCategory.TAMPERING:
                f"{component}のデータを改ざんできるか?",
            StrideCategory.REPUDIATION:
                f"{component}の操作を否認できるか?",
            StrideCategory.INFO_DISCLOSURE:
                f"{component}から情報が漏洩する経路はあるか?",
            StrideCategory.DENIAL_OF_SERVICE:
                f"{component}のサービスを妨害できるか?",
            StrideCategory.ELEVATION_OF_PRIVILEGE:
                f"{component}で権限昇格は可能か?",
        }
        # 各カテゴリについて脅威を検討
        for category, question in prompts.items():
            print(f"  検討: {question}")
        return [t for t in self.threats if t.target_component == component]

    def generate_report(self) -> str:
        """分析レポートを生成する"""
        lines = [f"# STRIDE分析レポート: {self.system_name}\n"]
        for cat in StrideCategory:
            cat_threats = [t for t in self.threats if t.category == cat]
            lines.append(f"\n## {cat.value} ({len(cat_threats)}件)")
            for t in cat_threats:
                lines.append(f"- [{t.target_component}] {t.description}")
                for m in t.mitigations:
                    lines.append(f"  - 緩和策: {m}")
        return "\n".join(lines)

# 使用例
analyzer = StrideAnalyzer("Eコマースサイト")
analyzer.add_threat(StrideThreat(
    category=StrideCategory.SPOOFING,
    target_component="ログインAPI",
    description="ブルートフォースによるアカウント乗っ取り",
    attack_vector="POST /api/login に大量の認証試行",
    mitigations=["レートリミット", "アカウントロックアウト", "MFA導入"],
))
```

---

## 3. DREADスコアリング

DREADはリスクを定量的に評価するためのスコアリングモデルである。

### 3.1 DREAD の5つの評価軸

| 軸 | 英語名 | 説明 | スコア範囲 |
|----|--------|------|-----------|
| D | Damage | 被害の大きさ | 1-10 |
| R | Reproducibility | 再現性の高さ | 1-10 |
| E | Exploitability | 攻撃の容易さ | 1-10 |
| A | Affected Users | 影響を受けるユーザー数 | 1-10 |
| D | Discoverability | 脆弱性の発見しやすさ | 1-10 |

```python
# コード例2: DREADスコアの計算
from dataclasses import dataclass

@dataclass
class DreadScore:
    damage: int           # 1-10: 被害の大きさ
    reproducibility: int  # 1-10: 再現性
    exploitability: int   # 1-10: 攻撃の容易さ
    affected_users: int   # 1-10: 影響範囲
    discoverability: int  # 1-10: 発見可能性

    def __post_init__(self):
        for field_name, value in self.__dict__.items():
            if not 1 <= value <= 10:
                raise ValueError(f"{field_name}は1-10の範囲で指定: {value}")

    @property
    def total(self) -> float:
        return (self.damage + self.reproducibility +
                self.exploitability + self.affected_users +
                self.discoverability) / 5.0

    @property
    def risk_level(self) -> str:
        score = self.total
        if score >= 8:
            return "Critical"
        elif score >= 6:
            return "High"
        elif score >= 4:
            return "Medium"
        return "Low"

# SQLインジェクションのDREADスコア
sqli_dread = DreadScore(
    damage=9,            # DB全体が危険にさらされる
    reproducibility=10,  # 100%再現可能
    exploitability=7,    # 自動化ツールが豊富
    affected_users=10,   # 全ユーザーが影響
    discoverability=8,   # スキャナーで容易に発見
)
print(f"SQLi DREAD Score: {sqli_dread.total} ({sqli_dread.risk_level})")
# => SQLi DREAD Score: 8.8 (Critical)
```

---

## 4. アタックツリー

攻撃目標を根（ルート）とし、達成手段を木構造で分解する手法。

```
              [ECサイトから顧客情報を窃取する]
                    /           \
                   /             \
     [Webアプリ経由]          [内部者経由]
        /      \                /      \
       /        \              /        \
  [SQLi]    [XSS->     [DB直接    [バックアップ
             セッション  アクセス]   ファイル窃取]
             窃取]
    /  \        |           |           |
[検索] [ログ   [Stored   [認証情報  [暗号化なし
 フォーム イン  XSSで     のハード    のバックアップ
 経由]  フォーム Cookie    コーディ    をUSBで持出]
        経由]  窃取]     ング]
```

```python
# コード例3: アタックツリーの構築
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class AttackNode:
    """アタックツリーのノード"""
    name: str
    description: str
    cost: int = 0          # 攻撃コスト（低いほど容易）
    difficulty: str = "Medium"  # Low/Medium/High
    children: List['AttackNode'] = field(default_factory=list)
    is_and: bool = False   # True=AND条件, False=OR条件

    def add_child(self, child: 'AttackNode') -> None:
        self.children.append(child)

    def min_cost(self) -> int:
        """最小攻撃コストを再帰的に計算する"""
        if not self.children:
            return self.cost
        child_costs = [c.min_cost() for c in self.children]
        if self.is_and:
            return sum(child_costs)  # AND: 全子ノードが必要
        return min(child_costs)      # OR: 最も安い経路

    def display(self, indent: int = 0) -> None:
        """ツリーを表示する"""
        connector = "AND" if self.is_and else "OR"
        prefix = "  " * indent
        print(f"{prefix}[{self.name}] (cost={self.cost}, {self.difficulty})")
        if self.children:
            print(f"{prefix}  ({connector})")
            for child in self.children:
                child.display(indent + 2)

# アタックツリーの構築例
root = AttackNode("顧客情報の窃取", "ECサイトの顧客データを不正に取得する")
web_attack = AttackNode("Webアプリ経由", "アプリケーション脆弱性を利用")
sqli = AttackNode("SQLインジェクション", "入力フォーム経由", cost=3, difficulty="Low")
xss = AttackNode("XSSでセッション窃取", "Stored XSS", cost=5, difficulty="Medium")
web_attack.add_child(sqli)
web_attack.add_child(xss)
root.add_child(web_attack)
root.display()
```

---

## 5. データフロー図（DFD）

DFDはシステムを通るデータの流れを可視化し、信頼境界（Trust Boundary）を明示する図である。

```
DFD レベル0（コンテキスト図）:

+----------+                              +----------+
|          |  --- HTTPリクエスト --->      |          |
|  ユーザー |                              | Webアプリ |
| (外部)   |  <--- HTTPレスポンス ---      | ケーション |
+----------+                              +----------+
                                               |
                 信頼境界                        |
  ===================================          |
                                               v
                                          +----------+
                                          |          |
                                          |  DB      |
                                          |          |
                                          +----------+


DFD レベル1（詳細図）:

                    信頼境界
  ==========================================
  |                                        |
  |  +-------+    +--------+    +------+   |
  |  | Web   | -> | API    | -> |  DB  |   |
  |  | Server|    | Server |    |      |   |
  |  +-------+    +--------+    +------+   |
  |      |             |                   |
  |      v             v                   |
  |  +-------+    +--------+              |
  |  | Static|    | Cache  |              |
  |  | Files |    | (Redis)|              |
  |  +-------+    +--------+              |
  ==========================================
```

```python
# コード例4: データフロー図の信頼境界分析
from dataclasses import dataclass, field
from typing import List, Set

@dataclass
class DataFlow:
    source: str
    destination: str
    data_type: str
    protocol: str
    encrypted: bool
    crosses_trust_boundary: bool

@dataclass
class TrustBoundary:
    name: str
    components: Set[str] = field(default_factory=set)

class DFDAnalyzer:
    """DFDに基づく脅威分析"""

    def __init__(self):
        self.flows: List[DataFlow] = []
        self.boundaries: List[TrustBoundary] = []

    def add_flow(self, flow: DataFlow) -> None:
        self.flows.append(flow)

    def add_boundary(self, boundary: TrustBoundary) -> None:
        self.boundaries.append(boundary)

    def find_risky_flows(self) -> List[DataFlow]:
        """信頼境界を越える非暗号化フローを検出する"""
        risky = []
        for flow in self.flows:
            if flow.crosses_trust_boundary and not flow.encrypted:
                risky.append(flow)
        return risky

    def analyze(self) -> dict:
        """全フローを分析してリスクサマリーを生成"""
        risky = self.find_risky_flows()
        return {
            "total_flows": len(self.flows),
            "boundary_crossing_flows": sum(
                1 for f in self.flows if f.crosses_trust_boundary
            ),
            "unencrypted_boundary_crossings": len(risky),
            "risky_flows": [
                f"{f.source} -> {f.destination} ({f.data_type})"
                for f in risky
            ],
        }

# 使用例
analyzer = DFDAnalyzer()
analyzer.add_flow(DataFlow(
    source="ブラウザ",
    destination="Webサーバー",
    data_type="認証情報",
    protocol="HTTPS",
    encrypted=True,
    crosses_trust_boundary=True,
))
analyzer.add_flow(DataFlow(
    source="APIサーバー",
    destination="データベース",
    data_type="クエリ",
    protocol="TCP",
    encrypted=False,  # 内部通信で暗号化なし -> リスク
    crosses_trust_boundary=True,
))
result = analyzer.analyze()
print(result)
```

---

## 6. 脅威モデリングの実践手順

### 6.1 4段階プロセス

```
Step 1              Step 2              Step 3              Step 4
+-------------+     +-------------+     +-------------+     +-------------+
| 何を作るか?  |     | 何が問題に   |     | どう対処    |     | 十分か?     |
|             | --> | なり得るか?  | --> | するか?    | --> |             |
| - DFD作成   |     | - STRIDE    |     | - 緩和      |     | - テスト    |
| - 資産特定   |     | - アタック   |     | - 転嫁      |     | - レビュー  |
| - 境界定義   |     |   ツリー    |     | - 受容      |     | - 検証     |
+-------------+     +-------------+     | - 回避      |     +-------------+
                                        +-------------+
```

```python
# コード例5: 脅威モデリングワークシートの自動生成
import json
from datetime import datetime
from typing import List, Dict

class ThreatModelWorksheet:
    """脅威モデリングワークシートの生成・管理"""

    def __init__(self, project: str, version: str):
        self.project = project
        self.version = version
        self.created = datetime.now().isoformat()
        self.components: List[Dict] = []
        self.threats: List[Dict] = []
        self.mitigations: List[Dict] = []

    def add_component(self, name: str, component_type: str,
                      trust_level: str) -> None:
        self.components.append({
            "name": name,
            "type": component_type,  # process/datastore/external/dataflow
            "trust_level": trust_level,  # trusted/untrusted/semi-trusted
        })

    def add_threat(self, stride_cat: str, component: str,
                   description: str, dread: dict) -> str:
        threat_id = f"T-{len(self.threats) + 1:03d}"
        self.threats.append({
            "id": threat_id,
            "stride_category": stride_cat,
            "component": component,
            "description": description,
            "dread_score": dread,
            "status": "identified",
        })
        return threat_id

    def add_mitigation(self, threat_id: str, strategy: str,
                       description: str) -> None:
        self.mitigations.append({
            "threat_id": threat_id,
            "strategy": strategy,  # mitigate/transfer/accept/avoid
            "description": description,
            "implemented": False,
        })

    def export(self) -> str:
        return json.dumps({
            "project": self.project,
            "version": self.version,
            "created": self.created,
            "components": self.components,
            "threats": self.threats,
            "mitigations": self.mitigations,
            "summary": {
                "total_threats": len(self.threats),
                "mitigated": sum(
                    1 for m in self.mitigations if m["implemented"]
                ),
                "pending": sum(
                    1 for m in self.mitigations if not m["implemented"]
                ),
            },
        }, indent=2, ensure_ascii=False)

# 使用例
ws = ThreatModelWorksheet("ECサイト", "v2.0")
ws.add_component("フロントエンド", "process", "untrusted")
ws.add_component("APIサーバー", "process", "trusted")
ws.add_component("PostgreSQL", "datastore", "trusted")

t1 = ws.add_threat(
    "Tampering", "APIサーバー",
    "SQLインジェクションによるデータ改ざん",
    {"D": 9, "R": 10, "E": 7, "A": 10, "D2": 8}
)
ws.add_mitigation(t1, "mitigate", "パラメータ化クエリの強制使用")
print(ws.export())
```

---

## アンチパターン

### アンチパターン1: 脅威モデリングの省略

「リリースが間に合わない」という理由で脅威モデリングを省略するパターン。本番環境でのセキュリティインシデント対応コストは、設計段階での脅威分析コストの数十倍から数百倍になる。最低限のSTRIDE分析だけでも実施すべきである。

### アンチパターン2: 一度きりの脅威モデリング

初回リリース時にだけ脅威モデリングを行い、その後更新しないパターン。システムは進化し、新しいコンポーネントや機能が追加されるたびに脅威も変化する。CI/CDパイプラインに脅威モデルレビューを組み込み、定期的に更新する必要がある。

### アンチパターン3: 網羅性の過度な追求

すべての脅威を100%洗い出そうとして分析が終わらないパターン。リスクベースで優先順位をつけ、高リスクの脅威から順に対処する実践的アプローチが重要である。

---

## FAQ

### Q1: 脅威モデリングは誰が行うべきですか?

開発チーム全体で取り組むべきである。セキュリティ専門家だけでなく、開発者・アーキテクト・運用担当者が参加することで、各視点からの脅威を漏れなく特定できる。特にアーキテクトはシステム全体の構造を把握しているため、信頼境界の定義に不可欠である。

### Q2: STRIDEとDREADはどちらを使うべきですか?

両方を組み合わせて使う。STRIDEは脅威の「分類と特定」に、DREADは特定した脅威の「優先順位付け」に使う。STRIDEで洗い出した脅威にDREADスコアを付けることで、対策の優先順位が明確になる。

### Q3: 小規模プロジェクトでも脅威モデリングは必要ですか?

規模に応じて簡易化すればよい。最低限、(1) DFDの作成、(2) 信頼境界の特定、(3) STRIDEの各カテゴリについて1つずつ脅威を検討する――この30分程度の作業でも、セキュリティ意識が大幅に向上する。

---

## まとめ

| 項目 | 要点 |
|------|------|
| STRIDE | 6カテゴリで脅威を体系的に分類・特定する手法 |
| DREAD | 5軸のスコアリングで脅威のリスクを定量評価する手法 |
| アタックツリー | 攻撃目標を木構造で分解し、最小コスト経路を特定する |
| DFD | データの流れと信頼境界を可視化し、脅威の発生箇所を特定する |
| 実践手順 | 分解→特定→評価→対策の4段階を反復的に実施する |

---

## 次に読むべきガイド

- [02-security-principles.md](./02-security-principles.md) -- 最小権限やゼロトラスト等の設計原則
- [../01-web-security/00-owasp-top10.md](../01-web-security/00-owasp-top10.md) -- 具体的なWeb脆弱性のSTRIDE分類
- [../04-application-security/00-secure-coding.md](../04-application-security/00-secure-coding.md) -- 脅威への対策としてのセキュアコーディング

---

## 参考文献

1. Adam Shostack, "Threat Modeling: Designing for Security" -- Wiley, 2014
2. Microsoft SDL Threat Modeling Tool -- https://www.microsoft.com/en-us/securityengineering/sdl/threatmodeling
3. OWASP Threat Modeling -- https://owasp.org/www-community/Threat_Modeling
4. Bruce Schneier, "Attack Trees" -- Dr. Dobb's Journal, 1999
