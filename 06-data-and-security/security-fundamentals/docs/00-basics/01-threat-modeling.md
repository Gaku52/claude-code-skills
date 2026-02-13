# 脅威モデリング

> システムに潜む脅威を体系的に洗い出すためのSTRIDE、DREAD、PASTA、アタックツリー、データフロー図、キルチェーン分析を用いた実践的手法を解説する。設計段階でのセキュリティ組み込みにより、修正コストを数十倍削減する脅威モデリングの全体像を習得する。

## この章で学ぶこと

1. **STRIDE** モデルを使った脅威の体系的分類と特定方法を理解する
2. **DREAD** スコアリングおよび **CVSS** によるリスクの定量的評価手法を習得する
3. **アタックツリーとデータフロー図（DFD）** を用いた脅威分析の実践手順を身につける
4. **脅威モデリングの組織導入**と継続的改善プロセスを計画できるようになる

## 前提知識

- [00-security-overview.md](./00-security-overview.md) -- CIA三原則とリスク評価の基礎
- ソフトウェアアーキテクチャの基本（クライアント・サーバー構成、API設計の概要）
- [../../authentication-and-authorization/docs/00-fundamentals/](../../authentication-and-authorization/docs/00-fundamentals/) -- 認証・認可の基本概念

---

## 1. 脅威モデリングとは

脅威モデリングとは、設計段階でシステムに対する潜在的な攻撃を体系的に特定・評価し、適切な対策を計画するプロセスである。

### WHY: なぜ脅威モデリングが必要か

```
修正コストの法則（NIST/IBM研究）:

  設計段階での修正     :  $100       (1x)
  実装段階での修正     :  $1,000     (10x)
  テスト段階での修正   :  $10,000    (100x)
  本番運用中の修正     :  $100,000+  (1000x)

  => 設計段階での脅威モデリングは最もコスト効率が高いセキュリティ活動
```

### 1.1 脅威モデリングの基本プロセス

```
脅威モデリングの4つのフェーズ:

+----------+    +----------+    +----------+    +----------+
| 1.対象の  | -> | 2.脅威の  | -> | 3.リスク  | -> | 4.対策の  |
| 分解     |    | 特定     |    | 評価     |    | 決定     |
| (DFD作成) |    | (STRIDE) |    | (DREAD)  |    | (緩和策) |
+----------+    +----------+    +----------+    +----------+
      |                                               |
      +-----------------------------------------------+
                     反復的に改善

Phase 1: 何を作るのか?（What are we building?）
  → DFD作成、資産特定、信頼境界の定義

Phase 2: 何が問題になり得るか?（What can go wrong?）
  → STRIDE、PASTA、アタックツリー分析

Phase 3: どれくらい危険か?（How bad is it?）
  → DREAD、CVSS、リスクマトリクス

Phase 4: どう対処するか?（What are we going to do about it?）
  → 緩和（Mitigate）、転嫁（Transfer）、受容（Accept）、回避（Avoid）
```

### 1.2 脅威モデリングの4つの質問（Adam Shostack の手法）

| 質問 | 目的 | 主なツール/手法 |
|------|------|----------------|
| 何を作っているか? | システムの理解と資産の特定 | DFD、アーキテクチャ図 |
| 何がうまくいかない可能性があるか? | 脅威の特定 | STRIDE、PASTA、キルチェーン |
| それについて何をするか? | 対策の決定 | 緩和・転嫁・受容・回避 |
| 十分な対策を行ったか? | 検証 | テスト、レビュー、ペネトレーションテスト |

---

## 2. STRIDE モデル

MicrosoftがSDL（Security Development Lifecycle）の一部として開発した脅威分類フレームワーク。各カテゴリは情報セキュリティの属性に1対1で対応する。

### 2.1 STRIDEの6カテゴリ

```
STRIDE と セキュリティ属性の対応関係:

  脅威カテゴリ          脅かされる属性         主な対策
  ─────────────────────────────────────────────────────
  S: Spoofing       →  Authentication     →  MFA, PKI
  T: Tampering      →  Integrity          →  ハッシュ, 署名
  R: Repudiation    →  Non-repudiation    →  監査ログ, 署名
  I: Info Disclosure→  Confidentiality    →  暗号化, ACL
  D: DoS            →  Availability       →  CDN, レートリミット
  E: Elevation      →  Authorization      →  最小権限, RBAC
```

| カテゴリ | 英語名 | 脅かされる属性 | 攻撃例 | DFD要素との対応 |
|---------|--------|---------------|--------|---------------|
| S: なりすまし | Spoofing | 真正性 | 偽ログインページ、セッションハイジャック、ARP spoofing | 外部エンティティ、プロセス |
| T: 改ざん | Tampering | 完全性 | SQLインジェクション、パラメータ改ざん、MITM | データフロー、データストア |
| R: 否認 | Repudiation | 否認防止 | ログの消去、証跡のない操作、タイムスタンプ偽装 | プロセス |
| I: 情報漏洩 | Information Disclosure | 機密性 | ディレクトリリスティング、エラーメッセージ、サイドチャネル | データフロー、データストア |
| D: サービス妨害 | Denial of Service | 可用性 | DDoS、リソース枯渇、アプリケーション層DoS | プロセス、データストア |
| E: 権限昇格 | Elevation of Privilege | 認可 | 水平/垂直権限昇格、バッファオーバーフロー、IDOR | プロセス |

### 2.2 STRIDE-per-Element 分析

DFDの各要素に対して、該当するSTRIDEカテゴリの脅威を検討する手法。

| DFD要素 | S | T | R | I | D | E |
|---------|---|---|---|---|---|---|
| 外部エンティティ | o | | | | | |
| プロセス | o | o | o | o | o | o |
| データフロー | | o | | o | o | |
| データストア | | o | o | o | o | |

```python
# コード例1: STRIDE-per-Element 分析ツール
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Set

class StrideCategory(Enum):
    SPOOFING = "S: なりすまし"
    TAMPERING = "T: 改ざん"
    REPUDIATION = "R: 否認"
    INFO_DISCLOSURE = "I: 情報漏洩"
    DENIAL_OF_SERVICE = "D: サービス妨害"
    ELEVATION_OF_PRIVILEGE = "E: 権限昇格"

class DFDElementType(Enum):
    EXTERNAL_ENTITY = "外部エンティティ"
    PROCESS = "プロセス"
    DATA_FLOW = "データフロー"
    DATA_STORE = "データストア"

# DFD要素ごとに該当するSTRIDEカテゴリを定義
STRIDE_PER_ELEMENT: Dict[DFDElementType, Set[StrideCategory]] = {
    DFDElementType.EXTERNAL_ENTITY: {
        StrideCategory.SPOOFING,
    },
    DFDElementType.PROCESS: {
        StrideCategory.SPOOFING,
        StrideCategory.TAMPERING,
        StrideCategory.REPUDIATION,
        StrideCategory.INFO_DISCLOSURE,
        StrideCategory.DENIAL_OF_SERVICE,
        StrideCategory.ELEVATION_OF_PRIVILEGE,
    },
    DFDElementType.DATA_FLOW: {
        StrideCategory.TAMPERING,
        StrideCategory.INFO_DISCLOSURE,
        StrideCategory.DENIAL_OF_SERVICE,
    },
    DFDElementType.DATA_STORE: {
        StrideCategory.TAMPERING,
        StrideCategory.REPUDIATION,
        StrideCategory.INFO_DISCLOSURE,
        StrideCategory.DENIAL_OF_SERVICE,
    },
}

@dataclass
class StrideThreat:
    """特定された脅威"""
    id: str
    category: StrideCategory
    element_name: str
    element_type: DFDElementType
    description: str
    attack_vector: str
    mitigations: List[str] = field(default_factory=list)
    priority: str = "未評価"  # Critical/High/Medium/Low/未評価

@dataclass
class StrideAnalyzer:
    """STRIDE-per-Element分析エンジン"""

    system_name: str
    threats: List[StrideThreat] = field(default_factory=list)
    _threat_counter: int = 0

    def analyze_element(self, name: str, element_type: DFDElementType,
                        description: str = "") -> List[str]:
        """DFD要素に対してSTRIDE分析を実行し、検討すべき質問を返す"""
        applicable = STRIDE_PER_ELEMENT.get(element_type, set())
        questions = []

        for category in applicable:
            question = self._generate_question(category, name, element_type)
            questions.append(question)

        return questions

    def _generate_question(self, category: StrideCategory,
                           element_name: str,
                           element_type: DFDElementType) -> str:
        """脅威カテゴリに基づいた分析質問を生成"""
        templates = {
            StrideCategory.SPOOFING:
                f"[{element_name}] に対して、なりすましは可能か？"
                f"正当な{element_type.value}であることをどう検証するか？",
            StrideCategory.TAMPERING:
                f"[{element_name}] のデータを攻撃者が改ざんできるか？"
                f"改ざんをどう検出するか？",
            StrideCategory.REPUDIATION:
                f"[{element_name}] の操作を後から否認できるか？"
                f"証跡は十分に記録されているか？",
            StrideCategory.INFO_DISCLOSURE:
                f"[{element_name}] から機密情報が漏洩する経路はあるか？"
                f"データは適切に暗号化されているか？",
            StrideCategory.DENIAL_OF_SERVICE:
                f"[{element_name}] のサービスを妨害する方法はあるか？"
                f"リソース制限は設定されているか？",
            StrideCategory.ELEVATION_OF_PRIVILEGE:
                f"[{element_name}] で権限昇格は可能か？"
                f"最小権限の原則は適用されているか？",
        }
        return templates.get(category, "")

    def add_threat(self, category: StrideCategory, element_name: str,
                   element_type: DFDElementType, description: str,
                   attack_vector: str,
                   mitigations: Optional[List[str]] = None) -> str:
        """脅威を登録"""
        self._threat_counter += 1
        threat_id = f"T-{self._threat_counter:03d}"
        threat = StrideThreat(
            id=threat_id,
            category=category,
            element_name=element_name,
            element_type=element_type,
            description=description,
            attack_vector=attack_vector,
            mitigations=mitigations or [],
        )
        self.threats.append(threat)
        return threat_id

    def generate_report(self) -> str:
        """分析レポートを生成"""
        lines = [
            f"# STRIDE分析レポート: {self.system_name}",
            f"## 脅威総数: {len(self.threats)}件\n",
        ]
        for cat in StrideCategory:
            cat_threats = [t for t in self.threats if t.category == cat]
            lines.append(f"### {cat.value} ({len(cat_threats)}件)")
            if not cat_threats:
                lines.append("  脅威なし\n")
                continue
            for t in cat_threats:
                lines.append(f"- **[{t.id}]** [{t.element_name}] {t.description}")
                lines.append(f"  - 攻撃ベクトル: {t.attack_vector}")
                lines.append(f"  - 優先度: {t.priority}")
                for m in t.mitigations:
                    lines.append(f"  - 緩和策: {m}")
            lines.append("")
        return "\n".join(lines)

# 使用例
analyzer = StrideAnalyzer("ECサイト決済システム")

# DFD要素に対するSTRIDE分析質問の生成
questions = analyzer.analyze_element(
    "決済API", DFDElementType.PROCESS, "クレジットカード決済処理"
)
for q in questions:
    print(f"  検討: {q}")

# 脅威の登録
analyzer.add_threat(
    StrideCategory.SPOOFING, "ログインAPI", DFDElementType.PROCESS,
    "ブルートフォースによるアカウント乗っ取り",
    "POST /api/login に大量の認証試行",
    ["レートリミット(5回/分)", "アカウントロックアウト(10分)", "MFA導入"],
)
analyzer.add_threat(
    StrideCategory.TAMPERING, "注文データフロー", DFDElementType.DATA_FLOW,
    "価格パラメータの改ざんによる不正な低額購入",
    "MITM/プロキシによるリクエストボディの改ざん",
    ["サーバーサイドでの価格再計算", "リクエスト署名検証", "TLS必須化"],
)
analyzer.add_threat(
    StrideCategory.INFO_DISCLOSURE, "ユーザーDB", DFDElementType.DATA_STORE,
    "SQLインジェクションによる全顧客データ漏洩",
    "検索フォームに ' OR 1=1 -- を入力",
    ["パラメータ化クエリ", "WAF導入", "DB最小権限"],
)

print(analyzer.generate_report())
```

---

## 3. DREADスコアリング

DREADはMicrosoftが開発したリスク定量評価モデルである。特定された脅威に優先順位をつけるために使用する。

### 3.1 DREAD の5つの評価軸

```
DREADスコアリングの構造:

  D ─ Damage（被害の大きさ）
  │   1: 軽微な影響  →  10: システム全体の壊滅
  │
  R ─ Reproducibility（再現性）
  │   1: 特殊条件下でのみ  →  10: 誰でも100%再現可能
  │
  E ─ Exploitability（攻撃の容易さ）
  │   1: 高度な専門知識が必要  →  10: 自動化ツールで即座に攻撃可能
  │
  A ─ Affected Users（影響を受けるユーザー数）
  │   1: 単一ユーザー  →  10: 全ユーザー
  │
  D ─ Discoverability（発見しやすさ）
      1: 発見が極めて困難  →  10: 公開情報から容易に発見

  合計スコア = (D + R + E + A + D) / 5
  → 8-10: Critical  → 6-7.9: High  → 4-5.9: Medium  → 1-3.9: Low
```

| 軸 | 英語名 | 説明 | 低（1-3） | 中（4-6） | 高（7-10） |
|----|--------|------|----------|----------|-----------|
| D | Damage | 被害の大きさ | ログ汚染程度 | 一部データの漏洩 | 全データ漏洩/システム破壊 |
| R | Reproducibility | 再現性 | 特定条件のみ | 時々再現 | 常に100%再現 |
| E | Exploitability | 攻撃の容易さ | 高度なスキル要 | ツール使用で可能 | 初心者でも自動ツールで可能 |
| A | Affected Users | 影響範囲 | 個人のみ | 一部のユーザー | 全ユーザー/管理者含む |
| D | Discoverability | 発見可能性 | 内部情報必要 | 推測可能 | 公開情報/自動スキャン |

```python
# コード例2: DREAD + CVSSハイブリッドスコアリング
from dataclasses import dataclass
from typing import Optional

@dataclass
class DreadScore:
    """DREADリスクスコアの計算"""
    damage: int           # 1-10: 被害の大きさ
    reproducibility: int  # 1-10: 再現性
    exploitability: int   # 1-10: 攻撃の容易さ
    affected_users: int   # 1-10: 影響範囲
    discoverability: int  # 1-10: 発見可能性

    def __post_init__(self):
        fields = {
            "damage": self.damage,
            "reproducibility": self.reproducibility,
            "exploitability": self.exploitability,
            "affected_users": self.affected_users,
            "discoverability": self.discoverability,
        }
        for name, value in fields.items():
            if not 1 <= value <= 10:
                raise ValueError(f"{name}は1-10の範囲: {value}")

    @property
    def total(self) -> float:
        """DREADスコア（平均値）"""
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

    @property
    def risk_color(self) -> str:
        """リスクレベルに対応する色コード"""
        colors = {
            "Critical": "RED",
            "High": "ORANGE",
            "Medium": "YELLOW",
            "Low": "GREEN",
        }
        return colors[self.risk_level]

    def breakdown(self) -> str:
        """スコアの内訳を表示"""
        items = [
            ("Damage", self.damage),
            ("Reproducibility", self.reproducibility),
            ("Exploitability", self.exploitability),
            ("Affected Users", self.affected_users),
            ("Discoverability", self.discoverability),
        ]
        lines = []
        for name, value in items:
            bar = "█" * value + "░" * (10 - value)
            lines.append(f"  {name:<20s} [{bar}] {value}/10")
        lines.append(f"  {'Total':<20s}  => {self.total:.1f}/10 ({self.risk_level})")
        return "\n".join(lines)

# 代表的な脅威のDREADスコア比較
threats = {
    "SQLインジェクション": DreadScore(9, 10, 7, 10, 8),
    "XSS (Stored)": DreadScore(7, 9, 6, 8, 7),
    "CSRF": DreadScore(6, 8, 5, 7, 6),
    "ブルートフォース": DreadScore(5, 10, 8, 3, 9),
    "DDoS": DreadScore(4, 10, 9, 10, 10),
}

print("=== 脅威 DREAD スコア比較 ===\n")
for name, score in sorted(
    threats.items(), key=lambda x: x[1].total, reverse=True
):
    print(f"--- {name} ---")
    print(score.breakdown())
    print()
```

### 3.2 DREAD vs CVSS 比較

| 特性 | DREAD | CVSS v3.1 |
|------|-------|-----------|
| 開発元 | Microsoft | FIRST |
| スコア範囲 | 1-10 | 0.0-10.0 |
| 評価軸 | 5軸（D, R, E, A, D） | 基本/現状/環境 |
| 主観性 | やや高い | 低い（標準化） |
| 学習コスト | 低い | 中～高い |
| 業界標準 | 非公式 | CVE採番で公式 |
| 適用場面 | 社内脅威モデリング | 脆弱性管理、パッチ優先度 |

---

## 4. アタックツリー

攻撃目標を根（ルート）とし、達成手段を木構造で分解する手法。Bruce Schneierが1999年に提唱した。

### 4.1 アタックツリーの基本構造

```
アタックツリーの例: ECサイトから顧客情報を窃取する

              [ECサイトから顧客情報を窃取する] (Root)
                    /              \
                   /                \
     [Webアプリ経由]             [内部者経由]
     (OR: いずれかで成功)        (OR: いずれかで成功)
        /       \                  /       \
       /         \                /         \
  [SQLi]     [XSS→           [DB直接     [バックアップ
   ($3K)      セッション       アクセス]    ファイル
              窃取]            ($10K)      窃取]
              ($5K)                       ($2K)
    /  \        |               |            |
[検索] [ログ   [Stored         [認証情報   [暗号化なし
フォーム イン   XSSで           のハード    のバックアップ
経由]  フォーム Cookie          コーディ    をUSBで持出]
       経由]   窃取]           ング]

AND条件: 複数の子ノードすべてが必要
OR条件: いずれかの子ノードで達成可能
コスト: 各リーフノードの攻撃コスト（低いほど攻撃が容易）
```

```python
# コード例3: アタックツリーの構築と最小コスト経路分析
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class AttackNode:
    """アタックツリーのノード"""
    name: str
    description: str = ""
    cost: int = 0             # 攻撃コスト（低いほど容易）
    difficulty: str = "Medium"  # Low/Medium/High
    probability: float = 0.5  # 成功確率（0.0-1.0）
    children: List['AttackNode'] = field(default_factory=list)
    is_and: bool = False      # True=AND条件, False=OR条件
    mitigations: List[str] = field(default_factory=list)

    def add_child(self, child: 'AttackNode') -> 'AttackNode':
        self.children.append(child)
        return self  # メソッドチェーン

    def min_cost(self) -> int:
        """最小攻撃コストを再帰的に計算"""
        if not self.children:
            return self.cost
        child_costs = [c.min_cost() for c in self.children]
        if self.is_and:
            return sum(child_costs)  # AND: 全子ノードのコスト合計
        return min(child_costs)      # OR: 最安経路

    def max_probability(self) -> float:
        """最大成功確率を再帰的に計算"""
        if not self.children:
            return self.probability
        child_probs = [c.max_probability() for c in self.children]
        if self.is_and:
            # AND: 全子ノードの確率の積
            result = 1.0
            for p in child_probs:
                result *= p
            return result
        return max(child_probs)  # OR: 最高確率の経路

    def find_cheapest_path(self) -> List[str]:
        """最もコストの低い攻撃経路を特定"""
        if not self.children:
            return [self.name]

        if self.is_and:
            # AND: すべての子を含む
            path = [self.name]
            for child in self.children:
                path.extend(child.find_cheapest_path())
            return path

        # OR: 最安の子を選択
        cheapest = min(self.children, key=lambda c: c.min_cost())
        return [self.name] + cheapest.find_cheapest_path()

    def display(self, indent: int = 0) -> None:
        """ツリーを視覚的に表示"""
        prefix = "  " * indent
        cost_str = f"${self.cost}K" if self.cost else ""
        prob_str = f"P={self.probability:.0%}" if not self.children else ""
        connector = "AND" if self.is_and else "OR"

        print(f"{prefix}[{self.name}] {cost_str} {prob_str}")
        if self.mitigations:
            for m in self.mitigations:
                print(f"{prefix}  >> 対策: {m}")
        if self.children:
            print(f"{prefix}  ({connector})")
            for child in self.children:
                child.display(indent + 2)

# アタックツリーの構築例
root = AttackNode("顧客情報の窃取", "ECサイトの顧客データを不正取得")

# Webアプリ経由
web_attack = AttackNode("Webアプリ経由", "アプリ脆弱性を利用")
sqli = AttackNode("SQLインジェクション", "入力フォーム経由",
                  cost=3, difficulty="Low", probability=0.7,
                  mitigations=["パラメータ化クエリ", "WAF", "入力バリデーション"])
xss = AttackNode("XSS→セッション窃取", "Stored XSSでCookie窃取",
                 cost=5, difficulty="Medium", probability=0.4,
                 mitigations=["CSP", "HttpOnly Cookie", "出力エスケープ"])
web_attack.add_child(sqli)
web_attack.add_child(xss)

# 内部者経由
insider = AttackNode("内部者経由", "組織内部からの攻撃")
db_direct = AttackNode("DB直接アクセス", "認証情報の不正利用",
                       cost=10, difficulty="High", probability=0.2,
                       mitigations=["最小権限", "特権アクセス管理", "監査ログ"])
backup_theft = AttackNode("バックアップ窃取", "暗号化なしバックアップ",
                          cost=2, difficulty="Low", probability=0.3,
                          mitigations=["バックアップ暗号化", "アクセス制限", "DLP"])
insider.add_child(db_direct)
insider.add_child(backup_theft)

root.add_child(web_attack)
root.add_child(insider)

# 分析結果
root.display()
print(f"\n最小攻撃コスト: ${root.min_cost()}K")
print(f"最大成功確率: {root.max_probability():.0%}")
print(f"最安経路: {' → '.join(root.find_cheapest_path())}")
```

---

## 5. データフロー図（DFD）と信頼境界

DFDはシステムを通るデータの流れを可視化し、信頼境界（Trust Boundary）を明示する図である。脅威モデリングにおける最も重要なツールである。

### 5.1 DFDの構成要素

```
DFD の4つの構成要素:

  +---------+       外部エンティティ（External Entity）
  | ユーザー |       システム外部のアクター
  +---------+

  (  処理  )         プロセス（Process）
  ( サーバー )       データを処理するコンポーネント

  ===========        データフロー（Data Flow）
  ==========>        データの移動方向を示す矢印

  [[  DB   ]]        データストア（Data Store）
  [[ 保存先 ]]       データの保管場所

  ────────────       信頼境界（Trust Boundary）
  ┃ 境界線  ┃       セキュリティレベルが変わる境界
  ────────────
```

### 5.2 DFDレベル

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
                                          | DB       |
                                          +----------+


DFD レベル1（詳細図）:

+--------+                   信頼境界
| ブラウザ |    ==========================================
| (外部)  |    |                                        |
+--------+    |  +-------+    +--------+    +------+   |
     |         |  | Web   | -> | API    | -> |  DB  |   |
     +-------->|  | Server|    | Server |    |      |   |
               |  +-------+    +--------+    +------+   |
               |      |             |            |      |
               |      v             v            |      |
               |  +-------+    +--------+        |      |
               |  | CDN/  |    | Cache  |        |      |
               |  | Static|    | (Redis)|        |      |
               |  +-------+    +--------+        |      |
               ==========================================
                                    |
                         信頼境界    |
                    ================|============
                                    v
                              +---------+
                              | 外部API  |
                              | (決済等) |
                              +---------+
```

```python
# コード例4: DFD解析と信頼境界を越えるフローの脅威自動検出
from dataclasses import dataclass, field
from typing import List, Set, Dict

@dataclass
class DFDComponent:
    """DFDのコンポーネント"""
    name: str
    component_type: str  # external_entity, process, data_store
    trust_zone: str       # untrusted, dmz, trusted, highly_trusted
    description: str = ""

@dataclass
class DataFlow:
    """データフロー"""
    source: str
    destination: str
    data_type: str
    protocol: str
    encrypted: bool
    authenticated: bool
    data_classification: str = "internal"  # public, internal, confidential, restricted

@dataclass
class TrustBoundary:
    """信頼境界"""
    name: str
    trust_level: str  # untrusted, dmz, trusted, highly_trusted
    components: Set[str] = field(default_factory=set)

class DFDThreatAnalyzer:
    """DFDに基づく自動脅威検出エンジン"""

    TRUST_HIERARCHY = ["untrusted", "dmz", "trusted", "highly_trusted"]

    def __init__(self, system_name: str):
        self.system_name = system_name
        self.components: Dict[str, DFDComponent] = {}
        self.flows: List[DataFlow] = []
        self.boundaries: List[TrustBoundary] = []

    def add_component(self, component: DFDComponent) -> None:
        self.components[component.name] = component

    def add_flow(self, flow: DataFlow) -> None:
        self.flows.append(flow)

    def add_boundary(self, boundary: TrustBoundary) -> None:
        self.boundaries.append(boundary)

    def _get_trust_level(self, component_name: str) -> str:
        """コンポーネントの信頼レベルを取得"""
        comp = self.components.get(component_name)
        return comp.trust_zone if comp else "untrusted"

    def _crosses_boundary(self, flow: DataFlow) -> bool:
        """フローが信頼境界を越えるか判定"""
        src_trust = self._get_trust_level(flow.source)
        dst_trust = self._get_trust_level(flow.destination)
        return src_trust != dst_trust

    def _trust_direction(self, flow: DataFlow) -> str:
        """信頼方向を判定（inbound/outbound/lateral）"""
        src_idx = self.TRUST_HIERARCHY.index(
            self._get_trust_level(flow.source)
        )
        dst_idx = self.TRUST_HIERARCHY.index(
            self._get_trust_level(flow.destination)
        )
        if src_idx < dst_idx:
            return "inbound"   # 低信頼→高信頼（要注意）
        elif src_idx > dst_idx:
            return "outbound"  # 高信頼→低信頼（情報漏洩リスク）
        return "lateral"

    def detect_threats(self) -> List[Dict]:
        """脅威を自動検出"""
        threats = []

        for flow in self.flows:
            flow_threats = []

            # 脅威1: 信頼境界を越える非暗号化フロー
            if self._crosses_boundary(flow) and not flow.encrypted:
                flow_threats.append({
                    "type": "TAMPERING/INFO_DISCLOSURE",
                    "severity": "High",
                    "description": (
                        f"信頼境界を越えるフロー "
                        f"({flow.source} → {flow.destination}) "
                        f"が暗号化されていない"
                    ),
                    "mitigation": "TLS/mTLS の導入",
                })

            # 脅威2: 認証なしの境界越えフロー
            if self._crosses_boundary(flow) and not flow.authenticated:
                direction = self._trust_direction(flow)
                if direction == "inbound":
                    flow_threats.append({
                        "type": "SPOOFING",
                        "severity": "Critical",
                        "description": (
                            f"低信頼ゾーンから高信頼ゾーンへの"
                            f"認証なしフロー "
                            f"({flow.source} → {flow.destination})"
                        ),
                        "mitigation": "認証メカニズムの実装",
                    })

            # 脅威3: 機密データの非暗号化フロー
            if flow.data_classification in ("confidential", "restricted"):
                if not flow.encrypted:
                    flow_threats.append({
                        "type": "INFO_DISCLOSURE",
                        "severity": "Critical",
                        "description": (
                            f"機密データ({flow.data_type}) が "
                            f"暗号化なしで転送されている"
                        ),
                        "mitigation": "転送時暗号化(TLS)と保存時暗号化の実装",
                    })

            for t in flow_threats:
                t["flow"] = f"{flow.source} → {flow.destination}"
                t["data_type"] = flow.data_type
                threats.append(t)

        return threats

    def generate_report(self) -> str:
        """脅威分析レポートを生成"""
        threats = self.detect_threats()
        lines = [
            f"# DFD脅威分析レポート: {self.system_name}",
            f"## コンポーネント数: {len(self.components)}",
            f"## データフロー数: {len(self.flows)}",
            f"## 検出された脅威: {len(threats)}件\n",
        ]

        by_severity = {}
        for t in threats:
            by_severity.setdefault(t["severity"], []).append(t)

        for severity in ["Critical", "High", "Medium", "Low"]:
            sev_threats = by_severity.get(severity, [])
            if sev_threats:
                lines.append(f"### {severity} ({len(sev_threats)}件)")
                for t in sev_threats:
                    lines.append(f"- [{t['type']}] {t['description']}")
                    lines.append(f"  フロー: {t['flow']}")
                    lines.append(f"  対策: {t['mitigation']}")
                lines.append("")

        return "\n".join(lines)

# 使用例
analyzer = DFDThreatAnalyzer("ECサイト")

# コンポーネント定義
analyzer.add_component(DFDComponent(
    "ブラウザ", "external_entity", "untrusted", "ユーザーのWebブラウザ"
))
analyzer.add_component(DFDComponent(
    "APIサーバー", "process", "trusted", "バックエンドAPI"
))
analyzer.add_component(DFDComponent(
    "データベース", "data_store", "highly_trusted", "PostgreSQL"
))

# データフロー定義
analyzer.add_flow(DataFlow(
    "ブラウザ", "APIサーバー", "認証情報",
    "HTTPS", encrypted=True, authenticated=True,
    data_classification="confidential",
))
analyzer.add_flow(DataFlow(
    "APIサーバー", "データベース", "SQLクエリ",
    "TCP", encrypted=False, authenticated=True,
    data_classification="internal",
))

threats = analyzer.detect_threats()
print(analyzer.generate_report())
```

---

## 6. PASTA（Process for Attack Simulation and Threat Analysis）

PASTAは7段階のリスク中心の脅威モデリング手法で、ビジネス目標とテクニカルな脅威分析を結合する。

### 6.1 PASTAの7段階

```
PASTA 7段階プロセス:

  Stage 1: ビジネス目標の定義
  +-------------------+
  | 事業要件の整理     |
  | リスク許容度の決定  |
  +--------+----------+
           |
  Stage 2: 技術スコープの定義
  +--------v----------+
  | インフラ図の作成    |
  | 技術仕様の整理     |
  +--------+----------+
           |
  Stage 3: アプリケーション分解
  +--------v----------+
  | DFDの作成          |
  | 信頼境界の定義     |
  +--------+----------+
           |
  Stage 4: 脅威分析
  +--------v----------+
  | 脅威情報の収集     |
  | 脅威ライブラリ参照  |
  +--------+----------+
           |
  Stage 5: 脆弱性分析
  +--------v----------+
  | CVE/CWE分析       |
  | スキャン結果統合    |
  +--------+----------+
           |
  Stage 6: 攻撃シミュレーション
  +--------v----------+
  | アタックツリー作成  |
  | 攻撃シナリオ実行   |
  +--------+----------+
           |
  Stage 7: リスク管理
  +--------v----------+
  | 残余リスクの評価    |
  | 対策の優先順位決定  |
  +-------------------+
```

### STRIDE vs PASTA 比較

| 特性 | STRIDE | PASTA |
|------|--------|-------|
| アプローチ | 技術中心 | リスク中心 |
| 段階数 | 明確な段階なし | 7段階 |
| ビジネス観点 | 弱い | 強い（Stage 1） |
| 攻撃シミュレーション | なし | あり（Stage 6） |
| 学習コスト | 低い | 中～高い |
| 適用規模 | 小～中規模 | 中～大規模 |
| 出力 | 脅威リスト | リスク管理計画 |

---

## 7. サイバーキルチェーン分析

Lockheed Martinが提唱した攻撃の段階モデル。脅威モデリングにおいて、攻撃者の行動パターンを理解するために使用する。

```
サイバーキルチェーンの7段階:

  1. 偵察          2. 武器化        3. 配送
  +----------+    +----------+    +----------+
  | 情報収集  |    | 攻撃ツール|    | マルウェア|
  | OSINT    | -> | 作成     | -> | 送付     |
  | スキャン  |    | ペイロード|    | フィッシン|
  +----------+    +----------+    | グ      |
                                  +----------+
                                       |
  7. 目的実行      6. C2通信        5. インストール  4. 攻撃実行
  +----------+    +----------+    +----------+    +----------+
  | データ窃取|    | 遠隔操作  |    | 永続化    |    | 脆弱性   |
  | 破壊     | <- | 横展開   | <- | バックドア| <- | 悪用     |
  | 暗号化   |    | 特権昇格  |    | 設置     |    | 実行     |
  +----------+    +----------+    +----------+    +----------+

各段階で検知・遮断のチャンスがある（多層防御の適用点）
```

---

## 8. 脅威モデリングの実践ワークフロー

```python
# コード例5: 脅威モデリングワークシートの完全実装
import json
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class ThreatModelWorksheet:
    """脅威モデリングワークシートの生成・管理"""

    project: str
    version: str
    author: str = ""
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    components: List[Dict] = field(default_factory=list)
    threats: List[Dict] = field(default_factory=list)
    mitigations: List[Dict] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    scope: str = ""

    def add_component(self, name: str, component_type: str,
                      trust_level: str, description: str = "") -> None:
        """DFDコンポーネントを追加"""
        self.components.append({
            "name": name,
            "type": component_type,
            "trust_level": trust_level,
            "description": description,
        })

    def add_threat(self, stride_cat: str, component: str,
                   description: str, attack_scenario: str,
                   dread: Optional[Dict] = None) -> str:
        """脅威を追加"""
        threat_id = f"T-{len(self.threats) + 1:03d}"
        dread_score = 0
        if dread:
            dread_score = sum(dread.values()) / len(dread)
        self.threats.append({
            "id": threat_id,
            "stride_category": stride_cat,
            "component": component,
            "description": description,
            "attack_scenario": attack_scenario,
            "dread_score": round(dread_score, 1) if dread else None,
            "dread_detail": dread,
            "status": "identified",
        })
        return threat_id

    def add_mitigation(self, threat_id: str, strategy: str,
                       description: str, effort: str = "medium",
                       assigned_to: str = "") -> None:
        """対策を追加"""
        self.mitigations.append({
            "threat_id": threat_id,
            "strategy": strategy,
            "description": description,
            "effort": effort,
            "assigned_to": assigned_to,
            "implemented": False,
        })

    def get_summary(self) -> Dict:
        """サマリーを生成"""
        severity_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        for t in self.threats:
            score = t.get("dread_score", 0) or 0
            if score >= 8:
                severity_counts["Critical"] += 1
            elif score >= 6:
                severity_counts["High"] += 1
            elif score >= 4:
                severity_counts["Medium"] += 1
            else:
                severity_counts["Low"] += 1

        return {
            "project": self.project,
            "version": self.version,
            "total_threats": len(self.threats),
            "severity_breakdown": severity_counts,
            "total_mitigations": len(self.mitigations),
            "mitigated": sum(1 for m in self.mitigations if m["implemented"]),
            "pending": sum(1 for m in self.mitigations if not m["implemented"]),
            "coverage": (
                f"{sum(1 for m in self.mitigations if m['implemented'])}"
                f"/{len(self.mitigations)}"
            ),
        }

    def export_json(self) -> str:
        """JSON形式でエクスポート"""
        return json.dumps({
            "project": self.project,
            "version": self.version,
            "author": self.author,
            "created": self.created,
            "scope": self.scope,
            "assumptions": self.assumptions,
            "components": self.components,
            "threats": self.threats,
            "mitigations": self.mitigations,
            "summary": self.get_summary(),
        }, indent=2, ensure_ascii=False)

# 使用例
ws = ThreatModelWorksheet(
    project="ECサイト決済システム",
    version="v2.0",
    author="セキュリティチーム",
    scope="決済フロー（カート → 注文確定 → 決済処理 → 完了通知）",
)
ws.assumptions = [
    "外部決済ゲートウェイ（Stripe）は信頼できるものとする",
    "TLS 1.3が全通信で使用されている",
    "管理者は信頼できるが、最小権限の原則は適用する",
]

ws.add_component("ブラウザ", "external_entity", "untrusted", "顧客のブラウザ")
ws.add_component("APIサーバー", "process", "trusted", "Node.js Express")
ws.add_component("PostgreSQL", "data_store", "highly_trusted", "注文・顧客DB")
ws.add_component("Stripe API", "external_entity", "semi-trusted", "決済処理")

t1 = ws.add_threat(
    "Tampering", "APIサーバー",
    "注文金額の改ざんによる不正な低額決済",
    "MITM/プロキシで /api/checkout のPOSTボディを改ざん",
    {"D": 9, "R": 10, "E": 6, "A": 8, "D2": 7},
)
ws.add_mitigation(t1, "mitigate",
    "サーバーサイドでの金額再計算（カートIDから商品価格を再取得）",
    effort="low", assigned_to="バックエンドチーム")

t2 = ws.add_threat(
    "Spoofing", "ブラウザ",
    "他ユーザーの注文情報閲覧（IDOR）",
    "GET /api/orders/{id} のidを変更して他者の注文を閲覧",
    {"D": 7, "R": 10, "E": 9, "A": 5, "D2": 8},
)
ws.add_mitigation(t2, "mitigate",
    "オーナーシップチェック: order.user_id == current_user.id",
    effort="low", assigned_to="バックエンドチーム")

summary = ws.get_summary()
print(f"プロジェクト: {summary['project']}")
print(f"脅威数: {summary['total_threats']}")
print(f"深刻度: {summary['severity_breakdown']}")
```

---

## アンチパターン

### アンチパターン1: 脅威モデリングの省略

「リリースが間に合わない」という理由で脅威モデリングを省略するパターン。

```python
# NG: 脅威モデリングを省略してリリース
class ReleaseProcess:
    def ship(self, feature):
        # 脅威モデリングなし → 本番で脆弱性が見つかる
        # 修正コスト: $100,000+
        deploy(feature)

# OK: 最小限の脅威モデリングを必ず実施
class SecureReleaseProcess:
    def ship(self, feature):
        # 最低限のSTRIDE分析（30分〜1時間）
        threats = self.quick_stride_analysis(feature)
        if any(t.risk_level == "Critical" for t in threats):
            raise SecurityGateError("Critical脅威が未対策です")
        # 対策完了後にリリース
        # 修正コスト: $100（設計段階）
        deploy(feature)
```

### アンチパターン2: 一度きりの脅威モデリング

初回リリース時にだけ脅威モデリングを行い、その後更新しないパターン。

```python
# NG: 初回のみ実施
threat_model = create_threat_model()  # v1.0でのみ実施
# v2.0で新機能追加 → 脅威モデル未更新 → 新しい脆弱性が放置

# OK: 変更のたびに差分分析
class ContinuousThreatModeling:
    def on_feature_change(self, feature_spec):
        """機能変更時に脅威モデルを差分更新"""
        # 1. 変更されたDFD要素を特定
        changed_elements = self.diff_dfd(feature_spec)
        # 2. 変更要素に対するSTRIDE分析
        new_threats = self.analyze_stride(changed_elements)
        # 3. 既存の脅威モデルに統合
        self.merge_threats(new_threats)
        # 4. レビューとサインオフ
        self.request_review(new_threats)
```

### アンチパターン3: 網羅性の過度な追求

すべての脅威を100%洗い出そうとして分析が終わらないパターン。リスクベースで優先順位をつけ、高リスクの脅威から順に対処する。パレートの法則（20%の脅威が80%のリスクを占める）を意識する。

---

## 実践演習

### 演習1（基礎）: STRIDE分析

以下のシステムの「ログインAPI」に対してSTRIDE分析を実施し、各カテゴリごとに1つ以上の脅威と対策を記述してください。

**システム概要**: ユーザーがメールアドレスとパスワードでログインするWebアプリケーション。ログイン成功時にJWTトークンを返却する。

<details>
<summary>模範解答</summary>

| カテゴリ | 脅威 | 攻撃手法 | 対策 |
|---------|------|---------|------|
| S: なりすまし | ブルートフォース攻撃 | 大量のパスワード試行 | レートリミット(5回/分)、アカウントロックアウト(30分)、MFA |
| S: なりすまし | クレデンシャルスタッフィング | 流出したID/PW一覧での試行 | パスワード漏洩チェック(HaveIBeenPwned API)、MFA |
| T: 改ざん | JWTトークンの改ざん | ヘッダーのalgをnoneに変更 | alg固定検証、RS256使用、署名検証の厳格化 |
| T: 改ざん | リクエストパラメータの改ざん | プロキシでemail/passwordフィールドを操作 | サーバーサイドバリデーション、TLS必須 |
| R: 否認 | ログイン操作の否認 | 「ログインしていない」と主張 | IP/UA/タイムスタンプ付き監査ログ、デバイスフィンガープリント |
| I: 情報漏洩 | エラーメッセージからの情報推測 | 「メールアドレスが存在しません」の応答 | 統一エラーメッセージ「認証に失敗しました」 |
| I: 情報漏洩 | タイミング攻撃 | 存在するアカウントと存在しないアカウントの応答時間差 | 定数時間比較、意図的な遅延 |
| D: サービス妨害 | ログインAPI枯渇攻撃 | 大量のログインリクエスト | レートリミット、CAPTCHA、CDN |
| E: 権限昇格 | JWT内のroleクレーム改ざん | JWT署名の脆弱性を利用 | 署名検証の厳格化、role情報はサーバーサイドで取得 |

</details>

### 演習2（応用）: DFD作成と信頼境界分析

以下のシステムのDFDを作成し、信頼境界を越えるデータフローを特定して、各フローの脅威を分析してください。

**システム概要**: モバイルアプリ → API Gateway → マイクロサービス（認証、商品、注文） → PostgreSQL → 外部決済API（Stripe）

<details>
<summary>模範解答</summary>

**DFD:**
```
信頼境界1（外部）          信頼境界2（DMZ）        信頼境界3（内部）
+------------------+     +------------------+   +------------------+
|  +-----------+   |     |  +-----------+   |   |  +-----------+   |
|  | モバイル   |-------->| API        |-------->| 認証       |   |
|  | アプリ    |   |     |  | Gateway   |   |   |  | サービス   |   |
|  +-----------+   |     |  +-----------+   |   |  +-----------+   |
|                  |     |       |          |   |       |          |
+------------------+     |       v          |   |  +-----------+   |
                         |  +-----------+   |   |  | 商品       |   |
                         |  | レート    |   |   |  | サービス   |   |
                         |  | リミッター |   |   |  +-----------+   |
                         |  +-----------+   |   |       |          |
                         +------------------+   |  +-----------+   |
                                                |  | 注文       |   |
信頼境界4（外部）                                |  | サービス   |   |
+------------------+                            |  +-----------+   |
|  +-----------+   |                            |       |          |
|  | Stripe    |<-------------------------------|  +----v------+   |
|  | API       |   |                            |  | PostgreSQL |   |
|  +-----------+   |                            |  +-----------+   |
+------------------+                            +------------------+
```

**信頼境界を越えるフローの脅威分析:**

| フロー | 境界 | 脅威 | 対策 |
|--------|------|------|------|
| モバイルアプリ → API Gateway | 外部→DMZ | なりすまし、改ざん、盗聴 | TLS 1.3、JWT Bearer認証、証明書ピンニング |
| API Gateway → 認証サービス | DMZ→内部 | なりすまし、改ざん | mTLS、サービスメッシュ |
| 注文サービス → PostgreSQL | 内部内 | SQLインジェクション、情報漏洩 | パラメータ化クエリ、最小権限DB接続、暗号化接続 |
| 注文サービス → Stripe API | 内部→外部 | 情報漏洩（カード情報） | PCI DSS準拠、トークナイゼーション、TLS |

</details>

### 演習3（発展）: アタックツリーとDREADスコアリング

以下のシナリオに対するアタックツリーを作成し、各リーフノードにDREADスコアを付与して、最もリスクの高い攻撃経路を特定してください。

**シナリオ**: オンラインバンキングシステムで、攻撃者が他者の口座から不正送金を実行する

<details>
<summary>模範解答</summary>

```
[他者の口座から不正送金] (Root, OR)
├── [アカウント乗っ取り] (OR)
│   ├── [フィッシング攻撃]
│   │   DREAD: D=9, R=8, E=7, A=3, D=6 → 6.6 (High)
│   ├── [ブルートフォース]
│   │   DREAD: D=9, R=10, E=4, A=1, D=8 → 6.4 (High)
│   └── [SIMスワッピング（MFA突破）]
│       DREAD: D=9, R=6, E=3, A=1, D=4 → 4.6 (Medium)
│
├── [アプリケーション脆弱性] (OR)
│   ├── [CSRF攻撃]
│   │   DREAD: D=9, R=8, E=6, A=5, D=7 → 7.0 (High)
│   ├── [IDOR（送金先IDの改ざん）]
│   │   DREAD: D=9, R=10, E=8, A=3, D=6 → 7.2 (High) ★最高リスク
│   └── [セッションハイジャック]
│       DREAD: D=9, R=7, E=5, A=3, D=5 → 5.8 (Medium)
│
└── [内部犯行] (OR)
    ├── [DB管理者による直接操作]
    │   DREAD: D=10, R=10, E=8, A=1, D=2 → 6.2 (High)
    └── [サポートスタッフの不正]
        DREAD: D=8, R=9, E=7, A=1, D=3 → 5.6 (Medium)
```

**最もリスクの高い経路**: IDOR（DREADスコア7.2）

**理由**: IDORは再現性が高く（R=10）、攻撃が比較的容易で（E=8）、被害が甚大（D=9）。送金APIのエンドポイントで、リクエストパラメータの`from_account_id`を他者のアカウントIDに変更するだけで攻撃が成立する可能性がある。

**優先対策**:
1. オーナーシップチェックの必須化（from_account_id == current_user.account_id）
2. 送金前の再認証（パスワードまたは生体認証）
3. 送金金額・頻度の異常検知（不正検知システム）
4. 送金通知の即時送信（メール + プッシュ通知）
</details>

---

## FAQ

### Q1: 脅威モデリングは誰が行うべきですか?

開発チーム全体で取り組むべきである。セキュリティ専門家だけでなく、以下のロールの参加が推奨される:
- **アーキテクト**: システム全体の構造把握、信頼境界の定義
- **開発者**: 実装上の脆弱性パターンの知識
- **運用担当者**: インフラレベルの脅威、監視要件
- **プロダクトオーナー**: ビジネスリスクの評価、優先順位の判断
- **セキュリティ専門家**: ファシリテーション、脅威ライブラリの提供

### Q2: STRIDEとDREADはどちらを使うべきですか?

両方を組み合わせて使う。STRIDEは脅威の「分類と特定」に、DREADは特定した脅威の「優先順位付け」に使う。STRIDEで洗い出した脅威にDREADスコアを付けることで、対策の優先順位が明確になる。大規模組織ではDREADの代わりにCVSSを使用するケースも多い。

### Q3: 小規模プロジェクトでも脅威モデリングは必要ですか?

規模に応じて簡易化すればよい。最低限以下の30分ワークショップを実施するだけでも効果がある:
1. ホワイトボードにDFDを描く（10分）
2. 信頼境界を特定する（5分）
3. STRIDEの各カテゴリについて1つずつ脅威を検討する（10分）
4. 上位3つの脅威の対策を決める（5分）

### Q4: 脅威モデリングの結果はどう管理すべきですか?

脅威モデルはバージョン管理し、以下のタイミングで更新すべきである:
- 新機能の追加時（差分分析）
- アーキテクチャの変更時（全体見直し）
- 新しい攻撃手法の発見時（脅威ライブラリの更新）
- インシデント発生後（教訓の反映）
- 定期レビュー（最低年1回）

ツールとしては、Microsoft Threat Modeling Tool、OWASP Threat Dragon、またはGit管理されたMarkdown/JSONが一般的である。

---

## まとめ

| 項目 | 要点 |
|------|------|
| STRIDE | 6カテゴリで脅威を体系的に分類・特定。DFD要素ごとに適用可能 |
| DREAD | 5軸のスコアリングで脅威のリスクを定量評価。優先順位の決定に使用 |
| アタックツリー | 攻撃目標を木構造で分解し、最小コスト・最大確率の経路を特定する |
| DFD | データの流れと信頼境界を可視化し、脅威の発生箇所を特定する |
| PASTA | 7段階のリスク中心手法。ビジネス目標との整合を重視 |
| キルチェーン | 攻撃の7段階を理解し、各段階での検知・防御ポイントを特定 |
| 実践手順 | 分解→特定→評価→対策の4段階を反復的に実施する |

---

## 次に読むべきガイド

- [00-security-overview.md](./00-security-overview.md) -- CIA三原則とリスク評価の基礎（復習）
- [02-security-principles.md](./02-security-principles.md) -- 脅威への対策としての設計原則
- [../01-web-security/00-owasp-top10.md](../01-web-security/00-owasp-top10.md) -- 具体的なWeb脆弱性のSTRIDE分類
- [../01-web-security/03-injection.md](../01-web-security/03-injection.md) -- インジェクション攻撃の詳細（STRIDE-T）
- [../../sql-and-query-mastery/docs/03-practical/](../../sql-and-query-mastery/docs/03-practical/) -- SQLインジェクション対策の実装

---

## 参考文献

1. Adam Shostack, "Threat Modeling: Designing for Security" -- Wiley, 2014
2. Microsoft SDL Threat Modeling Tool -- https://www.microsoft.com/en-us/securityengineering/sdl/threatmodeling
3. OWASP Threat Modeling -- https://owasp.org/www-community/Threat_Modeling
4. Bruce Schneier, "Attack Trees" -- Dr. Dobb's Journal, 1999
5. NIST SP 800-154 Guide to Data-Centric System Threat Modeling -- https://csrc.nist.gov/publications/detail/sp/800-154/draft
6. Lockheed Martin Cyber Kill Chain -- https://www.lockheedmartin.com/en-us/capabilities/cyber/cyber-kill-chain.html
7. PASTA Threat Modeling -- Tony UcedaVelez & Marco M. Morana, "Risk Centric Threat Modeling", Wiley, 2015
