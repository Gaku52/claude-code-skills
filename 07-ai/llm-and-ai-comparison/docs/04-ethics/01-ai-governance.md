# AI ガバナンス — 規制・ポリシー

> AI の開発・利用に関する法規制、企業ポリシー、国際的な枠組みを体系的に理解し、組織内でのガバナンス体制を構築するための知識を学ぶ。

---

## この章で学ぶこと

1. **法規制の動向** — EU AI Act、日本の AI ガイドライン、米国の大統領令など主要な規制フレームワーク
2. **組織のガバナンス体制** — AI 倫理委員会、リスク評価プロセス、インシデント対応の構築方法
3. **コンプライアンス実装** — 技術的・組織的な対応策の具体的な実装手法
4. **データガバナンス** — 学習データの品質管理、プライバシー保護、データリネージの確立
5. **説明責任と透明性** — モデルカード、説明可能AI（XAI）、監査対応の技術的実装

---

## 1. AI ガバナンスの全体像

### 1.1 ガバナンスの三層構造

```
+------------------------------------------------------------------+
|                    AI ガバナンスの構造                              |
+------------------------------------------------------------------+
|                                                                    |
|  +------------------------------------------------------------+  |
|  |  Layer 1: 国際・国家レベル                                   |  |
|  |  - EU AI Act, 日本AI事業者ガイドライン, 米国大統領令          |  |
|  |  - G7広島AIプロセス, OECD AI原則                             |  |
|  +------------------------------------------------------------+  |
|                                                                    |
|  +------------------------------------------------------------+  |
|  |  Layer 2: 組織・企業レベル                                   |  |
|  |  - AI倫理方針, リスク管理フレームワーク                       |  |
|  |  - ガバナンス委員会, 監査プロセス                             |  |
|  +------------------------------------------------------------+  |
|                                                                    |
|  +------------------------------------------------------------+  |
|  |  Layer 3: プロジェクト・システムレベル                        |  |
|  |  - 影響評価 (AIIA), 技術的安全対策                           |  |
|  |  - モデルカード, 監視ダッシュボード                           |  |
|  +------------------------------------------------------------+  |
|                                                                    |
+------------------------------------------------------------------+
```

### 1.2 ステークホルダーマップ

```
+----------+     +----------+     +----------+
| 規制当局  |     | 開発者   |     | 利用者   |
| (政府)    |     | (企業)   |     | (市民)   |
+----------+     +----------+     +----------+
     |                |                |
     v                v                v
+--------------------------------------------------+
|              AI ガバナンスフレームワーク             |
|                                                    |
|  法規制 <---> 自主規制 <---> 技術標準               |
|  (強制力あり)  (業界団体)    (IEEE, ISO)             |
+--------------------------------------------------+
     |                |                |
     v                v                v
+----------+     +----------+     +----------+
| 監査機関  |     | 研究機関  |     | 市民社会  |
+----------+     +----------+     +----------+
```

### 1.3 AI ガバナンス成熟度モデル

```python
# コード例: AI ガバナンス成熟度の自己評価ツール
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

class GovernanceMaturityLevel(IntEnum):
    """ガバナンス成熟度レベル"""
    AD_HOC = 1       # 場当たり的
    DEVELOPING = 2    # 発展途上
    DEFINED = 3       # 定義済み
    MANAGED = 4       # 管理された
    OPTIMIZING = 5    # 最適化中

@dataclass
class GovernanceMaturityAssessment:
    """AI ガバナンス成熟度の評価"""

    DIMENSIONS = {
        "ポリシーと戦略": {
            1: "AI倫理方針が存在しない",
            2: "基本的なAI利用方針がある",
            3: "包括的なAIガバナンスポリシーが策定されている",
            4: "ポリシーが定期的にレビュー・更新されている",
            5: "ポリシーが業界のベストプラクティスを先導している",
        },
        "組織体制": {
            1: "ガバナンスの責任者が不明確",
            2: "AI担当者が非公式に存在する",
            3: "AI倫理委員会が正式に設置されている",
            4: "CxOレベルにAI責任者がいる",
            5: "全部門に AI ガバナンスチャンピオンが配置されている",
        },
        "リスク管理": {
            1: "リスク評価プロセスがない",
            2: "基本的なリスクチェックリストがある",
            3: "体系的な影響評価（AIIA）プロセスがある",
            4: "リスクが定量的に管理され、KPIが設定されている",
            5: "予測的リスク管理と自動検知が実装されている",
        },
        "技術的対策": {
            1: "安全性対策が場当たり的",
            2: "基本的なフィルタリングが導入されている",
            3: "多層ガードレールとテストスイートがある",
            4: "CI/CDに安全性テストが統合されている",
            5: "自動レッドチームと継続的モニタリングが稼働している",
        },
        "透明性と説明責任": {
            1: "モデルの文書化がない",
            2: "基本的なモデルカードがある",
            3: "説明可能性の手法が導入されている",
            4: "監査ログが包括的に記録されている",
            5: "外部監査に対応可能な透明性体制がある",
        },
        "研修と文化": {
            1: "AI倫理に関する研修がない",
            2: "年1回のeラーニングがある",
            3: "定期的な実践的研修が実施されている",
            4: "ケーススタディの共有とナレッジベースが整備されている",
            5: "AI倫理が企業文化に深く根付いている",
        },
    }

    scores: dict = field(default_factory=dict)

    def evaluate(self, responses: dict[str, int]) -> GovernanceMaturityLevel:
        """各次元のスコアから総合成熟度レベルを判定する"""
        total = 0
        count = 0

        for dimension, score in responses.items():
            if dimension in self.DIMENSIONS:
                self.scores[dimension] = {
                    "score": score,
                    "description": self.DIMENSIONS[dimension].get(score, ""),
                    "max_score": 5,
                }
                total += score
                count += 1

        avg_score = total / count if count > 0 else 1
        return GovernanceMaturityLevel(min(round(avg_score), 5))

    def generate_roadmap(self, current_level: GovernanceMaturityLevel) -> list[dict]:
        """現在のレベルに基づく改善ロードマップを生成する"""
        roadmaps = {
            GovernanceMaturityLevel.AD_HOC: [
                {
                    "phase": "Phase 1: 基盤構築 (0-3ヶ月)",
                    "actions": [
                        "AI利用方針の策定と全社展開",
                        "AI責任者の任命",
                        "基本的なリスクチェックリストの導入",
                        "既存AIシステムの棚卸し",
                    ],
                },
            ],
            GovernanceMaturityLevel.DEVELOPING: [
                {
                    "phase": "Phase 2: プロセス整備 (3-6ヶ月)",
                    "actions": [
                        "AI倫理委員会の設置",
                        "AIIA（AI影響評価）プロセスの導入",
                        "監査ログの実装",
                        "基本的な安全性テストの導入",
                    ],
                },
            ],
            GovernanceMaturityLevel.DEFINED: [
                {
                    "phase": "Phase 3: 定量管理 (6-12ヶ月)",
                    "actions": [
                        "安全性KPIとSLOの設定",
                        "CI/CDへの安全性テスト統合",
                        "モデルカードの自動生成",
                        "インシデント対応プロセスの整備",
                    ],
                },
            ],
            GovernanceMaturityLevel.MANAGED: [
                {
                    "phase": "Phase 4: 高度化 (12-18ヶ月)",
                    "actions": [
                        "自動レッドチームの導入",
                        "予測的リスク管理の実装",
                        "外部監査への対応体制構築",
                        "業界連携・情報共有の推進",
                    ],
                },
            ],
            GovernanceMaturityLevel.OPTIMIZING: [
                {
                    "phase": "Phase 5: 最適化 (継続)",
                    "actions": [
                        "ベストプラクティスの外部公開",
                        "規制当局との協力関係構築",
                        "業界標準の策定への参画",
                        "AIガバナンスの研究開発",
                    ],
                },
            ],
        }

        return roadmaps.get(current_level, [])
```

---

## 2. 主要な法規制

### 2.1 EU AI Act

```python
# コード例 1: EU AI Act のリスクカテゴリ判定ツール
from enum import Enum
from dataclasses import dataclass

class RiskLevel(Enum):
    UNACCEPTABLE = "unacceptable"   # 禁止
    HIGH = "high"                    # 厳格な規制
    LIMITED = "limited"              # 透明性義務
    MINIMAL = "minimal"              # 自主規制

@dataclass
class AISystemClassification:
    name: str
    risk_level: RiskLevel
    obligations: list[str]
    examples: list[str]

# EU AI Act のリスク分類
CLASSIFICATIONS = [
    AISystemClassification(
        name="禁止されるAIシステム",
        risk_level=RiskLevel.UNACCEPTABLE,
        obligations=["使用禁止"],
        examples=[
            "ソーシャルスコアリング（政府による市民の社会信用評価）",
            "リアルタイム遠隔生体認証（公共空間、法執行目的以外）",
            "感情認識（職場・教育機関での使用）",
            "サブリミナル操作による行動誘導",
        ]
    ),
    AISystemClassification(
        name="ハイリスクAIシステム",
        risk_level=RiskLevel.HIGH,
        obligations=[
            "適合性評価の実施",
            "リスク管理システムの導入",
            "データガバナンスの確保",
            "技術文書の作成",
            "ログの記録と保持",
            "人間による監視の確保",
            "正確性・堅牢性・サイバーセキュリティの確保",
        ],
        examples=[
            "採用・人事評価システム",
            "信用スコアリング",
            "教育における成績評価",
            "法執行における犯罪予測",
            "生体認証による本人確認",
            "重要インフラの安全管理",
        ]
    ),
    AISystemClassification(
        name="限定リスクAIシステム",
        risk_level=RiskLevel.LIMITED,
        obligations=[
            "AIであることの明示（透明性義務）",
            "ディープフェイクの表示義務",
        ],
        examples=[
            "チャットボット",
            "感情認識システム（限定的用途）",
            "ディープフェイク生成",
        ]
    ),
    AISystemClassification(
        name="最小リスクAIシステム",
        risk_level=RiskLevel.MINIMAL,
        obligations=["任意の行動規範への準拠"],
        examples=[
            "スパムフィルター",
            "ゲームAI",
            "レコメンデーション（非高リスク）",
        ]
    ),
]

def classify_ai_system(system_description: str,
                        use_case: str) -> AISystemClassification:
    """AIシステムのリスクレベルを判定する（簡易版）"""
    high_risk_keywords = [
        "採用", "人事", "信用", "教育評価", "法執行",
        "生体認証", "重要インフラ", "医療診断"
    ]
    for keyword in high_risk_keywords:
        if keyword in use_case:
            return CLASSIFICATIONS[1]  # HIGH

    return CLASSIFICATIONS[3]  # MINIMAL
```

### 2.2 EU AI Act 汎用AI（GPAI）規制

```python
# コード例: 汎用AIモデル（GPAI）のコンプライアンスチェック
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GPAICompliance:
    """EU AI Act の汎用AIモデル（GPAI）に関するコンプライアンスチェック"""

    # GPAI モデルの分類
    # - 一般 GPAI: 基本的な義務
    # - システミックリスク GPAI: 追加義務（10^25 FLOP 以上の学習計算量）

    SYSTEMIC_RISK_THRESHOLD_FLOPS = 10**25  # 10^25 FLOP

    GENERAL_GPAI_OBLIGATIONS = [
        "技術文書の作成と維持",
        "AI Office への情報提供",
        "EU 著作権法の遵守",
        "学習データの要約の公開",
    ]

    SYSTEMIC_RISK_OBLIGATIONS = [
        "モデル評価の実施",
        "システミックリスクの評価と緩和",
        "重大インシデントの追跡と報告",
        "適切なサイバーセキュリティの確保",
        "エネルギー消費量の報告",
    ]

    model_name: str
    training_flops: float
    is_open_source: bool = False
    training_data_summary: Optional[str] = None
    technical_documentation: Optional[str] = None

    @property
    def is_systemic_risk(self) -> bool:
        """システミックリスクモデルかどうかを判定する"""
        return self.training_flops >= self.SYSTEMIC_RISK_THRESHOLD_FLOPS

    def check_compliance(self) -> dict:
        """コンプライアンス状態をチェックする"""
        results = {
            "model_name": self.model_name,
            "is_systemic_risk": self.is_systemic_risk,
            "classification": (
                "システミックリスクGPAI" if self.is_systemic_risk
                else "一般GPAI"
            ),
            "obligations": [],
            "compliance_status": [],
        }

        # 一般 GPAI の義務チェック
        for obligation in self.GENERAL_GPAI_OBLIGATIONS:
            status = self._check_obligation(obligation)
            results["obligations"].append({
                "obligation": obligation,
                "status": status,
                "category": "general",
            })

        # システミックリスク GPAI の追加義務
        if self.is_systemic_risk:
            for obligation in self.SYSTEMIC_RISK_OBLIGATIONS:
                status = self._check_obligation(obligation)
                results["obligations"].append({
                    "obligation": obligation,
                    "status": status,
                    "category": "systemic_risk",
                })

        # オープンソースの場合の例外
        if self.is_open_source and not self.is_systemic_risk:
            results["note"] = (
                "オープンソースモデルは一部の義務が免除される可能性があります "
                "(ただしシステミックリスクモデルは免除対象外)"
            )

        # 全体のコンプライアンス判定
        non_compliant = [
            o for o in results["obligations"]
            if o["status"] == "non_compliant"
        ]
        results["overall_status"] = (
            "compliant" if not non_compliant else "non_compliant"
        )
        results["non_compliant_count"] = len(non_compliant)

        return results

    def _check_obligation(self, obligation: str) -> str:
        """個別の義務のコンプライアンスをチェックする"""
        if "技術文書" in obligation:
            return "compliant" if self.technical_documentation else "non_compliant"
        if "学習データ" in obligation:
            return "compliant" if self.training_data_summary else "non_compliant"
        return "needs_review"  # 手動確認が必要

    def generate_compliance_report(self) -> str:
        """コンプライアンスレポートを生成する"""
        results = self.check_compliance()

        report = f"# GPAI コンプライアンスレポート\n\n"
        report += f"## モデル情報\n"
        report += f"- **モデル名**: {self.model_name}\n"
        report += f"- **学習計算量**: {self.training_flops:.2e} FLOP\n"
        report += f"- **分類**: {results['classification']}\n"
        report += f"- **オープンソース**: {'はい' if self.is_open_source else 'いいえ'}\n\n"

        report += f"## コンプライアンス状態\n"
        report += f"- **総合判定**: {results['overall_status']}\n"
        report += f"- **非準拠項目数**: {results['non_compliant_count']}\n\n"

        report += f"## 義務と状態\n\n"
        for ob in results["obligations"]:
            status_icon = {
                "compliant": "[OK]",
                "non_compliant": "[NG]",
                "needs_review": "[要確認]",
            }.get(ob["status"], "[?]")
            report += f"- {status_icon} {ob['obligation']} ({ob['category']})\n"

        return report
```

### 2.3 主要国の規制比較

| 項目 | EU AI Act | 日本 AI ガイドライン | 米国 AI 大統領令 (EO 14110) | 中国 AI 規制 | 英国 AI 規制 |
|------|-----------|---------------------|---------------------------|-------------|-------------|
| 法的拘束力 | あり (罰則付き) | なし (ガイドライン) | 一部あり (連邦機関向け) | あり (段階的) | なし (原則ベース) |
| リスク分類 | 4段階 | 原則ベース | セクター別 | 用途別 | セクター別 |
| 対象 | EU域内で利用される全AI | 日本国内の事業者 | 連邦政府 + 大規模AI | 中国国内の全サービス | 英国内のAIシステム |
| 罰則 | 最大3,500万EUR or 売上7% | なし | 連邦調達からの排除等 | 罰金 + 営業停止 | セクター別の既存罰則 |
| 施行時期 | 2024年段階的施行 | 2024年改訂 | 2023年10月 | 2023年〜段階的 | 2024年〜 |
| 特徴 | 包括的・横断的 | 柔軟・自主規制重視 | 安全保障重視 | コンテンツ規制重視 | pro-innovation |
| GPAI規制 | あり (10^25 FLOP) | なし | 報告義務あり | 生成AI管理弁法 | 検討中 |
| 域外適用 | あり | なし | 限定的 | あり | 検討中 |

### 2.4 日本の AI ガバナンス詳細

```python
# コード例: 日本の AI 事業者ガイドライン準拠チェッカー
class JapanAIGuidelineChecker:
    """日本のAI事業者ガイドライン（第1.0版）への準拠チェック"""

    # 10の基本原則
    PRINCIPLES = {
        "P1_human_centric": {
            "name": "人間中心の原則",
            "description": "AIは人間の能力を拡張し、人間が意思決定の主体であること",
            "checkpoints": [
                "AIの判断に対する人間のオーバーライド機能がある",
                "ユーザーがAIとの対話であることを認識できる",
                "AIの利用が人間の尊厳を侵害しない",
            ],
        },
        "P2_safety": {
            "name": "安全性の原則",
            "description": "AIシステムが社会に害を及ぼさないこと",
            "checkpoints": [
                "リスク評価が実施されている",
                "安全性テストが定期的に実施されている",
                "緊急停止機能が実装されている",
            ],
        },
        "P3_fairness": {
            "name": "公平性の原則",
            "description": "AIが不当な差別を行わないこと",
            "checkpoints": [
                "バイアス評価が実施されている",
                "多様なテストデータで評価されている",
                "公平性メトリクスが定義・監視されている",
            ],
        },
        "P4_privacy": {
            "name": "プライバシーの原則",
            "description": "個人のプライバシーが適切に保護されること",
            "checkpoints": [
                "個人情報保護法に準拠している",
                "データの最小化原則が適用されている",
                "適切な同意メカニズムがある",
            ],
        },
        "P5_security": {
            "name": "セキュリティの原則",
            "description": "AIシステムのセキュリティが確保されていること",
            "checkpoints": [
                "敵対的攻撃への耐性が評価されている",
                "アクセス制御が適切に設定されている",
                "インシデント対応プロセスがある",
            ],
        },
        "P6_transparency": {
            "name": "透明性の原則",
            "description": "AIの判断プロセスが適切に説明可能であること",
            "checkpoints": [
                "モデルカードが作成されている",
                "判断理由の説明機能がある",
                "利用規約でAIの利用を明示している",
            ],
        },
        "P7_accountability": {
            "name": "アカウンタビリティの原則",
            "description": "AIの開発・運用に関する説明責任が明確であること",
            "checkpoints": [
                "責任体制が明確に定義されている",
                "監査ログが適切に記録されている",
                "苦情対応プロセスがある",
            ],
        },
        "P8_education": {
            "name": "教育・リテラシーの原則",
            "description": "AI に関する適切な教育とリテラシー向上",
            "checkpoints": [
                "従業員向けAI倫理研修が実施されている",
                "利用者向けのガイドラインが提供されている",
                "AIリテラシー向上の取り組みがある",
            ],
        },
        "P9_fair_competition": {
            "name": "公正競争の原則",
            "description": "AIの利用が公正な競争環境を阻害しないこと",
            "checkpoints": [
                "市場独占につながる行為を行っていない",
                "データの不当な囲い込みを行っていない",
                "オープンな標準を尊重している",
            ],
        },
        "P10_innovation": {
            "name": "イノベーションの原則",
            "description": "ガバナンスがイノベーションを阻害しないこと",
            "checkpoints": [
                "リスクベースのアプローチを採用している",
                "実験的な取り組みに対する柔軟なプロセスがある",
                "技術的進歩に応じてガイドラインを更新している",
            ],
        },
    }

    def check_compliance(
        self, responses: dict[str, list[bool]]
    ) -> dict:
        """ガイドライン準拠状況をチェックする"""
        results = {
            "principles": [],
            "overall_compliance": 0.0,
            "non_compliant_areas": [],
        }

        total_checks = 0
        passed_checks = 0

        for principle_id, answers in responses.items():
            principle = self.PRINCIPLES.get(principle_id)
            if not principle:
                continue

            checkpoints = principle["checkpoints"]
            principle_passed = sum(
                1 for a in answers[:len(checkpoints)] if a
            )
            principle_total = len(checkpoints)

            total_checks += principle_total
            passed_checks += principle_passed

            compliance_rate = (
                principle_passed / principle_total
                if principle_total > 0 else 0
            )

            result = {
                "id": principle_id,
                "name": principle["name"],
                "compliance_rate": compliance_rate,
                "passed": principle_passed,
                "total": principle_total,
                "status": "compliant" if compliance_rate >= 0.67 else "non_compliant",
            }

            results["principles"].append(result)

            if compliance_rate < 0.67:
                results["non_compliant_areas"].append(principle["name"])

        results["overall_compliance"] = (
            passed_checks / total_checks if total_checks > 0 else 0
        )

        return results
```

---

## 3. 組織のガバナンス体制

### 3.1 AI 倫理委員会の構成

```python
# コード例: AI倫理委員会の構成と運営フレームワーク
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from typing import Optional

class CommitteeRole(Enum):
    CHAIR = "委員長"
    TECHNICAL = "技術委員"
    LEGAL = "法務委員"
    ETHICS = "倫理委員"
    BUSINESS = "事業委員"
    EXTERNAL = "外部委員"
    DATA_PROTECTION = "データ保護委員"

@dataclass
class CommitteeMember:
    name: str
    role: CommitteeRole
    department: str
    expertise: list[str]
    is_external: bool = False

@dataclass
class ReviewRequest:
    """AI倫理委員会へのレビュー申請"""
    id: str
    project_name: str
    requestor: str
    risk_level: str  # "high", "medium", "low"
    system_description: str
    intended_use: str
    affected_stakeholders: list[str]
    submitted_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # "pending", "under_review", "approved", "rejected", "conditional"
    decision: Optional[str] = None
    conditions: list[str] = field(default_factory=list)

class AIEthicsCommittee:
    """AI倫理委員会の運営管理"""

    def __init__(self):
        self.members: list[CommitteeMember] = []
        self.reviews: list[ReviewRequest] = []
        self.policies: list[dict] = []

    def add_member(self, member: CommitteeMember) -> None:
        self.members.append(member)

    def submit_review(self, request: ReviewRequest) -> str:
        """レビュー申請を受け付ける"""
        # リスクレベルに応じたレビュープロセスの決定
        if request.risk_level == "high":
            request.status = "under_review"
            required_quorum = max(len(self.members) * 2 // 3, 3)
            review_type = "full_committee"
        elif request.risk_level == "medium":
            request.status = "under_review"
            required_quorum = 3
            review_type = "subcommittee"
        else:
            request.status = "under_review"
            required_quorum = 1
            review_type = "fast_track"

        self.reviews.append(request)

        return (
            f"レビュー申請 {request.id} を受け付けました。\n"
            f"レビュータイプ: {review_type}\n"
            f"必要定足数: {required_quorum}名\n"
            f"想定期間: {'2-4週間' if review_type == 'full_committee' else '1週間' if review_type == 'subcommittee' else '3営業日'}"
        )

    def make_decision(
        self, review_id: str, decision: str,
        conditions: list[str] = None, rationale: str = ""
    ) -> dict:
        """レビュー結果の決定"""
        for review in self.reviews:
            if review.id == review_id:
                review.status = decision
                review.decision = rationale
                if conditions:
                    review.conditions = conditions

                return {
                    "review_id": review_id,
                    "project": review.project_name,
                    "decision": decision,
                    "conditions": conditions or [],
                    "rationale": rationale,
                    "decided_at": datetime.now().isoformat(),
                }

        return {"error": f"Review {review_id} not found"}

    def get_dashboard(self) -> dict:
        """委員会のダッシュボード情報を生成する"""
        return {
            "total_members": len(self.members),
            "external_members": sum(
                1 for m in self.members if m.is_external
            ),
            "pending_reviews": sum(
                1 for r in self.reviews if r.status == "pending"
            ),
            "under_review": sum(
                1 for r in self.reviews if r.status == "under_review"
            ),
            "total_reviews": len(self.reviews),
            "approval_rate": (
                sum(1 for r in self.reviews if r.status == "approved")
                / len(self.reviews)
                if self.reviews else 0
            ),
        }

# 使用例
committee = AIEthicsCommittee()

committee.add_member(CommitteeMember(
    name="田中太郎",
    role=CommitteeRole.CHAIR,
    department="CTO室",
    expertise=["AI安全性", "機械学習"],
))

committee.add_member(CommitteeMember(
    name="山田花子",
    role=CommitteeRole.LEGAL,
    department="法務部",
    expertise=["個人情報保護法", "EU AI Act"],
))

committee.add_member(CommitteeMember(
    name="鈴木教授",
    role=CommitteeRole.EXTERNAL,
    department="東京大学",
    expertise=["AI倫理", "社会学"],
    is_external=True,
))
```

### 3.2 AI 影響評価 (AIIA)

```python
# コード例 2: AI影響評価 (AIIA) テンプレート
@dataclass
class AIImpactAssessment:
    """AI Impact Assessment（AI影響評価）"""

    # 基本情報
    project_name: str
    system_description: str
    intended_use: str
    developer_team: str

    # リスク評価
    risk_categories: dict  # カテゴリ → リスクレベル

    # 公平性評価
    affected_groups: list[str]
    fairness_metrics: dict
    bias_mitigation: str

    # プライバシー評価
    personal_data_used: bool
    data_minimization: str
    consent_mechanism: str

    # 透明性
    explainability_method: str
    user_notification: str

    # 人間の監視
    human_oversight_level: str  # "full", "partial", "minimal"
    override_mechanism: str

    # 環境影響
    estimated_carbon_footprint: Optional[str] = None
    energy_efficiency_measures: Optional[str] = None

    # 承認
    approved_by: str = ""
    approval_date: str = ""
    review_schedule: str = ""  # "quarterly", "annually"

def generate_aiia_report(assessment: AIImpactAssessment) -> str:
    """AIIA レポートを生成する"""
    report = f"""
========================================
AI 影響評価レポート
========================================
プロジェクト: {assessment.project_name}
説明: {assessment.system_description}
想定用途: {assessment.intended_use}
開発チーム: {assessment.developer_team}

--- リスク評価 ---
"""
    for category, level in assessment.risk_categories.items():
        report += f"  {category}: {level}\n"

    report += f"""
--- 公平性 ---
影響を受けるグループ: {', '.join(assessment.affected_groups)}
バイアス緩和策: {assessment.bias_mitigation}
公平性メトリクス: {assessment.fairness_metrics}

--- プライバシー ---
個人データの使用: {'あり' if assessment.personal_data_used else 'なし'}
データ最小化: {assessment.data_minimization}
同意メカニズム: {assessment.consent_mechanism}

--- 透明性 ---
説明可能性手法: {assessment.explainability_method}
ユーザー通知: {assessment.user_notification}

--- 人間の監視 ---
監視レベル: {assessment.human_oversight_level}
オーバーライド: {assessment.override_mechanism}

--- 環境影響 ---
推定CO2排出量: {assessment.estimated_carbon_footprint or '未評価'}
省エネ対策: {assessment.energy_efficiency_measures or '未実施'}

--- 承認 ---
承認者: {assessment.approved_by}
承認日: {assessment.approval_date}
次回レビュー: {assessment.review_schedule}
"""
    return report
```

### 3.3 ガバナンスプロセスフロー

```
+------------------------------------------------------------------+
|                AI プロジェクトのガバナンスフロー                     |
+------------------------------------------------------------------+
|                                                                    |
|  企画段階                                                          |
|  +------------------+                                              |
|  | 1. 影響評価      | → リスクレベル判定                           |
|  |    (AIIA)        |   (High/Medium/Low)                         |
|  +------------------+                                              |
|          |                                                         |
|     [High Risk]                [Low Risk]                          |
|          |                         |                               |
|          v                         v                               |
|  +------------------+     +------------------+                     |
|  | 2. 倫理委員会    |     | 2. セルフ        |                     |
|  |    レビュー       |     |    チェックリスト |                     |
|  +------------------+     +------------------+                     |
|          |                         |                               |
|          v                         v                               |
|  開発段階                                                          |
|  +------------------+                                              |
|  | 3. 安全性テスト   | → レッドチーム、バイアステスト               |
|  +------------------+                                              |
|          |                                                         |
|          v                                                         |
|  +------------------+                                              |
|  | 4. 承認ゲート    | → 基準未達なら差し戻し                       |
|  +------------------+                                              |
|          |                                                         |
|          v                                                         |
|  運用段階                                                          |
|  +------------------+                                              |
|  | 5. 継続的監視    | → ドリフト検知、インシデント対応              |
|  +------------------+                                              |
|          |                                                         |
|          v                                                         |
|  +------------------+                                              |
|  | 6. 定期レビュー   | → 四半期/年次                                |
|  +------------------+                                              |
+------------------------------------------------------------------+
```

---

## 4. コンプライアンス実装

### 4.1 監査ログの実装

```python
# コード例 3: AI システムの包括的監査ログ
import json
import hashlib
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, Any

@dataclass
class AuditLogEntry:
    timestamp: str
    system_id: str
    model_version: str
    action: str          # "prediction", "decision", "override", "error"
    input_hash: str      # 入力データのハッシュ（個人情報を含まない）
    output: str
    confidence: float
    explanation: str
    user_id: str | None  # オペレーターID
    decision_type: str   # "automated", "human_reviewed", "human_overridden"
    metadata: dict = field(default_factory=dict)
    trace_id: Optional[str] = None  # 分散トレーシング用

class AIAuditLogger:
    """AI意思決定の監査ログを記録する"""

    def __init__(self, storage_backend, retention_days: int = 2555):
        """
        Args:
            storage_backend: ログの保存先
            retention_days: 保持期間（デフォルト7年 = EU AI Act要件）
        """
        self.storage = storage_backend
        self.retention_days = retention_days

    async def log_prediction(
        self, system_id: str, model_version: str,
        input_data: dict, output: str,
        confidence: float, explanation: str,
        operator_id: str = None,
        decision_type: str = "automated",
        trace_id: str = None,
        extra_metadata: dict = None
    ) -> str:
        """予測結果のログを記録する"""
        entry = AuditLogEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            system_id=system_id,
            model_version=model_version,
            action="prediction",
            input_hash=self._hash_input(input_data),
            output=output,
            confidence=confidence,
            explanation=explanation,
            user_id=operator_id,
            decision_type=decision_type,
            trace_id=trace_id,
            metadata={
                "input_features": list(input_data.keys()),
                "output_length": len(output),
                "retention_until": self._calculate_retention(),
                **(extra_metadata or {}),
            }
        )

        log_id = await self.storage.write(asdict(entry))
        return log_id

    async def log_human_override(
        self, system_id: str, original_prediction: str,
        override_value: str, operator_id: str,
        reason: str, trace_id: str = None
    ) -> str:
        """人間によるオーバーライドをログに記録する"""
        entry = AuditLogEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            system_id=system_id,
            model_version="N/A",
            action="override",
            input_hash="",
            output=override_value,
            confidence=1.0,
            explanation=reason,
            user_id=operator_id,
            decision_type="human_overridden",
            trace_id=trace_id,
            metadata={
                "original_prediction": original_prediction,
                "override_reason": reason,
                "retention_until": self._calculate_retention(),
            }
        )

        return await self.storage.write(asdict(entry))

    async def log_error(
        self, system_id: str, model_version: str,
        error_type: str, error_message: str,
        input_hash: str = "", trace_id: str = None
    ) -> str:
        """エラーイベントをログに記録する"""
        entry = AuditLogEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            system_id=system_id,
            model_version=model_version,
            action="error",
            input_hash=input_hash,
            output="",
            confidence=0.0,
            explanation=error_message,
            user_id=None,
            decision_type="automated",
            trace_id=trace_id,
            metadata={
                "error_type": error_type,
                "retention_until": self._calculate_retention(),
            }
        )

        return await self.storage.write(asdict(entry))

    async def query_audit_trail(
        self, system_id: str,
        start_time: datetime, end_time: datetime,
        action_filter: str = None
    ) -> list[dict]:
        """監査証跡を照会する"""
        query = {
            "system_id": system_id,
            "timestamp_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
        }
        if action_filter:
            query["action"] = action_filter

        return await self.storage.query(query)

    def _hash_input(self, data: dict) -> str:
        """個人情報を含まないハッシュを生成する"""
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]

    def _calculate_retention(self) -> str:
        """保持期限を計算する"""
        from datetime import timedelta
        retention_date = datetime.utcnow() + timedelta(days=self.retention_days)
        return retention_date.isoformat() + "Z"
```

### 4.2 モデルカードの自動生成

```python
# コード例 4: EU AI Act / 日本ガイドライン準拠のモデルカード自動生成
class ModelCardGenerator:
    """EU AI Act 準拠のモデルカードを生成する"""

    def generate(self, model_info: dict, eval_results: dict,
                 safety_results: dict) -> str:
        return f"""
# モデルカード: {model_info['name']}

## 基本情報
- **モデル名**: {model_info['name']}
- **バージョン**: {model_info['version']}
- **開発者**: {model_info['developer']}
- **リリース日**: {model_info['release_date']}
- **モデルタイプ**: {model_info['type']}
- **ライセンス**: {model_info['license']}

## 意図された用途
{model_info['intended_use']}

## 意図されていない用途
{model_info['out_of_scope_use']}

## 学習データ
- **データソース**: {model_info['training_data']['source']}
- **データサイズ**: {model_info['training_data']['size']}
- **前処理**: {model_info['training_data']['preprocessing']}

## 性能指標
| メトリクス | 全体 | グループA | グループB |
|-----------|------|----------|----------|
| 精度 | {eval_results['accuracy']:.3f} | {eval_results['accuracy_a']:.3f} | {eval_results['accuracy_b']:.3f} |
| F1 | {eval_results['f1']:.3f} | {eval_results['f1_a']:.3f} | {eval_results['f1_b']:.3f} |

## 安全性評価
- **有害コンテンツ拒否率**: {safety_results['refusal_rate']:.1%}
- **ジェイルブレイク耐性**: {safety_results['jailbreak_resistance']:.1%}
- **バイアススコア**: {safety_results['bias_score']:.3f}

## 制限事項
{model_info['limitations']}

## 倫理的考慮事項
{model_info['ethical_considerations']}
"""

    def generate_technical_documentation(
        self, model_info: dict,
        training_details: dict,
        evaluation_details: dict
    ) -> str:
        """EU AI Act 準拠の技術文書を生成する"""
        return f"""
# 技術文書: {model_info['name']}
## EU AI Act Article 11 準拠

### 1. システムの一般的説明
{model_info.get('general_description', 'N/A')}

### 2. 設計仕様
- **アーキテクチャ**: {training_details.get('architecture', 'N/A')}
- **パラメータ数**: {training_details.get('parameters', 'N/A')}
- **学習計算量**: {training_details.get('training_compute', 'N/A')} FLOP

### 3. 開発プロセス
- **学習手法**: {training_details.get('training_method', 'N/A')}
- **アライメント手法**: {training_details.get('alignment_method', 'N/A')}
- **安全性対策**: {training_details.get('safety_measures', 'N/A')}

### 4. 学習データ
- **データソース**: {training_details.get('data_source', 'N/A')}
- **データ量**: {training_details.get('data_size', 'N/A')}
- **データ品質管理**: {training_details.get('data_quality', 'N/A')}
- **バイアス対策**: {training_details.get('bias_mitigation', 'N/A')}

### 5. 評価結果
- **ベンチマーク結果**: {evaluation_details.get('benchmark_results', 'N/A')}
- **安全性テスト結果**: {evaluation_details.get('safety_results', 'N/A')}
- **バイアステスト結果**: {evaluation_details.get('bias_results', 'N/A')}

### 6. リスク管理
- **識別されたリスク**: {evaluation_details.get('identified_risks', 'N/A')}
- **緩和策**: {evaluation_details.get('mitigations', 'N/A')}
- **残存リスク**: {evaluation_details.get('residual_risks', 'N/A')}

### 7. 人間による監視
- **監視メカニズム**: {model_info.get('human_oversight', 'N/A')}
- **オーバーライド手順**: {model_info.get('override_procedure', 'N/A')}

### 8. サイバーセキュリティ
- **脅威モデル**: {model_info.get('threat_model', 'N/A')}
- **セキュリティ対策**: {model_info.get('security_measures', 'N/A')}
"""
```

### 4.3 データガバナンスの実装

```python
# コード例: 学習データのガバナンスフレームワーク
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class DataSource:
    """学習データソースの定義"""
    id: str
    name: str
    source_type: str  # "public", "licensed", "proprietary", "user_generated"
    license: str
    contains_pii: bool
    consent_obtained: bool
    collection_date: str
    geography: str  # データの地理的範囲
    language: str
    volume: str
    quality_score: float  # 0.0-1.0
    bias_assessment: Optional[str] = None

@dataclass
class DataLineageRecord:
    """データリネージの記録"""
    step: int
    operation: str
    input_sources: list[str]
    output_id: str
    timestamp: str
    operator: str
    description: str
    metadata: dict = field(default_factory=dict)

class TrainingDataGovernance:
    """学習データのガバナンス管理"""

    def __init__(self):
        self.data_sources: list[DataSource] = []
        self.lineage: list[DataLineageRecord] = []
        self.audit_trail: list[dict] = []

    def register_data_source(self, source: DataSource) -> dict:
        """データソースを登録する"""
        # バリデーション
        issues = []

        if source.contains_pii and not source.consent_obtained:
            issues.append(
                "PII を含むデータソースには同意の取得が必要です"
            )

        if source.source_type == "licensed" and not source.license:
            issues.append(
                "ライセンス情報が不足しています"
            )

        if source.quality_score < 0.5:
            issues.append(
                f"データ品質スコアが低い ({source.quality_score:.2f})"
            )

        self.data_sources.append(source)

        self.audit_trail.append({
            "timestamp": datetime.now().isoformat(),
            "action": "register_data_source",
            "source_id": source.id,
            "issues": issues,
        })

        return {
            "source_id": source.id,
            "registered": True,
            "issues": issues,
            "is_compliant": len(issues) == 0,
        }

    def record_lineage(self, record: DataLineageRecord) -> None:
        """データリネージを記録する"""
        self.lineage.append(record)

    def check_eu_ai_act_data_compliance(self) -> dict:
        """EU AI Act のデータガバナンス要件への準拠をチェックする"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_sources": len(self.data_sources),
            "checks": [],
        }

        # Article 10: データとデータガバナンス
        checks = [
            {
                "requirement": "学習データの品質管理プロセスがある",
                "status": all(
                    s.quality_score >= 0.5 for s in self.data_sources
                ),
            },
            {
                "requirement": "データの出所が記録されている",
                "status": all(
                    s.source_type != "" for s in self.data_sources
                ),
            },
            {
                "requirement": "バイアス評価が実施されている",
                "status": all(
                    s.bias_assessment is not None
                    for s in self.data_sources
                ),
            },
            {
                "requirement": "PII データに適切な同意がある",
                "status": all(
                    not s.contains_pii or s.consent_obtained
                    for s in self.data_sources
                ),
            },
            {
                "requirement": "データリネージが記録されている",
                "status": len(self.lineage) > 0,
            },
        ]

        results["checks"] = checks
        results["compliant_count"] = sum(
            1 for c in checks if c["status"]
        )
        results["overall_compliant"] = all(c["status"] for c in checks)

        return results

    def generate_data_summary(self) -> str:
        """EU AI Act 要求のデータ要約を生成する"""
        summary = "# 学習データ要約\n\n"
        summary += f"## 概要\n"
        summary += f"- データソース数: {len(self.data_sources)}\n"

        # ソースタイプ別集計
        type_counts = {}
        for source in self.data_sources:
            type_counts[source.source_type] = (
                type_counts.get(source.source_type, 0) + 1
            )

        summary += f"- ソースタイプ分布:\n"
        for stype, count in type_counts.items():
            summary += f"  - {stype}: {count}\n"

        # PII に関する情報
        pii_sources = [s for s in self.data_sources if s.contains_pii]
        summary += f"\n## 個人情報\n"
        summary += f"- PII含有データソース数: {len(pii_sources)}\n"
        summary += f"- 全件同意取得済み: {'はい' if all(s.consent_obtained for s in pii_sources) else 'いいえ'}\n"

        # 言語分布
        languages = set(s.language for s in self.data_sources)
        summary += f"\n## 言語\n"
        summary += f"- 対象言語: {', '.join(languages)}\n"

        return summary
```

---

## 5. 国際的な AI 原則の比較

| 原則 | OECD (2019) | EU AI Act (2024) | 日本 (2024) | IEEE Ethically Aligned Design | G7 広島AIプロセス |
|------|-------------|-------------------|-------------|-------------------------------|-----------------|
| 人間中心 | ◯ | ◯ | ◯ | ◯ | ◯ |
| 透明性 | ◯ | ◯ (義務) | ◯ | ◯ | ◯ |
| 公平性 | ◯ | ◯ (義務) | ◯ | ◯ | ◯ |
| 安全性 | ◯ | ◯ (義務) | ◯ | ◯ | ◯ |
| プライバシー | ◯ | GDPR連携 | ◯ | ◯ | ◯ |
| 説明責任 | ◯ | ◯ (罰則付き) | ◯ | ◯ | ◯ |
| イノベーション | ◯ | サンドボックス制度 | ◯ (重視) | △ | ◯ |
| 環境配慮 | △ | ◯ (エネルギー報告) | △ | △ | ◯ |
| セキュリティ | ◯ | ◯ (義務) | ◯ | ◯ | ◯ |
| 国際協調 | ◯ | 相互認証 | ◯ | △ | ◯ (主導) |

### 5.1 G7 広島AIプロセスの詳細

```python
# コード例: G7 広島AIプロセス — 生成AI開発者向け行動規範チェック
class HiroshimaAIProcessChecker:
    """G7広島AIプロセスの国際行動規範への準拠チェック"""

    # 11の指導原則
    GUIDING_PRINCIPLES = [
        {
            "id": "GP1",
            "title": "AIライフサイクル全体でのリスク管理",
            "checkpoints": [
                "開発段階でのリスク評価を実施している",
                "運用段階での継続的モニタリングがある",
                "リスク緩和策が文書化されている",
            ],
        },
        {
            "id": "GP2",
            "title": "悪用と誤用の特定と緩和",
            "checkpoints": [
                "悪用シナリオが体系的に分析されている",
                "レッドチーミングを定期的に実施している",
                "利用規約で禁止事項が明記されている",
            ],
        },
        {
            "id": "GP3",
            "title": "透明性と公開報告",
            "checkpoints": [
                "モデルカードが公開されている",
                "安全性評価結果が報告されている",
                "重大インシデントの報告体制がある",
            ],
        },
        {
            "id": "GP4",
            "title": "責任ある情報共有",
            "checkpoints": [
                "脆弱性情報を適切に共有している",
                "安全性ベストプラクティスを共有している",
                "業界団体や規制当局との協力体制がある",
            ],
        },
        {
            "id": "GP5",
            "title": "AIガバナンスポリシーの策定",
            "checkpoints": [
                "包括的なAI利用方針が策定されている",
                "プライバシーポリシーが整備されている",
                "定期的なポリシーレビューが実施されている",
            ],
        },
        {
            "id": "GP6",
            "title": "セキュリティの確保",
            "checkpoints": [
                "サイバーセキュリティ対策が実装されている",
                "モデルの不正アクセス防止策がある",
                "脆弱性管理プロセスがある",
            ],
        },
        {
            "id": "GP7",
            "title": "コンテンツの電子透かし",
            "checkpoints": [
                "AI生成コンテンツの識別手段がある",
                "電子透かしまたはメタデータが付与されている",
                "C2PA等の標準を採用している",
            ],
        },
        {
            "id": "GP8",
            "title": "安全研究への投資",
            "checkpoints": [
                "安全性に関する研究開発を行っている",
                "学術機関との連携がある",
                "安全性研究の成果を公開している",
            ],
        },
        {
            "id": "GP9",
            "title": "社会的課題への対応",
            "checkpoints": [
                "気候変動等の社会的課題への活用を検討している",
                "デジタルデバイドの解消に取り組んでいる",
                "多様なステークホルダーの意見を反映している",
            ],
        },
        {
            "id": "GP10",
            "title": "国際標準の策定支援",
            "checkpoints": [
                "国際標準化活動に参加している",
                "相互運用性の確保に取り組んでいる",
                "国際的な規制調和を支持している",
            ],
        },
        {
            "id": "GP11",
            "title": "データ入力の適切性確保",
            "checkpoints": [
                "学習データの品質管理を行っている",
                "著作権への配慮がされている",
                "個人情報保護が徹底されている",
            ],
        },
    ]

    def check_all_principles(
        self, responses: dict[str, list[bool]]
    ) -> dict:
        """全指導原則への準拠をチェックする"""
        results = {
            "principles": [],
            "overall_score": 0.0,
            "total_checkpoints": 0,
            "passed_checkpoints": 0,
        }

        for principle in self.GUIDING_PRINCIPLES:
            answers = responses.get(principle["id"], [])
            checkpoints = principle["checkpoints"]

            passed = sum(
                1 for i, a in enumerate(answers)
                if i < len(checkpoints) and a
            )
            total = len(checkpoints)

            results["total_checkpoints"] += total
            results["passed_checkpoints"] += passed

            results["principles"].append({
                "id": principle["id"],
                "title": principle["title"],
                "compliance_rate": passed / total if total > 0 else 0,
                "passed": passed,
                "total": total,
            })

        results["overall_score"] = (
            results["passed_checkpoints"] / results["total_checkpoints"]
            if results["total_checkpoints"] > 0 else 0
        )

        return results
```

---

## 6. 説明可能 AI (XAI) の実装

### 6.1 説明可能性の手法

```python
# コード例: LLM 出力の説明可能性の実装
class LLMExplainability:
    """LLM の出力に対する説明可能性を提供する"""

    def __init__(self, model, explanation_model=None):
        self.model = model
        self.explanation_model = explanation_model or model

    async def generate_with_explanation(
        self, prompt: str, system_prompt: str = ""
    ) -> dict:
        """説明付きの応答を生成する"""
        # 1. 通常の応答を生成
        response = await self.model.generate(
            prompt, system_prompt=system_prompt
        )

        # 2. 応答の根拠を生成
        explanation_prompt = f"""以下の質問と応答について、応答の根拠を説明してください。

質問: {prompt}
応答: {response}

根拠を以下の形式で説明してください:
1. この応答の主な根拠は何ですか？
2. どのような情報に基づいていますか？
3. この応答の確信度はどの程度ですか？
4. 代替的な応答はありますか？
5. この応答の限界は何ですか？"""

        explanation = await self.explanation_model.generate(
            explanation_prompt
        )

        # 3. 信頼度の推定
        confidence_prompt = f"""以下の応答の信頼度を0.0-1.0のスケールで評価してください。
応答: {response}
評価基準:
- 事実に基づいているか
- 曖昧さがないか
- 一般的に合意されている内容か

信頼度（数値のみ）:"""

        confidence_str = await self.explanation_model.generate(
            confidence_prompt
        )
        try:
            confidence = float(confidence_str.strip())
        except ValueError:
            confidence = 0.5

        return {
            "response": response,
            "explanation": explanation,
            "confidence": confidence,
            "metadata": {
                "model": self.model.__class__.__name__,
                "timestamp": datetime.now().isoformat(),
                "prompt_length": len(prompt),
                "response_length": len(response),
            },
        }

    async def generate_counterfactual_explanation(
        self, prompt: str, response: str
    ) -> dict:
        """反事実的説明を生成する — 入力がどう変われば結果が変わるか"""
        cf_prompt = f"""以下の質問と応答について、反事実的説明を提供してください。
つまり、質問をどのように変えれば、異なる応答が得られるかを説明してください。

質問: {prompt}
応答: {response}

以下を説明してください:
1. 質問のどの部分を変えると、異なる結論になりますか？
2. 具体的にどのような変更が必要ですか？
3. 変更後にどのような応答が予想されますか？"""

        counterfactual = await self.explanation_model.generate(cf_prompt)

        return {
            "original_prompt": prompt,
            "original_response": response,
            "counterfactual_explanation": counterfactual,
        }
```

---

## 7. 環境影響とサステナビリティ

### 7.1 AI の環境影響評価

```python
# コード例: AI モデルの環境影響評価
@dataclass
class CarbonFootprintEstimate:
    """AI モデルの炭素排出量推定"""
    training_kwh: float
    training_co2_kg: float
    inference_kwh_per_1k_requests: float
    inference_co2_kg_per_1k_requests: float
    total_annual_co2_kg: float
    equivalent_car_km: float
    equivalent_trees_needed: int

class AIEnvironmentalImpact:
    """AI モデルの環境影響を評価する"""

    # 地域別の炭素強度 (gCO2/kWh)
    CARBON_INTENSITY = {
        "us_average": 390,
        "eu_average": 230,
        "japan": 470,
        "france": 56,   # 原子力が多い
        "norway": 17,   # 水力が多い
        "india": 720,
        "china": 550,
    }

    # GPU の消費電力推定 (W)
    GPU_POWER = {
        "A100_80GB": 300,
        "H100": 700,
        "A10G": 150,
        "V100": 250,
    }

    def estimate_training_footprint(
        self,
        gpu_type: str,
        num_gpus: int,
        training_hours: float,
        region: str = "us_average",
        pue: float = 1.1,  # Power Usage Effectiveness
    ) -> CarbonFootprintEstimate:
        """学習時の炭素排出量を推定する"""
        gpu_power_w = self.GPU_POWER.get(gpu_type, 300)
        carbon_intensity = self.CARBON_INTENSITY.get(region, 390)

        # 学習時の電力消費 (kWh)
        training_kwh = (
            gpu_power_w * num_gpus * training_hours * pue / 1000
        )

        # CO2 排出量 (kg)
        training_co2_kg = training_kwh * carbon_intensity / 1000

        # 推論時の推定（1000リクエストあたり）
        inference_kwh = gpu_power_w * 0.001 * pue / 1000  # 1リクエスト≒1秒と仮定
        inference_co2_kg = inference_kwh * carbon_intensity / 1000

        # 年間推定（1日10万リクエストを仮定）
        annual_requests = 100_000 * 365
        annual_inference_co2 = (
            inference_co2_kg * annual_requests / 1000
        )
        total_annual_co2 = training_co2_kg + annual_inference_co2

        return CarbonFootprintEstimate(
            training_kwh=training_kwh,
            training_co2_kg=training_co2_kg,
            inference_kwh_per_1k_requests=inference_kwh * 1000,
            inference_co2_kg_per_1k_requests=inference_co2_kg * 1000,
            total_annual_co2_kg=total_annual_co2,
            equivalent_car_km=total_annual_co2 / 0.12,  # 120g/km
            equivalent_trees_needed=int(total_annual_co2 / 22),  # 22kg/年/木
        )

    def generate_environmental_report(
        self, estimate: CarbonFootprintEstimate, model_name: str
    ) -> str:
        """環境影響レポートを生成する"""
        return f"""
# 環境影響レポート: {model_name}

## 学習フェーズ
- 消費電力: {estimate.training_kwh:,.1f} kWh
- CO2排出量: {estimate.training_co2_kg:,.1f} kg

## 推論フェーズ（1,000リクエストあたり）
- 消費電力: {estimate.inference_kwh_per_1k_requests:.4f} kWh
- CO2排出量: {estimate.inference_co2_kg_per_1k_requests:.4f} kg

## 年間推定
- 総CO2排出量: {estimate.total_annual_co2_kg:,.1f} kg
- 自動車走行距離換算: {estimate.equivalent_car_km:,.0f} km
- 必要な樹木本数（吸収量換算）: {estimate.equivalent_trees_needed:,} 本

## 排出削減の推奨事項
1. 再生可能エネルギー比率の高い地域でのホスティング
2. モデルの蒸留による推論効率の向上
3. バッチ推論の活用によるGPU利用率の最適化
4. カーボンオフセットプログラムへの参加
"""
```

---

## 8. アンチパターン

### アンチパターン 1: 「倫理ウォッシング」

```
[誤り] AI倫理方針を策定するだけで、実際の開発プロセスに反映しない

  「当社はAI倫理を重視します」（ウェブサイトに掲載）
   → 開発チームは方針の存在すら知らない
   → リスク評価プロセスが存在しない
   → インシデント発生時に初めて方針を読む

[正解] ガバナンスを開発プロセスに組み込む
  1. 全プロジェクトで AIIA (影響評価) を必須化
  2. CI/CD パイプラインに安全性テストを組み込む
  3. 定期的な研修とケーススタディ共有
  4. インシデントレポートの公開と改善ループ
```

### アンチパターン 2: 「過剰規制による萎縮」

```
[誤り] リスクを恐れて全てのAIプロジェクトに最高レベルの規制を適用する

問題点:
  - 低リスクのプロジェクトまで倫理委員会の承認に数ヶ月待ち
  - イノベーション速度の致命的な低下
  - チームのモチベーション低下
  - 競合他社に対する競争力の喪失

[正解] リスクベースのアプローチ
  - Low Risk: セルフチェックリスト (1日)
  - Medium Risk: チームリーダーの承認 (1週間)
  - High Risk: 倫理委員会の審査 (2-4週間)
  - Unacceptable Risk: 中止判断

  リスクレベルに応じてプロセスの厳格さを調整する
```

### アンチパターン 3: 「チェックリスト形骸化」

```
[誤り] コンプライアンスチェックリストを機械的に「全てYes」で埋める

問題点:
  - 実質的なリスク評価が行われていない
  - 監査時に根拠を示せない
  - 真のリスクが見落とされる
  - 法的責任が発生した場合に不十分とみなされる

[正解] エビデンスベースのコンプライアンス
  - 各チェック項目に具体的なエビデンスを紐付ける
  - 第三者（社内の別チーム or 外部）によるレビューを実施
  - 「No」の場合のリスク受容判断を文書化
  - 定期的なチェックリスト自体のレビューと更新
```

### アンチパターン 4: 「規制追従のみの姿勢」

```
[誤り] 法律で要求される最低限の対応のみ行い、自主的な取り組みを行わない

問題点:
  - 法規制は技術の進歩に追いつかない
  - 新しいリスクが規制化されるまで対処されない
  - レピュテーションリスクが管理されない
  - ユーザーの信頼を失う

[正解] プロアクティブなガバナンス
  - 業界のベストプラクティスを積極的に取り入れる
  - 将来の規制を予測した対応を行う
  - 社会的責任としてのAI倫理への取り組み
  - 透明性レポートの自主的な公開
```

### アンチパターン 5: 「ガバナンスの属人化」

```
[誤り] AI ガバナンスが特定の個人（AI倫理責任者等）に依存している

問題点:
  - その人が退職すると体制が崩壊する
  - 知識の偏りにより盲点が生まれる
  - スケーラビリティがない
  - バス因子（Bus Factor）が1

[正解] 組織的・制度的なガバナンス体制
  - ガバナンスプロセスの文書化と標準化
  - 複数メンバーによる委員会制の採用
  - 各チームへの安全性チャンピオンの配置
  - ナレッジベースとトレーニングプログラムの整備
  - サクセッションプラン（後任者計画）の策定
```

---

## 9. 実務ユースケース

### ユースケース 1: 金融機関の AI ガバナンス体制構築

```python
# コード例: 金融機関向け AI ガバナンス体制のテンプレート
class FinancialAIGovernance:
    """金融機関向けの AI ガバナンス体制"""

    # 金融庁「AI 利用に関する原則」への対応
    FSA_PRINCIPLES = {
        "governance": "経営陣の責任のもとでの AI ガバナンス体制の構築",
        "fairness": "顧客への公正な取り扱いの確保",
        "transparency": "AI の利用に関する適切な説明",
        "data_quality": "データの品質と適切な管理",
        "risk_management": "AI 特有のリスクの適切な管理",
    }

    def generate_governance_framework(self, org_info: dict) -> dict:
        """ガバナンスフレームワークを生成する"""
        framework = {
            "organization": org_info["name"],
            "effective_date": datetime.now().isoformat(),
            "layers": {
                "board_level": {
                    "responsibilities": [
                        "AI戦略の承認",
                        "AIリスクアペタイトの設定",
                        "AI倫理方針の承認",
                        "重大インシデントの報告受領",
                    ],
                    "frequency": "四半期",
                },
                "management_level": {
                    "responsibilities": [
                        "AI倫理委員会の運営",
                        "ハイリスクAIシステムの承認",
                        "リスク管理フレームワークの監督",
                        "インシデント対応の指揮",
                    ],
                    "frequency": "月次",
                },
                "operational_level": {
                    "responsibilities": [
                        "AIIAの実施",
                        "安全性テストの実行",
                        "監視とモニタリング",
                        "インシデントの初期対応",
                    ],
                    "frequency": "日次",
                },
            },
            "three_lines_of_defense": {
                "first_line": {
                    "name": "AI開発・運用チーム",
                    "role": "リスクのオーナーシップ、日常的なリスク管理",
                },
                "second_line": {
                    "name": "リスク管理部門・コンプライアンス部門",
                    "role": "リスクフレームワークの策定、独立したモニタリング",
                },
                "third_line": {
                    "name": "内部監査部門",
                    "role": "ガバナンス体制の独立した評価",
                },
            },
        }

        return framework

    def generate_model_risk_management(self, model_info: dict) -> dict:
        """モデルリスク管理（MRM）フレームワーク"""
        return {
            "model_inventory": {
                "model_id": model_info.get("id"),
                "model_name": model_info.get("name"),
                "risk_tier": model_info.get("risk_tier", "medium"),
                "owner": model_info.get("owner"),
                "validation_date": model_info.get("validation_date"),
            },
            "validation_requirements": {
                "tier_1_critical": {
                    "independent_validation": True,
                    "frequency": "年次",
                    "backtesting": True,
                    "stress_testing": True,
                    "bias_testing": True,
                },
                "tier_2_significant": {
                    "independent_validation": True,
                    "frequency": "18ヶ月ごと",
                    "backtesting": True,
                    "stress_testing": False,
                    "bias_testing": True,
                },
                "tier_3_low": {
                    "independent_validation": False,
                    "frequency": "2年ごと",
                    "backtesting": False,
                    "stress_testing": False,
                    "bias_testing": True,
                },
            },
            "ongoing_monitoring": {
                "performance_metrics": ["accuracy", "f1", "auc"],
                "drift_detection": True,
                "threshold_alerts": True,
                "reporting_frequency": "月次",
            },
        }
```

---

## 10. FAQ

### Q1: 小規模スタートアップでも AI ガバナンスは必要ですか？

**A:** はい。ただし規模に応じた軽量なアプローチで十分です。

- **最低限**: AI 利用方針の策定（1ページ）、リスクチェックリスト（10項目）
- **中程度**: 影響評価テンプレートの導入、定期的なレビュー（四半期）
- **将来への備え**: EU AI Act 等の規制が中小企業にも適用される場合に備え、早期にプロセスを構築しておくことが競争優位になる

投資家や顧客からの信頼獲得の観点でも、早期のガバナンス体制構築は有効です。

### Q2: AI のインシデント対応はどう設計すべきですか？

**A:** 以下のフレームワークを推奨します。

1. **検知**: 監視システムによる自動検知 + ユーザー報告チャネル
2. **トリアージ**: 影響範囲と重大度の即時評価（30分以内）
3. **封じ込め**: 該当機能の停止またはフォールバック（人間による代替処理）
4. **根本原因分析**: データ起因か、モデル起因か、システム起因かの特定
5. **修正**: モデルの再学習、ガードレールの追加、データの修正
6. **事後レビュー**: インシデントレポートの作成と再発防止策の実装

EU AI Act では、ハイリスクAIのインシデント報告が義務化されているため、報告体制も整備が必要です。

### Q3: GDPR と AI 規制の関係は？

**A:** GDPR と EU AI Act は相互補完的です。

- **GDPR**: 個人データの処理に関する規制。AI の学習データ・推論データに適用
- **EU AI Act**: AI システムの開発・運用に関する規制。データに加えて、モデルの安全性・透明性を規制
- **重複領域**: プロファイリング、自動意思決定（GDPR第22条 + AI Act のハイリスクAI）
- **実務上**: 両方の要件を満たす統合的なコンプライアンスフレームワークが必要

### Q4: AI ガバナンスの ROI をどう示しますか？

**A:** 以下の観点で定量化できます。

- **リスク削減**: インシデントの防止による損失回避（罰金、訴訟、レピュテーション損失）
- **効率化**: 標準化されたプロセスによる開発速度の向上
- **競争優位**: 信頼されるAIブランドの構築による顧客獲得
- **規制対応コスト**: 事後対応と比べた事前対応のコスト優位性
- **具体例**: EU AI Act の罰則は最大3,500万EUR。ガバナンス体制の構築コストと比較すると、投資対効果は明確

### Q5: オープンソースAIモデルを使う場合のガバナンスは？

**A:** オープンソースモデルにも適切なガバナンスが必要です。

- **ライセンスの確認**: 商用利用可能か、派生物の制約はあるか
- **安全性評価**: 独自のレッドチームテストを実施（開発者の評価を鵜呑みにしない）
- **脆弱性管理**: コミュニティで報告される脆弱性を継続的にモニタリング
- **ガードレール追加**: モデル自体の安全性に依存せず、追加のガードレールを実装
- **EU AI Act**: オープンソースGPAIにも一定の義務がある（システミックリスクモデルは免除対象外）

### Q6: AI ガバナンスと既存のIT ガバナンスの関係は？

**A:** AI ガバナンスは既存の IT ガバナンスを拡張するものです。

- **共通点**: リスク管理、セキュリティ、監査、変更管理
- **AI 固有**: バイアス管理、説明可能性、人間の監視、モデルライフサイクル管理
- **統合アプローチ**: 既存の ITSM/ITIL フレームワークに AI 固有の要素を追加する形が効率的
- **避けるべきこと**: AI ガバナンスを完全に別組織で運用すると、サイロ化のリスク

---

## 11. まとめ

| 領域 | 対応事項 | ツール/手法 | 頻度 |
|------|----------|------------|------|
| 法規制対応 | リスク分類と義務の把握 | AI Act チェックリスト | プロジェクト開始時 |
| GPAI対応 | 技術文書・データ要約の準備 | コンプライアンスチェッカー | モデルリリース時 |
| 影響評価 | AIIA の実施 | テンプレート + レビュー | プロジェクトごと |
| 倫理委員会 | レビューと承認 | 委員会運営フレームワーク | 申請ごと |
| 透明性 | モデルカードの作成 | 自動生成ツール | モデル更新時 |
| 監査 | ログの記録と保持 | 監査ログシステム | 常時 |
| データガバナンス | データリネージとPII管理 | データガバナンスツール | 常時 |
| 環境影響 | 炭素排出量の推定と報告 | 環境影響評価ツール | 年次 |
| 監視 | バイアス・ドリフト検知 | ダッシュボード | 常時 |
| レビュー | ガバナンス体制の見直し | 定期レビュー会議 | 四半期 |
| インシデント | 対応プロセスの整備 | ランブック | インシデント発生時 |
| 研修 | AI倫理リテラシーの向上 | eラーニング + 実践研修 | 年次以上 |

---

## 次に読むべきガイド

- [AI セーフティ](./00-ai-safety.md) — アライメント・レッドチームの技術的手法
- [責任ある AI](../../../ai-analysis-guide/docs/03-applied/03-responsible-ai.md) — 公平性・説明可能性・プライバシーの実装
- [エージェントの安全性](../../../custom-ai-agents/docs/04-production/01-safety.md) — AI エージェント固有のガバナンス課題

---

## 参考文献

1. European Parliament. (2024). "Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act)." *Official Journal of the European Union*. https://eur-lex.europa.eu/eli/reg/2024/1689/oj
2. 経済産業省. (2024). 「AI事業者ガイドライン（第1.0版）」. https://www.meti.go.jp/shingikai/mono_info_service/ai_shakai_jisso/
3. OECD. (2019). "Recommendation of the Council on Artificial Intelligence." *OECD Legal Instruments*. https://legalinstruments.oecd.org/en/instruments/OECD-LEGAL-0449
4. The White House. (2023). "Executive Order on the Safe, Secure, and Trustworthy Development and Use of Artificial Intelligence." https://www.whitehouse.gov/briefing-room/presidential-actions/2023/10/30/executive-order-on-the-safe-secure-and-trustworthy-development-and-use-of-artificial-intelligence/
5. G7. (2023). "Hiroshima Process International Code of Conduct for Organizations Developing Advanced AI Systems." https://www.mofa.go.jp/ecm/ec/page5e_000076.html
6. 金融庁. (2024). 「AI利用に関する原則」. https://www.fsa.go.jp/
7. NIST. (2023). "AI Risk Management Framework (AI RMF 1.0)." https://airc.nist.gov/AI_RMF_Pub
