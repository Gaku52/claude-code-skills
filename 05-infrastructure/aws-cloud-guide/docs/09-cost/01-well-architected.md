# Well-Architected Framework — 6 つの柱とレビュープロセス

> AWS Well-Architected Framework の 6 つの柱を理解し、自社ワークロードを体系的にレビュー・改善するための実践ガイド。

---

## この章で学ぶこと

1. **6 つの柱** それぞれの設計原則とベストプラクティス
2. **Well-Architected Tool** を使ったワークロードレビューの進め方
3. **改善の優先順位付け** と継続的なアーキテクチャ改善プロセス

---

## 1. Well-Architected Framework の全体像

### 1.1 6 つの柱

```
┌──────────────────────────────────────────────────────────┐
│            AWS Well-Architected Framework                 │
│                   6 つの柱 (Pillars)                      │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ 1. 運用上の  │  │ 2. セキュリ │  │ 3. 信頼性    │   │
│  │   優秀性     │  │   ティ      │  │ (Reliability)│   │
│  │ (Operational │  │ (Security)  │  │              │   │
│  │  Excellence) │  │             │  │              │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ 4. パフォー  │  │ 5. コスト   │  │ 6. 持続可    │   │
│  │  マンス効率  │  │   最適化    │  │   能性       │   │
│  │ (Performance │  │ (Cost       │  │(Sustainability│  │
│  │  Efficiency) │  │ Optimization│  │             ) │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└──────────────────────────────────────────────────────────┘
```

### 1.2 各柱の関係性

```
                   ┌─────────────────┐
                   │   ビジネス価値   │
                   └────────┬────────┘
                            │
           ┌────────────────┼────────────────┐
           │                │                │
     ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
     │ セキュリティ│   │  信頼性   │   │ パフォー  │
     │ (基盤)     │   │ (基盤)    │   │  マンス   │
     └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
           │                │                │
           └────────────────┼────────────────┘
                            │
           ┌────────────────┼────────────────┐
           │                │                │
     ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
     │ 運用上の   │   │  コスト   │   │ 持続可    │
     │ 優秀性     │   │  最適化   │   │ 能性      │
     │ (横断)     │   │ (最適化)  │   │ (最適化)  │
     └───────────┘   └───────────┘   └───────────┘
```

---

## 2. 6 つの柱の詳細

### 2.1 柱 1: 運用上の優秀性（Operational Excellence）

```yaml
# 設計原則
principles:
  - 運用をコードとして管理 (IaC)
  - 小さく可逆的な変更を頻繁に行う
  - 運用手順を頻繁に改善する
  - 障害を予測する
  - 全ての運用上の障害から学ぶ

# チェックリスト例
checklist:
  organization:
    - チームの責任範囲が明確か
    - 運用の優先順位がビジネス目標と整合しているか
  prepare:
    - ワークロードの可観測性が設計されているか
    - デプロイ戦略 (Blue/Green, Canary) が定義されているか
  operate:
    - ランブックとプレイブックが整備されているか
    - ダッシュボードとアラートが適切に設定されているか
  evolve:
    - ポストモーテム (振り返り) プロセスがあるか
    - 改善項目がバックログに追加されているか
```

### 2.2 柱 2: セキュリティ（Security）

```yaml
principles:
  - 強力な ID 基盤を実装する
  - トレーサビリティを有効にする
  - 全レイヤーにセキュリティを適用する
  - セキュリティのベストプラクティスを自動化する
  - 転送中および保管中のデータを保護する
  - データに人の手を触れさせない
  - セキュリティイベントに備える

checklist:
  identity:
    - MFA が全 IAM ユーザーに強制されているか
    - ルートアカウントが日常業務で使われていないか
    - 最小権限の原則が適用されているか
  detection:
    - CloudTrail が全リージョンで有効か
    - GuardDuty が有効か
    - Security Hub が設定されているか
  protection:
    - VPC フローログが有効か
    - WAF が設定されているか
    - データが暗号化されているか (KMS)
```

### 2.3 柱 3: 信頼性（Reliability）

```yaml
principles:
  - 障害から自動的に復旧する
  - 復旧手順をテストする
  - 水平にスケールする
  - キャパシティの推測をやめる
  - 自動化で変更を管理する

checklist:
  foundations:
    - サービスクォータが適切に設定されているか
    - ネットワークトポロジが冗長化されているか
  workload_architecture:
    - マイクロサービス or SOA で障害分離ができているか
    - 分散システムでの障害処理が実装されているか
  change_management:
    - Auto Scaling が設定されているか
    - 変更がモニタリングされているか
  failure_management:
    - バックアップと DR 戦略が定義されているか
    - RTO/RPO が明確に定義されているか
```

### 2.4 6 つの柱のベストプラクティス要約

```python
# Well-Architected レビューの自動化スクリプト例
import boto3

def run_well_architected_review():
    """Well-Architected Tool でワークロードレビューを自動化"""
    wa_client = boto3.client("wellarchitected", region_name="ap-northeast-1")

    # ワークロード作成
    workload = wa_client.create_workload(
        WorkloadName="MyApp Production",
        Description="Main production workload",
        Environment="PRODUCTION",
        ArchitecturalDesign="https://wiki.example.com/architecture",
        ReviewOwner="architect@example.com",
        Lenses=["wellarchitected"],          # AWS Well-Architected Lens
        AwsRegions=["ap-northeast-1"],
        Tags={"Project": "myapp"},
    )

    workload_id = workload["WorkloadId"]

    # 質問一覧の取得
    answers = wa_client.list_answers(
        WorkloadId=workload_id,
        LensAlias="wellarchitected",
        PillarId="operationalExcellence",    # 柱を指定
    )

    for answer in answers["AnswerSummaries"]:
        print(f"Q: {answer['QuestionTitle']}")
        print(f"  Risk: {answer.get('Risk', 'UNANSWERED')}")
        print()

    return workload_id
```

### 2.5 Lens の活用

```bash
# 利用可能な Lens の一覧
aws wellarchitected list-lenses --query 'LensSummaries[*].{Name:LensName,Alias:LensAlias}'

# よく使う Lens:
# - wellarchitected          : AWS Well-Architected (デフォルト)
# - serverless               : Serverless Applications Lens
# - saas                     : SaaS Lens
# - foundational-technical-review : FTR Lens (APN パートナー向け)
# - machine-learning         : Machine Learning Lens
```

---

## 3. Well-Architected Tool でのレビュー

### 3.1 レビュープロセス

```
┌─────────────────────────────────────────────────────────┐
│         Well-Architected Review プロセス                  │
│                                                         │
│  Phase 1: 準備 (1-2日)                                  │
│  ┌───────────────────────────────────────┐              │
│  │ - アーキテクチャ図の準備               │              │
│  │ - ステークホルダーの特定               │              │
│  │ - 既知のリスクの整理                   │              │
│  └──────────────────┬────────────────────┘              │
│                     ▼                                   │
│  Phase 2: レビュー (2-5日)                               │
│  ┌───────────────────────────────────────┐              │
│  │ - 6つの柱ごとに質問に回答             │              │
│  │ - ベストプラクティスとの差分を特定     │              │
│  │ - リスクレベルの評価                   │              │
│  │   (High Risk / Medium Risk / No Risk) │              │
│  └──────────────────┬────────────────────┘              │
│                     ▼                                   │
│  Phase 3: 改善計画 (1-2日)                               │
│  ┌───────────────────────────────────────┐              │
│  │ - High Risk 項目の改善策を策定         │              │
│  │ - 優先順位とマイルストーンを設定       │              │
│  │ - 改善項目を Jira/Backlog に登録       │              │
│  └──────────────────┬────────────────────┘              │
│                     ▼                                   │
│  Phase 4: 実行と再レビュー (継続)                        │
│  ┌───────────────────────────────────────┐              │
│  │ - 改善を実施                           │              │
│  │ - 四半期ごとに再レビュー               │              │
│  │ - マイルストーンで進捗を記録           │              │
│  └───────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────┘
```

### 3.2 改善計画テンプレート

```markdown
## Well-Architected 改善計画

### High Risk Items (最優先)

| # | 柱 | 質問 | 現状のリスク | 改善アクション | 担当 | 期限 |
|---|---|------|------------|---------------|------|------|
| 1 | セキュリティ | 認証情報の管理 | ハードコード | Secrets Manager 導入 | @security | 2W |
| 2 | 信頼性 | バックアップ | 手動・不定期 | AWS Backup 自動化 | @infra | 3W |
| 3 | 運用 | モニタリング | ログ未収集 | CloudWatch + X-Ray | @sre | 4W |

### Medium Risk Items (次フェーズ)

| # | 柱 | 質問 | 改善アクション | 期限 |
|---|---|------|---------------|------|
| 4 | コスト | ライトサイジング | Compute Optimizer 適用 | Q2 |
| 5 | パフォーマンス | キャッシュ戦略 | ElastiCache 導入 | Q2 |
```

---

## 4. 比較表

### 4.1 6 つの柱 概要比較

| 柱 | 焦点 | 主要 AWS サービス | KPI 例 |
|----|------|------------------|--------|
| **運用上の優秀性** | 運用の自動化と継続改善 | CloudFormation, Systems Manager, CloudWatch | デプロイ頻度, MTTR |
| **セキュリティ** | データと資産の保護 | IAM, KMS, GuardDuty, Security Hub | 未対応の検出結果数 |
| **信頼性** | 障害復旧と可用性 | Route 53, ELB, Auto Scaling, Backup | 可用性 %, RTO/RPO |
| **パフォーマンス効率** | リソースの効率的な使用 | CloudFront, ElastiCache, Lambda | レイテンシ P99 |
| **コスト最適化** | 無駄の排除と価値最大化 | Cost Explorer, Budgets, Savings Plans | 月間コスト, SP カバー率 |
| **持続可能性** | 環境への影響最小化 | Graviton, Spot, サーバーレス | CO2 排出量推定 |

### 4.2 レビュー方式比較

| 方式 | 対象 | 所要時間 | コスト | 推奨場面 |
|------|------|---------|--------|---------|
| **セルフレビュー** | 自チーム | 2-5日 | 無料 | 定期レビュー |
| **AWS SA レビュー** | SA 支援 | 1-2週間 | 無料（Enterprise Support） | 初回レビュー |
| **パートナーレビュー** | APN パートナー | 2-4週間 | 有料 | 大規模ワークロード |
| **AWS Well-Architected Tool** | ツール支援 | 1-3日 | 無料 | 全ケースで利用推奨 |

---

## 5. アンチパターン

### 5.1 レビューを一度やって終わり

```
NG:
  リリース前に Well-Architected レビュー実施
  → "完了" として棚上げ
  → 1年後: アーキテクチャが変わり、リスクが再発

OK:
  四半期ごとの定期レビューサイクル
  ┌──────────────────────────────────┐
  │  Q1: フルレビュー                │
  │  Q2: High Risk 改善確認          │
  │  Q3: フルレビュー（再評価）      │
  │  Q4: 年間振り返り + 次年度計画   │
  └──────────────────────────────────┘
```

### 5.2 全ての柱を均等に扱う

```
NG:
  6つの柱 × 均等リソース配分
  → セキュリティのクリティカルな問題が後回しに

OK:
  リスクベースの優先順位付け
  1. セキュリティの High Risk → 即対応（1-2週間）
  2. 信頼性の High Risk → 次スプリント
  3. 運用の Medium Risk → バックログ
  4. コスト/パフォーマンス → 四半期計画
```

---

## 6. FAQ

### Q1. Well-Architected Review は誰が主導すべき？

**A.** ワークロードのテックリードまたはアーキテクトが主導し、開発・運用・セキュリティの各チームメンバーが参加する。AWS の Solutions Architect に初回の支援を依頼すると効率的。Enterprise Support 契約があれば無料で SA の支援を受けられる。

### Q2. 小規模なスタートアップでも Well-Architected は必要？

**A.** 規模に関わらず有用。ただし全ての質問に完璧に対応する必要はない。まずセキュリティと信頼性の High Risk 項目に集中し、ビジネスの成長に合わせて他の柱も強化していく段階的アプローチが現実的。

### Q3. Well-Architected Tool の結果は AWS に共有される？

**A.** 通常は共有されない。ただし「AWS Solutions Architect とワークロードを共有」を明示的に有効にした場合のみ、担当 SA がレビュー結果にアクセスできる。データは暗号化されアカウント所有者が管理権限を持つ。

---

## 7. まとめ

| 項目 | ポイント |
|------|---------|
| **6 つの柱** | 運用、セキュリティ、信頼性、パフォーマンス、コスト、持続可能性 |
| **レビューツール** | AWS Well-Architected Tool で質問に回答し、リスクを可視化 |
| **優先順位** | セキュリティ > 信頼性 > 運用 > パフォーマンス > コスト > 持続可能性 |
| **継続性** | 四半期ごとの定期レビュー、マイルストーンで進捗管理 |
| **Lens** | ワークロード種別に応じた専用 Lens を活用 |

---

## 次に読むべきガイド

- [00-cost-optimization.md](./00-cost-optimization.md) — コスト最適化の具体的な実践
- セキュリティガイド — IAM / KMS / WAF の詳細設計
- 信頼性ガイド — マルチ AZ / DR 戦略

---

## 参考文献

1. **AWS公式ドキュメント** — "AWS Well-Architected Framework" — https://docs.aws.amazon.com/wellarchitected/latest/framework/
2. **AWS公式ドキュメント** — "AWS Well-Architected Tool User Guide" — https://docs.aws.amazon.com/wellarchitected/latest/userguide/
3. **AWS公式ホワイトペーパー** — "AWS Well-Architected Framework: Six Pillars" — https://aws.amazon.com/architecture/well-architected/
4. **AWS公式ブログ** — "Well-Architected Labs" — https://www.wellarchitectedlabs.com/
