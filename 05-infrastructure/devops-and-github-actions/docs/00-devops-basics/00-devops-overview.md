# DevOps概要

> ソフトウェアの開発(Dev)と運用(Ops)を統合し、価値提供の速度と信頼性を最大化する文化・プラクティス体系

## この章で学ぶこと

1. DevOpsの文化的背景と5つの原則(CALMS)を理解する
2. DORA メトリクスによるパフォーマンス計測手法を習得する
3. DevOps導入のロードマップとアンチパターンを把握する

---

## 1. DevOpsとは何か

### 1.1 歴史的背景

従来のソフトウェア開発では、開発チームと運用チームが分断された組織構造(サイロ)で働いていた。開発チームは「新機能を早く出す」ことを、運用チームは「システムを安定させる」ことを目標とし、両者の間に構造的な対立が生まれていた。

```
従来のウォーターフォール型リリースサイクル:

企画 → 要件定義 → 設計 → 実装 → テスト → リリース → 運用
 |                                              |
 +--- 開発チームの責任範囲 ---+--- 運用チームの責任範囲 ---+
                              ^
                           "壁越し"
                        (Wall of Confusion)
```

2008年のAgile Conference、2009年のDevOpsdays Ghent で Patrick Debois が「DevOps」という用語を普及させ、開発と運用の壁を壊す運動が始まった。

### 1.2 DevOpsの定義

DevOpsは単一のツールや役職ではなく、以下の要素を統合した**文化・運動・プラクティス**である。

```
+-----------------------------------------------------------+
|                     DevOps の全体像                         |
+-----------------------------------------------------------+
|                                                           |
|   文化 (Culture)                                          |
|   +---------------------------------------------------+  |
|   | 自動化 (Automation)                                |  |
|   |   +---------------------------------------------+ |  |
|   |   | 計測 (Measurement)                           | |  |
|   |   |   +---------------------------------------+ | |  |
|   |   |   | 共有 (Sharing)                         | | |  |
|   |   |   |   +-------------------------------+   | | |  |
|   |   |   |   | リーン (Lean)                  |   | | |  |
|   |   |   |   +-------------------------------+   | | |  |
|   |   |   +---------------------------------------+ | |  |
|   |   +---------------------------------------------+ |  |
|   +---------------------------------------------------+  |
+-----------------------------------------------------------+
```

---

## 2. CALMS フレームワーク

DevOpsの成熟度を評価する5つの柱が CALMS である。

### 2.1 Culture（文化）

チーム間の信頼、責任共有、失敗から学ぶ姿勢。

```yaml
# 文化の具体例: Blameless Postmortem テンプレート
postmortem:
  title: "2024-01-15 API サーバーダウン"
  severity: SEV-1
  duration: "45 minutes"
  impact: "全ユーザーの API リクエストが失敗"
  timeline:
    - time: "14:00" event: "デプロイ実行"
    - time: "14:05" event: "エラーレート急上昇を検知"
    - time: "14:10" event: "ロールバック開始"
    - time: "14:45" event: "復旧確認"
  root_cause: "未テストの DB マイグレーションスクリプト"
  action_items:
    - "マイグレーションのステージング環境テスト必須化"
    - "カナリーデプロイの導入"
  blame: "個人を責めない。プロセスを改善する。"
```

### 2.2 Automation（自動化）

手作業を排除し、再現性と速度を確保する。

```bash
#!/bin/bash
# 手動デプロイ vs 自動化デプロイの対比

# --- 手動デプロイ（アンチパターン）---
ssh production-server
cd /var/www/app
git pull origin main
npm install
npm run build
pm2 restart app
# 問題: 手順書依存、ヒューマンエラー、再現性なし

# --- 自動化デプロイ ---
# GitHub Actions が PR マージ時に自動実行
# 1. テスト → 2. ビルド → 3. コンテナ作成 → 4. デプロイ → 5. ヘルスチェック
# 全てコードで定義され、バージョン管理される
```

### 2.3 Lean（リーン）

ムダの排除、小さなバッチ、フロー最適化。

```
バッチサイズとリスクの関係:

リスク
  ^
  |        *
  |       *
  |      *
  |    *
  |  *
  | *
  +*-----------> バッチサイズ
  小              大

小さなバッチ = 小さなリスク = 速いフィードバック
```

### 2.4 Measurement（計測）

改善にはデータが必要。計測しないものは改善できない。

```python
# DORA メトリクス計測の擬似コード
class DORAMetrics:
    def deployment_frequency(self, deployments: list, period_days: int) -> str:
        """デプロイ頻度: どのくらいの頻度でデプロイするか"""
        freq = len(deployments) / period_days
        if freq >= 1:
            return "Elite: 1日複数回"
        elif freq >= 1/7:
            return "High: 週1〜月1回"
        elif freq >= 1/30:
            return "Medium: 月1〜半年に1回"
        else:
            return "Low: 半年に1回未満"

    def lead_time_for_changes(self, commit_time, deploy_time) -> str:
        """変更リードタイム: コミットから本番デプロイまでの時間"""
        delta = deploy_time - commit_time
        if delta.total_seconds() < 86400:  # 1日未満
            return "Elite: 1日未満"
        elif delta.days < 7:
            return "High: 1日〜1週間"
        elif delta.days < 30:
            return "Medium: 1週間〜1ヶ月"
        else:
            return "Low: 1ヶ月超"

    def change_failure_rate(self, total_deploys, failed_deploys) -> str:
        """変更障害率: デプロイの何%が障害を引き起こすか"""
        rate = failed_deploys / total_deploys * 100
        if rate <= 5:
            return f"Elite: {rate}% (目標: 0-5%)"
        elif rate <= 15:
            return f"High: {rate}% (目標: 5-15%)"
        elif rate <= 30:
            return f"Medium: {rate}% (目標: 15-30%)"
        else:
            return f"Low: {rate}% (30%超)"

    def time_to_restore(self, incident_start, incident_end) -> str:
        """サービス復旧時間: 障害発生から復旧までの時間"""
        delta = incident_end - incident_start
        if delta.total_seconds() < 3600:
            return "Elite: 1時間未満"
        elif delta.total_seconds() < 86400:
            return "High: 1日未満"
        elif delta.days < 7:
            return "Medium: 1日〜1週間"
        else:
            return "Low: 1週間超"
```

### 2.5 Sharing（共有）

知識、責任、ツールの共有。

```yaml
# 共有のプラクティス例
sharing_practices:
  knowledge:
    - "ADR (Architecture Decision Records)"
    - "Runbook / Playbook"
    - "Tech Radar"
  responsibility:
    - "You build it, you run it"
    - "SRE のエラーバジェット"
    - "共同オンコール"
  tools:
    - "共通 CI/CD パイプライン"
    - "統一監視ダッシュボード"
    - "ChatOps (Slack + Bot)"
```

---

## 3. DORA メトリクス詳解

DORA (DevOps Research and Assessment) は Google Cloud が支援する研究プログラムで、ソフトウェアデリバリーのパフォーマンスを4つのメトリクスで定量化する。

### 3.1 四大メトリクス比較表

| メトリクス | Elite | High | Medium | Low |
|---|---|---|---|---|
| デプロイ頻度 | オンデマンド(1日複数回) | 週1〜月1回 | 月1〜半年に1回 | 半年に1回未満 |
| 変更リードタイム | 1時間未満 | 1日〜1週間 | 1週間〜1ヶ月 | 1ヶ月〜半年 |
| 変更障害率 | 0〜5% | 5〜15% | 15〜30% | 30%超 |
| サービス復旧時間 | 1時間未満 | 1日未満 | 1日〜1週間 | 1週間超 |

### 3.2 DevOps 成熟度モデル比較

| 段階 | プラクティス | ツール例 | 組織の特徴 |
|---|---|---|---|
| Level 0: 手動 | 手動デプロイ、手動テスト | FTP、手作業SSH | サイロ化した組織 |
| Level 1: 部分自動化 | CI導入、自動テスト一部 | Jenkins、基本的なスクリプト | 開発チーム内で自動化 |
| Level 2: CI/CD | 完全CI/CD、IaC導入 | GitHub Actions、Terraform | クロスファンクショナルチーム |
| Level 3: 継続的改善 | カナリーデプロイ、SLO | ArgoCD、Datadog | プラットフォームチーム |
| Level 4: 最適化 | カオスエンジニアリング、ML-Ops | Chaos Monkey、Feature Flags | 学習する組織 |

---

## 4. DevOps の無限ループ

```
          Plan → Code → Build → Test
         ↗                           ↘
    Monitor                           Release
         ↖                           ↙
          Operate ← Deploy ← Stage

+--------------------------------------------------+
|              DevOps ∞ ループ                       |
|                                                    |
|   ┌─────────── Dev ──────────┐                    |
|   │  Plan → Code → Build → Test                  |
|   │  ↑                       │                    |
|   │  │    ┌── Feedback ──┐   ↓                    |
|   │  │    │              │  Release                |
|   │  │    │              │   │                    |
|   │  │    ↓              │   ↓                    |
|   │  Monitor ← Operate ← Deploy                  |
|   │  └─────────── Ops ──────────┘                 |
|   └──────────────────────────────┘                |
+--------------------------------------------------+
```

---

## 5. Three Ways（3つの道）

Gene Kim の「The Phoenix Project」で提唱された DevOps の基本原則。

```
第1の道: フロー (Systems Thinking)
  Dev → Ops → Customer
  左から右へのフローを最大化

第2の道: フィードバック
  Dev ← Ops ← Customer
  右から左へのフィードバックを最大化

第3の道: 継続的学習
  +---→ 実験 → 失敗 → 学習 ---+
  |                             |
  +-------- 繰り返し ←---------+
  リスクを取り、反復し、習熟する
```

---

## 6. アンチパターン

### アンチパターン1: ツール先行型 DevOps

```
悪い例:
  "Kubernetes を導入したから DevOps できている"
  "CI/CD ツールを入れたから DevOps 完了"

問題:
  - 文化の変革なしにツールだけ導入
  - チーム間のサイロが残ったまま
  - ツールが複雑になり逆に生産性低下

改善:
  1. まず文化・プロセスの改善に着手
  2. 小さな自動化から始める
  3. ツールは問題解決の手段として選定
```

### アンチパターン2: DevOps チーム症候群

```
悪い例:
  組織図に「DevOps チーム」を新設し、
  開発チームと運用チームの間に第3のサイロを作る

  Dev Team  →  DevOps Team  →  Ops Team
              (新たな壁)      (依然として壁)

改善:
  - DevOps はチーム名ではなく文化
  - プラットフォームエンジニアリングチームとして
    開発者体験(DX)を向上させるツール・基盤を提供
  - "You build it, you run it" の原則
```

### アンチパターン3: 計測なき改善

```
悪い例:
  "感覚的に速くなった気がする"
  "たぶんデプロイ頻度が上がった"

問題:
  - 改善の根拠がない
  - 投資対効果を説明できない
  - 後退しても気づけない

改善:
  1. DORA メトリクスのベースラインを計測
  2. 改善施策ごとに効果を定量評価
  3. ダッシュボードで可視化・共有
```

---

## 7. FAQ

### Q1: DevOps エンジニアという職種は正しいのか？

DevOps は本来文化・プラクティスであり職種名ではない。しかし市場では「DevOps エンジニア」という職種が定着している。実態としてはインフラ自動化、CI/CD 構築、クラウド運用を担当するエンジニアを指すことが多い。最近では「Platform Engineer」「SRE (Site Reliability Engineer)」というより正確な名称が普及しつつある。

### Q2: DevOps と SRE の違いは何か？

DevOps は文化・原則のフレームワークであり、SRE は Google が生み出した DevOps の具体的実装の1つ。SRE はエラーバジェット、SLI/SLO/SLA、トイル削減など、より体系化されたプラクティスを持つ。DevOps が「何を(What)」すべきかを示し、SRE が「どのように(How)」実現するかを示すと理解できる。

### Q3: 小さなチームでも DevOps は必要か？

必要である。むしろ小さなチームこそ自動化の恩恵が大きい。5人のチームで週2時間の手動デプロイ作業があれば、年間520人時の損失になる。CI/CD を1日で構築すれば、その投資は1週間で回収できる。小さく始めて段階的に成熟度を上げるアプローチが推奨される。

### Q4: DevOps導入にどのくらいの期間がかかるか？

組織の規模と現状の成熟度による。基本的なCI/CDパイプラインは数日で構築できるが、文化の変革には6ヶ月〜2年を要する。DORA メトリクスの「Elite」レベルに到達するには、継続的な改善の積み重ねが必要。まずは3ヶ月で「Quick Win」を出し、組織の信頼を得ることが重要。

---

## まとめ

| 項目 | 要点 |
|---|---|
| DevOps の本質 | 文化・プラクティス・ツールの統合体 |
| CALMS | Culture, Automation, Lean, Measurement, Sharing |
| DORA メトリクス | デプロイ頻度、変更リードタイム、変更障害率、復旧時間 |
| Three Ways | フロー、フィードバック、継続的学習 |
| 最重要原則 | ツールより文化、計測して改善、小さく始める |
| アンチパターン | ツール先行、DevOpsチーム症候群、計測なき改善 |

---

## 次に読むべきガイド

- [CI/CD概念](./01-ci-cd-concepts.md) -- DevOps の中核プラクティスであるCI/CDを深掘り
- [Infrastructure as Code](./02-infrastructure-as-code.md) -- インフラ自動化の具体的手法
- [オブザーバビリティ](../03-monitoring/00-observability.md) -- 計測・監視の実践

---

## 参考文献

1. Gene Kim, Jez Humble, Patrick Debois, John Willis. *The DevOps Handbook*, 2nd Edition. IT Revolution Press, 2021.
2. Nicole Forsgren, Jez Humble, Gene Kim. *Accelerate: The Science of Lean Software and DevOps*. IT Revolution Press, 2018.
3. Google Cloud. "DORA | DevOps Research and Assessment." https://dora.dev/
4. Gene Kim. *The Phoenix Project: A Novel about IT, DevOps, and Helping Your Business Win*. IT Revolution Press, 2013.
5. Atlassian. "DevOps: Breaking the Development-Operations barrier." https://www.atlassian.com/devops
