# システム設計概要

> 大規模システムを「なぜ・どう設計するか」を体系的に学び、設計面接から実務まで通用する思考フレームワークを習得する。

## この章で学ぶこと

1. **システム設計の全体プロセス**: 4ステップ（要件明確化 → 見積もり → 高レベル設計 → 詳細設計）の各フェーズで問うべき質問と成果物を理解する
2. **要件分析の技法**: 機能要件（FR）と非機能要件（NFR）を体系的に切り分け、優先順位をつけ、トレードオフを分析する手法を身につける
3. **設計面接での構造化アプローチ**: 時間配分、質問の仕方、図の描き方、トレードオフの議論方法など、高評価を得るための実践的テクニックを習得する
4. **見積もり（Back-of-the-Envelope Estimation）**: QPS、ストレージ、帯域幅を瞬時に概算し、設計判断の根拠とする力を養う
5. **設計ドキュメントの書き方**: 実務で使える設計文書（Design Doc / RFC）の構成と記載ポイントを学ぶ

---

## 前提知識

このガイドを効果的に学ぶために、以下の知識があると望ましい。

| 前提知識 | 内容 | 参照リンク |
|---------|------|-----------|
| コンピュータネットワーク基礎 | TCP/IP、HTTP、DNS の基本 | [04-web-and-network](../../../../04-web-and-network/) |
| データベース基礎 | RDBMS、SQL、インデックスの基本概念 | [06-data-and-security](../../../../06-data-and-security/) |
| プログラミング基礎 | Python または任意の言語でコードが読めること | [02-programming](../../../../02-programming/) |
| ソフトウェア設計原則 | SOLID、結合度・凝集度の基礎 | [clean-code-principles: 00-principles](../../clean-code-principles/docs/00-principles/) |
| アーキテクチャパターン基礎 | MVC/MVVMなどの基礎概念 | [design-patterns-guide: 04-architectural](../../design-patterns-guide/docs/04-architectural/) |

---

## 1. システム設計とは何か

### 1.1 定義と目的

システム設計（System Design）とは、ビジネス要件を満たすソフトウェアシステムの**アーキテクチャ、コンポーネント、データフロー、インターフェース**を定義する活動である。単にコードを書く能力とは異なり、**スケーラビリティ、信頼性、保守性**を同時に満たす構造を導く力が求められる。

**なぜシステム設計が重要なのか？**

ソフトウェア開発における最も高コストなミスは「間違ったアーキテクチャの選択」である。コードレベルのバグは比較的容易に修正できるが、アーキテクチャの根本的な変更は数か月から数年を要し、しばしばシステムの書き直しにつながる。Twitterが2012年にRuby on RailsモノリスからJVMベースのマイクロサービスに移行するのに2年以上を費やしたのは、初期設計が成長に追いつけなかった典型例である。

```
設計の影響範囲と修正コスト:

  修正コスト（相対値）
    ^
100 │                                           ┌─────┐
    │                                           │ アー │
    │                                           │ キテ │
    │                                           │ クチ │
    │                                           │  ャ  │
 50 │                              ┌─────┐      │     │
    │                              │ DB  │      │     │
    │                              │スキー│      │     │
    │                 ┌─────┐      │ マ  │      │     │
 10 │    ┌─────┐      │ API │      │     │      │     │
    │    │コード│      │設計  │      │     │      │     │
  1 │    │バグ  │      │     │      │     │      │     │
    └────┴─────┴──────┴─────┴──────┴─────┴──────┴─────┴──→
         要件定義    設計       実装       運用
         フェーズ   フェーズ   フェーズ   フェーズ
```

### 1.2 システム設計の3つの柱

大規模システムの設計では、以下の3つの性質を同時に考慮する必要がある。

```
┌──────────────────────────────────────────────────────────────┐
│                 システム設計の3つの柱                          │
├──────────────────┬──────────────────┬────────────────────────┤
│  スケーラビリティ  │    信頼性         │    保守性              │
│  (Scalability)   │  (Reliability)   │  (Maintainability)     │
├──────────────────┼──────────────────┼────────────────────────┤
│ ・負荷増大に対応   │ ・障害時も動作    │ ・変更が容易           │
│ ・水平/垂直拡張    │ ・データ損失なし  │ ・チームが理解可能      │
│ ・レイテンシ維持   │ ・自動回復       │ ・テスト/デプロイ容易    │
│                  │                  │ ・技術的負債の制御       │
│ 詳細:            │ 詳細:            │ 詳細:                  │
│ → 01-scalability │ → 02-reliability │ → clean-code-principles│
└──────────────────┴──────────────────┴────────────────────────┘
```

### 1.3 システム設計が必要になる規模感

全てのシステムが大規模設計を必要とするわけではない。以下に規模ごとの典型的な構成を示す。

```python
# 規模別アーキテクチャ判断の目安
def recommend_architecture(dau: int, data_gb: float) -> dict:
    """DAUとデータ量からアーキテクチャを推奨"""
    if dau < 1_000:
        return {
            "architecture": "シンプルモノリス",
            "infra": "単一サーバー (VPS/PaaS)",
            "db": "単一RDB (PostgreSQL)",
            "cache": "不要（またはアプリ内キャッシュ）",
            "deployment": "手動 or 簡易CI/CD",
            "cost": "~$50/月",
            "team_size": "1-2人",
        }
    elif dau < 100_000:
        return {
            "architecture": "モノリス + 一部分離",
            "infra": "Web(2台) + DB(Primary/Replica)",
            "db": "RDB + Read Replica",
            "cache": "Redis (単一ノード)",
            "deployment": "CI/CD + Blue/Green",
            "cost": "~$500-2,000/月",
            "team_size": "3-10人",
        }
    elif dau < 10_000_000:
        return {
            "architecture": "サービス分割 (Modular Monolith or Microservices)",
            "infra": "Kubernetes or ECS + Auto Scaling",
            "db": "RDB (シャーディング) + NoSQL",
            "cache": "Redis Cluster + CDN",
            "deployment": "Canary/Rolling + Feature Flags",
            "cost": "~$10,000-100,000/月",
            "team_size": "20-100人",
        }
    else:
        return {
            "architecture": "マイクロサービス + イベント駆動",
            "infra": "マルチリージョン + マルチクラウド",
            "db": "分散DB + Data Lake + CQRS",
            "cache": "多層キャッシュ (L1/L2/CDN)",
            "deployment": "カスタムデプロイパイプライン",
            "cost": "$100,000+/月",
            "team_size": "100人+",
        }

# 出力例
for dau in [500, 50_000, 5_000_000, 50_000_000]:
    result = recommend_architecture(dau, dau * 0.001)
    print(f"\n=== DAU: {dau:>12,} ===")
    for k, v in result.items():
        print(f"  {k:20s}: {v}")

# 出力:
# === DAU:          500 ===
#   architecture        : シンプルモノリス
#   infra               : 単一サーバー (VPS/PaaS)
#   db                  : 単一RDB (PostgreSQL)
#   cache               : 不要（またはアプリ内キャッシュ）
#   deployment          : 手動 or 簡易CI/CD
#   cost                : ~$50/月
#   team_size           : 1-2人
# ...以下略
```

---

## 2. 設計プロセスの4ステップ

システム設計は、以下の4つのステップを順に進める。実務では反復的に行うが、設計面接では線形に進めるのが一般的である。

### 設計プロセス全体像

```
┌────────────────────────────────────────────────────────────────────────┐
│                     システム設計プロセス 4ステップ                       │
│                                                                        │
│  Step 1              Step 2              Step 3              Step 4    │
│  ┌──────────┐       ┌──────────┐       ┌──────────┐       ┌──────────┐│
│  │ 要件の    │──────→│ 見積もり  │──────→│ 高レベル  │──────→│ 詳細設計  ││
│  │ 明確化    │       │          │       │ 設計     │       │          ││
│  └──────────┘       └──────────┘       └──────────┘       └──────────┘│
│                                                                        │
│  成果物:             成果物:             成果物:             成果物:     │
│  ・FR/NFRリスト      ・QPS計算          ・コンポーネント図   ・API設計   │
│  ・制約条件          ・ストレージ計算    ・データフロー図     ・DB設計    │
│  ・スコープ定義      ・帯域幅計算       ・技術選定          ・障害対策   │
│                                                                        │
│  ←──── フィードバックループ（発見に基づいて前のステップに戻る）────→    │
└────────────────────────────────────────────────────────────────────────┘
```

### Step 1: 要件の明確化（Clarify Requirements）

要件の明確化は設計プロセスで**最も重要なステップ**である。ここを曖昧にすると、後のステップ全てが的外れになるリスクがある。

**なぜ要件明確化が最重要なのか？**

設計面接において、要件を明確にせず即座に設計に入る候補者は、ほぼ確実に低評価を受ける。理由は、実務で最も致命的な失敗は「要件の取り違え」から生じるためである。面接官は「曖昧な要件の中から本質を見抜く力」を見ている。

```python
# 要件明確化のフレームワーク
class RequirementAnalyzer:
    """要件を体系的に整理するフレームワーク"""

    def __init__(self, system_name: str):
        self.system_name = system_name
        self.functional_requirements = []
        self.non_functional_requirements = {}
        self.constraints = []
        self.assumptions = []
        self.out_of_scope = []

    def add_functional(self, requirement: str, priority: str = "must"):
        """機能要件を追加（must/should/could/won't で分類）"""
        self.functional_requirements.append({
            "requirement": requirement,
            "priority": priority,  # MoSCoW法
        })

    def set_nfr(self, category: str, target: str, rationale: str):
        """非機能要件を設定"""
        self.non_functional_requirements[category] = {
            "target": target,
            "rationale": rationale,
        }

    def add_constraint(self, constraint: str):
        """制約条件を追加"""
        self.constraints.append(constraint)

    def add_assumption(self, assumption: str):
        """前提条件（仮定）を追加"""
        self.assumptions.append(assumption)

    def add_out_of_scope(self, item: str):
        """スコープ外の機能を明示"""
        self.out_of_scope.append(item)

    def summary(self) -> str:
        """要件サマリーを出力"""
        lines = [f"=== {self.system_name} 要件サマリー ===\n"]

        # 機能要件（MoSCoW法で分類）
        lines.append("【機能要件】")
        for priority in ["must", "should", "could", "won't"]:
            items = [r for r in self.functional_requirements
                     if r["priority"] == priority]
            if items:
                label = {"must": "Must Have", "should": "Should Have",
                         "could": "Could Have", "won't": "Won't Have"}[priority]
                lines.append(f"  [{label}]")
                for item in items:
                    lines.append(f"    - {item['requirement']}")

        # 非機能要件
        lines.append("\n【非機能要件】")
        for category, detail in self.non_functional_requirements.items():
            lines.append(f"  {category}: {detail['target']}")
            lines.append(f"    理由: {detail['rationale']}")

        # 制約条件
        lines.append("\n【制約条件】")
        for c in self.constraints:
            lines.append(f"  - {c}")

        # 仮定
        lines.append("\n【仮定】")
        for a in self.assumptions:
            lines.append(f"  - {a}")

        # スコープ外
        lines.append("\n【スコープ外】")
        for o in self.out_of_scope:
            lines.append(f"  - {o}")

        return "\n".join(lines)


# 使用例: Twitter風SNSの設計
analyzer = RequirementAnalyzer("Twitter風SNS")

# 機能要件（MoSCoW法）
analyzer.add_functional("ツイートの投稿（テキスト、280文字以内）", "must")
analyzer.add_functional("ホームタイムラインの表示", "must")
analyzer.add_functional("ユーザーフォロー/アンフォロー", "must")
analyzer.add_functional("ツイートの「いいね」", "should")
analyzer.add_functional("リツイート機能", "should")
analyzer.add_functional("画像/動画の添付", "could")
analyzer.add_functional("ダイレクトメッセージ", "won't")  # 今回のスコープ外
analyzer.add_functional("広告表示", "won't")

# 非機能要件
analyzer.set_nfr("レイテンシ", "タイムライン表示 < 200ms (P99)",
                 "ユーザー体験の閾値。Googleの研究で200ms以上は体感遅延")
analyzer.set_nfr("可用性", "99.99% (年間ダウンタイム < 52分)",
                 "SNSは常時利用が前提。99.9%では年間8.7時間のダウン")
analyzer.set_nfr("スケーラビリティ", "DAU 3億、ピーク200万QPS",
                 "Twitterの実績ベース。成長を見込んで余裕を持つ")
analyzer.set_nfr("データ耐久性", "99.9999999% (9 nines)",
                 "ユーザーの投稿データは絶対に失えない")
analyzer.set_nfr("一貫性", "結果整合性（5秒以内）",
                 "強い一貫性は不要。タイムラインは多少の遅延が許容される")

# 制約条件
analyzer.add_constraint("予算: 初年度 $5M 以内")
analyzer.add_constraint("チーム: バックエンドエンジニア 20名")
analyzer.add_constraint("既存インフラ: AWS (移行不可)")

# 仮定
analyzer.add_assumption("DAU 3億のうち、1日あたり投稿するのは20%（6000万人）")
analyzer.add_assumption("1ユーザーあたりフォロー平均200人")
analyzer.add_assumption("読み:書き比率 = 100:1")

# スコープ外
analyzer.add_out_of_scope("ダイレクトメッセージ機能")
analyzer.add_out_of_scope("広告配信システム")
analyzer.add_out_of_scope("画像/動画のトランスコーディング")

print(analyzer.summary())

# 出力:
# === Twitter風SNS 要件サマリー ===
#
# 【機能要件】
#   [Must Have]
#     - ツイートの投稿（テキスト、280文字以内）
#     - ホームタイムラインの表示
#     - ユーザーフォロー/アンフォロー
#   [Should Have]
#     - ツイートの「いいね」
#     - リツイート機能
#   [Could Have]
#     - 画像/動画の添付
#   [Won't Have]
#     - ダイレクトメッセージ
#     - 広告表示
#
# 【非機能要件】
#   レイテンシ: タイムライン表示 < 200ms (P99)
#     理由: ユーザー体験の閾値。Googleの研究で200ms以上は体感遅延
#   可用性: 99.99% (年間ダウンタイム < 52分)
#     理由: SNSは常時利用が前提。99.9%では年間8.7時間のダウン
#   ...以下略
```

### Step 2: 見積もり（Back-of-the-Envelope Estimation）

設計面接でも実務でも、概算を即座に出す力が問われる。完璧な精度は不要だが、桁（オーダー）を間違えないことが重要である。

**なぜ見積もりが重要なのか？**

見積もりは設計判断の「根拠」となる。「キャッシュが必要かどうか」「DBのシャーディングが必要かどうか」「CDNを入れるべきか」といった判断は全て、QPS・データ量・帯域幅の概算に基づいて下される。見積もりなしの設計は、地図なしの航海に等しい。

```python
class SystemEstimator:
    """システム設計の見積もりツール"""

    # よく使う定数
    SECONDS_PER_DAY = 86_400
    SECONDS_PER_MONTH = 2_592_000  # 30日
    SECONDS_PER_YEAR = 31_536_000  # 365日

    def __init__(self, service_name: str, dau: int):
        self.service_name = service_name
        self.dau = dau

    def estimate_qps(self, actions_per_user_per_day: float,
                     read_write_ratio: int = 100,
                     peak_multiplier: float = 3.0) -> dict:
        """QPS（Queries Per Second）の見積もり"""
        write_qps = self.dau * actions_per_user_per_day / self.SECONDS_PER_DAY
        read_qps = write_qps * read_write_ratio
        return {
            "avg_write_qps": write_qps,
            "avg_read_qps": read_qps,
            "peak_write_qps": write_qps * peak_multiplier,
            "peak_read_qps": read_qps * peak_multiplier,
        }

    def estimate_storage(self, object_size_bytes: int,
                         objects_per_user_per_day: float,
                         retention_years: int = 5) -> dict:
        """ストレージ見積もり"""
        daily_objects = self.dau * objects_per_user_per_day
        daily_storage = daily_objects * object_size_bytes
        yearly_storage = daily_storage * 365
        total_storage = yearly_storage * retention_years
        return {
            "daily_new_objects": daily_objects,
            "daily_storage_tb": daily_storage / 1e12,
            "yearly_storage_tb": yearly_storage / 1e12,
            "total_storage_tb": total_storage / 1e12,
            "total_storage_pb": total_storage / 1e15,
        }

    def estimate_bandwidth(self, avg_response_size_bytes: int,
                           peak_read_qps: float) -> dict:
        """帯域幅見積もり"""
        bandwidth_bytes = peak_read_qps * avg_response_size_bytes
        return {
            "peak_bandwidth_gbps": bandwidth_bytes * 8 / 1e9,
            "peak_bandwidth_mbps": bandwidth_bytes * 8 / 1e6,
            "monthly_transfer_tb": (bandwidth_bytes * self.SECONDS_PER_MONTH) / 1e12,
        }

    def estimate_memory_for_cache(self, hot_data_percentage: float,
                                  daily_storage_bytes: float) -> dict:
        """キャッシュに必要なメモリ量の見積もり（80/20ルール）"""
        # 通常、20%のデータが80%のアクセスを占める
        cache_size = daily_storage_bytes * hot_data_percentage
        # Redis等のオーバーヘッド（約2倍）
        actual_memory = cache_size * 2
        return {
            "cache_data_gb": cache_size / 1e9,
            "actual_memory_gb": actual_memory / 1e9,
            "redis_nodes_64gb": max(1, int(actual_memory / (64 * 1e9)) + 1),
        }

    def full_report(self) -> str:
        """全見積もりのレポートを生成"""
        lines = [f"=== {self.service_name} 見積もりレポート ==="]
        lines.append(f"DAU: {self.dau:,}")

        # QPS
        qps = self.estimate_qps(2, 100, 3)
        lines.append(f"\n【QPS見積もり】")
        lines.append(f"  平均書き込み: {qps['avg_write_qps']:,.0f} QPS")
        lines.append(f"  平均読み込み: {qps['avg_read_qps']:,.0f} QPS")
        lines.append(f"  ピーク書き込み: {qps['peak_write_qps']:,.0f} QPS")
        lines.append(f"  ピーク読み込み: {qps['peak_read_qps']:,.0f} QPS")

        # ストレージ
        storage = self.estimate_storage(560, 2, 5)
        lines.append(f"\n【ストレージ見積もり】")
        lines.append(f"  1日の新規データ: {storage['daily_storage_tb']:.2f} TB")
        lines.append(f"  1年の新規データ: {storage['yearly_storage_tb']:.1f} TB")
        lines.append(f"  5年間の総データ: {storage['total_storage_tb']:.1f} TB "
                     f"({storage['total_storage_pb']:.2f} PB)")

        # 帯域幅
        bw = self.estimate_bandwidth(10_000, qps['peak_read_qps'])
        lines.append(f"\n【帯域幅見積もり】")
        lines.append(f"  ピーク帯域: {bw['peak_bandwidth_gbps']:.1f} Gbps")
        lines.append(f"  月間転送量: {bw['monthly_transfer_tb']:.0f} TB")

        # キャッシュ
        cache = self.estimate_memory_for_cache(
            0.2, storage['daily_storage_tb'] * 1e12)
        lines.append(f"\n【キャッシュ見積もり（ホットデータ20%）】")
        lines.append(f"  キャッシュデータ量: {cache['cache_data_gb']:.1f} GB")
        lines.append(f"  実メモリ必要量: {cache['actual_memory_gb']:.1f} GB")
        lines.append(f"  Redisノード数(64GB): {cache['redis_nodes_64gb']}")

        return "\n".join(lines)


# 使用例
estimator = SystemEstimator("Twitter風SNS", 300_000_000)
print(estimator.full_report())

# 出力:
# === Twitter風SNS 見積もりレポート ===
# DAU: 300,000,000
#
# 【QPS見積もり】
#   平均書き込み: 6,944 QPS
#   平均読み込み: 694,444 QPS
#   ピーク書き込み: 20,833 QPS
#   ピーク読み込み: 2,083,333 QPS
#
# 【ストレージ見積もり】
#   1日の新規データ: 0.34 TB
#   1年の新規データ: 122.6 TB
#   5年間の総データ: 613.2 TB (0.61 PB)
#
# 【帯域幅見積もり】
#   ピーク帯域: 166.7 Gbps
#   月間転送量: 54000 TB
#
# 【キャッシュ見積もり（ホットデータ20%）】
#   キャッシュデータ量: 67.2 GB
#   実メモリ必要量: 134.4 GB
#   Redisノード数(64GB): 3
```

### Step 3: 高レベル設計（High-Level Design）

高レベル設計では、システムの主要コンポーネントとそれらの関係を図に描き、データの流れを明らかにする。

**高レベル設計の原則:**

1. **クライアントからデータストアまでの全経路を描く**: リクエストが入ってレスポンスが返るまでの全経路
2. **各コンポーネントの責務を1文で説明できること**: 説明できないなら分割が不適切
3. **矢印にはプロトコルとデータを明記**: HTTP、gRPC、WebSocket、メッセージキュー等
4. **数値を添える**: QPS、レイテンシ目標、データサイズ

### Step 4: 詳細設計（Deep Dive）

高レベル設計の後、特にクリティカルなコンポーネントを掘り下げる。面接では面接官が指定するか、自分で「最もチャレンジングな部分」を選んで深掘りする。

**Deep Diveの対象を選ぶ基準:**

- システムのボトルネックになりそうな箇所
- スケーラビリティに最も影響する箇所
- データの一貫性が求められる箇所
- 障害時の影響が最も大きい箇所

---

## 3. 機能要件と非機能要件

### 3.1 機能要件（Functional Requirements）

機能要件とは、システムが「何をするか」を定義するものである。ユーザーの視点から見た「このシステムでできること」のリストとも言える。

```
┌──────────────────────────────────────────────────────────────┐
│                    機能要件の分類体系                          │
├──────────────────┬──────────────────┬────────────────────────┤
│   ユーザー機能    │   管理者機能      │    システム機能         │
├──────────────────┼──────────────────┼────────────────────────┤
│ ・ユーザー登録    │ ・ユーザー管理    │ ・通知送信             │
│ ・ログイン/認証   │ ・コンテンツ管理  │ ・バッチ処理           │
│ ・投稿作成/削除   │ ・統計ダッシュ    │ ・データ同期           │
│ ・検索           │ ・権限管理       │ ・監査ログ             │
│ ・タイムライン    │ ・設定管理       │ ・データエクスポート    │
└──────────────────┴──────────────────┴────────────────────────┘
```

### 3.2 非機能要件（Non-Functional Requirements）

非機能要件は「どのように実現するか」の品質特性を定義する。機能要件と異なり、暗黙的であることが多く、見落とされやすい。

```python
# 非機能要件の体系的整理
class NFRFramework:
    """非機能要件を網羅的にチェックするフレームワーク"""

    CATEGORIES = {
        "性能 (Performance)": {
            "レイテンシ": "応答時間の上限 (P50, P95, P99)",
            "スループット": "単位時間あたりの処理能力 (QPS/TPS)",
            "リソース効率": "CPU/メモリ/ディスク使用率の目標",
        },
        "信頼性 (Reliability)": {
            "可用性": "稼働率の目標 (99.9%, 99.99%等)",
            "耐障害性": "障害時の動作保証",
            "データ耐久性": "データ損失に対する保証",
            "災害復旧": "RPO (Recovery Point Objective) / RTO (Recovery Time Objective)",
        },
        "スケーラビリティ (Scalability)": {
            "負荷対応": "想定する最大負荷",
            "データ増大": "想定するデータ増加率",
            "地理的拡張": "マルチリージョン対応の必要性",
        },
        "セキュリティ (Security)": {
            "認証/認可": "認証方式、権限モデル",
            "データ保護": "暗号化要件 (at-rest, in-transit)",
            "コンプライアンス": "GDPR, HIPAA, PCI DSS等",
            "監査": "操作ログの保存要件",
        },
        "運用性 (Operability)": {
            "監視": "メトリクス、ログ、トレーシング",
            "デプロイ": "デプロイ頻度、ダウンタイム許容度",
            "保守性": "コードの理解しやすさ、変更容易性",
        },
    }

    @classmethod
    def generate_checklist(cls) -> str:
        """NFRチェックリストを生成"""
        lines = ["=== 非機能要件チェックリスト ===\n"]
        for category, items in cls.CATEGORIES.items():
            lines.append(f"[ ] {category}")
            for name, description in items.items():
                lines.append(f"    [ ] {name}: {description}")
            lines.append("")
        return "\n".join(lines)

print(NFRFramework.generate_checklist())

# 出力:
# === 非機能要件チェックリスト ===
#
# [ ] 性能 (Performance)
#     [ ] レイテンシ: 応答時間の上限 (P50, P95, P99)
#     [ ] スループット: 単位時間あたりの処理能力 (QPS/TPS)
#     [ ] リソース効率: CPU/メモリ/ディスク使用率の目標
#
# [ ] 信頼性 (Reliability)
#     [ ] 可用性: 稼働率の目標 (99.9%, 99.99%等)
#     [ ] 耐障害性: 障害時の動作保証
#     [ ] データ耐久性: データ損失に対する保証
#     [ ] 災害復旧: RPO / RTO
# ...以下略
```

### 3.3 可用性の数値と意味

非機能要件で最も頻出するのが「可用性」である。以下にSLAの数値とその意味を整理する。

```python
# 可用性とダウンタイムの関係
def availability_table():
    """可用性レベルごとのダウンタイムを計算"""
    print(f"{'可用性':>10s}  {'年間DT':>12s}  {'月間DT':>10s}  {'週間DT':>8s}  {'用途例'}")
    print("-" * 80)

    levels = [
        (99.0,    "内部ツール、開発環境"),
        (99.9,    "一般的なSaaS"),
        (99.95,   "ECサイト"),
        (99.99,   "金融、医療"),
        (99.999,  "通信インフラ"),
        (99.9999, "航空管制、原子力"),
    ]

    minutes_per_year = 365.25 * 24 * 60

    for avail, use_case in levels:
        downtime_pct = 100 - avail
        dt_year = minutes_per_year * downtime_pct / 100
        dt_month = dt_year / 12
        dt_week = dt_year / 52

        if dt_year >= 60:
            year_str = f"{dt_year / 60:.1f}時間"
        else:
            year_str = f"{dt_year:.1f}分"

        if dt_month >= 60:
            month_str = f"{dt_month / 60:.1f}時間"
        else:
            month_str = f"{dt_month:.1f}分"

        week_str = f"{dt_week:.1f}分"

        print(f"{avail:>9.4f}%  {year_str:>12s}  {month_str:>10s}  "
              f"{week_str:>8s}  {use_case}")

availability_table()

# 出力:
#    可用性        年間DT      月間DT    週間DT  用途例
# ----------------------------------------------------------
#  99.0000%      87.7時間     7.3時間   101.2分  内部ツール、開発環境
#  99.9000%       8.8時間      43.8分    10.1分  一般的なSaaS
#  99.9500%       4.4時間      21.9分     5.1分  ECサイト
#  99.9900%      52.6分        4.4分     1.0分  金融、医療
#  99.9990%       5.3分        0.4分     0.1分  通信インフラ
#  99.9999%       0.5分        0.0分     0.0分  航空管制、原子力
```

### 3.4 機能要件 vs 非機能要件の比較

| 項目 | 機能要件 (FR) | 非機能要件 (NFR) |
|------|--------------|-----------------|
| 定義 | システムが「何をするか」 | システムが「どのように動くか」 |
| 例 | ユーザー登録、検索、投稿 | レイテンシ、可用性、セキュリティ |
| 記述方法 | ユースケース、ユーザーストーリー | SLA/SLO、数値目標 |
| テスト方法 | 機能テスト、E2Eテスト | 負荷テスト、カオスエンジニアリング |
| 変更頻度 | 高い（機能追加が多い） | 低い（基盤的な要件） |
| 影響範囲 | 特定の機能に限定 | システム全体に波及 |
| 見落としの危険度 | 低い（明示的に要求される） | 高い（暗黙的であることが多い） |
| 優先度決定 | ビジネス価値で判断 | 技術的リスクで判断 |
| 記述の主体 | プロダクトマネージャー | アーキテクト / エンジニア |

---

## 4. 見積もり技法の詳細

### 4.1 レイテンシの数値を覚える

システム設計において、各操作のレイテンシを直感的に把握していることは極めて重要である。

```python
# Jeff Deanのレイテンシ数値（現代版概算）
latency_numbers = {
    # メモリ操作（ナノ秒オーダー）
    "L1 cache reference":                  ("0.5 ns",   0.5),
    "Branch mispredict":                   ("5 ns",     5),
    "L2 cache reference":                  ("7 ns",     7),
    "Mutex lock/unlock":                   ("25 ns",    25),
    "Main memory reference":               ("100 ns",   100),

    # ストレージ操作（マイクロ秒〜ミリ秒オーダー）
    "SSD random read (4KB)":               ("150 us",   150_000),
    "Read 1 MB sequentially from memory":  ("250 us",   250_000),
    "Read 1 MB sequentially from SSD":     ("1 ms",     1_000_000),
    "Read 1 MB sequentially from HDD":     ("20 ms",    20_000_000),
    "HDD disk seek":                       ("10 ms",    10_000_000),

    # ネットワーク操作（ミリ秒オーダー）
    "Send packet CA → Netherlands → CA":   ("150 ms",   150_000_000),
    "Same datacenter round trip":          ("0.5 ms",   500_000),
    "TCP handshake (same DC)":             ("1.5 ms",   1_500_000),
    "TLS handshake":                       ("10 ms",    10_000_000),
}

# ビジュアル比較（対数スケール）
print("=== レイテンシ比較（対数スケール） ===\n")
for operation, (label, ns) in latency_numbers.items():
    import math
    bar_length = int(math.log10(max(ns, 1)) * 4)
    bar = "#" * bar_length
    print(f"  {operation:45s} {label:>10s}  {bar}")

# 出力:
# === レイテンシ比較（対数スケール） ===
#
#   L1 cache reference                          0.5 ns
#   Branch mispredict                             5 ns  ##
#   L2 cache reference                            7 ns  ###
#   Mutex lock/unlock                            25 ns  #####
#   Main memory reference                       100 ns  ########
#   SSD random read (4KB)                       150 us  ####################
#   Read 1 MB sequentially from memory          250 us  #####################
#   Read 1 MB sequentially from SSD               1 ms  ########################
#   Read 1 MB sequentially from HDD              20 ms  #############################
#   HDD disk seek                                10 ms  ############################
#   Send packet CA → Netherlands → CA           150 ms  ################################
#   Same datacenter round trip                  0.5 ms  ######################
#   TCP handshake (same DC)                     1.5 ms  ########################
#   TLS handshake                                10 ms  ############################
```

### 4.2 2の冪乗テーブル

```python
# 設計面接でよく使う2の冪乗とデータサイズの対応表
print("=== 2の冪乗テーブル ===\n")
print(f"{'冪乗':>6s}  {'値':>20s}  {'バイト単位':>10s}  {'実用例'}")
print("-" * 75)

examples = {
    7:  ("128 B",     "HTTPヘッダー1つ分"),
    8:  ("256 B",     "小さなJSON応答"),
    10: ("1 KB",      "短いテキスト"),
    14: ("16 KB",     "典型的なHTML1ページ"),
    16: ("64 KB",     "TCPウィンドウサイズ"),
    20: ("1 MB",      "高画質写真1枚"),
    23: ("8 MB",      "一般的なキャッシュエントリ上限"),
    25: ("32 MB",     "短い動画クリップ"),
    30: ("1 GB",      "映画1本 (圧縮済み)"),
    33: ("8 GB",      "一般的なサーバーメモリ"),
    36: ("64 GB",     "Redisノード1台の推奨上限"),
    40: ("1 TB",      "大規模DBのデータファイル"),
    50: ("1 PB",      "大企業のデータウェアハウス"),
}

for power, (size, example) in examples.items():
    print(f"  2^{power:2d}  {2**power:>20,}  {size:>10s}  {example}")

# 出力:
#  冪乗                    値    バイト単位  実用例
# -----------------------------------------------------------------------
#   2^ 7                   128       128 B  HTTPヘッダー1つ分
#   2^ 8                   256       256 B  小さなJSON応答
#   2^10                 1,024        1 KB  短いテキスト
#   2^14                16,384       16 KB  典型的なHTML1ページ
#   2^16                65,536       64 KB  TCPウィンドウサイズ
#   2^20             1,048,576        1 MB  高画質写真1枚
#   ...以下略
```

### 4.3 見積もりの実践: 主要サービスの概算

```python
# 主要Webサービスの見積もりを瞬時に出す練習
services = {
    "URL短縮サービス": {
        "dau": 100_000_000,
        "writes_per_user_day": 0.1,  # 10人に1人が短縮
        "reads_per_write": 100,      # 1つのURLが100回クリック
        "data_per_record_bytes": 500, # URL + メタデータ
        "retention_years": 10,
    },
    "Instagram風画像共有": {
        "dau": 500_000_000,
        "writes_per_user_day": 0.5,
        "reads_per_write": 200,
        "data_per_record_bytes": 2_000_000,  # 平均2MB/画像
        "retention_years": 99,  # 永久保存
    },
    "チャットアプリ": {
        "dau": 200_000_000,
        "writes_per_user_day": 50,   # 1日50メッセージ
        "reads_per_write": 5,        # グループチャット平均
        "data_per_record_bytes": 200, # テキストメッセージ
        "retention_years": 5,
    },
}

for name, params in services.items():
    dau = params["dau"]
    write_qps = dau * params["writes_per_user_day"] / 86400
    read_qps = write_qps * params["reads_per_write"]
    daily_storage = dau * params["writes_per_user_day"] * params["data_per_record_bytes"]
    yearly_storage_tb = daily_storage * 365 / 1e12

    print(f"\n=== {name} ===")
    print(f"  DAU: {dau/1e6:.0f}M")
    print(f"  書き込みQPS: {write_qps:,.0f} (ピーク: {write_qps*3:,.0f})")
    print(f"  読み込みQPS: {read_qps:,.0f} (ピーク: {read_qps*3:,.0f})")
    print(f"  年間ストレージ: {yearly_storage_tb:.1f} TB")

# 出力:
# === URL短縮サービス ===
#   DAU: 100M
#   書き込みQPS: 116 (ピーク: 347)
#   読み込みQPS: 11,574 (ピーク: 34,722)
#   年間ストレージ: 1.8 TB
#
# === Instagram風画像共有 ===
#   DAU: 500M
#   書き込みQPS: 2,894 (ピーク: 8,681)
#   読み込みQPS: 578,704 (ピーク: 1,736,111)
#   年間ストレージ: 182,500.0 TB
#
# === チャットアプリ ===
#   DAU: 200M
#   書き込みQPS: 115,741 (ピーク: 347,222)
#   読み込みQPS: 578,704 (ピーク: 1,736,111)
#   年間ストレージ: 730.0 TB
```

---

## 5. 高レベル設計の描き方

### 5.1 典型的なWebアプリケーション構成

```
                           ┌──────────────┐
                           │    Client    │
                           │(Browser/App) │
                           └──────┬───────┘
                                  │ HTTPS
                           ┌──────▼───────┐
                           │     DNS      │
                           │ (Route 53)   │
                           └──────┬───────┘
                                  │ IP解決
                     ┌────────────▼────────────┐
                     │      CDN (CloudFront)   │
                     │   ・静的ファイル配信      │
                     │   ・エッジキャッシュ      │
                     └────────────┬────────────┘
                                  │ 動的リクエスト (miss)
                     ┌────────────▼────────────┐
                     │     Load Balancer (L7)   │
                     │   ・ヘルスチェック        │
                     │   ・SSL終端              │
                     │   ・レートリミット        │
                     └───┬────────┬────────┬───┘
                         │        │        │
                    ┌────▼──┐ ┌──▼────┐ ┌─▼─────┐
                    │App S1 │ │App S2 │ │App S3 │  ← Auto Scaling Group
                    │(API)  │ │(API)  │ │(API)  │
                    └───┬───┘ └───┬───┘ └───┬───┘
                        │        │         │
                   ┌────▼────────▼─────────▼────┐
                   │     Cache Layer (Redis)     │
                   │   ・セッション              │
                   │   ・ホットデータ             │
                   │   ・レートリミットカウンタ    │
                   └────────────┬────────────────┘
                                │ (cache miss)
              ┌─────────────────▼──────────────────┐
              │          Data Layer                 │
              │                                    │
              │  ┌──────────┐    ┌──────────────┐  │
              │  │ Primary  │───→│  Replica(s)  │  │
              │  │  (Write) │    │   (Read)     │  │
              │  └──────────┘    └──────────────┘  │
              │                                    │
              │  ┌──────────┐    ┌──────────────┐  │
              │  │  Object  │    │   Search     │  │
              │  │  Storage │    │   (ES/Solr)  │  │
              │  │  (S3)    │    │              │  │
              │  └──────────┘    └──────────────┘  │
              └────────────────────────────────────┘

              ┌────────────────────────────────────┐
              │       Async Processing             │
              │                                    │
              │  ┌──────────┐    ┌──────────────┐  │
              │  │  Message │───→│   Workers    │  │
              │  │  Queue   │    │  (Consumer)  │  │
              │  │ (Kafka)  │    │              │  │
              │  └──────────┘    └──────────────┘  │
              └────────────────────────────────────┘

              ┌────────────────────────────────────┐
              │       Observability                │
              │  ┌──────┐ ┌──────┐ ┌───────────┐  │
              │  │Metrics│ │ Logs │ │  Traces   │  │
              │  │(Prome)│ │(ELK) │ │ (Jaeger)  │  │
              │  └──────┘ └──────┘ └───────────┘  │
              └────────────────────────────────────┘
```

### 5.2 コンポーネント間通信パターン

```
┌───────────────────────────────────────────────────────────────────┐
│                コンポーネント間通信パターン                         │
├────────────────┬──────────────┬──────────────┬───────────────────┤
│   パターン      │  同期/非同期  │  結合度      │  ユースケース      │
├────────────────┼──────────────┼──────────────┼───────────────────┤
│ REST API       │  同期        │  高い        │ CRUD操作、外部API  │
│ gRPC           │  同期/非同期  │  高い        │ マイクロサービス間  │
│ GraphQL        │  同期        │  中程度      │ フロントエンド向け  │
│ Message Queue  │  非同期      │  低い        │ バックグラウンド処理│
│ Event Bus      │  非同期      │  非常に低い  │ イベント駆動       │
│ WebSocket      │  双方向      │  中程度      │ リアルタイム更新    │
│ SSE            │  サーバー→   │  低い        │ 通知、フィード更新  │
│ Webhook        │  非同期      │  低い        │ 外部連携           │
└────────────────┴──────────────┴──────────────┴───────────────────┘
```

### 5.3 設計面接の時間配分

```
■ 45分の設計面接の推奨時間配分

  0        5       10       15              35     40    45
  ├────────┼────────┼────────┼───────────────┼──────┼─────┤
  │ 要件   │ 見積   │ 高レベ │   詳細設計     │ まと │ Q&A │
  │ 確認   │ もり   │ ル設計 │  (Deep Dive)  │  め  │     │
  │  5min  │  5min  │  5min  │    20min      │ 5min │5min │
  └────────┴────────┴────────┴───────────────┴──────┴─────┘
    ↑                ↑                        ↑
    最も重要          ここで図を描く            最後に
    ここでスコープ    コンポーネント図           スケーリング
    を固める          データフロー図            ボトルネック
                                              を議論

■ 各フェーズのチェックリスト

  [要件確認 5min]
  □ コア機能を3つ以内に絞る
  □ DAU/規模感を確認
  □ 重要なNFRを2-3個特定
  □ スコープ外を明示

  [見積もり 5min]
  □ QPS (読み/書き) を計算
  □ ストレージ概算
  □ 必要なら帯域幅も

  [高レベル設計 5min]
  □ 主要コンポーネントを図に描く
  □ データフローを矢印で示す
  □ プロトコルを明記

  [詳細設計 20min]
  □ 最もクリティカルな2-3箇所を深掘り
  □ API設計 (エンドポイント、リクエスト/レスポンス)
  □ DBスキーマ (テーブル、インデックス)
  □ トレードオフを議論

  [まとめ 5min]
  □ ボトルネックの列挙と対策
  □ 将来のスケーリング戦略
  □ 監視すべきメトリクス
```

---

## 6. 詳細設計の実践テクニック

### 6.1 API設計の基本

```python
# REST API設計の体系的アプローチ
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class HttpMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"

@dataclass
class APIEndpoint:
    """REST APIエンドポイントの定義"""
    method: HttpMethod
    path: str
    description: str
    request_body: Optional[dict] = None
    query_params: Optional[dict] = None
    response: Optional[dict] = None
    rate_limit: str = "100 req/min"
    auth_required: bool = True

@dataclass
class APIDesign:
    """API設計文書"""
    service_name: str
    base_url: str
    version: str = "v1"
    endpoints: list = field(default_factory=list)

    def add_endpoint(self, endpoint: APIEndpoint):
        self.endpoints.append(endpoint)

    def generate_doc(self) -> str:
        lines = [f"=== {self.service_name} API設計 ==="]
        lines.append(f"Base URL: {self.base_url}/api/{self.version}\n")

        for ep in self.endpoints:
            lines.append(f"  {ep.method.value:6s} {ep.path}")
            lines.append(f"    説明: {ep.description}")
            lines.append(f"    認証: {'必要' if ep.auth_required else '不要'}")
            lines.append(f"    レート制限: {ep.rate_limit}")
            if ep.request_body:
                lines.append(f"    リクエスト: {ep.request_body}")
            if ep.query_params:
                lines.append(f"    クエリパラメータ: {ep.query_params}")
            if ep.response:
                lines.append(f"    レスポンス: {ep.response}")
            lines.append("")

        return "\n".join(lines)


# 使用例: Twitter風SNSのAPI設計
api = APIDesign("Twitter風SNS", "https://api.example.com")

api.add_endpoint(APIEndpoint(
    method=HttpMethod.POST,
    path="/tweets",
    description="新しいツイートを投稿",
    request_body={"text": "str (max 280)", "media_ids": "list[str] (optional)"},
    response={"tweet_id": "str", "created_at": "datetime"},
    rate_limit="300 req/hour",
))

api.add_endpoint(APIEndpoint(
    method=HttpMethod.GET,
    path="/timeline/home",
    description="ホームタイムラインを取得",
    query_params={"cursor": "str", "limit": "int (default=20, max=100)"},
    response={"tweets": "list[Tweet]", "next_cursor": "str"},
    rate_limit="450 req/15min",
))

api.add_endpoint(APIEndpoint(
    method=HttpMethod.GET,
    path="/users/{user_id}",
    description="ユーザープロフィールを取得",
    response={"user_id": "str", "name": "str", "followers_count": "int"},
    rate_limit="900 req/15min",
))

api.add_endpoint(APIEndpoint(
    method=HttpMethod.POST,
    path="/users/{user_id}/follow",
    description="ユーザーをフォロー",
    response={"success": "bool"},
    rate_limit="400 req/day",
))

api.add_endpoint(APIEndpoint(
    method=HttpMethod.DELETE,
    path="/tweets/{tweet_id}",
    description="ツイートを削除",
    response={"success": "bool"},
    rate_limit="300 req/hour",
))

print(api.generate_doc())

# 出力:
# === Twitter風SNS API設計 ===
# Base URL: https://api.example.com/api/v1
#
#   POST   /tweets
#     説明: 新しいツイートを投稿
#     認証: 必要
#     レート制限: 300 req/hour
#     リクエスト: {'text': 'str (max 280)', 'media_ids': 'list[str] (optional)'}
#     レスポンス: {'tweet_id': 'str', 'created_at': 'datetime'}
# ...以下略
```

### 6.2 データベーススキーマ設計

```python
# DBスキーマ設計のフレームワーク
class SchemaDesigner:
    """DBスキーマ設計を構造化するツール"""

    def __init__(self, db_type: str = "PostgreSQL"):
        self.db_type = db_type
        self.tables = []

    def add_table(self, name: str, columns: list, indexes: list = None,
                  notes: str = ""):
        self.tables.append({
            "name": name,
            "columns": columns,
            "indexes": indexes or [],
            "notes": notes,
        })

    def generate_ddl(self) -> str:
        lines = [f"-- {self.db_type} スキーマ定義\n"]
        for table in self.tables:
            lines.append(f"-- {table['notes']}" if table['notes'] else "")
            lines.append(f"CREATE TABLE {table['name']} (")
            col_lines = []
            for col in table['columns']:
                col_lines.append(f"    {col}")
            lines.append(",\n".join(col_lines))
            lines.append(");\n")

            for idx in table['indexes']:
                lines.append(f"CREATE INDEX {idx};")
            lines.append("")
        return "\n".join(lines)


# Twitter風SNSのスキーマ設計
schema = SchemaDesigner()

schema.add_table("users", [
    "user_id         BIGINT PRIMARY KEY",
    "username        VARCHAR(15) UNIQUE NOT NULL",
    "email           VARCHAR(255) UNIQUE NOT NULL",
    "display_name    VARCHAR(50) NOT NULL",
    "bio             VARCHAR(160)",
    "followers_count INT DEFAULT 0",
    "following_count INT DEFAULT 0",
    "created_at      TIMESTAMP DEFAULT NOW()",
    "updated_at      TIMESTAMP DEFAULT NOW()",
], indexes=[
    "idx_users_username ON users(username)",
    "idx_users_email ON users(email)",
], notes="ユーザー情報 - DAU 3億 → 約5億行想定")

schema.add_table("tweets", [
    "tweet_id    BIGINT PRIMARY KEY",  # Snowflake ID
    "user_id     BIGINT NOT NULL REFERENCES users(user_id)",
    "text        VARCHAR(280) NOT NULL",
    "media_urls  JSONB",
    "like_count  INT DEFAULT 0",
    "rt_count    INT DEFAULT 0",
    "created_at  TIMESTAMP DEFAULT NOW()",
], indexes=[
    "idx_tweets_user_id_created ON tweets(user_id, created_at DESC)",
    "idx_tweets_created ON tweets(created_at DESC)",
], notes="ツイート本体 - 1日6億件 → 年間2190億件。シャーディング必須")

schema.add_table("follows", [
    "follower_id   BIGINT NOT NULL REFERENCES users(user_id)",
    "followee_id   BIGINT NOT NULL REFERENCES users(user_id)",
    "created_at    TIMESTAMP DEFAULT NOW()",
    "PRIMARY KEY (follower_id, followee_id)",
], indexes=[
    "idx_follows_followee ON follows(followee_id)",
], notes="フォロー関係 - 双方向インデックスが重要")

schema.add_table("timeline_cache", [
    "user_id     BIGINT NOT NULL",
    "tweet_id    BIGINT NOT NULL",
    "author_id   BIGINT NOT NULL",
    "score       FLOAT NOT NULL",  # ランキングスコア
    "created_at  TIMESTAMP DEFAULT NOW()",
    "PRIMARY KEY (user_id, score DESC, tweet_id)",
], notes="タイムラインキャッシュ（Redisで実装するのが一般的）")

print(schema.generate_ddl())

# 出力:
# -- PostgreSQL スキーマ定義
#
# -- ユーザー情報 - DAU 3億 → 約5億行想定
# CREATE TABLE users (
#     user_id         BIGINT PRIMARY KEY,
#     username        VARCHAR(15) UNIQUE NOT NULL,
#     ...
# );
# CREATE INDEX idx_users_username ON users(username);
# ...以下略
```

### 6.3 トレードオフ分析の方法論

設計においてトレードオフの議論は最も評価される部分である。「正解」を選ぶことよりも、**選択肢を列挙し、各選択肢の長所・短所を論理的に比較する力**が問われる。

```
トレードオフの三角形:

              性能 (Performance)
                /\
               /  \
              /    \
             / 最適 \
            /  解は  \
           /  中央   \
          /    域    \
         /____________\
   コスト              信頼性
  (Cost)           (Reliability)

  3つ全てを最大化することは不可能
  → ビジネス要件に応じてバランスを取る
```

```python
# トレードオフ分析の構造化フレームワーク
class TradeoffAnalysis:
    """設計トレードオフを体系的に分析するツール"""

    def __init__(self, decision: str):
        self.decision = decision
        self.options = []

    def add_option(self, name: str, pros: list, cons: list,
                   cost: str, complexity: str, when_to_use: str):
        self.options.append({
            "name": name,
            "pros": pros,
            "cons": cons,
            "cost": cost,
            "complexity": complexity,
            "when_to_use": when_to_use,
        })

    def analyze(self) -> str:
        lines = [f"=== トレードオフ分析: {self.decision} ===\n"]
        for i, opt in enumerate(self.options, 1):
            lines.append(f"【選択肢{i}】{opt['name']}")
            lines.append(f"  長所:")
            for p in opt['pros']:
                lines.append(f"    + {p}")
            lines.append(f"  短所:")
            for c in opt['cons']:
                lines.append(f"    - {c}")
            lines.append(f"  コスト: {opt['cost']}")
            lines.append(f"  複雑さ: {opt['complexity']}")
            lines.append(f"  適する場面: {opt['when_to_use']}")
            lines.append("")
        return "\n".join(lines)


# タイムライン配信のトレードオフ分析
analysis = TradeoffAnalysis("タイムライン配信方式の選択")

analysis.add_option(
    "Fan-out on Write (プッシュモデル)",
    pros=[
        "タイムライン読み込みが高速（事前計算済み）",
        "読み込みQPSが高くても対応可能",
        "実装がシンプル（読み側）",
    ],
    cons=[
        "フォロワー数が多いユーザーの投稿時に大量の書き込み発生",
        "有名人（100万フォロワー）の投稿で書き込みQPSが爆発",
        "ストレージコストが高い（各ユーザーのタイムラインを保存）",
    ],
    cost="ストレージ: 高い、計算: 書き込み時に集中",
    complexity="中程度",
    when_to_use="フォロワー数の上限が低いサービス、読み込みが書き込みより圧倒的に多い場合",
)

analysis.add_option(
    "Fan-out on Read (プルモデル)",
    pros=[
        "書き込みが軽量（自分のフィードに書くだけ）",
        "有名人のツイートでも負荷が偏らない",
        "ストレージ効率が良い",
    ],
    cons=[
        "タイムライン表示時に全フォロイーの投稿を集約する必要あり",
        "レイテンシが高い（リアルタイム集約）",
        "フォロイーが多いユーザーの読み込みが遅い",
    ],
    cost="ストレージ: 低い、計算: 読み込み時に集中",
    complexity="中程度",
    when_to_use="書き込みQPSが非常に高い場合、フォロイー数が少ない場合",
)

analysis.add_option(
    "ハイブリッドアプローチ（Twitterの実方式）",
    pros=[
        "一般ユーザーはプッシュ、有名人はプルで最適化",
        "両方の長所を享受できる",
        "実際にTwitterが採用している実績あり",
    ],
    cons=[
        "実装が複雑（2つのパスを管理）",
        "「有名人」の閾値設定が必要",
        "デバッグが困難",
    ],
    cost="ストレージ: 中程度、計算: 分散",
    complexity="高い",
    when_to_use="大規模SNSで、フォロワー数のばらつきが大きい場合",
)

print(analysis.analyze())

# 出力:
# === トレードオフ分析: タイムライン配信方式の選択 ===
#
# 【選択肢1】Fan-out on Write (プッシュモデル)
#   長所:
#     + タイムライン読み込みが高速（事前計算済み）
#     + 読み込みQPSが高くても対応可能
#     + 実装がシンプル（読み側）
#   短所:
#     - フォロワー数が多いユーザーの投稿時に大量の書き込み発生
#     ...以下略
```

---

## 7. 設計ドキュメント（Design Doc）の書き方

実務では、設計はドキュメントに残して合意形成を行う。Google、Meta、Amazon等の大手テック企業では「Design Doc」や「RFC」と呼ばれる設計文書を書く文化がある。

### 7.1 Design Docのテンプレート

```python
# 設計ドキュメントのテンプレートを生成
class DesignDocTemplate:
    """Design Doc / RFCのテンプレート"""

    SECTIONS = [
        ("1. タイトルとメタデータ", [
            "プロジェクト名",
            "著者",
            "レビュアー",
            "ステータス (Draft / In Review / Approved / Implemented / Deprecated)",
            "最終更新日",
        ]),
        ("2. 概要 (Summary)", [
            "1段落で何を・なぜ・どのように解決するかを記述",
            "非技術者にも理解できるレベルで記載",
        ]),
        ("3. 背景と動機 (Background & Motivation)", [
            "なぜこの設計が必要なのか",
            "現状の問題点は何か",
            "ビジネス上の影響は何か",
        ]),
        ("4. 要件 (Requirements)", [
            "機能要件 (Must/Should/Could/Won't)",
            "非機能要件 (性能、可用性、セキュリティ等)",
            "制約条件",
        ]),
        ("5. 提案する設計 (Proposed Design)", [
            "アーキテクチャ図",
            "コンポーネント説明",
            "API設計",
            "データモデル/スキーマ",
            "主要なシーケンス図",
        ]),
        ("6. 代替案 (Alternatives Considered)", [
            "検討した他のアプローチと不採用の理由",
            "※最低2つの代替案を記載する",
        ]),
        ("7. トレードオフ (Trade-offs)", [
            "今回の設計で受け入れたトレードオフ",
            "将来変更が必要になる可能性のある箇所",
        ]),
        ("8. マイルストーン (Milestones)", [
            "フェーズ分けと各フェーズの成果物",
            "スケジュール概算",
        ]),
        ("9. セキュリティ考慮事項 (Security Considerations)", [
            "認証/認可",
            "データ暗号化",
            "既知のリスクと対策",
        ]),
        ("10. テスト計画 (Test Plan)", [
            "ユニットテスト",
            "統合テスト",
            "負荷テスト",
            "カオスエンジニアリング",
        ]),
        ("11. 監視とアラート (Monitoring & Alerting)", [
            "主要メトリクス（SLI/SLO）",
            "アラート条件",
            "ダッシュボード設計",
        ]),
        ("12. オープンクエスチョン (Open Questions)", [
            "まだ決まっていない事項",
            "レビューで議論が必要な事項",
        ]),
    ]

    @classmethod
    def generate(cls) -> str:
        lines = ["# [プロジェクト名] Design Doc\n"]
        for section, items in cls.SECTIONS:
            lines.append(f"## {section}\n")
            for item in items:
                lines.append(f"- {item}")
            lines.append("")
        return "\n".join(lines)

print(DesignDocTemplate.generate())
```

---

## 8. 比較表

### 比較表1: 設計アプローチの比較

| 観点 | トップダウン設計 | ボトムアップ設計 | ハイブリッド |
|------|-----------------|-----------------|-------------|
| 開始点 | ビジネス要件から | 技術コンポーネントから | 要件+技術的制約の両面から |
| 長所 | ビジネス価値に直結 | 技術的正確性が高い | バランスが取れる |
| 短所 | 技術的実現可能性の見落とし | 過剰設計のリスク | 時間がかかる |
| 適する場面 | 設計面接、新規プロジェクト | 既存システムのリファクタ | 大規模商用システム |
| リスク | 実装不可能な設計 | ビジネス要件との乖離 | 分析麻痺 |
| 成果物 | コンポーネント図中心 | クラス図・シーケンス図中心 | 多層の設計文書 |

### 比較表2: 設計面接 vs 実務の設計

| 項目 | 設計面接 | 実務の設計 |
|------|---------|-----------|
| 時間 | 45-60分 | 数週間〜数か月 |
| 深さ | 広く浅く | 特定箇所を深く |
| 完璧さ | 不要（思考プロセスが重要） | 実装可能なレベルで必要 |
| フィードバック | 面接官との対話 | チーム全体でのレビュー |
| 成果物 | ホワイトボードの図 | Design Doc + プロトタイプ |
| 重視される点 | コミュニケーション能力 | 技術的正確性 |
| トレードオフ | 口頭で議論 | 文書化して記録 |
| コスト見積もり | 概算レベル | 詳細な見積もり |
| テスト計画 | 言及程度 | 詳細なテスト戦略 |

### 比較表3: 主要な技術選択のトレードオフ

| 技術選択 | 選択肢A | 選択肢B | 判断基準 |
|---------|---------|---------|---------|
| DB | SQL (PostgreSQL) | NoSQL (MongoDB) | データの構造化度、JOIN必要性、スケール要件 |
| 通信 | REST | gRPC | レイテンシ要件、型安全性、ブラウザ対応 |
| キャッシュ | Redis | Memcached | データ構造の複雑さ、永続化要件 |
| MQ | Kafka | RabbitMQ | スループット、順序保証、replay可能性 |
| 検索 | Elasticsearch | PostgreSQL FTS | 検索の複雑さ、データ量、リアルタイム性 |
| コンテナ | Kubernetes | ECS/Fargate | 運用チームの能力、カスタマイズ要件 |

---

## 9. アンチパターン

### アンチパターン1: 要件確認なしの即設計

```
--- NG例 ---

面接官: 「Twitterを設計してください」
候補者: 「まずマイクロサービスで構成します。Kafkaでイベント駆動にして、
        Redisでキャッシュして、PostgreSQLにシャーディングして...」

何が問題か:
- 要件を理解する前に技術選定に走っている
- 面接官は「思考プロセス」を見たいのに、結論だけ述べている
- 全てのTwitterが同じ設計になるわけではない（規模、優先事項、制約が異なる）

--- OK例 ---

面接官: 「Twitterを設計してください」
候補者: 「ありがとうございます。いくつか確認させてください。

  機能について:
  1. コア機能はツイート投稿とタイムライン表示でよいですか？
  2. 検索、通知、DM はスコープに含みますか？
  3. 画像/動画のサポートは必要ですか？

  規模について:
  4. DAUはどの程度を想定しますか？（1万? 1億? 10億?）
  5. 1ユーザーあたりの平均フォロー数は？

  非機能要件:
  6. タイムライン表示のレイテンシ目標は？
  7. 可用性の目標は？（99.9%? 99.99%?）
  8. データの一貫性は強い一貫性が必要ですか、結果整合性で十分ですか？」

なぜ良いか:
- スコープを絞ることで、深い議論が可能になる
- 非機能要件を明確にすることで、技術選定の根拠が生まれる
- 面接官との対話が生まれ、コミュニケーション能力をアピールできる
```

### アンチパターン2: 銀の弾丸症候群

```
--- NG例 ---

「全ての問題はキャッシュで解決できる」
「マイクロサービスにすれば全て解決」
「NoSQLを使えばスケールする」
「Kubernetesに載せれば自動でスケールする」

なぜ問題か:
- 万能な技術は存在しない。全ての技術にはトレードオフがある
- キャッシュ → 無効化の複雑さ、データ不整合のリスク
- マイクロサービス → ネットワーク遅延、分散トランザクション、運用の複雑さ
- NoSQL → JOINの困難さ、ACID保証の弱さ
- Kubernetes → 学習コスト、運用の複雑さ、小規模では過剰

--- OK例 ---

「読み込みが書き込みの100倍多いので、Read-throughキャッシュが有効です。
  ただし、以下のトレードオフを考慮する必要があります:

  1. キャッシュ無効化戦略: TTL vs イベント駆動
     → TTLは実装が簡単だが古いデータが残る可能性
     → イベント駆動は最新だが実装が複雑

  2. キャッシュの一貫性:
     → Write-through: 書き込み時にキャッシュも更新。一貫性高いが書き込み遅い
     → Write-behind: 非同期でキャッシュ更新。速いがデータ損失リスク

  3. キャッシュ雪崩の対策:
     → TTLにジッターを追加して一斉失効を防ぐ
     → サーキットブレーカーでDB過負荷を防ぐ

  今回はTTL=5分 + Write-throughの組み合わせを提案します。
  理由は、5秒程度の古いデータは許容でき、書き込みQPSが低いためです。」
```

### アンチパターン3: 過剰設計（Over-Engineering）

```
--- NG例 ---

DAU 1,000 のスタートアップの初期設計:
- Kubernetes 20ノードクラスタ
- 6つのマイクロサービス
- Kafka + Elasticsearch + Redis Cluster
- マルチリージョンデプロイ
- カスタムサービスメッシュ

月額コスト: $15,000
チーム: エンジニア2名

何が問題か:
- トラフィックに対して10倍以上のインフラ
- 2名では運用が回らない
- 開発速度が遅くなり、ビジネスチャンスを逃す

--- OK例 ---

DAU 1,000 のスタートアップの適正設計:
- Heroku / Railway / PaaS 上のモノリスアプリ
- PostgreSQL (マネージド)
- Redis (セッション管理のみ)
- S3 (ファイルストレージ)

月額コスト: $100-300
チーム: エンジニア2名

ポイント:
- 「スケールする設計」はしつつ「スケールした実装」は不要
  例: DBアクセスはリポジトリパターンで抽象化しておくが、
      実装は単一DBのまま。シャーディングは必要になったら対応
- DAU 10万を超えるまでこの構成で十分
```

### アンチパターン4: 非機能要件の無視

```
--- NG例 ---

設計書に機能要件のみ記載:
- ユーザー登録ができる
- 商品を検索できる
- 購入処理ができる
（以上）

何が問題か:
- レイテンシ目標がないため、3秒かかっても「正しい」設計
- 可用性目標がないため、1日1時間ダウンしても「問題ない」
- セキュリティ要件がないため、平文パスワードでも「仕様通り」

--- OK例 ---

機能要件 + 非機能要件を明記:
- ユーザー登録ができる
  → レスポンスタイム: < 500ms (P99)
  → パスワードはbcryptでハッシュ化
  → メール認証必須

- 商品を検索できる
  → レスポンスタイム: < 200ms (P99)
  → 100万商品でもページング表示

- 購入処理ができる
  → 可用性: 99.99%
  → 二重購入防止（冪等性キー）
  → PCI DSS準拠
```

---

## 10. 実践演習

### 演習1（基礎）: URL短縮サービスの要件定義

**課題:** URL短縮サービス（bit.ly風）の要件を整理し、見積もりを行ってください。

**要求事項:**
1. 機能要件を3つ以上列挙する
2. 非機能要件を3つ以上列挙する（数値目標付き）
3. QPS、ストレージの見積もりを計算する

```python
# 演習1のスケルトンコード

class URLShortenerRequirements:
    """URL短縮サービスの要件定義"""

    def define_requirements(self):
        """ここに要件を記述"""
        # TODO: 機能要件
        functional = [
            # "長いURLを短いURLに変換する",
            # "短いURLにアクセスすると元のURLにリダイレクトする",
            # ...
        ]

        # TODO: 非機能要件
        non_functional = {
            # "レイテンシ": "リダイレクト < 100ms (P99)",
            # "可用性": "99.99%",
            # ...
        }

        return functional, non_functional

    def estimate(self):
        """ここに見積もりを記述"""
        # TODO: DAUの仮定
        dau = 0  # ???

        # TODO: QPS計算
        write_qps = 0  # ???
        read_qps = 0   # ???

        # TODO: ストレージ計算
        # 1レコードあたりのサイズは？
        # 保存期間は？

        return {
            "dau": dau,
            "write_qps": write_qps,
            "read_qps": read_qps,
        }
```

**期待される出力の例:**

```
=== URL短縮サービス 要件定義 ===

【機能要件】
  Must Have:
    - 長いURLを短いURLに変換する
    - 短いURLにアクセスすると元のURLにリダイレクトする (301/302)
    - 短いURLの有効期限を設定できる
  Should Have:
    - カスタムエイリアス (例: short.url/my-brand)
    - クリック統計の表示
  Could Have:
    - QRコード生成
    - A/Bテスト用の複数URL対応

【非機能要件】
  - リダイレクトレイテンシ: < 50ms (P99)
  - 可用性: 99.99%
  - 1日あたり1000万URL生成、10億リダイレクト
  - URL一意性の保証（衝突なし）
  - 短いURL長: 7文字（Base62で 62^7 = 3.5兆パターン）

【見積もり】
  DAU: 1億
  書き込みQPS: 116 (ピーク: 347)
  読み込みQPS: 11,574 (ピーク: 34,722)
  ストレージ: 1レコード500B × 1日1000万件 = 5GB/日 = 1.8TB/年
```

---

### 演習2（応用）: チャットシステムの高レベル設計

**課題:** LINEやSlack風のチャットシステムの高レベル設計を行ってください。

**要求事項:**
1. 要件を整理する（1対1チャット + グループチャット）
2. 主要コンポーネントを列挙し、ASCII図で構成を描く
3. メッセージ送信のデータフローを説明する
4. DBスキーマ（テーブル3つ以上）を設計する

```python
# 演習2のスケルトンコード

class ChatSystemDesign:
    """チャットシステムの設計"""

    def requirements(self):
        """要件定義"""
        pass  # TODO

    def high_level_design(self):
        """高レベル設計（ASCII図）"""
        diagram = """
        TODO: ここに構成図を描く

        ヒント: 以下のコンポーネントを含める
        - WebSocket Gateway（リアルタイム通信）
        - Chat Service（メッセージ処理）
        - Presence Service（オンライン状態管理）
        - Notification Service（プッシュ通知）
        - Message Store（メッセージ永続化）
        - File Storage（ファイル添付）
        """
        return diagram

    def data_flow(self):
        """メッセージ送信のデータフロー"""
        pass  # TODO: A → B にメッセージを送る際の全ステップ

    def db_schema(self):
        """DBスキーマ設計"""
        pass  # TODO: users, conversations, messages テーブル
```

**期待される出力の例:**

```
=== チャットシステム 高レベル設計 ===

  Client A ─── WebSocket ──→ WS Gateway ──→ Chat Service
                                              │
                                              ├──→ Message Store (Cassandra)
                                              ├──→ Notification Service
                                              │       └──→ Push (APNS/FCM)
                                              └──→ WS Gateway ──→ Client B
                                                     (B がオンラインの場合)

  [メッセージ送信フロー]
  1. Client A がWebSocketでメッセージ送信
  2. WS Gateway が Chat Service に転送
  3. Chat Service がメッセージをDBに永続化
  4. Client B がオンライン → WS Gateway 経由でリアルタイム配信
  5. Client B がオフライン → Notification Service でプッシュ通知

  [DBスキーマ]
  users (user_id PK, name, status, last_seen)
  conversations (conv_id PK, type, participants[], created_at)
  messages (msg_id PK, conv_id FK, sender_id FK, content, type, created_at)
```

---

### 演習3（発展）: 既存システムの改善提案

**課題:** 以下の「問題を抱えたシステム設計」を分析し、改善提案を行ってください。

```python
# 問題のあるシステム設計
"""
現状のECサイト設計:
- 単一のモノリスアプリケーション (Python/Django)
- 単一のPostgreSQL (16 vCPU, 64GB RAM)
- セッションはDjangoのDB-backedセッション
- 画像はローカルファイルシステムに保存
- 全文検索はPostgreSQLのLIKE検索
- バッチ処理はcronジョブ
- デプロイは手動SSH + git pull
- モニタリングはなし
- DAU: 50万、今後1年で200万に成長予定

問題:
1. ピーク時（セール期間）にDBのCPU使用率が95%に達する
2. 商品検索が遅い（2-3秒）
3. 画像の読み込みが遅い（特に海外ユーザー）
4. デプロイ時にダウンタイムが発生
5. 障害発生時に原因特定に時間がかかる
"""

# TODO: 上記の各問題に対して、以下の形式で改善提案を行う
# 1. 問題の根本原因分析
# 2. 短期的な改善案（1-2週間で実施可能）
# 3. 中期的な改善案（1-3か月で実施可能）
# 4. 各改善案のトレードオフ
```

**期待される出力の例:**

```
=== EC サイト改善提案 ===

【問題1: DBのCPU使用率95%】
  根本原因: 読み書きが同一DBに集中。インデックス不足の可能性もあり

  短期（1-2週間）:
    - スロークエリログの分析とインデックス追加
    - N+1クエリの修正（Django select_related/prefetch_related）
    - 重いクエリのキャッシュ化（Redis + TTL 5分）
    → トレードオフ: キャッシュ分のデータ遅延が発生

  中期（1-3か月）:
    - Read Replicaの導入（読み込みを分散）
    - DB-backedセッション → Redisセッションに移行
    - コネクションプーリングの最適化（PgBouncer）
    → トレードオフ: レプリケーションラグによる一貫性の低下

【問題2: 商品検索が遅い】
  根本原因: LIKE検索はフルテーブルスキャンになりO(n)

  短期: PostgreSQL の GIN インデックス + pg_trgm 拡張
  中期: Elasticsearch の導入
  → トレードオフ: ESの運用コスト vs 検索品質の向上

...以下、問題3-5についても同様に分析
```

---

## 11. FAQ

### Q1: システム設計の勉強はどこから始めるべきですか？

まず本ガイドの「00-fundamentals」セクションで基礎概念（スケーラビリティ、信頼性、CAP定理）を固める。次に「01-components」で個別コンポーネント（ロードバランサー、キャッシュ、メッセージキュー）を学ぶ。最後に「03-case-studies」で実際の設計問題を解く流れが効率的である。

**推奨学習パス:**

```
Week 1-2: 基礎概念
  → 00-system-design-overview.md (本ガイド)
  → 01-scalability.md
  → 02-reliability.md
  → 03-cap-theorem.md

Week 3-4: コンポーネント
  → 00-load-balancer.md
  → 01-caching.md
  → 02-message-queue.md
  → 03-cdn.md
  → 04-database-scaling.md

Week 5-6: アーキテクチャ
  → 00-monolith-vs-microservices.md
  → 01-clean-architecture.md
  → 02-ddd.md
  → 03-event-driven.md

Week 7-8: ケーススタディ（演習）
  → 00-url-shortener.md
  → 01-chat-system.md
  → 02-notification-system.md
  → 03-rate-limiter.md
  → 04-search-engine.md
```

書籍としては *Designing Data-Intensive Applications*（Martin Kleppmann著）が最も推奨される。

### Q2: 設計面接でホワイトボードに何を描けばよいですか？

最低限以下の要素を含める:

1. **クライアント**: ブラウザまたはモバイルアプリ
2. **ロードバランサー**: L4/L7の区別を明記
3. **アプリケーションサーバー群**: Auto Scaling Groupであることを示す
4. **キャッシュレイヤー**: Redis/Memcachedの選択理由を添えて
5. **データストア**: Primary/Replicaの構成、シャーディング有無

矢印でデータフローの方向を示し、各コンポーネント間のプロトコル（HTTP、gRPC、WebSocket等）を明記する。数値（QPS、レイテンシ目標）もボックスの近くに記載すると説得力が増す。

**ホワイトボードの図で差がつくポイント:**

- 各コンポーネントの数値（QPS、レイテンシ）を添える
- 障害ポイントとその対策を示す
- 非同期処理の経路を別の色/線種で描く
- ボトルネックになりうる箇所を明示する

### Q3: 実務とシステム設計面接の違いは何ですか？

面接では45分という制約の中で「思考プロセス」を見せることが重要であり、完璧な設計は求められない。実務では時間をかけて段階的に設計し、プロトタイプで検証する。

**具体的な違い:**

| 側面 | 面接 | 実務 |
|------|------|------|
| 目的 | 思考プロセスの評価 | 実装可能な設計の策定 |
| 時間 | 45-60分 | 数週間〜数か月 |
| 深さ | 全体像 + 1-2箇所の深掘り | 全箇所を詳細に |
| 不確実性 | 仮定を置いて進める | 関係者と合意形成 |
| フィードバック | 面接官との対話 | レビュー、プロトタイプ検証 |
| 結果の影響 | 採用判断のみ | ビジネスに直接影響 |

### Q4: CAP定理を設計面接で使いこなすにはどうすればよいですか？

CAP定理は「分散システムにおいてConsistency（一貫性）、Availability（可用性）、Partition Tolerance（分断耐性）の3つのうち、最大で2つしか同時に保証できない」という定理である。ただし、実務上はPartition Toleranceは避けられない（ネットワークは必ず分断する）ため、実質的にはCPとAPの選択となる。

設計面接では「このシステムはCPとAPのどちらを優先すべきか」という観点で議論すると良い。例えば:
- **金融取引システム → CP**: 残高の不整合は絶対に許されない
- **SNSのタイムライン → AP**: 数秒の遅延は許容、常に表示できることが重要

詳細は [CAP定理](./03-cap-theorem.md) で解説する。

### Q5: マイクロサービスはいつ導入すべきですか？

マイクロサービスは「いつ導入するか」よりも「本当に必要か」をまず問うべきである。以下のサインが複数出たら検討の余地がある:

1. **チーム規模**: 50人以上のエンジニアがモノリスを共有している
2. **デプロイ頻度**: 他チームの変更との衝突でデプロイが遅れる
3. **スケーリング**: 特定の機能だけ負荷が高いのにシステム全体をスケールしている
4. **技術的多様性**: 機能によって最適な技術スタックが異なる

逆に、チーム5人以下のスタートアップでマイクロサービスを導入するのは、ほぼ確実に過剰設計である。

詳細は [モノリス vs マイクロサービス](../02-architecture/00-monolith-vs-microservices.md) で解説する。

### Q6: システム設計で最もよくある失敗パターンは何ですか？

1. **要件を確認しない**: 面接でも実務でも最大の失敗原因
2. **早すぎる最適化**: 「将来10億ユーザーになるかも」で初日からシャーディング
3. **銀の弾丸思考**: 「マイクロサービスにすれば全て解決」
4. **非機能要件の無視**: 機能は動くが遅い・落ちる・セキュリティホールだらけ
5. **トレードオフを議論しない**: 一つの案だけ提示して代替案を示さない
6. **数値の裏付けがない**: 「たぶん大丈夫」で設計を進める

---

## 12. まとめ

| 項目 | ポイント |
|------|---------|
| システム設計の目的 | ビジネス要件を満たすスケーラブルで信頼性の高いアーキテクチャを定義する |
| 4ステップ | 要件確認 → 見積もり → 高レベル設計 → 詳細設計 |
| 要件の分類 | 機能要件（何をするか）と非機能要件（どう実現するか）の両方が必須 |
| MoSCoW法 | 機能要件をMust/Should/Could/Won'tで優先順位付け |
| 見積もりの重要性 | QPS、ストレージ、帯域幅を概算し設計判断の根拠とする |
| トレードオフ思考 | 性能・コスト・信頼性は同時最大化不可、ビジネスに応じて選択 |
| Design Doc | 設計は文書化し、代替案とトレードオフを記録する |
| 面接のコツ | 要件確認から始め、思考プロセスを声に出しながら進める |
| 過剰設計の回避 | 規模に合った設計を。DAU1000にKubernetesは不要 |
| 段階的進化 | モノリスから始めて、必要に応じてスケールアウトする |

---

## 次に読むべきガイド

- [スケーラビリティ](./01-scalability.md) -- 水平/垂直スケーリングの基礎と自動スケーリング
- [信頼性](./02-reliability.md) -- フォールトトレランスと冗長化設計
- [CAP定理](./03-cap-theorem.md) -- 分散システム設計の理論的基盤
- [ロードバランサー](../01-components/00-load-balancer.md) -- 負荷分散の実装パターン
- [キャッシング](../01-components/01-caching.md) -- キャッシュ戦略と無効化手法
- [メッセージキュー](../01-components/02-message-queue.md) -- 非同期処理と疎結合設計
- [モノリス vs マイクロサービス](../02-architecture/00-monolith-vs-microservices.md) -- アーキテクチャ選択の判断基準

### 関連する他のSkill

- [clean-code-principles](../../clean-code-principles/) -- コードレベルの設計原則（SOLID、結合度/凝集度）
- [design-patterns-guide](../../design-patterns-guide/) -- GoFデザインパターンとアーキテクチャパターン
  - [Repository Pattern](../../design-patterns-guide/docs/04-architectural/01-repository-pattern.md) -- データアクセスの抽象化
  - [Event Sourcing / CQRS](../../design-patterns-guide/docs/04-architectural/02-event-sourcing-cqrs.md) -- イベント駆動設計の応用

---

## 参考文献

1. Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly Media. -- システム設計の教科書的存在。データシステムの基礎から分散処理まで網羅
2. Xu, A. (2020). *System Design Interview -- An insider's guide*. Byte Code LLC. -- 設計面接の定番書。12の実際の設計問題を解説
3. Xu, A. (2022). *System Design Interview -- An Insider's Guide: Volume 2*. Byte Code LLC. -- 上記の続編。より高度な設計問題を扱う
4. Dean, J. & Barroso, L.A. (2013). "The Tail at Scale." *Communications of the ACM*, 56(2), 74-80. -- レイテンシのテール問題に関するGoogle論文
5. Google SRE Book -- https://sre.google/sre-book/table-of-contents/ -- GoogleのSREプラクティスを体系化した書籍
6. Amazon Builders' Library -- https://aws.amazon.com/builders-library/ -- Amazonが公開する分散システムのベストプラクティス集
7. Martin, R.C. (2017). *Clean Architecture*. Prentice Hall. -- アーキテクチャ設計の原則を解説
8. Brooks, F.P. (1975). *The Mythical Man-Month*. Addison-Wesley. -- ソフトウェア工学の古典。設計における本質的な難しさを論じる
