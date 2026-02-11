# 技術的負債

> 技術的負債とは、短期的な利益のために品質を犠牲にすることで将来の開発コストが増加する現象である。Ward Cunningham が1992年に提唱したメタファーを基に、負債の分類・可視化・計画的返済戦略を体系的に解説する

## この章で学ぶこと

1. **技術的負債の分類** — 意図的/無意識的、慎重/無謀の4象限モデルと具体例
2. **負債の可視化と定量化** — メトリクス収集、コスト算出、経営層への説明手法
3. **計画的な返済戦略** — 20%ルール、ボーイスカウトルール、技術的負債スプリント

---

## 1. 技術的負債の4象限モデル

### 1.1 Martin Fowler の4象限

```
          意図的 (Deliberate)
              |
  +-----------+-----------+
  |  慎重 × 意図的        |  無謀 × 意図的        |
  |                       |                       |
  | 「今はこの設計で      | 「テスト書く時間が    |
  |  リリースし、次の     |  ないから省略しよう」 |
  |  スプリントで改善」   |                       |
  +-----------+-----------+
  |  慎重 × 無意識的      |  無謀 × 無意識的      |
  |                       |                       |
  | 「今ならもっと良い    | 「レイヤー化って      |
  |  設計ができたのに」   |  何？」              |
  |  (学習による発見)     |  (スキル不足)        |
  +-----------+-----------+
              |
          無意識的 (Inadvertent)

  慎重 (Prudent) ----+---- 無謀 (Reckless)
```

### 1.2 負債の種類と「利子」

```
技術的負債のメタファー

  +------------------+
  |  元本 (Principal) |  = 品質が低いコード自体
  +------------------+
          |
          v  時間の経過
  +------------------+
  |  利子 (Interest)  |  = 品質が低いことで発生する追加コスト
  +------------------+
     |        |        |
     v        v        v
  バグ修正  機能追加  オンボーディング
  に2倍    に3倍     に2週間
  の時間   の時間    余計にかかる

  返済しないと利子が複利で膨らむ
  → いずれ「利子の支払いだけで精一杯」に
```

### 1.3 負債の具体例マッピング

```
ソースコード層
├── 重複コード (DRY 違反)
├── 長大なメソッド/クラス (God Object)
├── 不明瞭な命名
└── ハードコードされた値

アーキテクチャ層
├── 循環依存
├── レイヤー違反 (UI → DB 直接参照)
├── モノリスの肥大化
└── 不適切なデータモデル

テスト層
├── テストカバレッジの不足
├── Flaky テスト
├── テストの遅さ
└── 統合テスト偏重

インフラ/運用層
├── 手動デプロイ
├── 監視・アラートの不足
├── 古いライブラリ/フレームワーク
└── ドキュメントの陳腐化
```

---

## 2. 負債の可視化と定量化

### 2.1 メトリクス収集

```python
# 技術的負債メトリクス収集スクリプト
import subprocess
import json
from datetime import datetime

def collect_debt_metrics(repo_path: str) -> dict:
    """リポジトリの技術的負債指標を収集"""
    metrics = {}

    # 1. コード複雑度 (Cyclomatic Complexity)
    result = subprocess.run(
        ['radon', 'cc', repo_path, '-a', '-j'],
        capture_output=True, text=True
    )
    cc_data = json.loads(result.stdout)
    metrics['avg_complexity'] = cc_data.get('average', 0)
    metrics['high_complexity_functions'] = sum(
        1 for f in cc_data.get('functions', []) if f['complexity'] > 10
    )

    # 2. コード重複率
    result = subprocess.run(
        ['jscpd', repo_path, '--reporters', 'json', '--min-lines', '5'],
        capture_output=True, text=True
    )
    metrics['duplication_percentage'] = parse_duplication(result.stdout)

    # 3. 依存関係の古さ
    result = subprocess.run(
        ['pip', 'list', '--outdated', '--format=json'],
        capture_output=True, text=True
    )
    outdated = json.loads(result.stdout)
    metrics['outdated_dependencies'] = len(outdated)
    metrics['critical_updates'] = sum(
        1 for d in outdated if is_major_version_behind(d)
    )

    # 4. TODO/FIXME/HACK の数
    result = subprocess.run(
        ['grep', '-r', '-c', '-E', 'TODO|FIXME|HACK|XXX', repo_path],
        capture_output=True, text=True
    )
    metrics['todo_count'] = sum(int(line.split(':')[-1])
                                for line in result.stdout.strip().split('\n') if line)

    # 5. テストカバレッジ
    result = subprocess.run(
        ['pytest', '--cov', repo_path, '--cov-report=json'],
        capture_output=True, text=True
    )
    metrics['test_coverage'] = parse_coverage(result.stdout)

    return metrics
```

### 2.2 ダッシュボード表示

```python
# 技術的負債ダッシュボード (テキスト版)
def print_debt_dashboard(metrics: dict):
    print("=" * 60)
    print("  技術的負債ダッシュボード")
    print("=" * 60)
    print(f"  平均複雑度:          {metrics['avg_complexity']:.1f} "
          f"{'[OK]' if metrics['avg_complexity'] < 5 else '[WARNING]'}")
    print(f"  高複雑度関数:        {metrics['high_complexity_functions']} 件")
    print(f"  コード重複率:        {metrics['duplication_percentage']:.1f}% "
          f"{'[OK]' if metrics['duplication_percentage'] < 5 else '[WARNING]'}")
    print(f"  古い依存関係:        {metrics['outdated_dependencies']} 件")
    print(f"  TODO/FIXME:          {metrics['todo_count']} 件")
    print(f"  テストカバレッジ:    {metrics['test_coverage']:.1f}% "
          f"{'[OK]' if metrics['test_coverage'] > 80 else '[WARNING]'}")
    print("=" * 60)

    # 負債スコア算出 (0-100, 低いほど良い)
    score = calculate_debt_score(metrics)
    print(f"  技術的負債スコア:    {score}/100")
    if score < 30:
        print("  状態: 健全")
    elif score < 60:
        print("  状態: 要注意 — 計画的な返済を推奨")
    else:
        print("  状態: 危険 — 開発速度に深刻な影響")
```

### 2.3 コスト算出テンプレート

```python
# 技術的負債のコスト見積もり
def estimate_debt_cost():
    """技術的負債のビジネスコストを試算"""

    # 開発者の時間コスト
    developer_hourly_rate = 5000  # 円/時間
    team_size = 8

    # 負債によるオーバーヘッド (週あたり)
    overhead_hours_per_week = {
        'バグ修正の追加時間':        8,   # 本来2hで済む修正が+8h
        '機能追加の複雑性コスト':     12,  # レガシーコードとの格闘
        'オンボーディングコスト':      4,   # 新メンバーの理解に時間
        '手動テスト・デプロイ':        6,   # 自動化されていない作業
        'ライブラリ脆弱性対応':       2,   # 古い依存の緊急対応
    }

    total_weekly_hours = sum(overhead_hours_per_week.values())
    weekly_cost = total_weekly_hours * developer_hourly_rate
    annual_cost = weekly_cost * 50  # 50週/年

    print(f"週あたり無駄な工数: {total_weekly_hours}時間")
    print(f"週あたりコスト: {weekly_cost:,}円")
    print(f"年間コスト: {annual_cost:,}円")
    # → 年間コスト: 8,000,000円 (例)
```

---

## 3. 返済戦略

### 3.1 段階的返済計画

```
フェーズ 1 (即時): ボーイスカウトルール
  「コードを見つけた時より少しきれいにして去る」
  - 変数名のリネーム
  - 不要なコメントの削除
  - 小さなリファクタリング

フェーズ 2 (スプリント内): 20%ルール
  各スプリントの20%を技術的負債の返済に充てる
  - テスト追加
  - 重複コードの統合
  - 依存関係の更新

フェーズ 3 (四半期): 技術的負債スプリント
  四半期に1回、負債返済専用のスプリント
  - アーキテクチャの改善
  - 大規模リファクタリング
  - インフラ近代化
```

### 3.2 負債の優先度付け

```python
# 負債アイテムの優先度計算
from dataclasses import dataclass

@dataclass
class DebtItem:
    name: str
    impact: int           # ビジネスへの影響 (1-5)
    fix_effort: int       # 修正工数 (1-5, 1=小)
    frequency: int        # 影響頻度 (1-5, 5=毎日)
    risk: int             # リスク (1-5)

    @property
    def priority_score(self) -> float:
        """優先度スコア: 高いほど先に返済すべき"""
        return (self.impact * self.frequency * self.risk) / self.fix_effort

debt_items = [
    DebtItem("テストカバレッジ不足",   impact=5, fix_effort=3, frequency=5, risk=4),
    DebtItem("モノリス分割",           impact=4, fix_effort=5, frequency=3, risk=3),
    DebtItem("手動デプロイ",           impact=3, fix_effort=2, frequency=5, risk=4),
    DebtItem("古いReactバージョン",    impact=2, fix_effort=3, frequency=2, risk=3),
    DebtItem("ドキュメント陳腐化",     impact=2, fix_effort=1, frequency=3, risk=1),
]

# 優先度順にソート
sorted_items = sorted(debt_items, key=lambda x: x.priority_score, reverse=True)
for item in sorted_items:
    print(f"  {item.priority_score:.1f}  {item.name}")
```

---

## 4. 比較表

| 返済戦略 | コスト | 効果 | リスク | 適用場面 |
|---------|-------|------|--------|---------|
| ボーイスカウトルール | 最小 | 漸進的 | 最小 | 日常的な改善 |
| 20%ルール | 低 | 中期的 | 低 | スプリント内での改善 |
| 技術的負債スプリント | 中 | 大きい | 中 | 四半期ごとの集中改善 |
| Strangler Fig | 高 | 根本的 | 中 | レガシーシステムの置換 |
| フルリライト | 最高 | 根本的 | 最高 | 最終手段（非推奨） |

| 指標 | 健全 | 要注意 | 危険 |
|------|------|--------|------|
| テストカバレッジ | > 80% | 50-80% | < 50% |
| コード重複率 | < 3% | 3-10% | > 10% |
| 平均複雑度 (CC) | < 5 | 5-10 | > 10 |
| 古い依存関係 | < 5% | 5-20% | > 20% |
| デプロイ頻度 | 日次以上 | 週次 | 月次以下 |
| リードタイム | < 1日 | 1-7日 | > 7日 |

---

## 5. アンチパターン

### アンチパターン 1: 「後で直す」と言い続ける

```
BAD:
  Sprint 1: 「時間がないから後で直す」→ TODO コメント追加
  Sprint 2: 「今回も時間がない」→ TODO が増殖
  Sprint 6: 「もう誰も全体像がわからない」→ 手遅れ

GOOD: 負債を「発行」として記録し、返済計画を立てる
  1. 技術的負債をバックログアイテムとして起票
  2. 影響度・修正工数を見積もり
  3. 各スプリントで20%の時間を確保
  4. 四半期ごとに負債残高をレビュー
```

### アンチパターン 2: 全ての負債を同時に返済しようとする

```
BAD:
  「今月は技術的負債一掃月間だ！」
  → 全員が別々の負債に取り組む
  → 各改善が中途半端
  → 新機能が1ヶ月停止
  → ビジネス側の信頼を失う

GOOD: 優先度付けて1つずつ完了させる
  Sprint N:   テストカバレッジ60%→80% (最優先)
  Sprint N+1: CI/CDパイプライン構築
  Sprint N+2: 手動デプロイの自動化
  → 各スプリントで具体的な成果を出す
```

---

## 6. FAQ

### Q1. 技術的負債を経営層にどう説明する？

**A.** 金融メタファーで説明する。「技術的負債は住宅ローンのようなもので、毎月の利子（追加開発コスト）を払い続けている。年間推定800万円の利子を払っている。300万円を投資して元本を返済すれば、翌年から年500万円のコスト削減になる」。具体的な数値（開発速度の低下率、バグ修正にかかる時間）を示すことが説得力を高める。

### Q2. 技術的負債をゼロにすべきか？

**A.** ゼロにする必要はない。住宅ローン同様、「適切な量の負債」は戦略的に有用。重要なのは「負債の可視化」と「利子が制御可能な範囲内であること」。新規事業の初期フェーズでは意図的に負債を抱えて速度を優先し、プロダクトマーケットフィットが確認できたら計画的に返済するのが合理的。

### Q3. 技術的負債とビジネス要求のバランスはどう取る？

**A.** スプリント容量の20%を技術的負債に常に確保する「20%ルール」が広く使われている。これにより新機能開発を80%維持しつつ、負債が複利で膨らむのを防ぐ。緊急のビジネス要求時は一時的に100%を新機能に充てるが、次のスプリントで40%を負債返済に充てて帳尻を合わせる。重要なのは「負債を意識的に管理する文化」の構築。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 技術的負債の定義 | 品質を犠牲にして発生する将来の追加コスト |
| 4象限モデル | 意図的/無意識的 x 慎重/無謀で分類 |
| 利子の概念 | 返済しないと複利で膨らみ、開発速度が低下 |
| 可視化 | メトリクス収集、ダッシュボード、コスト試算 |
| 返済戦略 | ボーイスカウトルール → 20%ルール → 負債スプリント |
| 優先度付け | impact x frequency x risk / effort で算出 |
| バランス | 新機能80%:負債返済20%の配分を基本とする |

---

## 次に読むべきガイド

- [レガシーコード](./02-legacy-code.md) — レガシーコードへの安全な変更技法
- [継続的改善](./04-continuous-improvement.md) — CI/CD と品質メトリクスの自動化
- [コードレビューチェックリスト](../03-practices-advanced/04-code-review-checklist.md) — レビューで負債の発生を予防

---

## 参考文献

1. **Managing Technical Debt** — Philippe Kruchten, Robert Nord, Ipek Ozkaya (SEI/CMU, 2019) — 技術的負債の学術的・実践的フレームワーク
2. **Refactoring** — Martin Fowler (Addison-Wesley, 2018) — コード品質改善のカタログ
3. **Technical Debt Quadrant** — Martin Fowler (Blog, 2009) — https://martinfowler.com/bliki/TechnicalDebtQuadrant.html
4. **Accelerate** — Nicole Forsgren, Jez Humble, Gene Kim (IT Revolution, 2018) — DevOps メトリクスと組織パフォーマンス
