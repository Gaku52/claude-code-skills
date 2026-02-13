# 技術的負債

> 技術的負債（Technical Debt）とは、短期的な利益のために品質を犠牲にすることで将来の開発コストが増加する現象である。Ward Cunningham が1992年に OOPSLA で提唱したこのメタファーは、ソフトウェア開発における品質とスピードのトレードオフを金融の「借金」になぞらえたものであり、意思決定者・開発者双方が共通言語として使える強力な概念である。本ガイドでは、負債の分類・可視化・定量化・計画的返済戦略を体系的に解説し、負債を「管理可能な投資」として扱うためのフレームワークを提供する

## 前提知識

| トピック | 必要レベル | 参照ガイド |
|---------|----------|-----------|
| クリーンコード原則 | 基礎 | [原則と命名](../00-principles/00-naming.md) |
| コードスメル | 基礎 | [コードスメル](./00-code-smells.md) |
| リファクタリング技法 | 基礎 | [リファクタリング技法](./01-refactoring-techniques.md) |
| レガシーコード | 推奨 | [レガシーコード](./02-legacy-code.md) |
| テスト原則 | 推奨 | [テスト原則](../01-practices/04-testing-principles.md) |

## この章で学ぶこと

1. **技術的負債の本質と分類** -- Ward Cunningham の原義、Martin Fowler の4象限モデル、Steve McConnell の分類法を統合的に理解する
2. **負債の可視化と定量化** -- メトリクス収集、ホットスポット分析、コスト算出によって「見えない負債」を数値化する手法
3. **経営層への説明技法** -- 金融メタファーを活用し、投資対効果（ROI）として負債返済を提案する方法
4. **段階的返済戦略** -- ボーイスカウトルール、20%ルール、負債スプリント、Strangler Fig パターンの使い分け
5. **負債管理の組織文化** -- 負債を「悪」として隠すのではなく、透明に管理し戦略的に活用するチーム文化の構築

---

## 1. 技術的負債の本質

### 1.1 Ward Cunningham の原義

Ward Cunningham が1992年に述べた原文を正確に理解することが重要である:

```
「初回のコード出荷は借金をすることに似ている。少額の借金は、書き直しで
速やかに返済される限り、開発を加速させる。危険なのは、借金が返済されない
場合だ。未熟なコードに費やされるすべてのminute が、その借金の利子として
カウントされる。」
-- Ward Cunningham, OOPSLA 1992
```

重要なのは、Cunningham が「意図的に品質を下げること」を負債と呼んだのではなく、「現時点の理解で書いたコードが、後の学びによって不適切になる」ことを負債と呼んだ点である。

```
  Cunningham の原義              よくある誤解
  ┌─────────────────────┐      ┌─────────────────────┐
  │ 学習による知見の蓄積  │      │ 手抜きや雑なコード   │
  │ → 以前の設計が最適   │      │ → テスト省略、       │
  │   でなくなる         │      │   コピペ、ハック     │
  │ → リファクタリングで │      │ → 「負債だから       │
  │   返済する           │      │   仕方ない」         │
  └─────────────────────┘      └─────────────────────┘
```

### 1.2 Martin Fowler の4象限モデル

Martin Fowler は技術的負債を2つの軸で4象限に分類した:

```
                    意図的 (Deliberate)
                         |
    ┌────────────────────┼────────────────────┐
    │ 慎重 × 意図的       │ 無謀 × 意図的       │
    │                    │                    │
    │ 「今はこの設計で    │ 「テスト書く時間が  │
    │  リリースし、次の   │  ないから省略」     │
    │  スプリントで改善」 │                    │
    │                    │ 「動けばいい」      │
    │ [戦略的負債]        │ [怠惰な負債]        │
    │ 返済計画あり        │ 返済計画なし        │
    ├────────────────────┼────────────────────┤
    │ 慎重 × 無意識的     │ 無謀 × 無意識的     │
    │                    │                    │
    │ 「今ならもっと良い  │ 「レイヤー化って何？」│
    │  設計ができた」     │ 「SOLID？聞いたこと │
    │ (学習後に発見)      │  ない」             │
    │                    │                    │
    │ [発見的負債]        │ [無知な負債]         │
    │ 学習の証            │ スキル不足           │
    └────────────────────┼────────────────────┘
                         |
                    無意識的 (Inadvertent)

    慎重 (Prudent) ──────┼────── 無謀 (Reckless)
```

### 1.3 Steve McConnell の分類

Steve McConnell はさらに実務的な分類を提唱した:

```
技術的負債の分類 (McConnell)

1. 意図的な負債 (Intentional Debt)
   ├── 短期的: 「デモまでにMVPを出す。来週リファクタリング」
   ├── 長期的: 「このアーキテクチャで3年は持つ。その後再設計」
   └── 戦略的: 「市場投入を優先し、品質は段階的に向上」

2. 非意図的な負債 (Unintentional Debt)
   ├── 設計負債: 設計判断が後から不適切と判明
   ├── コード負債: 知識不足による低品質コード
   └── ビット腐敗: 時間経過による依存関係の陳腐化

3. 環境的な負債 (Environmental Debt)
   ├── プラットフォーム: OS・ランタイムのEOL
   ├── フレームワーク: メジャーバージョンの乖離
   └── ツールチェーン: ビルド・デプロイツールの陳腐化
```

### 1.4 負債の「利子」メカニズム

```
技術的負債のライフサイクル

  ┌──────────────┐
  │ 負債の発生     │  コード品質の妥協、設計上のショートカット
  └──────┬───────┘
         v
  ┌──────────────┐
  │ 元本 (Principal)│  = 品質が低いコード自体
  └──────┬───────┘
         v  時間経過 + コードベースの成長
  ┌──────────────┐
  │ 利子 (Interest) │  = 負債が存在することで発生する追加コスト
  └──────┬───────┘
         │
    ┌────┼────┬────────┬────────┐
    v    v    v        v        v
  バグ  機能追加  理解    オンボー  セキュリ
  修正  に3倍    に2倍   ディング  ティ
  に2倍  の時間   の時間  に+2週間  リスク
  の時間

  返済しないと利子が複利 (compound interest) で膨らむ
  ┌─────────────────────────────────────────────┐
  │                        ****                  │
  │                     ***                      │
  │ コスト            ***                         │
  │                 **                            │
  │               **         ← 複利で膨張する利子  │
  │             **                                │
  │           **                                  │
  │         **                                    │
  │       *                                       │
  │     *     ← 元本（初期の品質妥協）              │
  │   *                                           │
  │ *                                             │
  └─────────────────────────────────────────────┘
    T0    T1    T2    T3    T4    T5  → 時間
```

---

## 2. 負債の具体例マッピング

### 2.1 レイヤー別の技術的負債

```
技術的負債マップ

  ┌─────────────────────────────────────────────┐
  │  ソースコード層                               │
  │  ├── 重複コード (DRY 違反)                    │
  │  ├── 長大なメソッド/クラス (God Object)        │
  │  ├── 不明瞭な命名                            │
  │  ├── ハードコードされた値 (マジックナンバー)    │
  │  ├── 不適切な抽象化 (過剰/不足)               │
  │  └── 型安全性の欠如                           │
  ├─────────────────────────────────────────────┤
  │  アーキテクチャ層                              │
  │  ├── 循環依存                                │
  │  ├── レイヤー違反 (UI → DB 直接参照)           │
  │  ├── モノリスの肥大化                         │
  │  ├── 不適切なデータモデル                      │
  │  ├── API の一貫性欠如                         │
  │  └── コンポーネント境界の曖昧さ                │
  ├─────────────────────────────────────────────┤
  │  テスト層                                     │
  │  ├── テストカバレッジの不足                    │
  │  ├── Flaky テスト (不安定テスト)               │
  │  ├── テスト実行速度の遅さ                      │
  │  ├── 統合テスト偏重 (Ice Cream Cone)          │
  │  └── テストの可読性・保守性の低さ              │
  ├─────────────────────────────────────────────┤
  │  インフラ/運用層                               │
  │  ├── 手動デプロイ                             │
  │  ├── 監視・アラートの不足                      │
  │  ├── 古いライブラリ/フレームワーク              │
  │  ├── ドキュメントの陳腐化                      │
  │  ├── 環境差異 (dev ≠ staging ≠ prod)          │
  │  └── シークレット管理の不備                    │
  ├─────────────────────────────────────────────┤
  │  プロセス層                                   │
  │  ├── コードレビューの形骸化                    │
  │  ├── リリースプロセスの属人化                   │
  │  ├── ナレッジの暗黙知化                        │
  │  └── インシデント対応の未整備                   │
  └─────────────────────────────────────────────┘
```

### 2.2 負債の深刻度レベル

```python
"""技術的負債の深刻度分類"""
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta


class DebtSeverity(IntEnum):
    """負債の深刻度レベル"""
    TRIVIAL = 1    # 見た目の問題: 命名、フォーマット
    MINOR = 2      # 小さな設計問題: 重複コード、長いメソッド
    MAJOR = 3      # 重要な設計問題: 循環依存、レイヤー違反
    CRITICAL = 4   # 深刻な構造問題: セキュリティ脆弱性、データ不整合
    BLOCKER = 5    # 致命的: 本番障害リスク、スケーラビリティ限界


class DebtCategory(str):
    CODE = "code"
    ARCHITECTURE = "architecture"
    TEST = "test"
    INFRASTRUCTURE = "infrastructure"
    PROCESS = "process"


@dataclass
class TechnicalDebtItem:
    """技術的負債の個別アイテム"""
    id: str
    title: str
    description: str
    category: str
    severity: DebtSeverity

    # 影響度メトリクス
    impact_score: int          # ビジネスへの影響 (1-5)
    fix_effort_days: float     # 修正にかかる推定日数
    affected_frequency: int    # 影響を受ける頻度 (1-5, 5=毎日)
    risk_score: int            # リスク (1-5)

    # 追跡情報
    discovered_date: datetime = field(default_factory=datetime.now)
    reporter: str = ""
    assigned_to: Optional[str] = None
    target_sprint: Optional[str] = None
    status: str = "open"  # open, in_progress, resolved, wont_fix

    # 関連ファイル
    affected_files: list[str] = field(default_factory=list)
    related_tickets: list[str] = field(default_factory=list)

    @property
    def priority_score(self) -> float:
        """優先度スコア: 高いほど先に返済すべき

        計算式: (impact * frequency * risk) / effort
        - 影響が大きく、頻繁に発生し、リスクが高い負債を優先
        - 修正工数が小さい負債を優先 (Quick Win)
        """
        return (
            self.impact_score * self.affected_frequency * self.risk_score
        ) / max(self.fix_effort_days, 0.5)

    @property
    def annual_interest_cost(self) -> float:
        """年間利子コスト（概算、人日単位）

        利子 = 影響度 × 頻度 × 0.1日（1回あたりのオーバーヘッド）× 250営業日/年
        """
        overhead_per_occurrence = 0.1 * self.impact_score
        occurrences_per_year = self.affected_frequency * 50  # 週5日 × 50週
        return overhead_per_occurrence * occurrences_per_year

    @property
    def roi(self) -> float:
        """投資対効果: 年間利子削減額 / 修正工数

        ROI > 1: 1年以内に投資回収
        ROI > 3: 四半期以内に回収（高優先）
        """
        return self.annual_interest_cost / max(self.fix_effort_days, 0.5)

    @property
    def debt_age_days(self) -> int:
        """負債の経過日数"""
        return (datetime.now() - self.discovered_date).days

    def __str__(self) -> str:
        status_icon = {
            "open": "[OPEN]", "in_progress": "[WIP]",
            "resolved": "[DONE]", "wont_fix": "[SKIP]"
        }
        return (
            f"{status_icon.get(self.status, '[?]')} "
            f"[{self.severity.name}] {self.title} "
            f"(Priority: {self.priority_score:.1f}, ROI: {self.roi:.1f})"
        )
```

---

## 3. 負債の可視化と定量化

### 3.1 メトリクス収集

```python
"""技術的負債メトリクス収集フレームワーク"""
import subprocess
import json
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DebtMetrics:
    """収集された負債メトリクス"""
    # コード品質
    avg_complexity: float = 0.0
    max_complexity: float = 0.0
    high_complexity_count: int = 0       # CC > 10 の関数数
    very_high_complexity_count: int = 0  # CC > 20 の関数数

    # 重複
    duplication_percentage: float = 0.0
    duplicate_blocks: int = 0

    # 依存関係
    outdated_dependencies: int = 0
    critical_updates: int = 0            # メジャーバージョン遅れ
    known_vulnerabilities: int = 0

    # テスト
    test_coverage: float = 0.0
    flaky_test_count: int = 0
    test_execution_time_sec: float = 0.0

    # 保守性
    todo_count: int = 0                  # TODO/FIXME/HACK
    large_files_count: int = 0           # 500行超のファイル数
    large_functions_count: int = 0       # 50行超の関数数

    # 全体スコア
    timestamp: str = ""

    @property
    def debt_score(self) -> int:
        """技術的負債スコア (0-100, 低いほど健全)

        各カテゴリのスコアを重み付きで集計:
        - コード品質: 30%
        - テスト: 25%
        - 依存関係: 20%
        - 保守性: 25%
        """
        # コード品質スコア (0-30)
        complexity_score = min(self.avg_complexity / 15 * 30, 30)

        # テストスコア (0-25)
        coverage_penalty = max(0, (80 - self.test_coverage)) / 80 * 25

        # 依存関係スコア (0-20)
        dep_score = min((self.outdated_dependencies + self.known_vulnerabilities * 3) / 20 * 20, 20)

        # 保守性スコア (0-25)
        maintainability = min(
            (self.todo_count / 50 + self.large_files_count / 10 +
             self.duplication_percentage / 10) / 3 * 25, 25
        )

        return int(complexity_score + coverage_penalty + dep_score + maintainability)

    @property
    def health_status(self) -> str:
        score = self.debt_score
        if score < 25:
            return "HEALTHY"
        elif score < 50:
            return "CAUTION"
        elif score < 75:
            return "WARNING"
        else:
            return "CRITICAL"


def collect_debt_metrics(repo_path: str) -> DebtMetrics:
    """リポジトリの技術的負債指標を収集"""
    metrics = DebtMetrics(timestamp=datetime.now().isoformat())
    repo = Path(repo_path)

    # 1. コード複雑度 (Cyclomatic Complexity) -- radon
    try:
        result = subprocess.run(
            ["radon", "cc", str(repo / "src"), "-a", "-j"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            cc_data = json.loads(result.stdout)
            # radon の出力形式に応じてパース
            complexities = []
            for filepath, functions in cc_data.items():
                if isinstance(functions, list):
                    for func in functions:
                        complexities.append(func.get("complexity", 0))

            if complexities:
                metrics.avg_complexity = sum(complexities) / len(complexities)
                metrics.max_complexity = max(complexities)
                metrics.high_complexity_count = sum(1 for c in complexities if c > 10)
                metrics.very_high_complexity_count = sum(1 for c in complexities if c > 20)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass

    # 2. コード重複率 -- jscpd
    try:
        result = subprocess.run(
            ["jscpd", str(repo / "src"), "--reporters", "json",
             "--min-lines", "5", "--output", "/tmp/jscpd-report"],
            capture_output=True, text=True, timeout=120
        )
        report_path = Path("/tmp/jscpd-report/jscpd-report.json")
        if report_path.exists():
            report = json.loads(report_path.read_text())
            stats = report.get("statistics", {}).get("total", {})
            metrics.duplication_percentage = stats.get("percentage", 0)
            metrics.duplicate_blocks = stats.get("duplicates", 0)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # 3. 依存関係の古さ -- pip
    try:
        result = subprocess.run(
            ["pip", "list", "--outdated", "--format=json"],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            outdated = json.loads(result.stdout)
            metrics.outdated_dependencies = len(outdated)
            metrics.critical_updates = sum(
                1 for d in outdated
                if _is_major_version_behind(d.get("version", ""), d.get("latest_version", ""))
            )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # 4. セキュリティ脆弱性 -- safety
    try:
        result = subprocess.run(
            ["safety", "check", "--json"],
            capture_output=True, text=True, timeout=60
        )
        if result.stdout:
            vulns = json.loads(result.stdout)
            metrics.known_vulnerabilities = len(vulns) if isinstance(vulns, list) else 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # 5. TODO/FIXME/HACK の数
    try:
        result = subprocess.run(
            ["grep", "-r", "-c", "-E", "TODO|FIXME|HACK|XXX", str(repo / "src")],
            capture_output=True, text=True, timeout=30
        )
        metrics.todo_count = sum(
            int(line.split(":")[-1])
            for line in result.stdout.strip().split("\n")
            if line and ":" in line
        )
    except (subprocess.TimeoutExpired, ValueError):
        pass

    # 6. 大きなファイル/関数の検出
    try:
        for py_file in (repo / "src").rglob("*.py"):
            lines = py_file.read_text().split("\n")
            if len(lines) > 500:
                metrics.large_files_count += 1
    except (FileNotFoundError, PermissionError):
        pass

    return metrics


def _is_major_version_behind(current: str, latest: str) -> bool:
    """メジャーバージョンが異なるかチェック"""
    try:
        current_major = int(current.split(".")[0])
        latest_major = int(latest.split(".")[0])
        return latest_major > current_major
    except (ValueError, IndexError):
        return False
```

### 3.2 ダッシュボード表示

```python
"""技術的負債ダッシュボード"""

def print_debt_dashboard(metrics: DebtMetrics) -> None:
    """テキストベースの負債ダッシュボードを表示"""

    def status_icon(value: float, good: float, warn: float, higher_is_better: bool = False) -> str:
        if higher_is_better:
            if value >= good: return "[OK]"
            elif value >= warn: return "[WARN]"
            else: return "[CRIT]"
        else:
            if value <= good: return "[OK]"
            elif value <= warn: return "[WARN]"
            else: return "[CRIT]"

    width = 64
    print("=" * width)
    print("  技術的負債ダッシュボード".center(width))
    print(f"  {metrics.timestamp}".center(width))
    print("=" * width)

    print("\n  [コード品質]")
    print(f"    平均複雑度 (CC):     {metrics.avg_complexity:6.1f}  "
          f"{status_icon(metrics.avg_complexity, 5, 10)}")
    print(f"    最大複雑度 (CC):     {metrics.max_complexity:6.1f}  "
          f"{status_icon(metrics.max_complexity, 15, 25)}")
    print(f"    高複雑度関数 (>10):  {metrics.high_complexity_count:6d}  件")
    print(f"    超高複雑度 (>20):    {metrics.very_high_complexity_count:6d}  件")

    print("\n  [コード重複]")
    print(f"    重複率:              {metrics.duplication_percentage:6.1f}% "
          f"{status_icon(metrics.duplication_percentage, 3, 10)}")
    print(f"    重複ブロック数:      {metrics.duplicate_blocks:6d}  件")

    print("\n  [テスト]")
    print(f"    カバレッジ:          {metrics.test_coverage:6.1f}% "
          f"{status_icon(metrics.test_coverage, 80, 60, higher_is_better=True)}")
    print(f"    Flaky テスト:        {metrics.flaky_test_count:6d}  件")
    print(f"    実行時間:            {metrics.test_execution_time_sec:6.1f}  秒")

    print("\n  [依存関係]")
    print(f"    古い依存:            {metrics.outdated_dependencies:6d}  件")
    print(f"    メジャー遅れ:        {metrics.critical_updates:6d}  件")
    print(f"    既知脆弱性:          {metrics.known_vulnerabilities:6d}  件 "
          f"{status_icon(metrics.known_vulnerabilities, 0, 3)}")

    print("\n  [保守性]")
    print(f"    TODO/FIXME/HACK:     {metrics.todo_count:6d}  件")
    print(f"    巨大ファイル (>500L):{metrics.large_files_count:6d}  件")

    print("\n" + "-" * width)
    score = metrics.debt_score
    health = metrics.health_status
    bar_length = 40
    filled = int(score / 100 * bar_length)
    bar = "#" * filled + "-" * (bar_length - filled)

    print(f"  負債スコア: [{bar}] {score}/100")
    print(f"  健全性:     {health}")

    if health == "HEALTHY":
        print("  --> 健全な状態。現在の品質維持プラクティスを継続")
    elif health == "CAUTION":
        print("  --> 要注意。計画的な返済を開始し、悪化を防止")
    elif health == "WARNING":
        print("  --> 警告。開発速度に影響が出始めている。早急な対策を推奨")
    else:
        print("  --> 危険。開発速度に深刻な影響。負債スプリントの実施を推奨")

    print("=" * width)
```

### 3.3 ホットスポット分析

ホットスポットとは「変更頻度が高く、かつ複雑度が高い」コードのことである。Adam Tornhill の Code as a Crime Scene のアプローチに基づく:

```python
"""ホットスポット分析: 変更頻度 x 複雑度でリファクタリング優先箇所を特定"""
import subprocess
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Hotspot:
    """変更頻度と複雑度のホットスポット"""
    filepath: str
    change_count: int       # git log での変更回数
    complexity: float       # 平均循環的複雑度
    lines_of_code: int      # 行数
    bug_fix_count: int      # バグ修正での変更回数

    @property
    def hotspot_score(self) -> float:
        """ホットスポットスコア

        高い変更頻度 × 高い複雑度 = 高リスク
        バグ修正が多いファイルはさらに重み付け
        """
        bug_weight = 1.0 + (self.bug_fix_count / max(self.change_count, 1))
        return self.change_count * self.complexity * bug_weight

    @property
    def risk_level(self) -> str:
        score = self.hotspot_score
        if score > 500: return "CRITICAL"
        elif score > 200: return "HIGH"
        elif score > 100: return "MEDIUM"
        else: return "LOW"


def analyze_hotspots(
    repo_path: str,
    since: str = "6 months ago",
    top_n: int = 20
) -> list[Hotspot]:
    """Git履歴と複雑度からホットスポットを分析"""

    # 1. 変更頻度の収集 (git log)
    result = subprocess.run(
        ["git", "log", "--since", since, "--name-only",
         "--pretty=format:", "--diff-filter=M"],
        capture_output=True, text=True, cwd=repo_path
    )
    file_changes = Counter(
        line.strip() for line in result.stdout.split("\n")
        if line.strip() and line.strip().endswith(".py")
    )

    # 2. バグ修正での変更回数
    result = subprocess.run(
        ["git", "log", "--since", since, "--name-only",
         "--pretty=format:", "--grep=fix", "--grep=bug", "-i"],
        capture_output=True, text=True, cwd=repo_path
    )
    bug_changes = Counter(
        line.strip() for line in result.stdout.split("\n")
        if line.strip() and line.strip().endswith(".py")
    )

    # 3. 複雑度の収集 (radon)
    hotspots = []
    for filepath, change_count in file_changes.most_common(top_n * 2):
        full_path = Path(repo_path) / filepath
        if not full_path.exists():
            continue

        try:
            result = subprocess.run(
                ["radon", "cc", str(full_path), "-a", "-j"],
                capture_output=True, text=True, timeout=10
            )
            cc_data = json.loads(result.stdout)
            complexities = []
            for fpath, funcs in cc_data.items():
                if isinstance(funcs, list):
                    complexities.extend(f.get("complexity", 0) for f in funcs)

            avg_cc = sum(complexities) / len(complexities) if complexities else 1.0
            loc = len(full_path.read_text().split("\n"))

            hotspots.append(Hotspot(
                filepath=filepath,
                change_count=change_count,
                complexity=avg_cc,
                lines_of_code=loc,
                bug_fix_count=bug_changes.get(filepath, 0),
            ))
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
            continue

    # スコア順にソート
    hotspots.sort(key=lambda h: h.hotspot_score, reverse=True)
    return hotspots[:top_n]


def print_hotspot_report(hotspots: list[Hotspot]) -> None:
    """ホットスポットレポートを表示"""
    print("=" * 80)
    print("  ホットスポット分析レポート")
    print("=" * 80)
    print(f"  {'Rank':<5} {'File':<35} {'Changes':<8} {'CC':<6} {'Bugs':<5} {'Risk':<10}")
    print("-" * 80)

    for i, h in enumerate(hotspots, 1):
        short_path = h.filepath if len(h.filepath) < 33 else "..." + h.filepath[-30:]
        print(
            f"  {i:<5} {short_path:<35} {h.change_count:<8} "
            f"{h.complexity:<6.1f} {h.bug_fix_count:<5} {h.risk_level:<10}"
        )

    print("-" * 80)
    critical = sum(1 for h in hotspots if h.risk_level == "CRITICAL")
    high = sum(1 for h in hotspots if h.risk_level == "HIGH")
    print(f"  CRITICAL: {critical} ファイル  /  HIGH: {high} ファイル")
    print("=" * 80)
```

```
ホットスポット可視化 (変更頻度 vs 複雑度)

  複雑度 (CC)
  高  |                                 [payment.py]  ← 最優先
      |                        [order.py]
      |              [user_service.py]
      |
  中  |     [auth.py]
      |                   [email.py]
      |
      |  [config.py]                    [report.py]
  低  |     [utils.py]      [models.py]
      |
      +------------------------------------------------
      低            中              高
                    変更頻度

  右上 = ホットスポット（頻繁に変更 + 複雑 = 高リスク）
  左下 = 安全地帯（変更少ない + シンプル = 低リスク）
  右下 = 要監視（頻繁に変更だが複雑度は低い）
  左上 = 潜在リスク（複雑だが変更は少ない）
```

### 3.4 コスト算出テンプレート

```python
"""技術的負債のビジネスコスト算出"""
from dataclasses import dataclass


@dataclass
class DebtCostEstimation:
    """技術的負債のコスト見積もり"""

    # チーム情報
    team_size: int = 8
    developer_hourly_rate: int = 5000  # 円/時間
    working_hours_per_week: int = 40
    working_weeks_per_year: int = 50

    # 負債によるオーバーヘッド (時間/週/人)
    bug_fix_overhead: float = 1.0        # バグ修正の追加時間
    feature_dev_overhead: float = 1.5    # 機能追加の複雑性コスト
    onboarding_overhead: float = 0.5     # オンボーディングコスト（チーム平均）
    manual_process_overhead: float = 0.75 # 手動テスト・デプロイ
    incident_overhead: float = 0.25      # インシデント対応の追加時間
    context_switching: float = 0.5       # 技術的問題によるコンテキストスイッチ

    @property
    def weekly_overhead_per_person(self) -> float:
        """1人あたり週間オーバーヘッド（時間）"""
        return (
            self.bug_fix_overhead +
            self.feature_dev_overhead +
            self.onboarding_overhead +
            self.manual_process_overhead +
            self.incident_overhead +
            self.context_switching
        )

    @property
    def weekly_overhead_total(self) -> float:
        """チーム全体の週間オーバーヘッド（時間）"""
        return self.weekly_overhead_per_person * self.team_size

    @property
    def productivity_loss_percentage(self) -> float:
        """生産性損失率（%）"""
        return (self.weekly_overhead_per_person / self.working_hours_per_week) * 100

    @property
    def annual_cost(self) -> int:
        """年間コスト（円）"""
        return int(
            self.weekly_overhead_total *
            self.developer_hourly_rate *
            self.working_weeks_per_year
        )

    def print_report(self) -> None:
        """コストレポートを出力"""
        print("=" * 60)
        print("  技術的負債コストレポート")
        print("=" * 60)
        print(f"\n  チーム構成: {self.team_size}名")
        print(f"  時間単価:  {self.developer_hourly_rate:,}円/時間")

        print(f"\n  --- 週間オーバーヘッド (1人あたり) ---")
        items = [
            ("バグ修正の追加時間", self.bug_fix_overhead),
            ("機能追加の複雑性コスト", self.feature_dev_overhead),
            ("オンボーディングコスト", self.onboarding_overhead),
            ("手動プロセス", self.manual_process_overhead),
            ("インシデント対応", self.incident_overhead),
            ("コンテキストスイッチ", self.context_switching),
        ]
        for name, hours in items:
            print(f"    {name:<24s} {hours:5.2f} 時間/週")

        print(f"    {'─' * 36}")
        print(f"    {'合計':<24s} {self.weekly_overhead_per_person:5.2f} 時間/週/人")

        print(f"\n  --- 影響サマリ ---")
        print(f"    チーム週間オーバーヘッド: {self.weekly_overhead_total:,.0f} 時間/週")
        print(f"    生産性損失率:            {self.productivity_loss_percentage:.1f}%")
        print(f"    週間コスト:              {int(self.weekly_overhead_total * self.developer_hourly_rate):>12,} 円")
        print(f"    月間コスト:              {int(self.weekly_overhead_total * self.developer_hourly_rate * 4):>12,} 円")
        print(f"    年間コスト:              {self.annual_cost:>12,} 円")

        print(f"\n  --- 投資対効果 (例) ---")
        investment = self.annual_cost * 0.3  # 年間コストの30%を投資
        savings = self.annual_cost * 0.5     # 50%の改善を見込む
        print(f"    投資額 (30%の工数投入):  {int(investment):>12,} 円")
        print(f"    期待削減額 (50%改善):    {int(savings):>12,} 円")
        print(f"    純利益:                  {int(savings - investment):>12,} 円")
        print(f"    ROI:                     {(savings - investment) / investment * 100:>11.0f}%")
        print("=" * 60)


# 使用例
cost = DebtCostEstimation(
    team_size=8,
    developer_hourly_rate=5000,
    bug_fix_overhead=1.0,
    feature_dev_overhead=1.5,
    onboarding_overhead=0.5,
    manual_process_overhead=0.75,
    incident_overhead=0.25,
    context_switching=0.5,
)
cost.print_report()

# 出力例:
# チーム週間オーバーヘッド: 36.0 時間/週
# 生産性損失率:            11.3%
# 年間コスト:          9,000,000 円
# ROI:                          67%
```

---

## 4. 経営層への説明技法

### 4.1 金融メタファーの活用

経営層やステークホルダーに技術的負債を説明する際は、金融の借金メタファーを一貫して使う:

```
経営層向け説明テンプレート

┌─────────────────────────────────────────────────────┐
│  「技術的負債」= ソフトウェアの住宅ローン             │
│                                                     │
│  ● 元本: 品質が低いコード（建物の欠陥）              │
│  ● 利子: 開発速度の低下（毎月の追加修繕費）          │
│  ● 破産: 新機能が追加できない状態（建て直し必要）    │
│                                                     │
│  現状:                                              │
│    年間利子 = 約 900万円                             │
│    (開発者 8名 × 4.5時間/週 × 50週 × 5,000円/時)    │
│                                                     │
│  提案:                                              │
│    投資額 = 270万円 (3ヶ月 × 開発時間の20%)         │
│    期待効果 = 利子を50%削減 → 年450万円のコスト削減  │
│    回収期間 = 約7ヶ月                                │
│                                                     │
│  放置した場合:                                       │
│    利子は複利で年20-30%増加                          │
│    2年後の年間利子 = 約1,400万円                     │
│    新機能の開発速度は現在の60%に低下                  │
└─────────────────────────────────────────────────────┘
```

### 4.2 可視化グラフ

```
  開発速度の推移 (新機能に使える時間の割合)

  100% |***
       | * **
   80% |    * *
       |      * *
   60% |        * *
       |          * *      ← 負債を放置した場合
   40% |            * *
       |              * *
   20% |                **   ← 「利子の支払いだけで精一杯」
       |                  **
    0% +----+----+----+----+----+----+
       Y0   Y1   Y2   Y3   Y4   Y5

  vs.

  100% |***
       | * **
   80% |    * *
       |      * **          ← 負債を計画的に返済
   70% |         * *  * * * * * *
       |           **
   60% |                    ← 一時的に速度低下（返済投資）
       |
       +----+----+----+----+----+----+
       Y0   Y1   Y2   Y3   Y4   Y5
```

### 4.3 エレベーターピッチ

```
30秒で伝える技術的負債の説明:

「私たちは毎月、目に見えない『利子』を払っています。
 開発チームの 11% の時間が、過去の品質妥協のせいで
 本来不要な作業に費やされています。

 年間で約900万円に相当します。

 開発時間の20%を3ヶ月間投資することで、
 この利子を半減できます。

 7ヶ月で投資を回収し、
 その後は年間450万円のコスト削減が持続します。」
```

---

## 5. 返済戦略

### 5.1 戦略の全体像

```
返済戦略のピラミッド

                 ┌───────────┐
                 │ Strangler │  4-6ヶ月
                 │   Fig     │  根本的な置換
                 ├───────────┤
                 │  技術的    │  四半期ごと
                 │  負債      │  集中的な返済
                 │  スプリント │
                 ├───────────┤
                 │  20% ルール │  毎スプリント
                 │  (Sprint内  │  計画的な返済
                 │   の20%)    │
                 ├───────────┤
                 │   ボーイスカウト     │  毎日
                 │   ルール            │  小さな改善
                 └─────────────────────┘

  下から上に向かって:
  - コストが増加
  - 効果が増加
  - リスクが増加
  - 頻度が減少
```

### 5.2 ボーイスカウトルール

```python
"""ボーイスカウトルール: コードを見つけた時より少しきれいにして去る"""

# === 例1: 変数名の改善 ===

# Before (見つけた時の状態)
def calc(d, r):
    return d * r * 0.01

# After (少しきれいにして去る)
def calculate_discount(price: float, rate_percent: float) -> float:
    """割引額を計算する"""
    return price * rate_percent * 0.01


# === 例2: マジックナンバーの除去 ===

# Before
if user.login_attempts > 5:
    lock_account(user)

# After
MAX_LOGIN_ATTEMPTS = 5

if user.login_attempts > MAX_LOGIN_ATTEMPTS:
    lock_account(user)


# === 例3: 不要なコメントの削除と型ヒント追加 ===

# Before
# ユーザーを取得する
def get_user(id):
    # DBから取得
    user = db.query(User).filter(User.id == id).first()
    return user  # ユーザーを返す

# After
def get_user(user_id: int) -> User | None:
    return db.query(User).filter(User.id == user_id).first()
```

ボーイスカウトルールのガイドライン:

```
ボーイスカウトルールの適用範囲

  OK (適用すべき)              NG (スコープ外)
  ┌──────────────────┐        ┌──────────────────┐
  │ ● 変数名の改善     │        │ ● アーキテクチャ   │
  │ ● 型ヒント追加     │        │   の変更           │
  │ ● 不要コメント削除  │        │ ● 大規模な         │
  │ ● マジックナンバー  │        │   リファクタリング  │
  │   の定数化         │        │ ● API の変更       │
  │ ● import の整理    │        │ ● データモデルの    │
  │ ● 小さな関数抽出   │        │   変更             │
  └──────────────────┘        └──────────────────┘

  判断基準: 5分以内で完了し、
  動作を変えないリファクタリングか？
  → Yes: ボーイスカウトルール適用
  → No:  負債バックログに起票
```

### 5.3 20%ルール

```python
"""20%ルール: 各スプリントの20%を技術的負債返済に充てる"""

@dataclass
class SprintCapacity:
    """スプリントのキャパシティ管理"""
    total_story_points: int
    team_size: int
    sprint_days: int = 10  # 2週間スプリント

    @property
    def feature_capacity(self) -> int:
        """新機能に使えるポイント (80%)"""
        return int(self.total_story_points * 0.80)

    @property
    def debt_capacity(self) -> int:
        """負債返済に使えるポイント (20%)"""
        return int(self.total_story_points * 0.20)

    def plan_sprint(
        self,
        feature_backlog: list[dict],
        debt_backlog: list[TechnicalDebtItem]
    ) -> dict:
        """スプリント計画"""
        # 新機能を80%枠内で選択
        selected_features = []
        remaining_feature_points = self.feature_capacity
        for feature in feature_backlog:
            if feature["points"] <= remaining_feature_points:
                selected_features.append(feature)
                remaining_feature_points -= feature["points"]

        # 負債を20%枠内で選択（優先度順）
        selected_debt = []
        remaining_debt_points = self.debt_capacity
        sorted_debt = sorted(debt_backlog, key=lambda d: d.priority_score, reverse=True)
        for debt in sorted_debt:
            estimated_points = int(debt.fix_effort_days * 2)  # 1日 ≈ 2SP
            if estimated_points <= remaining_debt_points:
                selected_debt.append(debt)
                remaining_debt_points -= estimated_points

        return {
            "features": selected_features,
            "debt_items": selected_debt,
            "feature_points_used": self.feature_capacity - remaining_feature_points,
            "debt_points_used": self.debt_capacity - remaining_debt_points,
        }
```

```
20%ルールのスプリント配分

  Sprint 1        Sprint 2        Sprint 3
  ┌────────────┐  ┌────────────┐  ┌────────────┐
  │ Feature A  │  │ Feature C  │  │ Feature E  │
  │ Feature B  │  │ Feature D  │  │ Feature F  │
  │            │  │            │  │            │
  │   (80%)    │  │   (80%)    │  │   (80%)    │
  ├────────────┤  ├────────────┤  ├────────────┤
  │ Debt: テスト│  │ Debt: 依存 │  │ Debt: API  │
  │ カバレッジ  │  │ 更新       │  │ 統合       │
  │   (20%)    │  │   (20%)    │  │   (20%)    │
  └────────────┘  └────────────┘  └────────────┘

  例外: 緊急リリース時
  Sprint N (緊急)   Sprint N+1 (帳尻合わせ)
  ┌────────────┐    ┌────────────┐
  │ 緊急機能    │    │ Feature G  │
  │            │    │            │
  │  (100%)    │    │   (60%)    │
  │            │    ├────────────┤
  │            │    │ Debt返済   │
  │ 負債0%     │    │   (40%)    │
  └────────────┘    └────────────┘
```

### 5.4 技術的負債スプリント

```python
"""技術的負債スプリント: 四半期に1回の集中返済"""

@dataclass
class DebtSprint:
    """技術的負債スプリントの計画と実行"""
    quarter: str              # "2024-Q2"
    duration_days: int = 10   # 2週間
    team_size: int = 8

    def plan(self, debt_backlog: list[TechnicalDebtItem]) -> dict:
        """負債スプリントの計画

        選定基準:
        1. ROI が最も高い負債を優先
        2. 相互依存する負債をグループ化
        3. チームのスキルセットと負債の種類をマッチング
        """
        # ROI 順にソート
        sorted_debt = sorted(debt_backlog, key=lambda d: d.roi, reverse=True)

        # キャパシティ計算 (チーム × 日数)
        total_capacity_days = self.team_size * self.duration_days

        selected = []
        remaining_capacity = total_capacity_days

        for debt in sorted_debt:
            if debt.fix_effort_days <= remaining_capacity:
                selected.append(debt)
                remaining_capacity -= debt.fix_effort_days

        # 期待効果の算出
        total_annual_savings = sum(d.annual_interest_cost for d in selected)
        total_investment = sum(d.fix_effort_days for d in selected)

        return {
            "quarter": self.quarter,
            "selected_items": selected,
            "total_investment_days": total_investment,
            "total_annual_savings_days": total_annual_savings,
            "overall_roi": total_annual_savings / max(total_investment, 1),
            "remaining_capacity_days": remaining_capacity,
        }

    def execute_checklist(self) -> list[str]:
        """負債スプリント実施チェックリスト"""
        return [
            "[ ] Sprint Goal を明確に定義（測定可能な指標で）",
            "[ ] Before メトリクスを収集（複雑度、カバレッジ、重複率）",
            "[ ] 各負債アイテムに担当者をアサイン",
            "[ ] 各変更に対するテストを先に書く",
            "[ ] 小さなPRに分割（1PR = 1改善）",
            "[ ] デイリースタンドアップで進捗共有",
            "[ ] After メトリクスを収集",
            "[ ] Before/After の比較レポートを作成",
            "[ ] 振り返り（レトロスペクティブ）を実施",
            "[ ] 残った負債を次回に繰り越し",
        ]
```

### 5.5 負債バックログ管理

```python
"""技術的負債バックログの管理"""
from datetime import datetime
from typing import Optional


class TechnicalDebtBacklog:
    """技術的負債バックログの管理クラス"""

    def __init__(self):
        self._items: list[TechnicalDebtItem] = []

    def add(self, item: TechnicalDebtItem) -> None:
        """負債アイテムを追加"""
        self._items.append(item)

    def get_by_priority(self, top_n: int = 10) -> list[TechnicalDebtItem]:
        """優先度順に取得"""
        open_items = [i for i in self._items if i.status == "open"]
        return sorted(open_items, key=lambda i: i.priority_score, reverse=True)[:top_n]

    def get_by_roi(self, top_n: int = 10) -> list[TechnicalDebtItem]:
        """ROI順に取得（Quick Win 発見用）"""
        open_items = [i for i in self._items if i.status == "open"]
        return sorted(open_items, key=lambda i: i.roi, reverse=True)[:top_n]

    def get_quick_wins(self, max_effort_days: float = 1.0) -> list[TechnicalDebtItem]:
        """Quick Win: 少ない工数で高い効果が得られる負債"""
        return [
            i for i in self._items
            if i.status == "open" and i.fix_effort_days <= max_effort_days
        ]

    def get_stale_items(self, days: int = 90) -> list[TechnicalDebtItem]:
        """長期間放置されている負債"""
        return [
            i for i in self._items
            if i.status == "open" and i.debt_age_days > days
        ]

    def summary(self) -> dict:
        """バックログのサマリ"""
        open_items = [i for i in self._items if i.status == "open"]
        resolved = [i for i in self._items if i.status == "resolved"]

        total_effort = sum(i.fix_effort_days for i in open_items)
        total_interest = sum(i.annual_interest_cost for i in open_items)

        by_category = {}
        for item in open_items:
            cat = item.category
            if cat not in by_category:
                by_category[cat] = {"count": 0, "effort": 0.0}
            by_category[cat]["count"] += 1
            by_category[cat]["effort"] += item.fix_effort_days

        by_severity = {}
        for item in open_items:
            sev = item.severity.name
            by_severity[sev] = by_severity.get(sev, 0) + 1

        return {
            "total_open": len(open_items),
            "total_resolved": len(resolved),
            "total_effort_days": total_effort,
            "total_annual_interest_days": total_interest,
            "by_category": by_category,
            "by_severity": by_severity,
            "avg_age_days": (
                sum(i.debt_age_days for i in open_items) / len(open_items)
                if open_items else 0
            ),
        }

    def print_summary(self) -> None:
        """バックログサマリを表示"""
        s = self.summary()
        print("=" * 50)
        print("  技術的負債バックログ サマリ")
        print("=" * 50)
        print(f"  未対応: {s['total_open']} 件  /  解決済: {s['total_resolved']} 件")
        print(f"  合計修正工数: {s['total_effort_days']:.0f} 人日")
        print(f"  年間利子: {s['total_annual_interest_days']:.0f} 人日")
        print(f"  平均経過日数: {s['avg_age_days']:.0f} 日")

        print(f"\n  [カテゴリ別]")
        for cat, data in s["by_category"].items():
            print(f"    {cat:<20s} {data['count']:3d} 件  ({data['effort']:.0f} 人日)")

        print(f"\n  [深刻度別]")
        for sev, count in sorted(s["by_severity"].items()):
            print(f"    {sev:<12s} {count:3d} 件")
        print("=" * 50)
```

---

## 6. CI/CD との統合

### 6.1 GitHub Actions で負債トレンド追跡

```yaml
# .github/workflows/debt-tracking.yml
name: Technical Debt Tracking

on:
  schedule:
    - cron: '0 9 * * 1'  # 毎週月曜9時
  workflow_dispatch:

jobs:
  track-debt:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # 全履歴を取得

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install tools
        run: |
          pip install radon coverage safety
          npm install -g jscpd

      - name: Collect metrics
        run: |
          python scripts/collect_debt_metrics.py \
            --output debt-metrics.json

      - name: Check thresholds
        run: |
          python scripts/check_debt_thresholds.py \
            --input debt-metrics.json \
            --max-complexity 8 \
            --min-coverage 80 \
            --max-duplication 5

      - name: Update trend data
        run: |
          python scripts/update_debt_trend.py \
            --input debt-metrics.json \
            --trend-file debt-trend.json

      - name: Post to Slack (if degraded)
        if: failure()
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "技術的負債アラート: 品質メトリクスが閾値を超えました",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "詳細: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}"
                  }
                }
              ]
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

### 6.2 PR での負債チェック

```yaml
# .github/workflows/debt-check-pr.yml
name: Debt Check on PR

on:
  pull_request:
    branches: [main]

jobs:
  debt-impact:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Check complexity of changed files
        run: |
          # 変更されたPythonファイルの複雑度チェック
          git diff --name-only origin/main...HEAD -- '*.py' | while read file; do
            if [ -f "$file" ]; then
              radon cc "$file" -n C -s  # Cランク以上を表示
            fi
          done

      - name: Check for new TODOs
        run: |
          # 新しく追加された TODO/FIXME をチェック
          new_todos=$(git diff origin/main...HEAD | grep '^+' | grep -cE 'TODO|FIXME|HACK|XXX' || true)
          if [ "$new_todos" -gt 0 ]; then
            echo "::warning::${new_todos} 件の新しい TODO/FIXME が追加されています"
          fi

      - name: Coverage check
        run: |
          pytest --cov=src --cov-report=term --cov-fail-under=80
```

---

## 7. 比較表

### 7.1 返済戦略比較

| 戦略 | コスト | 効果範囲 | リスク | 効果発現 | 適用場面 | 推奨頻度 |
|------|:------:|:--------:|:------:|:--------:|---------|:--------:|
| ボーイスカウトルール | 最小 | 局所的 | 最小 | 即時 | 日常の小さな改善 | 毎日 |
| 20%ルール | 低 | 中程度 | 低 | 2-4週 | 計画的な品質向上 | 毎スプリント |
| 技術的負債スプリント | 中 | 広範 | 中 | 2-4週 | 蓄積した負債の集中返済 | 四半期 |
| Strangler Fig | 高 | 根本的 | 中-高 | 3-6ヶ月 | レガシーシステムの段階置換 | 年1-2回 |
| フルリライト | 最高 | 根本的 | 最高 | 6-18ヶ月 | 最終手段（非推奨） | 極稀 |

### 7.2 健全性指標

| 指標 | 健全 (Green) | 要注意 (Yellow) | 危険 (Red) | 測定ツール |
|------|:-----------:|:--------------:|:----------:|-----------|
| テストカバレッジ | > 80% | 50-80% | < 50% | coverage.py, Istanbul |
| コード重複率 | < 3% | 3-10% | > 10% | jscpd, SonarQube |
| 平均複雑度 (CC) | < 5 | 5-10 | > 10 | radon, ESLint |
| 古い依存関係 | < 5% | 5-20% | > 20% | pip list --outdated |
| デプロイ頻度 | 日次以上 | 週次 | 月次以下 | DORA metrics |
| リードタイム | < 1日 | 1-7日 | > 7日 | DORA metrics |
| 既知脆弱性 | 0件 | 1-5件 | > 5件 | Safety, Snyk, Trivy |
| TODO/FIXME 密度 | < 1/1000行 | 1-5/1000行 | > 5/1000行 | grep, SonarQube |

### 7.3 負債の4象限別対策

| 象限 | 例 | 対策 | 予防策 |
|------|-----|------|--------|
| 慎重 x 意図的 | MVP 優先の設計妥協 | 返済計画を負債発生時に起票 | 受入条件に「返済Sprint」を含める |
| 無謀 x 意図的 | テスト省略「時間がない」 | 20%ルールで段階的にテスト追加 | Definition of Done にテスト必須 |
| 慎重 x 無意識的 | 学習後に気づく設計不備 | 定期的なアーキテクチャレビュー | 継続的学習の文化、勉強会 |
| 無謀 x 無意識的 | スキル不足による低品質 | ペアプログラミング、コードレビュー | オンボーディング充実、メンタリング |

---

## 8. 演習問題

### 演習 1: 負債の分類と優先度付け（基礎）

以下の技術的負債アイテムを4象限に分類し、優先度スコアを計算せよ。

```
負債リスト:
A. 「リリース期限に間に合わせるため、入力バリデーションを省略した」
B. 「Reactのクラスコンポーネントで書いたが、今ならHooksで書き直す」
C. 「チームにテスト経験者がおらず、ユニットテストが書かれていない」
D. 「AWS Lambda のランタイムが Python 3.8（EOL済み）のまま」
E. 「DBアクセスをControllerから直接行っている（レイヤー違反）」
F. 「デモ用にハードコードした管理者パスワードがまだコードに残っている」

各アイテムについて:
1. 4象限での分類
2. DebtSeverity の割り当て
3. impact, fix_effort_days, frequency, risk の見積もり
4. priority_score と ROI の計算
5. 推奨する返済戦略
```

**期待される回答例（アイテムA）:**

```
A. 入力バリデーション省略
  象限: 無謀 × 意図的（「時間がない」という理由で品質を犠牲に）
  Severity: CRITICAL (セキュリティリスクを含む)
  impact=5, fix_effort=2.0日, frequency=5, risk=5
  priority_score = (5 × 5 × 5) / 2.0 = 62.5
  ROI = 年間利子25人日 / 2人日 = 12.5
  戦略: 即時対応（セキュリティリスクのため20%ルールを待たない）
```

### 演習 2: コスト試算と経営層プレゼン（応用）

以下のチーム状況から、技術的負債のコストを算出し、経営層向けのプレゼン資料（1ページ）を作成せよ。

```
チーム状況:
- チーム規模: 6名
- 開発者時間単価: 6,000円/時間
- テストカバレッジ: 45%
- 平均複雑度: 12.3
- コード重複率: 8%
- 古い依存関係: 15件
- 手動デプロイ（所要時間: 2時間/回、頻度: 週2回）
- バグ修正に平均8時間（業界平均の2倍）
- 新メンバーのオンボーディング: 6週間（業界平均の2倍）

課題:
1. DebtCostEstimation を使ってコストを算出
2. 30秒エレベーターピッチを作成
3. 3段階の返済計画（3ヶ月 / 6ヶ月 / 12ヶ月）を提案
4. 各段階の投資額と期待ROIを算出
```

**期待される出力の骨子:**

```
年間コスト ≈ 1,560万円
生産性損失率 ≈ 16.7%
3ヶ月計画: CI/CD + テスト基盤 → 投資 300万円、年間削減 500万円
6ヶ月計画: + リファクタリング → 追加投資 200万円、追加削減 300万円
12ヶ月計画: + アーキテクチャ改善 → 追加投資 400万円、追加削減 400万円
```

### 演習 3: ホットスポット分析と返済計画（発展）

実際のGitリポジトリに対してホットスポット分析を実施し、四半期の負債返済計画を策定せよ。

```
手順:
1. 自分のプロジェクト（またはOSSリポジトリ）でホットスポット分析を実行
2. 上位10ファイルのホットスポットを特定
3. 各ファイルの負債種類を分類（コード/アーキテクチャ/テスト/インフラ）
4. TechnicalDebtItem として登録
5. 四半期の負債スプリント計画を作成:
   - Sprint 1 (Week 1-2): Quick Win の返済
   - Sprint 2 (Week 3-4): High Priority の返済
   - Sprint 3-6: 20%ルールでの継続返済
6. Before/After のメトリクス目標を設定
7. 経営層への報告資料を作成

評価基準:
- ホットスポット分析の正確性
- 優先度付けの合理性
- 計画の実現可能性
- メトリクスの具体性
- 経営層向け説明の説得力
```

---

## 9. アンチパターン

### アンチパターン 1: 「後で直す」の無限延期

```
NG パターン:
  Sprint 1: 「時間がないから後で直す」→ TODO コメント追加
  Sprint 2: 「今回も時間がない」→ TODO が増殖
  Sprint 3: 「新機能の方が優先」→ TODO の山
  Sprint 6: 「もう誰も全体像がわからない」→ 手遅れ
  Sprint 10: 「フルリライトするしかない」→ 莫大なコスト

  原因: 負債が「見えない」ため、優先度が常に新機能の下になる

OK パターン:
  Sprint 1: TODO コメント → 即座に負債バックログに起票
  Sprint 2: 20%枠で最も ROI の高い負債を返済
  Sprint 3: 負債ダッシュボードでスコアを確認
  Sprint 4: 負債スコアが改善 → チームのモチベーション向上

  原因: 負債を「可視化」し、返済を「計画」に組み込む
```

### アンチパターン 2: 全負債の同時返済

```
NG パターン:
  「今月は技術的負債一掃月間だ！」
  → チーム全員が別々の負債に取り組む
  → 各改善が中途半端に終わる
  → 新機能開発が1ヶ月完全停止
  → ビジネス側からの信頼を失う
  → 「次は技術的負債月間はやらない」と言われる

OK パターン:
  「今四半期は上位3件の負債を完了させる」
  Sprint N:   テストカバレッジ 60% → 80% (チーム全員で集中)
  Sprint N+1: CI/CD パイプライン構築 (2名がリード)
  Sprint N+2: 手動デプロイの自動化 (インフラ担当)
  → 各スプリントで具体的な成果を出す
  → ビジネス側にも改善効果を報告
```

### アンチパターン 3: 負債スコアのゲーム化

```
NG パターン:
  「SonarQubeのスコアを A にすることが目標」
  → 意味のない微細な修正に時間を費やす
  → 本質的な構造問題は放置（スコアに反映されにくい）
  → スコアは改善したが、開発速度は変わらない

  = Goodhart's Law: メトリクスが目標になると指標としての価値を失う

OK パターン:
  「開発者体験（Developer Experience）の改善が目標」
  → メトリクスは「手段」であり「目的」ではない
  → 「ビルド時間が5分から1分に短縮」= 実感のある改善
  → 「新メンバーのオンボーディングが6週→3週」= ビジネス価値のある改善
  → メトリクスは改善の「検証」に使う
```

### アンチパターン 4: 負債の発生を全面禁止

```
NG パターン:
  「技術的負債は一切作らないルール！」
  → すべてのコードが「完璧」でないとマージできない
  → 開発速度が極端に低下
  → ビジネスチャンスを逃す
  → 開発者のモチベーション低下

OK パターン:
  「意図的な負債は戦略的に許可する」
  1. MVP リリースのため、一時的に設計を妥協 → OK（計画的）
  2. 負債発生時にバックログ起票を義務化
  3. 返済計画（いつ、誰が、どうやって）を記載
  4. 3スプリント以内に返済しない場合、自動エスカレーション

  ポイント: 「負債ゼロ」を目指すのではなく
           「負債を管理可能な範囲内に保つ」ことが目標
```

---

## 10. FAQ

### Q1. 技術的負債を経営層にどう説明する？

**A.** 金融メタファーを一貫して使う。「技術的負債は住宅ローンのようなもの。毎月の利子（追加開発コスト）を払い続けている。現在、年間推定900万円の利子を払っている。270万円を投資して元本を返済すれば、翌年から年450万円のコスト削減になる」。重要なのは、(1) 具体的な数値（開発速度の低下率、バグ修正にかかる時間）を示すこと、(2) 放置した場合のコスト増加（複利）を示すこと、(3) 投資対効果（ROI）として提案すること。セクション4の「エレベーターピッチ」テンプレートを活用するとよい。

### Q2. 技術的負債をゼロにすべきか？

**A.** ゼロにする必要はなく、ゼロにすべきでもない。住宅ローン同様、「適切な量の負債」は戦略的に有用である。重要なのは3つの条件: (1) 負債が可視化されていること（ダッシュボード、バックログ）、(2) 利子が制御可能な範囲内であること（生産性損失率 < 15%）、(3) 返済計画が存在すること（20%ルール、四半期スプリント）。新規事業の初期フェーズでは意図的に負債を抱えて速度を優先し、PMF（Product Market Fit）が確認できたら計画的に返済するのが合理的である。

### Q3. 技術的負債とビジネス要求のバランスはどう取る？

**A.** 「20%ルール」が最も広く使われているプラクティスである。スプリント容量の20%を常に技術的負債に確保する。これにより新機能開発を80%維持しつつ、負債が複利で膨らむのを防ぐ。緊急のビジネス要求時は一時的に100%を新機能に充てるが、次のスプリントで40%を負債返済に充てて帳尻を合わせる。重要なのは「負債返済はオプションではなく、Definition of Done の一部である」というチーム合意を形成すること。

### Q4. SonarQube などのツールで技術的負債を「日数」で表示しているが、どう解釈すべき？

**A.** SonarQube の「Technical Debt」は主にコードスメルの修正時間の合計であり、アーキテクチャ負債やテスト負債は含まれない。従って、SonarQube の数値は「負債の一部」であり「全体像」ではない。推奨するアプローチは、SonarQube をコード品質の1指標として使いつつ、本ガイドの DebtMetrics のような多面的なメトリクス収集を組み合わせること。また、SonarQube の「日数」は修正工数の見積もりであり、利子（放置コスト）は含まれていない点にも注意が必要。

### Q5. レガシーシステムの負債が膨大すぎてどこから手をつけていいかわからない

**A.** 以下の3ステップで着手する: (1) ホットスポット分析（セクション3.3）で「変更頻度が高く複雑度も高い」ファイルを特定する。上位5-10ファイルに集中する。(2) Quick Win を先に実施する。1日以内で完了し、ROI が高い負債（テスト追加、定数化、命名改善）を片付ける。成功体験がチームのモチベーションを高める。(3) Strangler Fig パターン（[レガシーコード](./02-legacy-code.md)を参照）で段階的にモジュールを置換する。全体を一度に改善しようとしないことが最も重要である。

### Q6. 技術的負債の「利子率」はどう見積もるのが正確か？

**A.** 正確な見積もりは困難だが、以下の proxy メトリクスが有用: (1) 変更リードタイム -- 同規模の変更が以前より何倍時間がかかるか。(2) バグ密度の推移 -- 新規コード1000行あたりのバグ数の増加率。(3) オンボーディング期間の変化 -- 新メンバーが独立して作業できるまでの期間。(4) デプロイ失敗率の推移。これらの指標が悪化傾向にあれば、利子が増加していると判断できる。3-6ヶ月単位でトレンドを追跡し、悪化が見られたら返済を強化する。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 技術的負債の定義 | 品質を犠牲にして発生する将来の追加コスト（Ward Cunningham, 1992） |
| 4象限モデル | 意図的/無意識的 x 慎重/無謀で分類（Martin Fowler） |
| 利子の概念 | 返済しないと複利で膨らみ、開発速度が持続的に低下する |
| 可視化 | メトリクス収集（複雑度・カバレッジ・重複・依存）、ダッシュボード、ホットスポット分析 |
| コスト算出 | 年間利子を金額で定量化し、ROI として返済を提案する |
| ボーイスカウトルール | 5分以内で完了する小さな改善を毎日実施 |
| 20%ルール | スプリント容量の20%を負債返済に常時確保 |
| 負債スプリント | 四半期に1回、ROI 上位の負債を集中返済 |
| 優先度付け | impact x frequency x risk / effort で算出、ROI も考慮 |
| アンチパターン | 無限延期、全負債同時返済、スコアのゲーム化、負債の全面禁止 |

---

## 次に読むべきガイド

- [レガシーコード](./02-legacy-code.md) -- Strangler Fig パターンなど、レガシーコードへの安全な変更技法
- [継続的改善](./04-continuous-improvement.md) -- CI/CD パイプラインと品質メトリクスの自動化
- [コードスメル](./00-code-smells.md) -- 負債の兆候を早期発見するためのスメルカタログ
- [リファクタリング技法](./01-refactoring-techniques.md) -- 負債返済の具体的な手段
- [テスト原則](../01-practices/04-testing-principles.md) -- テスト負債の解消と品質基盤の構築
- [コードレビューチェックリスト](../03-practices-advanced/04-code-review-checklist.md) -- レビューによる負債発生の予防
- [エラーハンドリング](../01-practices/02-error-handling.md) -- 堅牢なエラー処理による障害リスクの低減

---

## 参考文献

1. **Managing Technical Debt** -- Philippe Kruchten, Robert Nord, Ipek Ozkaya (Addison-Wesley, 2019) -- SEI/CMU による技術的負債の学術的・実践的フレームワーク。負債の分類・測定・管理のための体系的アプローチを提供
2. **Refactoring: Improving the Design of Existing Code, 2nd Edition** -- Martin Fowler (Addison-Wesley, 2018) -- コード品質改善のためのリファクタリングカタログ。負債返済の具体的手法として必読
3. **Technical Debt Quadrant** -- Martin Fowler (Blog, 2009) -- https://martinfowler.com/bliki/TechnicalDebtQuadrant.html -- 4象限モデルの原典
4. **Software Design X-Rays** -- Adam Tornhill (Pragmatic Programmers, 2018) -- Git 履歴を使ったホットスポット分析（Code as a Crime Scene）の実践ガイド。変更頻度 x 複雑度によるリファクタリング優先順位付けの手法
5. **Accelerate: The Science of Lean Software and DevOps** -- Nicole Forsgren, Jez Humble, Gene Kim (IT Revolution, 2018) -- DORA メトリクスと組織パフォーマンスの研究結果。技術的負債の放置が開発速度に与える影響のエビデンス
6. **A Mess is not a Technical Debt** -- Robert C. Martin (Blog, 2009) -- Uncle Bob による「乱雑なコード」と「技術的負債」の区別に関する重要な議論
7. **The Financial Implications of Technical Debt** -- Steve McConnell (2007) -- 意図的/非意図的な負債の分類と、負債管理のビジネスフレームワーク
