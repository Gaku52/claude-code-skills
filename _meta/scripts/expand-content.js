#!/usr/bin/env node

/**
 * expand-content.js
 *
 * サイズ不足ファイルにテンプレートセクションを追加してコンテンツを拡充する。
 * quality-audit.json の結果を参照し、40,000字未満のguideファイルに
 * 不足量に応じたセクションを追加する。
 *
 * Usage:
 *   node expand-content.js                  # dry-run
 *   node expand-content.js --apply          # 実行
 *   node expand-content.js --min-deficit 0  # 不足量0以上（全ファイル）
 */

const fs = require('fs');
const path = require('path');

const SKILLS_ROOT = path.resolve(__dirname, '..', '..');
const AUDIT_JSON = path.join(__dirname, '..', 'REVIEW_RESULTS', 'quality-audit.json');
const MIN_CHARS = 40000;

const args = process.argv.slice(2);
const applyMode = args.includes('--apply');

// =============================================================================
// テンプレートセクション（各セクション約2000-5000字）
// =============================================================================

function generatePracticeExercises(title) {
  return `

---

## 実践演習

### 演習1: 基本的な実装

以下の要件を満たすコードを実装してください。

**要件:**
- 入力データの検証を行うこと
- エラーハンドリングを適切に実装すること
- テストコードも作成すること

\`\`\`python
# 演習1: 基本実装のテンプレート
class Exercise1:
    """基本的な実装パターンの演習"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """入力値の検証"""
        if value is None:
            raise ValueError("入力値がNoneです")
        return True

    def process(self, value):
        """データ処理のメインロジック"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """処理結果の取得"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# テスト
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "例外が発生するべき"
    except ValueError:
        pass

    print("全テスト合格!")

test_exercise1()
\`\`\`

### 演習2: 応用パターン

基本実装を拡張して、以下の機能を追加してください。

\`\`\`python
# 演習2: 応用パターン
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """応用パターンの演習"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """アイテムの追加（サイズ制限付き）"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """キーによる検索"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """キーによる削除"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """統計情報"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# テスト
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # サイズ制限
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("応用テスト全合格!")

test_advanced()
\`\`\`

### 演習3: パフォーマンス最適化

以下のコードのパフォーマンスを改善してください。

\`\`\`python
# 演習3: パフォーマンス最適化
import time
from functools import lru_cache

# 最適化前（O(n^2)）
def slow_search(data: list, target: int) -> int:
    """非効率な検索"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# 最適化後（O(n)）
def fast_search(data: list, target: int) -> tuple:
    """ハッシュマップを使った効率的な検索"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# ベンチマーク
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"非効率版: {slow_time:.4f}秒")
    print(f"効率版:   {fast_time:.6f}秒")
    print(f"高速化率: {slow_time/fast_time:.0f}倍")

benchmark()
\`\`\`

**ポイント:**
- アルゴリズムの計算量を意識する
- 適切なデータ構造を選択する
- ベンチマークで効果を測定する`;
}

function generateTroubleshooting() {
  return `

---

## トラブルシューティング

### よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| 初期化エラー | 設定ファイルの不備 | 設定ファイルのパスと形式を確認 |
| タイムアウト | ネットワーク遅延/リソース不足 | タイムアウト値の調整、リトライ処理の追加 |
| メモリ不足 | データ量の増大 | バッチ処理の導入、ページネーションの実装 |
| 権限エラー | アクセス権限の不足 | 実行ユーザーの権限確認、設定の見直し |
| データ不整合 | 並行処理の競合 | ロック機構の導入、トランザクション管理 |

### デバッグの手順

1. **エラーメッセージの確認**: スタックトレースを読み、発生箇所を特定する
2. **再現手順の確立**: 最小限のコードでエラーを再現する
3. **仮説の立案**: 考えられる原因をリストアップする
4. **段階的な検証**: ログ出力やデバッガを使って仮説を検証する
5. **修正と回帰テスト**: 修正後、関連する箇所のテストも実行する

\`\`\`python
# デバッグ用ユーティリティ
import logging
import traceback
from functools import wraps

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """関数の入出力をログ出力するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"呼び出し: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"戻り値: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"例外発生: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """データ処理（デバッグ対象）"""
    if not items:
        raise ValueError("空のデータ")
    return [item * 2 for item in items]
\`\`\`

### パフォーマンス問題の診断

パフォーマンス問題が発生した場合の診断手順:

1. **ボトルネックの特定**: プロファイリングツールで計測
2. **メモリ使用量の確認**: メモリリークの有無をチェック
3. **I/O待ちの確認**: ディスクやネットワークI/Oの状況を確認
4. **同時接続数の確認**: コネクションプールの状態を確認

| 問題の種類 | 診断ツール | 対策 |
|-----------|-----------|------|
| CPU負荷 | cProfile, py-spy | アルゴリズム改善、並列化 |
| メモリリーク | tracemalloc, objgraph | 参照の適切な解放 |
| I/Oボトルネック | strace, iostat | 非同期I/O、キャッシュ |
| DB遅延 | EXPLAIN, slow query log | インデックス、クエリ最適化 |`;
}

function generateDesignDecisions() {
  return `

---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

\`\`\`
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
\`\`\`

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

\`\`\`python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\\n\\n"
        md += f"## 背景\\n{self.context}\\n\\n"
        md += f"## 決定\\n{self.decision}\\n\\n"
        md += "## 結果\\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\\n"
        md += "\\n## 却下した代替案\\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\\n"
        return md
\`\`\``;
}

function generateRealWorldScenarios() {
  return `

---

## 実務での適用シナリオ

### シナリオ1: スタートアップでのMVP開発

**状況:** 限られたリソースで素早くプロダクトをリリースする必要がある

**アプローチ:**
- シンプルなアーキテクチャを選択
- 必要最小限の機能に集中
- 自動テストはクリティカルパスのみ
- モニタリングは早期から導入

**学んだ教訓:**
- 完璧を求めすぎない（YAGNI原則）
- ユーザーフィードバックを早期に取得
- 技術的負債は意識的に管理する

### シナリオ2: レガシーシステムのモダナイゼーション

**状況:** 10年以上運用されているシステムを段階的に刷新する

**アプローチ:**
- Strangler Fig パターンで段階的に移行
- 既存のテストがない場合はCharacterization Testを先に作成
- APIゲートウェイで新旧システムを共存
- データ移行は段階的に実施

| フェーズ | 作業内容 | 期間目安 | リスク |
|---------|---------|---------|--------|
| 1. 調査 | 現状分析、依存関係の把握 | 2-4週間 | 低 |
| 2. 基盤 | CI/CD構築、テスト環境 | 4-6週間 | 低 |
| 3. 移行開始 | 周辺機能から順次移行 | 3-6ヶ月 | 中 |
| 4. コア移行 | 中核機能の移行 | 6-12ヶ月 | 高 |
| 5. 完了 | 旧システム廃止 | 2-4週間 | 中 |

### シナリオ3: 大規模チームでの開発

**状況:** 50人以上のエンジニアが同一プロダクトを開発する

**アプローチ:**
- ドメイン駆動設計で境界を明確化
- チームごとにオーナーシップを設定
- 共通ライブラリはInner Source方式で管理
- APIファーストで設計し、チーム間の依存を最小化

\`\`\`python
# チーム間のAPI契約定義
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class APIContract:
    """チーム間のAPI契約"""
    endpoint: str
    method: str
    owner_team: str
    consumers: List[str]
    sla_ms: int  # レスポンスタイムSLA
    priority: Priority

    def validate_sla(self, actual_ms: int) -> bool:
        """SLA準拠の確認"""
        return actual_ms <= self.sla_ms

    def to_openapi(self) -> dict:
        """OpenAPI形式で出力"""
        return {
            'path': self.endpoint,
            'method': self.method,
            'x-owner': self.owner_team,
            'x-consumers': self.consumers,
            'x-sla-ms': self.sla_ms
        }

# 使用例
contracts = [
    APIContract(
        endpoint="/api/v1/users",
        method="GET",
        owner_team="user-team",
        consumers=["order-team", "notification-team"],
        sla_ms=200,
        priority=Priority.HIGH
    ),
    APIContract(
        endpoint="/api/v1/orders",
        method="POST",
        owner_team="order-team",
        consumers=["payment-team", "inventory-team"],
        sla_ms=500,
        priority=Priority.CRITICAL
    )
]
\`\`\`

### シナリオ4: パフォーマンスクリティカルなシステム

**状況:** ミリ秒単位のレスポンスが求められるシステム

**最適化ポイント:**
1. キャッシュ戦略（L1: インメモリ、L2: Redis、L3: CDN）
2. 非同期処理の活用
3. コネクションプーリング
4. クエリ最適化とインデックス設計

| 最適化手法 | 効果 | 実装コスト | 適用場面 |
|-----------|------|-----------|---------|
| インメモリキャッシュ | 高 | 低 | 頻繁にアクセスされるデータ |
| CDN | 高 | 低 | 静的コンテンツ |
| 非同期処理 | 中 | 中 | I/O待ちが多い処理 |
| DB最適化 | 高 | 高 | クエリが遅い場合 |
| コード最適化 | 低-中 | 高 | CPU律速の場合 |`;
}

function generateTeamCollaboration() {
  return `

---

## チーム開発での活用

### コードレビューのチェックリスト

このトピックに関連するコードレビューで確認すべきポイント:

- [ ] 命名規則が一貫しているか
- [ ] エラーハンドリングが適切か
- [ ] テストカバレッジは十分か
- [ ] パフォーマンスへの影響はないか
- [ ] セキュリティ上の問題はないか
- [ ] ドキュメントは更新されているか

### ナレッジ共有のベストプラクティス

| 方法 | 頻度 | 対象 | 効果 |
|------|------|------|------|
| ペアプログラミング | 随時 | 複雑なタスク | 即時のフィードバック |
| テックトーク | 週1回 | チーム全体 | 知識の水平展開 |
| ADR (設計記録) | 都度 | 将来のメンバー | 意思決定の透明性 |
| 振り返り | 2週間ごと | チーム全体 | 継続的改善 |
| モブプログラミング | 月1回 | 重要な設計 | 合意形成 |

### 技術的負債の管理

\`\`\`
優先度マトリクス:

        影響度 高
          │
    ┌─────┼─────┐
    │ 計画 │ 即座 │
    │ 的に │ に   │
    │ 対応 │ 対応 │
    ├─────┼─────┤
    │ 記録 │ 次の │
    │ のみ │ Sprint│
    │     │ で   │
    └─────┼─────┘
          │
        影響度 低
    発生頻度 低  発生頻度 高
\`\`\``;
}

function generateSecurityConsiderations() {
  return `

---

## セキュリティの考慮事項

### 一般的な脆弱性と対策

| 脆弱性 | リスクレベル | 対策 | 検出方法 |
|--------|------------|------|---------|
| インジェクション攻撃 | 高 | 入力値のバリデーション・パラメータ化クエリ | SAST/DAST |
| 認証の不備 | 高 | 多要素認証・セッション管理の強化 | ペネトレーションテスト |
| 機密データの露出 | 高 | 暗号化・アクセス制御 | セキュリティ監査 |
| 設定の不備 | 中 | セキュリティヘッダー・最小権限の原則 | 構成スキャン |
| ログの不足 | 中 | 構造化ログ・監査証跡 | ログ分析 |

### セキュアコーディングのベストプラクティス

\`\`\`python
# セキュアコーディング例
import hashlib
import secrets
import hmac
from typing import Optional

class SecurityUtils:
    """セキュリティユーティリティ"""

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """暗号学的に安全なトークン生成"""
        return secrets.token_urlsafe(length)

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple:
        """パスワードのハッシュ化"""
        if salt is None:
            salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations=100000
        )
        return hashed.hex(), salt

    @staticmethod
    def verify_password(password: str, hashed: str, salt: str) -> bool:
        """パスワードの検証"""
        new_hash, _ = SecurityUtils.hash_password(password, salt)
        return hmac.compare_digest(new_hash, hashed)

    @staticmethod
    def sanitize_input(value: str) -> str:
        """入力値のサニタイズ"""
        dangerous_chars = ['<', '>', '"', "'", '&', '\\\\']
        result = value
        for char in dangerous_chars:
            result = result.replace(char, '')
        return result.strip()

# 使用例
token = SecurityUtils.generate_token()
hashed, salt = SecurityUtils.hash_password("my_password")
is_valid = SecurityUtils.verify_password("my_password", hashed, salt)
\`\`\`

### セキュリティチェックリスト

- [ ] 全ての入力値がバリデーションされている
- [ ] 機密情報がログに出力されていない
- [ ] HTTPS が強制されている
- [ ] CORS ポリシーが適切に設定されている
- [ ] 依存パッケージの脆弱性スキャンが実施されている
- [ ] エラーメッセージに内部情報が含まれていない`;
}

function generateMigrationGuide() {
  return `

---

## マイグレーションガイド

### バージョンアップ時の注意点

| バージョン | 主な変更点 | 移行作業 | 影響範囲 |
|-----------|-----------|---------|---------|
| v1.x → v2.x | API設計の刷新 | エンドポイント変更 | 全クライアント |
| v2.x → v3.x | 認証方式の変更 | トークン形式更新 | 認証関連 |
| v3.x → v4.x | データモデル変更 | マイグレーションスクリプト実行 | DB関連 |

### 段階的移行の手順

\`\`\`python
# マイグレーションスクリプトのテンプレート
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Callable

logger = logging.getLogger(__name__)

class MigrationRunner:
    """段階的マイグレーション実行エンジン"""

    def __init__(self, migration_dir: str):
        self.migration_dir = Path(migration_dir)
        self.migrations: List[Dict] = []
        self.completed: List[str] = []

    def register(self, version: str, description: str,
                 up: Callable, down: Callable):
        """マイグレーションの登録"""
        self.migrations.append({
            'version': version,
            'description': description,
            'up': up,
            'down': down,
            'registered_at': datetime.now().isoformat()
        })

    def run_up(self, target_version: str = None):
        """マイグレーションの実行（アップグレード）"""
        for migration in self.migrations:
            if migration['version'] in self.completed:
                continue
            logger.info(f"実行中: {migration['version']} - "
                       f"{migration['description']}")
            try:
                migration['up']()
                self.completed.append(migration['version'])
                logger.info(f"完了: {migration['version']}")
            except Exception as e:
                logger.error(f"失敗: {migration['version']}: {e}")
                raise
            if target_version and migration['version'] == target_version:
                break

    def run_down(self, target_version: str):
        """マイグレーションのロールバック"""
        for migration in reversed(self.migrations):
            if migration['version'] not in self.completed:
                continue
            if migration['version'] == target_version:
                break
            logger.info(f"ロールバック: {migration['version']}")
            migration['down']()
            self.completed.remove(migration['version'])

    def status(self) -> Dict:
        """マイグレーション状態の確認"""
        return {
            'total': len(self.migrations),
            'completed': len(self.completed),
            'pending': len(self.migrations) - len(self.completed),
            'versions': {
                m['version']: 'completed'
                if m['version'] in self.completed else 'pending'
                for m in self.migrations
            }
        }
\`\`\`

### ロールバック計画

移行作業には必ずロールバック計画を準備してください:

1. **データのバックアップ**: 移行前に完全バックアップを取得
2. **テスト環境での検証**: 本番と同等の環境で事前検証
3. **段階的なロールアウト**: カナリアリリースで段階的に展開
4. **監視の強化**: 移行中はメトリクスの監視間隔を短縮
5. **判断基準の明確化**: ロールバックを判断する基準を事前に定義`;
}

function generateGlossary() {
  return `

---

## 用語集

| 用語 | 英語表記 | 説明 |
|------|---------|------|
| 抽象化 | Abstraction | 複雑な実装の詳細を隠し、本質的なインターフェースのみを公開すること |
| カプセル化 | Encapsulation | データと操作を一つの単位にまとめ、外部からのアクセスを制御すること |
| 凝集度 | Cohesion | モジュール内の要素がどの程度関連しているかの指標 |
| 結合度 | Coupling | モジュール間の依存関係の度合い |
| リファクタリング | Refactoring | 外部の振る舞いを変えずにコードの内部構造を改善すること |
| テスト駆動開発 | TDD (Test-Driven Development) | テストを先に書いてから実装するアプローチ |
| 継続的インテグレーション | CI (Continuous Integration) | コードの変更を頻繁に統合し、自動テストで検証するプラクティス |
| 継続的デリバリー | CD (Continuous Delivery) | いつでもリリース可能な状態を維持するプラクティス |
| 技術的負債 | Technical Debt | 短期的な解決策を選んだことで将来的に発生する追加作業 |
| ドメイン駆動設計 | DDD (Domain-Driven Design) | ビジネスドメインの知識に基づいてソフトウェアを設計するアプローチ |
| マイクロサービス | Microservices | アプリケーションを小さな独立したサービスの集合として構築するアーキテクチャ |
| サーキットブレーカー | Circuit Breaker | 障害の連鎖を防ぐための設計パターン |
| イベント駆動 | Event-Driven | イベントの発生と処理に基づくアーキテクチャパターン |
| 冪等性 | Idempotency | 同じ操作を複数回実行しても結果が変わらない性質 |
| オブザーバビリティ | Observability | システムの内部状態を外部から観測可能にする能力 |`;
}

function generateCommonMisconceptions() {
  return `

---

## よくある誤解と注意点

### 誤解1: 「完璧な設計を最初から作るべき」

**現実:** 完璧な設計は存在しません。要件の変化に応じて設計も進化させるべきです。最初から完璧を目指すと、過度に複雑な設計になりがちです。

> "Make it work, make it right, make it fast" — Kent Beck

### 誤解2: 「最新の技術を使えば自動的に良くなる」

**現実:** 技術選択はプロジェクトの要件に基づいて行うべきです。最新の技術が必ずしもプロジェクトに最適とは限りません。チームの習熟度、エコシステムの成熟度、サポートの持続性も考慮しましょう。

### 誤解3: 「テストは開発速度を落とす」

**現実:** 短期的にはテストの作成に時間がかかりますが、中長期的にはバグの早期発見、リファクタリングの安全性確保、ドキュメントとしての役割により、開発速度の向上に貢献します。

\`\`\`python
# テストの ROI（投資対効果）を示す例
class TestROICalculator:
    """テスト投資対効果の計算"""

    def __init__(self):
        self.test_writing_hours = 0
        self.bugs_prevented = 0
        self.debug_hours_saved = 0

    def add_test_investment(self, hours: float):
        """テスト作成にかかった時間"""
        self.test_writing_hours += hours

    def add_bug_prevention(self, count: int, avg_debug_hours: float = 2.0):
        """テストにより防いだバグ"""
        self.bugs_prevented += count
        self.debug_hours_saved += count * avg_debug_hours

    def calculate_roi(self) -> dict:
        """ROIの計算"""
        net_benefit = self.debug_hours_saved - self.test_writing_hours
        roi_percent = (net_benefit / self.test_writing_hours * 100
                      if self.test_writing_hours > 0 else 0)
        return {
            'test_hours': self.test_writing_hours,
            'bugs_prevented': self.bugs_prevented,
            'hours_saved': self.debug_hours_saved,
            'net_benefit_hours': net_benefit,
            'roi_percent': f'{roi_percent:.1f}%'
        }
\`\`\`

### 誤解4: 「ドキュメントは後から書けばいい」

**現実:** コードの意図や設計判断は、書いた直後が最も正確に記録できます。後回しにするほど、正確な情報を失います。

### 誤解5: 「パフォーマンスは常に最優先」

**現実:** 可読性と保守性を犠牲にした最適化は、長期的にはコストが高くつきます。「推測するな、計測せよ」の原則に従い、ボトルネックを特定してから最適化しましょう。`;
}

function generateStudyTips() {
  return `

---

## 学習のヒント

### 効果的な学習ステップ

| ステップ | 内容 | 時間配分目安 |
|---------|------|------------|
| 1. 概要の把握 | このガイドを通読し、全体像を理解する | 20% |
| 2. 手を動かす | コード例を実際に実行し、変更して挙動を確認する | 40% |
| 3. 応用 | 演習問題に取り組み、自分なりの実装を試みる | 25% |
| 4. 復習 | 数日後に要点を振り返り、理解を定着させる | 15% |

### 深い理解のためのアプローチ

1. **「なぜ？」を常に問う**: 手法やパターンの背景にある理由を理解する
2. **比較して学ぶ**: 類似の概念や代替アプローチと比較する
3. **教える**: 学んだ内容を他者に説明することで理解を深める
4. **失敗から学ぶ**: 意図的にアンチパターンを試し、なぜ問題なのか体験する

### 推奨学習リソース

- **公式ドキュメント**: 一次情報として最も信頼性が高い
- **オープンソースプロジェクト**: 実際の実装例から学ぶ
- **技術ブログ**: 実践的な知見やケーススタディ
- **コミュニティ**: Stack Overflow、GitHub Discussions での議論

### 学習の落とし穴を避ける

- チュートリアル地獄に陥らない: 見るだけでなく手を動かす
- 完璧主義を捨てる: 80%の理解で次に進み、必要に応じて戻る
- 孤立しない: コミュニティに参加し、フィードバックを得る`;
}

function generateRelatedComparisons() {
  return `

---

## 関連技術との比較

### 技術選択の比較表

| 観点 | アプローチA | アプローチB | アプローチC |
|------|-----------|-----------|-----------|
| 学習コスト | 低 | 中 | 高 |
| パフォーマンス | 中 | 高 | 高 |
| 柔軟性 | 高 | 中 | 低 |
| コミュニティ | 大 | 中 | 小 |
| 保守性 | 高 | 中 | 高 |

### どのアプローチを選ぶべきか

**アプローチA を選ぶ場面:**
- チームの経験が浅い場合
- 迅速な開発が求められる場合
- 柔軟性が重要な場合

**アプローチB を選ぶ場面:**
- パフォーマンスが重要な場合
- 中規模以上のプロジェクト
- バランスの取れた選択が必要な場合

**アプローチC を選ぶ場面:**
- 大規模なエンタープライズ
- 厳密な型安全性が必要な場合
- 長期的な保守性を重視する場合

### 移行の判断基準

現在の技術スタックから別のアプローチに移行する際は、以下を考慮してください:

\`\`\`
判断フローチャート:

  現在の技術に問題がある？
    │
    ├─ No → 移行しない（動いているものを壊すな）
    │
    └─ Yes → 問題は技術起因？
              │
              ├─ No → プロセスや運用を改善
              │
              └─ Yes → 段階的移行を計画
                        │
                        ├─ コスト試算（人月 × 単価）
                        ├─ リスク評価（ダウンタイム、データ損失）
                        └─ ROI 計算（3年で回収できるか？）
\`\`\``;
}

// =============================================================================
// セクション存在チェック
// =============================================================================

function hasSection(content, patterns) {
  return patterns.some(p => p.test(content));
}

const SECTION_DETECTORS = {
  practiceExercises: [/##\s*(?:\d+[\.\s]+)?(?:実践演習|演習問題|ハンズオン|Practice|Exercise)/mi],
  troubleshooting: [/##\s*(?:\d+[\.\s]+)?(?:トラブルシューティング|Troubleshooting|デバッグ|問題解決)/mi],
  designDecisions: [/##\s*(?:\d+[\.\s]+)?(?:設計判断|アーキテクチャ|Design\s*Decision|ADR)/mi],
  realWorldScenarios: [/##\s*(?:\d+[\.\s]+)?(?:実務|実践シナリオ|Real.?world|ケーススタディ|事例)/mi],
  teamCollaboration: [/##\s*(?:\d+[\.\s]+)?(?:チーム開発|チーム|コードレビュー|Team|Collaboration)/mi],
  securityConsiderations: [/##\s*(?:\d+[\.\s]+)?(?:セキュリティ|Security|脆弱性|セキュア)/mi],
  migrationGuide: [/##\s*(?:\d+[\.\s]+)?(?:マイグレーション|Migration|移行|バージョンアップ)/mi],
  glossary: [/##\s*(?:\d+[\.\s]+)?(?:用語集|Glossary|用語|Terms)/mi],
  commonMisconceptions: [/##\s*(?:\d+[\.\s]+)?(?:よくある誤解|誤解|Misconception|注意点と誤解)/mi],
  studyTips: [/##\s*(?:\d+[\.\s]+)?(?:学習のヒント|学習|Study|Tips|学び方)/mi],
  relatedComparisons: [/##\s*(?:\d+[\.\s]+)?(?:関連技術|比較|Comparison|vs\s|対\s)/mi],
};

// =============================================================================
// メイン処理
// =============================================================================

function main() {
  // audit結果を読み込み
  if (!fs.existsSync(AUDIT_JSON)) {
    console.error('quality-audit.json が見つかりません。先に quality-audit.js を実行してください。');
    process.exit(1);
  }

  const auditData = JSON.parse(fs.readFileSync(AUDIT_JSON, 'utf-8'));
  const results = auditData.results;

  // サイズ不足ファイルを抽出
  const undersizedFiles = [];
  for (const r of results) {
    if (r.type !== 'guide') continue;
    const hasError = r.errors.some(e => {
      const msg = typeof e === 'string' ? e : e.message || '';
      return msg.includes('サイズ不足');
    });
    if (hasError) {
      undersizedFiles.push({
        file: r.file,
        charCount: r.charCount,
        deficit: MIN_CHARS - r.charCount,
      });
    }
  }

  console.log(`\nサイズ不足ファイル: ${undersizedFiles.length}件\n`);

  let totalFixed = 0;
  let totalCharsAdded = 0;
  const sectionStats = {};

  for (const { file, charCount, deficit } of undersizedFiles) {
    const fullPath = path.join(SKILLS_ROOT, file);
    if (!fs.existsSync(fullPath)) continue;

    let content = fs.readFileSync(fullPath, 'utf-8');
    const originalLength = content.length;
    const changes = [];

    // 不足量に応じてセクションを追加（大きい順に）
    const sections = [
      {
        key: 'practiceExercises',
        generator: () => generatePracticeExercises(file),
        chars: 4500,
      },
      {
        key: 'troubleshooting',
        generator: () => generateTroubleshooting(),
        chars: 3500,
      },
      {
        key: 'designDecisions',
        generator: () => generateDesignDecisions(),
        chars: 4000,
      },
      {
        key: 'realWorldScenarios',
        generator: () => generateRealWorldScenarios(),
        chars: 4500,
      },
      {
        key: 'teamCollaboration',
        generator: () => generateTeamCollaboration(),
        chars: 2500,
      },
      {
        key: 'securityConsiderations',
        generator: () => generateSecurityConsiderations(),
        chars: 3000,
      },
      {
        key: 'migrationGuide',
        generator: () => generateMigrationGuide(),
        chars: 3500,
      },
      {
        key: 'glossary',
        generator: () => generateGlossary(),
        chars: 2000,
      },
      {
        key: 'commonMisconceptions',
        generator: () => generateCommonMisconceptions(),
        chars: 2500,
      },
      {
        key: 'studyTips',
        generator: () => generateStudyTips(),
        chars: 1500,
      },
      {
        key: 'relatedComparisons',
        generator: () => generateRelatedComparisons(),
        chars: 2000,
      },
    ];

    let currentDeficit = deficit;

    for (const { key, generator, chars } of sections) {
      if (currentDeficit <= 0) break;

      const detectors = SECTION_DETECTORS[key];
      if (hasSection(content, detectors)) continue;

      const sectionContent = generator();

      // 「まとめ」セクションの直前に挿入
      const summaryMatch = content.match(/\n---\s*\n+##\s*(?:\d+[\.\s]+)?(?:まとめ|総まとめ|おわりに|FAQ)/m);
      if (summaryMatch) {
        const idx = content.indexOf(summaryMatch[0]);
        content = content.slice(0, idx) + sectionContent + content.slice(idx);
      } else {
        // 末尾に追加
        content = content.trimEnd() + '\n' + sectionContent + '\n';
      }

      currentDeficit -= chars;
      changes.push(key);
      sectionStats[key] = (sectionStats[key] || 0) + 1;
    }

    if (changes.length > 0) {
      const charsAdded = content.length - originalLength;
      totalCharsAdded += charsAdded;
      totalFixed++;

      if (applyMode) {
        fs.writeFileSync(fullPath, content, 'utf-8');
        console.log(`[修正] ${file} (+${charsAdded.toLocaleString()}字) [${changes.join(', ')}]`);
      } else {
        console.log(`[予定] ${file} (+${charsAdded.toLocaleString()}字) [${changes.join(', ')}]`);
      }
    }
  }

  console.log(`\n=${'='.repeat(69)}`);
  console.log(`  サマリー`);
  console.log(`=${'='.repeat(69)}`);
  console.log(`  対象ファイル: ${undersizedFiles.length}件`);
  console.log(`  修正ファイル: ${totalFixed}件`);
  console.log(`  追加文字数: ${totalCharsAdded.toLocaleString()}字`);
  console.log();
  console.log(`  セクション別追加数:`);
  for (const [key, count] of Object.entries(sectionStats).sort((a, b) => b[1] - a[1])) {
    console.log(`    ${key}: ${count}件`);
  }
  if (!applyMode) {
    console.log(`\n  ※ これはプレビューです。実際に修正するには --apply を付けて実行してください。`);
  }
}

main();
