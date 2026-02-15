# データエージェント

> 分析・可視化・洞察――データベースのクエリ、統計分析、グラフ生成を自律的に行うデータ分析エージェントの設計と実装。

## この章で学ぶこと

1. 自然言語からSQLへの変換（Text-to-SQL）とデータ分析パイプラインの設計
2. データ可視化の自動生成とインサイト抽出の実装パターン
3. データエージェントの安全性（読み取り専用、インジェクション防止）の確保
4. 複数データソースを横断する統合分析エージェントの構築
5. 本番運用におけるキャッシュ戦略、コスト最適化、モニタリング

---

## 1. データエージェントの全体像

```
データエージェントのパイプライン

[自然言語の質問]
  "先月の売上トップ10商品は？"
       |
       v
[質問理解] ── スキーマ情報を参照
       |
       v
[SQL生成] ── Text-to-SQL
       |
       v
[SQL検証] ── 安全性チェック（READ ONLY）
       |
       v
[SQL実行] ── DB接続・クエリ実行
       |
       v
[結果分析] ── 統計処理、パターン検出
       |
       v
[可視化生成] ── グラフ・チャート作成
       |
       v
[インサイト] ── 自然言語での洞察
```

### 1.1 データエージェントの分類

```
データエージェントの種類

┌────────────────┬────────────────┬────────────────┐
│  クエリエージェント │  分析エージェント  │ レポートエージェント │
│                │                │                │
│ 自然言語→SQL   │ 統計分析・ML    │ 定期レポート生成  │
│ 単発の質問応答  │ 多段階の分析    │ ダッシュボード更新 │
│ 即時レスポンス  │ 深い洞察の抽出  │ スケジュール実行  │
└────────────────┴────────────────┴────────────────┘
         │                │                │
         v                v                v
┌────────────────┬────────────────┬────────────────┐
│ パイプライン     │ 探索的エージェント│ 異常検知エージェント│
│ エージェント     │                │                │
│ ETL自動化      │ 仮説生成→検証   │ データ品質監視   │
│ データ変換      │ 反復的な深掘り  │ アラート通知     │
│ 品質チェック    │ レポート作成    │ 自動対応        │
└────────────────┴────────────────┴────────────────┘
```

### 1.2 アーキテクチャの全体構成

```python
# データエージェントのアーキテクチャ全体像
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class AgentRole(Enum):
    QUERY = "query"           # 単発クエリ実行
    ANALYST = "analyst"       # 多段階分析
    REPORTER = "reporter"     # レポート生成
    MONITOR = "monitor"       # 異常検知・監視

@dataclass
class DataAgentConfig:
    """データエージェントの統合設定"""
    role: AgentRole
    db_connections: dict[str, str]        # 名前→接続文字列
    max_rows: int = 1000                  # 結果の最大行数
    max_query_time_seconds: int = 30      # クエリタイムアウト
    enable_caching: bool = True           # 結果キャッシュ
    cache_ttl_seconds: int = 300          # キャッシュTTL
    pii_columns: set[str] = field(
        default_factory=lambda: {"email", "phone", "ssn", "credit_card", "address"}
    )
    allowed_operations: set[str] = field(
        default_factory=lambda: {"SELECT"}
    )
    model_name: str = "claude-sonnet-4-20250514"
    max_retries: int = 3
    enable_visualization: bool = True
    log_queries: bool = True              # クエリログ記録
    cost_limit_per_session: float = 1.0   # セッションあたりのAPI費用上限（USD）

@dataclass
class QueryResult:
    """クエリ結果の標準形式"""
    query: str
    columns: list[str]
    rows: list[tuple]
    row_count: int
    execution_time_ms: float
    truncated: bool = False
    error: Optional[str] = None
    cached: bool = False

    def to_dataframe(self):
        """pandas DataFrameに変換"""
        import pandas as pd
        return pd.DataFrame(self.rows, columns=self.columns)

    def summary(self) -> str:
        """結果の要約を文字列で返す"""
        if self.error:
            return f"エラー: {self.error}"
        lines = [
            f"カラム: {', '.join(self.columns)}",
            f"行数: {self.row_count}{'（切り詰め済み）' if self.truncated else ''}",
            f"実行時間: {self.execution_time_ms:.1f}ms",
        ]
        if self.cached:
            lines.append("（キャッシュ結果）")
        return "\n".join(lines)
```

---

## 2. Text-to-SQL

### 2.1 基本実装

```python
# Text-to-SQL エージェント
import anthropic
import sqlite3
import json
import time
import hashlib
from typing import Optional

class TextToSQLAgent:
    def __init__(self, db_path: str, config: Optional[DataAgentConfig] = None):
        self.client = anthropic.Anthropic()
        self.db_path = db_path
        self.schema = self._get_schema()
        self.config = config or DataAgentConfig(role=AgentRole.QUERY, db_connections={})
        self._query_cache: dict[str, QueryResult] = {}
        self._query_log: list[dict] = []

    def _get_schema(self) -> str:
        """データベーススキーマを取得"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table'"
        )
        schemas = cursor.fetchall()
        conn.close()
        return "\n".join(s[0] for s in schemas if s[0])

    def query(self, question: str) -> dict:
        """自然言語の質問をSQLに変換して実行"""

        # 1. SQL生成
        sql = self._generate_sql(question)

        # 2. 安全性チェック
        if not self._is_safe_query(sql):
            return {"error": "安全でないクエリが検出されました"}

        # 3. キャッシュチェック
        cache_key = self._cache_key(sql)
        if self.config.enable_caching and cache_key in self._query_cache:
            cached = self._query_cache[cache_key]
            cached.cached = True
            return {
                "question": question,
                "sql": sql,
                "results": {"columns": cached.columns, "rows": cached.rows},
                "interpretation": "（キャッシュされた結果）",
                "cached": True
            }

        # 4. 実行
        start_time = time.time()
        results = self._execute_sql(sql)
        execution_time = (time.time() - start_time) * 1000

        # 5. ログ記録
        if self.config.log_queries:
            self._query_log.append({
                "question": question,
                "sql": sql,
                "execution_time_ms": execution_time,
                "success": "error" not in results,
                "timestamp": time.time()
            })

        # 6. 結果の解釈
        interpretation = self._interpret_results(question, sql, results)

        return {
            "question": question,
            "sql": sql,
            "results": results,
            "interpretation": interpretation,
            "execution_time_ms": execution_time
        }

    def _cache_key(self, sql: str) -> str:
        """SQLからキャッシュキーを生成"""
        return hashlib.sha256(sql.strip().lower().encode()).hexdigest()

    def _generate_sql(self, question: str) -> str:
        """自然言語をSQLに変換"""
        response = self.client.messages.create(
            model=self.config.model_name,
            max_tokens=1024,
            messages=[{"role": "user", "content": f"""
データベーススキーマ:
{self.schema}

以下の質問に対するSQLクエリを生成してください。
SELECTのみ使用可能です（INSERT/UPDATE/DELETEは不可）。

質問: {question}

SQLクエリのみを出力（説明不要）:
"""}]
        )
        sql = response.content[0].text.strip()
        # ```sql ... ``` の形式を処理
        if sql.startswith("```"):
            sql = sql.split("\n", 1)[1].rsplit("```", 1)[0]
        return sql.strip()

    def _is_safe_query(self, sql: str) -> bool:
        """SQLの安全性をチェック"""
        dangerous_keywords = [
            "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
            "CREATE", "TRUNCATE", "EXEC", "EXECUTE",
            "GRANT", "REVOKE"
        ]
        sql_upper = sql.upper()
        return not any(kw in sql_upper for kw in dangerous_keywords)

    def _execute_sql(self, sql: str) -> list:
        """SQLを実行"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA query_only = ON")  # 読み取り専用を強制
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            return {"columns": columns, "rows": rows[:1000]}  # 最大1000行
        except Exception as e:
            return {"error": str(e)}
        finally:
            conn.close()

    def _interpret_results(self, question: str, sql: str, results: dict) -> str:
        """結果を自然言語で解釈"""
        response = self.client.messages.create(
            model=self.config.model_name,
            max_tokens=1024,
            messages=[{"role": "user", "content": f"""
質問: {question}
実行SQL: {sql}
結果: {json.dumps(results, ensure_ascii=False)[:3000]}

結果を分かりやすく日本語で解釈してください。
重要な数値やトレンドがあれば指摘してください。
"""}]
        )
        return response.content[0].text
```

### 2.2 スキーマ情報の強化

```python
# スキーマに説明を付加して精度向上
ENHANCED_SCHEMA = """
-- 商品テーブル
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,           -- 商品名
    category TEXT NOT NULL,        -- カテゴリ（electronics, clothing, food, etc.）
    price REAL NOT NULL,           -- 税抜価格（円）
    stock INTEGER DEFAULT 0,       -- 在庫数
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 注文テーブル
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,  -- 顧客ID
    product_id INTEGER NOT NULL,   -- 商品ID
    quantity INTEGER NOT NULL,     -- 数量
    total_price REAL NOT NULL,     -- 合計金額（税込）
    status TEXT DEFAULT 'pending', -- pending/confirmed/shipped/delivered/cancelled
    ordered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(id)
);

-- サンプルデータの例:
-- products: (1, 'ワイヤレスイヤホン', 'electronics', 12800, 150, '2025-01-01')
-- orders: (1, 101, 1, 2, 28160, 'delivered', '2025-01-15')
"""
```

### 2.3 動的スキーマ選択

```python
# 大規模DB向け: 質問に関連するテーブルだけを選択
class SchemaSelector:
    """100+テーブルのDBから関連テーブルを自動選択"""

    def __init__(self, db_path: str):
        self.client = anthropic.Anthropic()
        self.db_path = db_path
        self.table_catalog = self._build_catalog()

    def _build_catalog(self) -> dict[str, str]:
        """全テーブルの名前と説明のカタログを作成"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        conn.close()

        catalog = {}
        for name, sql in tables:
            if sql:
                # テーブル名とカラム名を抽出して要約
                catalog[name] = self._summarize_table(name, sql)
        return catalog

    def _summarize_table(self, name: str, create_sql: str) -> str:
        """テーブルのCREATE文からカラム名のリストを抽出"""
        import re
        columns = re.findall(r'(\w+)\s+(INTEGER|TEXT|REAL|BLOB|DATETIME|BOOLEAN)', create_sql)
        col_list = ", ".join(f"{c[0]}({c[1]})" for c in columns)
        return f"{name}: {col_list}"

    def select_tables(self, question: str, max_tables: int = 5) -> list[str]:
        """質問に関連するテーブルを選択"""
        catalog_text = "\n".join(
            f"- {name}: {desc}" for name, desc in self.table_catalog.items()
        )

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            messages=[{"role": "user", "content": f"""
以下のテーブル一覧から、質問に回答するために必要なテーブルを最大{max_tables}個選んでください。

テーブル一覧:
{catalog_text}

質問: {question}

テーブル名のみをカンマ区切りで出力:
"""}]
        )
        selected = response.content[0].text.strip().split(",")
        return [t.strip() for t in selected if t.strip() in self.table_catalog]

    def get_selected_schema(self, question: str) -> str:
        """質問に関連するテーブルのCREATE文のみを返す"""
        selected = self.select_tables(question)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        schemas = []
        for table_name in selected:
            cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            row = cursor.fetchone()
            if row and row[0]:
                schemas.append(row[0])

        conn.close()
        return "\n\n".join(schemas)
```

### 2.4 Few-shot SQL生成

```python
# Few-shot例を使ったSQL生成精度の向上
class FewShotSQLGenerator:
    """類似質問のSQL例を使って生成精度を向上"""

    def __init__(self, db_path: str):
        self.client = anthropic.Anthropic()
        self.db_path = db_path
        self.schema = self._get_schema(db_path)
        # Few-shot例のストア（本番ではベクトルDBを使用）
        self.examples: list[dict] = []

    def _get_schema(self, db_path: str) -> str:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
        schemas = cursor.fetchall()
        conn.close()
        return "\n".join(s[0] for s in schemas if s[0])

    def add_example(self, question: str, sql: str, description: str = ""):
        """Few-shot例を追加"""
        self.examples.append({
            "question": question,
            "sql": sql,
            "description": description
        })

    def find_similar_examples(self, question: str, top_k: int = 3) -> list[dict]:
        """質問に類似するFew-shot例を検索（簡易版: キーワードマッチ）"""
        import re
        question_words = set(re.findall(r'\w+', question.lower()))
        scored = []
        for ex in self.examples:
            ex_words = set(re.findall(r'\w+', ex["question"].lower()))
            overlap = len(question_words & ex_words)
            scored.append((overlap, ex))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:top_k]]

    def generate_sql(self, question: str) -> str:
        """Few-shot例を含めたSQL生成"""
        similar = self.find_similar_examples(question)

        examples_text = ""
        if similar:
            examples_text = "参考例:\n"
            for i, ex in enumerate(similar, 1):
                examples_text += f"""
例{i}:
  質問: {ex['question']}
  SQL: {ex['sql']}
"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": f"""
データベーススキーマ:
{self.schema}

{examples_text}

質問: {question}

上記の参考例を参考にして、SQLクエリを生成してください。
SELECTのみ使用可能です。
SQLクエリのみを出力:
"""}]
        )
        sql = response.content[0].text.strip()
        if sql.startswith("```"):
            sql = sql.split("\n", 1)[1].rsplit("```", 1)[0]
        return sql.strip()

    def learn_from_correction(self, question: str, corrected_sql: str):
        """ユーザーの修正からFew-shot例として学習"""
        self.add_example(
            question=question,
            sql=corrected_sql,
            description="ユーザー修正済み"
        )
```

### 2.5 SQLバリデーションの高度化

```python
import sqlparse
from typing import Optional

class SQLValidator:
    """多層のSQLバリデーション"""

    # 許可する構文要素
    ALLOWED_STATEMENT_TYPES = {"SELECT"}

    # 禁止するキーワード（大文字）
    FORBIDDEN_KEYWORDS = {
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
        "TRUNCATE", "EXEC", "EXECUTE", "GRANT", "REVOKE",
        "CALL", "LOAD", "REPLACE", "MERGE", "UPSERT",
        "ATTACH", "DETACH", "VACUUM", "REINDEX", "ANALYZE",
    }

    # 禁止する関数（SQLインジェクション対策）
    FORBIDDEN_FUNCTIONS = {
        "LOAD_FILE", "INTO OUTFILE", "INTO DUMPFILE",
        "SLEEP", "BENCHMARK", "WAITFOR",
    }

    def validate(self, sql: str) -> tuple[bool, Optional[str]]:
        """SQLを検証し、(is_valid, error_message)を返す"""
        checks = [
            self._check_statement_type,
            self._check_forbidden_keywords,
            self._check_forbidden_functions,
            self._check_subquery_depth,
            self._check_union_count,
            self._check_comment_injection,
            self._check_semicolon,
        ]

        for check in checks:
            is_valid, error = check(sql)
            if not is_valid:
                return False, error

        return True, None

    def _check_statement_type(self, sql: str) -> tuple[bool, Optional[str]]:
        """ステートメントタイプの検証"""
        parsed = sqlparse.parse(sql)
        if not parsed:
            return False, "空のSQL"
        stmt_type = parsed[0].get_type()
        if stmt_type and stmt_type.upper() not in self.ALLOWED_STATEMENT_TYPES:
            return False, f"許可されていないステートメント: {stmt_type}"
        return True, None

    def _check_forbidden_keywords(self, sql: str) -> tuple[bool, Optional[str]]:
        """禁止キーワードの検出"""
        tokens = sqlparse.parse(sql)[0].flatten()
        for token in tokens:
            if token.ttype is sqlparse.tokens.Keyword:
                if token.value.upper() in self.FORBIDDEN_KEYWORDS:
                    return False, f"禁止キーワード: {token.value}"
        return True, None

    def _check_forbidden_functions(self, sql: str) -> tuple[bool, Optional[str]]:
        """禁止関数の検出"""
        sql_upper = sql.upper()
        for func in self.FORBIDDEN_FUNCTIONS:
            if func in sql_upper:
                return False, f"禁止関数: {func}"
        return True, None

    def _check_subquery_depth(self, sql: str, max_depth: int = 3) -> tuple[bool, Optional[str]]:
        """サブクエリのネスト深さ制限"""
        depth = 0
        max_found = 0
        for char in sql:
            if char == '(':
                depth += 1
                max_found = max(max_found, depth)
            elif char == ')':
                depth -= 1
        if max_found > max_depth:
            return False, f"サブクエリが深すぎます（深さ{max_found}、最大{max_depth}）"
        return True, None

    def _check_union_count(self, sql: str, max_unions: int = 5) -> tuple[bool, Optional[str]]:
        """UNION句の数を制限"""
        union_count = sql.upper().count("UNION")
        if union_count > max_unions:
            return False, f"UNION句が多すぎます（{union_count}個、最大{max_unions}）"
        return True, None

    def _check_comment_injection(self, sql: str) -> tuple[bool, Optional[str]]:
        """コメントインジェクション検出"""
        if "--" in sql or "/*" in sql:
            return False, "SQLコメントは許可されていません"
        return True, None

    def _check_semicolon(self, sql: str) -> tuple[bool, Optional[str]]:
        """複数ステートメント防止"""
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        if len(statements) > 1:
            return False, "複数ステートメントは許可されていません"
        return True, None
```

---

## 3. データ可視化

### 3.1 グラフ自動生成

```python
# Pythonコード生成によるグラフ作成
class DataVisualizer:
    def __init__(self):
        self.client = anthropic.Anthropic()

    def generate_chart(self, data: dict, chart_request: str) -> str:
        """データに基づいてグラフを生成するPythonコードを作成"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": f"""
以下のデータに基づいて、matplotlibでグラフを作成するPythonコードを生成してください。

データ:
カラム: {data['columns']}
行数: {len(data['rows'])}
先頭5行: {data['rows'][:5]}

要望: {chart_request}

ルール:
- matplotlib + pandas を使用
- 日本語フォントは 'Hiragino Sans' を使用
- plt.savefig('chart.png', dpi=150, bbox_inches='tight') で保存
- 見やすい配色を使う
"""}]
        )
        return response.content[0].text

    def auto_visualize(self, data: dict) -> str:
        """データの特性に応じて最適なグラフを自動選択"""
        num_columns = len(data["columns"])
        num_rows = len(data["rows"])

        if num_rows == 0:
            return "データがありません"

        # データの特性を分析
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{"role": "user", "content": f"""
以下のデータに最適なグラフ種類を1つ選んでください。

カラム: {data['columns']}
先頭3行: {data['rows'][:3]}
行数: {num_rows}

選択肢: bar（棒）, line（折れ線）, pie（円）, scatter（散布）, heatmap（ヒートマップ）, table（表のまま）

最適なグラフ種類とその理由を1行で:
"""}]
        )
        return response.content[0].text
```

### 3.2 高度な可視化テンプレート

```python
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from pathlib import Path

# 日本語フォント設定
matplotlib.rcParams['font.family'] = 'Hiragino Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

class ChartTemplates:
    """再利用可能なグラフテンプレート集"""

    # 共通のカラーパレット
    COLORS = [
        '#4e79a7', '#f28e2b', '#e15759', '#76b7b2',
        '#59a14f', '#edc948', '#b07aa1', '#ff9da7',
        '#9c755f', '#bab0ac'
    ]

    @staticmethod
    def sales_trend(df: pd.DataFrame, date_col: str, value_col: str,
                    title: str = "売上推移", output_path: str = "chart.png"):
        """売上推移の折れ線グラフ"""
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(df[date_col], df[value_col],
                color=ChartTemplates.COLORS[0],
                linewidth=2, marker='o', markersize=4)

        # 移動平均線を追加
        if len(df) >= 7:
            ma7 = df[value_col].rolling(window=7).mean()
            ax.plot(df[date_col], ma7,
                    color=ChartTemplates.COLORS[1],
                    linewidth=1.5, linestyle='--',
                    label='7日移動平均')

        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel("日付", fontsize=12)
        ax.set_ylabel("金額（円）", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Y軸のフォーマット（千円単位）
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: f'{x/1000:.0f}K')
        )

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path

    @staticmethod
    def category_comparison(df: pd.DataFrame, cat_col: str, value_col: str,
                            title: str = "カテゴリ比較",
                            output_path: str = "chart.png"):
        """カテゴリ別の横棒グラフ"""
        fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.4)))

        sorted_df = df.sort_values(value_col, ascending=True)
        colors = [ChartTemplates.COLORS[i % len(ChartTemplates.COLORS)]
                  for i in range(len(sorted_df))]

        bars = ax.barh(sorted_df[cat_col], sorted_df[value_col], color=colors)

        # 値ラベルを追加
        for bar, val in zip(bars, sorted_df[value_col]):
            ax.text(bar.get_width() + max(sorted_df[value_col]) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:,.0f}', va='center', fontsize=10)

        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel("金額（円）", fontsize=12)
        ax.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path

    @staticmethod
    def pie_chart(df: pd.DataFrame, label_col: str, value_col: str,
                  title: str = "構成比", output_path: str = "chart.png",
                  top_n: int = 8):
        """円グラフ（上位N個＋その他）"""
        fig, ax = plt.subplots(figsize=(10, 8))

        sorted_df = df.sort_values(value_col, ascending=False)

        if len(sorted_df) > top_n:
            top = sorted_df.head(top_n)
            other_sum = sorted_df.iloc[top_n:][value_col].sum()
            other_row = pd.DataFrame({
                label_col: ["その他"],
                value_col: [other_sum]
            })
            plot_df = pd.concat([top, other_row], ignore_index=True)
        else:
            plot_df = sorted_df

        wedges, texts, autotexts = ax.pie(
            plot_df[value_col],
            labels=plot_df[label_col],
            autopct='%1.1f%%',
            colors=ChartTemplates.COLORS[:len(plot_df)],
            startangle=90,
            pctdistance=0.85
        )

        for text in autotexts:
            text.set_fontsize(10)
            text.set_fontweight('bold')

        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path

    @staticmethod
    def heatmap(df: pd.DataFrame, title: str = "ヒートマップ",
                output_path: str = "chart.png"):
        """相関行列やクロス集計のヒートマップ"""
        fig, ax = plt.subplots(figsize=(10, 8))

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return None

        corr = numeric_df.corr()
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr.columns)

        # 値を各セルに表示
        for i in range(len(corr)):
            for j in range(len(corr)):
                text_color = 'white' if abs(corr.iloc[i, j]) > 0.5 else 'black'
                ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                        ha='center', va='center', color=text_color, fontsize=9)

        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path

    @staticmethod
    def multi_metric_dashboard(
        data: dict[str, pd.DataFrame],
        title: str = "ダッシュボード",
        output_path: str = "dashboard.png"
    ):
        """複数指標のダッシュボード（2x2グリッド）"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)

        panels = list(data.items())[:4]

        for idx, (panel_title, df) in enumerate(panels):
            ax = axes[idx // 2][idx % 2]
            if len(df.columns) >= 2:
                x_col, y_col = df.columns[0], df.columns[1]
                if pd.api.types.is_numeric_dtype(df[y_col]):
                    ax.bar(df[x_col].astype(str), df[y_col],
                           color=ChartTemplates.COLORS[idx])
                    ax.set_title(panel_title, fontsize=14, fontweight='bold')
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
```

### 3.3 インタラクティブ可視化

```python
# Plotlyを使ったインタラクティブグラフ生成
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class InteractiveVisualizer:
    """Plotlyベースのインタラクティブ可視化"""

    def sales_dashboard(self, df: pd.DataFrame) -> go.Figure:
        """売上ダッシュボード"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '日次売上推移', 'カテゴリ別売上',
                '時間帯別注文数', '顧客セグメント'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "pie"}]
            ]
        )

        # 日次売上
        if 'date' in df.columns and 'revenue' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['date'], y=df['revenue'],
                           mode='lines+markers', name='売上'),
                row=1, col=1
            )

        # カテゴリ別
        if 'category' in df.columns and 'revenue' in df.columns:
            cat_data = df.groupby('category')['revenue'].sum().reset_index()
            fig.add_trace(
                go.Bar(x=cat_data['category'], y=cat_data['revenue'],
                       name='カテゴリ'),
                row=1, col=2
            )

        fig.update_layout(
            height=800,
            title_text="売上ダッシュボード",
            showlegend=True,
            template="plotly_white"
        )
        return fig

    def time_series_with_anomaly(self, df: pd.DataFrame,
                                 date_col: str, value_col: str) -> go.Figure:
        """異常値ハイライト付き時系列グラフ"""
        # 異常値の検出（IQR法）
        q1 = df[value_col].quantile(0.25)
        q3 = df[value_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        normal = df[(df[value_col] >= lower) & (df[value_col] <= upper)]
        anomalies = df[(df[value_col] < lower) | (df[value_col] > upper)]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=normal[date_col], y=normal[value_col],
            mode='lines+markers', name='通常値',
            line=dict(color='#4e79a7')
        ))

        fig.add_trace(go.Scatter(
            x=anomalies[date_col], y=anomalies[value_col],
            mode='markers', name='異常値',
            marker=dict(color='red', size=12, symbol='x')
        ))

        # 正常範囲の帯を追加
        fig.add_hrect(
            y0=lower, y1=upper,
            fillcolor="lightgreen", opacity=0.1,
            annotation_text="正常範囲"
        )

        fig.update_layout(
            title="時系列データ（異常値ハイライト）",
            template="plotly_white",
            height=500
        )
        return fig
```

### 3.4 分析パイプライン

```python
# 複数ステップの分析パイプライン
class AnalysisPipeline:
    def __init__(self):
        self.client = anthropic.Anthropic()

    def comprehensive_analysis(self, db_path: str, topic: str) -> dict:
        """トピックに関する包括的なデータ分析"""
        agent = TextToSQLAgent(db_path)

        # Step 1: 概要統計
        overview = agent.query(f"{topic}の全体概要（件数、合計、平均）")

        # Step 2: トレンド分析
        trend = agent.query(f"{topic}の月次推移")

        # Step 3: トップN分析
        top_items = agent.query(f"{topic}のトップ10")

        # Step 4: 分布分析
        distribution = agent.query(f"{topic}のカテゴリ別分布")

        # Step 5: 前期比較
        comparison = agent.query(f"{topic}の前月比較")

        # Step 6: 統合インサイト
        insights = self._generate_insights({
            "overview": overview,
            "trend": trend,
            "top_items": top_items,
            "distribution": distribution,
            "comparison": comparison
        })

        return {
            "overview": overview,
            "trend": trend,
            "top_items": top_items,
            "distribution": distribution,
            "comparison": comparison,
            "insights": insights
        }

    def _generate_insights(self, analysis_results: dict) -> str:
        """複数の分析結果を統合してインサイトを生成"""
        results_text = json.dumps(analysis_results, ensure_ascii=False, default=str)[:5000]

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": f"""
以下の分析結果を統合して、ビジネスインサイトを3-5個抽出してください。

分析結果:
{results_text}

各インサイトには以下を含めてください:
1. 発見事項（何が分かったか）
2. ビジネスへの影響（なぜ重要か）
3. 推奨アクション（何をすべきか）

フォーマット:
### インサイト1: [タイトル]
- 発見: ...
- 影響: ...
- アクション: ...
"""}]
        )
        return response.content[0].text

    def anomaly_analysis(self, db_path: str, metric: str,
                         period: str = "過去30日") -> dict:
        """異常値検出と原因分析"""
        agent = TextToSQLAgent(db_path)

        # Step 1: ベースラインデータの取得
        baseline = agent.query(
            f"{metric}の{period}の日次データ（日付、値）"
        )

        # Step 2: 統計的な異常検出
        stats = agent.query(
            f"{metric}の{period}の平均、標準偏差、最小、最大"
        )

        # Step 3: 異常日の特定
        anomalies = agent.query(
            f"{metric}が通常の2倍以上または半分以下だった日"
        )

        # Step 4: 異常の原因候補を分析
        if anomalies.get("results", {}).get("rows"):
            cause_analysis = self._analyze_causes(
                agent, metric, anomalies
            )
        else:
            cause_analysis = "異常値は検出されませんでした"

        return {
            "baseline": baseline,
            "statistics": stats,
            "anomalies": anomalies,
            "cause_analysis": cause_analysis
        }

    def _analyze_causes(self, agent: TextToSQLAgent,
                        metric: str, anomalies: dict) -> str:
        """異常値の原因を分析"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": f"""
{metric}で異常値が検出されました:
{json.dumps(anomalies, ensure_ascii=False, default=str)[:2000]}

この異常の原因として考えられるものを3つ挙げ、
それぞれ確認するためのSQLクエリを提案してください。
"""}]
        )
        return response.content[0].text
```

---

## 4. 複数データソースの統合

### 4.1 マルチソースエージェント

```python
# 複数のデータソースを横断する分析エージェント
from abc import ABC, abstractmethod

class DataSource(ABC):
    """データソースの抽象インターフェース"""

    @abstractmethod
    def get_schema(self) -> str:
        pass

    @abstractmethod
    def execute_query(self, sql: str) -> QueryResult:
        pass

    @abstractmethod
    def get_sample_data(self, table: str, limit: int = 5) -> list[dict]:
        pass

class SQLiteSource(DataSource):
    def __init__(self, db_path: str, name: str = "sqlite"):
        self.db_path = db_path
        self.name = name

    def get_schema(self) -> str:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
        schemas = cursor.fetchall()
        conn.close()
        return "\n".join(s[0] for s in schemas if s[0])

    def execute_query(self, sql: str) -> QueryResult:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA query_only = ON")
        cursor = conn.cursor()
        start = time.time()
        try:
            cursor.execute(sql)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            elapsed = (time.time() - start) * 1000
            return QueryResult(
                query=sql, columns=columns, rows=rows[:1000],
                row_count=len(rows), execution_time_ms=elapsed,
                truncated=len(rows) > 1000
            )
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return QueryResult(
                query=sql, columns=[], rows=[], row_count=0,
                execution_time_ms=elapsed, error=str(e)
            )
        finally:
            conn.close()

    def get_sample_data(self, table: str, limit: int = 5) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table} LIMIT {limit}")
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        return [dict(zip(columns, row)) for row in rows]

class PostgreSQLSource(DataSource):
    def __init__(self, connection_string: str, name: str = "postgres"):
        self.connection_string = connection_string
        self.name = name

    def get_schema(self) -> str:
        import psycopg2
        conn = psycopg2.connect(self.connection_string)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position
        """)
        rows = cursor.fetchall()
        conn.close()

        tables: dict[str, list] = {}
        for table, col, dtype, nullable in rows:
            if table not in tables:
                tables[table] = []
            null_str = "NULL" if nullable == "YES" else "NOT NULL"
            tables[table].append(f"    {col} {dtype} {null_str}")

        schemas = []
        for table, cols in tables.items():
            schemas.append(f"CREATE TABLE {table} (\n" + ",\n".join(cols) + "\n);")
        return "\n\n".join(schemas)

    def execute_query(self, sql: str) -> QueryResult:
        import psycopg2
        conn = psycopg2.connect(self.connection_string)
        conn.set_session(readonly=True)
        cursor = conn.cursor()
        start = time.time()
        try:
            cursor.execute(sql)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            elapsed = (time.time() - start) * 1000
            return QueryResult(
                query=sql, columns=columns, rows=rows[:1000],
                row_count=len(rows), execution_time_ms=elapsed,
                truncated=len(rows) > 1000
            )
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return QueryResult(
                query=sql, columns=[], rows=[], row_count=0,
                execution_time_ms=elapsed, error=str(e)
            )
        finally:
            conn.close()

    def get_sample_data(self, table: str, limit: int = 5) -> list[dict]:
        import psycopg2
        conn = psycopg2.connect(self.connection_string)
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table} LIMIT {limit}")
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        return [dict(zip(columns, row)) for row in rows]


class MultiSourceAgent:
    """複数データソースを横断する分析エージェント"""

    def __init__(self, sources: dict[str, DataSource]):
        self.sources = sources
        self.client = anthropic.Anthropic()
        self.validator = SQLValidator()

    def query(self, question: str) -> dict:
        """自然言語の質問に対して適切なデータソースを選択して実行"""

        # 1. 関連するデータソースの選択
        source_name = self._select_source(question)
        source = self.sources[source_name]

        # 2. スキーマを取得
        schema = source.get_schema()

        # 3. SQL生成
        sql = self._generate_sql(question, schema, source_name)

        # 4. バリデーション
        is_valid, error = self.validator.validate(sql)
        if not is_valid:
            return {"error": f"SQLバリデーションエラー: {error}"}

        # 5. 実行
        result = source.execute_query(sql)

        # 6. 解釈
        interpretation = self._interpret(question, result)

        return {
            "source": source_name,
            "question": question,
            "sql": sql,
            "result": result,
            "interpretation": interpretation
        }

    def cross_source_analysis(self, question: str) -> dict:
        """複数データソースにまたがる分析"""
        # 各ソースから関連データを収集
        partial_results = {}
        for name, source in self.sources.items():
            schema = source.get_schema()
            sub_question = self._decompose_question(question, name, schema)
            if sub_question:
                sql = self._generate_sql(sub_question, schema, name)
                is_valid, _ = self.validator.validate(sql)
                if is_valid:
                    partial_results[name] = source.execute_query(sql)

        # 結果を統合して分析
        unified_insight = self._unify_results(question, partial_results)
        return {
            "question": question,
            "partial_results": partial_results,
            "unified_insight": unified_insight
        }

    def _select_source(self, question: str) -> str:
        """質問に最適なデータソースを選択"""
        source_descriptions = []
        for name, source in self.sources.items():
            schema_summary = source.get_schema()[:500]
            source_descriptions.append(f"- {name}: {schema_summary}")

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=64,
            messages=[{"role": "user", "content": f"""
以下のデータソースから、質問に最適なものを1つ選んでください。

データソース:
{chr(10).join(source_descriptions)}

質問: {question}

データソース名のみ出力:
"""}]
        )
        name = response.content[0].text.strip()
        return name if name in self.sources else list(self.sources.keys())[0]

    def _generate_sql(self, question: str, schema: str, source_name: str) -> str:
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": f"""
データソース: {source_name}
スキーマ:
{schema}

質問: {question}
SELECTのみ使用可能。SQLクエリのみ出力:
"""}]
        )
        sql = response.content[0].text.strip()
        if sql.startswith("```"):
            sql = sql.split("\n", 1)[1].rsplit("```", 1)[0]
        return sql.strip()

    def _decompose_question(self, question: str, source_name: str,
                            schema: str) -> Optional[str]:
        """質問をデータソースごとのサブ質問に分解"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            messages=[{"role": "user", "content": f"""
元の質問: {question}

データソース「{source_name}」のスキーマ:
{schema[:500]}

このデータソースで回答可能な部分があれば、
そのサブ質問を出力してください。
回答不可能なら「SKIP」と出力:
"""}]
        )
        result = response.content[0].text.strip()
        return None if result == "SKIP" else result

    def _interpret(self, question: str, result: QueryResult) -> str:
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": f"""
質問: {question}
SQL: {result.query}
結果: カラム={result.columns}, 行数={result.row_count}
先頭5行: {result.rows[:5]}

結果を日本語で解釈してください:
"""}]
        )
        return response.content[0].text

    def _unify_results(self, question: str,
                       partial_results: dict[str, QueryResult]) -> str:
        summaries = []
        for name, result in partial_results.items():
            summaries.append(
                f"[{name}] {result.summary()}\n先頭3行: {result.rows[:3]}"
            )

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": f"""
質問: {question}

各データソースからの結果:
{chr(10).join(summaries)}

複数のデータソースの結果を統合して、
包括的な回答を日本語で作成してください:
"""}]
        )
        return response.content[0].text
```

---

## 5. 安全性設計

```
データエージェントのセキュリティ層

Layer 1: クエリ生成制限
  └── SELECT文のみ生成するようプロンプトで制約

Layer 2: SQLバリデーション
  └── 生成されたSQLの構文を静的解析

Layer 3: DB接続制限
  └── READ ONLYユーザーで接続

Layer 4: 結果サイズ制限
  └── 最大行数、最大カラム数を制限

Layer 5: 機密データマスキング
  └── PII（個人情報）を除外

Layer 6: クエリレート制限
  └── 単位時間あたりのクエリ数を制限

Layer 7: 監査ログ
  └── 全クエリの記録と異常検知
```

```python
# 多層セキュリティの実装
import re
import logging
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)

class SecureDataAgent:
    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.max_rows = 1000
        self.max_columns = 50
        self.pii_columns = {"email", "phone", "ssn", "credit_card", "address", "password"}
        self.validator = SQLValidator()
        self._rate_limiter = RateLimiter(max_queries=100, window_seconds=3600)
        self._audit_log: list[dict] = []

    def execute_safely(self, sql: str, user_id: str = "anonymous") -> dict:
        """多層セキュリティを適用してクエリを実行"""
        start_time = time.time()

        # Layer 0: レート制限チェック
        if not self._rate_limiter.allow(user_id):
            self._log_audit(user_id, sql, "RATE_LIMITED")
            return {"error": "レート制限に達しました。しばらくしてから再試行してください"}

        # Layer 1: 構文バリデーション
        is_valid, error = self.validator.validate(sql)
        if not is_valid:
            self._log_audit(user_id, sql, f"VALIDATION_FAILED: {error}")
            return {"error": f"SQLバリデーションエラー: {error}"}

        # Layer 2: SQLインジェクション検出
        if self._detect_injection(sql):
            self._log_audit(user_id, sql, "INJECTION_DETECTED")
            logger.warning(f"SQLインジェクション検出: user={user_id}, sql={sql[:100]}")
            return {"error": "不正なクエリパターンが検出されました"}

        # Layer 3: READ ONLY接続
        conn = self._get_readonly_connection()

        try:
            cursor = conn.cursor()

            # Layer 4: クエリタイムアウト設定
            cursor.execute(f"SET statement_timeout = '30s'")
            cursor.execute(sql)

            # Layer 5: 結果サイズ制限
            columns = [desc[0] for desc in cursor.description]
            if len(columns) > self.max_columns:
                self._log_audit(user_id, sql, "TOO_MANY_COLUMNS")
                return {"error": f"カラム数が多すぎます（{len(columns)}、最大{self.max_columns}）"}

            results = cursor.fetchmany(self.max_rows)
            truncated = cursor.fetchone() is not None

            # Layer 6: PIIマスキング
            masked_results = self._mask_pii(columns, results)

            elapsed = (time.time() - start_time) * 1000
            self._log_audit(user_id, sql, "SUCCESS", elapsed)

            return {
                "columns": columns,
                "rows": masked_results,
                "row_count": len(masked_results),
                "truncated": truncated,
                "execution_time_ms": elapsed
            }
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            self._log_audit(user_id, sql, f"ERROR: {str(e)}", elapsed)
            return {"error": str(e)}
        finally:
            conn.close()

    def _detect_injection(self, sql: str) -> bool:
        """SQLインジェクションパターンの検出"""
        injection_patterns = [
            r";\s*(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE)",  # 複数文+危険操作
            r"UNION\s+ALL\s+SELECT\s+NULL",                  # UNION NULLパターン
            r"'\s*OR\s+'1'\s*=\s*'1",                        # OR 1=1
            r"'\s*OR\s+1\s*=\s*1",                           # OR 1=1(数値)
            r"CHAR\s*\(\s*\d+\s*\)",                         # CHAR関数
            r"0x[0-9a-fA-F]+",                               # 16進リテラル
            r"INFORMATION_SCHEMA",                            # メタデータアクセス
            r"pg_catalog",                                    # PostgreSQLカタログ
            r"sqlite_master",                                 # SQLiteマスターテーブル
        ]
        for pattern in injection_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                return True
        return False

    def _get_readonly_connection(self):
        """読み取り専用のDB接続を返す"""
        import psycopg2
        conn = psycopg2.connect(
            host=self.db_config["host"],
            port=self.db_config.get("port", 5432),
            dbname=self.db_config["dbname"],
            user=self.db_config.get("readonly_user", "readonly"),
            password=self.db_config.get("readonly_password", ""),
        )
        conn.set_session(readonly=True, autocommit=True)
        return conn

    def _mask_pii(self, columns: list, rows: list) -> list:
        """個人情報カラムをマスク"""
        pii_indices = {
            i for i, col in enumerate(columns)
            if col.lower() in self.pii_columns
            or any(pii in col.lower() for pii in self.pii_columns)
        }
        if not pii_indices:
            return rows

        return [
            tuple(
                "***MASKED***" if i in pii_indices else val
                for i, val in enumerate(row)
            )
            for row in rows
        ]

    def _log_audit(self, user_id: str, sql: str, status: str,
                   execution_time_ms: float = 0):
        """監査ログの記録"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "sql": sql[:500],
            "status": status,
            "execution_time_ms": execution_time_ms
        }
        self._audit_log.append(entry)
        logger.info(f"AUDIT: {status} user={user_id} time={execution_time_ms:.1f}ms")

    def get_audit_summary(self, hours: int = 24) -> dict:
        """監査ログのサマリー"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [
            e for e in self._audit_log
            if datetime.fromisoformat(e["timestamp"]) > cutoff
        ]
        status_counts = defaultdict(int)
        for entry in recent:
            status_counts[entry["status"]] += 1
        return {
            "total_queries": len(recent),
            "status_counts": dict(status_counts),
            "unique_users": len(set(e["user_id"] for e in recent))
        }


class RateLimiter:
    """スライディングウィンドウ方式のレート制限"""

    def __init__(self, max_queries: int = 100, window_seconds: int = 3600):
        self.max_queries = max_queries
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def allow(self, user_id: str) -> bool:
        """リクエストを許可するか判定"""
        now = time.time()
        cutoff = now - self.window_seconds

        # 期限切れのリクエストを削除
        self._requests[user_id] = [
            t for t in self._requests[user_id] if t > cutoff
        ]

        if len(self._requests[user_id]) >= self.max_queries:
            return False

        self._requests[user_id].append(now)
        return True

    def remaining(self, user_id: str) -> int:
        """残りのリクエスト数"""
        now = time.time()
        cutoff = now - self.window_seconds
        current = len([t for t in self._requests.get(user_id, []) if t > cutoff])
        return max(0, self.max_queries - current)
```

---

## 6. キャッシュとパフォーマンス最適化

### 6.1 クエリキャッシュ

```python
import hashlib
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class CacheEntry:
    """キャッシュエントリ"""
    result: QueryResult
    created_at: float
    ttl_seconds: float
    access_count: int = 0
    last_accessed: float = 0

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds


class QueryCache:
    """階層キャッシュ: メモリ → ディスク"""

    def __init__(self, max_memory_entries: int = 100,
                 disk_cache_dir: Optional[str] = None,
                 default_ttl: float = 300):
        self._memory: dict[str, CacheEntry] = {}
        self.max_memory = max_memory_entries
        self.disk_dir = Path(disk_cache_dir) if disk_cache_dir else None
        self.default_ttl = default_ttl
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

        if self.disk_dir:
            self.disk_dir.mkdir(parents=True, exist_ok=True)

    def get(self, sql: str) -> Optional[QueryResult]:
        """キャッシュからクエリ結果を取得"""
        key = self._key(sql)

        # メモリキャッシュを確認
        if key in self._memory:
            entry = self._memory[key]
            if not entry.is_expired:
                entry.access_count += 1
                entry.last_accessed = time.time()
                self._stats["hits"] += 1
                return entry.result
            else:
                del self._memory[key]

        # ディスクキャッシュを確認
        if self.disk_dir:
            disk_path = self.disk_dir / f"{key}.pkl"
            if disk_path.exists():
                try:
                    with open(disk_path, "rb") as f:
                        entry = pickle.load(f)
                    if not entry.is_expired:
                        # メモリに昇格
                        self._memory[key] = entry
                        entry.access_count += 1
                        entry.last_accessed = time.time()
                        self._stats["hits"] += 1
                        return entry.result
                    else:
                        disk_path.unlink()
                except Exception:
                    disk_path.unlink(missing_ok=True)

        self._stats["misses"] += 1
        return None

    def put(self, sql: str, result: QueryResult,
            ttl: Optional[float] = None):
        """クエリ結果をキャッシュに保存"""
        key = self._key(sql)
        entry = CacheEntry(
            result=result,
            created_at=time.time(),
            ttl_seconds=ttl or self.default_ttl,
            last_accessed=time.time()
        )

        # メモリキャッシュに保存
        if len(self._memory) >= self.max_memory:
            self._evict()
        self._memory[key] = entry

        # ディスクキャッシュにも保存
        if self.disk_dir:
            disk_path = self.disk_dir / f"{key}.pkl"
            with open(disk_path, "wb") as f:
                pickle.dump(entry, f)

    def invalidate(self, sql: str):
        """特定のキャッシュを無効化"""
        key = self._key(sql)
        self._memory.pop(key, None)
        if self.disk_dir:
            (self.disk_dir / f"{key}.pkl").unlink(missing_ok=True)

    def clear(self):
        """全キャッシュをクリア"""
        self._memory.clear()
        if self.disk_dir:
            for f in self.disk_dir.glob("*.pkl"):
                f.unlink()

    def stats(self) -> dict:
        """キャッシュ統計"""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0
        return {
            **self._stats,
            "hit_rate": f"{hit_rate:.1%}",
            "memory_entries": len(self._memory),
        }

    def _key(self, sql: str) -> str:
        normalized = " ".join(sql.strip().lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _evict(self):
        """LRU方式でキャッシュを削除"""
        if not self._memory:
            return
        oldest_key = min(
            self._memory, key=lambda k: self._memory[k].last_accessed
        )
        del self._memory[oldest_key]
        self._stats["evictions"] += 1
```

### 6.2 クエリ最適化

```python
class QueryOptimizer:
    """生成されたSQLクエリの最適化"""

    def optimize(self, sql: str, schema: str) -> str:
        """SQLクエリを最適化"""
        optimizations = [
            self._add_limit_if_missing,
            self._optimize_select_star,
            self._suggest_index_hints,
        ]

        optimized = sql
        for opt in optimizations:
            optimized = opt(optimized, schema)

        return optimized

    def _add_limit_if_missing(self, sql: str, schema: str) -> str:
        """LIMITがない場合に追加"""
        sql_upper = sql.upper().strip()
        if "LIMIT" not in sql_upper and "GROUP BY" not in sql_upper:
            if not sql.strip().endswith(";"):
                return f"{sql}\nLIMIT 1000"
            else:
                return f"{sql[:-1]}\nLIMIT 1000;"
        return sql

    def _optimize_select_star(self, sql: str, schema: str) -> str:
        """SELECT * を必要なカラムに限定（ログ出力のみ）"""
        if "SELECT *" in sql.upper() or "SELECT  *" in sql.upper():
            logger.info(
                "パフォーマンス警告: SELECT * の使用を検出。"
                "必要なカラムのみを指定することを推奨します。"
            )
        return sql

    def _suggest_index_hints(self, sql: str, schema: str) -> str:
        """インデックス利用のヒントを出力"""
        where_match = re.search(r'WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|$)',
                                sql, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1)
            columns = re.findall(r'(\w+)\s*[=<>!]', where_clause)
            if columns:
                logger.info(
                    f"推奨インデックス: WHERE句で使用されるカラム {columns} "
                    "にインデックスがあるか確認してください。"
                )
        return sql
```

---

## 7. 比較表

### 7.1 Text-to-SQLアプローチ比較

| アプローチ | 精度 | 柔軟性 | 実装コスト | 安全性 | スケーラビリティ |
|-----------|------|--------|-----------|--------|--------------|
| 直接SQL生成 | 中 | 高 | 低 | 低 | 中 |
| テンプレートベース | 高 | 低 | 中 | 高 | 低 |
| Few-shot + 検証 | 高 | 中 | 中 | 中 | 高 |
| Self-correction | 最高 | 高 | 高 | 中 | 高 |
| パース→SQL | 高 | 中 | 高 | 高 | 中 |
| 動的スキーマ選択 | 高 | 高 | 高 | 高 | 最高 |

### 7.2 データ分析ツール比較

| ツール | 対話的 | SQL不要 | 可視化 | コスト | カスタマイズ性 |
|--------|--------|---------|--------|--------|-------------|
| データエージェント | はい | はい | 自動 | API費用 | 高 |
| Jupyter Notebook | 手動 | いいえ | 手動コード | 無料 | 最高 |
| Tableau / Looker | GUI | はい | ドラッグ&ドロップ | 高額 | 中 |
| pandas + matplotlib | 手動 | いいえ | 手動コード | 無料 | 最高 |
| Metabase | GUI | はい | テンプレート | 無料/有料 | 低 |
| Streamlit + LLM | はい | はい | コード生成 | API費用 | 高 |

### 7.3 データソース別の接続方式

| データソース | 接続方式 | READ ONLY設定 | 備考 |
|-------------|---------|--------------|------|
| SQLite | ファイルパス | PRAGMA query_only | ローカル開発向け |
| PostgreSQL | psycopg2 | SET SESSION READ ONLY | 本番推奨 |
| MySQL | mysql-connector | READ ONLYユーザー | 権限設定が必要 |
| BigQuery | google-cloud-bigquery | IAMロール | コスト注意（スキャン課金） |
| Snowflake | snowflake-connector | WAREHOUSE READONLY | クレジット消費に注意 |
| DuckDB | duckdb | access_mode='read_only' | 分析特化、高速 |

---

## 8. エラーハンドリングと自己修正

### 8.1 基本の自己修正パターン

```python
# SQL生成の自己修正パターン
class SelfCorrectingAgent(TextToSQLAgent):
    def query_with_retry(self, question: str, max_retries: int = 3) -> dict:
        """SQLエラー時に自己修正して再試行"""
        sql = self._generate_sql(question)
        errors_history = []

        for attempt in range(max_retries):
            if not self._is_safe_query(sql):
                return {"error": "安全でないクエリ"}

            results = self._execute_sql(sql)

            if "error" not in results:
                return {
                    "sql": sql,
                    "results": results,
                    "attempts": attempt + 1,
                    "errors_history": errors_history
                }

            # エラー履歴を蓄積
            errors_history.append({
                "attempt": attempt + 1,
                "sql": sql,
                "error": results["error"]
            })

            # エラーからの自己修正
            sql = self._fix_sql(question, sql, results["error"], errors_history)

        return {
            "error": f"{max_retries}回試行後も失敗",
            "errors_history": errors_history
        }

    def _fix_sql(self, question: str, bad_sql: str, error: str,
                 history: list[dict]) -> str:
        """エラーメッセージに基づいてSQLを修正"""
        history_text = ""
        if len(history) > 1:
            history_text = "\n過去の試行:\n"
            for h in history[:-1]:
                history_text += f"  SQL: {h['sql']}\n  エラー: {h['error']}\n"

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": f"""
以下のSQLがエラーになりました。修正してください。

スキーマ: {self.schema}
元の質問: {question}
失敗したSQL: {bad_sql}
エラー: {error}
{history_text}
重要: 過去に失敗した同じSQLを生成しないでください。
修正したSQLのみ出力:
"""}]
        )
        return response.content[0].text.strip()
```

### 8.2 段階的クエリ分解

```python
class DecomposingAgent:
    """複雑な質問を段階的に分解して実行"""

    def __init__(self, db_path: str):
        self.client = anthropic.Anthropic()
        self.agent = TextToSQLAgent(db_path)

    def query_complex(self, question: str) -> dict:
        """複雑な質問を分解して順次実行"""

        # Step 1: 質問の複雑度を判定
        complexity = self._assess_complexity(question)

        if complexity == "simple":
            return self.agent.query(question)

        # Step 2: サブ質問に分解
        sub_questions = self._decompose(question)

        # Step 3: 各サブ質問を順次実行
        sub_results = []
        for i, sq in enumerate(sub_questions):
            result = self.agent.query(sq)
            sub_results.append({
                "sub_question": sq,
                "result": result,
                "step": i + 1
            })

        # Step 4: 結果を統合
        final_answer = self._synthesize(question, sub_results)

        return {
            "question": question,
            "complexity": complexity,
            "sub_results": sub_results,
            "final_answer": final_answer
        }

    def _assess_complexity(self, question: str) -> str:
        """質問の複雑度を判定"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=32,
            messages=[{"role": "user", "content": f"""
以下の質問を1つのSQLで回答できますか？

質問: {question}

回答: 「simple」または「complex」のみ
"""}]
        )
        return response.content[0].text.strip().lower()

    def _decompose(self, question: str) -> list[str]:
        """質問をサブ質問に分解"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{"role": "user", "content": f"""
以下の複雑な質問を、それぞれ1つのSQLで回答できるサブ質問に分解してください。

質問: {question}

データベーススキーマ: {self.agent.schema}

サブ質問を番号付きリストで出力（各行1つのサブ質問）:
"""}]
        )
        lines = response.content[0].text.strip().split("\n")
        return [
            re.sub(r'^\d+[\.\)]\s*', '', line).strip()
            for line in lines
            if line.strip() and re.match(r'^\d+', line.strip())
        ]

    def _synthesize(self, question: str, sub_results: list[dict]) -> str:
        """サブ結果を統合して最終回答を生成"""
        results_text = ""
        for sr in sub_results:
            results_text += f"""
Step {sr['step']}: {sr['sub_question']}
結果: {json.dumps(sr['result'].get('results', {}), ensure_ascii=False, default=str)[:500]}
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": f"""
元の質問: {question}

各ステップの結果:
{results_text}

これらの結果を統合して、元の質問に対する包括的な回答を作成してください:
"""}]
        )
        return response.content[0].text
```

---

## 9. 本番運用パターン

### 9.1 Streamlitベースのデータ分析UI

```python
# streamlit_app.py
import streamlit as st

def main():
    st.set_page_config(page_title="データ分析エージェント", layout="wide")
    st.title("データ分析エージェント")

    # サイドバー: 設定
    with st.sidebar:
        st.header("設定")
        db_path = st.text_input("データベースパス", "data/sample.db")
        model = st.selectbox("モデル", [
            "claude-sonnet-4-20250514",
            "claude-haiku-4-20250514"
        ])
        max_rows = st.slider("最大行数", 10, 1000, 100)

    # エージェントの初期化
    if "agent" not in st.session_state:
        config = DataAgentConfig(
            role=AgentRole.QUERY,
            db_connections={"main": db_path},
            max_rows=max_rows,
            model_name=model
        )
        st.session_state.agent = SelfCorrectingAgent(db_path, config)
        st.session_state.history = []

    # チャット履歴の表示
    for entry in st.session_state.history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["interpretation"])
            if entry.get("sql"):
                with st.expander("実行SQL"):
                    st.code(entry["sql"], language="sql")
            if entry.get("chart_path"):
                st.image(entry["chart_path"])

    # 入力
    question = st.chat_input("データについて質問してください")
    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("分析中..."):
                result = st.session_state.agent.query_with_retry(question)

            if "error" in result:
                st.error(result["error"])
            else:
                st.write(result.get("interpretation", ""))

                with st.expander("実行SQL"):
                    st.code(result["sql"], language="sql")

                # 結果テーブル
                if result.get("results", {}).get("rows"):
                    import pandas as pd
                    df = pd.DataFrame(
                        result["results"]["rows"],
                        columns=result["results"]["columns"]
                    )
                    st.dataframe(df, use_container_width=True)

                    # 自動可視化
                    visualizer = DataVisualizer()
                    chart_type = visualizer.auto_visualize(result["results"])
                    st.info(f"推奨グラフ: {chart_type}")

                # 履歴に追加
                st.session_state.history.append({
                    "question": question,
                    "sql": result.get("sql"),
                    "interpretation": result.get("interpretation", ""),
                })

if __name__ == "__main__":
    main()
```

### 9.2 APIサーバー

```python
# FastAPIベースのデータ分析APIサーバー
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

app = FastAPI(title="Data Agent API", version="1.0.0")

class QueryRequest(BaseModel):
    question: str = Field(..., description="自然言語の質問")
    db_name: str = Field(default="main", description="データベース名")
    max_rows: int = Field(default=100, le=1000, description="最大行数")
    enable_visualization: bool = Field(default=False, description="可視化を含む")

class QueryResponse(BaseModel):
    question: str
    sql: str
    columns: list[str]
    rows: list[list]
    row_count: int
    interpretation: str
    execution_time_ms: float
    cached: bool = False
    chart_url: Optional[str] = None

class AnalysisRequest(BaseModel):
    topic: str = Field(..., description="分析トピック")
    db_name: str = Field(default="main")
    analysis_type: str = Field(
        default="comprehensive",
        description="分析タイプ: comprehensive, trend, anomaly"
    )

# グローバルエージェントインスタンス
agents: dict[str, SecureDataAgent] = {}
cache = QueryCache(max_memory_entries=500, default_ttl=300)

@app.on_event("startup")
async def startup():
    """起動時にエージェントを初期化"""
    db_configs = {
        "main": {"host": "localhost", "dbname": "analytics", "readonly_user": "reader"},
        "logs": {"host": "localhost", "dbname": "logs", "readonly_user": "reader"},
    }
    for name, config in db_configs.items():
        agents[name] = SecureDataAgent(config)

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """自然言語でデータを問い合わせ"""
    if request.db_name not in agents:
        raise HTTPException(404, f"データベース '{request.db_name}' が見つかりません")

    agent = agents[request.db_name]

    # キャッシュチェック（SQLが不明なため質問ベース）
    cache_key = f"{request.db_name}:{request.question}"
    cached_result = cache.get(cache_key)
    if cached_result:
        return QueryResponse(
            question=request.question,
            sql=cached_result.query,
            columns=cached_result.columns,
            rows=[list(r) for r in cached_result.rows],
            row_count=cached_result.row_count,
            interpretation="（キャッシュ結果）",
            execution_time_ms=0,
            cached=True
        )

    try:
        result = agent.execute_safely(request.question)
        if "error" in result:
            raise HTTPException(400, result["error"])

        return QueryResponse(
            question=request.question,
            sql=result.get("sql", ""),
            columns=result["columns"],
            rows=result["rows"],
            row_count=result["row_count"],
            interpretation=result.get("interpretation", ""),
            execution_time_ms=result["execution_time_ms"],
        )
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    """包括的なデータ分析を実行"""
    pipeline = AnalysisPipeline()
    try:
        if request.analysis_type == "comprehensive":
            result = pipeline.comprehensive_analysis(
                request.db_name, request.topic
            )
        elif request.analysis_type == "anomaly":
            result = pipeline.anomaly_analysis(
                request.db_name, request.topic
            )
        else:
            raise HTTPException(400, f"未対応の分析タイプ: {request.analysis_type}")

        return result
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "databases": list(agents.keys())}

@app.get("/cache/stats")
async def cache_stats():
    return cache.stats()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 9.3 Slackボット統合

```python
# Slackボットとしてデータエージェントを運用
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

slack_app = App(token="xoxb-your-bot-token")
agent = SelfCorrectingAgent("data/analytics.db")

@slack_app.event("app_mention")
def handle_mention(event, say):
    """メンション時にデータ分析を実行"""
    question = event["text"].split(">", 1)[-1].strip()
    user = event["user"]

    if not question:
        say("質問を入力してください。例: @DataBot 先月の売上は？")
        return

    say(f"<@{user}> 分析中です... :hourglass:")

    try:
        result = agent.query_with_retry(question)

        if "error" in result:
            say(f"<@{user}> エラー: {result['error']}")
            return

        # 結果をフォーマット
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*質問:* {question}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*回答:*\n{result.get('interpretation', '結果なし')}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"```sql\n{result.get('sql', 'N/A')}\n```"
                }
            }
        ]

        # テーブル形式の結果（10行まで）
        results_data = result.get("results", {})
        if results_data.get("rows"):
            columns = results_data["columns"]
            rows = results_data["rows"][:10]
            table = " | ".join(columns) + "\n"
            table += " | ".join(["---"] * len(columns)) + "\n"
            for row in rows:
                table += " | ".join(str(v) for v in row) + "\n"
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"```\n{table}\n```"}
            })

        say(blocks=blocks)

    except Exception as e:
        say(f"<@{user}> 予期しないエラーが発生しました: {str(e)}")

@slack_app.command("/data-query")
def handle_command(ack, say, command):
    """スラッシュコマンドでのクエリ実行"""
    ack()
    question = command["text"]
    user = command["user_id"]

    result = agent.query_with_retry(question)
    if "error" in result:
        say(f"<@{user}> エラー: {result['error']}")
    else:
        say(f"<@{user}>\n{result.get('interpretation', '')}")

if __name__ == "__main__":
    handler = SocketModeHandler(slack_app, "xapp-your-app-token")
    handler.start()
```

---

## 10. モニタリングとコスト管理

### 10.1 コスト追跡

```python
class CostTracker:
    """API呼び出しコストの追跡"""

    # Anthropic APIの料金（2025年時点の概算、USD）
    PRICING = {
        "claude-sonnet-4-20250514": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
        "claude-haiku-4-20250514": {"input": 0.25 / 1_000_000, "output": 1.25 / 1_000_000},
    }

    def __init__(self):
        self._sessions: dict[str, list[dict]] = defaultdict(list)

    def record(self, session_id: str, model: str,
               input_tokens: int, output_tokens: int):
        """API呼び出しを記録"""
        pricing = self.PRICING.get(model, {"input": 0, "output": 0})
        cost = (input_tokens * pricing["input"] +
                output_tokens * pricing["output"])

        self._sessions[session_id].append({
            "timestamp": time.time(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost
        })

    def session_cost(self, session_id: str) -> float:
        """セッションの合計コスト"""
        return sum(e["cost_usd"] for e in self._sessions.get(session_id, []))

    def daily_report(self) -> dict:
        """日次コストレポート"""
        today = datetime.now().date()
        today_start = datetime.combine(today, datetime.min.time()).timestamp()

        total_cost = 0
        total_calls = 0
        model_costs: dict[str, float] = defaultdict(float)

        for session_id, entries in self._sessions.items():
            for entry in entries:
                if entry["timestamp"] >= today_start:
                    total_cost += entry["cost_usd"]
                    total_calls += 1
                    model_costs[entry["model"]] += entry["cost_usd"]

        return {
            "date": str(today),
            "total_cost_usd": round(total_cost, 4),
            "total_api_calls": total_calls,
            "model_breakdown": {
                k: round(v, 4) for k, v in model_costs.items()
            },
            "active_sessions": len(self._sessions)
        }
```

### 10.2 パフォーマンスモニタリング

```python
class PerformanceMonitor:
    """クエリパフォーマンスの監視"""

    def __init__(self):
        self._metrics: list[dict] = []

    def record_query(self, sql: str, execution_time_ms: float,
                     row_count: int, success: bool):
        """クエリメトリクスを記録"""
        self._metrics.append({
            "timestamp": time.time(),
            "sql_length": len(sql),
            "execution_time_ms": execution_time_ms,
            "row_count": row_count,
            "success": success
        })

    def get_summary(self, last_n: int = 100) -> dict:
        """パフォーマンスサマリー"""
        recent = self._metrics[-last_n:]
        if not recent:
            return {"message": "メトリクスなし"}

        times = [m["execution_time_ms"] for m in recent]
        success_count = sum(1 for m in recent if m["success"])

        return {
            "total_queries": len(recent),
            "success_rate": f"{success_count / len(recent):.1%}",
            "avg_execution_ms": round(sum(times) / len(times), 1),
            "p50_execution_ms": round(sorted(times)[len(times) // 2], 1),
            "p95_execution_ms": round(sorted(times)[int(len(times) * 0.95)], 1),
            "max_execution_ms": round(max(times), 1),
            "avg_row_count": round(
                sum(m["row_count"] for m in recent) / len(recent), 1
            ),
        }

    def slow_queries(self, threshold_ms: float = 5000) -> list[dict]:
        """スロークエリの検出"""
        return [
            m for m in self._metrics
            if m["execution_time_ms"] > threshold_ms
        ]
```

---

## 11. アンチパターン

### アンチパターン1: 全データの取得

```python
# NG: テーブル全体をLLMに渡す
sql = "SELECT * FROM orders"  # 100万行!
results = execute(sql)
llm.analyze(results)  # コンテキスト超過

# OK: 集計してから分析
sql = """
SELECT
    DATE(ordered_at) as date,
    COUNT(*) as order_count,
    SUM(total_price) as revenue
FROM orders
GROUP BY DATE(ordered_at)
ORDER BY date DESC
LIMIT 30
"""
```

### アンチパターン2: スキーマ情報なしでのSQL生成

```python
# NG: スキーマを渡さずにSQL生成
llm.generate("売上を教えて")  # テーブル名もカラム名も知らない → ハルシネーション

# OK: スキーマ + サンプルデータを必ず提供
llm.generate(f"""
スキーマ: {schema}
サンプル行: {sample_rows}
質問: 売上を教えて
""")
```

### アンチパターン3: 書き込み権限での接続

```python
# NG: 管理者権限でデータエージェントを接続
conn = psycopg2.connect(
    user="admin",      # 全権限あり
    password="secret",
    dbname="production"
)

# OK: 読み取り専用ユーザーで接続
conn = psycopg2.connect(
    user="readonly_agent",  # SELECT権限のみ
    password="readonly_pass",
    dbname="production"
)
conn.set_session(readonly=True)
```

### アンチパターン4: エラーハンドリングなしの本番運用

```python
# NG: エラーを握りつぶす
def query(question):
    sql = generate_sql(question)
    return execute(sql)  # エラー時にクラッシュ

# OK: 多層のエラーハンドリング
def query_safely(question):
    try:
        sql = generate_sql(question)
        is_valid, error = validator.validate(sql)
        if not is_valid:
            return {"error": f"バリデーション失敗: {error}", "sql": sql}

        result = execute_with_timeout(sql, timeout=30)
        if result.error:
            # 自己修正を試行
            corrected = self_correct(question, sql, result.error)
            result = execute_with_timeout(corrected, timeout=30)

        return result
    except TimeoutError:
        return {"error": "クエリがタイムアウトしました（30秒）"}
    except Exception as e:
        logger.error(f"予期しないエラー: {e}", exc_info=True)
        return {"error": "内部エラーが発生しました"}
```

### アンチパターン5: キャッシュなしの同一クエリ繰り返し

```python
# NG: 同じ質問を毎回LLMに送信
for department in departments:
    # 50部門 × (SQL生成 + 解釈) = 100回のAPI呼び出し
    result = agent.query(f"{department}の売上は？")

# OK: パラメータ化 + キャッシュ
template_sql = agent.generate_sql("各部門の売上は？")
# → SELECT department, SUM(revenue) FROM sales GROUP BY department
result = agent.execute(template_sql)  # 1回のDB呼び出しで全部門取得
```

---

## 12. 実務シナリオ別ガイド

### シナリオ1: ECサイトの売上分析

```python
# ECサイト分析エージェントの具体的な使用例
class ECommerceAnalyst:
    """ECサイト専用の分析エージェント"""

    def __init__(self, db_path: str):
        self.agent = SelfCorrectingAgent(db_path)
        self.visualizer = DataVisualizer()

    def daily_report(self) -> dict:
        """日次売上レポートの自動生成"""
        queries = {
            "summary": "本日の売上合計、注文数、平均注文額",
            "hourly": "本日の時間帯別売上推移",
            "top_products": "本日の売上トップ10商品",
            "categories": "本日のカテゴリ別売上構成比",
            "comparison": "前日比と前週同日比の売上比較",
            "new_customers": "本日の新規顧客数と売上",
            "cancellations": "本日のキャンセル件数と金額"
        }

        results = {}
        for key, question in queries.items():
            results[key] = self.agent.query_with_retry(question)

        return results

    def customer_cohort_analysis(self, months: int = 6) -> dict:
        """顧客コホート分析"""
        return self.agent.query_with_retry(f"""
過去{months}ヶ月の月別顧客コホート分析:
各月に初回購入した顧客グループが、
その後の各月にどれだけリピート購入したかを
月別のリテンション率で表示
""")

    def product_recommendation_data(self, product_id: int) -> dict:
        """商品レコメンデーション用データ"""
        return self.agent.query_with_retry(f"""
商品ID {product_id} と一緒に購入されることが多い商品トップ10
（同じ注文に含まれている商品を集計）
""")
```

### シナリオ2: SaaS指標ダッシュボード

```python
class SaaSMetricsAgent:
    """SaaS KPI自動分析エージェント"""

    def __init__(self, db_path: str):
        self.agent = SelfCorrectingAgent(db_path)

    def calculate_mrr(self) -> dict:
        """月次経常収益（MRR）の計算"""
        return self.agent.query_with_retry("""
今月のMRR（Monthly Recurring Revenue）を計算:
- 新規MRR: 今月新規契約の月額合計
- 拡大MRR: 今月アップグレードした顧客の差額合計
- 縮小MRR: 今月ダウングレードした顧客の差額合計
- 解約MRR: 今月解約した顧客の月額合計
- 純MRR: 新規 + 拡大 - 縮小 - 解約
""")

    def churn_analysis(self) -> dict:
        """解約分析"""
        return self.agent.query_with_retry("""
過去12ヶ月の月別解約率と解約理由の分析:
- 月別の解約顧客数と解約率
- 解約理由のカテゴリ別内訳
- プラン別の解約率比較
- 解約前の利用状況（最終ログイン日からの日数）
""")

    def ltv_analysis(self) -> dict:
        """顧客生涯価値（LTV）分析"""
        return self.agent.query_with_retry("""
プラン別の顧客LTV（Life Time Value）分析:
- 各プランの平均契約期間（月数）
- 各プランの月額料金
- 計算LTV = 月額料金 × 平均契約期間
- プラン別の顧客数
""")
```

---

## 13. FAQ

### Q1: 大規模データベース（100+テーブル）でのText-to-SQLの精度は？

テーブル数が多い場合、すべてのスキーマをプロンプトに含めるとノイズが増えて精度が下がる。対策:
- **関連テーブルの自動選択**: 質問から関連テーブルを2段階で絞り込み（1段目: キーワードマッチ、2段目: LLM選択）
- **スキーマ要約**: テーブルのdescriptionのみ先に渡し、選択後にCREATE TABLE文を渡す
- **SchemaSelector クラス**: セクション2.3で実装した動的スキーマ選択を使用

### Q2: リアルタイムダッシュボードへの応用は？

データエージェントはアドホック分析に適しているが、リアルタイムダッシュボードには向かない（レイテンシ+コスト）。推奨アプローチ:
- **アドホック分析**: データエージェント
- **定期レポート**: エージェントで一度クエリを生成 → 定期実行に移行
- **リアルタイム**: 従来のBIツール（Metabase等）

### Q3: データの鮮度をどう保証する？

- **タイムスタンプ付きの回答**: 「このデータは2025年1月31日時点のものです」
- **データ更新日時の確認**: メタデータテーブルで最終更新日を確認
- **キャッシュ有効期限**: 同じクエリのキャッシュに有効期限を設定

### Q4: Text-to-SQLの精度を測定するには？

```python
# SQL生成精度のベンチマーク
class SQLAccuracyBenchmark:
    def __init__(self, agent: TextToSQLAgent):
        self.agent = agent
        self.test_cases: list[dict] = []

    def add_test(self, question: str, expected_sql: str,
                 expected_result: list = None):
        self.test_cases.append({
            "question": question,
            "expected_sql": expected_sql,
            "expected_result": expected_result
        })

    def run(self) -> dict:
        correct = 0
        results = []
        for tc in self.test_cases:
            generated = self.agent._generate_sql(tc["question"])
            # 結果ベースの比較（SQL文字列ではなく実行結果で判定）
            gen_result = self.agent._execute_sql(generated)
            exp_result = self.agent._execute_sql(tc["expected_sql"])

            match = (gen_result.get("rows") == exp_result.get("rows"))
            if match:
                correct += 1
            results.append({
                "question": tc["question"],
                "generated_sql": generated,
                "expected_sql": tc["expected_sql"],
                "match": match
            })

        return {
            "total": len(self.test_cases),
            "correct": correct,
            "accuracy": f"{correct/len(self.test_cases):.1%}" if self.test_cases else "N/A",
            "details": results
        }
```

### Q5: コスト削減のベストプラクティスは？

| 戦略 | 効果 | 実装難易度 |
|------|------|-----------|
| クエリキャッシュ | 同一質問のAPI呼び出し削減 | 低 |
| 小さいモデルで事前分類 | SQL生成以外をHaikuで処理 | 低 |
| Few-shot例でプロンプト最適化 | トークン数削減+精度向上 | 中 |
| バッチ処理（定期レポート化） | リアルタイム呼び出し削減 | 中 |
| スキーマ圧縮 | 入力トークン削減 | 中 |

### Q6: 複数のデータベース方言への対応は？

```python
# データベース方言の抽象化
class SQLDialect:
    """SQL方言の差異を吸収"""

    DIALECTS = {
        "sqlite": {
            "current_date": "DATE('now')",
            "date_diff": "JULIANDAY({end}) - JULIANDAY({start})",
            "limit": "LIMIT {n}",
            "string_concat": "{a} || {b}",
        },
        "postgresql": {
            "current_date": "CURRENT_DATE",
            "date_diff": "({end}::date - {start}::date)",
            "limit": "LIMIT {n}",
            "string_concat": "{a} || {b}",
        },
        "mysql": {
            "current_date": "CURDATE()",
            "date_diff": "DATEDIFF({end}, {start})",
            "limit": "LIMIT {n}",
            "string_concat": "CONCAT({a}, {b})",
        },
    }

    @classmethod
    def get_prompt_hint(cls, dialect: str) -> str:
        """SQL方言のヒントをプロンプトに含める"""
        d = cls.DIALECTS.get(dialect, {})
        hints = [f"データベース: {dialect}"]
        for key, val in d.items():
            hints.append(f"  {key}: {val}")
        return "\n".join(hints)
```

---

## まとめ

| 項目 | 内容 |
|------|------|
| コアフロー | 質問理解→SQL生成→検証→実行→分析→可視化 |
| Text-to-SQL | スキーマ情報+質問→SELECT文を生成（Few-shot強化） |
| 安全性 | 7層防御: バリデーション→READ ONLY→PIIマスキング→レート制限→監査ログ |
| 可視化 | データ特性に応じた自動グラフ選択（matplotlib/Plotly） |
| 自己修正 | SQLエラー時にエラー履歴を含めて修正（最大3回） |
| 複数ソース | DataSource抽象化で異種DB横断分析 |
| キャッシュ | メモリ→ディスクの階層キャッシュ、LRU方式 |
| 本番運用 | Streamlit UI / FastAPI / Slackボット |
| 核心原則 | 集計してからLLMに渡す。全データを読まない |

## 次に読むべきガイド

- [../04-production/00-deployment.md](../04-production/00-deployment.md) -- データエージェントのデプロイ
- [../00-fundamentals/03-memory-systems.md](../00-fundamentals/03-memory-systems.md) -- RAGとベクトル検索
- [../01-patterns/02-workflow-agents.md](../01-patterns/02-workflow-agents.md) -- 分析ワークフロー
- [../02-implementation/04-evaluation.md](../02-implementation/04-evaluation.md) -- 評価とベンチマーク

## 参考文献

1. Rajkumar, N. et al., "Evaluating the Text-to-SQL Capabilities of Large Language Models" (2022) -- https://arxiv.org/abs/2204.00498
2. Pourreza, M. et al., "DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction" (2023) -- https://arxiv.org/abs/2304.11015
3. Vanna AI -- https://vanna.ai/
4. LangChain SQL Agent -- https://python.langchain.com/docs/use_cases/sql/
5. Anthropic Tool Use Documentation -- https://docs.anthropic.com/en/docs/build-with-claude/tool-use
