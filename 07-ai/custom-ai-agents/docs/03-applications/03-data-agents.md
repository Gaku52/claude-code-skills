# データエージェント

> 分析・可視化・洞察――データベースのクエリ、統計分析、グラフ生成を自律的に行うデータ分析エージェントの設計と実装。

## この章で学ぶこと

1. 自然言語からSQLへの変換（Text-to-SQL）とデータ分析パイプラインの設計
2. データ可視化の自動生成とインサイト抽出の実装パターン
3. データエージェントの安全性（読み取り専用、インジェクション防止）の確保

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

---

## 2. Text-to-SQL

### 2.1 基本実装

```python
# Text-to-SQL エージェント
import anthropic
import sqlite3
import json

class TextToSQLAgent:
    def __init__(self, db_path: str):
        self.client = anthropic.Anthropic()
        self.db_path = db_path
        self.schema = self._get_schema()

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

        # 3. 実行
        results = self._execute_sql(sql)

        # 4. 結果の解釈
        interpretation = self._interpret_results(question, sql, results)

        return {
            "question": question,
            "sql": sql,
            "results": results,
            "interpretation": interpretation
        }

    def _generate_sql(self, question: str) -> str:
        """自然言語をSQLに変換"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
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
            model="claude-sonnet-4-20250514",
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

---

## 3. データ可視化

### 3.1 グラフ自動生成

```python
# Pythonコード生成によるグラフ作成
class DataVisualizer:
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

### 3.2 分析パイプライン

```python
# 複数ステップの分析パイプライン
class AnalysisPipeline:
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

        # Step 5: 統合インサイト
        insights = self._generate_insights({
            "overview": overview,
            "trend": trend,
            "top_items": top_items,
            "distribution": distribution
        })

        return {
            "overview": overview,
            "trend": trend,
            "top_items": top_items,
            "distribution": distribution,
            "insights": insights
        }
```

---

## 4. 安全性設計

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
```

```python
# 多層セキュリティの実装
class SecureDataAgent:
    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.max_rows = 1000
        self.pii_columns = {"email", "phone", "ssn", "credit_card"}

    def execute_safely(self, sql: str) -> dict:
        # Layer 1: 構文チェック
        if not self._validate_syntax(sql):
            return {"error": "無効なSQL構文"}

        # Layer 2: 危険な操作の検出
        if self._has_dangerous_operations(sql):
            return {"error": "許可されていない操作"}

        # Layer 3: READ ONLY接続
        conn = self._get_readonly_connection()

        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchmany(self.max_rows)

            # Layer 4: PIIマスキング
            columns = [desc[0] for desc in cursor.description]
            masked_results = self._mask_pii(columns, results)

            return {
                "columns": columns,
                "rows": masked_results,
                "truncated": cursor.fetchone() is not None
            }
        finally:
            conn.close()

    def _mask_pii(self, columns: list, rows: list) -> list:
        """個人情報カラムをマスク"""
        pii_indices = {
            i for i, col in enumerate(columns)
            if col.lower() in self.pii_columns
        }
        return [
            tuple(
                "***MASKED***" if i in pii_indices else val
                for i, val in enumerate(row)
            )
            for row in rows
        ]
```

---

## 5. 比較表

### 5.1 Text-to-SQLアプローチ比較

| アプローチ | 精度 | 柔軟性 | 実装コスト | 安全性 |
|-----------|------|--------|-----------|--------|
| 直接SQL生成 | 中 | 高 | 低 | 低 |
| テンプレートベース | 高 | 低 | 中 | 高 |
| Few-shot + 検証 | 高 | 中 | 中 | 中 |
| Self-correction | 最高 | 高 | 高 | 中 |
| パース→SQL | 高 | 中 | 高 | 高 |

### 5.2 データ分析ツール比較

| ツール | 対話的 | SQL不要 | 可視化 | コスト |
|--------|--------|---------|--------|--------|
| データエージェント | はい | はい | 自動 | API費用 |
| Jupyter Notebook | 手動 | いいえ | 手動コード | 無料 |
| Tableau / Looker | GUI | はい | ドラッグ&ドロップ | 高額 |
| pandas + matplotlib | 手動 | いいえ | 手動コード | 無料 |
| Metabase | GUI | はい | テンプレート | 無料/有料 |

---

## 6. エラーハンドリングと自己修正

```python
# SQL生成の自己修正パターン
class SelfCorrectingAgent(TextToSQLAgent):
    def query_with_retry(self, question: str, max_retries: int = 3) -> dict:
        """SQLエラー時に自己修正して再試行"""
        sql = self._generate_sql(question)

        for attempt in range(max_retries):
            if not self._is_safe_query(sql):
                return {"error": "安全でないクエリ"}

            results = self._execute_sql(sql)

            if "error" not in results:
                return {
                    "sql": sql,
                    "results": results,
                    "attempts": attempt + 1
                }

            # エラーからの自己修正
            sql = self._fix_sql(question, sql, results["error"])

        return {"error": f"{max_retries}回試行後も失敗"}

    def _fix_sql(self, question: str, bad_sql: str, error: str) -> str:
        """エラーメッセージに基づいてSQLを修正"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": f"""
以下のSQLがエラーになりました。修正してください。

スキーマ: {self.schema}
元の質問: {question}
失敗したSQL: {bad_sql}
エラー: {error}

修正したSQLのみ出力:
"""}]
        )
        return response.content[0].text.strip()
```

---

## 7. アンチパターン

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

---

## 8. FAQ

### Q1: 大規模データベース（100+テーブル）でのText-to-SQLの精度は？

テーブル数が多い場合、すべてのスキーマをプロンプトに含めるとノイズが増えて精度が下がる。対策:
- **関連テーブルの自動選択**: 質問から関連テーブルを2段階で絞り込み（1段目: キーワードマッチ、2段目: LLM選択）
- **スキーマ要約**: テーブルのdescriptionのみ先に渡し、選択後にCREATE TABLE文を渡す

### Q2: リアルタイムダッシュボードへの応用は？

データエージェントはアドホック分析に適しているが、リアルタイムダッシュボードには向かない（レイテンシ+コスト）。推奨アプローチ:
- **アドホック分析**: データエージェント
- **定期レポート**: エージェントで一度クエリを生成 → 定期実行に移行
- **リアルタイム**: 従来のBIツール（Metabase等）

### Q3: データの鮮度をどう保証する？

- **タイムスタンプ付きの回答**: 「このデータは2025年1月31日時点のものです」
- **データ更新日時の確認**: メタデータテーブルで最終更新日を確認
- **キャッシュ有効期限**: 同じクエリのキャッシュに有効期限を設定

---

## まとめ

| 項目 | 内容 |
|------|------|
| コアフロー | 質問理解→SQL生成→実行→分析→可視化 |
| Text-to-SQL | スキーマ情報+質問→SELECT文を生成 |
| 安全性 | READ ONLY、SQLバリデーション、PIIマスキング |
| 可視化 | データ特性に応じた自動グラフ選択 |
| 自己修正 | SQLエラー時にエラーメッセージを基に修正 |
| 核心原則 | 集計してからLLMに渡す。全データを読まない |

## 次に読むべきガイド

- [../04-production/00-deployment.md](../04-production/00-deployment.md) — データエージェントのデプロイ
- [../00-fundamentals/03-memory-systems.md](../00-fundamentals/03-memory-systems.md) — RAGとベクトル検索
- [../01-patterns/02-workflow-agents.md](../01-patterns/02-workflow-agents.md) — 分析ワークフロー

## 参考文献

1. Rajkumar, N. et al., "Evaluating the Text-to-SQL Capabilities of Large Language Models" (2022) — https://arxiv.org/abs/2204.00498
2. Pourreza, M. et al., "DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction" (2023) — https://arxiv.org/abs/2304.11015
3. Vanna AI — https://vanna.ai/
