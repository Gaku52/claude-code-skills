# SAST/DAST

> 静的解析 (SAST) と動的解析 (DAST) の特性を理解し、SonarQube や OWASP ZAP を活用して CI/CD パイプラインにセキュリティテストを組み込むガイド

## この章で学ぶこと

1. **SAST の原理と実践** — ソースコードの静的解析による脆弱性の早期発見
2. **DAST の原理と実践** — 実行中アプリケーションに対する動的なセキュリティテスト
3. **CI/CD 統合** — セキュリティテストを開発フローに自然に組み込む方法
4. **IAST とハイブリッド手法** — SAST と DAST を補完する次世代アプローチ
5. **シークレットスキャン** — ソースコードに埋め込まれた秘密情報の検出
6. **運用ベストプラクティス** — トリアージ、チューニング、組織的導入戦略

---

## 1. SAST と DAST の全体像

### テスト手法の分類

```
+----------------------------------------------------------+
|            アプリケーションセキュリティテスト                 |
|----------------------------------------------------------|
|                                                          |
|  SAST (Static Application Security Testing)              |
|  +-- ソースコードを解析                                    |
|  +-- ビルド前に実行可能                                    |
|  +-- 行番号レベルで問題箇所を特定                           |
|  +-- 偽陽性が多い傾向                                     |
|                                                          |
|  DAST (Dynamic Application Security Testing)             |
|  +-- 実行中のアプリを外部からテスト                         |
|  +-- デプロイ後に実行                                     |
|  +-- 実際に悪用可能な脆弱性を発見                           |
|  +-- ソースコード不要 (ブラックボックス)                     |
|                                                          |
|  IAST (Interactive Application Security Testing)         |
|  +-- アプリ内にエージェントを埋め込み                       |
|  +-- リアルタイムで検出                                    |
|  +-- SAST + DAST のハイブリッド                            |
|                                                          |
|  SCA (Software Composition Analysis)                     |
|  +-- 依存ライブラリの脆弱性を検出                          |
|  +-- → 別章「依存関係セキュリティ」で詳述                   |
|                                                          |
|  RASP (Runtime Application Self-Protection)              |
|  +-- 本番環境でリアルタイム防御                             |
|  +-- 攻撃検出時に自動ブロック                              |
|  +-- WAF との連携で多層防御を実現                           |
+----------------------------------------------------------+
```

### セキュリティテスト手法のライフサイクル配置

```
  コード作成    コミット     ビルド      テスト     ステージング    本番
     |            |          |          |            |           |
  IDE Plugin   pre-commit  SAST       単体テスト    DAST        RASP
  リアルタイム  Semgrep    SonarQube   セキュリティ  OWASP ZAP   監視
  lint         gitleaks   CodeQL      テスト       Nuclei      WAF
     |            |          |          |            |           |
     v            v          v          v            v           v
  [即時FB]    [数秒]     [数分]      [数分]      [数十分]     [常時]
```

この図が示すように、セキュリティテストは開発ライフサイクルの左側 (Shift Left) に配置するほど修正コストが低くなる。SAST は最も左に位置し、DAST はデプロイ後に位置する。両者を組み合わせることでカバレッジを最大化できる。

### SAST vs DAST 比較

| 項目 | SAST | DAST |
|------|------|------|
| 解析対象 | ソースコード / バイトコード | 実行中のアプリケーション |
| 実行タイミング | 開発中 / コミット時 | デプロイ後 / ステージング |
| 検出できる脆弱性 | インジェクション、ハードコード秘密、安全でない関数 | XSS、認証不備、設定ミス |
| 偽陽性 | 多い (30-70%) | 少ない (5-20%) |
| 偽陰性 | ビジネスロジック脆弱性を見逃す | コード内部の問題を見逃す |
| 言語依存 | あり (言語別パーサ) | なし (プロトコルベース) |
| 修正の容易さ | 行番号特定で容易 | 根本原因特定が難しい場合あり |
| 速度 | 中程度 (分-時間) | 遅い (時間単位) |
| 必要な環境 | ソースコードのみ | 実行環境が必要 |
| 認証テスト | 不得意 | 得意 |
| 設定ミス検出 | 限定的 | 得意 |
| スケーラビリティ | 高い (並列化容易) | 中程度 (実行環境必要) |

### SAST・DAST の検出範囲の違い

```
          SAST が得意               両方で検出可能           DAST が得意
  +-------------------------+  +-------------------+  +-----------------------+
  | ハードコード秘密         |  | SQL インジェクション|  | 認証・認可の不備       |
  | バッファオーバーフロー   |  | XSS               |  | CSRF                  |
  | 安全でない乱数生成       |  | パストラバーサル   |  | セッション管理の不備   |
  | デッドコード             |  | コマンドインジェク |  | HTTP ヘッダの設定ミス  |
  | 未使用変数               |  |   ション           |  | CORS 設定ミス          |
  | 型安全性の問題           |  | SSRF               |  | レートリミットの欠如   |
  | NULL ポインタ参照        |  | XXE                |  | 情報漏洩 (エラー応答)  |
  | リソースリーク           |  |                    |  | TLS/SSL 設定不備      |
  +-------------------------+  +-------------------+  +-----------------------+
```

---

## 2. SAST の実践

### 2.1 SAST の動作原理

SAST ツールは以下のステップでソースコードを解析する。

```
ソースコード → 字句解析 → 構文解析 (AST生成) → 意味解析 → データフロー解析 → パターンマッチ → 結果
     |              |              |                |              |              |          |
  .py, .js      トークン化     抽象構文木       型推論・      汚染追跡      ルール照合   脆弱性
  .go, .java    分割          構築             スコープ      (taint        (OWASP等)   レポート
                                               解決          analysis)
```

**データフロー解析 (Taint Analysis)** は SAST の中核技術である。ユーザー入力 (source) から危険な操作 (sink) までのデータの流れを追跡し、途中でサニタイズされていない場合に脆弱性として報告する。

```python
# Taint Analysis の例
def handler(request):
    user_input = request.GET.get('query')   # Source: ユーザー入力
    #                    |
    #              データが流れる
    #                    |
    #                    v
    result = db.execute(                     # Sink: SQL 実行
        f"SELECT * FROM users WHERE name = '{user_input}'"  # 脆弱性検出!
    )
    return result

# サニタイズされている場合は安全と判定
def safe_handler(request):
    user_input = request.GET.get('query')   # Source: ユーザー入力
    #                    |
    #            サニタイズ処理
    #                    |
    #                    v
    sanitized = escape(user_input)           # Sanitizer
    result = db.execute(                     # Sink: SQL 実行
        "SELECT * FROM users WHERE name = %s", [sanitized]   # 安全と判定
    )
    return result
```

### 2.2 SonarQube によるコード解析

#### SonarQube のアーキテクチャ

```
+-------------------+       +-------------------+       +-------------------+
|  SonarQube Server |       |   Elasticsearch   |       |    PostgreSQL     |
|  (Web UI +        |<----->|  (検索エンジン)    |       |  (データ保存)      |
|   Compute Engine) |       +-------------------+       +-------------------+
+-------------------+                                          ^
        ^                                                      |
        |  結果送信                                             |
        |                                                      |
+-------------------+                                          |
|  SonarScanner     |------------------------------------------+
|  (解析実行)       |  解析結果をDBに保存
+-------------------+
        ^
        |  ソースコード読み込み
        |
+-------------------+
|  プロジェクト      |
|  ソースコード      |
+-------------------+
```

#### 基本設定

```properties
# sonar-project.properties
sonar.projectKey=myapp
sonar.projectName=My Application
sonar.projectVersion=1.0
sonar.sources=src
sonar.tests=tests
sonar.language=js
sonar.javascript.lcov.reportPaths=coverage/lcov.info

# セキュリティルールの重点設定
sonar.issue.ignore.multicriteria=e1
sonar.issue.ignore.multicriteria.e1.ruleKey=javascript:S1234
sonar.issue.ignore.multicriteria.e1.resourceKey=**/test/**

# エンコーディング設定
sonar.sourceEncoding=UTF-8

# 除外パターン
sonar.exclusions=**/node_modules/**,**/vendor/**,**/dist/**
sonar.test.exclusions=**/test/**,**/*.test.js
```

#### Quality Gate の設定

```json
// SonarQube Quality Gate 設定例 (API 経由)
// POST /api/qualitygates/create
{
  "name": "Security-Focused Gate",
  "conditions": [
    {
      "metric": "new_security_hotspots_reviewed",
      "op": "LT",
      "error": "100"
    },
    {
      "metric": "new_vulnerabilities",
      "op": "GT",
      "error": "0"
    },
    {
      "metric": "new_security_rating",
      "op": "GT",
      "error": "1"
    },
    {
      "metric": "new_coverage",
      "op": "LT",
      "error": "80"
    }
  ]
}
```

#### Quality Gate の各メトリクス説明

| メトリクス | 説明 | 推奨閾値 |
|-----------|------|---------|
| `new_vulnerabilities` | 新規脆弱性の数 | 0 (1つでもあればブロック) |
| `new_security_hotspots_reviewed` | セキュリティホットスポットのレビュー率 | 100% |
| `new_security_rating` | セキュリティレーティング (A-E) | A (1) のみ通過 |
| `new_coverage` | 新規コードのテストカバレッジ | 80% 以上 |
| `new_duplicated_lines_density` | 新規コードの重複率 | 3% 以下 |

#### GitHub Actions での SonarQube 統合

```yaml
# GitHub Actions での SonarQube 統合
name: Code Quality
on: [push, pull_request]

jobs:
  sonarqube:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # 差分解析に必要

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install dependencies
        run: npm ci

      - name: Run tests with coverage
        run: npm test -- --coverage

      - name: SonarQube Scan
        uses: sonarsource/sonarqube-scan-action@master
        env:
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
          SONAR_HOST_URL: ${{ secrets.SONAR_HOST_URL }}

      - name: Quality Gate Check
        uses: sonarsource/sonarqube-quality-gate-action@master
        timeout-minutes: 5
        env:
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}

      - name: Post results to PR
        if: github.event_name == 'pull_request' && failure()
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: 'SonarQube Quality Gate failed. Please check the SonarQube dashboard for details.'
            })
```

### 2.3 Semgrep (軽量 SAST)

#### Semgrep の特徴と動作原理

Semgrep は「コードのための grep」として設計された軽量 SAST ツールである。正規表現ではなく、AST (抽象構文木) レベルでパターンマッチを行うため、コードのフォーマットやスタイルの違いに影響されない。

```
通常の grep:                        Semgrep:
  テキストレベルのマッチ              AST レベルのマッチ

  "eval(" で検索                     pattern: eval($X)
    → eval( にマッチ                   → eval(user_input) にマッチ
    → "// eval(" にもマッチ (偽陽性)   → コメント内は無視 (偽陽性なし)
    → 改行を跨ぐと見逃し (偽陰性)     → 改行を跨いでもマッチ
```

#### カスタムルールの作成

```yaml
# .semgrep.yml - カスタムルール
rules:
  - id: hardcoded-secret
    patterns:
      - pattern: |
          $KEY = "..."
      - metavariable-regex:
          metavariable: $KEY
          regex: '(?i)(password|secret|api_key|token)'
    message: "ハードコードされたシークレットが検出されました"
    severity: ERROR
    languages: [python, javascript, go]
    metadata:
      cwe: "CWE-798: Use of Hard-coded Credentials"
      owasp: "A07:2021 - Identification and Authentication Failures"

  - id: sql-injection
    patterns:
      - pattern: |
          cursor.execute(f"... {$VAR} ...")
    message: "SQL インジェクションの可能性: パラメータ化クエリを使用してください"
    severity: ERROR
    languages: [python]
    metadata:
      cwe: "CWE-89: SQL Injection"
      owasp: "A03:2021 - Injection"
    fix: |
      cursor.execute("... %s ...", ($VAR,))

  - id: unsafe-deserialization
    pattern: pickle.loads(...)
    message: "安全でないデシリアライゼーション: 信頼できないデータに pickle を使用しないでください"
    severity: WARNING
    languages: [python]
    metadata:
      cwe: "CWE-502: Deserialization of Untrusted Data"

  - id: open-redirect
    patterns:
      - pattern: redirect($URL)
      - pattern-not: redirect("/...")
    message: "オープンリダイレクトの可能性: 外部 URL へのリダイレクトを検証してください"
    severity: WARNING
    languages: [python]

  - id: missing-csrf-protection
    patterns:
      - pattern: |
          @app.route("...", methods=["POST"])
          def $FUNC(...):
              ...
      - pattern-not-inside: |
          @csrf_protect
          ...
    message: "POST エンドポイントに CSRF 保護がありません"
    severity: WARNING
    languages: [python]

  - id: insecure-random
    patterns:
      - pattern-either:
          - pattern: random.random()
          - pattern: random.randint(...)
          - pattern: Math.random()
    message: "セキュリティ目的には暗号学的に安全な乱数生成器を使用してください"
    severity: WARNING
    languages: [python, javascript]
    fix-regex:
      regex: "random\\.random\\(\\)"
      replacement: "secrets.token_hex(32)"
```

#### Semgrep の高度なパターン

```yaml
# 高度なパターンマッチングの例
rules:
  # taint モード: source から sink への汚染追跡
  - id: tainted-sql-query
    mode: taint
    pattern-sources:
      - pattern: request.args.get(...)
      - pattern: request.form.get(...)
      - pattern: request.json[...]
    pattern-sinks:
      - pattern: db.execute($QUERY, ...)
      - pattern: cursor.execute($QUERY, ...)
    pattern-sanitizers:
      - pattern: sanitize(...)
      - pattern: escape(...)
    message: "ユーザー入力が SQL クエリに直接使用されています"
    severity: ERROR
    languages: [python]

  # join モード: 複数条件の組み合わせ
  - id: jwt-without-verification
    patterns:
      - pattern: jwt.decode($TOKEN, ...)
      - pattern-not: jwt.decode($TOKEN, ..., verify=True, ...)
      - pattern-not: jwt.decode($TOKEN, ..., algorithms=[...], ...)
    message: "JWT の署名検証が無効になっている可能性があります"
    severity: ERROR
    languages: [python]
```

#### Semgrep の実行コマンド

```bash
# Semgrep の基本実行
semgrep --config auto .                    # 自動ルール選択
semgrep --config .semgrep.yml .            # カスタムルール
semgrep --config p/owasp-top-ten .         # OWASP Top 10 ルール
semgrep --config p/javascript .            # JavaScript 専用ルール
semgrep --config p/python .               # Python 専用ルール
semgrep --config p/golang .               # Go 専用ルール

# 出力形式の指定
semgrep --config auto --json -o results.json .     # JSON 形式
semgrep --config auto --sarif -o results.sarif .   # SARIF 形式 (GitHub連携用)
semgrep --config auto --emacs .                    # Emacs 形式

# CI/CD ゲート (エラーがあればビルド失敗)
semgrep --config auto --error .

# 特定ファイルのみスキャン
semgrep --config auto --include="*.py" .
semgrep --config auto --exclude="tests/*" .

# 差分のみスキャン (高速化)
semgrep --config auto --baseline-commit HEAD~1 .

# Semgrep CI (Semgrep App 連携)
SEMGREP_APP_TOKEN=xxx semgrep ci
```

### 2.4 CodeQL による高度な解析

CodeQL は GitHub が開発した SAST ツールで、コードをデータベースとして構築し、SQL ライクなクエリ言語でパターンを検索できる。

```ql
// CodeQL クエリ例: SQL インジェクションの検出
import javascript
import DataFlow::PathGraph

class SqlInjectionConfig extends TaintTracking::Configuration {
  SqlInjectionConfig() { this = "SqlInjectionConfig" }

  override predicate isSource(DataFlow::Node source) {
    exists(Express::RequestExpr req |
      source.asExpr() = req.getAPropertyRead("query").getAPropertyRead(_)
    )
  }

  override predicate isSink(DataFlow::Node sink) {
    exists(DatabaseAccess db |
      sink.asExpr() = db.getAQueryArgument()
    )
  }
}

from SqlInjectionConfig cfg, DataFlow::PathNode source, DataFlow::PathNode sink
where cfg.hasFlowPath(source, sink)
select sink, source, sink,
  "この SQL クエリは $@ から来たユーザー入力を含んでいます。",
  source.getNode(), "ユーザー入力"
```

```yaml
# GitHub Actions での CodeQL 統合
name: CodeQL Analysis
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 6 * * 1'  # 毎週月曜 6:00 UTC

jobs:
  analyze:
    runs-on: ubuntu-latest
    permissions:
      security-events: write
    strategy:
      matrix:
        language: [javascript, python]
    steps:
      - uses: actions/checkout@v4

      - name: Initialize CodeQL
        uses: github/codeql-action/init@v3
        with:
          languages: ${{ matrix.language }}
          queries: +security-extended,security-and-quality

      - name: Autobuild
        uses: github/codeql-action/autobuild@v3

      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v3
        with:
          category: "/language:${{ matrix.language }}"
```

### 2.5 言語別 SAST ツールの選定ガイド

#### SAST ツールの比較

| ツール | 対応言語 | 速度 | カスタムルール | コスト | 特徴 |
|--------|---------|------|-------------|-------|------|
| SonarQube | 30+ | 中 | はい | CE: 無料 | 品質管理の統合ダッシュボード |
| Semgrep | 30+ | 高速 | YAML ベース | OSS: 無料 | 軽量・カスタムルールが容易 |
| CodeQL | 10+ | 遅い | QL 言語 | GitHub 無料 | 高精度なデータフロー解析 |
| Bandit | Python | 高速 | プラグイン | 無料 | Python 特化で導入簡単 |
| ESLint Security | JavaScript | 高速 | ルールベース | 無料 | eslint-plugin-security |
| gosec | Go | 高速 | AST ベース | 無料 | Go 特化 |
| Brakeman | Ruby | 高速 | - | 無料 | Rails 特化 |
| SpotBugs | Java | 中 | プラグイン | 無料 | Find Security Bugs プラグイン |
| PHPStan | PHP | 高速 | ルール拡張 | 無料 | 型解析ベース |

#### 言語別の推奨構成

```
Python プロジェクト:
  必須: Semgrep + Bandit
  推奨: + SonarQube (品質管理)
  任意: + CodeQL (GitHub 利用時)

JavaScript/TypeScript プロジェクト:
  必須: Semgrep + ESLint Security
  推奨: + SonarQube (品質管理)
  任意: + CodeQL (GitHub 利用時)

Go プロジェクト:
  必須: Semgrep + gosec
  推奨: + SonarQube (品質管理)
  任意: + staticcheck (追加lint)

Java プロジェクト:
  必須: SonarQube + SpotBugs (Find Security Bugs)
  推奨: + Semgrep
  任意: + CodeQL (GitHub 利用時)

Ruby/Rails プロジェクト:
  必須: Brakeman + Semgrep
  推奨: + SonarQube (品質管理)
```

### 2.6 Bandit (Python 特化 SAST) の活用

```bash
# Bandit のインストールと実行
pip install bandit

# 基本実行
bandit -r ./src/

# 重大度でフィルタ
bandit -r ./src/ -ll  # Medium 以上のみ

# 特定テストのみ実行
bandit -r ./src/ -t B601,B602,B603  # シェルインジェクション系

# JSON 出力
bandit -r ./src/ -f json -o bandit-results.json

# 除外設定
bandit -r ./src/ --exclude tests/,docs/
```

```ini
# .bandit - Bandit 設定ファイル
[bandit]
exclude = tests,docs,venv
tests = B101,B102,B103,B104,B105,B106,B107,B108,B110
# B301-B303: pickle 関連
# B601-B603: シェルインジェクション
# B608: SQL インジェクション

skips = B101  # assert の使用 (テストでは必要)

[bandit.plugins.hardcoded_password_string]
word_list = password,pass,passwd,pwd,secret,token,api_key,apikey
```

```python
# Bandit が検出する典型的な脆弱性

# B602: subprocess with shell=True (高リスク)
import subprocess
user_input = input("Command: ")
subprocess.call(user_input, shell=True)  # 検出!

# B608: SQL injection
query = "SELECT * FROM users WHERE id = '%s'" % user_id  # 検出!

# B105: Hardcoded password
password = "super_secret_123"  # 検出!

# B301: Pickle usage
import pickle
data = pickle.loads(untrusted_data)  # 検出!

# B104: Binding to all interfaces
from flask import Flask
app = Flask(__name__)
app.run(host='0.0.0.0')  # 検出!
```

---

## 3. DAST の実践

### 3.1 DAST の動作原理

DAST ツールは実行中のアプリケーションに対して、攻撃者の視点からテストリクエストを送信する。

```
DAST ツール                    テスト対象アプリ
+------------------+          +------------------+
|                  |   HTTP   |                  |
|  1. クローラー    |--------->|  Web サーバー     |
|    サイトマップ   |<---------|  (Nginx等)       |
|    構築          |   応答   |                  |
|                  |          |  アプリケーション  |
|  2. パッシブ     |--------->|  サーバー         |
|    スキャナー     |<---------|  (Django等)      |
|    通信監視      |          |                  |
|                  |          |  データベース      |
|  3. アクティブ   |--------->|                  |
|    スキャナー     |<---------|                  |
|    攻撃送信      |          |                  |
|                  |          +------------------+
|  4. レポート生成  |
|                  |
+------------------+

攻撃リクエストの例:
  通常: GET /users?id=123
  XSS:  GET /users?id=<script>alert(1)</script>
  SQLi: GET /users?id=123' OR '1'='1
  Path: GET /users/../../../etc/passwd
```

### 3.2 OWASP ZAP による動的テスト

#### ZAP のテストフロー

```
ZAP のテストフロー:

  +-- Spider (クロール) --+
  |  サイトマップ構築      |
  |  Ajax Spider も利用可  |
  +-----------------------+
            |
            v
  +-- Passive Scan -------+
  |  通信を観察して検出     |
  |  (速い、低リスク)      |
  |  例: ヘッダの欠如      |
  |      Cookie の属性     |
  +-----------------------+
            |
            v
  +-- Active Scan ---------+
  |  攻撃リクエスト送信     |
  |  (遅い、サーバ負荷あり) |
  |  例: SQL Injection      |
  |      XSS               |
  |      Path Traversal    |
  +------------------------+
            |
            v
  +-- レポート生成 --------+
  |  HTML / JSON / XML     |
  |  SARIF (GitHub連携)    |
  +------------------------+
```

#### ZAP のスキャンポリシー設定

```xml
<!-- ZAP スキャンポリシー設定例 -->
<configuration>
  <scanner>
    <!-- SQL Injection テスト -->
    <plugin id="40018" enabled="true" strength="HIGH" threshold="MEDIUM"/>
    <!-- XSS テスト -->
    <plugin id="40012" enabled="true" strength="HIGH" threshold="LOW"/>
    <!-- Path Traversal テスト -->
    <plugin id="6" enabled="true" strength="MEDIUM" threshold="MEDIUM"/>
    <!-- Remote Code Execution -->
    <plugin id="40014" enabled="true" strength="HIGH" threshold="HIGH"/>
    <!-- CSRF テスト -->
    <plugin id="40003" enabled="true" strength="MEDIUM" threshold="LOW"/>
  </scanner>
  <spider>
    <maxDuration>10</maxDuration>
    <maxDepth>5</maxDepth>
    <threadCount>5</threadCount>
  </spider>
</configuration>
```

#### ZAP の認証設定

多くの Web アプリケーションは認証が必要である。ZAP で認証済みスキャンを行う方法を示す。

```python
from zapv2 import ZAPv2
import time

zap = ZAPv2(apikey='your-api-key', proxies={
    'http': 'http://localhost:8080',
    'https': 'http://localhost:8080',
})

target = 'https://staging.example.com'

# 認証設定
context_id = zap.context.new_context('authenticated-context')

# ログインページの URL パターンを含める
zap.context.include_in_context(context_id, f'{target}.*')

# フォームベース認証の設定
auth_method_config = (
    f'loginUrl={target}/login&'
    f'loginRequestData=username={{%username%}}&password={{%password%}}'
)
zap.authentication.set_authentication_method(
    context_id, 'formBasedAuthentication', auth_method_config
)

# ログイン状態の検証パターン
zap.authentication.set_logged_in_indicator(context_id, '\\QWelcome\\E')
zap.authentication.set_logged_out_indicator(context_id, '\\QLogin\\E')

# ユーザーの追加
user_id = zap.users.new_user(context_id, 'test-user')
zap.users.set_authentication_credentials(
    context_id, user_id,
    f'username=testuser&password=TestPass123!'
)
zap.users.set_user_enabled(context_id, user_id, True)

# 認証済みスキャンの実行
zap.forcedUser.set_forced_user(context_id, user_id)
zap.forcedUser.set_forced_user_mode_enabled(True)
```

### 3.3 ZAP の API を使った自動テスト

```python
from zapv2 import ZAPv2
import time
import json
import sys

# ZAP に接続
zap = ZAPv2(apikey='your-api-key', proxies={
    'http': 'http://localhost:8080',
    'https': 'http://localhost:8080',
})

target = 'https://staging.example.com'

def run_zap_scan(target_url, scan_type='full'):
    """ZAP による自動セキュリティスキャン

    Args:
        target_url: スキャン対象の URL
        scan_type: 'baseline' (パッシブのみ) または 'full' (アクティブ含む)

    Returns:
        tuple: (high_alerts, medium_alerts, low_alerts)
    """

    print(f"[*] Target: {target_url}")
    print(f"[*] Scan type: {scan_type}")

    # Step 1: Spider (クロール)
    print("[*] Phase 1: Spidering target...")
    scan_id = zap.spider.scan(target_url)
    while int(zap.spider.status(scan_id)) < 100:
        progress = zap.spider.status(scan_id)
        print(f"    Spider progress: {progress}%")
        time.sleep(2)

    urls_found = len(zap.spider.results(scan_id))
    print(f"[+] Spider found {urls_found} URLs")

    # Ajax Spider (SPA 対応)
    print("[*] Phase 1.5: Ajax Spidering...")
    zap.ajaxSpider.scan(target_url)
    timeout = 120  # 2分のタイムアウト
    while zap.ajaxSpider.status == 'running' and timeout > 0:
        time.sleep(5)
        timeout -= 5
    zap.ajaxSpider.stop()
    print(f"[+] Ajax Spider found additional URLs")

    # Step 2: Passive Scan の完了を待つ
    print("[*] Phase 2: Waiting for passive scan...")
    while int(zap.pscan.records_to_scan) > 0:
        remaining = zap.pscan.records_to_scan
        print(f"    Records remaining: {remaining}")
        time.sleep(1)
    print("[+] Passive scan completed")

    # Step 3: Active Scan (baseline でなければ)
    if scan_type == 'full':
        print("[*] Phase 3: Active scanning...")
        scan_id = zap.ascan.scan(target_url)
        while int(zap.ascan.status(scan_id)) < 100:
            progress = zap.ascan.status(scan_id)
            print(f"    Active scan progress: {progress}%")
            time.sleep(5)
        print("[+] Active scan completed")
    else:
        print("[*] Phase 3: Skipped (baseline scan)")

    # Step 4: 結果を取得・分類
    alerts = zap.core.alerts(baseurl=target_url)
    high_alerts = [a for a in alerts if a['risk'] == 'High']
    medium_alerts = [a for a in alerts if a['risk'] == 'Medium']
    low_alerts = [a for a in alerts if a['risk'] == 'Low']
    info_alerts = [a for a in alerts if a['risk'] == 'Informational']

    print(f"\n[=] Results Summary:")
    print(f"    High:          {len(high_alerts)}")
    print(f"    Medium:        {len(medium_alerts)}")
    print(f"    Low:           {len(low_alerts)}")
    print(f"    Informational: {len(info_alerts)}")

    # 重複除去した脆弱性タイプの一覧
    unique_alerts = set()
    for alert in high_alerts + medium_alerts:
        unique_alerts.add(f"[{alert['risk']}] {alert['alert']}")
    if unique_alerts:
        print(f"\n[!] Unique vulnerability types found:")
        for ua in sorted(unique_alerts):
            print(f"    - {ua}")

    # HTML レポート出力
    with open('zap-report.html', 'w') as f:
        f.write(zap.core.htmlreport())
    print(f"\n[+] HTML report saved to zap-report.html")

    # JSON レポート出力
    with open('zap-report.json', 'w') as f:
        json.dump(alerts, f, indent=2)
    print(f"[+] JSON report saved to zap-report.json")

    return high_alerts, medium_alerts, low_alerts

# 実行
high, medium, low = run_zap_scan(target, scan_type='full')

# CI/CD ゲート判定
if high:
    print("\n[FAIL] CRITICAL: High-risk vulnerabilities found!")
    for alert in high:
        print(f"  - {alert['alert']}: {alert['url']}")
    sys.exit(1)
elif medium:
    print("\n[WARN] Medium-risk vulnerabilities found (not blocking)")
    sys.exit(0)
else:
    print("\n[PASS] No high/medium-risk vulnerabilities found")
    sys.exit(0)
```

### 3.4 ZAP の CI/CD 統合

```yaml
# GitHub Actions での ZAP スキャン
name: DAST Scan
on:
  deployment_status:

jobs:
  zap-baseline:
    if: github.event.deployment_status.state == 'success'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: ZAP Baseline Scan
        uses: zaproxy/action-baseline@v0.12.0
        with:
          target: ${{ github.event.deployment.payload.url }}
          rules_file_name: '.zap/rules.tsv'
          cmd_options: '-a'
          issue_title: 'ZAP Baseline Scan Report'
          fail_action: true

      - name: Upload ZAP Report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: zap-baseline-report
          path: report_html.html

  zap-full:
    needs: zap-baseline
    if: contains(github.event.deployment.environment, 'staging')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: ZAP Full Scan
        uses: zaproxy/action-full-scan@v0.10.0
        with:
          target: ${{ github.event.deployment.payload.url }}
          rules_file_name: '.zap/rules.tsv'
          cmd_options: '-a -j'

      - name: Upload Full Scan Report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: zap-full-report
          path: report_html.html
```

#### ZAP ルールファイルの設定

```tsv
# .zap/rules.tsv
# ルールID	動作	説明
10010	IGNORE	Cookie No HttpOnly Flag (低リスク、他で対応)
10011	IGNORE	Cookie Without Secure Flag (開発環境のため)
10015	WARN	Incomplete or No Cache-control Header Set
10017	FAIL	Cross-Domain JavaScript Source File Inclusion
10020	FAIL	X-Frame-Options Header Not Set
10021	FAIL	X-Content-Type-Options Header Missing
10038	FAIL	Content Security Policy (CSP) Header Not Set
40012	FAIL	Cross Site Scripting (Reflected)
40014	FAIL	Cross Site Scripting (Persistent)
40018	FAIL	SQL Injection
90022	FAIL	Application Error Disclosure
```

### 3.5 Nuclei による高速 DAST

Nuclei は YAML テンプレートベースの高速脆弱性スキャナである。

```yaml
# nuclei テンプレート例: カスタム脆弱性チェック
id: custom-admin-panel-check
info:
  name: Admin Panel Detection
  author: security-team
  severity: medium
  description: 公開されている管理画面の検出
  tags: admin,panel,exposure

requests:
  - method: GET
    path:
      - "{{BaseURL}}/admin"
      - "{{BaseURL}}/admin/login"
      - "{{BaseURL}}/wp-admin"
      - "{{BaseURL}}/administrator"
      - "{{BaseURL}}/manage"
    matchers-condition: or
    matchers:
      - type: status
        status:
          - 200
      - type: word
        words:
          - "admin"
          - "login"
          - "dashboard"
        condition: or
```

```bash
# Nuclei の実行
nuclei -u https://staging.example.com -t nuclei-templates/    # テンプレート指定
nuclei -u https://staging.example.com -tags cve               # CVE テンプレート
nuclei -u https://staging.example.com -severity critical,high  # 重大度フィルタ
nuclei -l urls.txt -t nuclei-templates/ -o results.json -json  # バッチ実行
```

---

## 4. IAST (Interactive Application Security Testing)

### 4.1 IAST の動作原理

IAST はアプリケーション内部にエージェント (インストゥルメンテーション) を埋め込み、実行時にリアルタイムで脆弱性を検出する。SAST と DAST の利点を組み合わせたハイブリッドアプローチである。

```
DAST (外部テスト)                IAST エージェント (内部監視)
+------------------+            +----------------------------------+
|  テストリクエスト  |  ------->  |  Web アプリケーション              |
|  送信             |            |  +----------------------------+  |
|                  |            |  | IAST Agent                 |  |
|                  |            |  | +-- HTTP リクエスト解析     |  |
|  レスポンス受信   |  <-------  |  | +-- データフロー追跡       |  |
|                  |            |  | +-- SQL クエリ監視         |  |
+------------------+            |  | +-- ファイルアクセス監視   |  |
                                |  | +-- 外部呼び出し監視       |  |
                                |  +----------------------------+  |
                                +----------------------------------+
                                           |
                                           v
                                +----------------------------+
                                | 脆弱性レポート              |
                                | - ソースコードの行番号       |
                                | - データフローの完全な経路   |
                                | - 実際に悪用可能かどうか     |
                                +----------------------------+
```

### 4.2 IAST vs SAST vs DAST

| 特性 | SAST | DAST | IAST |
|------|------|------|------|
| 偽陽性率 | 高い (30-70%) | 低い (5-20%) | 非常に低い (<5%) |
| 偽陰性率 | 中程度 | 中程度 | 低い |
| コード行番号特定 | はい | いいえ | はい |
| 実行環境の必要性 | 不要 | 必要 | 必要 |
| パフォーマンス影響 | なし | 外部からの負荷 | 2-5% のオーバーヘッド |
| ビジネスロジック検出 | 困難 | 部分的 | 可能 |
| 導入コスト | 低い | 低い | 高い |
| 代表的ツール | Semgrep, SonarQube | ZAP, Nuclei | Contrast, Hdiv |

### 4.3 IAST エージェントの導入例 (Java)

```java
// Java アプリケーションへの IAST エージェント導入
// JVM 起動オプションに追加
// java -javaagent:/path/to/contrast.jar -jar myapp.jar

// もしくは Dockerfile で設定
// FROM eclipse-temurin:21-jre
// COPY contrast.jar /opt/contrast/
// ENV JAVA_TOOL_OPTIONS="-javaagent:/opt/contrast/contrast.jar"
// COPY myapp.jar /app/
// CMD ["java", "-jar", "/app/myapp.jar"]
```

```yaml
# contrast_security.yaml - IAST エージェント設定
api:
  url: https://app.contrastsecurity.com/Contrast
  api_key: ${CONTRAST_API_KEY}
  service_key: ${CONTRAST_SERVICE_KEY}

agent:
  java:
    enable_assess: true        # IAST 機能有効化
    enable_protect: false      # RASP 機能 (本番用)
    assess:
      enable_sampling: true
      sampling_window: 180
      sampling_request_frequency: 5
    rules:
      disabled_rules:
        - cookie-flags-missing  # テスト環境では無視
```

---

## 5. CI/CD パイプラインへの統合

### 5.1 セキュリティテストの配置

```
開発者の PC    →    CI/CD Pipeline    →    ステージング    →    本番
    |                    |                      |                |
 [pre-commit]      [ビルド時]              [デプロイ後]      [継続的]
    |                    |                      |                |
  Semgrep          SonarQube              OWASP ZAP         ランタイム
  (即時)           Semgrep                Nuclei             監視
                   SCA (Trivy)            (DAST)             (IAST/RASP)
                   シークレットスキャン                         WAF
                   (SAST + SCA)

修正コスト:  $1          $10                   $100             $1000
  (Shift Left するほど修正コストが低い)
```

### 5.2 統合パイプラインの完全版

```yaml
# .github/workflows/security-pipeline.yml
name: Security Pipeline
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # ====================================
  # Stage 1: 静的解析 (並列実行)
  # ====================================
  sast-semgrep:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Semgrep Scan
        uses: returntocorp/semgrep-action@v1
        with:
          config: >-
            p/owasp-top-ten
            p/r2c-security-audit
            .semgrep.yml
          generateSarif: true

      - name: Upload SARIF
        if: always()
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: semgrep.sarif

  sast-sonarqube:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: SonarQube Scan
        uses: sonarsource/sonarqube-scan-action@master
        env:
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
          SONAR_HOST_URL: ${{ secrets.SONAR_HOST_URL }}

      - name: Quality Gate
        uses: sonarsource/sonarqube-quality-gate-action@master
        timeout-minutes: 5
        env:
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}

  # ====================================
  # Stage 1b: SCA + シークレットスキャン (並列)
  # ====================================
  sca:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Trivy filesystem scan
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          severity: 'HIGH,CRITICAL'
          exit-code: '1'
          format: 'sarif'
          output: 'trivy-fs-results.sarif'

      - name: Upload Trivy SARIF
        if: always()
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: trivy-fs-results.sarif

  secret-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # 全履歴をスキャン

      - name: Gitleaks scan
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  # ====================================
  # Stage 2: コンテナスキャン
  # ====================================
  container-scan:
    needs: [sast-semgrep, sast-sonarqube, sca, secret-scan]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build container image
        run: docker build -t ${{ env.IMAGE_NAME }}:${{ github.sha }} .

      - name: Trivy image scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: '${{ env.IMAGE_NAME }}:${{ github.sha }}'
          severity: 'CRITICAL'
          exit-code: '1'
          format: 'sarif'
          output: 'trivy-image-results.sarif'

      - name: Upload Trivy Image SARIF
        if: always()
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: trivy-image-results.sarif

  # ====================================
  # Stage 3: DAST (ステージング環境)
  # ====================================
  dast:
    needs: container-scan
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Deploy to staging
        id: deploy
        run: |
          # ステージング環境へデプロイ
          echo "staging_url=https://staging.example.com" >> $GITHUB_OUTPUT

      - name: Wait for deployment
        run: |
          for i in $(seq 1 30); do
            if curl -sf "${{ steps.deploy.outputs.staging_url }}/health"; then
              echo "Deployment is healthy"
              exit 0
            fi
            sleep 10
          done
          echo "Deployment health check failed"
          exit 1

      - name: ZAP Baseline Scan
        uses: zaproxy/action-baseline@v0.12.0
        with:
          target: ${{ steps.deploy.outputs.staging_url }}
          rules_file_name: '.zap/rules.tsv'
          cmd_options: '-a'
          fail_action: true

      - name: ZAP Full Scan (週次)
        if: github.event.schedule != ''
        uses: zaproxy/action-full-scan@v0.10.0
        with:
          target: ${{ steps.deploy.outputs.staging_url }}

  # ====================================
  # Stage 4: セキュリティレポート集約
  # ====================================
  security-summary:
    needs: [sast-semgrep, sast-sonarqube, sca, secret-scan, container-scan]
    if: always()
    runs-on: ubuntu-latest
    steps:
      - name: Security Summary
        run: |
          echo "## Security Scan Summary" >> $GITHUB_STEP_SUMMARY
          echo "| Check | Status |" >> $GITHUB_STEP_SUMMARY
          echo "|-------|--------|" >> $GITHUB_STEP_SUMMARY
          echo "| SAST (Semgrep) | ${{ needs.sast-semgrep.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| SAST (SonarQube) | ${{ needs.sast-sonarqube.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| SCA (Trivy) | ${{ needs.sca.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| Secrets (Gitleaks) | ${{ needs.secret-scan.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| Container Scan | ${{ needs.container-scan.result }} |" >> $GITHUB_STEP_SUMMARY
```

### 5.3 pre-commit フックでの SAST 統合

```yaml
# .pre-commit-config.yaml
repos:
  # Semgrep (SAST)
  - repo: https://github.com/returntocorp/semgrep
    rev: v1.50.0
    hooks:
      - id: semgrep
        args: ['--config', 'auto', '--error']

  # Gitleaks (シークレットスキャン)
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks

  # Bandit (Python SAST)
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.7
    hooks:
      - id: bandit
        args: ['-ll', '-ii']

  # ESLint Security (JavaScript SAST)
  - repo: https://github.com/pre-commit/mirrors-eslint
    rev: v8.56.0
    hooks:
      - id: eslint
        additional_dependencies:
          - eslint-plugin-security@2.1.0

  # Hadolint (Dockerfile lint)
  - repo: https://github.com/hadolint/hadolint
    rev: v2.12.0
    hooks:
      - id: hadolint
```

```bash
# pre-commit のセットアップ
pip install pre-commit
pre-commit install
pre-commit install --hook-type commit-msg

# 全ファイルに対して実行 (初回)
pre-commit run --all-files

# 特定のフックのみ実行
pre-commit run semgrep --all-files
pre-commit run gitleaks --all-files
```

---

## 6. シークレットスキャン

### 6.1 シークレットスキャンの重要性

ソースコードに埋め込まれた秘密情報 (API キー、パスワード、証明書など) は最もよく悪用される脆弱性の一つである。GitGuardian の調査によると、2023 年に公開リポジトリで検出されたシークレットは 1,280 万件に上る。

```
シークレット漏洩の典型的な流れ:

  開発者が API キーを         Git に commit        GitHub に push
  ソースコードに埋め込む  →   して履歴に残る   →   して公開される
        |                         |                      |
        v                         v                      v
  「一時的なテスト」の        git rm しても          ボットが数分以内に
  つもりで放置              履歴に残り続ける       検出して悪用開始
```

### 6.2 gitleaks の活用

```bash
# gitleaks のインストールと実行
# macOS
brew install gitleaks

# 現在のリポジトリをスキャン
gitleaks detect --source . --report-format json --report-path gitleaks-report.json

# Git 履歴全体をスキャン
gitleaks detect --source . --report-format json --report-path gitleaks-report.json --log-opts="--all"

# 特定のコミット範囲のみスキャン
gitleaks detect --source . --log-opts="HEAD~10..HEAD"

# 差分のみスキャン (CI/CD 用、高速)
gitleaks protect --staged  # ステージングされたファイルのみ
```

```toml
# .gitleaks.toml - カスタムルール設定
title = "Custom Gitleaks Configuration"

[allowlist]
description = "Global allowlist"
paths = [
    '''test/.*''',
    '''.*_test\.go''',
    '''.*\.test\.js''',
    '''fixtures/.*''',
    '''__mocks__/.*''',
]
regexTarget = "match"

# AWS Access Key
[[rules]]
id = "aws-access-key"
description = "AWS Access Key ID"
regex = '''AKIA[0-9A-Z]{16}'''
tags = ["aws", "credentials"]
entropy = 3.5

# AWS Secret Key
[[rules]]
id = "aws-secret-key"
description = "AWS Secret Access Key"
regex = '''(?i)aws_secret_access_key\s*[:=]\s*['"]?([A-Za-z0-9/+=]{40})['"]?'''
tags = ["aws", "credentials"]

# Generic API Key
[[rules]]
id = "generic-api-key"
description = "Generic API Key"
regex = '''(?i)(api[_-]?key|apikey)\s*[:=]\s*['"][a-zA-Z0-9]{20,}['"]'''
tags = ["api", "generic"]

# GitHub Token
[[rules]]
id = "github-token"
description = "GitHub Personal Access Token"
regex = '''ghp_[a-zA-Z0-9]{36}'''
tags = ["github", "token"]

# Slack Webhook
[[rules]]
id = "slack-webhook"
description = "Slack Webhook URL"
regex = '''https://hooks\.slack\.com/services/T[a-zA-Z0-9]{8}/B[a-zA-Z0-9]{8}/[a-zA-Z0-9]{24}'''
tags = ["slack", "webhook"]

# Private Key
[[rules]]
id = "private-key"
description = "Private Key"
regex = '''-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----'''
tags = ["key", "private"]

# JWT
[[rules]]
id = "jwt-token"
description = "JSON Web Token"
regex = '''eyJ[A-Za-z0-9-_]+\.eyJ[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+'''
tags = ["jwt", "token"]

# ルール固有の allowlist
[[rules]]
id = "generic-password"
description = "Generic Password"
regex = '''(?i)(password|passwd|pwd)\s*[:=]\s*['"][^'"]{8,}['"]'''
tags = ["password"]
[rules.allowlist]
regexes = [
    '''(?i)password\s*[:=]\s*['"](placeholder|example|changeme|xxx|test)['"]''',
]
```

### 6.3 シークレット漏洩時の対応手順

```
シークレット漏洩が発覚した場合の対応フロー:

  1. 即座にシークレットを無効化
     +-- API キー: ダッシュボードから失効
     +-- パスワード: 即座に変更
     +-- 証明書: 失効リストに追加
     |
  2. 影響範囲の調査
     +-- シークレットを使用したアクセスログを確認
     +-- 不正利用の痕跡を調査
     +-- 影響を受けたシステムの特定
     |
  3. Git 履歴からの削除 (オプション)
     +-- git filter-branch または BFG Repo-Cleaner を使用
     +-- ただし既にクローンされた可能性を考慮
     |
  4. 新しいシークレットの発行
     +-- より安全な管理方法 (Vault, AWS Secrets Manager 等) で管理
     +-- 環境変数経由でアプリに注入
     |
  5. 再発防止策
     +-- pre-commit フックに gitleaks を追加
     +-- CI/CD パイプラインにシークレットスキャンを統合
     +-- チームへの教育
```

```bash
# BFG Repo-Cleaner によるシークレット削除
# ※ 最終手段として使用。チーム全員に影響する

# 1. ベアクローンを作成
git clone --mirror https://github.com/user/repo.git

# 2. BFG でシークレットを含むファイルを削除
java -jar bfg.jar --replace-text passwords.txt repo.git

# 3. クリーンアップ
cd repo.git
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 4. プッシュ (force push が必要)
git push
```

---

## 7. 結果のトリアージと管理

### 7.1 トリアージプロセス

```
スキャン結果
    |
    v
+-- 重大度分類 --+
|               |
|  Critical     | → 即座に修正 (24時間以内) → ビルドをブロック
|  High         | → 即座に修正 (48時間以内) → ビルドをブロック
|  Medium       | → 次スプリントで対応 → 警告のみ
|  Low          | → バックログに追加 → 情報提供
|  Info         | → 記録のみ → 無視可
|               |
+-- 偽陽性判定 --+
|               |
|  偽陽性確認    | → 抑制ルールに追加
|  理由を文書化  | → チームで共有
|               |
+---------------+
```

### 7.2 偽陽性の管理

```yaml
# Semgrep の偽陽性抑制

# インラインでの抑制 (理由を必ず記載)
def get_system_info():
    # nosemgrep: hardcoded-secret  # システム定数、シークレットではない
    DEFAULT_REGION = "us-east-1"
    return DEFAULT_REGION

# ファイルレベルでの抑制 (.semgrepignore)
# .semgrepignore
tests/
fixtures/
**/testdata/**
vendor/
```

```properties
# SonarQube の偽陽性抑制

# インラインで抑制
String query = buildQuery(param); // NOSONAR - パラメータはバリデーション済み

# SonarQube UI でも抑制可能:
# Issues → Won't Fix / False Positive をマーク
# 理由をコメントとして記録
```

### 7.3 メトリクスとダッシュボード

```
推奨メトリクス:

  +-- セキュリティメトリクス --+
  |                           |
  |  脆弱性密度               |  脆弱性数 / 1000行コード
  |  平均修正時間 (MTTR)      |  検出から修正までの平均日数
  |  偽陽性率                 |  偽陽性数 / 全検出数
  |  スキャンカバレッジ       |  スキャン対象リポジトリ / 全リポジトリ
  |  Quality Gate 通過率     |  通過ビルド数 / 全ビルド数
  |  重大脆弱性のバックログ   |  未修正の Critical/High 件数
  |                           |
  +---------------------------+

  目標値の例:
  - 脆弱性密度: < 1.0 (1000行あたり)
  - MTTR (Critical): < 24時間
  - MTTR (High): < 1週間
  - 偽陽性率: < 20%
  - スキャンカバレッジ: > 95%
  - Quality Gate 通過率: > 90%
```

---

## 8. アンチパターンとベストプラクティス

### アンチパターン 1: セキュリティスキャンの結果を無視

```
NG: スキャン結果が 200 件の警告を出すが全て無視
  → 偽陽性と本物の脆弱性が混在し、全てが放置される
  → 「アラート疲れ」が発生してツール自体が無視される

OK: トリアージプロセスを設定
  1. Critical/High → 即座に修正 (ビルドをブロック)
  2. Medium → 次スプリントで対応
  3. Low/Info → バックログに追加
  4. 偽陽性 → 抑制ルールに追加して文書化
```

### アンチパターン 2: SAST だけで安心する

```
NG: SAST のみ実施し「セキュリティテスト完了」とする
  → SAST は認証フロー・認可ロジックの不備を検出できない
  → ビジネスロジックの脆弱性は見逃される

OK: SAST + DAST + SCA の組み合わせ
  SAST → コードの脆弱性パターン検出
  DAST → 実際の攻撃シミュレーション
  SCA  → 依存関係の既知脆弱性検出
  手動ペネトレーションテスト → ビジネスロジックの検証
```

### アンチパターン 3: 全ルールを有効にして始める

```
NG: SAST ツール導入初日に全ルールを有効化
  → 数千件の警告が出てチームが圧倒される
  → 「ツールが使えない」という印象になり放置される

OK: 段階的導入
  Week 1: Critical ルールのみ有効化 (10-20件の検出)
  Week 2: High ルールを追加
  Week 4: Medium ルールを追加
  Month 2: カスタムルールの追加開始
  Month 3: Quality Gate を徐々に厳格化
```

### アンチパターン 4: スキャン結果を開発者に丸投げ

```
NG: スキャン結果を開発者に送り「全部直してください」
  → セキュリティの知識がないと修正方法が分からない
  → 優先順位が不明で放置される

OK: セキュリティチャンピオン制度
  1. 各チームにセキュリティチャンピオンを配置
  2. トリアージ会議で優先順位を決定
  3. 修正ガイダンスを脆弱性レポートに含める
  4. 定期的なセキュリティトレーニングを実施
```

### アンチパターン 5: DAST を本番環境で実行

```
NG: Active Scan を本番環境に対して実行
  → データ破損のリスク
  → サービス停止のリスク
  → 攻撃として検知されアラートが発生

OK: 環境の使い分け
  本番環境: Baseline Scan (パッシブスキャン) のみ
  ステージング環境: Full Scan (アクティブスキャン)
  開発環境: 開発者による手動テスト
```

---

## 9. 演習問題

### 演習 1: Semgrep カスタムルールの作成

以下のコードに含まれる脆弱性を検出する Semgrep ルールを作成せよ。

```python
# vulnerable_app.py
from flask import Flask, request, render_template_string
import subprocess
import os

app = Flask(__name__)

@app.route('/search')
def search():
    query = request.args.get('q', '')
    # 脆弱性 1: テンプレートインジェクション
    return render_template_string(f'<h1>Results for: {query}</h1>')

@app.route('/ping')
def ping():
    host = request.args.get('host', '')
    # 脆弱性 2: コマンドインジェクション
    result = subprocess.run(f'ping -c 1 {host}', shell=True, capture_output=True)
    return result.stdout.decode()

@app.route('/file')
def read_file():
    filename = request.args.get('name', '')
    # 脆弱性 3: パストラバーサル
    filepath = os.path.join('/var/data', filename)
    with open(filepath) as f:
        return f.read()
```

<details>
<summary>解答例</summary>

```yaml
rules:
  - id: ssti-flask
    patterns:
      - pattern: render_template_string(f"...", ...)
      - pattern-not: render_template_string("...", ...)
    message: "Flask テンプレートインジェクション (SSTI): render_template_string に f-string を使用しないでください"
    severity: ERROR
    languages: [python]
    metadata:
      cwe: "CWE-1336: Server-Side Template Injection"

  - id: command-injection-subprocess
    patterns:
      - pattern: subprocess.run(f"...", shell=True, ...)
      - pattern: subprocess.call(f"...", shell=True, ...)
      - pattern: subprocess.Popen(f"...", shell=True, ...)
    message: "コマンドインジェクション: f-string と shell=True の組み合わせは危険です"
    severity: ERROR
    languages: [python]
    metadata:
      cwe: "CWE-78: OS Command Injection"

  - id: path-traversal-open
    patterns:
      - pattern: |
          $PATH = os.path.join(..., $INPUT)
          ...
          open($PATH, ...)
      - pattern-not-inside: |
          if os.path.realpath($PATH).startswith(...):
              ...
    message: "パストラバーサル: ファイルパスを検証してください"
    severity: ERROR
    languages: [python]
    metadata:
      cwe: "CWE-22: Path Traversal"
```
</details>

### 演習 2: CI/CD パイプラインの設計

以下の要件を満たす GitHub Actions ワークフローを設計せよ。

- Python + React (TypeScript) のフルスタックアプリケーション
- PR 時: SAST (Semgrep + Bandit) + SCA (Trivy) + シークレットスキャン
- main マージ時: 上記 + コンテナスキャン + DAST (ZAP Baseline)
- 週次: Full DAST スキャン
- 結果は SARIF 形式で GitHub Security タブに統合

<details>
<summary>解答例 (骨格)</summary>

```yaml
name: Full Security Pipeline
on:
  pull_request:
    branches: [main]
  push:
    branches: [main]
  schedule:
    - cron: '0 2 * * 1'  # 毎週月曜 02:00 UTC

jobs:
  # PR + main: SAST
  sast:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Semgrep
        run: semgrep --config auto --sarif -o semgrep.sarif .
      - name: Bandit
        run: bandit -r backend/ -f sarif -o bandit.sarif
      - name: Upload SARIF
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: semgrep.sarif

  # PR + main: SCA
  sca:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: aquasecurity/trivy-action@master
        with:
          scan-type: fs
          format: sarif
          output: trivy.sarif

  # PR + main: Secret Scan
  secrets:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: gitleaks/gitleaks-action@v2

  # main only: Container Scan
  container:
    if: github.event_name == 'push'
    needs: [sast, sca, secrets]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t app:test .
      - uses: aquasecurity/trivy-action@master
        with:
          image-ref: app:test
          severity: CRITICAL,HIGH

  # main only: DAST Baseline
  dast-baseline:
    if: github.event_name == 'push'
    needs: container
    runs-on: ubuntu-latest
    steps:
      - uses: zaproxy/action-baseline@v0.12.0
        with:
          target: https://staging.example.com

  # Weekly: Full DAST
  dast-full:
    if: github.event_name == 'schedule'
    runs-on: ubuntu-latest
    steps:
      - uses: zaproxy/action-full-scan@v0.10.0
        with:
          target: https://staging.example.com
```
</details>

### 演習 3: トリアージの実践

以下の SAST/DAST スキャン結果を、重大度・対応優先度で分類し、各項目について対応方針を示せ。

| # | ツール | 検出内容 | ファイル |
|---|--------|---------|---------|
| 1 | Semgrep | SQL Injection in user query | src/db/users.py:42 |
| 2 | SonarQube | Hardcoded password in config | src/config.py:15 |
| 3 | ZAP | Missing X-Content-Type-Options header | / |
| 4 | Trivy | CVE-2024-1234 (CVSS 9.8) in log4j | pom.xml |
| 5 | Bandit | Use of assert in production code | src/auth/verify.py:88 |
| 6 | ZAP | Server version disclosed in header | / |
| 7 | Semgrep | eval() with user input | src/api/dynamic.py:23 |
| 8 | gitleaks | AWS Access Key in commit history | - |

<details>
<summary>解答例</summary>

| # | 重大度 | 優先度 | 対応方針 |
|---|--------|--------|---------|
| 1 | Critical | 即時 (24h) | パラメータ化クエリに書き換え。ビルドブロック |
| 2 | High | 即時 (48h) | 環境変数または Secrets Manager に移行 |
| 3 | Low | バックログ | Web サーバー設定で X-Content-Type-Options: nosniff を追加 |
| 4 | Critical | 即時 (24h) | log4j を修正バージョンにアップデート |
| 5 | Low | バックログ | assert を proper な例外処理に置き換え (ただし影響は限定的) |
| 6 | Info | 記録のみ | Server ヘッダの削除を検討 |
| 7 | Critical | 即時 (24h) | eval を安全な代替手段に置き換え |
| 8 | Critical | 即時 | AWS キーを即座に失効させ、新しいキーを発行。BFG で履歴からも削除 |
</details>

---

## 10. FAQ

### Q1. SAST の偽陽性が多すぎる場合はどうするか?

まず、ルールの重大度を HIGH/CRITICAL に絞ってノイズを減らす。次に、偽陽性をインラインコメント (`// NOSONAR`, `// nosemgrep`) で抑制し、その理由を文書化する。チーム全体でトリアージのルールを決め、定期的にルール設定を見直すことが重要である。

段階的導入アプローチも有効で、最初は Critical ルールのみ有効化し、チームが慣れてきたら段階的にルールを追加する。偽陽性率が 30% を超える場合はルールの見直しが必要である。

### Q2. DAST はどの環境で実行すべきか?

本番環境に対して DAST を実行するのはリスクが高い (データ破損やサービス影響)。ステージング環境に本番と同等の構成を用意し、そこで実行するのが標準的である。Baseline Scan (パッシブのみ) であれば本番に対しても比較的安全に実行できる。

環境ごとの推奨スキャンタイプ:
- **開発環境**: 開発者による手動テスト、ZAP プロキシモード
- **ステージング環境**: Full Scan (Active Scan 含む)
- **本番環境**: Baseline Scan のみ (パッシブスキャン)

### Q3. SonarQube と Semgrep のどちらを選ぶべきか?

SonarQube はコード品質全般 (バグ、コードスメル、カバレッジ) を統合管理でき、ダッシュボードが充実している。Semgrep はセキュリティに特化し、カスタムルール作成が容易で実行速度が速い。両者は補完関係にあり、併用するのが理想的である。

- **小規模チーム / スタートアップ**: Semgrep のみで始めて十分
- **中-大規模組織**: SonarQube (品質管理) + Semgrep (セキュリティ特化)
- **GitHub 中心のワークフロー**: CodeQL + Semgrep

### Q4. SAST/DAST の導入に組織の抵抗がある場合は?

段階的に導入することが重要である。

1. **啓蒙フェーズ**: セキュリティインシデントの事例を共有し、必要性を認識してもらう
2. **パイロットフェーズ**: 1つのプロジェクトで導入し、効果を実証する
3. **展開フェーズ**: 成功事例をもとに他プロジェクトに展開する
4. **最初は「情報提供モード」**: ビルドをブロックせず、レポートのみ出力する
5. **安定後にゲート化**: 偽陽性が十分に管理できたらビルドブロックを有効化する

### Q5. IAST は導入すべきか?

IAST は SAST と DAST の弱点を補完する優れた技術だが、導入コストが高い (商用ツールが多い) ことと、本番環境でのパフォーマンスオーバーヘッドを考慮する必要がある。

- **推奨する場合**: セキュリティ要件が厳しい (金融、医療)、SAST/DAST の偽陽性に悩んでいる
- **推奨しない場合**: 予算が限られている、SAST + DAST で十分なカバレッジが得られている

### Q6. スキャン時間が CI/CD パイプラインを遅くする場合は?

以下の最適化を検討する:

1. **差分スキャン**: 変更されたファイルのみスキャン (`semgrep --baseline-commit`)
2. **並列実行**: SAST/SCA/シークレットスキャンを並列に実行
3. **キャッシュ活用**: SonarQube のインクリメンタル解析を有効化
4. **スキャン分離**: PR 時は SAST のみ、マージ時に Full Scan
5. **週次スキャン**: Full DAST は週次スケジュールで実行

### Q7. マイクロサービスアーキテクチャでの SAST/DAST 戦略は?

マイクロサービス環境では、サービスごとに SAST を実行し、結合テスト環境で DAST を実行する。API Gateway を経由したテストと、サービス間通信のテストの両方が必要である。

```
API Gateway → DAST (外部向けAPI)
    |
    +-- Service A → SAST (個別)
    +-- Service B → SAST (個別)
    +-- Service C → SAST (個別)
    |
API 間通信 → Contract Testing + DAST
```

### Q8. コンプライアンス要件 (PCI DSS, SOC2等) で SAST/DAST は必須か?

主要なコンプライアンスフレームワークにおけるセキュリティテスト要件:

| フレームワーク | SAST | DAST | ペネトレーションテスト |
|---------------|------|------|--------------------|
| PCI DSS v4.0 | 推奨 | 推奨 | 必須 (年次) |
| SOC 2 Type II | 推奨 | 推奨 | 推奨 |
| HIPAA | 推奨 | 推奨 | 推奨 |
| ISO 27001 | 推奨 | 推奨 | 推奨 |
| NIST SP 800-53 | 必須 (SA-11) | 必須 (SA-11) | 必須 (CA-8) |

---

## まとめ

| 項目 | 要点 |
|------|------|
| SAST | コード内の脆弱性パターンを早期検出 (Semgrep, SonarQube, CodeQL) |
| DAST | 実行中アプリへの攻撃シミュレーション (OWASP ZAP, Nuclei) |
| IAST | SAST + DAST のハイブリッド、低偽陽性率 (Contrast, Hdiv) |
| SCA | 依存ライブラリの既知脆弱性を検出 (Trivy) |
| シークレットスキャン | コードに埋め込まれた秘密を検出 (gitleaks) |
| CI/CD 統合 | SAST→コンテナスキャン→DAST の段階的パイプライン |
| トリアージ | 重大度別の対応 SLA を設定し偽陽性を管理 |
| Shift Left | 開発ライフサイクルの早期にテストを配置し修正コストを削減 |
| 段階的導入 | Critical ルールから始め、徐々にルールとゲートを厳格化 |

---

## 次に読むべきガイド

- [セキュアコーディング](./00-secure-coding.md) — SAST が検出する脆弱性の根本対策
- [依存関係セキュリティ](./01-dependency-security.md) — SCA の詳細
- [コンテナセキュリティ](./02-container-security.md) — コンテナイメージのスキャン

---

## 参考文献

1. **OWASP Testing Guide** — https://owasp.org/www-project-web-security-testing-guide/
2. **OWASP ZAP Documentation** — https://www.zaproxy.org/docs/
3. **Semgrep Documentation** — https://semgrep.dev/docs/
4. **NIST SP 800-218 — Secure Software Development Framework** — https://csrc.nist.gov/publications/detail/sp/800-218/final
5. **SonarQube Documentation** — https://docs.sonarqube.org/latest/
6. **CodeQL Documentation** — https://codeql.github.com/docs/
7. **Gitleaks Documentation** — https://github.com/gitleaks/gitleaks
8. **OWASP SAST Tools** — https://owasp.org/www-community/Source_Code_Analysis_Tools
9. **OWASP DAST Tools** — https://owasp.org/www-community/Vulnerability_Scanning_Tools
10. **NIST SP 800-53 Rev.5 — Security and Privacy Controls** — https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final
