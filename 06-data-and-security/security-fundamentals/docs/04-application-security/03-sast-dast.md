# SAST/DAST

> 静的解析 (SAST) と動的解析 (DAST) の特性を理解し、SonarQube や OWASP ZAP を活用して CI/CD パイプラインにセキュリティテストを組み込むガイド

## この章で学ぶこと

1. **SAST の原理と実践** — ソースコードの静的解析による脆弱性の早期発見
2. **DAST の原理と実践** — 実行中アプリケーションに対する動的なセキュリティテスト
3. **CI/CD 統合** — セキュリティテストを開発フローに自然に組み込む方法

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
+----------------------------------------------------------+
```

### SAST vs DAST 比較

| 項目 | SAST | DAST |
|------|------|------|
| 解析対象 | ソースコード / バイトコード | 実行中のアプリケーション |
| 実行タイミング | 開発中 / コミット時 | デプロイ後 / ステージング |
| 検出できる脆弱性 | インジェクション、ハードコード秘密、安全でない関数 | XSS、認証不備、設定ミス |
| 偽陽性 | 多い (30-70%) | 少ない |
| 偽陰性 | ビジネスロジック脆弱性を見逃す | コード内部の問題を見逃す |
| 言語依存 | あり (言語別パーサ) | なし (プロトコルベース) |
| 修正の容易さ | 行番号特定で容易 | 根本原因特定が難しい場合あり |
| 速度 | 中程度 (分-時間) | 遅い (時間単位) |

---

## 2. SAST の実践

### SonarQube によるコード解析

```yaml
# sonar-project.properties
sonar.projectKey=myapp
sonar.projectName=My Application
sonar.sources=src
sonar.tests=tests
sonar.language=js
sonar.javascript.lcov.reportPaths=coverage/lcov.info

# セキュリティルールの重点設定
sonar.issue.ignore.multicriteria=e1
sonar.issue.ignore.multicriteria.e1.ruleKey=javascript:S1234
sonar.issue.ignore.multicriteria.e1.resourceKey=**/test/**
```

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
```

### Semgrep (軽量 SAST)

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

  - id: sql-injection
    patterns:
      - pattern: |
          cursor.execute(f"... {$VAR} ...")
    message: "SQL インジェクションの可能性: パラメータ化クエリを使用してください"
    severity: ERROR
    languages: [python]

  - id: unsafe-deserialization
    pattern: pickle.loads(...)
    message: "安全でないデシリアライゼーション: 信頼できないデータに pickle を使用しないでください"
    severity: WARNING
    languages: [python]
```

```bash
# Semgrep の実行
semgrep --config auto .                    # 自動ルール選択
semgrep --config .semgrep.yml .            # カスタムルール
semgrep --config p/owasp-top-ten .         # OWASP Top 10 ルール
semgrep --config p/javascript .            # JavaScript 専用ルール

# CI/CD ゲート (エラーがあればビルド失敗)
semgrep --config auto --error .
```

### SAST ツールの比較

| ツール | 対応言語 | 速度 | カスタムルール | コスト |
|--------|---------|------|-------------|-------|
| SonarQube | 30+ | 中 | はい | CE: 無料 |
| Semgrep | 30+ | 高速 | YAML ベース | OSS: 無料 |
| CodeQL | 10+ | 遅い | QL 言語 | GitHub 無料 |
| Bandit | Python | 高速 | プラグイン | 無料 |
| ESLint Security | JavaScript | 高速 | ルールベース | 無料 |
| gosec | Go | 高速 | AST ベース | 無料 |

---

## 3. DAST の実践

### OWASP ZAP による動的テスト

```
ZAP のテストフロー:

  +-- Spider (クロール) --+
  |  サイトマップ構築      |
  +-----------------------+
            |
            v
  +-- Passive Scan -------+
  |  通信を観察して検出     |
  |  (速い、低リスク)      |
  +-----------------------+
            |
            v
  +-- Active Scan ---------+
  |  攻撃リクエスト送信     |
  |  (遅い、サーバ負荷あり) |
  +------------------------+
            |
            v
  +-- レポート生成 --------+
  |  HTML / JSON / XML     |
  +------------------------+
```

### ZAP の API を使った自動テスト

```python
from zapv2 import ZAPv2
import time

# ZAP に接続
zap = ZAPv2(apikey='your-api-key', proxies={
    'http': 'http://localhost:8080',
    'https': 'http://localhost:8080',
})

target = 'https://staging.example.com'

def run_zap_scan(target_url):
    """ZAP による自動セキュリティスキャン"""

    # Step 1: Spider (クロール)
    print("Spidering target...")
    scan_id = zap.spider.scan(target_url)
    while int(zap.spider.status(scan_id)) < 100:
        time.sleep(2)
    print(f"Spider found {len(zap.spider.results(scan_id))} URLs")

    # Step 2: Passive Scan の完了を待つ
    while int(zap.pscan.records_to_scan) > 0:
        time.sleep(1)

    # Step 3: Active Scan
    print("Active scanning...")
    scan_id = zap.ascan.scan(target_url)
    while int(zap.ascan.status(scan_id)) < 100:
        time.sleep(5)
        print(f"  Progress: {zap.ascan.status(scan_id)}%")

    # Step 4: 結果を取得
    alerts = zap.core.alerts(baseurl=target_url)
    high_alerts = [a for a in alerts if a['risk'] == 'High']
    medium_alerts = [a for a in alerts if a['risk'] == 'Medium']

    print(f"Results: {len(high_alerts)} High, {len(medium_alerts)} Medium")

    # HTML レポート出力
    with open('zap-report.html', 'w') as f:
        f.write(zap.core.htmlreport())

    return high_alerts, medium_alerts

high, medium = run_zap_scan(target)
if high:
    print("CRITICAL: High-risk vulnerabilities found!")
    exit(1)
```

### ZAP の CI/CD 統合

```yaml
# GitHub Actions での ZAP スキャン
name: DAST Scan
on:
  deployment_status:

jobs:
  zap-scan:
    if: github.event.deployment_status.state == 'success'
    runs-on: ubuntu-latest
    steps:
      - name: ZAP Baseline Scan
        uses: zaproxy/action-baseline@v0.12.0
        with:
          target: ${{ github.event.deployment.payload.url }}
          rules_file_name: '.zap/rules.tsv'
          cmd_options: '-a'

      - name: ZAP Full Scan (ステージング環境のみ)
        if: contains(github.event.deployment.environment, 'staging')
        uses: zaproxy/action-full-scan@v0.10.0
        with:
          target: ${{ github.event.deployment.payload.url }}
```

---

## 4. CI/CD パイプラインへの統合

### セキュリティテストの配置

```
開発者の PC    →    CI/CD Pipeline    →    ステージング    →    本番
    |                    |                      |                |
 [pre-commit]      [ビルド時]              [デプロイ後]      [継続的]
    |                    |                      |                |
  Semgrep          SonarQube              OWASP ZAP         ランタイム
  (即時)           Semgrep                Nuclei             監視
                   SCA (Trivy)            (DAST)             (IAST)
                   シークレットスキャン
                   (SAST + SCA)
```

### 統合パイプラインの例

```yaml
# .github/workflows/security-pipeline.yml
name: Security Pipeline
on: [push, pull_request]

jobs:
  # Stage 1: 静的解析 (並列実行)
  sast:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        tool: [semgrep, sonarqube, trivy-fs]
    steps:
      - uses: actions/checkout@v4

      - name: Semgrep
        if: matrix.tool == 'semgrep'
        run: |
          pip install semgrep
          semgrep --config auto --error --json -o semgrep-results.json .

      - name: Trivy filesystem scan
        if: matrix.tool == 'trivy-fs'
        run: |
          trivy fs --severity HIGH,CRITICAL --exit-code 1 .

  # Stage 2: コンテナスキャン
  container-scan:
    needs: sast
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t myapp:${{ github.sha }} .
      - name: Trivy image scan
        run: trivy image --exit-code 1 --severity CRITICAL myapp:${{ github.sha }}

  # Stage 3: DAST (ステージング環境)
  dast:
    needs: container-scan
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging
        run: echo "Deploy to staging..."

      - name: ZAP Baseline Scan
        uses: zaproxy/action-baseline@v0.12.0
        with:
          target: 'https://staging.example.com'
```

---

## 5. シークレットスキャン

### git-secrets / gitleaks の活用

```bash
# gitleaks のインストールと実行
gitleaks detect --source . --report-format json --report-path gitleaks-report.json

# pre-commit hook として設定
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks
```

```toml
# .gitleaks.toml - カスタムルール
[allowlist]
paths = [
    '''test/.*''',
    '''.*_test\.go''',
]

[[rules]]
id = "aws-access-key"
description = "AWS Access Key ID"
regex = '''AKIA[0-9A-Z]{16}'''
tags = ["aws", "credentials"]

[[rules]]
id = "generic-api-key"
description = "Generic API Key"
regex = '''(?i)(api[_-]?key|apikey)\s*[:=]\s*['"][a-zA-Z0-9]{20,}['"]'''
tags = ["api", "generic"]
```

---

## 6. アンチパターン

### アンチパターン 1: セキュリティスキャンの結果を無視

```
NG: スキャン結果が 200 件の警告を出すが全て無視
  → 偽陽性と本物の脆弱性が混在し、全てが放置される

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

---

## 7. FAQ

### Q1. SAST の偽陽性が多すぎる場合はどうするか?

まず、ルールの重大度を HIGH/CRITICAL に絞ってノイズを減らす。次に、偽陽性をインラインコメント (`// NOSONAR`, `// nosemgrep`) で抑制し、その理由を文書化する。チーム全体でトリアージのルールを決め、定期的にルール設定を見直すことが重要である。

### Q2. DAST はどの環境で実行すべきか?

本番環境に対して DAST を実行するのはリスクが高い (データ破損やサービス影響)。ステージング環境に本番と同等の構成を用意し、そこで実行するのが標準的である。Baseline Scan (パッシブのみ) であれば本番に対しても比較的安全に実行できる。

### Q3. SonarQube と Semgrep のどちらを選ぶべきか?

SonarQube はコード品質全般 (バグ、コードスメル、カバレッジ) を統合管理でき、ダッシュボードが充実している。Semgrep はセキュリティに特化し、カスタムルール作成が容易で実行速度が速い。両者は補完関係にあり、併用するのが理想的である。

---

## まとめ

| 項目 | 要点 |
|------|------|
| SAST | コード内の脆弱性パターンを早期検出 (Semgrep, SonarQube) |
| DAST | 実行中アプリへの攻撃シミュレーション (OWASP ZAP) |
| SCA | 依存ライブラリの既知脆弱性を検出 (Trivy) |
| シークレットスキャン | コードに埋め込まれた秘密を検出 (gitleaks) |
| CI/CD 統合 | SAST→コンテナスキャン→DAST の段階的パイプライン |
| トリアージ | 重大度別の対応 SLA を設定し偽陽性を管理 |

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
