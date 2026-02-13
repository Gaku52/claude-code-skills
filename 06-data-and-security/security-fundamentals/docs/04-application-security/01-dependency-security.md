# 依存関係セキュリティ

> SCA (Software Composition Analysis) による脆弱性検出、Dependabot による自動更新、SBOM による可視化まで、サードパーティ依存関係のリスクを管理するガイド。サプライチェーン攻撃の内部メカニズム、推移的依存関係の深層分析、規制対応まで網羅する。

## この章で学ぶこと

1. **サプライチェーン攻撃の脅威モデル** — タイポスクワッティング、依存関係混乱攻撃、アカウント乗っ取りの手法と防御策
2. **SCA ツールの活用と CI/CD 統合** — Dependabot、Snyk、Trivy による脆弱性の自動検出・修正・ゲーティング
3. **SBOM (Software Bill of Materials)** — ソフトウェア部品表の生成・管理・規制対応
4. **依存関係のロック・固定・監査** — 再現可能なビルドとライセンスコンプライアンスの確保

## 前提知識

| トピック | 参照先 |
|---------|--------|
| セキュアコーディングの基礎 | [セキュアコーディング](./00-secure-coding.md) |
| パッケージマネージャの基本 | npm, pip, Go modules の基礎知識 |
| CI/CD パイプラインの概念 | [コンテナセキュリティ](./02-container-security.md) |
| 暗号化とハッシュの基礎 | [暗号化基礎](../02-cryptography/) |

---

## 1. 依存関係のリスク

### WHY: なぜ依存関係セキュリティが重要か

現代のソフトウェアは平均して 80-90% がオープンソースの依存関係で構成されている。あなたが書いたコードが 10% でも、残り 90% のサードパーティコードに脆弱性があれば、アプリケーション全体が危険にさらされる。Log4Shell (CVE-2021-44228) は、ほぼ全ての Java アプリケーションに影響を与え、依存関係管理の不備がいかに壊滅的な結果をもたらすかを世界に示した。

### サプライチェーン攻撃の手法と内部メカニズム

```
┌──────────────────────────────────────────────────────────────┐
│              サプライチェーン攻撃のベクター                       │
│──────────────────────────────────────────────────────────────│
│                                                              │
│  [直接攻撃]                                                   │
│  +-- 依存パッケージの乗っ取り (アカウント侵害)                   │
│  │   → メンテナの npm/PyPI アカウントを侵害                     │
│  │   → 正規パッケージに悪意あるコードを注入                      │
│  │   → 例: ua-parser-js (2021) 週間DL 800万のパッケージ侵害     │
│  │                                                            │
│  +-- タイポスクワッティング (lodash → lodahs, reqeusts)         │
│  │   → 類似名パッケージを公開して誤インストールを誘導             │
│  │   → PyPI では平均して月100件以上の悪意あるパッケージ発見       │
│  │                                                            │
│  +-- 依存関係の混乱攻撃 (Dependency Confusion)                 │
│      → 内部パッケージ名と同名の公開パッケージを高バージョンで公開  │
│      → パッケージマネージャが公開版を優先してインストール          │
│      → 例: Alex Birsan が Apple/Microsoft/PayPal で実証 (2021) │
│                                                              │
│  [間接攻撃]                                                   │
│  +-- 推移的依存関係の脆弱性 (A→B→C、Cに脆弱性)                 │
│  +-- ビルドシステムの侵害 (CI/CD パイプライン)                   │
│  +-- 悪意あるプレ/ポストインストールスクリプト                    │
│  +-- ソーシャルエンジニアリングによるメンテナ権限取得              │
│                                                              │
│  [既知の重大事例]                                               │
│  +-- event-stream (2018): 暗号通貨窃取コード注入                │
│  +-- ua-parser-js (2021): マイナー & パスワード窃取              │
│  +-- Log4Shell (2021): Log4j のリモートコード実行               │
│  +-- colors/faker (2022): メンテナによる意図的破壊               │
│  +-- xz-utils (2024): 2年がかりのバックドア挿入                 │
└──────────────────────────────────────────────────────────────┘
```

### 依存関係の深さの問題

```
あなたのアプリケーション
  ├── express (直接依存: 1個)
  │   ├── body-parser
  │   │   ├── bytes
  │   │   ├── content-type
  │   │   ├── raw-body
  │   │   │   └── bytes, iconv-lite, unpipe
  │   │   └── ...
  │   ├── cookie
  │   ├── debug
  │   │   └── ms
  │   ├── ...
  │   └── (推移的依存: 30+ パッケージ)
  │
  直接依存 1 個 → 推移的依存 30+ 個
  典型的な Node.js アプリ: 直接依存 20個 → 推移的依存 1000+ 個
  典型的な Java アプリ: 直接依存 50個 → 推移的依存 500+ 個
```

### 依存関係混乱攻撃の仕組み（詳細）

```python
# ========================================
# 依存関係混乱攻撃のシナリオ
# ========================================

# 1. 社内で "mycompany-utils" というパッケージを内部レジストリで使用
# requirements.txt:
#   mycompany-utils==1.0.0

# 2. 攻撃者が PyPI に "mycompany-utils" を version 99.0.0 で公開
#    → setup.py に悪意あるインストールスクリプトを仕込む

# 3. pip が公開 PyPI の高バージョン (99.0.0) を優先インストール
# pip install mycompany-utils
# → PyPI の 99.0.0 がインストールされる（内部レジストリの 1.0.0 ではなく）

# ========================================
# 防御策
# ========================================

# 方法1: pip.conf で内部レジストリのみを参照
# [global]
# index-url = https://internal.pypi.mycompany.com/simple/
# extra-index-url = (設定しない → 公開 PyPI を参照しない)

# 方法2: .npmrc でスコープごとにレジストリを指定 (npm)
# @mycompany:registry=https://npm.mycompany.com/
# registry=https://registry.npmjs.org/

# 方法3: PyPI にプレースホルダパッケージを公開
#   → 内部パッケージと同名の空パッケージを公開し、攻撃者の先手を打つ

# 方法4: ハッシュ検証で予期しないパッケージの変更を検出
# pip install --require-hashes -r requirements.txt
# requirements.txt:
#   mycompany-utils==1.0.0 \
#     --hash=sha256:abc123def456...
```

---

## 2. SCA (Software Composition Analysis)

### SCA ツールの比較

| ツール | 対応言語 | 無料枠 | 特徴 | 脆弱性DB | 修正PR自動生成 |
|--------|---------|--------|------|---------|--------------|
| Dependabot | 多言語 | GitHub 無料 | GitHub ネイティブ統合 | GitHub Advisory DB | あり |
| Snyk | 多言語 | OSS 無料 | 優先順位付きの修正提案 | Snyk DB | あり |
| Trivy | 多言語 + コンテナ | 完全無料 | コンテナイメージ対応 | NVD + 独自 | なし |
| OWASP Dep-Check | Java, .NET 中心 | 完全無料 | NVD データベース連携 | NVD | なし |
| npm audit | JavaScript | 無料 | npm 標準機能 | GitHub Advisory DB | `npm audit fix` |
| pip-audit | Python | 無料 | OSV データベース連携 | OSV | なし |
| Renovate | 多言語 | 完全無料 | 高度なカスタマイズ | 多数 | あり |

### Dependabot の設定（詳細版）

```yaml
# .github/dependabot.yml
version: 2
updates:
  # ========================================
  # npm 依存関係
  # ========================================
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
      timezone: "Asia/Tokyo"
    open-pull-requests-limit: 10
    reviewers:
      - "security-team"
    labels:
      - "dependencies"
      - "security"
    # セキュリティアップデートは即座に、バージョンアップデートはグループ化
    groups:
      production-dependencies:
        dependency-type: "production"
        update-types:
          - "minor"
          - "patch"
      dev-dependencies:
        dependency-type: "development"
    ignore:
      # メジャーバージョンアップは手動で対応
      - dependency-name: "*"
        update-types: ["version-update:semver-major"]
    # セキュリティアドバイザリ対応は例外（メジャーでも自動PR）
    # → GitHub Security Advisories が自動的に処理

  # ========================================
  # Python 依存関係
  # ========================================
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    groups:
      python-deps:
        patterns: ["*"]
        update-types: ["minor", "patch"]

  # ========================================
  # Docker イメージ
  # ========================================
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
    # ベースイメージの更新は必ず追従
    ignore: []

  # ========================================
  # GitHub Actions のバージョン管理
  # ========================================
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    # Actions のバージョン固定は特に重要（サプライチェーン攻撃対策）
```

### GitHub Security Advisories の自動スキャンパイプライン

```yaml
# .github/workflows/security-scan.yml
name: Security Scan
on:
  push:
    branches: [main]
  pull_request:
  schedule:
    - cron: '0 0 * * *'  # 毎日 0:00 UTC

permissions:
  contents: read
  security-events: write

jobs:
  # ========================================
  # npm audit
  # ========================================
  npm-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install dependencies
        run: npm ci

      - name: npm audit (high/critical only)
        run: npm audit --audit-level=high

  # ========================================
  # Trivy filesystem scan
  # ========================================
  trivy-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Trivy vulnerability scan
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          severity: 'HIGH,CRITICAL'
          exit-code: '1'  # 脆弱性発見時にビルド失敗
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload Trivy scan results to Security tab
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'

  # ========================================
  # pip-audit (Python)
  # ========================================
  pip-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install pip-audit
        run: pip install pip-audit

      - name: Run pip-audit
        run: pip-audit -r requirements.txt --desc --fix --dry-run
```

### Trivy によるスキャン（詳細）

```bash
# ========================================
# 基本的なスキャンコマンド
# ========================================

# ファイルシステムスキャン (プロジェクト全体)
trivy fs --severity HIGH,CRITICAL .

# コンテナイメージスキャン
trivy image --severity HIGH,CRITICAL myapp:latest

# 特定の脆弱性を無視（正当な理由がある場合）
trivy fs --severity HIGH,CRITICAL --ignore-unfixed .

# JSON 形式で出力（CI/CD パイプライン向け）
trivy fs --format json --output results.json .

# SBOM 出力付きスキャン
trivy fs --format cyclonedx --output sbom.json .

# ========================================
# 出力例
# ========================================
# myapp (npm)
# ============
# Total: 5 (HIGH: 3, CRITICAL: 2)
#
# ┌──────────────┬───────────────────┬──────────┬────────────┬────────────┐
# │   Library    │   Vulnerability   │ Severity │ Installed  │   Fixed    │
# ├──────────────┼───────────────────┼──────────┼────────────┼────────────┤
# │ lodash       │ CVE-2021-23337    │ HIGH     │ 4.17.20    │ 4.17.21    │
# │ express      │ CVE-2024-XXXX     │ CRITICAL │ 4.17.1     │ 4.18.2     │
# │ jsonwebtoken │ CVE-2022-23529    │ HIGH     │ 8.5.1      │ 9.0.0      │
# │ axios        │ CVE-2023-45857    │ HIGH     │ 1.5.0      │ 1.6.0      │
# │ node-forge   │ CVE-2022-24771    │ CRITICAL │ 1.2.1      │ 1.3.0      │
# └──────────────┴───────────────────┴──────────┴────────────┴────────────┘

# ========================================
# .trivyignore — 正当な理由で無視する脆弱性
# ========================================
# CVE-2021-23337  # lodash: 本アプリでは該当コードパスを使用していない
# CVE-2023-XXXXX  # テスト用依存関係のため本番影響なし
```

### Snyk による統合スキャン

```bash
# Snyk CLIのインストールと使用
npm install -g snyk

# プロジェクトのスキャン
snyk test

# 監視モード（新規脆弱性の継続的検出）
snyk monitor

# 修正可能な脆弱性の自動修正
snyk fix

# コンテナイメージのスキャン
snyk container test myapp:latest

# IaC のスキャン
snyk iac test terraform/
```

```python
# Snyk API を使った自動脆弱性レポート生成
import requests
import json
from datetime import datetime

def generate_vulnerability_report(org_id: str, api_token: str) -> dict:
    """Snyk API から組織全体の脆弱性レポートを生成"""
    headers = {
        'Authorization': f'token {api_token}',
        'Content-Type': 'application/json',
    }

    # 全プロジェクトの脆弱性を取得
    response = requests.get(
        f'https://api.snyk.io/rest/orgs/{org_id}/issues',
        headers=headers,
        params={'version': '2024-01-01', 'limit': 100},
    )
    response.raise_for_status()
    issues = response.json()

    # 重大度別に集計
    severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
    for issue in issues.get('data', []):
        severity = issue['attributes']['effective_severity_level']
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    return {
        'timestamp': datetime.utcnow().isoformat(),
        'total_issues': len(issues.get('data', [])),
        'severity_breakdown': severity_counts,
        'org_id': org_id,
    }
```

---

## 3. SBOM (Software Bill of Materials)

### SBOM の WHY: なぜ部品表が必要か

SBOM は「ソフトウェアの成分表示」である。食品に原材料表示が義務付けられているように、ソフトウェアにも「何で作られているか」の透明性が求められている。Log4Shell の際、多くの組織が「自社のどのシステムが Log4j を使っているか」を把握できず、対応に数週間を要した。SBOM があれば、脆弱性が公開された瞬間に影響範囲を即座に特定できる。

```
┌──────────────────────────────────────────────────────────────┐
│                      SBOM (部品表)                             │
│──────────────────────────────────────────────────────────────│
│  アプリケーション: MyApp v2.1.0                                │
│  ビルド日時: 2024-03-15T10:30:00Z                             │
│  ビルド環境: Node.js 20.11.0 / npm 10.2.4                     │
│                                                              │
│  コンポーネント一覧:                                           │
│  ┌───────────────────────┬─────────┬──────┬────────────────┐ │
│  │ パッケージ名           │ バージョン│ ライセンス│ 種別           │ │
│  ├───────────────────────┼─────────┼──────┼────────────────┤ │
│  │ express               │ 4.18.2  │ MIT  │ 直接依存       │ │
│  │ lodash                │ 4.17.21 │ MIT  │ 直接依存       │ │
│  │ body-parser           │ 1.20.2  │ MIT  │ 推移的依存     │ │
│  │ debug                 │ 4.3.4   │ MIT  │ 推移的依存     │ │
│  │ ...                   │ ...     │ ...  │ ...            │ │
│  └───────────────────────┴─────────┴──────┴────────────────┘ │
│                                                              │
│  各コンポーネントに含まれる情報:                                │
│  - パッケージ名・バージョン                                    │
│  - ライセンス (SPDX 識別子)                                   │
│  - 供給元 (サプライヤー)                                      │
│  - 既知の脆弱性 (CVE)                                        │
│  - 暗号ハッシュ (改竄検知)                                    │
│  - 依存関係の親子関係ツリー                                    │
└──────────────────────────────────────────────────────────────┘
```

### SBOM フォーマットの比較

| 項目 | SPDX | CycloneDX |
|------|------|-----------|
| 策定 | Linux Foundation | OWASP |
| ISO 標準 | ISO/IEC 5962:2021 | ECMA-424 |
| フォーマット | JSON, RDF, Tag-Value, YAML | JSON, XML, Protobuf |
| 重点 | ライセンスコンプライアンス | セキュリティ |
| 脆弱性情報 | 外部参照 | VEX (Vulnerability Exploitability eXchange) 統合 |
| サービス記述 | 限定的 | API/サービスも記述可能 |
| 生成ツール | syft, trivy, spdx-tools | cdxgen, trivy, syft |
| 推奨用途 | ライセンス監査、法務 | セキュリティ運用、脆弱性管理 |

### SBOM の生成と活用

```bash
# ========================================
# SBOM 生成ツール
# ========================================

# syft で CycloneDX 形式の SBOM を生成
syft dir:. -o cyclonedx-json > sbom.cyclonedx.json

# syft で SPDX 形式の SBOM を生成
syft dir:. -o spdx-json > sbom.spdx.json

# Trivy で SBOM 生成
trivy fs --format cyclonedx --output sbom.json .

# コンテナイメージから SBOM 生成
syft myapp:latest -o cyclonedx-json > image-sbom.json

# npm で SBOM を生成 (npm 10+)
npm sbom --sbom-format cyclonedx

# cdxgen で SBOM 生成（複数言語対応）
npx @cyclonedx/cdxgen -o sbom.json

# ========================================
# SBOM からの脆弱性スキャン
# ========================================

# grype で SBOM から脆弱性をスキャン
grype sbom:sbom.cyclonedx.json

# Trivy で SBOM スキャン
trivy sbom sbom.cyclonedx.json

# ========================================
# SBOM の検証
# ========================================

# CycloneDX の形式検証
cyclonedx validate --input-file sbom.json --input-format json
```

```python
# SBOM を CI/CD で管理するスクリプト
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

def generate_and_analyze_sbom(
    project_dir: str,
    output_dir: str,
    severity_threshold: str = "HIGH",
) -> dict:
    """SBOM を生成して脆弱性分析を実行"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%S')

    # Step 1: CycloneDX SBOM を生成
    sbom_file = output_path / f"sbom-{timestamp}.json"
    result = subprocess.run(
        ["syft", f"dir:{project_dir}", "-o", "cyclonedx-json"],
        capture_output=True, text=True, check=True,
    )
    sbom = json.loads(result.stdout)
    sbom_file.write_text(json.dumps(sbom, indent=2))

    # Step 2: 脆弱性スキャン
    vuln_file = output_path / f"vulnerabilities-{timestamp}.json"
    vuln_result = subprocess.run(
        ["grype", f"sbom:{sbom_file}", "-o", "json"],
        capture_output=True, text=True,
    )
    vulnerabilities = json.loads(vuln_result.stdout)
    vuln_file.write_text(json.dumps(vulnerabilities, indent=2))

    # Step 3: 分析結果の集計
    matches = vulnerabilities.get("matches", [])
    severity_counts = {}
    for match in matches:
        sev = match.get("vulnerability", {}).get("severity", "unknown")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    # Step 4: ライセンス分析
    components = sbom.get("components", [])
    license_counts = {}
    for comp in components:
        for lic in comp.get("licenses", []):
            lic_id = lic.get("license", {}).get("id", "unknown")
            license_counts[lic_id] = license_counts.get(lic_id, 0) + 1

    return {
        "sbom_path": str(sbom_file),
        "vulnerability_path": str(vuln_file),
        "component_count": len(components),
        "vulnerability_count": len(matches),
        "severity_breakdown": severity_counts,
        "license_breakdown": license_counts,
        "timestamp": timestamp,
    }
```

### VEX (Vulnerability Exploitability eXchange)

```json
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.5",
  "vulnerabilities": [
    {
      "id": "CVE-2021-23337",
      "source": { "name": "NVD" },
      "analysis": {
        "state": "not_affected",
        "justification": "code_not_reachable",
        "detail": "lodash.template() は本アプリケーションで使用していないため影響なし。コードパス分析により確認済み。",
        "response": ["will_not_fix"]
      },
      "affects": [
        {
          "ref": "pkg:npm/lodash@4.17.20"
        }
      ]
    }
  ]
}
```

---

## 4. 依存関係のロックと固定

### ロックファイルの WHY

```
┌──────────────────────────────────────────────────────────────┐
│  ロックファイルなし:                                           │
│  package.json: "lodash": "^4.17.0"                           │
│  → 開発環境: 4.17.20 (ある時点のインストール)                  │
│  → CI 環境:  4.17.21 (最新パッチ)                             │
│  → 本番環境: 4.17.19 (キャッシュからインストール)              │
│  → 環境によってバージョンが異なり再現性がない                   │
│  → 本番でのみ発生する脆弱性を見逃す可能性                      │
│                                                              │
│  ロックファイルあり:                                           │
│  package-lock.json: "lodash": "4.17.21"                      │
│  → 全環境で 4.17.21 を使用 (完全な再現性)                     │
│  → 依存関係の推移的バージョンも固定                             │
│  → ハッシュ値で改竄を検知                                      │
└──────────────────────────────────────────────────────────────┘
```

### 言語別ロックファイルとベストプラクティス

```bash
# ========================================
# JavaScript / Node.js
# ========================================
# CI では npm ci を使用 (lock ファイルに厳密に従う)
npm ci  # npm install ではなく ci を使用

# lock ファイルは必ず Git 管理
git add package-lock.json
git commit -m "Update lock file"

# ========================================
# Python
# ========================================
# pip-compile で requirements.txt をハッシュ付きで生成
pip-compile --generate-hashes requirements.in > requirements.txt

# ハッシュ検証付きインストール
pip install --require-hashes -r requirements.txt

# ========================================
# Go
# ========================================
# go.sum が自動生成される (チェックサム検証)
go mod verify

# 依存関係の整理
go mod tidy

# ========================================
# Rust
# ========================================
# Cargo.lock は必ずコミット
cargo build  # Cargo.lock 自動生成

# 依存関係の監査
cargo audit
```

### 推移的依存関係の強制バージョン指定

```json
// package.json — npm overrides
{
  "overrides": {
    "lodash": "4.17.21",
    "minimist": ">=1.2.6",
    "json5": ">=2.2.2"
  }
}
```

```json
// package.json — yarn resolutions
{
  "resolutions": {
    "lodash": "4.17.21",
    "**/minimist": ">=1.2.6"
  }
}
```

```
# pip — constraints.txt
# pip install -c constraints.txt -r requirements.txt
cryptography>=41.0.0
urllib3>=2.0.7
certifi>=2023.7.22
```

---

## 5. ライセンスコンプライアンス

### WHY: なぜライセンスチェックが必要か

オープンソースのライセンスには、商用利用制限やソースコード公開義務を課すものがある。GPL ライセンスのコードを商用製品に含めると、製品全体のソースコード公開が求められる可能性がある。知らずに含めた推移的依存関係が原因で法的問題になったケースもある。

### ライセンスリスク分類

| リスクレベル | ライセンス | 条件 | 商用利用 |
|-------------|----------|------|---------|
| 低リスク | MIT, BSD-2, ISC | 著作権表示のみ | 自由 |
| 低リスク | Apache-2.0 | 著作権+特許条項 | 自由 |
| 中リスク | LGPL-2.1/3.0 | 動的リンクなら可 | 条件付き |
| 高リスク | GPL-2.0/3.0 | 派生物も GPL | 制限あり |
| 高リスク | AGPL-3.0 | ネットワーク経由でも公開義務 | 厳しい制限 |
| 不明 | ライセンスなし | 著作権法により使用不可 | 使用禁止 |

```bash
# ========================================
# ライセンスチェックの自動化
# ========================================

# JavaScript: license-checker
npx license-checker --summary
npx license-checker --failOn "GPL-3.0;AGPL-3.0;SSPL-1.0"
npx license-checker --production --json > licenses.json

# Python: pip-licenses
pip install pip-licenses
pip-licenses --format=table
pip-licenses --fail-on="GNU General Public License v3 (GPLv3)"
pip-licenses --format=json > licenses.json

# Go: go-licenses
go install github.com/google/go-licenses@latest
go-licenses check ./...
go-licenses csv ./... > licenses.csv

# マルチ言語: FOSSA
# fossa analyze
# fossa test  # ポリシー違反でビルド失敗
```

---

## 6. アンチパターン集

### アンチパターン 1: ロックファイルを Git 管理しない

```bash
# NG: .gitignore にロックファイルを追加
echo "package-lock.json" >> .gitignore
echo "yarn.lock" >> .gitignore

# OK: ロックファイルを必ずコミット
git add package-lock.json
git commit -m "Update lock file"

# CI では npm ci を使用 (lock ファイルに基づく厳密インストール)
# npm install は lock ファイルを更新してしまうため CI では使用しない
```

**影響**: ビルドごとに異なるバージョンが使用され、既知の脆弱性が混入するリスクがある。また、ビルドの再現性が失われデバッグが困難になる。

### アンチパターン 2: 脆弱性アラートの放置

```
NG: Dependabot アラートを無視し続ける
  → 93件の Critical/High 脆弱性が放置
  → 攻撃者が既知の CVE を用いてエクスプロイトを実行
  → CVE 公開から平均15日で攻撃が開始されるという研究結果

OK: SLA を設けて対応
  Critical: 24時間以内に対応
  High:     1週間以内に対応
  Medium:   1ヶ月以内に対応
  Low:      次のスプリントで対応

  対応の優先順位付け:
  1. 攻撃コードが公開されている (Exploit Available)
  2. ネットワーク経由で攻撃可能 (Network Attack Vector)
  3. 本番環境で使用されている依存関係
  4. CVSS スコア
```

### アンチパターン 3: `npm install --ignore-scripts` を恒常的に使用

```bash
# NG: postinstall スクリプトの問題を避けるために常時無視
npm install --ignore-scripts

# → ネイティブモジュールのビルドが行われず本番で障害
# → セキュリティ対策にもなっていない（インストール後に手動で実行する可能性）

# OK: 信頼できるパッケージのみ使用 + npm audit で検証
npm ci
npm audit --audit-level=high
```

---

## 7. 実践演習

### 演習1: npm プロジェクトの脆弱性スキャンと修正（基礎）

**課題**: 以下の package.json を持つプロジェクトについて、脆弱性をスキャンし、修正計画を立てよ。
```json
{
  "dependencies": {
    "express": "4.17.1",
    "lodash": "4.17.19",
    "jsonwebtoken": "8.5.1"
  }
}
```

<details>
<summary>模範解答</summary>

```bash
# Step 1: 脆弱性スキャン
npm audit
# → express: 複数の DoS 脆弱性 (Medium-High)
# → lodash: プロトタイプ汚染 (High)
# → jsonwebtoken: アルゴリズム混同攻撃 (Critical)

# Step 2: 修正可能な脆弱性を自動修正
npm audit fix

# Step 3: メジャーバージョンアップが必要な場合
npm audit fix --force  # 注意: 破壊的変更の可能性あり

# Step 4: 手動で修正計画を作成
# 1. lodash 4.17.19 → 4.17.21 (パッチ、互換性問題なし)
# 2. express 4.17.1 → 4.19.2 (マイナー、テスト必要)
# 3. jsonwebtoken 8.5.1 → 9.0.0 (メジャー、API変更あり)
#    → API の変更点を確認: jose ライブラリへの移行も検討

# Step 5: 修正後の再スキャン
npm audit
# → 0 vulnerabilities found

# Step 6: lock ファイルのコミット
git add package.json package-lock.json
git commit -m "fix: update dependencies to address security vulnerabilities"
```

</details>

### 演習2: SBOM の生成と脆弱性分析（応用）

**課題**: 自身のプロジェクト（または適当なOSSプロジェクト）に対して、以下を実行せよ。
1. CycloneDX 形式の SBOM を生成
2. SBOM から脆弱性をスキャン
3. 検出された脆弱性を重大度別に分類
4. 修正計画を策定（修正版への更新 or VEX による影響なし宣言）

<details>
<summary>模範解答</summary>

```bash
# Step 1: SBOM 生成
syft dir:. -o cyclonedx-json > sbom.json

# Step 2: 脆弱性スキャン
grype sbom:sbom.json -o table

# Step 3: 結果の分析
grype sbom:sbom.json -o json | python3 -c "
import json, sys
data = json.load(sys.stdin)
matches = data.get('matches', [])
severity_map = {}
for m in matches:
    sev = m['vulnerability']['severity']
    severity_map[sev] = severity_map.get(sev, 0) + 1
    pkg = m['artifact']['name']
    ver = m['artifact']['version']
    cve = m['vulnerability']['id']
    fixed = m['vulnerability'].get('fix', {}).get('versions', ['N/A'])
    print(f'{sev:10s} {cve:20s} {pkg}@{ver} → fix: {fixed}')
print()
print('Summary:', severity_map)
"

# Step 4: 修正計画の策定
# Critical: 即座にバージョン更新
# High: 今週中に対応
# Medium: バックログに追加
# 影響なし: VEX ドキュメントを作成して理由を記録
```

</details>

### 演習3: 依存関係混乱攻撃のシミュレーションと防御（発展）

**課題**: 依存関係混乱攻撃のシナリオを理解し、防御策を .npmrc と pip.conf に実装せよ。
- 社内パッケージ名: `@mycompany/auth-utils`（npm）, `mycompany-auth-utils`（pip）
- 内部レジストリ: `https://npm.mycompany.com/`, `https://pypi.mycompany.com/simple/`
- 攻撃者が公開レジストリに同名パッケージを公開するシナリオを想定

<details>
<summary>模範解答</summary>

```ini
# .npmrc — npm の依存関係混乱攻撃防御
# スコープ付きパッケージは内部レジストリを参照
@mycompany:registry=https://npm.mycompany.com/

# 公開パッケージは npm 公式レジストリ
registry=https://registry.npmjs.org/

# パッケージのインテグリティ検証を有効化
package-lock=true
```

```ini
# pip.conf — Python の依存関係混乱攻撃防御
[global]
# 内部パッケージのみを使用する場合:
index-url = https://pypi.mycompany.com/simple/
# 外部パッケージも必要な場合:
extra-index-url = https://pypi.org/simple/
# ただし extra-index-url は混乱攻撃に脆弱

# より安全な方法: 社内 PyPI サーバで全パッケージをプロキシ
# (DevPI や Artifactory を使用)
# index-url = https://devpi.mycompany.com/root/pypi+simple/
```

```python
# 追加防御: 公開レジストリにプレースホルダを登録するスクリプト
# setup.py for placeholder package
from setuptools import setup

setup(
    name="mycompany-auth-utils",
    version="0.0.1",
    description="This is a placeholder package. "
                "Do not install. "
                "See https://internal.mycompany.com/docs",
    author="MyCompany Security Team",
    url="https://mycompany.com",
    python_requires=">=99",  # インストール不能にする
    classifiers=[
        "Development Status :: 7 - Inactive",
    ],
)
```

```yaml
# CI/CD で依存関係のソースを検証
# .github/workflows/dependency-check.yml
name: Dependency Source Check
on: [pull_request]
jobs:
  check-sources:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Verify no unexpected registries
        run: |
          # package-lock.json に予期しないレジストリのURLがないか確認
          if grep -v "registry.npmjs.org\|npm.mycompany.com" package-lock.json | grep -q "resolved.*http"; then
            echo "ERROR: Unexpected registry found in package-lock.json"
            exit 1
          fi
```

</details>

---

## 8. FAQ

### Q1. 推移的依存関係の脆弱性はどう対処するか?

直接依存のバージョンを上げることで推移的依存も更新されるケースが多い。それが不可能な場合は npm の `overrides`、yarn の `resolutions`、pip の constraints で特定バージョンを強制できる。ただし互換性の問題が起きうるためテストを十分に行うこと。最終手段として、脆弱な依存関係を使用しているパッケージの代替を探すか、パッチを当てた fork を作成する。

### Q2. SBOM の提供は義務か?

米国の大統領令 14028 (2021) により、連邦政府向けソフトウェアでは SBOM の提供が求められている。EU のサイバーレジリエンス法 (CRA) でも SBOM が要件化されている。日本では 2023 年に経済産業省が「SBOM 導入の手引き」を公開し、重要インフラ分野での導入を推進している。民間でも取引先からの要求が増加しており、早期の導入が推奨される。

### Q3. 内部パッケージのスコープ保護はどうすればよいか?

npm では `@myorg/` スコープを組織で予約登録する。依存関係混乱攻撃を防ぐため、内部パッケージ名と同名のパッケージを公開レジストリにプレースホルダとして登録する方法がある。.npmrc でレジストリのスコープ設定を正しく行い、CI/CD で package-lock.json の resolved URL を検証することで未知のレジストリからのインストールを検出できる。

### Q4. Dependabot と Renovate のどちらを使うべきか?

Dependabot は GitHub ネイティブで設定が簡単、追加コスト不要で始められる。Renovate はより細かいカスタマイズが可能で、グループ化、自動マージ条件、複数パッケージマネージャの統合管理に優れている。大規模プロジェクトや複雑な依存関係管理には Renovate が適している。両者は併用も可能だが、PR の重複に注意が必要。

### Q5. ゼロデイ脆弱性が発見された場合の緊急対応手順は?

1. SBOM を用いて影響を受けるシステムを即座に特定する。2. WAF ルールや仮想パッチで暫定的に攻撃を遮断する。3. 修正バージョンがリリースされ次第、CI/CD で自動テスト→デプロイする。4. 修正バージョンが存在しない場合、該当コードパスの無効化や代替ライブラリへの切り替えを検討する。5. 事後にインシデントレビューを行い、検出→修正の所要時間を計測し改善する。

---

## まとめ

| 項目 | 要点 | 推奨ツール |
|------|------|-----------|
| サプライチェーンリスク | 推移的依存・タイポスクワッティング・乗っ取りに注意 | - |
| SCA ツール | 脆弱性を自動検出し CI/CD でゲーティング | Dependabot + Trivy |
| SBOM | CycloneDX/SPDX で部品表を生成し脆弱性を追跡 | syft + grype |
| ロックファイル | 必ず Git 管理し CI では厳密インストール | npm ci, pip --require-hashes |
| 脆弱性対応 SLA | Critical 24h、High 1週間の対応基準を設定 | - |
| ライセンス | GPL/AGPL 等の制約を自動チェック | license-checker, pip-licenses |
| 依存関係混乱防御 | スコープ保護、レジストリ設定、プレースホルダ登録 | .npmrc, pip.conf |
| VEX | 影響がない脆弱性を文書化して誤検知を管理 | CycloneDX VEX |

---

## 次に読むべきガイド

- [コンテナセキュリティ](./02-container-security.md) — コンテナイメージの依存関係スキャンと最小化
- [SAST/DAST](./03-sast-dast.md) — コード自体の脆弱性検査と SCA との組み合わせ
- [セキュアコーディング](./00-secure-coding.md) — コードレベルの脆弱性防止
- [IaCセキュリティ](../05-cloud-security/02-infrastructure-as-code-security.md) — インフラコードの依存関係管理
- [暗号化基礎](../02-cryptography/) — 署名検証・ハッシュの理論
- [SQLとクエリの基礎](../../sql-and-query-mastery/00-basics/) — ORM/SQL のセキュリティ

---

## 参考文献

1. **OWASP Dependency-Check** — https://owasp.org/www-project-dependency-check/
2. **NIST SP 800-218 — Secure Software Development Framework (SSDF)** — https://csrc.nist.gov/publications/detail/sp/800-218/final
3. **CycloneDX Specification** — https://cyclonedx.org/specification/overview/
4. **GitHub Dependabot Documentation** — https://docs.github.com/en/code-security/dependabot
5. **NTIA SBOM Minimum Elements** — https://www.ntia.doc.gov/report/2021/minimum-elements-software-bill-materials-sbom
6. **Alex Birsan — Dependency Confusion** — https://medium.com/@alex.birsan/dependency-confusion-4a5d60fec610
