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

```python
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
```

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
| DB遅延 | EXPLAIN, slow query log | インデックス、クエリ最適化 |

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

```
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
```

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

```python
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
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```

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

```python
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
```

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
| コード最適化 | 低-中 | 高 | CPU律速の場合 |

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

```
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
```

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

```python
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
        dangerous_chars = ['<', '>', '"', "'", '&', '\\']
        result = value
        for char in dangerous_chars:
            result = result.replace(char, '')
        return result.strip()

# 使用例
token = SecurityUtils.generate_token()
hashed, salt = SecurityUtils.hash_password("my_password")
is_valid = SecurityUtils.verify_password("my_password", hashed, salt)
```

### セキュリティチェックリスト

- [ ] 全ての入力値がバリデーションされている
- [ ] 機密情報がログに出力されていない
- [ ] HTTPS が強制されている
- [ ] CORS ポリシーが適切に設定されている
- [ ] 依存パッケージの脆弱性スキャンが実施されている
- [ ] エラーメッセージに内部情報が含まれていない

---

## マイグレーションガイド

### バージョンアップ時の注意点

| バージョン | 主な変更点 | 移行作業 | 影響範囲 |
|-----------|-----------|---------|---------|
| v1.x → v2.x | API設計の刷新 | エンドポイント変更 | 全クライアント |
| v2.x → v3.x | 認証方式の変更 | トークン形式更新 | 認証関連 |
| v3.x → v4.x | データモデル変更 | マイグレーションスクリプト実行 | DB関連 |

### 段階的移行の手順

```python
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
```

### ロールバック計画

移行作業には必ずロールバック計画を準備してください:

1. **データのバックアップ**: 移行前に完全バックアップを取得
2. **テスト環境での検証**: 本番と同等の環境で事前検証
3. **段階的なロールアウト**: カナリアリリースで段階的に展開
4. **監視の強化**: 移行中はメトリクスの監視間隔を短縮
5. **判断基準の明確化**: ロールバックを判断する基準を事前に定義
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


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

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
- SQLとクエリの基礎 — ORM/SQL のセキュリティ

---

## 参考文献

1. **OWASP Dependency-Check** — https://owasp.org/www-project-dependency-check/
2. **NIST SP 800-218 — Secure Software Development Framework (SSDF)** — https://csrc.nist.gov/publications/detail/sp/800-218/final
3. **CycloneDX Specification** — https://cyclonedx.org/specification/overview/
4. **GitHub Dependabot Documentation** — https://docs.github.com/en/code-security/dependabot
5. **NTIA SBOM Minimum Elements** — https://www.ntia.doc.gov/report/2021/minimum-elements-software-bill-materials-sbom
6. **Alex Birsan — Dependency Confusion** — https://medium.com/@alex.birsan/dependency-confusion-4a5d60fec610
