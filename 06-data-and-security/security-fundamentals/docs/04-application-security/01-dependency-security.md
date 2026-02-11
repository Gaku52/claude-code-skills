# 依存関係セキュリティ

> SCA (Software Composition Analysis) による脆弱性検出、Dependabot による自動更新、SBOM による可視化まで、サードパーティ依存関係のリスクを管理するガイド

## この章で学ぶこと

1. **依存関係のリスク** — サプライチェーン攻撃の手法と依存関係に潜む脆弱性の脅威
2. **SCA ツールの活用** — Dependabot、Snyk、Trivy による脆弱性の自動検出と修正
3. **SBOM (Software Bill of Materials)** — ソフトウェア部品表によるサプライチェーンの可視化

---

## 1. 依存関係のリスク

### サプライチェーン攻撃の手法

```
+----------------------------------------------------------+
|            サプライチェーン攻撃のベクター                    |
|----------------------------------------------------------|
|                                                          |
|  [直接攻撃]                                               |
|  +-- 依存パッケージの乗っ取り (アカウント侵害)              |
|  +-- タイポスクワッティング (lodash → lodahs)              |
|  +-- 依存関係の混乱攻撃 (内部パッケージ名偽装)              |
|                                                          |
|  [間接攻撃]                                               |
|  +-- 推移的依存関係の脆弱性 (A→B→C、Cに脆弱性)            |
|  +-- ビルドシステムの侵害 (CI/CD パイプライン)              |
|  +-- 悪意あるプレ/ポストインストールスクリプト               |
|                                                          |
|  [既知の事例]                                              |
|  +-- event-stream (2018): 暗号通貨窃取コード注入           |
|  +-- ua-parser-js (2021): マイナー & パスワード窃取         |
|  +-- Log4Shell (2021): Log4j のリモートコード実行          |
|  +-- colors/faker (2022): メンテナによる意図的破壊          |
+----------------------------------------------------------+
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
  node_modules の管理対象が爆発的に増加
```

---

## 2. SCA (Software Composition Analysis)

### SCA ツールの比較

| ツール | 対応言語 | 無料枠 | 特徴 |
|--------|---------|--------|------|
| Dependabot | 多言語 | GitHub 無料 | GitHub ネイティブ統合 |
| Snyk | 多言語 | OSS 無料 | 修正 PR 自動生成 |
| Trivy | 多言語 + コンテナ | 完全無料 | コンテナイメージも対応 |
| OWASP Dep-Check | Java, .NET 中心 | 完全無料 | NVD データベース連携 |
| npm audit | JavaScript | 無料 | npm 標準機能 |
| pip-audit | Python | 無料 | OSV データベース連携 |

### Dependabot の設定

```yaml
# .github/dependabot.yml
version: 2
updates:
  # npm 依存関係
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
    # セキュリティアップデートは即座に
    # バージョンアップデートはグループ化
    groups:
      production-dependencies:
        dependency-type: "production"
        update-types:
          - "minor"
          - "patch"
      dev-dependencies:
        dependency-type: "development"
    ignore:
      # メジャーバージョンアップは手動で
      - dependency-name: "*"
        update-types: ["version-update:semver-major"]

  # Docker イメージ
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

### GitHub Security Advisories の自動通知

```yaml
# .github/workflows/security-scan.yml
name: Security Scan
on:
  push:
    branches: [main]
  pull_request:
  schedule:
    - cron: '0 0 * * *'  # 毎日 0:00

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: npm audit
        run: npm audit --audit-level=high

      - name: Trivy vulnerability scan
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          severity: 'HIGH,CRITICAL'
          exit-code: '1'  # 脆弱性発見時にビルド失敗
```

### Trivy によるスキャン

```bash
# ファイルシステムスキャン (プロジェクト全体)
trivy fs --severity HIGH,CRITICAL .

# コンテナイメージスキャン
trivy image --severity HIGH,CRITICAL myapp:latest

# SBOM 出力付きスキャン
trivy fs --format cyclonedx --output sbom.json .

# 出力例:
# library     | vulnerability | severity | installed | fixed
# ------------|-------------- |----------|-----------|------
# lodash      | CVE-2021-23337| HIGH     | 4.17.20   | 4.17.21
# express     | CVE-2024-xxxx | CRITICAL | 4.17.1    | 4.18.2
```

---

## 3. SBOM (Software Bill of Materials)

### SBOM の概念

```
+----------------------------------------------------------+
|                    SBOM (部品表)                           |
|----------------------------------------------------------|
|  アプリケーション: MyApp v2.1.0                            |
|                                                          |
|  コンポーネント一覧:                                      |
|  +-- express@4.18.2       (MIT)      直接依存             |
|  +-- lodash@4.17.21      (MIT)      直接依存             |
|  +-- body-parser@1.20.2  (MIT)      推移的依存           |
|  +-- debug@4.3.4         (MIT)      推移的依存           |
|  +-- ...                                                |
|                                                          |
|  各コンポーネントに含まれる情報:                            |
|  - パッケージ名・バージョン                                |
|  - ライセンス                                            |
|  - 供給元 (サプライヤー)                                  |
|  - 既知の脆弱性 (CVE)                                    |
|  - 依存関係の親子関係                                     |
+----------------------------------------------------------+
```

### SBOM フォーマットの比較

| 項目 | SPDX | CycloneDX |
|------|------|-----------|
| 策定 | Linux Foundation | OWASP |
| ISO 標準 | ISO/IEC 5962:2021 | ECMA-424 |
| フォーマット | JSON, RDF, Tag-Value | JSON, XML, Protobuf |
| 用途 | ライセンスコンプライアンス重視 | セキュリティ重視 |
| 脆弱性情報 | 外部参照 | VEX 統合 |
| ツール | syft, trivy | cdxgen, trivy |

### SBOM の生成と活用

```bash
# syft で CycloneDX 形式の SBOM を生成
syft dir:. -o cyclonedx-json > sbom.cyclonedx.json

# syft で SPDX 形式の SBOM を生成
syft dir:. -o spdx-json > sbom.spdx.json

# grype で SBOM から脆弱性をスキャン
grype sbom:sbom.cyclonedx.json

# npm で SBOM を生成 (npm 9+)
npm sbom --sbom-format cyclonedx
```

```python
# SBOM を CI/CD で管理するスクリプト
import json
import subprocess
from datetime import datetime

def generate_and_store_sbom(project_dir: str, output_dir: str):
    """SBOM を生成してアーティファクトとして保存"""

    # CycloneDX SBOM を生成
    result = subprocess.run(
        ["syft", f"dir:{project_dir}", "-o", "cyclonedx-json"],
        capture_output=True, text=True,
    )
    sbom = json.loads(result.stdout)

    # メタデータを追加
    sbom["metadata"] = {
        "timestamp": datetime.utcnow().isoformat(),
        "tools": [{"name": "syft", "version": "0.100.0"}],
    }

    # 保存
    output_path = f"{output_dir}/sbom-{datetime.now().strftime('%Y%m%d')}.json"
    with open(output_path, 'w') as f:
        json.dump(sbom, f, indent=2)

    # 脆弱性スキャン
    vuln_result = subprocess.run(
        ["grype", f"sbom:{output_path}", "-o", "json"],
        capture_output=True, text=True,
    )
    vulnerabilities = json.loads(vuln_result.stdout)

    return {
        "sbom_path": output_path,
        "component_count": len(sbom.get("components", [])),
        "vulnerability_count": len(vulnerabilities.get("matches", [])),
    }
```

---

## 4. 依存関係のロックと固定

### ロックファイルの重要性

```
+----------------------------------------------------------+
|  ロックファイルなし:                                       |
|  package.json: "lodash": "^4.17.0"                       |
|  → 開発環境: 4.17.20                                     |
|  → CI: 4.17.21 (最新)                                    |
|  → 本番: 4.17.19 (キャッシュ)                             |
|  (環境によってバージョンが異なる = 再現性がない)             |
|                                                          |
|  ロックファイルあり:                                       |
|  package-lock.json: "lodash": "4.17.21"                  |
|  → 全環境で 4.17.21 を使用 (再現性が保証される)            |
+----------------------------------------------------------+
```

```bash
# package-lock.json を CI で厳密に使用
npm ci  # npm install ではなく ci を使用

# pip で requirements.txt を固定
pip freeze > requirements.txt
pip install -r requirements.txt --require-hashes

# Go モジュールのチェックサム検証
# go.sum が自動生成される
go mod verify
```

---

## 5. ライセンスコンプライアンス

```bash
# license-checker でライセンスを一覧表示
npx license-checker --summary

# 禁止ライセンスのチェック
npx license-checker --failOn "GPL-3.0;AGPL-3.0"

# Python のライセンスチェック
pip install pip-licenses
pip-licenses --fail-on="GNU General Public License v3 (GPLv3)"
```

---

## 6. アンチパターン

### アンチパターン 1: ロックファイルを Git 管理しない

```bash
# NG: .gitignore にロックファイルを追加
echo "package-lock.json" >> .gitignore
echo "yarn.lock" >> .gitignore

# OK: ロックファイルを必ずコミット
git add package-lock.json
git commit -m "Update lock file"

# CI では npm ci を使用 (lock ファイルに基づく厳密インストール)
```

**影響**: ビルドごとに異なるバージョンが使用され、既知の脆弱性が混入するリスクがある。

### アンチパターン 2: 脆弱性アラートの放置

```
NG: Dependabot アラートを無視し続ける
  → 93件の Critical/High 脆弱性が放置
  → 攻撃者が既知の脆弱性を悪用

OK: SLA を設けて対応
  Critical: 24時間以内に対応
  High:     1週間以内に対応
  Medium:   1ヶ月以内に対応
  Low:      次のスプリントで対応
```

---

## 7. FAQ

### Q1. 推移的依存関係の脆弱性はどう対処するか?

直接依存のバージョンを上げることで推移的依存も更新されるケースが多い。それが不可能な場合は npm の `overrides`、yarn の `resolutions`、pip の `constraints` で特定バージョンを強制できる。ただし互換性の問題が起きうるためテストを十分に行うこと。

### Q2. SBOM の提供は義務か?

米国の大統領令 14028 (2021) により、連邦政府向けソフトウェアでは SBOM の提供が求められている。EU のサイバーレジリエンス法 (CRA) でも SBOM が要件化される見込みである。民間でも取引先からの要求が増えている。

### Q3. 内部パッケージのスコープ保護はどうすればよいか?

npm では `@myorg/` スコープを組織で予約登録する。依存関係混乱攻撃を防ぐため、内部レジストリに該当名のパッケージを公開レジストリにもプレースホルダとして登録する方法もある。.npmrc でレジストリのスコープ設定を正しく行うことが重要である。

---

## まとめ

| 項目 | 要点 |
|------|------|
| サプライチェーンリスク | 推移的依存・タイポスクワッティング・乗っ取りに注意 |
| SCA ツール | Dependabot + Trivy で脆弱性を自動検出 |
| SBOM | CycloneDX/SPDX で部品表を生成し脆弱性を追跡 |
| ロックファイル | 必ず Git 管理し CI では厳密インストール |
| 脆弱性対応 SLA | Critical 24h、High 1週間の対応基準を設定 |
| ライセンス | GPL/AGPL 等の制約を自動チェック |

---

## 次に読むべきガイド

- [コンテナセキュリティ](./02-container-security.md) — コンテナイメージの依存関係スキャン
- [SAST/DAST](./03-sast-dast.md) — コード自体の脆弱性検査
- [セキュアコーディング](./00-secure-coding.md) — コードレベルの脆弱性防止

---

## 参考文献

1. **OWASP Dependency-Check** — https://owasp.org/www-project-dependency-check/
2. **NIST SP 800-218 — Secure Software Development Framework (SSDF)** — https://csrc.nist.gov/publications/detail/sp/800-218/final
3. **CycloneDX Specification** — https://cyclonedx.org/specification/overview/
4. **GitHub Dependabot Documentation** — https://docs.github.com/en/code-security/dependabot
