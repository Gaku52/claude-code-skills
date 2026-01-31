# Security & License Management - セキュリティとライセンス管理

## 📋 目次

1. [概要](#概要)
2. [セキュリティ管理](#セキュリティ管理)
3. [脆弱性スキャンツール](#脆弱性スキャンツール)
4. [セキュリティアップデート戦略](#セキュリティアップデート戦略)
5. [ライセンス管理](#ライセンス管理)
6. [ライセンススキャンツール](#ライセンススキャンツール)
7. [実践例](#実践例)

## 概要

依存関係のセキュリティとライセンス管理は、プロジェクトの安全性と法的コンプライアンスを確保するために不可欠です。

**このガイドで学べること:**
- 脆弱性の早期発見と対応
- セキュリティアップデート戦略
- ライセンス互換性の確認
- 自動化ツールの活用
- コンプライアンス対応

## セキュリティ管理

### セキュリティリスクの種類

**1. 既知の脆弱性（CVE）:**
```bash
# 例: CVE-2024-12345
# 影響: Remote Code Execution (RCE)
# 深刻度: Critical
# 対象: package@1.0.0 - 1.2.3
# 修正: package@1.2.4以降
```

**2. 供給チェーン攻撃:**
```bash
# 悪意のあるパッケージの混入
# - タイポスクワッティング（名前の類似）
# - アカウント乗っ取り
# - 依存関係の改ざん
```

**3. ライセンス違反:**
```bash
# 商用利用禁止ライセンスの使用
# GPL等のコピーレフトライセンス
```

### セキュリティチェックの基本

**npm audit:**
```bash
# 監査実行
npm audit

# 出力例:
# found 3 vulnerabilities (2 moderate, 1 high)
#
# Moderate  Inefficient Regular Expression Complexity
# Package   minimatch
# Patched in >=3.0.5
# Dependency of @babel/core
# Path @babel/core > minimatch

# 詳細レポート（JSON）
npm audit --json > audit-report.json

# 自動修正
npm audit fix

# 破壊的変更も含めて修正（慎重に）
npm audit fix --force

# 深刻度でフィルタ
npm audit --audit-level=high
```

**yarn audit:**
```bash
# 監査実行
yarn audit

# 詳細情報
yarn audit --verbose

# JSON出力
yarn audit --json
```

**pnpm audit:**
```bash
# 監査実行
pnpm audit

# 自動修正
pnpm audit --fix
```

### プラットフォーム別セキュリティチェック

**Swift Package Manager:**
```bash
# 依存関係の確認
swift package show-dependencies

# 古いパッケージの確認
# Xcodeの Package Dependencies で確認

# セキュリティアドバイザリ
# GitHub Security Advisories を有効化
```

**CocoaPods:**
```bash
# 古いバージョン確認
pod outdated --verbose

# 特定Podの詳細
pod spec cat Alamofire
```

**Python (pip):**
```bash
# pip-auditのインストール
pip install pip-audit

# 監査実行
pip-audit

# 自動修正
pip-audit --fix

# requirements.txtのスキャン
pip-audit -r requirements.txt
```

**Poetry:**
```bash
# 古いパッケージ確認
poetry show --outdated

# セキュリティチェック（plugin使用）
poetry self add poetry-plugin-audit
poetry audit
```

## 脆弱性スキャンツール

### GitHub Dependabot

**設定（.github/dependabot.yml）:**
```yaml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "daily"
    # セキュリティアップデートのみ
    open-pull-requests-limit: 10
    labels:
      - "security"
      - "dependencies"
    reviewers:
      - "security-team"
    assignees:
      - "tech-lead"

  # セキュリティアラートの設定
  # リポジトリ Settings → Security → Dependabot alerts を有効化
```

**アラート通知:**
```yaml
# .github/workflows/security-alerts.yml
name: Security Alerts

on:
  repository_vulnerability_alert:

jobs:
  notify:
    runs-on: ubuntu-latest
    steps:
      - name: Notify Slack
        uses: slackapi/slack-github-action@v1
        with:
          webhook-url: ${{ secrets.SLACK_WEBHOOK }}
          payload: |
            {
              "text": "🚨 Security Alert: ${{ github.event.alert.title }}"
            }
```

### Snyk

**インストール:**
```bash
# CLIインストール
npm install -g snyk

# 認証
snyk auth

# プロジェクトスキャン
snyk test

# 継続的監視
snyk monitor

# 自動修正
snyk fix

# Docker imageスキャン
snyk container test my-image:latest

# IaCスキャン
snyk iac test
```

**CI/CD統合:**
```yaml
# .github/workflows/security.yml
name: Security Scan

on:
  push:
    branches: [main, develop]
  pull_request:

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Snyk
        uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          args: --severity-threshold=high
```

**snyk.io設定:**
```json
{
  "language-settings": {
    "javascript": {
      "ignoreUnresolved": false
    }
  },
  "ignore": {
    "SNYK-JS-MINIMIST-559764": {
      "reason": "No fix available, low risk in our use case",
      "expires": "2025-12-31"
    }
  }
}
```

### OWASP Dependency-Check

```bash
# インストール
npm install -g dependency-check

# スキャン実行
dependency-check --project "My App" --scan ./

# レポート生成
dependency-check --project "My App" --scan ./ --format HTML

# CI統合
dependency-check --project "My App" --scan ./ --failOnCVSS 7
```

### npm-audit-resolver

```bash
# インストール
npm install -g npm-audit-resolver

# インタラクティブに脆弱性を解決
npm-audit-resolver

# 監査ファイル生成
npm-audit-resolver --audit-file audit.json

# CI/CDでチェック
npm-audit-resolver check
```

## セキュリティアップデート戦略

### 深刻度別対応フロー

```markdown
## セキュリティアップデート対応ポリシー

### Critical（緊急）
- **対応時間**: 即時（24時間以内）
- **プロセス**:
  1. アラート受信後、即座に影響範囲確認
  2. 緊急パッチ適用
  3. 最小限のテスト実施
  4. 緊急デプロイ
  5. 事後検証・監視強化

### High（高）
- **対応時間**: 48時間以内
- **プロセス**:
  1. 影響範囲詳細調査
  2. パッチ適用
  3. 通常テスト実施
  4. スプリント外でデプロイ
  5. 監視

### Medium（中）
- **対応時間**: 1週間以内
- **プロセス**:
  1. 次回スプリントで計画
  2. パッチ適用
  3. 完全テスト実施
  4. 通常デプロイ

### Low（低）
- **対応時間**: 次回定期メンテナンス時
- **プロセス**:
  1. バックログに追加
  2. 定期更新時に対応
```

### 自動セキュリティアップデート

**GitHub Actions例:**
```yaml
# .github/workflows/auto-security-update.yml
name: Auto Security Update

on:
  schedule:
    # 毎日午前9時（JST）
    - cron: '0 0 * * *'
  workflow_dispatch:

jobs:
  security-update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'

      - name: Install dependencies
        run: npm ci

      - name: Security audit
        id: audit
        run: |
          npm audit --audit-level=high --json > audit.json || true
          echo "vulnerabilities=$(cat audit.json | jq '.metadata.vulnerabilities.total')" >> $GITHUB_OUTPUT

      - name: Fix vulnerabilities
        if: steps.audit.outputs.vulnerabilities > 0
        run: npm audit fix

      - name: Run tests
        run: npm test

      - name: Create Pull Request
        if: steps.audit.outputs.vulnerabilities > 0
        uses: peter-evans/create-pull-request@v5
        with:
          commit-message: "chore(security): fix vulnerabilities"
          title: "🔒 Security: Fix vulnerabilities"
          body: |
            ## Security Update

            Automatically generated PR to fix security vulnerabilities.

            **Vulnerabilities fixed**: ${{ steps.audit.outputs.vulnerabilities }}

            Please review and merge.
          branch: security/auto-update
          labels: security, automated
```

### 脆弱性の評価とトリアージ

```typescript
// security-triage.ts
interface Vulnerability {
  id: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  package: string;
  version: string;
  title: string;
  description: string;
  cvss: number;
}

function triageVulnerability(vuln: Vulnerability): 'immediate' | 'scheduled' | 'deferred' {
  // 1. 深刻度チェック
  if (vuln.severity === 'critical' || vuln.cvss >= 9.0) {
    return 'immediate';
  }

  // 2. パッケージの重要性チェック
  const criticalPackages = ['express', 'react', 'next'];
  if (criticalPackages.includes(vuln.package) && vuln.severity === 'high') {
    return 'immediate';
  }

  // 3. 本番環境での使用チェック
  if (isProductionDependency(vuln.package)) {
    if (vuln.severity === 'high') return 'scheduled';
    return 'deferred';
  }

  // 4. 開発依存関係
  return 'deferred';
}
```

## ライセンス管理

### ライセンスの種類と互換性

**主要ライセンスの分類:**

```markdown
## ライセンス互換性マトリクス

### ✅ 商用利用可能（推奨）

1. **MIT License**
   - 特徴: 最も寛容、再配布・改変自由
   - 義務: 著作権表示・ライセンス文の保持
   - 例: React, Vue.js, Node.js

2. **Apache License 2.0**
   - 特徴: 特許権の明示的付与
   - 義務: 著作権表示、変更箇所の明示
   - 例: Kubernetes, Android, TensorFlow

3. **BSD License (2-Clause, 3-Clause)**
   - 特徴: MITに類似、シンプル
   - 義務: 著作権表示
   - 例: FreeBSD, Django

4. **ISC License**
   - 特徴: MITとほぼ同じ、より簡潔
   - 義務: 著作権表示
   - 例: npm, Node.js core modules

### ⚠️ 要検討（条件付き使用可）

1. **LGPL (Lesser GPL)**
   - 特徴: 動的リンクは許可
   - 義務: ソースコード開示（改変時）
   - 注意: 静的リンクは慎重に

2. **Mozilla Public License 2.0 (MPL)**
   - 特徴: ファイル単位のコピーレフト
   - 義務: MPLファイルの改変はソース開示
   - 例: Firefox

3. **Eclipse Public License (EPL)**
   - 特徴: 弱いコピーレフト
   - 義務: EPL部分の改変はソース開示
   - 例: Eclipse IDE

### ❌ 商用利用禁止・要注意

1. **GPL (GNU General Public License)**
   - 特徴: 強いコピーレフト
   - 義務: 全体のソースコード開示必須
   - リスク: プロプライエタリソフトウェアには不適

2. **AGPL (Affero GPL)**
   - 特徴: GPLより厳格（ネットワーク経由も開示義務）
   - 義務: SaaS提供時もソース開示
   - リスク: Webサービスでは危険

3. **CC BY-NC (Creative Commons Non-Commercial)**
   - 特徴: 商用利用禁止
   - リスク: ビジネス利用不可

4. **Proprietary / Custom License**
   - 特徴: 独自ライセンス
   - リスク: 個別確認必須
```

### ライセンス互換性チェック

```bash
# プロジェクトのライセンス = MIT の場合

✅ 使用可能:
- MIT
- Apache 2.0
- BSD
- ISC

⚠️ 要検討:
- LGPL（動的リンクのみ）
- MPL 2.0

❌ 使用不可:
- GPL
- AGPL
- 商用利用禁止ライセンス
```

## ライセンススキャンツール

### license-checker (npm)

```bash
# インストール
npm install -g license-checker

# 基本スキャン
license-checker

# JSON出力
license-checker --json > licenses.json

# CSV出力
license-checker --csv > licenses.csv

# 許可されたライセンスのみ
license-checker --onlyAllow="MIT;Apache-2.0;BSD-2-Clause;BSD-3-Clause;ISC"

# 禁止ライセンスチェック
license-checker --failOn="GPL;AGPL;LGPL"

# サマリー表示
license-checker --summary
```

**CI統合:**
```yaml
# .github/workflows/license-check.yml
name: License Check

on:
  push:
    branches: [main]
  pull_request:

jobs:
  check-licenses:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4

      - name: Install dependencies
        run: npm ci

      - name: Check licenses
        run: |
          npx license-checker --failOn "GPL;AGPL;LGPL;CC-BY-NC" \
            --onlyAllow "MIT;Apache-2.0;BSD-2-Clause;BSD-3-Clause;ISC;0BSD;Unlicense"
```

### npm-license-crawler

```bash
# インストール
npm install -g npm-license-crawler

# スキャン実行
npm-license-crawler --onlyDirectDependencies

# Markdown出力
npm-license-crawler --markdown licenses.md

# 除外設定
npm-license-crawler --exclude "test,docs"
```

### licensee (Ruby)

```bash
# インストール
gem install licensee

# スキャン
licensee detect

# 詳細情報
licensee detect --json
```

### CocoaPods License Acknowledgements

```ruby
# Podfile
plugin 'cocoapods-acknowledgements'

post_install do |installer|
  require 'cocoapods-acknowledgements'
  Pod::Acknowledgements.generate(installer)
end
```

```bash
# インストール後、Settings.bundleにライセンス情報が生成される
# Settings → Acknowledgements で表示可能
```

### Swift Package Manager

```bash
# 依存関係とライセンス確認（手動）
swift package show-dependencies

# 各パッケージのLICENSEファイルを確認
# GitHubリポジトリで確認
```

**自動化スクリプト例:**
```swift
// Scripts/generate-licenses.swift
import Foundation

struct Package {
    let name: String
    let url: String
    let license: String
}

// Package.resolvedから読み込み
let packages = parsePackageResolved()

// ライセンス情報を収集
for package in packages {
    let license = fetchLicense(from: package.url)
    print("\(package.name): \(license)")
}
```

## 実践例

### セキュリティスキャンの定期実行

**毎日の自動スキャン:**
```bash
#!/bin/bash
# scripts/daily-security-scan.sh

echo "🔍 Starting daily security scan..."

# npm audit
echo "Running npm audit..."
npm audit --audit-level=moderate

# Snyk
echo "Running Snyk..."
snyk test --severity-threshold=high

# License check
echo "Checking licenses..."
npx license-checker --failOn "GPL;AGPL"

echo "✅ Security scan completed"
```

**cron設定:**
```bash
# 毎日午前9時に実行
0 9 * * * cd /path/to/project && ./scripts/daily-security-scan.sh
```

### ライセンス表示ページの生成

**自動生成スクリプト:**
```javascript
// scripts/generate-license-page.js
const checker = require('license-checker');
const fs = require('fs');

checker.init({
  start: '.',
  production: true,
}, (err, packages) => {
  if (err) {
    console.error(err);
    process.exit(1);
  }

  let html = `
<!DOCTYPE html>
<html>
<head>
  <title>Open Source Licenses</title>
  <style>
    body { font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
    .package { border: 1px solid #ddd; margin: 10px 0; padding: 15px; }
    .package-name { font-weight: bold; font-size: 18px; }
    .license { color: #666; }
  </style>
</head>
<body>
  <h1>Open Source Licenses</h1>
`;

  Object.keys(packages).sort().forEach(name => {
    const pkg = packages[name];
    html += `
    <div class="package">
      <div class="package-name">${name}</div>
      <div class="license">License: ${pkg.licenses}</div>
      <div>Repository: <a href="${pkg.repository}">${pkg.repository}</a></div>
    </div>
`;
  });

  html += `
</body>
</html>
`;

  fs.writeFileSync('public/licenses.html', html);
  console.log('✅ License page generated: public/licenses.html');
});
```

**package.json:**
```json
{
  "scripts": {
    "generate-licenses": "node scripts/generate-license-page.js"
  }
}
```

### iOS アプリでのライセンス表示

**Settings.bundle方式:**
```bash
# CocoaPods plugin使用
# Podfile
plugin 'cocoapods-acknowledgements'

post_install do |installer|
  require 'cocoapods-acknowledgements'
  Pod::Acknowledgements.generate(installer)
end
```

**専用画面方式:**
```swift
// LicensesViewController.swift
import UIKit

struct License {
    let name: String
    let license: String
    let url: String
}

class LicensesViewController: UITableViewController {
    let licenses: [License] = [
        License(
            name: "Alamofire",
            license: "MIT",
            url: "https://github.com/Alamofire/Alamofire/blob/master/LICENSE"
        ),
        License(
            name: "Realm",
            license: "Apache 2.0",
            url: "https://github.com/realm/realm-swift/blob/master/LICENSE"
        ),
    ]

    override func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return licenses.count
    }

    override func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "LicenseCell", for: indexPath)
        let license = licenses[indexPath.row]
        cell.textLabel?.text = license.name
        cell.detailTextLabel?.text = license.license
        return cell
    }

    override func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        let license = licenses[indexPath.row]
        if let url = URL(string: license.url) {
            UIApplication.shared.open(url)
        }
    }
}
```

### セキュリティインシデント対応フロー

```markdown
## セキュリティインシデント対応プロセス

### Phase 1: 検知（Detection）
1. アラート受信（Dependabot, Snyk, GitHub Advisory）
2. 深刻度・影響範囲の初期評価
3. インシデントチケット作成

### Phase 2: 評価（Assessment）
1. 詳細な影響範囲調査
   - 使用箇所の特定
   - エクスプロイト可能性の確認
   - ビジネスインパクト評価
2. 対応優先度の決定
3. 関係者への通知

### Phase 3: 対応（Response）
1. パッチ適用
   ```bash
   npm audit fix
   # または
   npm install package@fixed-version
   ```
2. テスト実施
   - ユニットテスト
   - 統合テスト
   - E2Eテスト
3. デプロイ準備

### Phase 4: デプロイ（Deploy）
1. ステージング環境でデプロイ
2. 動作確認
3. 本番環境デプロイ
4. 監視強化

### Phase 5: 事後対応（Post-Incident）
1. インシデントレポート作成
2. 再発防止策の検討
3. ドキュメント更新
4. チーム共有
```

## まとめ

### セキュリティ管理のベストプラクティス

1. **継続的スキャン**
   - 毎日の自動スキャン
   - CI/CDでの監査
   - Dependabot/Snyk活用

2. **迅速な対応**
   - Critical: 24時間以内
   - High: 48時間以内
   - Medium: 1週間以内

3. **自動化**
   - セキュリティアップデート
   - ライセンスチェック
   - レポート生成

### ライセンス管理のベストプラクティス

1. **許可ライセンスの明確化**
   - MIT, Apache 2.0, BSD, ISC: ✅
   - GPL, AGPL: ❌

2. **定期チェック**
   - CI/CDでの自動チェック
   - 新規依存追加時の確認

3. **ライセンス表示**
   - アプリ内表示
   - ドキュメント化

### チェックリスト

- [ ] セキュリティスキャンツールを導入している
- [ ] Dependabot/Renovateを設定している
- [ ] セキュリティアップデートポリシーを策定している
- [ ] 許可ライセンスリストを作成している
- [ ] CI/CDでライセンスチェックを実施している
- [ ] ライセンス表示を実装している
- [ ] インシデント対応フローを策定している
- [ ] 定期的なレビューを実施している

---

**関連ガイド:**
- [Package Manager Best Practices](./01-package-managers.md)
- [Version Management](./02-version-management.md)
