# 📝 Incident Recording - 問題記録の実践ガイド

> **目的**: 開発中に発生した問題・エラー・失敗を即座に記録し、体系的に管理する手法を習得する

## 📚 目次

1. [インシデント記録の重要性](#インシデント記録の重要性)
2. [記録すべきインシデントの種類](#記録すべきインシデントの種類)
3. [効果的な記録フォーマット](#効果的な記録フォーマット)
4. [記録タイミングとワークフロー](#記録タイミングとワークフロー)
5. [ツールと自動化](#ツールと自動化)
6. [実践例](#実践例)
7. [チーム運用](#チーム運用)

---

## インシデント記録の重要性

### なぜ記録が必要か

**記録の価値**:
- **再発防止**: 同じ問題を繰り返さない
- **知識の蓄積**: チーム全体の学習資産
- **時間短縮**: 過去の解決策を即座に参照
- **品質向上**: パターン分析による改善
- **透明性**: 問題の可視化と共有

**記録しない場合のコスト**:
```
❌ 記録なし
├─ 同じ問題に何度も遭遇
├─ 解決策を毎回ゼロから考える
├─ チームメンバーが独立に同じ問題と格闘
├─ 根本原因が放置される
└─ 新メンバーが同じ罠にはまる

✅ 記録あり
├─ 問題発生時に即座に参照
├─ 解決済み問題は5分で解決
├─ チーム全体で知識共有
├─ パターン認識による予防
└─ オンボーディング資料として活用
```

### 記録による効果

**定量的効果**:
- デバッグ時間: **平均50%削減**
- 再発率: **70%減少**
- 新メンバーのオンボーディング: **30%短縮**

**定性的効果**:
- チームの心理的安全性向上
- 失敗からの学習文化
- 継続的改善のサイクル

---

## 記録すべきインシデントの種類

### 1. Critical（緊急・致命的）

**定義**: システム停止、データ損失、セキュリティ侵害など

**記録例**:
```markdown
# [CRITICAL] 本番環境でデータベース接続が全断

## 発生日時
2025-12-31 14:23:00 JST

## 影響範囲
- 全ユーザーがサービス利用不可（約45分間）
- 推定影響ユーザー数: 約12,000名

## 発生状況
- デプロイ後5分でDB接続エラーが大量発生
- ヘルスチェックが全てFAIL
- アラートが10件/分で発火

## 初動対応
1. 14:25 - インシデント検知（監視アラート）
2. 14:27 - 緊急対策チーム招集
3. 14:30 - 直前のデプロイをロールバック
4. 14:35 - サービス復旧確認
5. 15:08 - 完全復旧宣言

## 根本原因
- 新しいDB接続プール設定が原因
- `maxConnections: 10` → 本番負荷に対して不足
- ステージング環境では低負荷のため検知できず

## 恒久対策
- [x] DB接続設定を `maxConnections: 100` に変更
- [x] ステージング環境に負荷テスト追加
- [x] デプロイ前チェックリストに接続プール設定確認を追加
- [ ] 自動負荷テストCI/CD統合（1週間以内）

## 学んだ教訓
1. ステージング環境は本番と同等の負荷テストが必須
2. DB関連設定変更は特に慎重に
3. ロールバック手順の事前確認が重要

## 担当者
- 検知: @monitoring-team
- 初動対応: @sre-team
- 根本対策: @backend-team
```

### 2. High（重大）

**定義**: 主要機能の障害、重大なバグ、パフォーマンス問題

**記録例**:
```markdown
# [HIGH] iOS アプリでログイン後に画面が真っ白

## 発生日時
2025-12-30 10:15:00 JST

## 影響範囲
- iOS 15.0-15.2 のユーザーのみ（全体の約8%）
- Android、iOS 16+ は正常

## 発生状況
- v2.5.0 リリース直後から報告開始
- App Store レビューで低評価が増加
- カスタマーサポートへの問い合わせ増加

## 再現手順
1. iOS 15.1 デバイスでアプリを起動
2. ログイン画面でメールアドレス・パスワード入力
3. ログインボタンをタップ
4. → 画面が真っ白になり、操作不能

## 調査結果
- iOS 15では `NavigationStack` が未対応
- `NavigationView` との互換性問題
- Xcode の Preview では正常動作（iOS 16+）

## 解決策
```swift
// 修正前（iOS 15 で動作しない）
struct ContentView: View {
    var body: some View {
        NavigationStack {
            MainTabView()
        }
    }
}

// 修正後（iOS 15+ 対応）
struct ContentView: View {
    var body: some View {
        if #available(iOS 16.0, *) {
            NavigationStack {
                MainTabView()
            }
        } else {
            NavigationView {
                MainTabView()
            }
            .navigationViewStyle(.stack)
        }
    }
}
```

## 対応履歴
- 12/30 10:15 - 問題報告受領
- 12/30 11:00 - 再現確認
- 12/30 13:30 - 修正コード実装
- 12/30 15:00 - テスト完了
- 12/30 16:00 - v2.5.1 緊急リリース申請
- 12/31 09:00 - App Store 承認、配信開始

## 予防策
- [x] CI/CD に iOS 15 実機テスト追加
- [x] 最小サポートバージョンでの動作確認を必須化
- [x] チェックリストに「下位OS対応確認」を追加

## 担当者
- 報告: @customer-support
- 調査・修正: @ios-team
- テスト: @qa-team
```

### 3. Medium（中程度）

**定義**: 限定的な機能障害、軽微なバグ、使いにくさ

**記録例**:
```markdown
# [MEDIUM] ダークモードで一部テキストが読めない

## 発生日時
2025-12-28

## 影響範囲
- ダークモード利用ユーザーのみ
- 設定画面の一部テキスト

## 発生状況
ユーザーから「設定画面の説明文が真っ黒で読めない」との報告

## 再現手順
1. アプリ設定でダークモード有効化
2. 設定画面を開く
3. → 説明文（グレーテキスト）が背景と同化して読めない

## 原因
```swift
// 問題のコード
Text("この設定を有効にすると...")
    .foregroundColor(.gray)  // ライトモード用の固定色
```

ダークモードでも `.gray` が暗い色のままで背景と区別がつかない。

## 解決策
```swift
// 修正後
Text("この設定を有効にすると...")
    .foregroundColor(.secondary)  // システムカラー使用
```

## 予防策
- 固定色ではなくセマンティックカラーを使用
- デザインレビューでダークモード確認を必須化

## 担当者
- 報告: @user-feedback
- 修正: @ios-team
```

### 4. Low（軽微）

**定義**: 小さな不具合、改善要望、ドキュメント不備

**記録例**:
```markdown
# [LOW] README の環境構築手順が古い

## 発生日時
2025-12-29

## 問題
- README に記載の Node.js バージョンが v14
- 実際のプロジェクトは v20 必須
- 新メンバーが環境構築時に混乱

## 対応
```markdown
<!-- 修正前 -->
- Node.js v14 以上

<!-- 修正後 -->
- Node.js v20 以上
- npm v10 以上
```

## 予防策
- README にバージョン自動チェックスクリプト追加
- CI/CD で README の整合性チェック

## 担当者
- 報告: @new-member
- 修正: @devops-team
```

### 5. Knowledge（ナレッジ）

**定義**: エラーではないが、知っておくべき情報

**記録例**:
```markdown
# [KNOWLEDGE] Swift Package Manager のキャッシュクリア方法

## 状況
SPM でパッケージ更新後も古いバージョンがビルドされる問題が頻発

## 解決策
```bash
# Xcode の Derived Data を削除
rm -rf ~/Library/Developer/Xcode/DerivedData

# SPM キャッシュをクリア
rm -rf ~/Library/Caches/org.swift.swiftpm

# Xcode を再起動
```

または Xcode メニューから:
```
Product > Clean Build Folder (Cmd+Shift+K)
File > Packages > Reset Package Caches
```

## 予防
- パッケージバージョン変更時は必ずキャッシュクリア
- CI/CD では常にクリーンビルド

## 参考
- [Apple Developer Forums](https://developer.apple.com/forums/)
- チーム Wiki: SPM トラブルシューティング

## 担当者
- 記録: @ios-team
```

---

## 効果的な記録フォーマット

### 基本構成要素

**必須項目**:
```markdown
# [優先度] タイトル（1行で問題を要約）

## 発生日時
YYYY-MM-DD HH:MM:SS

## 影響範囲
誰が、何が、どの程度影響を受けたか

## 発生状況
何が起きたか（事実のみ）

## 原因
なぜ起きたか

## 解決策
どう解決したか（コード例含む）

## 予防策
今後どう防ぐか

## 担当者
誰が対応したか
```

### 優れた記録の原則

**5W1H を明確に**:
- **When**: いつ発生したか
- **Where**: どこで発生したか（環境、画面、機能）
- **Who**: 誰が影響を受けたか
- **What**: 何が起きたか
- **Why**: なぜ起きたか
- **How**: どう解決したか

**STAR フォーマット**:
- **Situation**: 状況（発生時の背景）
- **Task**: 課題（解決すべき問題）
- **Action**: 行動（実施した対応）
- **Result**: 結果（対応の成果）

### 具体的な記録例

**❌ 悪い例**:
```markdown
# ログインできない

エラーが出た。直した。
```

問題点:
- いつ発生したか不明
- どんなエラーか不明
- どう直したか不明
- 再発防止策なし

**✅ 良い例**:
```markdown
# [HIGH] ログイン時に "Invalid token" エラーで認証失敗

## 発生日時
2025-12-31 09:00:00 JST

## 影響範囲
- 全ユーザー（推定影響: 100%）
- Web/iOS/Android 全プラットフォーム

## 発生状況
本番環境デプロイ（v3.2.0）直後から、すべてのログイン試行が失敗。
エラーメッセージ: "Invalid token"

## 再現手順
1. ログイン画面でメールアドレス・パスワード入力
2. ログインボタンをタップ
3. → "Invalid token" エラー表示

## 原因
JWT トークンの署名アルゴリズムを HS256 から RS256 に変更したが、
バックエンドの環境変数 `JWT_PUBLIC_KEY` が設定されていなかった。

```javascript
// 問題のコード (backend/auth/jwt.js:45)
const publicKey = process.env.JWT_PUBLIC_KEY; // undefined
jwt.verify(token, publicKey, { algorithms: ['RS256'] }); // 常に失敗
```

## 解決策
1. 環境変数 `JWT_PUBLIC_KEY` を設定
```bash
export JWT_PUBLIC_KEY="$(cat public_key.pem)"
```

2. 環境変数チェック処理を追加
```javascript
// 修正後
const publicKey = process.env.JWT_PUBLIC_KEY;
if (!publicKey) {
    throw new Error('JWT_PUBLIC_KEY is not set');
}
jwt.verify(token, publicKey, { algorithms: ['RS256'] });
```

## 対応履歴
- 09:00 - 問題検知（監視アラート）
- 09:05 - 原因特定（ログ分析）
- 09:10 - 環境変数設定
- 09:12 - サービス復旧確認

## 予防策
- [x] 起動時に必須環境変数をチェックするスクリプト追加
- [x] デプロイチェックリストに環境変数確認を追加
- [x] ステージング環境で本番と同じ設定を使用
- [ ] 環境変数管理ツール（dotenv-vault）導入検討

## 参考情報
- Commit: `abc1234`
- PR: #456
- 関連ドキュメント: docs/auth/jwt-migration.md

## 担当者
- 検知: @monitoring
- 対応: @backend-team
- レビュー: @sre-team
```

---

## 記録タイミングとワークフロー

### 記録すべきタイミング

**1. 問題発生時（即座に記録）**

```
問題発生
   ↓
[1分以内] 速報記録（タイトル・影響範囲のみ）
   ↓
調査・対応
   ↓
[対応完了後] 詳細記録（原因・解決策・予防策）
```

**速報記録の例**:
```markdown
# [CRITICAL] 本番環境 API が 500 エラー

## 発生日時
2025-12-31 14:23:00

## 影響範囲
全ユーザー、全 API エンドポイント

## 状況
調査中...

## 担当者
@sre-team
```

**詳細記録の例** (問題解決後):
```markdown
# [CRITICAL] 本番環境 API が 500 エラー

## 発生日時
2025-12-31 14:23:00

## 影響範囲
全ユーザー、全 API エンドポイント（約15分間）

## 原因
メモリリークによる OOM (Out of Memory)

## 解決策
サーバー再起動 + メモリリーク修正

## 予防策
- メモリ監視アラート追加
- 定期的なメモリプロファイリング

## 詳細
（詳細な調査結果・コード例）

## 担当者
@sre-team
```

### ワークフロー統合

**Git との統合**:
```bash
# 問題修正時のコミットメッセージ
git commit -m "fix: resolve login token error

Incident: INC-2025-001
Severity: HIGH
Root cause: Missing JWT_PUBLIC_KEY environment variable
Reference: incidents/2025/001-login-token-error.md
"
```

**Issue Tracker との連携**:
```markdown
# GitHub Issue

## Issue #123: ログイン時にトークンエラー

**Labels**: `bug`, `high-priority`, `backend`

**Incident Report**: [INC-2025-001](incidents/2025/001-login-token-error.md)

**Status**: Resolved

**Resolution**:
- Root cause identified and fixed
- Prevention measures implemented
- Incident documented for future reference
```

**CI/CD との統合**:
```yaml
# .github/workflows/incident-check.yml
name: Incident Check

on:
  push:
    paths:
      - 'incidents/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Validate incident format
        run: |
          # インシデント記録の形式チェック
          python scripts/validate_incident.py

      - name: Update incident index
        run: |
          # インシデント一覧を自動更新
          python scripts/update_incident_index.py

      - name: Notify team
        run: |
          # チームに通知
          curl -X POST $SLACK_WEBHOOK \
            -d '{"text": "New incident recorded: ${{ github.event.head_commit.message }}"}'
```

---

## ツールと自動化

### 1. ファイルベース管理

**ディレクトリ構造**:
```
incidents/
├── 2025/
│   ├── 001-login-token-error.md
│   ├── 002-ios-white-screen.md
│   └── 003-dark-mode-text.md
├── templates/
│   ├── critical.md
│   ├── high.md
│   ├── medium.md
│   └── knowledge.md
├── README.md
└── INDEX.md  # 自動生成される一覧
```

**テンプレート** (`templates/high.md`):
```markdown
# [HIGH] タイトルを入力

## 発生日時
YYYY-MM-DD HH:MM:SS

## 影響範囲
-

## 発生状況


## 再現手順
1.
2.
3.

## 原因


## 解決策
```[language]
# コード例
```

## 対応履歴
- HH:MM -

## 予防策
- [ ]

## 担当者
- 報告: @
- 対応: @
```

**自動生成スクリプト** (`scripts/new-incident.sh`):
```bash
#!/bin/bash

# 使い方: ./new-incident.sh HIGH "ログインエラー"

SEVERITY=$1
TITLE=$2
YEAR=$(date +%Y)
MONTH=$(date +%m)
DAY=$(date +%d)
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# インシデント番号を自動採番
LAST_NUM=$(ls incidents/$YEAR/ | grep -oE '^[0-9]+' | sort -n | tail -1)
NEXT_NUM=$(printf "%03d" $((10#$LAST_NUM + 1)))

FILENAME="incidents/$YEAR/${NEXT_NUM}-${TITLE// /-}.md"

# テンプレートからコピー
cp "incidents/templates/${SEVERITY,,}.md" "$FILENAME"

# 日時を自動挿入
sed -i '' "s/YYYY-MM-DD HH:MM:SS/$TIMESTAMP/" "$FILENAME"

echo "Created: $FILENAME"
echo "Please edit the file and commit."

# エディタで開く
${EDITOR:-vim} "$FILENAME"
```

### 2. Notion データベース

**データベース構造**:
```
Incidents Database
├─ ID (auto)
├─ Title (text)
├─ Severity (select: CRITICAL, HIGH, MEDIUM, LOW, KNOWLEDGE)
├─ Status (select: Open, In Progress, Resolved)
├─ Date (date)
├─ Impact (text)
├─ Root Cause (text)
├─ Solution (text)
├─ Prevention (text)
├─ Assignee (person)
├─ Tags (multi-select)
└─ Related Issues (relation)
```

**ビューの活用**:
- **Table View**: 全インシデント一覧
- **Board View**: ステータス別（Open / In Progress / Resolved）
- **Calendar View**: 発生日時でカレンダー表示
- **Gallery View**: カード形式で視覚的に
- **Timeline View**: 時系列で表示

### 3. GitHub Issues / Projects

**Issue テンプレート** (`.github/ISSUE_TEMPLATE/incident.md`):
```markdown
---
name: Incident Report
about: Record an incident for tracking and learning
title: '[SEVERITY] Brief description'
labels: incident
assignees: ''
---

## Incident Details

**Severity**: [CRITICAL / HIGH / MEDIUM / LOW / KNOWLEDGE]

**Date & Time**: YYYY-MM-DD HH:MM:SS

**Impact**:
-

**Status**: Open

## Description

### What happened?


### Reproduction steps
1.
2.
3.

## Investigation

### Root cause


### Solution
```[language]

```

## Prevention

- [ ]
- [ ]

## Related

- PR: #
- Commit:
- Documentation:
```

**プロジェクトボード**:
```
Incident Tracking Board
├─ 📥 New
├─ 🔍 Investigating
├─ 🛠 Fixing
├─ ✅ Resolved
└─ 📚 Documented
```

### 4. Slack 統合

**インシデント報告ボット**:
```javascript
// Slack Bot (Node.js)
const { App } = require('@slack/bolt');

const app = new App({
  token: process.env.SLACK_BOT_TOKEN,
  signingSecret: process.env.SLACK_SIGNING_SECRET
});

// /incident コマンド
app.command('/incident', async ({ command, ack, respond }) => {
  await ack();

  // モーダルを表示
  await respond({
    response_type: 'ephemeral',
    blocks: [
      {
        type: 'section',
        text: {
          type: 'mrkdwn',
          text: '*New Incident Report*\nPlease fill in the details:'
        }
      },
      {
        type: 'input',
        block_id: 'title',
        label: { type: 'plain_text', text: 'Title' },
        element: { type: 'plain_text_input', action_id: 'title_input' }
      },
      {
        type: 'input',
        block_id: 'severity',
        label: { type: 'plain_text', text: 'Severity' },
        element: {
          type: 'static_select',
          action_id: 'severity_select',
          options: [
            { text: { type: 'plain_text', text: 'CRITICAL' }, value: 'critical' },
            { text: { type: 'plain_text', text: 'HIGH' }, value: 'high' },
            { text: { type: 'plain_text', text: 'MEDIUM' }, value: 'medium' },
            { text: { type: 'plain_text', text: 'LOW' }, value: 'low' }
          ]
        }
      },
      {
        type: 'input',
        block_id: 'description',
        label: { type: 'plain_text', text: 'Description' },
        element: { type: 'plain_text_input', action_id: 'description_input', multiline: true }
      },
      {
        type: 'actions',
        elements: [
          {
            type: 'button',
            text: { type: 'plain_text', text: 'Submit' },
            action_id: 'submit_incident',
            style: 'primary'
          }
        ]
      }
    ]
  });
});

// インシデント送信処理
app.action('submit_incident', async ({ ack, body, client }) => {
  await ack();

  const { title, severity, description } = extractFormData(body);

  // GitHub Issue 作成
  await createGitHubIssue(title, severity, description);

  // Notion に記録
  await createNotionPage(title, severity, description);

  // チャンネルに通知
  await client.chat.postMessage({
    channel: '#incidents',
    text: `🚨 New Incident: [${severity.toUpperCase()}] ${title}`,
    blocks: [
      {
        type: 'section',
        text: {
          type: 'mrkdwn',
          text: `*[${severity.toUpperCase()}] ${title}*\n${description}`
        }
      },
      {
        type: 'context',
        elements: [
          {
            type: 'mrkdwn',
            text: `Reported by <@${body.user.id}> | View in <https://github.com/org/repo/issues|GitHub> | <https://notion.so/incidents|Notion>`
          }
        ]
      }
    ]
  });
});

app.start(3000);
```

### 5. CLI ツール

**インシデント記録 CLI**:
```javascript
#!/usr/bin/env node

// incident-cli.js
const inquirer = require('inquirer');
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

async function main() {
  console.log('📝 Incident Recording CLI\n');

  const answers = await inquirer.prompt([
    {
      type: 'list',
      name: 'severity',
      message: 'Severity:',
      choices: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'KNOWLEDGE']
    },
    {
      type: 'input',
      name: 'title',
      message: 'Title:',
      validate: input => input.length > 0
    },
    {
      type: 'input',
      name: 'impact',
      message: 'Impact (who/what was affected):',
      validate: input => input.length > 0
    },
    {
      type: 'editor',
      name: 'description',
      message: 'Description (opens editor):'
    },
    {
      type: 'confirm',
      name: 'createIssue',
      message: 'Create GitHub Issue?',
      default: true
    }
  ]);

  // インシデントファイル作成
  const year = new Date().getFullYear();
  const timestamp = new Date().toISOString();
  const incidentDir = path.join('incidents', year.toString());

  if (!fs.existsSync(incidentDir)) {
    fs.mkdirSync(incidentDir, { recursive: true });
  }

  const lastNum = getLastIncidentNumber(incidentDir);
  const nextNum = String(lastNum + 1).padStart(3, '0');
  const filename = `${nextNum}-${answers.title.toLowerCase().replace(/\s+/g, '-')}.md`;
  const filepath = path.join(incidentDir, filename);

  const content = generateIncidentMarkdown(answers, timestamp);
  fs.writeFileSync(filepath, content);

  console.log(`\n✅ Incident recorded: ${filepath}`);

  // GitHub Issue 作成
  if (answers.createIssue) {
    const issueTitle = `[${answers.severity}] ${answers.title}`;
    const issueBody = `Incident Report: ${filepath}\n\n${answers.description}`;

    try {
      const issueUrl = execSync(
        `gh issue create --title "${issueTitle}" --body "${issueBody}" --label incident`,
        { encoding: 'utf-8' }
      ).trim();

      console.log(`✅ GitHub Issue created: ${issueUrl}`);

      // インシデントファイルにIssue URLを追記
      fs.appendFileSync(filepath, `\n## Related\n- Issue: ${issueUrl}\n`);
    } catch (error) {
      console.error('❌ Failed to create GitHub Issue:', error.message);
    }
  }

  // Git commit
  try {
    execSync(`git add ${filepath}`);
    execSync(`git commit -m "docs: add incident ${nextNum} - ${answers.title}"`);
    console.log('✅ Changes committed');
  } catch (error) {
    console.error('❌ Failed to commit:', error.message);
  }
}

function getLastIncidentNumber(dir) {
  const files = fs.readdirSync(dir);
  const numbers = files
    .map(f => parseInt(f.split('-')[0]))
    .filter(n => !isNaN(n));
  return numbers.length > 0 ? Math.max(...numbers) : 0;
}

function generateIncidentMarkdown(answers, timestamp) {
  return `# [${answers.severity}] ${answers.title}

## 発生日時
${timestamp}

## 影響範囲
${answers.impact}

## 発生状況
${answers.description}

## 原因
（調査中）

## 解決策
（対応中）

## 予防策
- [ ]

## 担当者
- 報告: @${process.env.USER}
`;
}

main().catch(console.error);
```

**使い方**:
```bash
# インストール
npm install -g incident-cli

# 実行
incident
```

---

## 実践例

### ケース1: API エラーの記録

**状況**: 本番環境で突然 API が 500 エラーを返し始めた

**記録プロセス**:

**1. 速報記録（1分以内）**:
```bash
# CLI で速報作成
incident --quick --severity CRITICAL --title "API 500 errors in production"
```

生成されるファイル:
```markdown
# [CRITICAL] API 500 errors in production

## 発生日時
2025-12-31 14:23:00

## 影響範囲
調査中

## 状況
本番環境の全 API エンドポイントが 500 エラー

## 担当者
@sre-team

---
*This is a quick incident report. Full details will be added after resolution.*
```

**2. 調査・対応中の更新**:
```markdown
# [CRITICAL] API 500 errors in production

## 発生日時
2025-12-31 14:23:00

## 影響範囲
全ユーザー、全 API エンドポイント

## 状況
本番環境の全 API エンドポイントが 500 エラー

## 調査進捗
- [x] 14:23 - 問題検知
- [x] 14:25 - ログ確認開始
- [x] 14:30 - 原因特定（DB接続プール枯渇）
- [ ] 対応中

## 暫定対応
サーバー再起動実施中...

## 担当者
@sre-team
```

**3. 解決後の完全記録**:
```markdown
# [CRITICAL] API 500 errors in production

## 発生日時
2025-12-31 14:23:00 JST

## 影響範囲
- 全ユーザー（推定 12,000 名）
- 全 API エンドポイント
- 継続時間: 約 15 分

## 発生状況
デプロイ後 5 分で、全 API リクエストが 500 エラーを返し始めた。
ヘルスチェックも失敗し、アラートが大量発火。

## 原因
DB 接続プールの設定ミス。

```javascript
// 問題のコード (config/database.js:12)
const pool = new Pool({
  max: 10,  // 本番負荷に対して不足
  idleTimeoutMillis: 30000
});
```

本番環境では同時接続数が平均 50、ピーク時 100 に達するため、
`max: 10` では接続プールが即座に枯渇。

## 解決策

**即時対応**:
1. サーバー再起動で一時的に接続プールをリセット
2. 接続プール設定を緊急変更

```javascript
// 修正後 (config/database.js:12)
const pool = new Pool({
  max: 100,  // 本番負荷に対応
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000
});
```

**恒久対策**:
```javascript
// 環境変数から設定を取得
const pool = new Pool({
  max: parseInt(process.env.DB_POOL_MAX) || 100,
  min: parseInt(process.env.DB_POOL_MIN) || 10,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000
});

// 起動時にプール状態を監視
pool.on('connect', () => {
  console.log('New client connected to pool');
});

pool.on('error', (err) => {
  console.error('Unexpected pool error:', err);
  // アラート送信
});
```

## 対応履歴
- 14:23 - 問題検知（監視アラート）
- 14:25 - 初動チーム招集
- 14:30 - 原因特定（DB 接続プール枯渇）
- 14:32 - サーバー再起動
- 14:35 - 接続プール設定変更
- 14:38 - サービス復旧確認
- 15:00 - 完全復旧宣言

## 予防策
- [x] DB 接続プール設定を環境変数化
- [x] ステージング環境で負荷テスト実施
- [x] 接続プール監視アラート追加
- [x] デプロイチェックリストに接続プール設定確認を追加
- [ ] 自動負荷テスト CI/CD 統合（1週間以内）

## メトリクス
- **MTTR** (Mean Time To Repair): 15分
- **影響ユーザー数**: 約 12,000 名
- **ダウンタイム**: 15分
- **推定売上損失**: 約 $5,000

## 学んだ教訓
1. **ステージング環境は本番同等の負荷テストが必須**
   - 低負荷では検知できない問題が本番で顕在化
2. **DB 関連設定変更は特に慎重に**
   - 接続プール、タイムアウト、リトライ戦略など
3. **ロールバック手順の事前確認が重要**
   - 緊急時に迅速な対応が可能

## 参考情報
- Commit: `abc1234`
- PR: #456
- Related Incidents: INC-2025-003 (類似問題)
- Documentation: docs/operations/database-tuning.md

## 担当者
- 検知: @monitoring-team
- 初動対応: @sre-team
- 根本対策: @backend-team
- レビュー: @tech-lead
```

### ケース2: iOS ビルドエラー

**状況**: `Archive` 時に謎のビルドエラー

**記録**:
```markdown
# [MEDIUM] Xcode Archive 時に "Command PhaseScriptExecution failed" エラー

## 発生日時
2025-12-30 16:45:00

## 影響範囲
- iOS チーム全員
- App Store へのリリース作業がブロック

## 発生状況
`Product > Archive` 実行時にビルドが失敗。
エラーメッセージが不明瞭で、原因特定に時間がかかった。

## エラーメッセージ
```
Command PhaseScriptExecution failed with a nonzero exit code
```

## 原因
Firebase Crashlytics の Run Script Phase で、
`GOOGLE_APP_ID` 環境変数が未設定だったため。

```bash
# Run Script Phase の内容
"${PODS_ROOT}/FirebaseCrashlytics/run"

# エラー詳細（Build Report から確認）
Error: Missing GOOGLE_APP_ID
```

## 解決策

**1. `GoogleService-Info.plist` を確認**:
```xml
<!-- GoogleService-Info.plist -->
<key>GOOGLE_APP_ID</key>
<string>1:1234567890:ios:abcdef1234567890</string>
```

**2. Run Script Phase に環境変数を追加**:
```bash
# Xcode > Build Phases > Run Script
export GOOGLE_APP_ID=$(cat GoogleService-Info.plist | grep -A 1 "GOOGLE_APP_ID" | tail -1 | sed 's/.*<string>\(.*\)<\/string>.*/\1/')

"${PODS_ROOT}/FirebaseCrashlytics/run"
```

または、より簡潔に:
```bash
# GoogleService-Info.plist が Info.plist に統合されている場合
/usr/libexec/PlistBuddy -c "Print :GOOGLE_APP_ID" "${BUILT_PRODUCTS_DIR}/${INFOPLIST_PATH}" > /dev/null
"${PODS_ROOT}/FirebaseCrashlytics/run"
```

**3. ビルド成功確認**:
```bash
# Clean Build
rm -rf ~/Library/Developer/Xcode/DerivedData
Product > Clean Build Folder

# Archive
Product > Archive
```

## 予防策
- [x] チーム Wiki に Firebase 初期設定手順を追加
- [x] プロジェクト README に環境構築チェックリスト追加
- [x] CI/CD で Archive ビルドテストを追加

## 参考情報
- Firebase Crashlytics Docs: https://firebase.google.com/docs/crashlytics/get-started?platform=ios
- Stack Overflow: https://stackoverflow.com/questions/...

## 担当者
- 報告: @ios-team
- 調査: @ios-team
- 解決: @ios-team
```

### ケース3: ナレッジ共有

**状況**: よく発生する問題をナレッジとして記録

**記録**:
```markdown
# [KNOWLEDGE] Swift Package Manager のキャッシュ問題解決法

## 問題
SPM でパッケージ更新後も古いバージョンがビルドされる問題が頻発。

## 症状
- Package.swift でバージョン変更しても反映されない
- ビルドエラー: "Module 'XXX' has no member 'YYY'"
- Clean Build しても解決しない

## 解決策

### 方法1: Xcode メニューから
```
1. Product > Clean Build Folder (Cmd+Shift+K)
2. File > Packages > Reset Package Caches
3. File > Packages > Update to Latest Package Versions
4. Xcode を再起動
5. ビルドを再実行
```

### 方法2: ターミナルから
```bash
# Derived Data を削除
rm -rf ~/Library/Developer/Xcode/DerivedData

# SPM キャッシュをクリア
rm -rf ~/Library/Caches/org.swift.swiftpm

# Xcode を再起動
killall Xcode
open YourProject.xcodeproj

# ビルド
xcodebuild clean build
```

### 方法3: Package.resolved を削除
```bash
# Package.resolved を削除して再生成
rm YourProject.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved

# Xcode で再度パッケージ解決
open YourProject.xcodeproj
# File > Packages > Resolve Package Versions
```

## 予防策
- パッケージバージョン変更時は必ずキャッシュクリア
- CI/CD では常にクリーンビルド
- `.gitignore` に Derived Data を追加

```gitignore
# Xcode
DerivedData/
*.xcworkspace/xcshareddata/swiftpm/
```

## スクリプト化
```bash
# clean-spm.sh
#!/bin/bash

echo "🧹 Cleaning SPM caches..."

rm -rf ~/Library/Developer/Xcode/DerivedData
rm -rf ~/Library/Caches/org.swift.swiftpm
rm -rf *.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved

echo "✅ SPM caches cleared!"
echo "Please restart Xcode and rebuild."
```

## 参考
- Apple Developer Forums: https://developer.apple.com/forums/
- SPM Documentation: https://swift.org/package-manager/
- チーム Wiki: SPM トラブルシューティング

## タグ
`swift`, `spm`, `xcode`, `cache`, `troubleshooting`

## 更新履歴
- 2025-12-30: 初版作成 (@ios-team)
- 2025-12-31: スクリプト化セクション追加 (@ios-team)
```

---

## チーム運用

### 1. 記録文化の醸成

**チームガイドライン**:
```markdown
# インシデント記録ガイドライン

## 基本原則

### 1. 失敗は学習の機会
- 失敗を責めない
- 問題を隠さない
- オープンに共有する

### 2. 全員が記録者
- エンジニアだけでなく、全員が記録
- QA、デザイナー、PM も記録可能
- 小さな問題でも記録する

### 3. 即座に記録
- 問題発生後 10分以内に速報
- 詳細は後から追記 OK
- 完璧を求めない

## 記録するタイミング

✅ **記録すべき**:
- ビルドエラーで 10分以上詰まった
- ドキュメントと実際の動作が違った
- 設定ミスで時間を無駄にした
- ライブラリのバージョン問題
- 本番環境での不具合
- ユーザーからの問い合わせ

❌ **記録不要**:
- タイポによる単純ミス
- 既知の問題（既存インシデントを参照）
- 1分で解決した問題

## 記録フォーマット

最低限必要な情報:
1. **タイトル**: 何が起きたか（1行）
2. **日時**: いつ起きたか
3. **原因**: なぜ起きたか
4. **解決策**: どう解決したか

## レビュー

- 週次でインシデントレビュー会を実施
- パターン分析で再発防止
- チーム全体で学びを共有
```

### 2. レビュー会の実施

**週次インシデントレビュー**:
```markdown
# 週次インシデントレビュー

## 目的
- 今週発生した問題の共有
- パターン認識と再発防止
- チーム学習

## アジェンダ（45分）

### 1. インシデントサマリー（10分）
- 今週のインシデント件数
- 重大度別の内訳
- 頻出カテゴリ

### 2. 主要インシデントの深掘り（25分）
- CRITICAL/HIGH のインシデントを1-2件ピックアップ
- 担当者がプレゼン
- Q&A、ディスカッション

### 3. パターン分析（5分）
- 繰り返し発生している問題はないか
- 共通の根本原因はないか

### 4. アクションアイテム（5分）
- 予防策の優先順位付け
- 担当者・期限の設定
```

**レビュー資料例**:
```markdown
# 週次インシデントレビュー - 2025年 第1週

## サマリー

| 項目 | 数値 |
|------|------|
| 総インシデント数 | 12件 |
| CRITICAL | 1件 |
| HIGH | 3件 |
| MEDIUM | 5件 |
| LOW | 2件 |
| KNOWLEDGE | 1件 |

## カテゴリ別

| カテゴリ | 件数 |
|----------|------|
| Backend API | 4件 |
| iOS ビルド | 3件 |
| デプロイ | 2件 |
| 環境設定 | 2件 |
| その他 | 1件 |

## 主要インシデント

### 1. [CRITICAL] 本番環境 API が 500 エラー
- **発生日時**: 12/31 14:23
- **原因**: DB 接続プール設定ミス
- **MTTR**: 15分
- **影響**: 全ユーザー（約12,000名）
- **学び**: ステージング環境での負荷テスト強化
- **アクション**:
  - [x] 接続プール監視アラート追加
  - [ ] 自動負荷テスト CI/CD 統合（担当: @devops、期限: 1/7）

### 2. [HIGH] iOS Archive ビルド失敗
- **発生日時**: 12/30 16:45
- **原因**: Firebase 環境変数未設定
- **MTTR**: 2時間
- **影響**: iOS チーム、リリース遅延
- **学び**: Firebase 初期設定ドキュメント不足
- **アクション**:
  - [x] チーム Wiki に設定手順追加
  - [x] CI/CD で Archive テスト追加

## パターン分析

### 繰り返し発生している問題
1. **環境変数の設定漏れ**（3件）
   - 本番/ステージング/開発環境で設定が異なる
   - → 環境変数管理ツール導入を検討
2. **SPM キャッシュ問題**（2件）
   - パッケージ更新後にキャッシュが原因でビルドエラー
   - → 自動キャッシュクリアスクリプト導入

## アクションアイテム

| アクション | 担当 | 期限 | 優先度 |
|-----------|------|------|--------|
| 自動負荷テスト CI/CD 統合 | @devops | 1/7 | HIGH |
| 環境変数管理ツール調査 | @backend | 1/10 | MEDIUM |
| SPM キャッシュクリアスクリプト | @ios | 1/5 | LOW |

## Next Steps
- [ ] 来週も継続してパターン監視
- [ ] 月次レポートで経営層に報告
```

### 3. メトリクス管理

**追跡すべき指標**:
```markdown
## インシデントメトリクス

### 1. 量的指標
- **総インシデント数**: 週次/月次
- **重大度別件数**: CRITICAL/HIGH/MEDIUM/LOW
- **カテゴリ別件数**: Backend/Frontend/iOS/Android/DevOps

### 2. 時間指標
- **MTTD** (Mean Time To Detect): 検知までの平均時間
- **MTTR** (Mean Time To Repair): 修復までの平均時間
- **MTTF** (Mean Time To Failure): 障害間隔の平均時間

### 3. 影響指標
- **影響ユーザー数**: 推定
- **ダウンタイム**: 分/時間
- **売上損失**: 推定金額

### 4. 学習指標
- **再発率**: 同じ問題の再発頻度
- **予防策実施率**: 予防策の実施割合
- **ナレッジ活用率**: 過去のインシデントが参照された回数
```

**ダッシュボード例**:
```markdown
# インシデントダッシュボード

## 今月のサマリー

| 指標 | 今月 | 先月 | 変化 |
|------|------|------|------|
| 総インシデント数 | 48件 | 52件 | ▼ 7.7% |
| CRITICAL | 2件 | 3件 | ▼ 33% |
| MTTR | 32分 | 45分 | ▼ 28.9% |
| 再発インシデント | 5件 | 8件 | ▼ 37.5% |

## 傾向

### 📉 改善傾向
- MTTR が大幅短縮（45分 → 32分）
- 再発インシデント減少

### 📈 注意が必要
- iOS 関連インシデントが増加（10件 → 15件）
- 環境設定問題が継続

## トップ5カテゴリ

1. iOS ビルド - 15件
2. Backend API - 12件
3. デプロイ - 8件
4. 環境設定 - 7件
5. Frontend - 6件
```

---

## まとめ

### 記録のベストプラクティス

**記録の5原則**:
1. **即座に記録**: 問題発生後10分以内に速報
2. **具体的に記録**: 5W1Hを明確に
3. **コード例を含める**: 再現・解決に必要なコード
4. **予防策を明記**: 今後どう防ぐか
5. **チームで共有**: 個人の学びをチームの資産に

**記録フォーマット**:
```markdown
# [優先度] タイトル（1行で要約）

## 発生日時
YYYY-MM-DD HH:MM:SS

## 影響範囲
誰が、何が、どの程度影響を受けたか

## 原因
なぜ起きたか

## 解決策
どう解決したか（コード例含む）

## 予防策
今後どう防ぐか

## 担当者
誰が対応したか
```

### ツール選択ガイド

| ツール | 適している場合 | 不適な場合 |
|--------|--------------|-----------|
| **Markdown + Git** | 小規模チーム、シンプル、バージョン管理重視 | 複雑な検索、リレーション管理 |
| **Notion** | 視覚的、検索、リレーション | 自動化、API統合 |
| **GitHub Issues** | コードと密接、PR連携 | 非エンジニアの利用 |
| **Slack** | リアルタイム通知、ディスカッション | 長期保存、構造化データ |
| **Jira/Linear** | プロジェクト管理と統合 | 軽量・シンプルな記録 |

### チェックリスト

**インシデント記録時**:
- [ ] タイトルは1行で問題を要約しているか
- [ ] 発生日時は正確か
- [ ] 影響範囲は明確か
- [ ] 原因は特定できているか
- [ ] 解決策は再現可能か（コード例含む）
- [ ] 予防策は具体的か
- [ ] 担当者は明記されているか
- [ ] 関連情報（PR、Issue、Commit）はリンクされているか

**チーム運用**:
- [ ] 週次でインシデントレビュー会を実施
- [ ] メトリクスを追跡・可視化
- [ ] 再発防止策の進捗を管理
- [ ] ナレッジベースとして活用
- [ ] 新メンバーのオンボーディング資料として活用

---

## 次のステップ

1. **02-root-cause-analysis.md**: 根本原因分析（RCA）の手法
2. **03-prevention-strategy.md**: 再発防止と継続的改善

**関連スキル**:
- `lessons-learned`: 教訓データベースの構築
- `testing-strategy`: テスト戦略でインシデント予防
- `code-review`: レビューで問題を早期発見

---

*このガイドは実践的なインシデント記録の基礎です。継続的に進化させていきましょう。*
