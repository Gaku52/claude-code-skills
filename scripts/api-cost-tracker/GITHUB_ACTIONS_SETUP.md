# GitHub Actions セットアップ手順

GitHub Actions で自動的に日次・週次・月次レポートを送信する設定方法。

## ⚠️ 重要: セキュリティ

このワークフローは **Gaku52（リポジトリオーナー）のみ** が実行できるように制限されています。

```yaml
if: |
  github.repository_owner == 'Gaku52' &&
  (github.actor == 'Gaku52' || github.event_name == 'schedule')
```

- フォークしたユーザーは実行できません
- 他のコントリビューターも実行できません
- スケジュール実行は自動的に実行されます
- 手動実行（workflow_dispatch）はあなただけが可能

## GitHub Secrets の設定

### 1. GitHub リポジトリで Secrets を追加

https://github.com/Gaku52/claude-code-skills/settings/secrets/actions

「New repository secret」をクリックして以下を追加:

#### 必須の Secrets

| Name | Value (例) | 説明 |
|------|-----------|------|
| `API_COST_EMAIL_PROVIDER` | `smtp` | メール送信方法 (smtp または aws-ses) |
| `API_COST_SMTP_HOST` | `smtp.gmail.com` | SMTPサーバー |
| `API_COST_SMTP_PORT` | `587` | SMTPポート |
| `API_COST_SMTP_USER` | `your-email@gmail.com` | SMTPユーザー名 |
| `API_COST_SMTP_PASS` | `xxxx-xxxx-xxxx-xxxx` | Gmailアプリパスワード |
| `API_COST_SMTP_FROM` | `your-email@gmail.com` | 送信元メールアドレス |
| `API_COST_NOTIFY_EMAIL` | `your-email@gmail.com` | 受信先メールアドレス |

#### 閾値の Secrets

| Name | Value (例) | 説明 |
|------|-----------|------|
| `API_COST_THRESHOLD_DAILY` | `10` | 日次閾値（ドル） |
| `API_COST_THRESHOLD_WEEKLY` | `50` | 週次閾値（ドル） |
| `API_COST_THRESHOLD_MONTHLY` | `200` | 月次閾値（ドル） |

#### AWS SES を使う場合（オプション）

| Name | Value | 説明 |
|------|-------|------|
| `API_COST_EMAIL_PROVIDER` | `aws-ses` | AWS SES を使用 |
| `API_COST_AWS_REGION` | `us-east-1` | AWSリージョン |
| `API_COST_AWS_ACCESS_KEY_ID` | `AKIA...` | AWS Access Key |
| `API_COST_AWS_SECRET_ACCESS_KEY` | `xxxxx` | AWS Secret Key |
| `API_COST_AWS_SES_FROM` | `noreply@yourdomain.com` | 送信元（認証済み） |

### 2. Gmail アプリパスワードの取得

1. https://myaccount.google.com/apppasswords にアクセス
2. 「アプリパスワード」を生成
3. 16桁のパスワード（例: `xxxx-xxxx-xxxx-xxxx`）を `API_COST_SMTP_PASS` に設定

## ワークフローの実行

### 自動実行（スケジュール）

以下のスケジュールで自動実行されます（UTC時間）:

- **日次レポート**: 毎日 00:00 UTC (09:00 JST)
- **週次レポート**: 毎週月曜 00:00 UTC (09:00 JST)
- **月次レポート**: 毎月1日 00:00 UTC (09:00 JST)

### 手動実行

GitHub リポジトリの「Actions」タブから手動実行できます:

1. https://github.com/Gaku52/claude-code-skills/actions
2. 左側から実行したいワークフローを選択
   - `API Cost - Daily Report`
   - `API Cost - Weekly Report`
   - `API Cost - Monthly Report`
3. 「Run workflow」ボタンをクリック
4. 「Run workflow」を再度クリックして確認

**注意**: あなた（Gaku52）がログインしている必要があります。

## トラブルシューティング

### ワークフローが実行されない

**原因**: フォークしたリポジトリや他のユーザーは実行できません。

**確認**:
```yaml
if: github.repository_owner == 'Gaku52' && github.actor == 'Gaku52'
```

### メール送信が失敗する

**確認項目**:
1. GitHub Secrets が正しく設定されているか
2. Gmail アプリパスワードが正しいか
3. ワークフローログでエラーを確認

### スケジュールが動かない

GitHub Actions のスケジュール実行は最大15分の遅延があります。また、リポジトリの activity が低い場合、スケジュールが遅延することがあります。

手動実行（workflow_dispatch）で正常に動作するか確認してください。

## セキュリティベストプラクティス

### ✅ 実装済み

- [x] アクター制限（Gaku52のみ）
- [x] リポジトリオーナー制限
- [x] Secrets による機密情報管理

### 推奨設定

#### Branch Protection

https://github.com/Gaku52/claude-code-skills/settings/branches

`main` ブランチに以下を設定:
- ✅ Require pull request reviews before merging
- ✅ Require status checks to pass before merging
- ✅ Restrict who can push to matching branches (あなたのみ)

#### Actions Permissions

https://github.com/Gaku52/claude-code-skills/settings/actions

- ✅ Allow select actions and reusable workflows
- ✅ Require approval for all outside collaborators

## データベースについて

GitHub Actions は毎回クリーンな環境で実行されるため、データベースは永続化されません。

**現在の仕様**:
- 毎回新しいデータベースを作成
- レポートは空の状態で送信される

**永続化が必要な場合**:
1. **ローカル環境で実行** (推奨)
   - PM2 で scheduler を起動
   - データベースがローカルに保存される

2. **GitHub Actions で永続化**
   - Artifacts でデータベースをアップロード/ダウンロード
   - または外部ストレージ（AWS S3等）を使用

## まとめ

- ✅ あなた（Gaku52）だけが実行可能
- ✅ GitHub Secrets で機密情報を管理
- ✅ 毎日/毎週/毎月自動でレポート送信
- ✅ 手動実行も可能

データの永続化が必要な場合は、ローカル環境での実行を推奨します。
