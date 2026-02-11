# Claude API Cost Tracker & Email Notifier

Claude APIの使用量（トークン数）を追跡し、料金を計算してメールで通知するシステム。

## 機能

1. **使用量トラッキング**: Claude APIレスポンスから入力・出力トークン数を記録
2. **料金計算**: Sonnet 4.5の料金レート($3/MTok入力、$15/MTok出力)で計算
3. **メール通知**:
   - 日次レポート（毎日の使用量と料金）
   - 週次レポート（週間の集計）
   - 閾値アラート（設定金額を超えた場合）

## アーキテクチャ

```
┌─────────────────┐
│ Claude API Call │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Usage Tracker   │  ← APIレスポンスからトークン数を抽出
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Local DB        │  ← SQLite に使用履歴を保存
│ (usage.db)      │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Cost Calculator │  ← 料金計算
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Email Notifier  │  ← nodemailer / AWS SES でメール送信
└─────────────────┘
```

## セットアップ

### 1. 依存関係のインストール

```bash
cd scripts/api-cost-tracker
npm install
```

### 2. 環境変数の設定

```bash
# .env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password
NOTIFY_EMAIL=recipient@example.com

# 料金閾値（ドル）
COST_THRESHOLD_DAILY=10
COST_THRESHOLD_WEEKLY=50
COST_THRESHOLD_MONTHLY=200

# レポート送信時刻（24時間形式）
DAILY_REPORT_TIME=09:00
WEEKLY_REPORT_DAY=monday
```

### 3. データベース初期化

```bash
npm run init-db
```

## 使用方法

### トラッキングの統合

```typescript
import { trackUsage } from './scripts/api-cost-tracker/tracker';

// Claude API呼び出し後
const response = await anthropic.messages.create({
  model: "claude-sonnet-4-20250514",
  messages: [{ role: "user", content: "Hello" }]
});

// 使用量を記録
await trackUsage({
  modelId: response.model,
  inputTokens: response.usage.input_tokens,
  outputTokens: response.usage.output_tokens,
  timestamp: new Date(),
  requestId: response.id,
});
```

### 手動レポート生成

```bash
# 今日の使用量レポート
npm run report:daily

# 今週の使用量レポート
npm run report:weekly

# 今月の使用量レポート
npm run report:monthly
```

### スケジューラー起動

```bash
# 定期レポート送信を開始
npm run start:scheduler

# またはPM2で永続化
pm2 start ecosystem.config.js
```

## 料金レート

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| Claude Sonnet 4.5 | $3.00 | $15.00 |
| Claude Sonnet 3.5 | $3.00 | $15.00 |
| Claude Haiku 3.5 | $0.80 | $4.00 |
| Claude Opus 3 | $15.00 | $75.00 |

## データベーススキーマ

```sql
CREATE TABLE usage_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id TEXT NOT NULL,
  model_id TEXT NOT NULL,
  input_tokens INTEGER NOT NULL,
  output_tokens INTEGER NOT NULL,
  input_cost REAL NOT NULL,
  output_cost REAL NOT NULL,
  total_cost REAL NOT NULL,
  timestamp DATETIME NOT NULL,
  metadata TEXT
);

CREATE INDEX idx_timestamp ON usage_history(timestamp);
CREATE INDEX idx_model_id ON usage_history(model_id);
```

## レポートサンプル

### 日次レポート

```
Claude API Daily Usage Report - 2025-12-27
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model: claude-sonnet-4-20250514
Requests: 42

Input Tokens:  125,234 ($0.38)
Output Tokens: 89,456  ($1.34)
Total Cost:    $1.72

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Daily Cost: $1.72
Monthly Total (MTD): $12.45
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### アラートメール

```
⚠️ Cost Alert: Daily Threshold Exceeded

Current daily cost: $11.23
Threshold: $10.00

Please review your API usage.

View detailed report: https://dashboard.anthropic.com
```

## トラブルシューティング

### メール送信が失敗する

```bash
# SMTPサーバー接続テスト
npm run test:email

# Gmailの場合は「アプリパスワード」を使用
# https://myaccount.google.com/apppasswords
```

### データベースエラー

```bash
# データベースをリセット
npm run reset-db

# バックアップから復元
npm run restore-db backup/usage-2025-12-27.db
```

## ライセンス

MIT
