# デプロイ戦略

> Blue-Green、Canary、Rolling、Feature Flag など主要なデプロイ戦略を体系的に理解し、安全で高速なリリースを実現する

## この章で学ぶこと

1. **主要デプロイ戦略の仕組みと使い分け** — Blue-Green、Canary、Rolling Update、Recreate の動作原理と適用条件
2. **Feature Flag によるリリース制御** — コードデプロイとリリースを分離し、段階的ロールアウトを実現する手法
3. **ロールバック設計とリスク軽減** — 障害発生時に即座に復旧するための仕組みと運用プロセス
4. **DB マイグレーション戦略** — デプロイと連携したデータベーススキーマ変更の安全な実行方法
5. **自動化されたデプロイパイプラインの設計** — GitHub Actions と連携した CI/CD パイプラインの構築

---

## 1. デプロイ戦略の全体像

```
┌─────────────────────────────────────────────────────┐
│              デプロイ戦略の分類                        │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌───────────┐   ┌───────────┐   ┌───────────┐    │
│  │ Recreate  │   │  Rolling  │   │ Blue-Green│    │
│  │  (全停止)  │   │ (段階置換) │   │ (環境切替) │    │
│  └─────┬─────┘   └─────┬─────┘   └─────┬─────┘    │
│        │               │               │           │
│        ▼               ▼               ▼           │
│  ダウンタイム有    ゼロダウンタイム  ゼロダウンタイム   │
│  最もシンプル     リソース効率良    即座にロールバック  │
│                                                     │
│  ┌───────────┐   ┌───────────┐                     │
│  │  Canary   │   │  A/B Test │                     │
│  │ (段階公開) │   │ (比較検証) │                     │
│  └─────┬─────┘   └─────┬─────┘                     │
│        │               │                           │
│        ▼               ▼                           │
│  リスク最小化     データ駆動判断                      │
└─────────────────────────────────────────────────────┘
```

### 1.1 選択基準マトリクス

```
デプロイ戦略を選ぶ際の判断フロー:

  ダウンタイムは許容できるか？
       │
  ┌────┴────┐
  │ Yes     │ No
  │         │
  ↓         ├── リソースコストを最小化したいか？
  Recreate  │         │
            │    ┌────┴────┐
            │    │ Yes     │ No
            │    │         │
            │    ↓         ├── 段階的なリスク検証が必要か？
            │    Rolling   │         │
            │              │    ┌────┴────┐
            │              │    │ Yes     │ No
            │              │    │         │
            │              │    ↓         ↓
            │              │    Canary    Blue-Green
            │              │
            │              └── A/B テストが必要か？ → A/B Test
            │
            └── Feature Flag でリリースを制御するか？ → Feature Flag
```

---

## 2. Recreate デプロイ

全インスタンスを停止してから新バージョンを起動する、最もシンプルな戦略。

```
Recreate デプロイの流れ:

時間軸 ──────────────────────────────────────────►

ステップ1: 旧バージョンが稼働中
  [v1] [v1] [v1] ◄── 全トラフィック

ステップ2: 全インスタンスを停止（ダウンタイム開始）
  [---] [---] [---]  ← サービス停止

ステップ3: 新バージョンを起動（ダウンタイム終了）
  [v2] [v2] [v2] ◄── 全トラフィック
```

```yaml
# Kubernetes Recreate デプロイ
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  replicas: 3
  strategy:
    type: Recreate  # 全 Pod を停止後に新 Pod を起動
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
        - name: myapp
          image: myapp:v2.0.0
          ports:
            - containerPort: 3000
```

```
Recreate が適するケース:
  - 開発/ステージング環境
  - バッチ処理サーバー
  - DB スキーマの破壊的変更を伴うデプロイ
  - 旧バージョンと新バージョンの共存が不可能な場合
  - コスト制約が厳しく追加リソースを確保できない場合
```

---

## 3. Blue-Green デプロイ

2つの同一環境（Blue/Green）を用意し、トラフィックを一括で切り替える戦略。

```yaml
# docker-compose.blue-green.yml
version: "3.8"

services:
  app-blue:
    image: myapp:v1.0.0
    ports:
      - "8081:3000"
    environment:
      - ENV=production
      - SLOT=blue

  app-green:
    image: myapp:v1.1.0
    ports:
      - "8082:3000"
    environment:
      - ENV=production
      - SLOT=green

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - app-blue
      - app-green
```

```nginx
# nginx.conf — Blue-Green切替
upstream active_backend {
    # 現在のアクティブスロットを指定
    # Blue → Green に切り替えるときはここを変更
    server app-green:3000;
}

upstream standby_backend {
    server app-blue:3000;
}

server {
    listen 80;

    location / {
        proxy_pass http://active_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # ヘルスチェック用
    location /health {
        proxy_pass http://active_backend/health;
    }
}
```

```
Blue-Green デプロイの流れ:

時間軸 ──────────────────────────────────────────►

ステップ1: Blue(v1) が稼働中
  [Blue v1] ◄── 全トラフィック
  [Green   ] (待機)

ステップ2: Green に v2 をデプロイ
  [Blue v1] ◄── 全トラフィック
  [Green v2] (起動・テスト中)

ステップ3: ヘルスチェック通過後に切替
  [Blue v1 ] (待機 = ロールバック先)
  [Green v2] ◄── 全トラフィック

ステップ4: 問題なければ Blue を解放
  [Blue    ] (解放 or 次バージョン用)
  [Green v2] ◄── 全トラフィック
```

### 3.1 AWS ALB を使った Blue-Green

```yaml
# GitHub Actions での Blue-Green デプロイ
name: Blue-Green Deploy
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read
    steps:
      - uses: actions/checkout@v4

      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: ap-northeast-1

      - name: Determine target group
        id: target
        run: |
          # 現在のアクティブターゲットグループを取得
          ACTIVE_TG=$(aws elbv2 describe-rules \
            --listener-arn ${{ secrets.ALB_LISTENER_ARN }} \
            --query 'Rules[0].Actions[0].TargetGroupArn' --output text)

          if echo "$ACTIVE_TG" | grep -q "blue"; then
            echo "deploy_tg=green" >> "$GITHUB_OUTPUT"
            echo "active_tg=blue" >> "$GITHUB_OUTPUT"
          else
            echo "deploy_tg=blue" >> "$GITHUB_OUTPUT"
            echo "active_tg=green" >> "$GITHUB_OUTPUT"
          fi

      - name: Deploy to standby environment
        run: |
          # スタンバイ環境にデプロイ
          aws ecs update-service \
            --cluster production \
            --service myapp-${{ steps.target.outputs.deploy_tg }} \
            --task-definition myapp:latest \
            --force-new-deployment

          # サービスが安定するまで待機
          aws ecs wait services-stable \
            --cluster production \
            --services myapp-${{ steps.target.outputs.deploy_tg }}

      - name: Health check on standby
        run: |
          HEALTH_URL="https://${{ steps.target.outputs.deploy_tg }}.internal.example.com/health"
          for i in $(seq 1 10); do
            STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL")
            if [ "$STATUS" = "200" ]; then
              echo "Health check passed"
              exit 0
            fi
            echo "Attempt $i/10: Status $STATUS"
            sleep 5
          done
          echo "Health check failed"
          exit 1

      - name: Switch traffic
        run: |
          # ALB のリスナールールを更新してトラフィックを切り替え
          aws elbv2 modify-rule \
            --rule-arn ${{ secrets.ALB_RULE_ARN }} \
            --actions Type=forward,TargetGroupArn=${{ secrets[format('TG_{0}_ARN', steps.target.outputs.deploy_tg)] }}

          echo "Traffic switched to ${{ steps.target.outputs.deploy_tg }}"

      - name: Verify deployment
        run: |
          sleep 30  # DNS 伝播待ち
          STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://api.example.com/health)
          if [ "$STATUS" != "200" ]; then
            echo "::error::Deployment verification failed! Rolling back..."
            # ロールバック: 元のターゲットグループに戻す
            aws elbv2 modify-rule \
              --rule-arn ${{ secrets.ALB_RULE_ARN }} \
              --actions Type=forward,TargetGroupArn=${{ secrets[format('TG_{0}_ARN', steps.target.outputs.active_tg)] }}
            exit 1
          fi
          echo "Deployment verified successfully"
```

---

## 4. Canary デプロイ

新バージョンへのトラフィックを少量ずつ段階的に増やし、問題がないことを確認しながら全体に展開する。

```yaml
# Kubernetes - Canary デプロイ (Ingress ベース)
# stable deployment (v1)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp-stable
  labels:
    app: myapp
    version: stable
spec:
  replicas: 9
  selector:
    matchLabels:
      app: myapp
      version: stable
  template:
    metadata:
      labels:
        app: myapp
        version: stable
    spec:
      containers:
        - name: myapp
          image: myapp:v1.0.0
          ports:
            - containerPort: 3000
---
# canary deployment (v2)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp-canary
  labels:
    app: myapp
    version: canary
spec:
  replicas: 1  # 全10台中1台 = 10%
  selector:
    matchLabels:
      app: myapp
      version: canary
  template:
    metadata:
      labels:
        app: myapp
        version: canary
    spec:
      containers:
        - name: myapp
          image: myapp:v2.0.0
          ports:
            - containerPort: 3000
---
# 共通 Service (両方にルーティング)
apiVersion: v1
kind: Service
metadata:
  name: myapp-service
spec:
  selector:
    app: myapp  # version ラベルは指定しない
  ports:
    - port: 80
      targetPort: 3000
```

### 4.1 Istio を使った高度な Canary デプロイ

```yaml
# Istio VirtualService でトラフィック比率を制御
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: myapp
spec:
  hosts:
    - myapp.example.com
  http:
    - route:
        - destination:
            host: myapp-stable
            port:
              number: 3000
          weight: 90
        - destination:
            host: myapp-canary
            port:
              number: 3000
          weight: 10
```

### 4.2 AWS App Mesh を使った Canary

```yaml
# AWS App Mesh Route で Canary トラフィック分割
apiVersion: appmesh.k8s.aws/v1beta2
kind: VirtualRouter
metadata:
  name: myapp-router
spec:
  listeners:
    - portMapping:
        port: 3000
        protocol: http
  routes:
    - name: canary-route
      httpRoute:
        match:
          prefix: /
        action:
          weightedTargets:
            - virtualNodeRef:
                name: myapp-stable
              weight: 90
            - virtualNodeRef:
                name: myapp-canary
              weight: 10
```

### 4.3 Canary の段階的ロールアウトスクリプト

```bash
#!/bin/bash
# canary-rollout.sh — Canary の段階的ロールアウト
set -euo pipefail

STAGES=(5 10 25 50 75 100)  # トラフィック比率(%)
OBSERVATION_TIME=300          # 各段階の観測時間(秒)
ERROR_THRESHOLD=5             # エラー率の閾値(%)

for WEIGHT in "${STAGES[@]}"; do
  echo "=== Stage: ${WEIGHT}% canary traffic ==="

  # トラフィック比率を更新
  kubectl patch virtualservice myapp --type=json \
    -p="[{\"op\":\"replace\",\"path\":\"/spec/http/0/route/1/weight\",\"value\":${WEIGHT}},
         {\"op\":\"replace\",\"path\":\"/spec/http/0/route/0/weight\",\"value\":$((100-WEIGHT))}]"

  echo "Observing for ${OBSERVATION_TIME}s..."
  sleep "${OBSERVATION_TIME}"

  # メトリクスを取得してエラー率を確認
  ERROR_RATE=$(curl -s "http://prometheus:9090/api/v1/query?query=rate(http_requests_total{version=\"canary\",code=~\"5..\"}[5m])/rate(http_requests_total{version=\"canary\"}[5m])*100" \
    | jq -r '.data.result[0].value[1] // "0"')

  echo "Current error rate: ${ERROR_RATE}%"

  if (( $(echo "${ERROR_RATE} > ${ERROR_THRESHOLD}" | bc -l) )); then
    echo "::error::Error rate ${ERROR_RATE}% exceeds threshold ${ERROR_THRESHOLD}%. Rolling back!"
    kubectl patch virtualservice myapp --type=json \
      -p='[{"op":"replace","path":"/spec/http/0/route/1/weight","value":0},
           {"op":"replace","path":"/spec/http/0/route/0/weight","value":100}]'
    exit 1
  fi

  if [ "${WEIGHT}" -eq 100 ]; then
    echo "Canary rollout complete! Scaling down stable version."
    kubectl scale deployment myapp-stable --replicas=0
    kubectl scale deployment myapp-canary --replicas=10
  fi
done
```

---

## 5. Rolling Update

既存インスタンスを段階的に新バージョンに置き換える。Kubernetes のデフォルト戦略。

```yaml
# Kubernetes Rolling Update 設定
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  replicas: 6
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2        # 一度に追加できる Pod 数
      maxUnavailable: 1  # 同時に停止できる Pod 数
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
        - name: myapp
          image: myapp:v2.0.0
          ports:
            - containerPort: 3000
          readinessProbe:
            httpGet:
              path: /health
              port: 3000
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 3000
            initialDelaySeconds: 15
            periodSeconds: 20
```

```
Rolling Update の流れ (replicas=6, maxSurge=2, maxUnavailable=1):

ステップ1: 初期状態
  [v1] [v1] [v1] [v1] [v1] [v1]  (6台稼働)

ステップ2: 新 Pod を追加、旧 Pod を1台停止
  [v1] [v1] [v1] [v1] [v1] [--] [v2] [v2]  (5台稼働 + 2台起動中)

ステップ3: v2 が Ready になったら次の旧 Pod を置換
  [v1] [v1] [v1] [v1] [--] [v2] [v2] [v2]  (6台稼働 + 1台起動中)

ステップ4: 繰り返し
  [v1] [v1] [v1] [--] [v2] [v2] [v2] [v2]

ステップ5: 完了
  [v2] [v2] [v2] [v2] [v2] [v2]  (6台すべて v2)
```

### 5.1 Rolling Update のパラメータチューニング

```
maxSurge と maxUnavailable の設定ガイド:

高速デプロイ（リソースに余裕がある場合）:
  maxSurge: 50%
  maxUnavailable: 25%
  → 最大で全体の 150% のリソースを使用
  → 4分の1が一度に停止

安全重視デプロイ（本番環境）:
  maxSurge: 1
  maxUnavailable: 0
  → 常にレプリカ数以上の Pod が稼働
  → 1台ずつ慎重に置換

バランス（推奨）:
  maxSurge: 25%
  maxUnavailable: 25%
  → Kubernetes のデフォルト
  → 多くのケースで適切
```

---

## 6. Feature Flag によるリリース制御

### 6.1 Feature Flag の概念

```
Feature Flag の基本概念:

  デプロイ ≠ リリース

  従来:
    コードデプロイ = ユーザーへの公開
    → デプロイの失敗 = サービス障害

  Feature Flag:
    コードデプロイ → Flag OFF (ユーザーに見えない)
    検証完了 → Flag ON (段階的に公開)
    問題発生 → Flag OFF (即座に復旧)
    → デプロイとリリースが分離される
```

### 6.2 Feature Flag の実装

```typescript
// feature-flag.ts — シンプルな Feature Flag 実装
interface FeatureFlag {
  name: string;
  enabled: boolean;
  rolloutPercentage: number;  // 0-100
  allowedUsers?: string[];
  metadata?: Record<string, unknown>;
}

class FeatureFlagService {
  private flags: Map<string, FeatureFlag> = new Map();

  constructor(private readonly configSource: FlagConfigSource) {}

  async initialize(): Promise<void> {
    const config = await this.configSource.fetch();
    for (const flag of config.flags) {
      this.flags.set(flag.name, flag);
    }
  }

  isEnabled(flagName: string, userId?: string): boolean {
    const flag = this.flags.get(flagName);
    if (!flag) return false;
    if (!flag.enabled) return false;

    // 特定ユーザーに許可されている場合
    if (userId && flag.allowedUsers?.includes(userId)) {
      return true;
    }

    // パーセンテージロールアウト
    if (flag.rolloutPercentage < 100) {
      const hash = this.hashUserId(userId ?? 'anonymous');
      return (hash % 100) < flag.rolloutPercentage;
    }

    return true;
  }

  private hashUserId(userId: string): number {
    let hash = 0;
    for (let i = 0; i < userId.length; i++) {
      const char = userId.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash |= 0; // 32bit整数に変換
    }
    return Math.abs(hash);
  }
}

// 使用例
const featureFlags = new FeatureFlagService(remoteConfigSource);
await featureFlags.initialize();

if (featureFlags.isEnabled('new-checkout-flow', currentUser.id)) {
  renderNewCheckout();
} else {
  renderLegacyCheckout();
}
```

### 6.3 Feature Flag のライフサイクル管理

```typescript
// feature-flag-lifecycle.ts — Flag のライフサイクル管理
interface FlagLifecycle {
  name: string;
  createdAt: Date;
  createdBy: string;
  status: 'development' | 'testing' | 'rollout' | 'fully-rolled-out' | 'cleanup';
  expiresAt?: Date;  // 期限（これを過ぎたら削除対象）
  jiraTicket?: string;  // 関連チケット
}

class FlagLifecycleManager {
  // 期限切れの Flag を検出
  findExpiredFlags(flags: FlagLifecycle[]): FlagLifecycle[] {
    const now = new Date();
    return flags.filter(flag =>
      flag.expiresAt && flag.expiresAt < now &&
      flag.status === 'fully-rolled-out'
    );
  }

  // Flag の棚卸しレポート生成
  generateAuditReport(flags: FlagLifecycle[]): string {
    const lines = [
      '# Feature Flag 棚卸しレポート',
      `生成日: ${new Date().toISOString()}`,
      '',
      '| Flag | ステータス | 作成日 | 期限 | 担当者 |',
      '|------|----------|-------|------|--------|',
    ];

    for (const flag of flags) {
      const isExpired = flag.expiresAt && flag.expiresAt < new Date();
      lines.push(
        `| ${flag.name} | ${flag.status} ${isExpired ? '(期限切れ!)' : ''} | ${flag.createdAt.toISOString().slice(0, 10)} | ${flag.expiresAt?.toISOString().slice(0, 10) ?? '-'} | ${flag.createdBy} |`
      );
    }

    return lines.join('\n');
  }
}
```

### 6.4 Feature Flag と GitHub Actions の連携

```yaml
# Feature Flag の状態を CI で検証
name: Feature Flag Audit
on:
  schedule:
    - cron: '0 9 * * 1'  # 毎週月曜
  workflow_dispatch:

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Check for stale feature flags
        run: |
          # コード内の Feature Flag 参照を検索
          echo "## Feature Flag 使用状況" > flag-report.md

          # isEnabled('flag-name') パターンを検索
          grep -rn "isEnabled\|featureFlag\|FEATURE_" src/ --include="*.ts" --include="*.tsx" | \
            grep -oP "(?:isEnabled|featureFlag)\(['\"]([^'\"]+)" | \
            sort -u > found-flags.txt

          echo "検出された Flag:" >> flag-report.md
          cat found-flags.txt >> flag-report.md

      - name: Create issue if stale flags found
        if: steps.check.outputs.stale_count > 0
        uses: peter-evans/create-issue-from-file@v5
        with:
          title: "Feature Flag 棚卸し: 古い Flag が検出されました"
          content-filepath: flag-report.md
          labels: tech-debt,feature-flags
```

---

## 7. DB マイグレーション戦略

### 7.1 Expand and Contract パターン

```
Expand and Contract パターンの3段階:

Phase 1: Expand（拡張）
  - 新カラム/テーブルを追加
  - 旧コードはそのまま動作する
  - 新コードは新旧両方に対応

  ALTER TABLE users ADD COLUMN email_verified BOOLEAN DEFAULT false;
  -- 既存の行は false がセットされる
  -- 旧コードは email_verified を参照しないので影響なし

Phase 2: Migrate（移行）
  - 全インスタンスが新バージョンになったことを確認
  - データの移行を実行
  - 新カラムのみを参照するようコードを更新

  UPDATE users SET email_verified = true WHERE verified_at IS NOT NULL;
  -- バッチ処理でデータを移行

Phase 3: Contract（縮小）
  - 旧カラム/テーブルを削除
  - 不要なコードを削除

  ALTER TABLE users DROP COLUMN verified_at;
  -- 旧カラムを安全に削除
```

### 7.2 安全なマイグレーションのルール

```
安全なマイグレーション操作:
  ✅ カラムの追加（デフォルト値付き）
  ✅ テーブルの追加
  ✅ インデックスの追加（CONCURRENTLY）
  ✅ カラムの NULL 許容化

危険なマイグレーション操作:
  ❌ カラムの削除（旧バージョンが参照している可能性）
  ❌ カラムの型変更
  ❌ カラムのリネーム
  ❌ NOT NULL 制約の追加（既存データが違反する可能性）
  ❌ テーブルの削除

危険な操作の安全な実行方法:
  カラムの削除:
    1. コードからカラム参照を削除してデプロイ
    2. 全インスタンスが新バージョンになったことを確認
    3. カラムを削除

  カラムのリネーム:
    1. 新カラムを追加
    2. 新旧カラムの両方に書き込むコードをデプロイ
    3. データを新カラムに移行
    4. 旧カラムの参照を削除してデプロイ
    5. 旧カラムを削除
```

### 7.3 マイグレーションと GitHub Actions

```yaml
# マイグレーションの自動実行ワークフロー
name: DB Migration
on:
  push:
    branches: [main]
    paths:
      - 'migrations/**'

jobs:
  migrate:
    runs-on: ubuntu-latest
    environment: production
    permissions:
      id-token: write
      contents: read
    steps:
      - uses: actions/checkout@v4

      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: ap-northeast-1

      - name: Run migration (dry-run)
        run: |
          # まずドライランで確認
          npx prisma migrate diff \
            --from-schema-datasource prisma/schema.prisma \
            --to-migrations migrations/ \
            --shadow-database-url "$SHADOW_DB_URL"
        env:
          SHADOW_DB_URL: ${{ secrets.SHADOW_DATABASE_URL }}

      - name: Apply migration
        run: npx prisma migrate deploy
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}

      - name: Verify migration
        run: |
          # マイグレーション後のスキーマ検証
          npx prisma validate
```

---

## 8. 自動ロールバック

### 8.1 メトリクスベースの自動ロールバック

```yaml
# AWS Lambda + SAM の自動ロールバック
Resources:
  ApiFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: dist/lambda.handler
      Runtime: nodejs20.x
      AutoPublishAlias: live
      DeploymentPreference:
        Type: Canary10Percent5Minutes
        Alarms:
          - !Ref ApiErrorAlarm
          - !Ref ApiLatencyAlarm

  # エラー率のアラーム
  ApiErrorAlarm:
    Type: AWS::CloudWatch::Alarm
    Properties:
      MetricName: 5XXError
      Namespace: AWS/ApiGateway
      Statistic: Sum
      Period: 60
      EvaluationPeriods: 2
      Threshold: 10
      ComparisonOperator: GreaterThanThreshold
      TreatMissingData: notBreaching

  # レイテンシのアラーム
  ApiLatencyAlarm:
    Type: AWS::CloudWatch::Alarm
    Properties:
      MetricName: Latency
      Namespace: AWS/ApiGateway
      ExtendedStatistic: p99
      Period: 60
      EvaluationPeriods: 2
      Threshold: 3000  # 3秒
      ComparisonOperator: GreaterThanThreshold
```

### 8.2 Kubernetes の自動ロールバック

```yaml
# Argo Rollouts を使った自動ロールバック
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: myapp
spec:
  replicas: 10
  strategy:
    canary:
      steps:
        - setWeight: 10
        - pause: { duration: 5m }
        - analysis:
            templates:
              - templateName: success-rate
        - setWeight: 50
        - pause: { duration: 5m }
        - analysis:
            templates:
              - templateName: success-rate
        - setWeight: 100
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
        - name: myapp
          image: myapp:v2.0.0
---
# 分析テンプレート
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: success-rate
spec:
  metrics:
    - name: success-rate
      interval: 30s
      count: 5
      successCondition: result[0] >= 0.95
      failureLimit: 3
      provider:
        prometheus:
          address: http://prometheus:9090
          query: |
            sum(rate(http_requests_total{app="myapp",version="canary",code!~"5.."}[5m]))
            /
            sum(rate(http_requests_total{app="myapp",version="canary"}[5m]))
```

---

## 9. デプロイ戦略比較表

| 特性 | Recreate | Rolling | Blue-Green | Canary |
|------|----------|---------|------------|--------|
| ダウンタイム | あり | なし | なし | なし |
| ロールバック速度 | 遅い（再デプロイ） | 中（段階的） | 即座（切替） | 即座（トラフィック変更） |
| リソースコスト | 低い | 中 | 高い（2倍） | 中〜高 |
| リスク | 高い | 中 | 低い | 最も低い |
| 複雑さ | 最低 | 低い | 中 | 高い |
| DB マイグレーション | 容易 | 注意が必要 | 注意が必要 | 注意が必要 |
| 適用規模 | 小規模 | 中〜大規模 | 中〜大規模 | 大規模 |
| テスト容易性 | 低い | 中 | 高い（スタンバイで確認可） | 高い（少量トラフィックで確認） |
| 自動化難易度 | 低い | 低い | 中 | 高い |

| Feature Flag 比較 | 自前実装 | LaunchDarkly | Unleash (OSS) | AWS AppConfig |
|-------------------|---------|-------------|---------------|---------------|
| 導入コスト | 低い | 高い | 中 | 中 |
| 運用負荷 | 高い | 低い | 中 | 低い |
| リアルタイム更新 | 要実装 | 対応 | 対応 | 対応 |
| ターゲティング | 要実装 | 高機能 | 中機能 | 基本的 |
| 監査ログ | 要実装 | 対応 | 対応 | 対応 |
| セルフホスト | 可能 | 不可 | 可能 | 不可 |

---

## 10. アンチパターン

### アンチパターン 1: ビッグバンデプロイ

```
[悪い例] 全変更を一度にデプロイ

- 3ヶ月分の変更を一括リリース
- テスト環境と本番環境の差異が大きい
- 問題発生時にどの変更が原因か特定困難
- ロールバックすると全機能が巻き戻る

[良い例] 小さく頻繁にデプロイ

- 1機能ずつ Feature Flag で保護してデプロイ
- 段階的にロールアウト（1% → 10% → 50% → 100%）
- 問題の切り分けが容易
- 該当 Flag だけ OFF にすれば即復旧
```

### アンチパターン 2: ロールバック手順の未整備

```
[悪い例]
- デプロイ手順書はあるがロールバック手順がない
- 障害発生後に慌ててロールバック方法を調査
- DBマイグレーションの巻き戻しが不可能
- 夜間リリースで担当者が不在

[良い例]
- デプロイごとにロールバック手順を文書化
- 自動ロールバックの閾値を設定（エラー率 > 5% で自動復旧）
- DBマイグレーションは前方互換を維持（カラム追加 → コード変更 → カラム削除）
- ロールバック訓練を定期的に実施
```

### アンチパターン 3: Feature Flag の放置

```
[悪い例]
- 100% ロールアウト済みの Flag がコードに残り続ける
- 古い Flag の分岐が複雑に絡み合い保守困難
- Flag の ON/OFF 状態を把握している人がいない

[良い例]
- Flag のライフサイクルを管理（作成日、期限）
- 100% 展開後は技術的負債チケットを起票し、Flag を削除
- Flag の棚卸しを月次で実施
```

### アンチパターン 4: ヘルスチェックの不備

```
[悪い例]
- ヘルスチェックエンドポイントが常に 200 を返す
- DB 接続が切れてもヘルスチェックが成功する
- ヘルスチェックの間隔が長すぎて異常検知が遅い

[良い例]
- ヘルスチェックで実際の依存関係（DB、外部API）を検証
- Readiness Probe と Liveness Probe を適切に分離
- 浅いヘルスチェック（/health）と深いヘルスチェック（/health/deep）を用意
```

```typescript
// health-check.ts — 適切なヘルスチェック実装
app.get('/health', (req, res) => {
  // 浅いヘルスチェック: アプリケーションが起動しているか
  res.json({ status: 'ok', uptime: process.uptime() });
});

app.get('/health/deep', async (req, res) => {
  // 深いヘルスチェック: 依存関係が正常か
  const checks = {
    database: await checkDatabase(),
    redis: await checkRedis(),
    externalApi: await checkExternalApi(),
  };

  const allHealthy = Object.values(checks).every(c => c.status === 'ok');

  res.status(allHealthy ? 200 : 503).json({
    status: allHealthy ? 'ok' : 'degraded',
    checks,
    timestamp: new Date().toISOString(),
  });
});
```

---

## 11. FAQ

### Q1: Blue-Green と Canary はどちらを選ぶべきですか？

トラフィック量とリスク許容度で判断します。トラフィックが少ない場合や、迅速な全体切替が必要な場合は Blue-Green が適しています。大規模サービスで段階的にリスクを抑えたい場合は Canary が有効です。また、Blue-Green はインフラコストが2倍になるため、コスト制約も考慮してください。

### Q2: DB マイグレーションを伴うデプロイで注意すべきことは？

**Expand and Contract パターン**を使用します。(1) 新カラム/テーブルを追加（Expand）、(2) 新旧コードが共存できる状態にする、(3) 全インスタンスが新バージョンに切り替わった後に旧カラムを削除（Contract）。破壊的変更（カラム削除、型変更）を直接行うと、ローリング更新中に旧バージョンのインスタンスが参照エラーを起こします。

### Q3: Feature Flag の粒度はどのくらいが適切ですか？

ユーザーから見える**機能単位**で切るのが基本です。1つの Flag が複数の独立した機能を制御すると、片方だけ無効化できず運用が困難になります。一方、細かすぎる Flag（例: ボタンの色変更ごと）は管理コストが増大します。目安として「この Flag を OFF にしたとき、ユーザー体験として一貫しているか」を判断基準にしてください。

### Q4: デプロイ頻度はどのくらいが理想ですか？

DORA メトリクスでは「Elite」パフォーマーは1日に複数回デプロイしています。ただし、頻度そのものが目標ではなく、「小さな変更を安全に、自信を持ってデプロイできる」状態を目指すべきです。Feature Flag を活用すれば、コードのデプロイ頻度を上げつつ、リリース（ユーザーへの公開）は慎重に制御できます。

### Q5: Rolling Update で新旧バージョンが混在する期間のリスクは？

API のバージョニング（v1/v2）や、前方互換性のあるスキーマ設計で対処します。具体的には、(1) 新しいフィールドは追加のみ（削除しない）、(2) クライアントは未知のフィールドを無視する、(3) DB マイグレーションは Expand and Contract パターンに従う、というルールを徹底します。

### Q6: 自動ロールバックの閾値はどう設定すべきですか？

サービスの SLA とベースラインメトリクスに基づいて設定します。一般的な目安は、(1) エラー率: ベースラインの2倍以上（例: 通常0.1%なら閾値0.2%）、(2) レイテンシ: P99がベースラインの1.5倍以上、(3) リクエスト成功率: 99%を下回った場合。閾値が厳しすぎると誤検知でロールバックが頻発するため、段階的に調整してください。

---

## 12. デプロイ運用のベストプラクティス

### 12.1 デプロイチェックリスト

```
デプロイ前チェック:
  [  ] 全テスト（unit / integration / e2e）がパスしている
  [  ] コードレビューが承認されている
  [  ] DB マイグレーションが Expand and Contract に従っている
  [  ] 環境変数・シークレットの追加/変更がドキュメント化されている
  [  ] ロールバック手順が明確になっている
  [  ] Feature Flag の状態が確認済みである
  [  ] 依存サービスへの影響が評価されている

デプロイ中チェック:
  [  ] ヘルスチェックが正常を返している
  [  ] エラー率がベースラインを超えていない
  [  ] レイテンシが許容範囲内である
  [  ] ログに異常なエラーパターンが出ていない

デプロイ後チェック:
  [  ] Smoke テストがパスしている
  [  ] 主要なユーザーフロー（ログイン、購入等）が正常動作する
  [  ] メトリクスダッシュボードで異常がない
  [  ] 前バージョンの Feature Flag がクリーンアップ対象としてマークされている
```

### 12.2 デプロイ頻度と組織パフォーマンス

```
DORA メトリクスによるパフォーマンス分類:

指標                    | Elite        | High         | Medium       | Low
デプロイ頻度            | 日に複数回   | 週〜月1回    | 月〜半年1回  | 半年以上
変更リードタイム        | 1時間未満    | 1日〜1週間   | 1週間〜1ヶ月 | 1ヶ月以上
変更失敗率              | 0-15%        | 16-30%       | 16-30%       | 46-60%
サービス復旧時間        | 1時間未満    | 1日未満      | 1日〜1週間   | 6ヶ月以上

改善のアプローチ:
1. テスト自動化 → 変更失敗率の低減
2. CI/CD パイプライン最適化 → リードタイム短縮
3. Feature Flag + Canary → デプロイ頻度向上
4. 自動ロールバック + 監視 → 復旧時間短縮
```

### 12.3 デプロイ通知テンプレート

```yaml
# .github/workflows/deploy-notification.yml
name: Deploy Notification

on:
  workflow_call:
    inputs:
      environment:
        required: true
        type: string
      version:
        required: true
        type: string
      status:
        required: true
        type: string

jobs:
  notify:
    runs-on: ubuntu-latest
    steps:
      - name: Post Slack Notification
        uses: slackapi/slack-github-action@v2.0.0
        with:
          webhook: ${{ secrets.SLACK_DEPLOY_WEBHOOK }}
          webhook-type: incoming-webhook
          payload: |
            {
              "blocks": [
                {
                  "type": "header",
                  "text": {
                    "type": "plain_text",
                    "text": "Deploy ${{ inputs.status == 'success' && 'Completed' || 'Failed' }}"
                  }
                },
                {
                  "type": "section",
                  "fields": [
                    { "type": "mrkdwn", "text": "*Environment:*\n`${{ inputs.environment }}`" },
                    { "type": "mrkdwn", "text": "*Version:*\n`${{ inputs.version }}`" },
                    { "type": "mrkdwn", "text": "*Actor:*\n${{ github.actor }}" },
                    { "type": "mrkdwn", "text": "*Status:*\n${{ inputs.status == 'success' && ':white_check_mark: Success' || ':x: Failed' }}" }
                  ]
                },
                {
                  "type": "actions",
                  "elements": [
                    {
                      "type": "button",
                      "text": { "type": "plain_text", "text": "View Run" },
                      "url": "${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}"
                    }
                  ]
                }
              ]
            }
```

---

## まとめ

| 項目 | 要点 |
|------|------|
| Recreate | 最もシンプルだがダウンタイムあり。開発環境やバッチ向き |
| Rolling Update | K8s デフォルト。段階的置換でゼロダウンタイム |
| Blue-Green | 2環境を用意し即座に切替。ロールバックが最速 |
| Canary | 少量トラフィックで検証。リスク最小化 |
| Feature Flag | デプロイとリリースを分離。段階的ロールアウトに必須 |
| DB マイグレーション | Expand and Contract パターンで前方互換を維持 |
| ロールバック | 手順を事前に整備し、自動ロールバック閾値を設定 |
| ヘルスチェック | 浅い/深いの2段階で依存関係まで検証 |
| デプロイ頻度 | 小さく頻繁にデプロイし、Feature Flag でリリースを制御 |
| DORA メトリクス | デプロイ頻度・リードタイム・失敗率・復旧時間を継続計測 |

---

## 次に読むべきガイド

- [01-cloud-deployment.md](./01-cloud-deployment.md) — AWS/Vercel/Cloudflare Workers へのクラウドデプロイ実践
- [02-container-deployment.md](./02-container-deployment.md) — ECS/Kubernetes でのコンテナデプロイ
- [03-release-management.md](./03-release-management.md) — セマンティックバージョニングとリリース管理

---

## 参考文献

1. **Accelerate** — Nicole Forsgren, Jez Humble, Gene Kim (2018) — デプロイ頻度とリードタイムの科学的分析
2. **Continuous Delivery** — Jez Humble, David Farley (2010) — デプロイパイプラインの原典
3. **Kubernetes Documentation - Deployments** — https://kubernetes.io/docs/concepts/workloads/controllers/deployment/ — Rolling Update の公式リファレンス
4. **Martin Fowler - Feature Toggles** — https://martinfowler.com/articles/feature-toggles.html — Feature Flag のパターン分類
5. **Argo Rollouts** — https://argoproj.github.io/rollouts/ — Kubernetes の高度なデプロイ戦略
6. **DORA Metrics** — https://dora.dev/research/ — デプロイ頻度と組織パフォーマンスの研究
