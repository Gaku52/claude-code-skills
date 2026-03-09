# デプロイ戦略

> Blue-Green、Canary、Rolling、Feature Flag など主要なデプロイ戦略を体系的に理解し、安全で高速なリリースを実現する

## この章で学ぶこと

1. **主要デプロイ戦略の仕組みと使い分け** — Blue-Green、Canary、Rolling Update、Recreate の動作原理と適用条件
2. **Feature Flag によるリリース制御** — コードデプロイとリリースを分離し、段階的ロールアウトを実現する手法
3. **ロールバック設計とリスク軽減** — 障害発生時に即座に復旧するための仕組みと運用プロセス
4. **DB マイグレーション戦略** — デプロイと連携したデータベーススキーマ変更の安全な実行方法
5. **自動化されたデプロイパイプラインの設計** — GitHub Actions と連携した CI/CD パイプラインの構築


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解

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
            grep -oP "(?:isEnabled|featureFlag)\('\"" | \
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

## 実践演習

### 演習1: 基本的な実装

以下の要件を満たすコードを実装してください。

**要件:**
- 入力データの検証を行うこと
- エラーハンドリングを適切に実装すること
- テストコードも作成すること

```python
# 演習1: 基本実装のテンプレート
class Exercise1:
    """基本的な実装パターンの演習"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """入力値の検証"""
        if value is None:
            raise ValueError("入力値がNoneです")
        return True

    def process(self, value):
        """データ処理のメインロジック"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """処理結果の取得"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# テスト
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "例外が発生するべき"
    except ValueError:
        pass

    print("全テスト合格!")

test_exercise1()
```

### 演習2: 応用パターン

基本実装を拡張して、以下の機能を追加してください。

```python
# 演習2: 応用パターン
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """応用パターンの演習"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """アイテムの追加（サイズ制限付き）"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """キーによる検索"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """キーによる削除"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """統計情報"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# テスト
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # サイズ制限
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("応用テスト全合格!")

test_advanced()
```

### 演習3: パフォーマンス最適化

以下のコードのパフォーマンスを改善してください。

```python
# 演習3: パフォーマンス最適化
import time
from functools import lru_cache

# 最適化前（O(n^2)）
def slow_search(data: list, target: int) -> int:
    """非効率な検索"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# 最適化後（O(n)）
def fast_search(data: list, target: int) -> tuple:
    """ハッシュマップを使った効率的な検索"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# ベンチマーク
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"非効率版: {slow_time:.4f}秒")
    print(f"効率版:   {fast_time:.6f}秒")
    print(f"高速化率: {slow_time/fast_time:.0f}倍")

benchmark()
```

**ポイント:**
- アルゴリズムの計算量を意識する
- 適切なデータ構造を選択する
- ベンチマークで効果を測定する

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


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

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
