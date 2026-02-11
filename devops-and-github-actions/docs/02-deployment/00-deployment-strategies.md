# デプロイ戦略

> Blue-Green、Canary、Rolling、Feature Flag など主要なデプロイ戦略を体系的に理解し、安全で高速なリリースを実現する

## この章で学ぶこと

1. **主要デプロイ戦略の仕組みと使い分け** — Blue-Green、Canary、Rolling Update、Recreate の動作原理と適用条件
2. **Feature Flag によるリリース制御** — コードデプロイとリリースを分離し、段階的ロールアウトを実現する手法
3. **ロールバック設計とリスク軽減** — 障害発生時に即座に復旧するための仕組みと運用プロセス

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

---

## 2. Blue-Green デプロイ

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

---

## 3. Canary デプロイ

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

---

## 4. Rolling Update

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

---

## 5. Feature Flag によるリリース制御

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

---

## 6. デプロイ戦略比較表

| 特性 | Recreate | Rolling | Blue-Green | Canary |
|------|----------|---------|------------|--------|
| ダウンタイム | あり | なし | なし | なし |
| ロールバック速度 | 遅い（再デプロイ） | 中（段階的） | 即座（切替） | 即座（トラフィック変更） |
| リソースコスト | 低い | 中 | 高い（2倍） | 中〜高 |
| リスク | 高い | 中 | 低い | 最も低い |
| 複雑さ | 最低 | 低い | 中 | 高い |
| DB マイグレーション | 容易 | 注意が必要 | 注意が必要 | 注意が必要 |
| 適用規模 | 小規模 | 中〜大規模 | 中〜大規模 | 大規模 |

| Feature Flag 比較 | 自前実装 | LaunchDarkly | Unleash (OSS) | AWS AppConfig |
|-------------------|---------|-------------|---------------|---------------|
| 導入コスト | 低い | 高い | 中 | 中 |
| 運用負荷 | 高い | 低い | 中 | 低い |
| リアルタイム更新 | 要実装 | 対応 | 対応 | 対応 |
| ターゲティング | 要実装 | 高機能 | 中機能 | 基本的 |
| 監査ログ | 要実装 | 対応 | 対応 | 対応 |
| セルフホスト | 可能 | 不可 | 可能 | 不可 |

---

## 7. アンチパターン

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

---

## 8. FAQ

### Q1: Blue-Green と Canary はどちらを選ぶべきですか？

トラフィック量とリスク許容度で判断します。トラフィックが少ない場合や、迅速な全体切替が必要な場合は Blue-Green が適しています。大規模サービスで段階的にリスクを抑えたい場合は Canary が有効です。また、Blue-Green はインフラコストが2倍になるため、コスト制約も考慮してください。

### Q2: DB マイグレーションを伴うデプロイで注意すべきことは？

**Expand and Contract パターン**を使用します。(1) 新カラム/テーブルを追加（Expand）、(2) 新旧コードが共存できる状態にする、(3) 全インスタンスが新バージョンに切り替わった後に旧カラムを削除（Contract）。破壊的変更（カラム削除、型変更）を直接行うと、ローリング更新中に旧バージョンのインスタンスが参照エラーを起こします。

### Q3: Feature Flag の粒度はどのくらいが適切ですか？

ユーザーから見える**機能単位**で切るのが基本です。1つの Flag が複数の独立した機能を制御すると、片方だけ無効化できず運用が困難になります。一方、細かすぎる Flag（例: ボタンの色変更ごと）は管理コストが増大します。目安として「この Flag を OFF にしたとき、ユーザー体験として一貫しているか」を判断基準にしてください。

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
