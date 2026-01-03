# 統計的手法と実験設計の標準化

## 目次
1. [統計的厳密性の要件](#統計的厳密性の要件)
2. [サンプルサイズの決定](#サンプルサイズの決定)
3. [統計検定の選択](#統計検定の選択)
4. [効果量の計算](#効果量の計算)
5. [結果の報告形式](#結果の報告形式)
6. [再現性の確保](#再現性の確保)

---

## 統計的厳密性の要件

### MIT修士論文レベルの基準

**必須要件**:
1. ✅ サンプルサイズ n ≥ 30
2. ✅ 95% 信頼区間の報告
3. ✅ p値 < 0.05 (できれば p < 0.001)
4. ✅ 効果量 (Cohen's d または R²)
5. ✅ 多重比較の補正 (必要に応じて)

### なぜこれらが重要か?

**サンプルサイズ n ≥ 30**:
- 中心極限定理により、正規分布に近似可能
- t分布が正規分布に収束
- 信頼性の高い統計的推論

**95% 信頼区間**:
- 真の値が含まれる範囲を示す
- 点推定値だけでなく、不確実性も示す

**p値 < 0.001**:
- p < 0.05: 統計的に有意
- p < 0.001: 非常に強い有意性 (MIT修士論文レベル)

**効果量**:
- p値だけでは実用的重要性が不明
- Cohen's d: 標準化された差の大きさ
- R²: モデルの説明力

---

## サンプルサイズの決定

### 1. 事前計算 (A Priori Power Analysis)

**目的**: 必要なサンプルサイズを事前に決定

**手順**:

**ステップ1: 効果量の設定**
```
小さい効果: d = 0.2
中程度の効果: d = 0.5
大きい効果: d = 0.8
```

**ステップ2: 検出力 (Power) の設定**
```
標準: 1 - β = 0.80 (80%)
推奨: 1 - β = 0.90 (90%)
```

**ステップ3: 有意水準の設定**
```
α = 0.05 (標準)
α = 0.001 (厳密)
```

**ステップ4: サンプルサイズの計算**

**対応のある t検定の場合**:
```
n = (Z_α/2 + Z_β)² × 2σ² / d²

例: d = 0.5, α = 0.001, β = 0.10
Z_0.0005 = 3.29 (両側検定)
Z_0.10 = 1.28

n = (3.29 + 1.28)² × 2 × 1² / 0.5²
  = 20.9² × 2 / 0.25
  = 437 / 0.25
  = 166.6 ≈ 167
```

**実用的な最小値**:
```
n = 30 (中心極限定理の適用)
```

### 2. TypeScript実装

```typescript
/**
 * サンプルサイズ計算 (対応のある t検定)
 */
function calculateSampleSize(
  effectSize: number,      // Cohen's d
  alpha: number = 0.001,   // 有意水準
  power: number = 0.90     // 検出力
): number {
  // Z値の近似計算 (正規分布)
  const zAlpha = normalInv(1 - alpha / 2)  // 両側検定
  const zBeta = normalInv(power)

  const n = Math.pow(zAlpha + zBeta, 2) * 2 / Math.pow(effectSize, 2)
  return Math.ceil(n)
}

/**
 * 正規分布の逆関数 (近似)
 */
function normalInv(p: number): number {
  // Beasley-Springer-Moro algorithm
  const a = [
    2.50662823884,
    -18.61500062529,
    41.39119773534,
    -25.44106049637
  ]
  const b = [
    -8.47351093090,
    23.08336743743,
    -21.06224101826,
    3.13082909833
  ]
  const c = [
    0.3374754822726147,
    0.9761690190917186,
    0.1607979714918209,
    0.0276438810333863,
    0.0038405729373609,
    0.0003951896511919,
    0.0000321767881768,
    0.0000002888167364,
    0.0000003960315187
  ]

  if (p < 0.02425) {
    const q = Math.sqrt(-2 * Math.log(p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((q + c[6]) * q + c[7]) * q + c[8]) * q + 1)
  } else if (p < 0.97575) {
    const q = p - 0.5
    const r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r) * q) /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + 1))
  } else {
    const q = Math.sqrt(-2 * Math.log(1 - p))
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
           ((((q + c[6]) * q + c[7]) * q + c[8]) * q + 1)
  }
}

// 使用例
const n1 = calculateSampleSize(0.5, 0.001, 0.90)  // n = 167
const n2 = calculateSampleSize(0.8, 0.001, 0.90)  // n = 66
const n3 = calculateSampleSize(1.0, 0.001, 0.90)  // n = 42
console.log(`効果量 d=0.5: n=${n1}`)
console.log(`効果量 d=0.8: n=${n2}`)
console.log(`効果量 d=1.0: n=${n3}`)
```

---

## 統計検定の選択

### フローチャート

```
データの種類は?
│
├─ 連続データ (時間、サイズなど)
│  │
│  ├─ 2群の比較
│  │  │
│  │  ├─ 対応あり (同じサンプルのBefore/After)
│  │  │  → 対応のある t検定
│  │  │
│  │  └─ 対応なし (異なるサンプル)
│  │     → 独立した t検定
│  │
│  └─ 3群以上の比較
│     │
│     ├─ 対応あり
│     │  → 反復測定ANOVA
│     │
│     └─ 対応なし
│        → 一元配置ANOVA
│
└─ カテゴリーデータ (成功/失敗など)
   │
   ├─ 2×2 分割表
   │  → カイ二乗検定 または Fisher正確確率検定
   │
   └─ それ以上
      → カイ二乗検定
```

### 1. 対応のある t検定

**用途**: Before/After の比較

**前提条件**:
- ペアの差が正規分布 (n ≥ 30 なら緩和)
- 対応がある (同じサンプル)

**TypeScript実装**:
```typescript
interface TTestResult {
  t: number         // t統計量
  df: number        // 自由度
  p: number         // p値 (両側検定)
  ci: [number, number]  // 95%信頼区間
  meanDiff: number  // 平均差
  sd: number        // 差の標準偏差
  d: number         // Cohen's d
}

function pairedTTest(before: number[], after: number[]): TTestResult {
  if (before.length !== after.length) {
    throw new Error("Arrays must have the same length")
  }

  const n = before.length
  const diff = before.map((b, i) => b - after[i])

  // 平均差
  const meanDiff = diff.reduce((a, b) => a + b, 0) / n

  // 標準偏差
  const variance = diff.reduce((sum, d) => sum + Math.pow(d - meanDiff, 2), 0) / (n - 1)
  const sd = Math.sqrt(variance)

  // t統計量
  const t = meanDiff / (sd / Math.sqrt(n))

  // 自由度
  const df = n - 1

  // p値 (両側検定) - t分布の累積分布関数を使用
  const p = 2 * (1 - tCDF(Math.abs(t), df))

  // 95%信頼区間
  const tCrit = tInv(0.025, df)  // 両側検定
  const margin = tCrit * (sd / Math.sqrt(n))
  const ci: [number, number] = [meanDiff - margin, meanDiff + margin]

  // Cohen's d (効果量)
  const d = meanDiff / sd

  return { t, df, p, ci, meanDiff, sd, d }
}

// 使用例
const before = [12.5, 13.2, 11.8, 14.1, 12.9, ...]  // n=30
const after = [10.2, 11.5, 9.8, 12.3, 10.7, ...]

const result = pairedTTest(before, after)
console.log(`t(${result.df}) = ${result.t.toFixed(2)}, p = ${result.p.toFixed(4)}`)
console.log(`Mean diff = ${result.meanDiff.toFixed(2)} (95% CI [${result.ci[0].toFixed(2)}, ${result.ci[1].toFixed(2)}])`)
console.log(`Cohen's d = ${result.d.toFixed(2)}`)
```

### 2. 独立した t検定

**用途**: 2つの異なるグループの比較

**TypeScript実装**:
```typescript
function independentTTest(group1: number[], group2: number[]): TTestResult {
  const n1 = group1.length
  const n2 = group2.length

  const mean1 = group1.reduce((a, b) => a + b, 0) / n1
  const mean2 = group2.reduce((a, b) => a + b, 0) / n2
  const meanDiff = mean1 - mean2

  const var1 = group1.reduce((sum, x) => sum + Math.pow(x - mean1, 2), 0) / (n1 - 1)
  const var2 = group2.reduce((sum, x) => sum + Math.pow(x - mean2, 2), 0) / (n2 - 1)

  // プールされた分散 (Welch's t-test の場合は異なる)
  const pooledVar = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
  const pooledSd = Math.sqrt(pooledVar)

  // t統計量
  const t = meanDiff / (pooledSd * Math.sqrt(1 / n1 + 1 / n2))

  // 自由度
  const df = n1 + n2 - 2

  // p値
  const p = 2 * (1 - tCDF(Math.abs(t), df))

  // 95%信頼区間
  const tCrit = tInv(0.025, df)
  const margin = tCrit * pooledSd * Math.sqrt(1 / n1 + 1 / n2)
  const ci: [number, number] = [meanDiff - margin, meanDiff + margin]

  // Cohen's d
  const d = meanDiff / pooledSd

  return { t, df, p, ci, meanDiff, sd: pooledSd, d }
}
```

---

## 効果量の計算

### 1. Cohen's d (標準化された平均差)

**定義**:
```
d = (M₁ - M₂) / SD_pooled
```

**解釈**:
| Cohen's d | 効果の大きさ |
|-----------|------------|
| 0.2 | 小さい |
| 0.5 | 中程度 |
| 0.8 | 大きい |
| 1.2+ | 非常に大きい |

**TypeScript実装**:
```typescript
function cohensD(group1: number[], group2: number[]): number {
  const n1 = group1.length
  const n2 = group2.length

  const mean1 = group1.reduce((a, b) => a + b, 0) / n1
  const mean2 = group2.reduce((a, b) => a + b, 0) / n2

  const var1 = group1.reduce((sum, x) => sum + Math.pow(x - mean1, 2), 0) / (n1 - 1)
  const var2 = group2.reduce((sum, x) => sum + Math.pow(x - mean2, 2), 0) / (n2 - 1)

  const pooledSd = Math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

  return (mean1 - mean2) / pooledSd
}
```

### 2. R² (決定係数)

**用途**: 線形回帰、ログ-ログプロット

**定義**:
```
R² = 1 - (SS_residual / SS_total)
```

**解釈**:
- R² = 0.9: モデルが90%の変動を説明
- R² > 0.95: 非常に良い適合
- R² > 0.99: ほぼ完全な適合 (理論計算量の検証)

**TypeScript実装**:
```typescript
interface RegressionResult {
  slope: number        // 傾き
  intercept: number    // 切片
  r2: number          // 決定係数
  residuals: number[] // 残差
}

function linearRegression(x: number[], y: number[]): RegressionResult {
  const n = x.length

  const meanX = x.reduce((a, b) => a + b, 0) / n
  const meanY = y.reduce((a, b) => a + b, 0) / n

  const ssXY = x.reduce((sum, xi, i) => sum + (xi - meanX) * (y[i] - meanY), 0)
  const ssXX = x.reduce((sum, xi) => sum + Math.pow(xi - meanX, 2), 0)

  const slope = ssXY / ssXX
  const intercept = meanY - slope * meanX

  // 予測値と残差
  const predicted = x.map(xi => slope * xi + intercept)
  const residuals = y.map((yi, i) => yi - predicted[i])

  // R²
  const ssRes = residuals.reduce((sum, r) => sum + r * r, 0)
  const ssTot = y.reduce((sum, yi) => sum + Math.pow(yi - meanY, 2), 0)
  const r2 = 1 - (ssRes / ssTot)

  return { slope, intercept, r2, residuals }
}

// ログ-ログプロット用
function logLogRegression(n: number[], time: number[]): RegressionResult {
  const logN = n.map(x => Math.log(x))
  const logTime = time.map(t => Math.log(t))
  return linearRegression(logN, logTime)
}

// 使用例: 理論計算量の検証
const n = [100, 200, 500, 1000, 2000, 5000, 10000]
const time = [2.1, 4.5, 12.3, 26.8, 55.2, 142.1, 298.5]  // O(n log n)の例

const result = logLogRegression(n, time)
console.log(`傾き = ${result.slope.toFixed(2)} (理論値: 1.0 for O(n log n))`)
console.log(`R² = ${result.r2.toFixed(4)}`)  // 0.9999が期待される
```

---

## 結果の報告形式

### 1. テーブル形式

**標準フォーマット**:
```markdown
| メトリクス | Before | After | 改善率 | t値 | p値 | 効果量 (d) |
|---------|--------|-------|--------|-----|-----|-----------|
| 処理時間 | 12.5ms (±2.1) | 4.8ms (±0.9) | -61.6% | t(29)=25.3 | <0.001 | d=4.8 |
| メモリ | 185MB (±12) | 92MB (±8) | -50.3% | t(29)=42.1 | <0.001 | d=9.2 |
```

**含めるべき情報**:
- ✅ 平均値 ± 標準偏差
- ✅ 改善率 (%)
- ✅ t値と自由度: t(df)
- ✅ p値 (<0.001 が理想)
- ✅ 効果量 (Cohen's d)

### 2. 信頼区間の報告

**フォーマット**:
```markdown
**測定結果 (n=30)**:
- 処理時間: 4.8ms (SD=0.9ms, 95% CI [4.5, 5.1])
- メモリ使用量: 92MB (SD=8MB, 95% CI [89, 95])
```

### 3. ログ-ログプロット結果

**フォーマット**:
```markdown
**計算量の検証** (ログ-ログプロット):
- 理論計算量: O(n log n)
- 実測傾き: 1.02 (理論値: 1.0)
- R² = 0.9998 (理論との完全な一致)
```

---

## 再現性の確保

### 1. 実験環境の記録

**必須情報**:
```markdown
**測定環境**:
- CPU: Apple M1 Pro (8コア)
- メモリ: 16GB
- OS: macOS 14.2
- Node.js: v20.10.0
- TypeScript: 5.3.3
```

### 2. 乱数シードの固定

**TypeScript実装**:
```typescript
class SeededRandom {
  private seed: number

  constructor(seed: number) {
    this.seed = seed
  }

  next(): number {
    this.seed = (this.seed * 9301 + 49297) % 233280
    return this.seed / 233280
  }

  nextInt(min: number, max: number): number {
    return Math.floor(this.next() * (max - min + 1)) + min
  }
}

// 使用例
const rng = new SeededRandom(42)  // 固定シード
const randomData = Array.from({ length: 30 }, () => rng.nextInt(1, 100))
```

### 3. データの保存

**CSV形式で保存**:
```typescript
function saveResults(results: Array<{ n: number, time: number }>, filename: string) {
  const csv = [
    "n,time",
    ...results.map(r => `${r.n},${r.time}`)
  ].join("\n")

  fs.writeFileSync(filename, csv)
}
```

---

## 完全な実験テンプレート

次のファイル (`02-experiment-template.ts`) で、実行可能なテンプレートを提供します。

**このファイルの内容**:
- ✅ サンプルサイズ計算
- ✅ t検定 (対応あり・なし)
- ✅ 信頼区間計算
- ✅ Cohen's d 計算
- ✅ ログ-ログ回帰
- ✅ 結果のフォーマット

---

## まとめ

### MIT修士論文レベルの統計基準

| 要件 | 基準 | 理由 |
|------|------|------|
| サンプルサイズ | n ≥ 30 | 中心極限定理 |
| 信頼区間 | 95% CI | 不確実性の明示 |
| 有意水準 | p < 0.001 | 強い有意性 |
| 効果量 | Cohen's d, R² | 実用的重要性 |
| 再現性 | 環境・シード記録 | 追試可能 |

### 次のステップ

- ✅ このファイル: 統計手法の理論
- 次: 02-experiment-template.ts (実行可能なテンプレート)
- 最後: 03-reporting-template.md (報告書のテンプレート)

---

**統計的厳密性は MIT修士論文の基盤** ✓
