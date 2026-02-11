# ウェアラブル — Apple Watch / Galaxy Watch、健康モニタリング、AIフィットネス

> スマートウォッチを中心としたウェアラブルデバイスのAI活用を解説する。センサーデータの取得から健康モニタリング、AIフィットネスコーチング、そして開発手法まで網羅する。

---

## この章で学ぶこと

1. **ウェアラブルセンサー技術** — 光学心拍、加速度、血中酸素などセンサーの仕組みと精度
2. **健康モニタリングAI** — 不整脈検出、睡眠分析、ストレス推定のアルゴリズム
3. **AIフィットネスの開発** — HealthKit/Health Connect APIを使ったアプリ開発

---

## 1. ウェアラブルセンサーの構成

```
┌─────────────────────────────────────────────────┐
│          スマートウォッチ センサー構成              │
│                                                   │
│  ┌──────────────┐  ┌──────────────┐              │
│  │ PPG (光学式)  │  │ ECG (心電図)  │              │
│  │ 心拍数        │  │ 不整脈検出    │              │
│  │ SpO2 (血中O2) │  │ 心房細動      │              │
│  └──────────────┘  └──────────────┘              │
│                                                   │
│  ┌──────────────┐  ┌──────────────┐              │
│  │ 加速度計      │  │ ジャイロスコープ│              │
│  │ 歩数/転倒検出 │  │ 姿勢推定      │              │
│  │ 睡眠段階      │  │ ワークアウト   │              │
│  └──────────────┘  └──────────────┘              │
│                                                   │
│  ┌──────────────┐  ┌──────────────┐              │
│  │ 気圧計        │  │ 温度センサー  │              │
│  │ 高度/階段検出 │  │ 皮膚温度      │              │
│  └──────────────┘  └──────────────┘              │
│                                                   │
│  ┌──────────────┐  ┌──────────────┐              │
│  │ GPS          │  │ NPU (一部機種)│              │
│  │ 位置/ルート   │  │ オンデバイスAI │              │
│  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────┘
```

### 1.1 健康データの処理フロー

```
┌─────────────────────────────────────────────────┐
│          健康モニタリング AIパイプライン           │
│                                                   │
│  センサー群                                       │
│  (PPG/加速度/温度)                                │
│      │                                            │
│      ▼                                            │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  │ ノイズ除去 │──▶│ 特徴量   │──▶│ オンデバイス│    │
│  │ フィルタ  │   │ 抽出     │   │ ML推論    │     │
│  │ (ローパス)│   │ (HRV等)  │   │ (Core ML) │     │
│  └──────────┘   └──────────┘   └──────────┘     │
│                                       │           │
│                                       ▼           │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  │ アラート  │◀──│ トレンド  │◀──│ HealthKit │     │
│  │ 通知     │   │ 分析     │   │ / Health  │     │
│  │ (異常時) │   │ (長期)   │   │ Connect   │     │
│  └──────────┘   └──────────┘   └──────────┘     │
└─────────────────────────────────────────────────┘
```

---

## 2. コード例

### コード例 1: Apple HealthKit で心拍データを取得

```swift
import HealthKit

class HeartRateMonitor {
    let healthStore = HKHealthStore()

    func requestAuthorization() {
        let heartRateType = HKQuantityType.quantityType(
            forIdentifier: .heartRate)!
        let typesToRead: Set<HKSampleType> = [heartRateType]

        healthStore.requestAuthorization(
            toShare: nil, read: typesToRead
        ) { success, error in
            if success {
                print("HealthKit認証成功")
                self.fetchRecentHeartRates()
            }
        }
    }

    func fetchRecentHeartRates() {
        let heartRateType = HKQuantityType.quantityType(
            forIdentifier: .heartRate)!

        // 過去24時間のデータを取得
        let predicate = HKQuery.predicateForSamples(
            withStart: Date().addingTimeInterval(-86400),
            end: Date(), options: .strictEndDate
        )

        let query = HKSampleQuery(
            sampleType: heartRateType,
            predicate: predicate,
            limit: HKObjectQueryNoLimit,
            sortDescriptors: [
                NSSortDescriptor(key: HKSampleSortIdentifierStartDate,
                               ascending: false)
            ]
        ) { _, samples, error in
            guard let samples = samples as? [HKQuantitySample] else { return }

            for sample in samples.prefix(10) {
                let bpm = sample.quantity.doubleValue(
                    for: HKUnit(from: "count/min"))
                let date = sample.startDate
                print("\(date): \(Int(bpm)) BPM")
            }

            // 異常検出
            let avgBPM = samples.map {
                $0.quantity.doubleValue(for: HKUnit(from: "count/min"))
            }.reduce(0, +) / Double(samples.count)

            if avgBPM > 100 {
                print("⚠ 安静時心拍数が高めです: \(Int(avgBPM)) BPM")
            }
        }

        healthStore.execute(query)
    }
}
```

### コード例 2: Android Health Connect で歩数データを取得

```kotlin
import androidx.health.connect.client.HealthConnectClient
import androidx.health.connect.client.records.StepsRecord
import androidx.health.connect.client.request.ReadRecordsRequest
import androidx.health.connect.client.time.TimeRangeFilter
import java.time.Instant
import java.time.temporal.ChronoUnit

suspend fun readStepsData(healthClient: HealthConnectClient) {
    val now = Instant.now()
    val startOfDay = now.truncatedTo(ChronoUnit.DAYS)

    val request = ReadRecordsRequest(
        recordType = StepsRecord::class,
        timeRangeFilter = TimeRangeFilter.between(startOfDay, now)
    )

    val response = healthClient.readRecords(request)
    var totalSteps = 0L

    for (record in response.records) {
        totalSteps += record.count
        println("${record.startTime} - ${record.endTime}: ${record.count}歩")
    }

    println("本日の合計: ${totalSteps}歩")

    // 目標達成チェック
    val goal = 10000L
    val progress = (totalSteps.toFloat() / goal * 100).toInt()
    println("目標達成率: ${progress}% ($totalSteps / $goal)")

    if (totalSteps >= goal) {
        println("目標達成おめでとうございます！")
    } else {
        println("あと${goal - totalSteps}歩です。頑張りましょう！")
    }
}
```

### コード例 3: 心拍変動（HRV）分析でストレス推定

```python
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

def analyze_hrv(rr_intervals_ms):
    """
    RR間隔（心拍間隔）からHRV指標を計算しストレスを推定
    rr_intervals_ms: 連続するRR間隔（ミリ秒）のリスト
    """
    rr = np.array(rr_intervals_ms)

    # --- 時間領域指標 ---
    mean_rr = np.mean(rr)
    sdnn = np.std(rr, ddof=1)  # 全RR間隔の標準偏差
    rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))  # 連続差のRMS

    # --- 周波数領域指標 ---
    # RR間隔を等間隔に補間
    cumulative_time = np.cumsum(rr) / 1000.0
    f_interp = interp1d(cumulative_time, rr, kind='cubic')
    fs = 4.0  # 4Hzでリサンプリング
    t_uniform = np.arange(cumulative_time[0], cumulative_time[-1], 1/fs)
    rr_uniform = f_interp(t_uniform)

    # パワースペクトル密度
    freqs, psd = signal.welch(rr_uniform, fs=fs, nperseg=256)

    # LF (0.04-0.15Hz): 交感神経 + 副交感神経
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    lf_power = np.trapz(psd[lf_mask], freqs[lf_mask])

    # HF (0.15-0.4Hz): 副交感神経
    hf_mask = (freqs >= 0.15) & (freqs < 0.4)
    hf_power = np.trapz(psd[hf_mask], freqs[hf_mask])

    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else float('inf')

    # --- ストレス推定 ---
    # RMSSD低下 + LF/HF比上昇 → ストレス高
    if rmssd < 20 and lf_hf_ratio > 3.0:
        stress_level = "高"
    elif rmssd < 40 or lf_hf_ratio > 2.0:
        stress_level = "中"
    else:
        stress_level = "低"

    return {
        "mean_hr": 60000 / mean_rr,
        "sdnn_ms": sdnn,
        "rmssd_ms": rmssd,
        "lf_power": lf_power,
        "hf_power": hf_power,
        "lf_hf_ratio": lf_hf_ratio,
        "stress_level": stress_level
    }

# 使用例
rr_data = [820, 835, 790, 810, 845, 800, 815, 830, 795, 810]
result = analyze_hrv(rr_data)
print(f"心拍数: {result['mean_hr']:.0f} BPM")
print(f"RMSSD: {result['rmssd_ms']:.1f} ms")
print(f"LF/HF比: {result['lf_hf_ratio']:.2f}")
print(f"ストレスレベル: {result['stress_level']}")
```

### コード例 4: 加速度データから睡眠段階を推定

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def extract_sleep_features(accel_data, window_sec=30, fs=50):
    """
    加速度計データから睡眠特徴量を抽出
    accel_data: shape (N, 3) - x, y, z 軸の加速度
    """
    window_size = window_sec * fs
    features_list = []

    for i in range(0, len(accel_data) - window_size, window_size):
        window = accel_data[i:i + window_size]
        magnitude = np.sqrt(np.sum(window**2, axis=1))

        features = {
            'mean_mag': np.mean(magnitude),
            'std_mag': np.std(magnitude),
            'max_mag': np.max(magnitude),
            'activity_count': np.sum(np.abs(np.diff(magnitude)) > 0.1),
            'zero_crossing': np.sum(np.diff(np.sign(magnitude - np.mean(magnitude))) != 0),
            'energy': np.sum(magnitude**2) / len(magnitude),
            'entropy': -np.sum(np.histogram(magnitude, bins=20, density=True)[0]
                              * np.log2(np.histogram(magnitude, bins=20, density=True)[0] + 1e-10))
        }
        features_list.append(features)

    return features_list

def classify_sleep_stages(features):
    """
    睡眠段階分類: 覚醒 / 浅い睡眠 / 深い睡眠 / REM
    （学習済みモデルを使用する想定）
    """
    # 閾値ベースの簡易分類
    stages = []
    for f in features:
        if f['activity_count'] > 50:
            stages.append('覚醒')
        elif f['std_mag'] < 0.01:
            stages.append('深い睡眠')
        elif f['zero_crossing'] > 30:
            stages.append('REM')
        else:
            stages.append('浅い睡眠')
    return stages

# 使用例
accel = np.random.randn(50 * 3600 * 8, 3) * 0.05  # 8時間分
features = extract_sleep_features(accel)
stages = classify_sleep_stages(features)
print(f"睡眠段階: {len(stages)}エポック分析完了")
print(f"深い睡眠: {stages.count('深い睡眠')}エポック "
      f"({stages.count('深い睡眠') / len(stages) * 100:.1f}%)")
```

### コード例 5: watchOS — ワークアウトセッション管理

```swift
import HealthKit
import WatchKit

class WorkoutManager: NSObject, HKWorkoutSessionDelegate,
                      HKLiveWorkoutBuilderDelegate {

    let healthStore = HKHealthStore()
    var session: HKWorkoutSession?
    var builder: HKLiveWorkoutBuilder?

    func startRunningWorkout() {
        let config = HKWorkoutConfiguration()
        config.activityType = .running
        config.locationType = .outdoor

        do {
            session = try HKWorkoutSession(
                healthStore: healthStore, configuration: config
            )
            builder = session?.associatedWorkoutBuilder()

            session?.delegate = self
            builder?.delegate = self

            builder?.dataSource = HKLiveWorkoutDataSource(
                healthStore: healthStore,
                workoutConfiguration: config
            )

            let startDate = Date()
            session?.startActivity(with: startDate)
            builder?.beginCollection(withStart: startDate) { success, error in
                if success {
                    print("ワークアウト開始: ランニング")
                }
            }
        } catch {
            print("エラー: \(error)")
        }
    }

    // リアルタイムデータ更新
    func workoutBuilder(_ workoutBuilder: HKLiveWorkoutBuilder,
                       didCollectDataOf collectedTypes: Set<HKSampleType>) {
        for type in collectedTypes {
            guard let quantityType = type as? HKQuantityType else { continue }

            if let stats = workoutBuilder.statistics(for: quantityType) {
                switch quantityType {
                case HKQuantityType.quantityType(forIdentifier: .heartRate):
                    let bpm = stats.mostRecentQuantity()?.doubleValue(
                        for: HKUnit(from: "count/min")) ?? 0
                    print("心拍数: \(Int(bpm)) BPM")

                    // AIゾーン判定
                    let zone = heartRateZone(bpm: bpm, maxHR: 190)
                    print("ゾーン: \(zone)")

                case HKQuantityType.quantityType(forIdentifier: .distanceWalkingRunning):
                    let km = stats.sumQuantity()?.doubleValue(
                        for: .meterUnit(with: .kilo)) ?? 0
                    print("距離: \(String(format: "%.2f", km)) km")

                default: break
                }
            }
        }
    }

    func heartRateZone(bpm: Double, maxHR: Double) -> String {
        let percentage = bpm / maxHR * 100
        switch percentage {
        case ..<60: return "ゾーン1 (回復)"
        case 60..<70: return "ゾーン2 (脂肪燃焼)"
        case 70..<80: return "ゾーン3 (有酸素)"
        case 80..<90: return "ゾーン4 (無酸素)"
        default: return "ゾーン5 (最大)"
        }
    }

    func workoutSession(_ workoutSession: HKWorkoutSession,
                       didChangeTo toState: HKWorkoutSessionState,
                       from fromState: HKWorkoutSessionState,
                       date: Date) {}
    func workoutSession(_ workoutSession: HKWorkoutSession,
                       didFailWithError error: Error) {}
}
```

---

## 3. 比較表

### 比較表 1: 主要スマートウォッチ比較

| 項目 | Apple Watch Ultra 2 | Galaxy Watch 6 Classic | Garmin Fenix 8 | Pixel Watch 3 |
|------|-------------------|---------------------|---------------|-------------|
| OS | watchOS 11 | Wear OS 5 | Garmin OS | Wear OS 5 |
| センサー | PPG, ECG, SpO2, 温度 | PPG, ECG, BIA, SpO2 | PPG, SpO2 | PPG, ECG, SpO2 |
| NPU | S9 SiP | Exynos W940 | なし | Tensor搭載 |
| バッテリー | 36時間 | 40時間 | 28日間 | 24時間 |
| 睡眠分析 | 睡眠段階+呼吸 | 睡眠スコア+いびき | Advanced Sleep | 睡眠段階 |
| AI機能 | ダブルタップ、Siri | Galaxy AI、BIA分析 | Training Readiness | Fitbit AI |
| 価格帯 | ¥128,800〜 | ¥52,800〜 | ¥139,800〜 | ¥52,800〜 |

### 比較表 2: 健康モニタリング精度

| 指標 | 測定原理 | 精度（臨床比） | 制限事項 |
|------|---------|-------------|---------|
| 心拍数（PPG） | 光学式（緑色LED） | ±2〜5 BPM | 運動中は精度低下 |
| ECG（心電図） | 電気信号 | 医療グレード近い | 心房細動検出のみ |
| SpO2（血中酸素） | 赤色/赤外LED | ±2〜3% | 暗い肌色で精度低下 |
| 体組成（BIA） | 生体インピーダンス | ±3〜5%体脂肪率 | Galaxy Watch限定 |
| 皮膚温度 | 赤外線 | ±0.1℃ | 環境温度の影響 |
| 睡眠段階 | 加速度+心拍 | PSG比70〜80%一致 | 寝返り少ない人で低下 |

---

## 4. アンチパターン

### アンチパターン 1: ウェアラブルデータを医療診断と同等に扱う

```
❌ 悪い例:
Apple Watch の不整脈通知だけで「心房細動」と自己診断
SpO2が95%を下回ったらすぐに「肺疾患」と判断

✅ 正しいアプローチ:
- ウェアラブルデータは「スクリーニング」として活用
- 異常が続く場合は医療機関を受診
- Apple Watch ECG は「心房細動の可能性」を示唆するのみ
- FDA/医療機器認証の範囲を理解する
```

### アンチパターン 2: バッテリーを考慮せずセンサーを常時フル稼働

```
❌ 悪い例:
全センサー（PPG+加速度+GPS+SpO2）を1秒間隔で常時取得
→ バッテリーが数時間で枯渇

✅ 正しいアプローチ:
- 通常時: 心拍5分間隔、加速度バッチ処理
- 運動時: 心拍1秒間隔、GPS連続
- 睡眠時: 加速度低頻度、心拍5分間隔
- コンテキスト適応型サンプリング（activity recognitionベース）
```

---

## 5. FAQ

### Q1: スマートウォッチの心拍計はどの程度正確ですか？

**A:** 安静時は医療グレードに近い精度（±2 BPM）ですが、高強度運動中は±5〜10 BPMの誤差が生じます。これは腕の動きによるセンサー密着度の変化（モーションアーティファクト）が原因です。正確な計測にはバンドを適度に締め、手首の骨から1cm上に装着することが重要です。

### Q2: 睡眠トラッキングは信頼できますか？

**A:** 総睡眠時間の推定は比較的正確（PSG比±15分程度）ですが、睡眠段階（深い睡眠/REM/浅い睡眠）の分類は70〜80%の一致率です。これは医療用の脳波計（PSG）と比べると限界がありますが、長期トレンドの把握には十分有用です。

### Q3: ウェアラブルアプリ開発で最も重要な点は何ですか？

**A:** バッテリー効率です。ウェアラブルは小型バッテリーで動作するため、センサー取得頻度・バックグラウンド処理・通信量を最小限に抑える必要があります。Core Motionのバッチ処理、HealthKitのバックグラウンドデリバリー、BLEでの低頻度データ同期が推奨パターンです。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 主要センサー | PPG（心拍）、加速度計、ECG、SpO2、温度 |
| 健康モニタリング | 心拍/HRV/睡眠/SpO2 をAIで分析 |
| AI活用 | 不整脈検出、ストレス推定、ワークアウト最適化 |
| 開発API | HealthKit（Apple）、Health Connect（Android） |
| バッテリー | コンテキスト適応型サンプリングが必須 |
| 注意点 | 医療診断の代替ではなくスクリーニングツール |

---

## 次に読むべきガイド

- [AI PC — NPU搭載PCとローカルLLM](../01-computing/00-ai-pcs.md)
- [エッジAI — Jetson、Coral、推論最適化](../01-computing/02-edge-ai.md)
- [スマートホーム — Matter、Thread、AIオートメーション](../02-emerging/02-smart-home.md)

---

## 参考文献

1. **Apple** — "Using Apple Watch for Health Research," apple.com/healthcare, 2024
2. **Perez, M.V. et al.** — "Large-Scale Assessment of a Smartwatch to Identify Atrial Fibrillation," NEJM, 2019
3. **Samsung** — "Galaxy Watch BIA Body Composition Analysis," samsung.com, 2024
4. **Google** — "Health Connect API Documentation," developer.android.com, 2024
