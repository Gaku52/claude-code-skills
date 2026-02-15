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

### 1.1 PPG（光電容積脈波）センサーの動作原理

PPG（Photoplethysmography）は、ウェアラブルの心拍計測の基盤技術である。緑色LEDを皮膚に照射し、血管の容積変化による光の吸収変動を検出する。

```
┌─────────────────────────────────────────────────┐
│          PPG センサー 動作原理                     │
│                                                   │
│  ┌───────────────────────────────┐               │
│  │      皮膚表面                  │               │
│  │  ←←← 反射光（フォトダイオード検出）            │
│  │                                │               │
│  │  LED(緑/赤/赤外) ───→ 照射光    │               │
│  │       │                        │               │
│  │       ▼                        │               │
│  │  ┌─────────────┐              │               │
│  │  │ 毛細血管     │              │               │
│  │  │ 血液量変化    │              │               │
│  │  │ ↕ 収縮/拡張  │              │               │
│  │  └─────────────┘              │               │
│  └───────────────────────────────┘               │
│                                                   │
│  緑色LED (525nm): 心拍数計測                       │
│  赤色LED (660nm) + 赤外LED (940nm): SpO2計測       │
│                                                   │
│  心拍時 → 血液量増加 → 光吸収増加 → 信号減少       │
│  拡張期 → 血液量減少 → 光吸収減少 → 信号増加       │
│                                                   │
│  信号処理: ローパスフィルタ → ピーク検出 → BPM算出  │
└─────────────────────────────────────────────────┘
```

### 1.2 健康データの処理フロー

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

### 1.3 BIA（生体インピーダンス分析）による体組成測定

Galaxy Watch に搭載されている BIA センサーは、微弱電流を体に流して電気抵抗を測定することで体組成を推定する。

```
┌─────────────────────────────────────────────────┐
│          BIA 体組成分析 フロー                     │
│                                                   │
│  ステップ1: 電極接触                              │
│  ┌─────────────┐                                 │
│  │ 裏面電極(2点) │ ── 微弱交流電流 (50kHz) ──     │
│  │ 手首接触      │                    │           │
│  └─────────────┘                    │           │
│                                      │           │
│  ステップ2: 反対側電極で受信                      │
│  ┌─────────────┐                    │           │
│  │ 側面ボタン    │ ◀── 電流受信 ──────┘           │
│  │ (指で触れる)  │                                │
│  └─────────────┘                                │
│                                                   │
│  ステップ3: インピーダンス計算                     │
│  Z = V / I (電圧 / 電流)                          │
│                                                   │
│  ステップ4: 体組成推定                            │
│  ┌──────────────────────────────┐               │
│  │ 体脂肪率 = f(Z, 身長, 体重, 年齢, 性別)      │
│  │ 骨格筋量 = f(Z, 身長, 体重)                   │
│  │ 基礎代謝 = f(骨格筋量, 体脂肪率)             │
│  │ 体水分量 = f(Z, 身長)                        │
│  └──────────────────────────────┘               │
│                                                   │
│  制限事項:                                        │
│  - 食後・運動後は精度低下                          │
│  - ペースメーカー装着者は使用不可                  │
│  - 医療グレードの DEXA スキャンと比べて ±3-5%      │
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
                print("安静時心拍数が高めです: \(Int(avgBPM)) BPM")
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

### コード例 6: HealthKit バックグラウンドデリバリー（リアルタイム通知）

HealthKit のバックグラウンドデリバリー機能を使うことで、新しい健康データが記録された際にアプリをバックグラウンドで起動してデータを処理できる。

```swift
import HealthKit
import UserNotifications

class HealthBackgroundMonitor {
    let healthStore = HKHealthStore()

    /// バックグラウンドデリバリーの設定
    func enableBackgroundDelivery() {
        let heartRateType = HKQuantityType.quantityType(
            forIdentifier: .heartRate)!

        // バックグラウンドで心拍データ更新を受け取る
        healthStore.enableBackgroundDelivery(
            for: heartRateType,
            frequency: .immediate  // 即時通知
        ) { success, error in
            if success {
                print("バックグラウンドデリバリー有効化成功")
                self.setupObserverQuery()
            }
        }
    }

    /// オブザーバークエリで新しいデータを監視
    func setupObserverQuery() {
        let heartRateType = HKQuantityType.quantityType(
            forIdentifier: .heartRate)!

        let query = HKObserverQuery(
            sampleType: heartRateType,
            predicate: nil
        ) { [weak self] query, completionHandler, error in
            // 新しい心拍データが記録された
            self?.checkForAbnormalHeartRate()
            completionHandler()
        }

        healthStore.execute(query)
    }

    /// 異常心拍の検出とユーザー通知
    func checkForAbnormalHeartRate() {
        let heartRateType = HKQuantityType.quantityType(
            forIdentifier: .heartRate)!

        // 直近5分間のデータを取得
        let predicate = HKQuery.predicateForSamples(
            withStart: Date().addingTimeInterval(-300),
            end: Date(), options: .strictEndDate
        )

        let query = HKSampleQuery(
            sampleType: heartRateType,
            predicate: predicate,
            limit: HKObjectQueryNoLimit,
            sortDescriptors: nil
        ) { _, samples, _ in
            guard let samples = samples as? [HKQuantitySample],
                  !samples.isEmpty else { return }

            let bpmValues = samples.map {
                $0.quantity.doubleValue(for: HKUnit(from: "count/min"))
            }

            let avgBPM = bpmValues.reduce(0, +) / Double(bpmValues.count)
            let maxBPM = bpmValues.max() ?? 0
            let minBPM = bpmValues.min() ?? 0

            // 異常検出ロジック
            // 1. 安静時頻脈（100BPM以上が5分以上継続）
            if avgBPM > 100 {
                self.sendAlert(
                    title: "高心拍数検出",
                    body: "安静時心拍数が\(Int(avgBPM)) BPMです。" +
                          "長時間続く場合は医療機関にご相談ください。"
                )
            }

            // 2. 徐脈（40BPM以下）
            if minBPM < 40 && minBPM > 0 {
                self.sendAlert(
                    title: "低心拍数検出",
                    body: "心拍数が\(Int(minBPM)) BPMに低下しました。"
                )
            }

            // 3. 心拍変動の急激な変化
            if bpmValues.count >= 3 {
                let diffs = zip(bpmValues, bpmValues.dropFirst()).map {
                    abs($0.0 - $0.1)
                }
                let maxDiff = diffs.max() ?? 0
                if maxDiff > 30 {
                    self.sendAlert(
                        title: "心拍変動異常",
                        body: "短時間で\(Int(maxDiff)) BPMの変動を検出しました。"
                    )
                }
            }
        }

        healthStore.execute(query)
    }

    func sendAlert(title: String, body: String) {
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = .default

        let request = UNNotificationRequest(
            identifier: UUID().uuidString,
            content: content,
            trigger: nil  // 即時通知
        )

        UNUserNotificationCenter.current().add(request)
    }
}
```

### コード例 7: Core Motion によるバッチ処理（歩行分析AI）

```swift
import CoreMotion
import CoreML

class GaitAnalyzer {
    let motionManager = CMMotionManager()
    let pedometer = CMPedometer()
    var accelerometerBuffer: [(x: Double, y: Double, z: Double, t: Double)] = []

    /// 加速度計のバッチ収集開始
    func startGaitCollection() {
        guard motionManager.isAccelerometerAvailable else { return }

        // 50Hzでサンプリング（バッテリー効率を考慮）
        motionManager.accelerometerUpdateInterval = 1.0 / 50.0

        motionManager.startAccelerometerUpdates(to: .main) {
            [weak self] data, error in
            guard let data = data else { return }

            self?.accelerometerBuffer.append((
                x: data.acceleration.x,
                y: data.acceleration.y,
                z: data.acceleration.z,
                t: data.timestamp
            ))

            // 10秒分のデータが溜まったら分析
            if let self = self,
               self.accelerometerBuffer.count >= 500 {
                self.analyzeGait()
            }
        }
    }

    /// 歩行パターン分析
    func analyzeGait() {
        let buffer = accelerometerBuffer
        accelerometerBuffer.removeAll()

        // 特徴量の計算
        let magnitudes = buffer.map {
            sqrt($0.x * $0.x + $0.y * $0.y + $0.z * $0.z)
        }

        let mean = magnitudes.reduce(0, +) / Double(magnitudes.count)
        let variance = magnitudes.map { ($0 - mean) * ($0 - mean) }
            .reduce(0, +) / Double(magnitudes.count)
        let stdDev = sqrt(variance)

        // ステップ検出（ゼロクロッシング法）
        let detrended = magnitudes.map { $0 - mean }
        var stepCount = 0
        for i in 1..<detrended.count {
            if detrended[i-1] < 0 && detrended[i] >= 0 {
                stepCount += 1
            }
        }

        // 歩行対称性の計算
        let halfLen = magnitudes.count / 2
        let firstHalf = Array(magnitudes[..<halfLen])
        let secondHalf = Array(magnitudes[halfLen...])
        let symmetryScore = calculateSymmetry(firstHalf, secondHalf)

        // 歩行品質スコア
        let gaitScore = evaluateGaitQuality(
            stepRegularity: stdDev,
            symmetry: symmetryScore,
            cadence: Double(stepCount) * 6.0  // 10秒間 → 1分換算
        )

        print("歩数(10秒): \(stepCount)")
        print("ケイデンス: \(stepCount * 6) steps/min")
        print("対称性スコア: \(String(format: "%.2f", symmetryScore))")
        print("歩行品質スコア: \(String(format: "%.0f", gaitScore))/100")
    }

    func calculateSymmetry(_ a: [Double], _ b: [Double]) -> Double {
        let minLen = min(a.count, b.count)
        var correlation = 0.0
        let meanA = a.prefix(minLen).reduce(0, +) / Double(minLen)
        let meanB = b.prefix(minLen).reduce(0, +) / Double(minLen)

        var numerator = 0.0
        var denomA = 0.0
        var denomB = 0.0

        for i in 0..<minLen {
            let da = a[i] - meanA
            let db = b[i] - meanB
            numerator += da * db
            denomA += da * da
            denomB += db * db
        }

        let denom = sqrt(denomA * denomB)
        return denom > 0 ? numerator / denom : 0
    }

    func evaluateGaitQuality(stepRegularity: Double,
                             symmetry: Double,
                             cadence: Double) -> Double {
        // 正常範囲: ケイデンス 100-130, 対称性 0.8+
        var score = 100.0

        if cadence < 80 || cadence > 150 {
            score -= 20  // 異常なケイデンス
        }
        if symmetry < 0.7 {
            score -= 30  // 非対称な歩行
        } else if symmetry < 0.85 {
            score -= 15
        }
        if stepRegularity > 0.3 {
            score -= 15  // 不規則な歩行
        }

        return max(0, min(100, score))
    }
}
```

### コード例 8: Health Connect を使った睡眠データの書き込みと分析

```kotlin
import androidx.health.connect.client.HealthConnectClient
import androidx.health.connect.client.records.SleepSessionRecord
import androidx.health.connect.client.records.SleepStageRecord
import androidx.health.connect.client.request.ReadRecordsRequest
import androidx.health.connect.client.time.TimeRangeFilter
import java.time.Instant
import java.time.ZoneOffset
import java.time.LocalDate
import java.time.Duration

class SleepAnalyzer(private val healthClient: HealthConnectClient) {

    /**
     * 睡眠セッションの記録
     */
    suspend fun recordSleepSession(
        startTime: Instant,
        endTime: Instant,
        stages: List<Pair<Instant, SleepStageRecord.StageType>>
    ) {
        // 睡眠段階レコードの作成
        val stageRecords = stages.zipWithNext().map { (current, next) ->
            SleepStageRecord(
                startTime = current.first,
                startZoneOffset = ZoneOffset.ofHours(9),
                endTime = next.first,
                endZoneOffset = ZoneOffset.ofHours(9),
                stage = current.second
            )
        }

        // 睡眠セッションレコードの作成
        val sessionRecord = SleepSessionRecord(
            startTime = startTime,
            startZoneOffset = ZoneOffset.ofHours(9),
            endTime = endTime,
            endZoneOffset = ZoneOffset.ofHours(9),
            stages = stageRecords.map {
                SleepSessionRecord.Stage(
                    startTime = it.startTime,
                    endTime = it.endTime,
                    stage = when (it.stage) {
                        SleepStageRecord.StageType.AWAKE ->
                            SleepSessionRecord.STAGE_TYPE_AWAKE
                        SleepStageRecord.StageType.LIGHT ->
                            SleepSessionRecord.STAGE_TYPE_LIGHT
                        SleepStageRecord.StageType.DEEP ->
                            SleepSessionRecord.STAGE_TYPE_DEEP
                        SleepStageRecord.StageType.REM ->
                            SleepSessionRecord.STAGE_TYPE_REM
                        else -> SleepSessionRecord.STAGE_TYPE_UNKNOWN
                    }
                )
            }
        )

        healthClient.insertRecords(listOf(sessionRecord))
        println("睡眠セッションを記録しました")
    }

    /**
     * 過去7日間の睡眠分析レポート
     */
    suspend fun generateWeeklySleepReport(): SleepReport {
        val now = Instant.now()
        val weekAgo = now.minus(Duration.ofDays(7))

        val request = ReadRecordsRequest(
            recordType = SleepSessionRecord::class,
            timeRangeFilter = TimeRangeFilter.between(weekAgo, now)
        )

        val sessions = healthClient.readRecords(request).records

        val dailyStats = sessions.map { session ->
            val duration = Duration.between(session.startTime, session.endTime)
            val stages = session.stages

            val deepSleep = stages
                .filter { it.stage == SleepSessionRecord.STAGE_TYPE_DEEP }
                .sumOf { Duration.between(it.startTime, it.endTime).toMinutes() }

            val remSleep = stages
                .filter { it.stage == SleepSessionRecord.STAGE_TYPE_REM }
                .sumOf { Duration.between(it.startTime, it.endTime).toMinutes() }

            val awakeTime = stages
                .filter { it.stage == SleepSessionRecord.STAGE_TYPE_AWAKE }
                .sumOf { Duration.between(it.startTime, it.endTime).toMinutes() }

            DailySleep(
                date = session.startTime,
                totalMinutes = duration.toMinutes(),
                deepMinutes = deepSleep,
                remMinutes = remSleep,
                awakeMinutes = awakeTime,
                sleepEfficiency = ((duration.toMinutes() - awakeTime).toFloat()
                    / duration.toMinutes() * 100)
            )
        }

        return SleepReport(
            averageDuration = dailyStats.map { it.totalMinutes }.average(),
            averageDeepSleep = dailyStats.map { it.deepMinutes }.average(),
            averageRemSleep = dailyStats.map { it.remMinutes }.average(),
            averageEfficiency = dailyStats.map { it.sleepEfficiency.toDouble() }.average(),
            sleepScore = calculateSleepScore(dailyStats),
            recommendation = generateRecommendation(dailyStats)
        )
    }

    private fun calculateSleepScore(stats: List<DailySleep>): Int {
        var score = 100
        val avgDuration = stats.map { it.totalMinutes }.average()
        val avgEfficiency = stats.map { it.sleepEfficiency.toDouble() }.average()
        val avgDeep = stats.map { it.deepMinutes }.average()

        // 睡眠時間: 7-9時間が理想
        if (avgDuration < 360) score -= 25       // 6時間未満
        else if (avgDuration < 420) score -= 10  // 6-7時間
        else if (avgDuration > 600) score -= 5   // 10時間超過

        // 睡眠効率: 85%以上が理想
        if (avgEfficiency < 75) score -= 20
        else if (avgEfficiency < 85) score -= 10

        // 深い睡眠: 総睡眠時間の15-25%が理想
        val deepRatio = avgDeep / avgDuration * 100
        if (deepRatio < 10) score -= 15
        else if (deepRatio < 15) score -= 5

        return score.coerceIn(0, 100)
    }

    private fun generateRecommendation(stats: List<DailySleep>): String {
        val avgDuration = stats.map { it.totalMinutes }.average()
        val avgDeep = stats.map { it.deepMinutes }.average()

        return when {
            avgDuration < 360 -> "睡眠時間が不足しています。就寝時間を30分早めることをお勧めします。"
            avgDeep < avgDuration * 0.10 -> "深い睡眠が少なめです。就寝前のカフェインやスクリーンタイムを減らしましょう。"
            else -> "良好な睡眠パターンです。この調子を維持しましょう。"
        }
    }
}

data class DailySleep(
    val date: Instant,
    val totalMinutes: Long,
    val deepMinutes: Long,
    val remMinutes: Long,
    val awakeMinutes: Long,
    val sleepEfficiency: Float
)

data class SleepReport(
    val averageDuration: Double,
    val averageDeepSleep: Double,
    val averageRemSleep: Double,
    val averageEfficiency: Double,
    val sleepScore: Int,
    val recommendation: String
)
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
| 価格帯 | 128,800円〜 | 52,800円〜 | 139,800円〜 | 52,800円〜 |

### 比較表 2: 健康モニタリング精度

| 指標 | 測定原理 | 精度（臨床比） | 制限事項 |
|------|---------|-------------|---------|
| 心拍数（PPG） | 光学式（緑色LED） | ±2〜5 BPM | 運動中は精度低下 |
| ECG（心電図） | 電気信号 | 医療グレード近い | 心房細動検出のみ |
| SpO2（血中酸素） | 赤色/赤外LED | ±2〜3% | 暗い肌色で精度低下 |
| 体組成（BIA） | 生体インピーダンス | ±3〜5%体脂肪率 | Galaxy Watch限定 |
| 皮膚温度 | 赤外線 | ±0.1℃ | 環境温度の影響 |
| 睡眠段階 | 加速度+心拍 | PSG比70〜80%一致 | 寝返り少ない人で低下 |

### 比較表 3: 開発プラットフォーム比較

| 項目 | watchOS (Apple) | Wear OS (Google) | Tizen (Samsung旧) | Garmin SDK |
|------|----------------|-----------------|------------------|-----------|
| 言語 | Swift | Kotlin | C/C++ | Monkey C |
| 健康API | HealthKit | Health Connect | Samsung Health SDK | Garmin Health SDK |
| ML推論 | Core ML | TFLite | TFLite | なし |
| UI フレームワーク | SwiftUI | Jetpack Compose | EFL | Garmin UI |
| バックグラウンド | Background Delivery | WorkManager | Background Service | なし（制限大） |
| センサーアクセス | Core Motion | SensorManager | Sensor API | Sensor API |
| コンプリケーション | ClockKit | Tiles API | Watchface Studio | Data Fields |
| デバッグ | Xcode + シミュレータ | Android Studio | Tizen Studio | Garmin Simulator |
| ストア配信 | App Store | Google Play | Galaxy Store | Connect IQ Store |

### 比較表 4: HRV 指標の臨床的意義

| HRV 指標 | 計算方法 | 正常範囲 | 臨床的意義 | ウェアラブル対応 |
|----------|---------|---------|-----------|---------------|
| RMSSD | 連続RR差のRMS | 20-100 ms | 副交感神経活動の指標 | Apple Watch, Oura |
| SDNN | RR間隔の標準偏差 | 50-150 ms | 自律神経全体の変動 | Garmin, Whoop |
| LF/HF比 | 低周波/高周波比 | 1.0-3.0 | 交感/副交感バランス | 計算が必要 |
| pNN50 | 50ms以上の差の割合 | 5-40% | 副交感神経トーン | Apple Watch |
| DFA alpha1 | Detrended分析 | 0.75-1.0 | 運動強度の指標 | Garmin (一部) |

---

## 4. ユースケース

### ユースケース 1: 不整脈（心房細動）の早期発見

Apple Watch の ECG 機能と不規則なリズム通知機能は、FDA（米国食品医薬品局）と日本の医薬品医療機器総合機構（PMDA）の認可を受けている。Stanford Apple Heart Study（2019年）では、419,297人の参加者のうち0.52%に不規則なリズムが検出され、そのうち34%が心房細動と診断された。

```
┌─────────────────────────────────────────────────┐
│          心房細動検出のフロー                      │
│                                                   │
│  常時モニタリング                                  │
│  PPG 信号 → 不規則リズム検出 AI                    │
│       │                                           │
│       ├── 正常 → 何もしない                        │
│       │                                           │
│       └── 不規則検出 → 通知                        │
│                │                                   │
│                ▼                                   │
│  ユーザーがECGを記録（30秒間）                     │
│       │                                           │
│       ├── 洞調律 → 「正常な心拍リズム」            │
│       ├── 心房細動 → 「心房細動の兆候あり」        │
│       └── 判定不能 → 「再計測を推奨」             │
│                │                                   │
│                ▼                                   │
│  PDF レポート生成 → 医師と共有                     │
└─────────────────────────────────────────────────┘
```

**開発者が注意すべき点:**
- Apple Watch の ECG は「心房細動の可能性を示唆する」スクリーニングツールであり、確定診断はできない
- 医療機器認証の範囲を理解し、アプリ内で医療診断を行うような表現をしないこと
- 各国の医療機器規制（FDA 510(k)、CE マーキング、PMDA 認可）に準拠する必要がある

### ユースケース 2: AIフィットネスコーチング

ウェアラブルデータを活用した個別化されたトレーニング推奨システムの実装パターン。

```python
class AIFitnessCoach:
    """
    ウェアラブルデータに基づくAIフィットネスコーチ
    心拍ゾーン、HRV、睡眠スコアから最適なトレーニングを推奨
    """
    def __init__(self, user_profile):
        self.max_hr = 220 - user_profile['age']  # 最大心拍数の推定
        self.resting_hr = user_profile.get('resting_hr', 60)
        self.fitness_level = user_profile.get('fitness_level', 'intermediate')

    def calculate_training_readiness(self, hrv_rmssd, sleep_score,
                                      previous_training_load):
        """
        トレーニング準備度スコアの計算
        HRV、睡眠、前日のトレーニング負荷から総合判定
        """
        # HRV スコア (0-40点): 自律神経回復度
        hrv_baseline = 45.0  # 個人のベースライン（学習で更新）
        hrv_score = min(40, max(0, (hrv_rmssd / hrv_baseline) * 30))

        # 睡眠スコア (0-30点)
        sleep_component = min(30, sleep_score * 0.3)

        # 回復スコア (0-30点): 前日の負荷が高いほど低い
        recovery_score = max(0, 30 - previous_training_load * 0.3)

        total = hrv_score + sleep_component + recovery_score

        return {
            'total_score': round(total),
            'hrv_component': round(hrv_score),
            'sleep_component': round(sleep_component),
            'recovery_component': round(recovery_score),
            'recommendation': self._get_recommendation(total)
        }

    def _get_recommendation(self, readiness_score):
        if readiness_score >= 80:
            return {
                'intensity': 'high',
                'suggestion': '高強度トレーニング推奨（インターバル、テンポ走）',
                'hr_zone_target': 'ゾーン4-5 (80-100% max HR)',
                'duration_minutes': 45
            }
        elif readiness_score >= 60:
            return {
                'intensity': 'moderate',
                'suggestion': '中強度トレーニング推奨（ペース走、筋トレ）',
                'hr_zone_target': 'ゾーン3 (70-80% max HR)',
                'duration_minutes': 60
            }
        elif readiness_score >= 40:
            return {
                'intensity': 'low',
                'suggestion': '軽めのトレーニング推奨（ジョグ、ヨガ、ストレッチ）',
                'hr_zone_target': 'ゾーン1-2 (50-70% max HR)',
                'duration_minutes': 30
            }
        else:
            return {
                'intensity': 'rest',
                'suggestion': '完全休養推奨。回復に集中してください。',
                'hr_zone_target': 'なし',
                'duration_minutes': 0
            }

    def analyze_workout(self, hr_data, duration_minutes):
        """ワークアウト後の分析"""
        hr_reserve = self.max_hr - self.resting_hr

        trimp = 0  # Training Impulse
        for hr in hr_data:
            hr_ratio = (hr - self.resting_hr) / hr_reserve
            trimp += hr_ratio * 0.64 * (2.718 ** (1.92 * hr_ratio))

        trimp *= (duration_minutes / len(hr_data))

        zone_distribution = self._calculate_zone_distribution(hr_data)

        return {
            'trimp_score': round(trimp, 1),
            'training_effect': self._training_effect(trimp),
            'zone_distribution': zone_distribution,
            'peak_hr': max(hr_data),
            'avg_hr': sum(hr_data) / len(hr_data),
            'calories_estimated': round(trimp * 0.8)
        }

    def _calculate_zone_distribution(self, hr_data):
        zones = {f'ゾーン{i}': 0 for i in range(1, 6)}
        for hr in hr_data:
            pct = hr / self.max_hr * 100
            if pct < 60: zones['ゾーン1'] += 1
            elif pct < 70: zones['ゾーン2'] += 1
            elif pct < 80: zones['ゾーン3'] += 1
            elif pct < 90: zones['ゾーン4'] += 1
            else: zones['ゾーン5'] += 1

        total = len(hr_data)
        return {k: f"{v/total*100:.1f}%" for k, v in zones.items()}

    def _training_effect(self, trimp):
        if trimp < 50: return "リカバリー"
        elif trimp < 100: return "有酸素基盤強化"
        elif trimp < 200: return "有酸素能力向上"
        elif trimp < 300: return "高強度持久力向上"
        else: return "オーバーリーチ（注意）"

# 使用例
coach = AIFitnessCoach({'age': 35, 'resting_hr': 55})
readiness = coach.calculate_training_readiness(
    hrv_rmssd=52.0, sleep_score=82, previous_training_load=65
)
print(f"トレーニング準備度: {readiness['total_score']}/100")
print(f"推奨: {readiness['recommendation']['suggestion']}")
```

### ユースケース 3: 転倒検出と緊急通報

Apple Watch の転倒検出機能は、加速度計とジャイロスコープのデータからAIモデルで転倒パターンを識別する。65歳以上のユーザーではデフォルトで有効化されている。

```
┌─────────────────────────────────────────────────┐
│          転倒検出 アルゴリズムフロー               │
│                                                   │
│  加速度 + ジャイロ データ (100Hz)                  │
│       │                                           │
│       ▼                                           │
│  ┌───────────────────┐                           │
│  │ 衝撃検出           │                           │
│  │ 加速度 > 3G ?      │                           │
│  └─────────┬─────────┘                           │
│            │ Yes                                  │
│            ▼                                      │
│  ┌───────────────────┐                           │
│  │ 転倒パターン照合    │                           │
│  │ - 急落下 + 衝撃     │                           │
│  │ - 回転 + 停止       │                           │
│  │ - つまずき + 前傾   │                           │
│  └─────────┬─────────┘                           │
│            │ マッチ                                │
│            ▼                                      │
│  ┌───────────────────┐                           │
│  │ 転倒後の動作確認    │                           │
│  │ 1分間の不動検出     │                           │
│  └─────────┬─────────┘                           │
│            │ 動きなし                              │
│            ▼                                      │
│  ┌───────────────────┐                           │
│  │ 緊急SOS発動         │                           │
│  │ - 位置情報送信      │                           │
│  │ - 緊急連絡先に通知  │                           │
│  │ - 119番自動発信     │                           │
│  └───────────────────┘                           │
└─────────────────────────────────────────────────┘
```

---

## 5. トラブルシューティング

### 問題 1: PPG 心拍数が不正確

**症状:** 運動中の心拍数が実際よりも大幅に高い/低い値を示す。特にランニングやウエイトトレーニング中に顕著。

**原因分析:**
1. モーションアーティファクト: 腕の動きがPPGセンサーの光学信号に干渉
2. バンドの装着位置が不適切: 骨の上に装着すると血管が少なく精度低下
3. タトゥーや暗い肌色: 光の吸収/反射特性が変わる

**解決策:**
- バンドを手首の骨（尺骨茎状突起）から1cm上に装着し、適度にフィットさせる
- 高強度運動時は胸部心拍ベルト（Polar H10等）を併用する
- 運動前にバンドを少し締め直す
- 寒冷環境ではウォームアップで末梢血流を確保してから計測

### 問題 2: HealthKit/Health Connect のデータが同期されない

**症状:** ウォッチで計測したデータがスマートフォンのアプリに反映されない。

**原因分析:**
```
┌─────────────────────────────────────────────────┐
│  データ同期のチェックポイント                      │
│                                                   │
│  1. 権限設定 → HealthKit/Health Connect の        │
│     読み取り/書き込み権限が正しいか？              │
│                                                   │
│  2. Bluetooth接続 → ウォッチとスマートフォンが     │
│     ペアリングされているか？                       │
│                                                   │
│  3. バックグラウンド更新 → アプリのバックグラウンド │
│     リフレッシュが有効か？                          │
│                                                   │
│  4. ストレージ → デバイスのストレージ空きがあるか？ │
│                                                   │
│  5. OS バージョン → 互換性のあるバージョンか？     │
└─────────────────────────────────────────────────┘
```

**解決策 (iOS):**
```swift
// HealthKit の権限確認
let status = healthStore.authorizationStatus(
    for: HKQuantityType.quantityType(forIdentifier: .heartRate)!
)

switch status {
case .notDetermined:
    // まだ権限をリクエストしていない
    requestAuthorization()
case .sharingAuthorized:
    // 書き込み権限あり（読み取り権限は確認できない仕様）
    print("権限OK")
case .sharingDenied:
    // 権限が拒否されている
    // 設定アプリに誘導
    print("設定 > プライバシー > ヘルスケア から権限を有効にしてください")
}
```

### 問題 3: バッテリーの急速消耗

**症状:** スマートウォッチのバッテリーが通常より早く消耗する（1日もたない）。

**原因と対策:**

| 原因 | 消費電力への影響 | 対策 |
|------|---------------|------|
| 常時表示ディスプレイ (AOD) | +20-30% | 使用しない時間帯はOFF |
| GPS連続使用 | +40-60% | ワークアウト時のみGPS有効 |
| 心拍の高頻度サンプリング | +15-25% | 通常時は5分間隔に設定 |
| SpO2 常時測定 | +10-20% | 睡眠時のみ有効化 |
| Wi-Fi 常時接続 | +10-15% | Bluetooth のみに切替 |
| 通知の過多 | +5-10% | 不要な通知をフィルタリング |

### 問題 4: 睡眠トラッキングの誤検出

**症状:** 昼寝や映画鑑賞中にベッドで寝転がっているだけで「睡眠」と判定される。就寝/起床時刻が実際と30分以上ずれる。

**対策:**
- ベッドタイムスケジュールを手動設定して検出範囲を絞る
- 加速度だけでなく心拍変動データも考慮するアルゴリズムにする（開発者向け）
- 就寝前にフォーカスモード（おやすみモード）を有効にし、ウォッチに就寝の意図を伝える

---

## 6. パフォーマンス最適化

### 最適化 1: コンテキスト適応型サンプリング

ウェアラブルの最大の制約はバッテリーである。全センサーを常時フル稼働するのではなく、ユーザーの状態（安静/歩行/運動/睡眠）に応じてサンプリングレートを動的に変更する。

```
┌─────────────────────────────────────────────────┐
│  コンテキスト適応型サンプリング戦略                │
│                                                   │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ Activity  │──▶│ サンプリング│──▶│ バッテリー│   │
│  │ Recognition│   │ レート調整  │   │ 延長効果  │   │
│  └──────────┘   └──────────┘   └──────────┘    │
│                                                   │
│  状態        心拍    加速度   GPS    SpO2         │
│  ────────   ─────  ─────  ────   ────          │
│  安静時      5分     OFF    OFF    OFF   → 36h   │
│  歩行中      1分     25Hz   OFF    OFF   → 24h   │
│  運動中      1秒     50Hz   1Hz    OFF   → 8h    │
│  睡眠中      5分     10Hz   OFF    15分  → 30h   │
│  異常検出    連続     50Hz   OFF    連続  → 4h    │
│                                                   │
│  Core Motion の CMMotionActivityManager で状態検出 │
│  → 状態変化時にセンサー設定を動的に切り替え        │
└─────────────────────────────────────────────────┘
```

### 最適化 2: オンデバイスMLモデルの量子化

```python
# Core ML モデルの量子化で推論速度とバッテリー効率を改善
import coremltools as ct

# 学習済みモデルの読み込み
model = ct.models.MLModel("HeartRateClassifier.mlpackage")

# INT8 量子化（メモリ使用量を75%削減）
quantized_model = ct.models.neural_network.quantization_utils.quantize_weights(
    model,
    nbits=8,  # 32bit → 8bit
    quantization_mode="linear"
)
quantized_model.save("HeartRateClassifier_INT8.mlpackage")

# パフォーマンス比較
# FP32: 推論時間 5ms, モデルサイズ 2.4MB, メモリ 8MB
# INT8: 推論時間 2ms, モデルサイズ 0.6MB, メモリ 2MB
# Neural Engine 使用時はさらに高速化
```

### 最適化 3: BLE（Bluetooth Low Energy）データ転送の最適化

ウォッチからスマートフォンへのデータ転送は BLE で行われるが、接続・転送ごとにバッテリーを消費する。データのバッチ処理と圧縮が重要。

```swift
// BLE データ転送の最適化パターン
class OptimizedBLETransfer {
    // NG: 毎秒データを送信
    // → BLE接続の維持とデータ転送で電力消費大

    // OK: データをバッファリングし、5分ごとにまとめて送信
    var dataBuffer: [HealthSample] = []
    let transferInterval: TimeInterval = 300  // 5分

    func addSample(_ sample: HealthSample) {
        dataBuffer.append(sample)

        // バッファが閾値に達したら送信
        if dataBuffer.count >= 300 {  // 5分 × 1Hz = 300サンプル
            transferData()
        }
    }

    func transferData() {
        // Delta圧縮: 前のサンプルとの差分のみ送信
        let compressed = deltaCompress(dataBuffer)

        // BLE MTU に合わせてチャンク分割
        let chunks = compressed.chunked(into: 512)  // 512 bytes/chunk

        for chunk in chunks {
            peripheral.writeValue(chunk, for: characteristic, type: .withResponse)
        }

        dataBuffer.removeAll()
    }

    func deltaCompress(_ samples: [HealthSample]) -> Data {
        // 心拍値の差分エンコーディング
        // [72, 73, 72, 74, 71] → [72, +1, -1, +2, -3]
        // 差分は小さい値なので少ないビット数で表現可能
        var encoded = Data()
        encoded.append(contentsOf: withUnsafeBytes(of: samples[0].value) {
            Array($0)
        })

        for i in 1..<samples.count {
            let diff = Int8(samples[i].value - samples[i-1].value)
            encoded.append(contentsOf: withUnsafeBytes(of: diff) { Array($0) })
        }

        return encoded
    }
}
```

---

## 7. 設計パターン

### パターン 1: リアクティブ健康データパイプライン

健康データは継続的に生成されるストリームであるため、リアクティブプログラミングパターンが適している。Combine（iOS）/ Flow（Android）を使った実装パターン。

```swift
import Combine
import HealthKit

class ReactiveHealthPipeline {
    private var cancellables = Set<AnyCancellable>()
    private let healthStore = HKHealthStore()

    /// 心拍データのリアクティブストリーム
    func heartRateStream() -> AnyPublisher<Double, Never> {
        let subject = PassthroughSubject<Double, Never>()

        let heartRateType = HKQuantityType.quantityType(
            forIdentifier: .heartRate)!

        let query = HKAnchoredObjectQuery(
            type: heartRateType,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { _, samples, _, _, _ in
            guard let samples = samples as? [HKQuantitySample] else { return }
            for sample in samples {
                let bpm = sample.quantity.doubleValue(
                    for: HKUnit(from: "count/min"))
                subject.send(bpm)
            }
        }

        query.updateHandler = { _, samples, _, _, _ in
            guard let samples = samples as? [HKQuantitySample] else { return }
            for sample in samples {
                let bpm = sample.quantity.doubleValue(
                    for: HKUnit(from: "count/min"))
                subject.send(bpm)
            }
        }

        healthStore.execute(query)

        return subject.eraseToAnyPublisher()
    }

    /// パイプライン構築: フィルタ → 分析 → アラート
    func setupPipeline() {
        heartRateStream()
            // ノイズ除去: 生理学的にありえない値を除外
            .filter { $0 > 30 && $0 < 250 }
            // 移動平均: 5サンプルの平均を取る
            .scan([Double]()) { buffer, newValue in
                var buf = buffer
                buf.append(newValue)
                if buf.count > 5 { buf.removeFirst() }
                return buf
            }
            .map { buffer -> Double in
                buffer.reduce(0, +) / Double(buffer.count)
            }
            // 異常検出
            .sink { avgBPM in
                if avgBPM > 120 {
                    NotificationCenter.default.post(
                        name: .highHeartRate,
                        object: nil,
                        userInfo: ["bpm": avgBPM]
                    )
                }
            }
            .store(in: &cancellables)
    }
}

extension Notification.Name {
    static let highHeartRate = Notification.Name("highHeartRate")
}
```

### パターン 2: ローカルファースト + クラウド同期アーキテクチャ

ウェアラブルアプリでは「ローカルで即座に動作し、クラウドに非同期で同期する」パターンが基本。ネットワーク接続がなくてもデータ記録と分析が継続できるようにする。

```
┌─────────────────────────────────────────────────┐
│  ローカルファースト + クラウド同期                  │
│                                                   │
│  ┌──────────────────────────────┐               │
│  │ ウォッチ (watchOS/Wear OS)    │               │
│  │                               │               │
│  │  センサー → ローカルDB          │               │
│  │              (Core Data/Room)  │               │
│  │              │                 │               │
│  │              ├── リアルタイム分析│               │
│  │              │   (オンデバイスML)│               │
│  │              │                 │               │
│  │              └── 通知/アラート  │               │
│  └──────────────┬───────────────┘               │
│                 │ BLE/Wi-Fi                       │
│  ┌──────────────▼───────────────┐               │
│  │ スマートフォン                  │               │
│  │                               │               │
│  │  ローカルDB ←→ 同期エンジン     │               │
│  │              │                 │               │
│  │              ├── 高度な分析     │               │
│  │              │   (LLMベースの   │               │
│  │              │    レコメンド)   │               │
│  │              │                 │               │
│  │              └── ダッシュボード │               │
│  └──────────────┬───────────────┘               │
│                 │ HTTPS (Wi-Fi接続時のみ)         │
│  ┌──────────────▼───────────────┐               │
│  │ クラウド                       │               │
│  │                               │               │
│  │  長期データ保存                 │               │
│  │  集約分析（人口ベース比較）     │               │
│  │  AIモデル学習・配信             │               │
│  └──────────────────────────────┘               │
└─────────────────────────────────────────────────┘
```

---

## 8. アンチパターン

### アンチパターン 1: ウェアラブルデータを医療診断と同等に扱う

```
悪い例:
Apple Watch の不整脈通知だけで「心房細動」と自己診断
SpO2が95%を下回ったらすぐに「肺疾患」と判断

正しいアプローチ:
- ウェアラブルデータは「スクリーニング」として活用
- 異常が続く場合は医療機関を受診
- Apple Watch ECG は「心房細動の可能性」を示唆するのみ
- FDA/医療機器認証の範囲を理解する
```

### アンチパターン 2: バッテリーを考慮せずセンサーを常時フル稼働

```
悪い例:
全センサー（PPG+加速度+GPS+SpO2）を1秒間隔で常時取得
→ バッテリーが数時間で枯渇

正しいアプローチ:
- 通常時: 心拍5分間隔、加速度バッチ処理
- 運動時: 心拍1秒間隔、GPS連続
- 睡眠時: 加速度低頻度、心拍5分間隔
- コンテキスト適応型サンプリング（activity recognitionベース）
```

### アンチパターン 3: プライバシーを軽視した健康データの取り扱い

```
悪い例:
- 健康データを暗号化せずにクラウドに送信
- ユーザーの同意なくデータを第三者と共有
- 健康データと個人識別情報を同じDBに保存

正しいアプローチ:
- HIPAA/GDPR/個人情報保護法 に準拠
- 健康データは暗号化して保存・転送（AES-256、TLS 1.3）
- データの匿名化/仮名化を必ず実施
- HealthKit / Health Connect の権限モデルに従う
- ユーザーがいつでもデータを削除できるようにする
- Apple の HealthKit Review Guidelines を遵守
```

### アンチパターン 4: 単一センサーに過度に依存

```
悪い例:
PPGの心拍数だけで運動強度を判定
→ モーションアーティファクトで大きく外れる可能性

正しいアプローチ:
- センサーフュージョン: PPG + 加速度 + ジャイロを組み合わせる
- 加速度でモーションアーティファクトを検出し、PPG信号を補正
- 信頼度スコアを付与し、低信頼度のデータには注意を促す
- 複数センサーの整合性チェック（心拍が急上昇しても加速度が
  変わらない場合はアーティファクトの可能性が高い）
```

---

## 9. エッジケース分析

### エッジケース 1: 寒冷環境でのPPG精度低下

寒冷環境（0℃以下）では末梢血管が収縮し、手首の血流量が大幅に減少する。これによりPPGセンサーの信号対雑音比（SNR）が低下し、心拍数の測定精度が大きく悪化する。

```
環境温度と PPG 精度の関係:

温度        血管状態        PPG 精度     対策
──────    ──────────    ──────      ──────────
25℃以上    正常拡張        ±2 BPM      通常通り
15-25℃    やや収縮        ±3-5 BPM    通常通り
5-15℃     収縮            ±5-10 BPM   装着位置を確認
0℃以下    強い収縮        ±10-20 BPM  胸部ベルト推奨
-10℃以下  極度の収縮      測定不能      手袋の下に装着

開発者向け対策:
1. 信号品質インジケータ (SQI) を計算し、低品質時は
   ユーザーに通知する
2. 加速度計データとの相関を確認し、アーティファクトを除去
3. 寒冷環境を温度センサーで検出し、自動的に測定間隔を
   長くする（バッテリー節約 + 誤データ防止）
```

### エッジケース 2: タトゥーがある手首でのセンサー精度

タトゥーの色素（特に濃い色のインク）はPPGセンサーの光を吸収し、信号品質を大幅に低下させる。Apple は公式に「一部のタトゥーがセンサー性能に影響する場合がある」と認めている。

**影響の程度:**
| タトゥーの色 | 影響レベル | 対策 |
|------------|----------|------|
| 黒/濃紺 | 高（測定不能の場合あり） | 反対側の手首に装着 |
| 赤/緑 | 中 | 信頼度低下を受容 |
| 薄い色/細い線 | 低〜なし | 通常通り使用可能 |
| 白/UV | ほぼなし | 通常通り使用可能 |

---

## 10. 開発者チェックリスト

### ウェアラブルアプリ開発の品質チェック

```
[ ] HealthKit / Health Connect の権限リクエストは適切なタイミングで行っているか
[ ] バックグラウンドでのセンサーアクセスはコンテキスト適応型か
[ ] バッテリー消費のテストを最低48時間実施したか
[ ] センサーデータに信頼度スコアを付与しているか
[ ] モーションアーティファクトの除去処理を実装しているか
[ ] オフラインでもデータ記録が継続できるか
[ ] 健康データの暗号化（保存時・転送時）を実装しているか
[ ] HIPAA/GDPR/個人情報保護法の要件を確認したか
[ ] 医療機器規制（FDA 510(k)/CE/PMDA）の対象かどうか確認したか
[ ] 睡眠/運動/安静時のセンサーサンプリングレートを最適化したか
[ ] BLE データ転送のバッチ処理を実装しているか
[ ] アプリの医療的表現が規制の範囲内か確認したか
[ ] 異常検出のアラート頻度が適切か（過多は無視される）
[ ] ユーザーがデータの削除・エクスポートを行えるか
[ ] コンプリケーション/Tilesのデータ更新頻度を最適化したか
```

---

## FAQ

### Q1: スマートウォッチの心拍計はどの程度正確ですか？

**A:** 安静時は医療グレードに近い精度（±2 BPM）ですが、高強度運動中は±5〜10 BPMの誤差が生じます。これは腕の動きによるセンサー密着度の変化（モーションアーティファクト）が原因です。正確な計測にはバンドを適度に締め、手首の骨から1cm上に装着することが重要です。

### Q2: 睡眠トラッキングは信頼できますか？

**A:** 総睡眠時間の推定は比較的正確（PSG比±15分程度）ですが、睡眠段階（深い睡眠/REM/浅い睡眠）の分類は70〜80%の一致率です。これは医療用の脳波計（PSG）と比べると限界がありますが、長期トレンドの把握には十分有用です。

### Q3: ウェアラブルアプリ開発で最も重要な点は何ですか？

**A:** バッテリー効率です。ウェアラブルは小型バッテリーで動作するため、センサー取得頻度・バックグラウンド処理・通信量を最小限に抑える必要があります。Core Motionのバッチ処理、HealthKitのバックグラウンドデリバリー、BLEでの低頻度データ同期が推奨パターンです。

### Q4: ウェアラブルの健康データを医療に活用できますか？

**A:** 可能ですが、いくつかの条件があります。第一に、データの精度とバリデーションが臨床研究で確認されていること。Apple Watch のECG機能はFDA De Novo認可を取得しており、心房細動のスクリーニングとして医師も参照可能です。第二に、医療機器としての認証が必要な場合があること。ウェルネスアプリとして販売するか医療機器として販売するかで規制が大きく異なります。Apple の ResearchKit / CareKit フレームワークは、臨床研究でのデータ収集を標準化する目的で設計されています。

### Q5: 複数のウェアラブルデバイスのデータを統合するにはどうすればよいですか？

**A:** HealthKit（iOS）と Health Connect（Android）がデータ統合の標準プラットフォームです。複数のウェアラブル（Apple Watch + Oura Ring + Garmin など）からのデータが統一されたスキーマでこれらのプラットフォームに保存されます。開発者はこれらのAPIを通じて統合されたデータにアクセスできます。ただし、同じ種類のデータ（例: 心拍数）が複数のソースから来る場合、どのソースを優先するかのロジックが必要です。一般的には「最もデバイスに近いソース」（直接計測 > 推定値）を優先します。

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
| プライバシー | HIPAA/GDPR準拠、暗号化、匿名化が必須 |
| センサーフュージョン | 複数センサーの組み合わせで精度向上 |

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
5. **Bent, B. et al.** — "Investigating sources of inaccuracy in wearable optical heart rate sensors," npj Digital Medicine, 2020
6. **Castaneda, D. et al.** — "A review on wearable photoplethysmography sensors and their potential future applications in health care," IJBS, 2018
