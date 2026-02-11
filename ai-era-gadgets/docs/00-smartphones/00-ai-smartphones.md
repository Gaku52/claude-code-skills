# AIスマートフォン — NPU搭載チップとオンデバイスAI

> スマートフォンに搭載されるNPU（Neural Processing Unit）の仕組み、Google PixelやiPhoneのAI機能、そしてオンデバイスAIがもたらす新たなユーザー体験を体系的に解説する。

---

## この章で学ぶこと

1. **NPU（Neural Processing Unit）の仕組み** — CPU/GPU/NPUの役割分担とAI処理の高速化原理
2. **主要プラットフォームのAI機能** — Google Pixel（Tensor）とiPhone（A/Mシリーズ）の具体的なAI活用事例
3. **オンデバイスAIの設計思想** — クラウド依存を減らしプライバシーと低レイテンシを両立する技術

---

## 1. NPUアーキテクチャの基礎

### 1.1 SoC内でのNPUの位置づけ

```
┌─────────────────────────────────────────────────┐
│              スマートフォン SoC                    │
│                                                   │
│  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │
│  │  CPU     │  │  GPU     │  │  NPU / Neural   │  │
│  │ (汎用)  │  │ (並列)  │  │  Engine (AI特化) │  │
│  │ 4+4core │  │ Adreno/ │  │  INT8/FP16最適化 │  │
│  │         │  │ Mali    │  │  ~45 TOPS        │  │
│  └─────────┘  └─────────┘  └─────────────────┘  │
│                                                   │
│  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │
│  │  ISP    │  │  DSP    │  │  Memory (LPDDR5) │  │
│  │ (カメラ)│  │ (信号) │  │  8〜16 GB        │  │
│  └─────────┘  └─────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────┘
```

### 1.2 NPUの処理性能比較

```
処理性能（TOPS: Trillion Operations Per Second）

Apple A18 Pro (Neural Engine)    ████████████████████████████████████ 35 TOPS
Snapdragon 8 Gen 3 (Hexagon)    █████████████████████████████████████████████ 45 TOPS
Google Tensor G4                 ████████████████████████████████ 32 TOPS
MediaTek Dimensity 9300          ██████████████████████████████████████ 38 TOPS
Samsung Exynos 2400              ██████████████████████████████████████████ 40 TOPS
```

---

## 2. コード例

### コード例 1: Android — ML Kit でオンデバイス画像ラベリング

```kotlin
// build.gradle
// implementation 'com.google.mlkit:image-labeling:17.0.8'

import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.label.ImageLabeling
import com.google.mlkit.vision.label.defaults.ImageLabelerOptions

fun labelImage(bitmap: Bitmap) {
    val image = InputImage.fromBitmap(bitmap, 0)
    val options = ImageLabelerOptions.Builder()
        .setConfidenceThreshold(0.7f) // 70%以上の信頼度のみ
        .build()
    val labeler = ImageLabeling.getClient(options)

    labeler.process(image)
        .addOnSuccessListener { labels ->
            for (label in labels) {
                println("${label.text}: ${label.confidence}")
                // 例: "Cat: 0.95", "Animal: 0.92"
            }
        }
        .addOnFailureListener { e ->
            println("エラー: ${e.message}")
        }
}
```

### コード例 2: iOS — Core ML でオンデバイス推論

```swift
import CoreML
import Vision

func classifyImage(_ image: CGImage) {
    // MobileNetV3 モデルをロード
    guard let model = try? VNCoreMLModel(
        for: MobileNetV3(configuration: .init()).model
    ) else { return }

    let request = VNCoreMLRequest(model: model) { request, error in
        guard let results = request.results as? [VNClassificationObservation] else { return }

        // 上位3件を表示
        for result in results.prefix(3) {
            print("\(result.identifier): \(result.confidence * 100)%")
        }
    }

    let handler = VNImageRequestHandler(cgImage: image)
    try? handler.perform([request])
}
```

### コード例 3: TensorFlow Lite — NPU デリゲート活用

```python
import tensorflow as tf

# モデルの量子化（FP32 → INT8）でNPU最適化
converter = tf.lite.TFLiteConverter.from_saved_model("model_dir")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]  # NPU向け量子化

# 代表的なデータセットで量子化キャリブレーション
def representative_dataset():
    for data in calibration_data:
        yield [data.astype("float32")]

converter.representative_dataset = representative_dataset
tflite_model = converter.convert()

# 保存
with open("model_quantized.tflite", "wb") as f:
    f.write(tflite_model)

print(f"モデルサイズ: {len(tflite_model) / 1024 / 1024:.1f} MB")
```

### コード例 4: NNAPI（Android Neural Networks API）ベンチマーク

```kotlin
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.nnapi.NnApiDelegate

fun benchmarkNPU(modelPath: String) {
    // NNAPI デリゲート（NPU優先）
    val nnApiDelegate = NnApiDelegate(
        NnApiDelegate.Options().apply {
            setAllowFp16(true)             // FP16演算を許可
            setExecutionPreference(         // NPU優先実行
                NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED
            )
        }
    )

    val options = Interpreter.Options().apply {
        addDelegate(nnApiDelegate)
        setNumThreads(4)
    }

    val interpreter = Interpreter(loadModelFile(modelPath), options)

    // ウォームアップ + 計測
    val input = FloatArray(224 * 224 * 3)
    val output = Array(1) { FloatArray(1000) }

    repeat(10) { interpreter.run(input, output) } // ウォームアップ

    val start = System.nanoTime()
    repeat(100) { interpreter.run(input, output) }
    val elapsed = (System.nanoTime() - start) / 1_000_000.0 / 100

    println("平均推論時間: ${elapsed}ms")  // NPU: ~3ms, CPU: ~25ms
}
```

### コード例 5: Pixel の Gemini Nano — オンデバイス要約

```kotlin
// Android AICore API（Pixel 8 Pro 以降）
import com.google.android.gms.aicore.GenerativeModel
import com.google.android.gms.aicore.GenerateContentRequest

suspend fun summarizeOnDevice(text: String): String {
    val model = GenerativeModel("gemini-nano")

    val request = GenerateContentRequest.newBuilder()
        .addText("以下のテキストを3行で要約してください:\n$text")
        .build()

    val response = model.generateContent(request)
    return response.text ?: "要約を生成できませんでした"
}

// 使用例
val article = "長い記事のテキスト..."
val summary = summarizeOnDevice(article)
println(summary)
// → "1. AIスマートフォンはNPUを搭載..."
// → "2. オンデバイス処理でプライバシーを保護..."
// → "3. クラウド不要で低遅延応答を実現..."
```

---

## 3. オンデバイスAI vs クラウドAI

### 比較表 1: 処理方式の違い

```
┌─────────────────────────────────────────────────────┐
│             AI処理のデータフロー比較                    │
│                                                       │
│  【クラウドAI】                                       │
│  端末 ──→ ネットワーク ──→ クラウド ──→ ネットワーク ──→ 端末│
│       (100〜500ms)   (推論)    (100〜500ms)           │
│                                                       │
│  【オンデバイスAI】                                    │
│  端末 ──→ NPU ──→ 結果                                │
│       (3〜20ms)                                       │
│                                                       │
│  【ハイブリッドAI】                                    │
│  端末 ──→ NPU(前処理/軽量推論)                         │
│       └──→ クラウド(高度な推論) ──→ 端末               │
└─────────────────────────────────────────────────────┘
```

| 項目 | オンデバイスAI | クラウドAI |
|------|--------------|-----------|
| レイテンシ | 3〜20ms | 100ms〜数秒 |
| プライバシー | データが端末内に留まる | サーバーに送信される |
| オフライン動作 | 可能 | 不可 |
| モデルサイズ | 数MB〜数GB（量子化必須） | 制限なし（数百GB可） |
| 精度 | やや低い（量子化損失） | 高い（FP32/BF16） |
| コスト | 端末負荷（バッテリー消費） | サーバー運用コスト |
| 更新 | OS/アプリ更新が必要 | サーバー側で即時更新 |

### 比較表 2: 主要チップセットのAI性能

| チップセット | メーカー | NPU性能 (TOPS) | 対応モデル | オンデバイスLLM |
|-------------|---------|---------------|-----------|---------------|
| A18 Pro | Apple | 35 | iPhone 16 Pro | Apple Intelligence |
| Snapdragon 8 Gen 3 | Qualcomm | 45 | Galaxy S24等 | Llama 2 7B対応 |
| Tensor G4 | Google | 32 | Pixel 9 | Gemini Nano |
| Dimensity 9300 | MediaTek | 38 | 各社ハイエンド | Llama 2対応 |
| Exynos 2400 | Samsung | 40 | Galaxy S24(一部) | Galaxy AI |

---

## 4. アンチパターン

### アンチパターン 1: すべてのAI処理をオンデバイスに押し込む

```
❌ 悪い例:
7Bパラメータの大規模LLMをスマートフォンで直接動かそうとする
→ メモリ不足、バッテリー急消耗、応答が数十秒かかる

✅ 正しいアプローチ:
- 軽量タスク（画像分類、音声認識）→ オンデバイス
- 重いタスク（長文生成、複雑な推論）→ クラウド
- 前処理はオンデバイス、最終推論はクラウド → ハイブリッド
```

### アンチパターン 2: NPU非対応モデルをそのまま配備する

```
❌ 悪い例:
FP32モデル（500MB）をそのままTFLiteに変換してデプロイ
→ NPUでなくCPUフォールバックが発生、10倍遅くなる

✅ 正しいアプローチ:
1. INT8量子化でモデルサイズを1/4に圧縮
2. NNAPI/Core ML デリゲートを明示的に指定
3. ベンチマークでNPU実行を確認（CPU比で5〜10倍高速なら成功）
```

---

## 5. FAQ

### Q1: NPUとGPUの違いは何ですか？

**A:** GPUは汎用的な並列計算（ゲーム描画、科学計算など）に最適化されていますが、NPUはニューラルネットワークの推論（行列積、畳み込み）に特化した回路です。NPUはINT8/FP16の低精度演算を効率よく行うことで、消費電力あたりの性能（TOPS/W）がGPUの数倍に達します。

### Q2: オンデバイスAIでどの程度のモデルが動きますか？

**A:** 2024〜2025年時点で、スマートフォンでは以下が現実的です:
- **画像分類/物体検出**: MobileNetV3、EfficientNet-Lite（数MB）
- **音声認識**: Whisper tiny/base（40〜140MB）
- **LLM**: Gemini Nano（1.8B/3.25Bパラメータ）、Phi-3 Mini（3.8B、量子化後2GB程度）
- 7Bパラメータ以上のLLMは12GB以上のRAMが必要で、フラグシップ機に限定されます。

### Q3: Apple IntelligenceとGoogle AIの違いは何ですか？

**A:** 主な違いは以下の通りです:
- **Apple Intelligence**: プライバシー最優先。Private Cloud Compute でクラウド処理時もデータを暗号化。Siri + ChatGPT連携。
- **Google AI（Pixel）**: Gemini Nano をオンデバイスで実行。Google検索/アプリとの深い統合。Cloud AI との連携が得意。
- 両者ともオンデバイス処理を基本としつつ、高度なタスクはクラウドにオフロードするハイブリッド方式です。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| NPU の役割 | AI推論に特化したプロセッサ。INT8/FP16で省電力・高速処理 |
| 主要チップ | Apple Neural Engine, Qualcomm Hexagon, Google Tensor TPU |
| オンデバイスAI | プライバシー保護・低遅延・オフライン動作が利点 |
| 量子化 | FP32→INT8で4倍圧縮、NPU活用の必須技術 |
| ハイブリッド構成 | 軽量タスクはオンデバイス、重いタスクはクラウドが最適解 |
| 今後の展望 | 端末LLMの高度化、マルチモーダルAIの標準化 |

---

## 次に読むべきガイド

- [AIカメラ — 計算フォトグラフィとAI編集](./01-ai-cameras.md)
- [AIアシスタント — Siri/Google Assistant/Alexa](./02-ai-assistants.md)
- [AI PC — NPU搭載PCとローカルLLM](../01-computing/00-ai-pcs.md)

---

## 参考文献

1. **Qualcomm** — "Snapdragon 8 Gen 3 Mobile Platform," qualcomm.com, 2024
2. **Apple** — "Apple Intelligence Technical Overview," developer.apple.com, 2024
3. **Google** — "AICore and Gemini Nano on Android," developer.android.com, 2024
4. **ARM** — "Ethos-U NPU Architecture Reference," developer.arm.com, 2024
5. **TensorFlow** — "TensorFlow Lite for Mobile and Edge Devices," tensorflow.org, 2024
