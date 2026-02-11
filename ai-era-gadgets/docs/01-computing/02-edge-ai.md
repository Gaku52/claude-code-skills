# エッジAIガイド

> NPU、Google Coral、NVIDIA Jetsonを活用し、クラウドに依存しないローカルAI推論を実現する

## この章で学ぶこと

1. **エッジAIの基本概念** — クラウドAIとの違い、NPU/VPU/TPUの役割
2. **主要プラットフォーム** — NVIDIA Jetson、Google Coral、Apple Neural Engine の実践的活用法
3. **最適化テクニック** — モデル量子化、プルーニング、知識蒸留によるエッジ向けモデル最適化

---

## 1. エッジAIの基本概念

### クラウドAI vs エッジAI

```
+-----------------------------+     +-----------------------------+
|       クラウドAI             |     |       エッジAI               |
+-----------------------------+     +-----------------------------+
|                             |     |                             |
|  デバイス → ネットワーク →   |     |  デバイス内で推論完結        |
|  クラウドサーバーで推論 →    |     |                             |
|  結果をデバイスに返す        |     |  +--------+                |
|                             |     |  | カメラ  |                |
|  +------+    +--------+    |     |  +---+----+                |
|  |デバイス| → |クラウド | → |     |      |                     |
|  +------+    | GPU/TPU |   |     |  +---v--------+            |
|       ↑      +--------+   |     |  | NPU/GPU     |            |
|       |          |         |     |  | ローカル推論 |            |
|       +----------+         |     |  +---+---------+            |
|   遅延: 50-500ms           |     |      |                     |
|   要インターネット          |     |  +---v----+               |
|                             |     |  | 結果    |  遅延: 1-10ms |
+-----------------------------+     |  +--------+  オフライン可 |
                                    +-----------------------------+
```

### エッジAIの利点と課題

```
+-----------------------------------------------------------+
|                エッジAIの利点                                |
+-----------------------------------------------------------+
|  低遅延      | 1-10ms のリアルタイム推論                    |
|  プライバシー | データがデバイスから出ない                    |
|  帯域節約    | 大量データのクラウド転送が不要                |
|  オフライン  | インターネット接続不要                        |
|  コスト      | クラウドAPI課金なし                            |
+-----------------------------------------------------------+
|                エッジAIの課題                                |
+-----------------------------------------------------------+
|  計算制約    | 限られた演算能力・メモリ                      |
|  モデル制約  | 大規模モデルは動作不可                        |
|  電力制約    | バッテリー駆動デバイスでの消費電力            |
|  更新        | モデルアップデートの配信が複雑                |
+-----------------------------------------------------------+
```

---

## 2. エッジAIアクセラレータの種類

### アクセラレータ比較表

| アクセラレータ | 代表製品 | 演算性能(TOPS) | 消費電力 | 主な用途 |
|--------------|---------|---------------|---------|---------|
| NPU (Neural Processing Unit) | Apple Neural Engine, Qualcomm Hexagon | 11-45 TOPS | 1-5W | スマートフォン、PC |
| Edge TPU | Google Coral | 4 TOPS | 2W | IoTデバイス、カメラ |
| Jetson (GPU) | NVIDIA Jetson Orin | 20-275 TOPS | 7-60W | ロボット、自動運転 |
| VPU (Vision Processing Unit) | Intel Movidius | 1-4 TOPS | 1-2W | カメラ、ドローン |
| FPGA | Xilinx/AMD Versal | カスタム | 5-75W | 産業用、カスタムAI |
| ASIC | Google Edge TPU, Hailo | 4-26 TOPS | 1-5W | 特定用途向け |

### TOPS (Tera Operations Per Second) の目安

```
+-----------------------------------------------------------+
|  TOPS と実行可能なモデルの目安                               |
+-----------------------------------------------------------+
|                                                           |
|  1 TOPS   |█|            画像分類 (MobileNet)             |
|  4 TOPS   |███|          物体検出 (SSD MobileNet)         |
|  10 TOPS  |███████|      顔認識 + ポーズ推定               |
|  20 TOPS  |█████████████| セマンティックセグメンテーション  |
|  40 TOPS  |████████████████████████|  小規模LLM (1-3B)    |
|  100 TOPS |████████████████████████████████████|  7B LLM  |
|  275 TOPS |██████████████████████████████████████████████| |
|            自動運転レベル、複数モデル同時実行               |
+-----------------------------------------------------------+
```

---

## 3. NVIDIA Jetson

### Jetson 製品ラインナップ

| モデル | GPU | CPU | メモリ | AI性能 | 消費電力 | 用途 |
|--------|-----|-----|--------|--------|---------|------|
| Jetson Orin Nano | 1024 CUDA cores | 6-core A78AE | 4-8GB | 20-40 TOPS | 7-15W | 入門、軽量AI |
| Jetson Orin NX | 1024 CUDA cores | 8-core A78AE | 8-16GB | 70-100 TOPS | 10-25W | ロボット、ドローン |
| Jetson AGX Orin | 2048 CUDA cores | 12-core A78AE | 32-64GB | 200-275 TOPS | 15-60W | 自動運転、産業用 |

### コード例1: Jetson での推論（TensorRT）

```python
# Jetson で TensorRT を使った高速推論
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

def load_engine(engine_path):
    """TensorRT エンジンのロード"""
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, \
         trt.Runtime(logger) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def infer(engine, input_data):
    """推論実行"""
    context = engine.create_execution_context()

    # 入出力バッファの確保
    h_input = np.ascontiguousarray(input_data)
    h_output = np.empty(output_shape, dtype=np.float32)

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    # ホスト→デバイス転送
    cuda.memcpy_htod(d_input, h_input)

    # 推論実行
    context.execute_v2([int(d_input), int(d_output)])

    # デバイス→ホスト転送
    cuda.memcpy_dtoh(h_output, d_output)

    return h_output

# ONNX → TensorRT 変換コマンド
# trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
```

### コード例2: Jetson でのリアルタイム物体検出

```python
import jetson_inference
import jetson_utils

# モデルロード（自動的に TensorRT に最適化される）
net = jetson_inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# カメラ入力
camera = jetson_utils.videoSource("csi://0")  # CSI カメラ
display = jetson_utils.videoOutput("display://0")

while display.IsStreaming():
    img = camera.Capture()

    # 物体検出（GPU で推論）
    detections = net.Detect(img)

    for det in detections:
        print(f"検出: {net.GetClassDesc(det.ClassID)} "
              f"信頼度: {det.Confidence:.2f} "
              f"位置: ({det.Left:.0f},{det.Top:.0f})-({det.Right:.0f},{det.Bottom:.0f})")

    display.Render(img)
    display.SetStatus(f"FPS: {net.GetNetworkFPS():.0f}")
```

---

## 4. Google Coral

### コード例3: Coral Edge TPU での推論

```python
# Google Coral Edge TPU を使った画像分類
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from PIL import Image

# Edge TPU 用にコンパイル済みモデルをロード
interpreter = make_interpreter("mobilenet_v2_1.0_224_quant_edgetpu.tflite")
interpreter.allocate_tensors()

# ラベル読み込み
labels = read_label_file("imagenet_labels.txt")

# 画像の前処理と推論
image = Image.open("cat.jpg").resize(
    common.input_size(interpreter), Image.LANCZOS
)
common.set_input(interpreter, image)

# 推論実行（Edge TPU で高速処理）
interpreter.invoke()

# 結果取得
classes = classify.get_classes(interpreter, top_k=5)
for c in classes:
    print(f"{labels.get(c.id, c.id)}: {c.score:.4f}")

# 推論時間の計測
import time
start = time.perf_counter()
for _ in range(100):
    interpreter.invoke()
elapsed = (time.perf_counter() - start) / 100
print(f"推論時間: {elapsed*1000:.1f} ms ({1/elapsed:.0f} FPS)")
```

### モデルのEdge TPU向け変換フロー

```
+-------------------+     +--------------------+     +-------------------+
|  学習済みモデル    | --> | TFLite 変換         | --> | INT8 量子化        |
|  (PyTorch/TF)     |     | (float32 → float32)|     | (float32 → int8)  |
+-------------------+     +--------------------+     +-------------------+
                                                            |
                                                            v
+-------------------+     +--------------------+     +-------------------+
| Edge TPU で推論   | <-- | Edge TPU Compiler  | <-- | 量子化済み        |
| 4 TOPS / 2W       |     | edgetpu_compiler   |     | TFLite モデル     |
+-------------------+     +--------------------+     +-------------------+
```

---

## 5. Apple Neural Engine / NPU搭載PC

### コード例4: Core ML でのオンデバイス推論

```python
# Python (coremltools) でモデル変換
import coremltools as ct
import torch

# PyTorch モデルを Core ML に変換
model = torchvision.models.mobilenet_v2(pretrained=True)
model.eval()

example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(shape=(1, 3, 224, 224), scale=1/255.0)],
    compute_units=ct.ComputeUnit.ALL,  # CPU + GPU + Neural Engine
)
mlmodel.save("MobileNetV2.mlpackage")
```

```swift
// Swift での推論（Neural Engine 自動活用）
import CoreML
import Vision

let model = try! MobileNetV2(configuration: .init())

let request = VNCoreMLRequest(model: try! VNCoreMLModel(for: model.model)) {
    request, error in
    guard let results = request.results as? [VNClassificationObservation] else { return }
    let top = results.prefix(3)
    for result in top {
        print("\(result.identifier): \(result.confidence * 100)%")
    }
}

// Neural Engine で推論: ~1ms / 画像
```

### コード例5: Qualcomm AI Engine (Snapdragon)

```python
# Qualcomm AI Engine Direct (QNN) での推論
# スマートフォン上の Hexagon NPU を活用

# モデル変換: ONNX → QNN
# qnn-onnx-converter --input_network model.onnx \
#     --output_path model.cpp \
#     --input_dim "input" 1,3,224,224

# 量子化（Hexagon NPU 向け）
# qnn-net-run --model model.so \
#     --backend libQnnHtp.so \
#     --input_list input_list.txt

# Android アプリでの使用 (Java/Kotlin)
# Qualcomm Neural Processing SDK を使用
```

---

## 6. モデル最適化テクニック

### 最適化手法の比較表

| 手法 | サイズ削減 | 速度向上 | 精度影響 | 実装難度 |
|------|-----------|---------|---------|---------|
| INT8 量子化 (PTQ) | 75% | 2-4x | 1-3%低下 | 低 |
| INT8 量子化 (QAT) | 75% | 2-4x | 0.5-1%低下 | 中 |
| プルーニング (構造化) | 50-90% | 1.5-3x | 1-5%低下 | 中 |
| 知識蒸留 | 50-90% | 2-10x | 1-3%低下 | 高 |
| ONNX Runtime | - | 1.2-2x | なし | 低 |
| TensorRT | - | 2-5x | ≤1%低下 | 中 |

---

## 7. アンチパターン

### アンチパターン1: エッジデバイスのスペックを無視したモデル選択

```
NG: Jetson Nano (4GB) に 7B パラメータの LLM をデプロイしようとする
    → メモリ不足で動作不能

OK: デバイスのスペックに合わせたモデルを選択
    Jetson Nano 4GB  → MobileNet, EfficientNet-Lite (画像)
                      → DistilBERT, TinyLlama 1.1B (テキスト)
    Jetson Orin 32GB → YOLOv8, SAM (画像)
                      → Llama 3 8B 4bit量子化 (テキスト)
```

### アンチパターン2: 量子化なしでエッジにデプロイ

```
NG: FP32 モデルをそのままエッジデバイスに配置
    → メモリ消費が大きく、推論が遅い

OK: エッジ向けに必ず量子化を行う
    1. Post-Training Quantization (PTQ) — 再学習不要
    2. Quantization-Aware Training (QAT) — 精度が重要な場合
    3. ターゲットデバイスの対応精度に合わせる
       Edge TPU → INT8 必須
       Jetson   → FP16 または INT8
       Apple ANE → FP16
```

---

## FAQ

### Q1. Raspberry Pi でAI推論は実用的か？

Raspberry Pi 5 は CPU 推論で MobileNet 程度なら 30-50ms（20-30 FPS）で動作する。Coral USB Accelerator を接続すると Edge TPU で 5-10ms に高速化される。リアルタイム性が不要な用途（定期的な画像分類、音声コマンド認識）では十分実用的。

### Q2. エッジデバイスでLLMは動くか？

2025年時点で、Jetson AGX Orin（64GB）なら Llama 3 8B の4bit量子化版が 10-20 tokens/sec で動作する。スマートフォンでは Phi-3 Mini (3.8B) や Gemma 2B が実用的。Apple M4 搭載MacならLlama 3 70B の4bit量子化も可能。

### Q3. エッジAI開発の入門にはどのデバイスが良い？

予算1万円以下なら Coral USB Accelerator + Raspberry Pi。予算5万円以下なら Jetson Orin Nano。Apple デバイスを持っているなら Core ML + Create ML で追加投資なしに始められる。

---

## まとめ

| 概念 | 要点 |
|------|------|
| エッジAI | デバイス上でAI推論を完結、低遅延・プライバシー保護 |
| NPU | ニューラルネットワーク処理特化チップ |
| NVIDIA Jetson | 高性能エッジGPU、TensorRT対応 |
| Google Coral | Edge TPU搭載、低消費電力INT8推論 |
| Apple Neural Engine | iPhoneやMacに内蔵のAIアクセラレータ |
| 量子化 | FP32→INT8で4倍のメモリ削減・速度向上 |
| TensorRT | NVIDIAの推論最適化エンジン |
| TOPS | AI演算性能の指標（Tera Operations Per Second） |

---

## 次に読むべきガイド

- **01-computing/03-cloud-ai-hardware.md** — クラウドAIハードウェア：TPU、Inferentia
- **01-computing/01-gpu-computing.md** — GPU：NVIDIA/AMD、CUDA
- **02-emerging/01-robotics.md** — ロボティクス：Boston Dynamics、Figure

---

## 参考文献

1. **NVIDIA Jetson 公式ドキュメント** https://developer.nvidia.com/embedded-computing
2. **Google Coral 公式** https://coral.ai/docs/
3. **Apple Core ML 公式** https://developer.apple.com/documentation/coreml
4. **TensorFlow Lite for Microcontrollers** https://www.tensorflow.org/lite/microcontrollers
