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

### エッジAIアーキテクチャパターン

```
+-----------------------------------------------------------+
|  エッジAI デプロイメントパターン                             |
+-----------------------------------------------------------+
|                                                           |
|  パターン1: 完全エッジ                                     |
|  [デバイス] → [NPU推論] → [結果] → [アクション]            |
|  用途: セキュリティカメラ、自動運転                         |
|                                                           |
|  パターン2: エッジ + クラウド連携                           |
|  [デバイス] → [エッジ前処理] → [クラウドで精密推論]         |
|  用途: 音声アシスタント（ウェイクワードはエッジ）            |
|                                                           |
|  パターン3: フェデレーテッドラーニング                       |
|  [デバイス群] → [ローカル学習] → [勾配のみクラウドに集約]   |
|  用途: 医療データ、キーボード予測                           |
|                                                           |
|  パターン4: エッジメッシュ                                  |
|  [デバイス群] → [ローカルP2P通信] → [分散推論]              |
|  用途: 工場IoT、スマートビルディング                        |
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

### NPU世代別性能比較

| NPU | メーカー | 世代 | TOPS | 搭載デバイス | 特徴 |
|-----|---------|------|------|-------------|------|
| Neural Engine (16コア) | Apple | A16/M2 | 15.8 TOPS | iPhone 14 Pro, MacBook Air | 低消費電力、Core ML最適化 |
| Neural Engine (16コア) | Apple | A17 Pro/M3 | 35 TOPS | iPhone 15 Pro, MacBook Pro | AV1デコード対応 |
| Neural Engine (16コア) | Apple | M4 | 38 TOPS | iPad Pro 2024 | ハードウェアレイトレーシング |
| Hexagon DSP | Qualcomm | Gen 3 | 45 TOPS | Snapdragon 8 Gen 3 | INT4対応、Micro NPU |
| NPU | Intel | Meteor Lake | 10 TOPS | Core Ultra | Windows AI PC |
| NPU | AMD | Ryzen AI | 16 TOPS | Ryzen 7040 | XDNA アーキテクチャ |
| NPU | Qualcomm | X Elite | 45 TOPS | Snapdragon X Elite | Copilot+ PC |

---

## 3. NVIDIA Jetson

### Jetson 製品ラインナップ

| モデル | GPU | CPU | メモリ | AI性能 | 消費電力 | 用途 |
|--------|-----|-----|--------|--------|---------|------|
| Jetson Orin Nano | 1024 CUDA cores | 6-core A78AE | 4-8GB | 20-40 TOPS | 7-15W | 入門、軽量AI |
| Jetson Orin NX | 1024 CUDA cores | 8-core A78AE | 8-16GB | 70-100 TOPS | 10-25W | ロボット、ドローン |
| Jetson AGX Orin | 2048 CUDA cores | 12-core A78AE | 32-64GB | 200-275 TOPS | 15-60W | 自動運転、産業用 |

### Jetson開発環境のセットアップ

```bash
# JetPack SDK のインストール（推奨）
# JetPack 6.x はJetson Orin シリーズ向け

# 1. NVIDIA SDK Managerをホストマシンにインストール
# https://developer.nvidia.com/sdk-manager

# 2. Jetsonデバイスをリカバリモードで接続
# ボタン操作でリカバリモードに入る

# 3. SDK Manager でフラッシュ
# JetPack 6.0 = Ubuntu 22.04 + CUDA 12.2 + cuDNN + TensorRT

# 4. 初期設定後の確認
jetson_release  # JetPack バージョン確認
nvidia-smi      # GPU状態確認（注: jtopの方が詳細）

# jtop のインストール（Jetson用モニタリングツール）
sudo pip3 install jetson-stats
sudo systemctl restart jtop.service
jtop  # GPU/CPU/メモリ/温度をリアルタイム表示

# Docker + NVIDIA Container Runtime の確認
sudo docker run --runtime nvidia --rm nvcr.io/nvidia/l4t-base:r36.2.0 \
    nvidia-smi
```

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

### コード例3: Jetson での YOLOv8 実行

```python
from ultralytics import YOLO
import cv2

# YOLOv8 モデルをロード
model = YOLO("yolov8n.pt")

# TensorRT にエクスポート（Jetson上で実行）
model.export(format="engine", half=True, device=0)

# TensorRT エンジンで推論
model_trt = YOLO("yolov8n.engine")

# カメラからリアルタイム推論
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 推論実行
    results = model_trt(frame, stream=True, verbose=False)

    for result in results:
        annotated = result.plot()
        cv2.imshow("YOLOv8 on Jetson", annotated)

        # 検出結果の詳細
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = result.names[cls]
            print(f"{label}: {conf:.2f}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### コード例4: Jetson でのマルチストリーム推論

```python
import threading
from queue import Queue
import cv2
import numpy as np

class MultiStreamInference:
    """複数カメラからの同時推論パイプライン"""

    def __init__(self, model_path, num_streams=4):
        self.model = load_tensorrt_engine(model_path)
        self.num_streams = num_streams
        self.result_queues = [Queue(maxsize=10) for _ in range(num_streams)]
        self.running = True

    def capture_and_infer(self, stream_id, source):
        """各ストリームのキャプチャ→推論ループ"""
        cap = cv2.VideoCapture(source)
        cuda_stream = cuda.Stream()  # 非同期CUDA Stream

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            # 前処理（リサイズ、正規化）
            input_tensor = preprocess(frame, target_size=(640, 640))

            # 非同期推論（CUDAストリームで並列実行）
            output = self.model.infer_async(
                input_tensor,
                stream=cuda_stream
            )
            cuda_stream.synchronize()

            # 後処理（NMS、バウンディングボックス描画）
            detections = postprocess(output, conf_threshold=0.5)

            self.result_queues[stream_id].put({
                'stream_id': stream_id,
                'frame': frame,
                'detections': detections,
            })

        cap.release()

    def start(self, sources):
        """全ストリームを起動"""
        threads = []
        for i, source in enumerate(sources):
            t = threading.Thread(
                target=self.capture_and_infer,
                args=(i, source)
            )
            t.daemon = True
            t.start()
            threads.append(t)
        return threads

# 使用例: 4台のカメラから同時推論
pipeline = MultiStreamInference("yolov8n.engine", num_streams=4)
sources = [
    "rtsp://192.168.1.101/stream",
    "rtsp://192.168.1.102/stream",
    "rtsp://192.168.1.103/stream",
    "rtsp://192.168.1.104/stream",
]
pipeline.start(sources)
```

---

## 4. Google Coral

### コード例5: Coral Edge TPU での推論

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

### Coral Edge TPU のセットアップ手順

```bash
# Coral USB Accelerator の場合

# 1. Edge TPU ランタイムのインストール
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
    sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    sudo apt-key add -
sudo apt-get update
sudo apt-get install libedgetpu1-std  # 標準クロック
# sudo apt-get install libedgetpu1-max  # 最大クロック（発熱注意）

# 2. PyCoral のインストール
pip3 install pycoral

# 3. モデルのダウンロード
wget https://github.com/google-coral/test_data/raw/master/\
mobilenet_v2_1.0_224_quant_edgetpu.tflite

# 4. Edge TPU Compiler のインストール
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
    sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
sudo apt-get install edgetpu-compiler

# 5. カスタムモデルのコンパイル
edgetpu_compiler my_model_quant.tflite
# → my_model_quant_edgetpu.tflite が生成される
```

### コード例6: Coral での物体検出パイプライン

```python
from pycoral.adapters import detect
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from PIL import Image, ImageDraw
import time

class CoralObjectDetector:
    """Coral Edge TPU を使った物体検出"""

    def __init__(self, model_path, labels_path, threshold=0.5):
        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.labels = read_label_file(labels_path)
        self.threshold = threshold
        self.input_size = common.input_size(self.interpreter)

    def detect(self, image):
        """画像から物体を検出"""
        # リサイズ
        resized = image.resize(self.input_size, Image.LANCZOS)
        common.set_input(self.interpreter, resized)

        # 推論
        start = time.perf_counter()
        self.interpreter.invoke()
        inference_time = time.perf_counter() - start

        # 結果取得
        objs = detect.get_objects(
            self.interpreter,
            score_threshold=self.threshold
        )

        results = []
        for obj in objs:
            bbox = obj.bbox
            # 元画像サイズにスケール
            scale_x = image.width / self.input_size[0]
            scale_y = image.height / self.input_size[1]

            results.append({
                'label': self.labels.get(obj.id, str(obj.id)),
                'score': float(obj.score),
                'bbox': (
                    int(bbox.xmin * scale_x),
                    int(bbox.ymin * scale_y),
                    int(bbox.xmax * scale_x),
                    int(bbox.ymax * scale_y),
                ),
            })

        return results, inference_time

    def draw_results(self, image, results):
        """検出結果を画像に描画"""
        draw = ImageDraw.Draw(image)
        for r in results:
            bbox = r['bbox']
            draw.rectangle(bbox, outline='red', width=2)
            draw.text(
                (bbox[0], bbox[1] - 15),
                f"{r['label']}: {r['score']:.2f}",
                fill='red'
            )
        return image

# 使用例
detector = CoralObjectDetector(
    "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
    "coco_labels.txt",
    threshold=0.5
)

image = Image.open("street.jpg")
results, time_ms = detector.detect(image)
print(f"推論時間: {time_ms*1000:.1f}ms, 検出数: {len(results)}")
for r in results:
    print(f"  {r['label']}: {r['score']:.2f} @ {r['bbox']}")
```

---

## 5. Apple Neural Engine / NPU搭載PC

### コード例7: Core ML でのオンデバイス推論

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

### コード例8: Core ML でのオンデバイスLLM推論

```swift
import CoreML

class OnDeviceLLM {
    let model: MLModel
    let tokenizer: BPETokenizer

    init(modelPath: String, tokenizerPath: String) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all  // CPU + GPU + Neural Engine

        self.model = try MLModel(contentsOf: URL(fileURLWithPath: modelPath),
                                  configuration: config)
        self.tokenizer = try BPETokenizer(mergesFile: tokenizerPath)
    }

    func generate(prompt: String, maxTokens: Int = 100) throws -> String {
        var tokens = tokenizer.encode(prompt)
        var generatedText = ""

        for _ in 0..<maxTokens {
            // 入力準備
            let inputArray = try MLMultiArray(shape: [1, NSNumber(value: tokens.count)],
                                             dataType: .int32)
            for (i, token) in tokens.enumerated() {
                inputArray[i] = NSNumber(value: token)
            }

            // 推論（Neural Engineで高速実行）
            let input = try MLDictionaryFeatureProvider(
                dictionary: ["input_ids": MLFeatureValue(multiArray: inputArray)]
            )
            let output = try model.prediction(from: input)

            // 次のトークンを取得
            guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
                break
            }
            let nextToken = argmax(logits)

            if nextToken == tokenizer.eosToken {
                break
            }

            tokens.append(nextToken)
            generatedText += tokenizer.decode([nextToken])
        }

        return generatedText
    }

    private func argmax(_ array: MLMultiArray) -> Int {
        var maxVal: Float = -Float.infinity
        var maxIdx = 0
        let lastDim = array.shape.last!.intValue
        let offset = (array.count - lastDim)

        for i in 0..<lastDim {
            let val = array[offset + i].floatValue
            if val > maxVal {
                maxVal = val
                maxIdx = i
            }
        }
        return maxIdx
    }
}
```

### コード例9: Qualcomm AI Engine (Snapdragon)

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

### Windows AI PC (NPU対応) でのモデル実行

```python
# ONNX Runtime で NPU を活用（Windows AI PC）
import onnxruntime as ort
import numpy as np

class WindowsNPUInference:
    """Windows AI PC の NPU を使った推論"""

    def __init__(self, model_path):
        # NPU (DML) を使用するセッション設定
        providers = [
            ('DmlExecutionProvider', {
                'device_id': 0,
                'enable_dynamic_graph_fusion': True,
            }),
            'CPUExecutionProvider',  # フォールバック
        ]

        self.session = ort.InferenceSession(
            model_path,
            providers=providers
        )

        # 入出力情報の取得
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_data):
        """NPU上で推論実行"""
        result = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )
        return result[0]

    @staticmethod
    def check_npu_availability():
        """NPU の利用可能性を確認"""
        providers = ort.get_available_providers()
        print(f"利用可能なプロバイダ: {providers}")
        if 'DmlExecutionProvider' in providers:
            print("DirectML (NPU/GPU) が利用可能")
            return True
        else:
            print("DirectML が利用不可 — CPUフォールバック")
            return False

# 使用例
WindowsNPUInference.check_npu_availability()
inferencer = WindowsNPUInference("model.onnx")
result = inferencer.predict(input_array)
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

### コード例10: Post-Training Quantization (PTQ)

```python
import torch
from torch.quantization import quantize_dynamic, quantize_static

# 動的量子化（推論時に量子化）
# 最も簡単な方法、精度劣化が少ない
model_fp32 = load_model("resnet50.pth")
model_int8 = quantize_dynamic(
    model_fp32,
    {torch.nn.Linear, torch.nn.Conv2d},  # 量子化対象レイヤー
    dtype=torch.qint8
)

# モデルサイズの比較
import os
torch.save(model_fp32.state_dict(), "model_fp32.pth")
torch.save(model_int8.state_dict(), "model_int8.pth")
size_fp32 = os.path.getsize("model_fp32.pth") / 1e6
size_int8 = os.path.getsize("model_int8.pth") / 1e6
print(f"FP32: {size_fp32:.1f}MB → INT8: {size_int8:.1f}MB ({size_int8/size_fp32*100:.0f}%)")
```

### コード例11: 知識蒸留

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """知識蒸留の損失関数"""

    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, labels):
        # ソフトターゲット損失（教師の知識を蒸留）
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # ハードターゲット損失（正解ラベルとの損失）
        hard_loss = F.cross_entropy(student_logits, labels)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

def train_with_distillation(teacher, student, train_loader, epochs=50):
    """教師モデルの知識を生徒モデルに蒸留"""
    teacher.eval()  # 教師は推論モードで固定
    student.train()

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    criterion = DistillationLoss(temperature=4.0, alpha=0.5)

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 教師の推論（勾配計算不要）
            with torch.no_grad():
                teacher_logits = teacher(images)

            # 生徒の推論
            student_logits = student(images)

            # 蒸留損失
            loss = criterion(student_logits, teacher_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss = {total_loss / len(train_loader):.4f}")

# 使用例
# 教師: ResNet-50 (25.6M params, 97.8MB)
# 生徒: MobileNet-V3 (5.4M params, 21.8MB)
teacher = torchvision.models.resnet50(pretrained=True)
student = torchvision.models.mobilenet_v3_small(pretrained=False)
train_with_distillation(teacher, student, train_loader)
```

### コード例12: 構造化プルーニング

```python
import torch
import torch.nn.utils.prune as prune

def structured_pruning(model, amount=0.3):
    """構造化プルーニング（チャネル単位で削除）"""

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # L1ノルムベースでチャネルをプルーニング
            prune.ln_structured(
                module,
                name='weight',
                amount=amount,
                n=1,  # L1ノルム
                dim=0  # 出力チャネル次元
            )
            # プルーニングを永続化
            prune.remove(module, 'weight')

    return model

def evaluate_pruned_model(model, test_loader):
    """プルーニング後のモデル評価"""
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    zero_params = sum((p == 0).sum().item() for p in model.parameters())
    sparsity = zero_params / total_params

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.to(device)).sum().item()

    accuracy = correct / total
    print(f"スパース率: {sparsity*100:.1f}%")
    print(f"精度: {accuracy*100:.1f}%")
    print(f"パラメータ数: {total_params:,} (ゼロ: {zero_params:,})")

    return accuracy, sparsity
```

### ONNX によるクロスプラットフォーム変換

```python
import torch
import onnx
import onnxruntime as ort

def export_to_onnx(model, input_shape, output_path):
    """PyTorch モデルを ONNX 形式にエクスポート"""
    model.eval()
    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
    )

    # モデルの検証
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX エクスポート完了: {output_path}")

def optimize_onnx(input_path, output_path):
    """ONNX モデルの最適化"""
    import onnxoptimizer

    model = onnx.load(input_path)
    optimized = onnxoptimizer.optimize(model, [
        'eliminate_deadend',
        'eliminate_identity',
        'eliminate_nop_dropout',
        'fuse_bn_into_conv',
        'fuse_consecutive_transposes',
    ])
    onnx.save(optimized, output_path)
    print(f"最適化完了: {output_path}")

def benchmark_onnx(model_path, input_shape, num_runs=100):
    """ONNX Runtime での推論速度ベンチマーク"""
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    dummy = np.random.randn(*input_shape).astype(np.float32)

    # ウォームアップ
    for _ in range(10):
        session.run(None, {input_name: dummy})

    # ベンチマーク
    import time
    start = time.perf_counter()
    for _ in range(num_runs):
        session.run(None, {input_name: dummy})
    elapsed = (time.perf_counter() - start) / num_runs

    print(f"推論時間: {elapsed*1000:.2f}ms ({1/elapsed:.0f} FPS)")
```

---

## 7. エッジAIの実践ユースケース

### ユースケース別推奨構成

| ユースケース | 推奨デバイス | モデル | 性能目安 |
|-------------|------------|--------|---------|
| スマートカメラ（人検出） | Coral USB + RPi | SSD MobileNet (INT8) | 30 FPS, 2W |
| 製造ライン検品 | Jetson Orin NX | YOLOv8m (FP16) | 60 FPS, 25W |
| 自動運転L2 | Jetson AGX Orin | 複数モデル同時 | 30 FPS, 60W |
| スマホ写真加工 | Apple Neural Engine | Core ML最適化モデル | リアルタイム, 1W |
| 音声コマンド | Coral + マイク | ウェイクワード検出 (INT8) | <10ms, 0.5W |
| ドローン追跡 | Jetson Orin Nano | YOLOv8n (FP16) | 60 FPS, 15W |

### コード例13: エッジAIでの異常検知

```python
import numpy as np
from sklearn.ensemble import IsolationForest
import onnxruntime as ort

class EdgeAnomalyDetector:
    """エッジデバイスでの製造ライン異常検知"""

    def __init__(self, model_path, threshold=-0.5):
        self.session = ort.InferenceSession(model_path)
        self.threshold = threshold
        self.feature_buffer = []
        self.alert_callback = None

    def extract_features(self, image):
        """画像から特徴量を抽出"""
        input_name = self.session.get_inputs()[0].name
        preprocessed = self.preprocess(image)
        features = self.session.run(None, {input_name: preprocessed})[0]
        return features.flatten()

    def detect(self, image):
        """異常検知"""
        features = self.extract_features(image)

        # 統計的異常検知（エッジで軽量に動作）
        self.feature_buffer.append(features)
        if len(self.feature_buffer) < 100:
            return {'status': 'collecting', 'samples': len(self.feature_buffer)}

        # Isolation Forestで異常スコア計算
        if len(self.feature_buffer) == 100:
            self.model = IsolationForest(contamination=0.05)
            self.model.fit(np.array(self.feature_buffer))

        score = self.model.decision_function([features])[0]
        is_anomaly = score < self.threshold

        if is_anomaly and self.alert_callback:
            self.alert_callback(image, score)

        return {
            'status': 'anomaly' if is_anomaly else 'normal',
            'score': float(score),
            'threshold': self.threshold,
        }

    def preprocess(self, image):
        """画像前処理"""
        resized = cv2.resize(image, (224, 224))
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized.transpose(2, 0, 1), 0)
```

---

## 8. アンチパターン

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

### アンチパターン3: 熱管理を無視した連続運転

```
NG: Jetson を冷却なしで24/7運用
    → サーマルスロットリングで性能50%低下
    → 最悪の場合、デバイス損傷

OK: 適切な熱管理
    1. ヒートシンク + ファンの装着（必須）
    2. 動作温度のモニタリング（jtopで監視）
    3. パワーモード設定
       ・最大性能: 60W（冷却十分な場合のみ）
       ・バランス: 30W（通常運用）
       ・省電力: 15W（バッテリー/小型筐体）
    4. 筐体の通気設計
    5. 周囲温度が40度以上ならデレーティング適用
```

### アンチパターン4: OTA更新の仕組みなしでデプロイ

```
NG: モデルをデバイスに焼き込んで設置、更新手段なし
    → バグ修正もモデル改善も現地作業が必要

OK: OTA（Over-The-Air）更新の仕組みを構築
    1. A/Bパーティション方式でのモデル更新
    2. 更新失敗時のロールバック機構
    3. モデルバージョン管理とメタデータ
    4. 帯域制限環境での差分更新
    5. 更新前後の精度検証テスト
```

---

## 9. トラブルシューティング

### よくある問題と解決策

| 問題 | 原因 | 解決策 |
|------|------|--------|
| OOM (Out of Memory) | モデルが大きすぎる | 量子化、モデル軽量化、バッチサイズ=1 |
| 推論が遅い | FP32のまま、最適化なし | TensorRT/ONNX Runtime変換、FP16化 |
| サーマルスロットリング | 冷却不足 | ヒートシンク/ファン追加、パワーモード変更 |
| Edge TPU で一部レイヤーがCPU実行 | 未対応オペレーション | モデル構造変更、対応オペのみ使用 |
| NPU使用率が低い | フレームワークの設定不備 | compute_units設定確認、プロファイリング |
| モデル精度の劣化 | 量子化による精度低下 | QAT適用、キャリブレーションデータの改善 |

---

## FAQ

### Q1. Raspberry Pi でAI推論は実用的か？

Raspberry Pi 5 は CPU 推論で MobileNet 程度なら 30-50ms（20-30 FPS）で動作する。Coral USB Accelerator を接続すると Edge TPU で 5-10ms に高速化される。リアルタイム性が不要な用途（定期的な画像分類、音声コマンド認識）では十分実用的。

### Q2. エッジデバイスでLLMは動くか？

2025年時点で、Jetson AGX Orin（64GB）なら Llama 3 8B の4bit量子化版が 10-20 tokens/sec で動作する。スマートフォンでは Phi-3 Mini (3.8B) や Gemma 2B が実用的。Apple M4 搭載MacならLlama 3 70B の4bit量子化も可能。

### Q3. エッジAI開発の入門にはどのデバイスが良い？

予算1万円以下なら Coral USB Accelerator + Raspberry Pi。予算5万円以下なら Jetson Orin Nano。Apple デバイスを持っているなら Core ML + Create ML で追加投資なしに始められる。

### Q4. TensorRT と ONNX Runtime、どちらを使うべきか？

NVIDIA GPU（Jetson含む）限定ならTensorRTが最速。クロスプラットフォーム（CPU、GPU、NPU）で動かしたいならONNX Runtime。両方試してベンチマークを取るのが最善。TensorRTはONNX形式の入力も受け付けるため、まずONNXにエクスポートしてからTensorRT変換が一般的なフロー。

### Q5. エッジAIとクラウドAI、どちらを選ぶべきか？

判断基準: 1) レイテンシ要件（10ms以下ならエッジ一択）、2) プライバシー要件（医療・金融データはエッジ推奨）、3) インターネット接続の安定性（不安定ならエッジ）、4) モデルの複雑さ（70B+パラメータはクラウド）、5) コスト構造（大量推論ならエッジが安い、散発的ならクラウドが安い）。実際には両方を組み合わせたハイブリッドアーキテクチャが最適解になることが多い。

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
| 知識蒸留 | 大規模モデルの知識を軽量モデルに転移 |
| ONNX | クロスプラットフォームモデル交換フォーマット |

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
5. **ONNX Runtime** https://onnxruntime.ai/docs/
6. **TensorRT Developer Guide** https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/
