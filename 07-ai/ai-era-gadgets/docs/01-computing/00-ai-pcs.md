# AI PC — NPU搭載PC、Copilot+ PC、ローカルLLM、Snapdragon X

> AI PC の定義と技術要件を解説する。NPU搭載プロセッサ、Microsoft Copilot+ PC 仕様、Snapdragon X Elite / Intel Core Ultra / AMD Ryzen AI の比較、そしてローカルLLMの実行方法まで網羅する。

---

## この章で学ぶこと

1. **AI PCの定義と要件** — NPU 40+ TOPS、Copilot+ PC認定条件、Windows AI機能
2. **主要プロセッサのNPU比較** — Snapdragon X / Intel Core Ultra / AMD Ryzen AI
3. **ローカルLLMの実行** — Ollama / LM Studio / llama.cpp によるオンデバイス推論

---

## 1. AI PCのアーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                    AI PC アーキテクチャ                        │
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │    CPU       │  │    GPU       │  │       NPU            │  │
│  │ (汎用処理)  │  │ (グラフィック│  │ (AI推論専用)          │  │
│  │ P+Eコア     │  │  + AI学習)  │  │ INT8/INT4最適化       │  │
│  │ ~100 TOPS   │  │ ~100 TOPS   │  │ 40〜75 TOPS          │  │
│  │ (FP32)      │  │ (FP16)      │  │ (INT8)               │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│          │                │                    │              │
│          └────────────────┼────────────────────┘              │
│                           │                                   │
│                   ┌───────▼──────┐                            │
│                   │ Windows ML / │                            │
│                   │ DirectML /   │                            │
│                   │ ONNX Runtime │                            │
│                   └──────────────┘                            │
│                           │                                   │
│                   ┌───────▼──────┐                            │
│                   │ Copilot+     │                            │
│                   │ AI機能群     │                            │
│                   │ (Recall等)  │                            │
│                   └──────────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

### 1.1 NPU推論パイプライン

```
┌─────────────────────────────────────────────┐
│          NPU推論パイプライン                   │
│                                               │
│  入力データ                                   │
│    │                                          │
│    ▼                                          │
│  ┌──────────┐                                │
│  │ 量子化済み │  INT8/INT4モデル               │
│  │ モデル    │  (ONNX形式)                    │
│  └──────────┘                                │
│    │                                          │
│    ▼                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ MAC Array│→ │ Activation│→ │ Output   │   │
│  │ (行列積) │  │ (活性化)  │  │ (後処理) │   │
│  │ INT8×INT8│  │ ReLU/GELU │  │ Softmax  │   │
│  └──────────┘  └──────────┘  └──────────┘   │
│                                    │          │
│                                    ▼          │
│                              推論結果          │
│                              (3〜10ms)        │
└─────────────────────────────────────────────┘
```

### 1.2 AI PC のソフトウェアスタック詳細

```
┌──────────────────────────────────────────────────────┐
│              AI PC ソフトウェアスタック                  │
│                                                        │
│  ┌────────────────────────────────────────────────┐   │
│  │ アプリケーション層                                │   │
│  │ Copilot, Adobe Creative Suite, DaVinci Resolve  │   │
│  │ Visual Studio (IntelliCode), ブラウザ AI機能      │   │
│  └───────────────────────┬────────────────────────┘   │
│                           │                            │
│  ┌───────────────────────▼────────────────────────┐   │
│  │ AI フレームワーク層                               │   │
│  │ ONNX Runtime | DirectML | OpenVINO | Core ML    │   │
│  │ PyTorch | TensorFlow | llama.cpp                │   │
│  └───────────────────────┬────────────────────────┘   │
│                           │                            │
│  ┌───────────────────────▼────────────────────────┐   │
│  │ API / ランタイム層                                │   │
│  │ Windows ML | NNAPI | Vulkan Compute              │   │
│  │ CUDA (NVIDIA) | ROCm (AMD) | oneAPI (Intel)     │   │
│  └───────────────────────┬────────────────────────┘   │
│                           │                            │
│  ┌───────────────────────▼────────────────────────┐   │
│  │ ドライバ / ハードウェア抽象化層                     │   │
│  │ NPU Driver | GPU Driver | CPU Microcode          │   │
│  └───────────────────────┬────────────────────────┘   │
│                           │                            │
│  ┌───────────────────────▼────────────────────────┐   │
│  │ ハードウェア層                                    │   │
│  │ NPU (Hexagon/AI Boost/XDNA) | GPU | CPU         │   │
│  └────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

---

## 2. コード例

### コード例 1: Windows ML でNPU推論

```csharp
using Microsoft.AI.MachineLearning;

// ONNX モデルをNPUで実行
async Task RunOnNPU()
{
    // モデルの読み込み
    var model = await LearningModel.LoadFromFilePath("model.onnx");

    // NPU デバイスを指定
    var device = new LearningModelDevice(
        LearningModelDeviceKind.DirectXHighPerformance // NPU優先
    );
    var session = new LearningModelSession(model, device);

    // 入力データの準備
    var binding = new LearningModelBinding(session);
    var inputTensor = TensorFloat.CreateFromArray(
        new long[] { 1, 3, 224, 224 },
        imageData
    );
    binding.Bind("input", inputTensor);

    // 推論実行
    var result = await session.EvaluateAsync(binding, "inference");
    var output = result.Outputs["output"] as TensorFloat;

    Console.WriteLine($"推論完了: {output.GetAsVectorView()[0]}");
}
```

### コード例 2: Ollama でローカルLLM実行

```bash
# Ollama のインストールと実行

# モデルのダウンロードと実行
ollama pull llama3.1:8b        # Llama 3.1 8B (4.7GB)
ollama pull gemma2:9b          # Gemma 2 9B (5.4GB)
ollama pull phi3:mini           # Phi-3 Mini 3.8B (2.3GB)

# 対話開始
ollama run llama3.1:8b

# API経由で利用（REST API）
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Pythonでクイックソートを実装してください",
  "stream": false
}'
```

```python
# Python からOllamaを利用
import ollama

# チャット
response = ollama.chat(model='llama3.1:8b', messages=[
    {'role': 'system', 'content': '日本語で簡潔に回答してください。'},
    {'role': 'user', 'content': 'AI PCのNPUとは何ですか？'}
])
print(response['message']['content'])

# ストリーミング
for chunk in ollama.chat(
    model='llama3.1:8b',
    messages=[{'role': 'user', 'content': 'Hello'}],
    stream=True
):
    print(chunk['message']['content'], end='', flush=True)
```

### コード例 3: llama.cpp でNPU/GPU活用

```bash
# llama.cpp のビルド（Vulkan GPU対応）
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_VULKAN=ON  # GPUバックエンド
cmake --build build --config Release

# 量子化モデルの実行（Q4_K_M: 品質とサイズのバランス）
./build/bin/llama-cli \
  -m models/llama-3.1-8b-q4_k_m.gguf \
  -p "AI PCの利点を3つ挙げてください：" \
  -n 256 \
  --n-gpu-layers 33 \    # GPU層数
  --threads 8 \           # CPUスレッド数
  --temp 0.7
```

```python
# llama-cpp-python で利用
from llama_cpp import Llama

llm = Llama(
    model_path="models/llama-3.1-8b-q4_k_m.gguf",
    n_ctx=4096,        # コンテキスト長
    n_gpu_layers=33,   # GPU層（-1で全層GPU）
    n_threads=8,       # CPUスレッド数
    verbose=False
)

output = llm(
    "AI PCのNPUについて説明してください。",
    max_tokens=256,
    temperature=0.7,
    stop=["。\n\n"]
)
print(output['choices'][0]['text'])
# トークン/秒を確認（NPU/GPU活用時は20-40 tok/s）
```

### コード例 4: ONNX Runtime でクロスプラットフォームNPU推論

```python
import onnxruntime as ort
import numpy as np

# 利用可能なプロバイダを確認
print("利用可能:", ort.get_available_providers())
# → ['DmlExecutionProvider', 'QNNExecutionProvider', 'CPUExecutionProvider']

# NPU (QNN) > GPU (DML) > CPU の優先順位で実行
providers = ['QNNExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']

session = ort.InferenceSession(
    "model_int8.onnx",
    providers=providers,
    provider_options=[
        {'backend_path': 'QnnHtp.dll'},  # NPU (Qualcomm HTP)
        {},                               # GPU (DirectML)
        {}                                # CPU
    ]
)

# 推論
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
result = session.run(None, {"input": input_data})

# 実行プロバイダを確認
active = session.get_providers()
print(f"使用中: {active[0]}")  # → 'QNNExecutionProvider'（NPU）
```

### コード例 5: Copilot+ PC の Phi Silica（Windows AI API）

```python
# Windows AI Foundation Model API（Windows 11 24H2以降）
import windowsai

async def use_phi_silica():
    """Copilot+ PC 内蔵の Phi Silica モデルを使用"""

    # オンデバイスLLM（Phi Silica）にアクセス
    model = await windowsai.LanguageModel.create_async()

    # テキスト生成
    result = await model.generate_async(
        prompt="この会議メモを要約してください：...",
        max_tokens=200
    )
    print(result.text)

    # ストリーミング生成
    async for token in model.generate_stream_async(
        prompt="メールの返信案を書いてください",
        max_tokens=300
    ):
        print(token, end="", flush=True)

# ※ Copilot+ PC (NPU 40+ TOPS) でのみ実行可能
```

### コード例 6: OpenVINO でIntel NPU推論

```python
import openvino as ov
import numpy as np

def run_on_intel_npu(model_path: str, input_data: np.ndarray):
    """Intel Core Ultra の NPU (AI Boost) で推論"""

    core = ov.Core()

    # 利用可能なデバイスを確認
    devices = core.available_devices
    print(f"利用可能デバイス: {devices}")
    # → ['CPU', 'GPU', 'NPU']

    # モデルの読み込みとコンパイル
    model = core.read_model(model_path)

    # NPU向けに最適化してコンパイル
    compiled_model = core.compile_model(
        model,
        device_name="NPU",
        config={
            "NPU_COMPILER_TYPE": "DRIVER",
            "PERFORMANCE_HINT": "LATENCY",
        }
    )

    # 推論リクエスト
    infer_request = compiled_model.create_infer_request()
    infer_request.set_input_tensor(ov.Tensor(input_data))

    # 推論実行と計測
    import time
    start = time.perf_counter()
    infer_request.infer()
    elapsed = (time.perf_counter() - start) * 1000

    output = infer_request.get_output_tensor().data
    print(f"推論時間: {elapsed:.2f}ms (NPU)")

    return output

# INT8量子化モデルで推論
input_image = np.random.randn(1, 3, 224, 224).astype(np.float32)
result = run_on_intel_npu("mobilenet_v3_int8.xml", input_image)
```

### コード例 7: Ryzen AI (XDNA NPU) でのモデル実行

```python
# AMD Ryzen AI Software を使ったNPU推論
import vitis_ai_runtime as vai

def run_on_ryzen_ai(model_path: str, input_data):
    """AMD Ryzen AI (XDNA 2 NPU) で推論"""

    # Vitis AI ランタイムの初期化
    runner = vai.Runner(model_path)

    # 入力テンソルの準備
    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()

    # NPUで推論実行
    job_id = runner.execute_async(input_data, output_tensors)
    runner.wait(job_id)

    return output_tensors

# ONNX → Vitis AI 変換フロー
# 1. ONNX モデルを準備
# 2. Vitis AI Quantizer でINT8量子化
# 3. Vitis AI Compiler でXDNA向けにコンパイル
# 4. ランタイムで実行

# 変換コマンド例:
# vai_q_onnx quantize --model model.onnx \
#     --output_model model_int8.onnx \
#     --calibration_data_reader calibration_data
```

### コード例 8: ローカルLLMベンチマークスクリプト

```python
import time
import subprocess
import json
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    model_name: str
    quantization: str
    tokens_per_second: float
    time_to_first_token: float
    memory_usage_gb: float
    backend: str

def benchmark_ollama_model(model_name: str, prompt: str, num_runs: int = 5):
    """Ollama モデルの性能ベンチマーク"""
    results = []

    for i in range(num_runs):
        start = time.perf_counter()

        # Ollama API を呼び出し
        response = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/generate",
             "-d", json.dumps({
                 "model": model_name,
                 "prompt": prompt,
                 "stream": False,
                 "options": {"num_predict": 128}
             })],
            capture_output=True, text=True
        )

        elapsed = time.perf_counter() - start
        data = json.loads(response.stdout)

        tokens = data.get("eval_count", 0)
        eval_duration = data.get("eval_duration", 1) / 1e9  # ns → s
        prompt_eval_duration = data.get("prompt_eval_duration", 0) / 1e9

        results.append({
            "tokens": tokens,
            "tokens_per_sec": tokens / eval_duration if eval_duration > 0 else 0,
            "ttft": prompt_eval_duration,
            "total_time": elapsed,
        })

    # 平均値を計算
    avg_tps = sum(r["tokens_per_sec"] for r in results) / len(results)
    avg_ttft = sum(r["ttft"] for r in results) / len(results)

    print(f"=== {model_name} ベンチマーク結果 ===")
    print(f"  平均生成速度: {avg_tps:.1f} tokens/sec")
    print(f"  平均TTFT: {avg_ttft*1000:.0f}ms")
    print(f"  実行回数: {num_runs}")

    return BenchmarkResult(
        model_name=model_name,
        quantization="Q4_K_M",
        tokens_per_second=avg_tps,
        time_to_first_token=avg_ttft,
        memory_usage_gb=0,
        backend="ollama"
    )

# 複数モデルのベンチマーク
models = ["phi3:mini", "llama3.1:8b", "gemma2:9b", "mistral:7b"]
prompt = "Explain the concept of NPU in AI PCs in 3 sentences."

for model in models:
    benchmark_ollama_model(model, prompt)
```

---

## 3. 比較表

### 比較表 1: 主要AI PCプロセッサ比較

| 項目 | Snapdragon X Elite | Intel Core Ultra 200V | AMD Ryzen AI 300 |
|------|-------------------|---------------------|-----------------|
| NPU性能 | 45 TOPS | 48 TOPS | 50 TOPS |
| NPU名称 | Hexagon NPU | Intel AI Boost (NPU4) | XDNA 2 (Ryzen AI) |
| CPU | Oryon 12コア | P+Eコア (16スレッド) | Zen 5 (16スレッド) |
| GPU | Adreno X1-85 | Intel Arc (Xe2) | Radeon 890M |
| 電力効率 | 優秀（ARM） | 良好 | 良好 |
| Copilot+ PC | 対応 | 対応 | 対応 |
| ローカルLLM | 7B〜13Bモデル | 7B〜13Bモデル | 7B〜13Bモデル |
| 対応OS | Windows 11 (ARM) | Windows 11 | Windows 11 |

### 比較表 2: ローカルLLM実行ツール比較

| ツール | GUI | API | 量子化対応 | NPU対応 | マルチモデル | 難易度 |
|--------|-----|-----|-----------|---------|------------|--------|
| Ollama | なし（CLI） | REST API | GGUF (Q4/Q5/Q8) | 限定的 | 同時実行可 | 低 |
| LM Studio | あり | OpenAI互換 | GGUF全般 | 限定的 | 切り替え式 | 低 |
| llama.cpp | なし（CLI） | HTTP Server | GGUF全般 | Vulkan/CUDA | 単一 | 中 |
| vLLM | なし | OpenAI互換 | AWQ/GPTQ | CUDA | バッチ推論 | 高 |
| GPT4All | あり | Python API | GGUF | 限定的 | 切り替え式 | 低 |

### 比較表 3: GGUF量子化レベルと品質

| 量子化レベル | ビット幅 | 7Bモデルサイズ | 品質（Perplexity） | 推論速度 | 推奨用途 |
|------------|---------|-------------|------------------|---------|---------|
| Q2_K | 2-3bit | ~2.8GB | 低（損失大） | 最速 | 速度最優先・テスト用 |
| Q3_K_M | 3bit | ~3.3GB | やや低い | 速い | メモリ制約が厳しい場合 |
| Q4_K_M | 4bit | ~4.1GB | 良好 | 速い | 最も推奨（バランス良） |
| Q5_K_M | 5bit | ~4.8GB | 高い | 中程度 | 品質重視 |
| Q6_K | 6bit | ~5.5GB | 非常に高い | やや遅い | 高品質推論 |
| Q8_0 | 8bit | ~7.2GB | 最高 | 遅い | 品質最優先 |
| FP16 | 16bit | ~14GB | 基準 | 最遅 | 研究・評価用 |

### 比較表 4: Copilot+ PC AI機能一覧

| 機能 | 説明 | 必要NPU | 処理場所 | 対応バージョン |
|------|------|---------|---------|-------------|
| Recall | 画面履歴のAI検索 | 40+ TOPS | NPU（オンデバイス） | Windows 11 24H2 |
| Live Captions | リアルタイム翻訳字幕 | 40+ TOPS | NPU | Windows 11 24H2 |
| Image Creator | テキストから画像生成 | 40+ TOPS | NPU | Windows 11 24H2 |
| Cocreator (Paint) | AIアシスト描画 | 40+ TOPS | NPU | Windows 11 24H2 |
| Windows Studio Effects | 背景ぼかし、視線補正 | 10+ TOPS | NPU | Windows 11 23H2+ |
| Copilot | AIアシスタント | 不要 | クラウド | Windows 11全般 |

---

## 4. 実践的なユースケースと応用例

### ユースケース 1: ローカルRAG（検索拡張生成）システム

```python
import ollama
import chromadb
from sentence_transformers import SentenceTransformer

class LocalRAGSystem:
    """
    AI PC上で完全ローカルに動作するRAGシステム
    プライバシー保護: データが端末外に出ない
    """

    def __init__(self):
        # 埋め込みモデル（ローカル実行）
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        # ベクトルDB（ローカル永続化）
        self.chroma = chromadb.PersistentClient(path="./local_vectordb")
        self.collection = self.chroma.get_or_create_collection("documents")

    def add_document(self, text: str, metadata: dict = None):
        """ドキュメントをベクトルDBに追加"""
        embedding = self.embedder.encode(text).tolist()
        doc_id = f"doc_{self.collection.count()}"
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[metadata or {}]
        )
        print(f"ドキュメント追加: {doc_id}")

    def query(self, question: str, n_results: int = 3) -> str:
        """質問に対してRAG応答を生成"""
        # 類似ドキュメントを検索
        query_embedding = self.embedder.encode(question).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        context = "\n\n".join(results['documents'][0])

        # ローカルLLMで応答生成
        response = ollama.chat(
            model='llama3.1:8b',
            messages=[
                {'role': 'system',
                 'content': f'以下のコンテキストに基づいて回答してください。\n\nコンテキスト:\n{context}'},
                {'role': 'user', 'content': question}
            ]
        )
        return response['message']['content']

# 使用例
rag = LocalRAGSystem()
rag.add_document("AI PCにはNPUが搭載され、40 TOPS以上の性能を持つ...")
rag.add_document("Copilot+ PCはMicrosoftが定義したAI PC規格...")
answer = rag.query("Copilot+ PCの要件は？")
print(answer)
```

### ユースケース 2: ローカルコード補完サーバー

```python
from llama_cpp import Llama
from flask import Flask, request, jsonify

app = Flask(__name__)

# コード補完専用モデルをロード
llm = Llama(
    model_path="models/codellama-7b-instruct-q4_k_m.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,  # 全層GPU
    n_threads=8,
)

@app.route('/v1/completions', methods=['POST'])
def code_completion():
    """VS Code / JetBrains 互換のコード補完API"""
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 128)

    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.2,  # コード補完は低温推奨
        stop=["\n\n", "```", "def ", "class "],
    )

    return jsonify({
        "choices": [{
            "text": output['choices'][0]['text'],
            "finish_reason": output['choices'][0]['finish_reason']
        }],
        "usage": output['usage']
    })

# VS Code の Continue 拡張機能と連携可能
# config.json: {"model": "codellama", "apiBase": "http://localhost:5000"}
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### ユースケース 3: NPUを活用したリアルタイム映像処理

```python
import onnxruntime as ort
import cv2
import numpy as np
import time

class NPUVideoProcessor:
    """NPUでリアルタイム映像処理（背景ぼかし、物体検出）"""

    def __init__(self):
        # 背景セグメンテーションモデル（NPU実行）
        self.seg_session = ort.InferenceSession(
            "selfie_segmentation_int8.onnx",
            providers=['QNNExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
        )

        # 物体検出モデル（NPU実行）
        self.det_session = ort.InferenceSession(
            "yolov8n_int8.onnx",
            providers=['QNNExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
        )

        print(f"セグメンテーション: {self.seg_session.get_providers()[0]}")
        print(f"物体検出: {self.det_session.get_providers()[0]}")

    def process_frame(self, frame):
        """1フレームを処理（背景ぼかし + 物体検出）"""
        # 背景セグメンテーション
        input_seg = cv2.resize(frame, (256, 256))
        input_seg = input_seg.astype(np.float32) / 255.0
        input_seg = np.transpose(input_seg, (2, 0, 1))[np.newaxis]

        mask = self.seg_session.run(None, {"input": input_seg})[0]
        mask = cv2.resize(mask[0, 0], (frame.shape[1], frame.shape[0]))

        # 背景ぼかし適用
        blurred = cv2.GaussianBlur(frame, (21, 21), 0)
        mask_3d = np.stack([mask] * 3, axis=-1)
        result = (frame * mask_3d + blurred * (1 - mask_3d)).astype(np.uint8)

        return result

    def run_camera(self):
        """カメラ入力でリアルタイム処理"""
        cap = cv2.VideoCapture(0)
        fps_counter = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start = time.perf_counter()
            result = self.process_frame(frame)
            elapsed = time.perf_counter() - start

            fps_counter.append(1.0 / elapsed)
            if len(fps_counter) > 30:
                fps_counter.pop(0)
            avg_fps = sum(fps_counter) / len(fps_counter)

            cv2.putText(result, f"FPS: {avg_fps:.1f} (NPU)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("AI PC NPU Demo", result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

# NPU実行時: 30+ FPS (背景ぼかし + 物体検出)
# CPU実行時: 5-10 FPS
```

---

## 5. トラブルシューティングガイド

### 問題 1: Ollama で「out of memory」エラー

```
症状: ollama run llama3.1:8b が "out of memory" でクラッシュ

診断手順:
1. 利用可能メモリの確認
   $ free -h (Linux) / タスクマネージャー (Windows)
   → 最低でもモデルサイズ + 2GB の空きRAMが必要

2. 量子化レベルの変更
   $ ollama pull llama3.1:8b    # Q4_K_M (4.7GB) → RAM 8GB+
   $ ollama pull phi3:mini       # (2.3GB) → RAM 6GB+
   $ ollama pull gemma2:2b       # (1.6GB) → RAM 4GB+

3. GPU VRAM の確認（GPU オフロード時）
   $ nvidia-smi  # NVIDIA
   → GPUメモリ不足の場合、n_gpu_layers を下げる

4. スワップの確認
   → スワップ使用量が増えている場合、推論速度が大幅低下
   → モデルを小さくするか、RAMを増設
```

### 問題 2: NPUが認識されない

```
症状: ONNX Runtime で QNNExecutionProvider が表示されない

対処法:
1. NPUドライバの確認
   → デバイスマネージャー > ニューラルプロセッサ
   → ドライバが最新版か確認

2. ONNX Runtime バージョンの確認
   → NPU対応は ort 1.17+ が必要
   $ pip install onnxruntime-qnn  # Qualcomm NPU向け
   $ pip install onnxruntime-directml  # DirectML (GPU + NPU)

3. Windows バージョンの確認
   → NPU完全対応は Windows 11 24H2 以降
   → 設定 > システム > バージョン情報で確認

4. モデルの互換性確認
   → NPUはINT8量子化モデルのみ対応
   → FP32/FP16モデルはGPU/CPUにフォールバック
```

### 問題 3: ローカルLLMの推論速度が遅い

```
症状: 期待した tokens/sec が出ない

チェックリスト:
□ GPU オフロードが有効か？
  → llama.cpp: --n-gpu-layers 33 (全層GPU)
  → Ollama: 自動検出だが、CUDA/ROCm が未インストールの場合CPU実行

□ コンテキスト長が長すぎないか？
  → n_ctx=4096 → 8192 にすると速度が約半分に
  → 必要最小限のコンテキスト長を設定

□ バッチサイズは適切か？
  → n_batch=512 がデフォルト。メモリに余裕があれば 1024 に

□ 量子化レベルは適切か？
  → Q4_K_M が速度と品質のベストバランス
  → Q2_K は最速だが品質低下が大きい

□ バックグラウンドプロセスがCPU/GPUを占有していないか？
  → ブラウザのGPUアクセラレーション等を一時無効化
```

---

## 6. パフォーマンス最適化Tips

### Tip 1: モデルサイズとRAMの関係

```
┌────────────────────────────────────────────────┐
│         RAM別 推奨モデルサイズ早見表             │
│                                                  │
│  RAM 8GB  ━━━━━━━━━━━━━━━━━━━                  │
│           Phi-3 Mini 3.8B (Q4) ✓                │
│           Gemma 2 2B (Q4) ✓                     │
│           Llama 3.1 8B (Q4) △ (ギリギリ)        │
│                                                  │
│  RAM 16GB ━━━━━━━━━━━━━━━━━━━━━━━━━━━━         │
│           Llama 3.1 8B (Q4) ✓                   │
│           Mistral 7B (Q4) ✓                     │
│           Gemma 2 9B (Q4) ✓                     │
│           CodeLlama 13B (Q4) △                   │
│                                                  │
│  RAM 32GB ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│           Llama 3.1 8B (Q8) ✓                   │
│           Mixtral 8x7B (Q4) ✓                   │
│           CodeLlama 34B (Q4) △                   │
│                                                  │
│  RAM 64GB ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
│           Llama 3.1 70B (Q4) ✓                  │
│           Qwen 72B (Q4) ✓                       │
│           Command R+ 104B (Q4) △                 │
│                                                  │
│  ✓ = 快適動作  △ = 動作するが遅い場合あり        │
└────────────────────────────────────────────────┘
```

### Tip 2: NPU vs GPU vs CPU の使い分け

| ワークロード | 推奨 | 理由 |
|------------|------|------|
| 常時背景ぼかし（ビデオ会議） | NPU | 省電力で常時実行 |
| LLM推論（チャット） | GPU | VRAM活用で高速 |
| バッチ画像分類 | NPU | INT8量子化で高効率 |
| 画像生成（Stable Diffusion） | GPU | FP16/BF16が必要 |
| 音声認識（リアルタイム） | NPU | 低遅延が重要 |
| コード補完 | GPU+CPU | 中程度の速度で十分 |
| RAG検索（埋め込み生成） | NPU | 小モデルの高速推論 |
| 動画編集AIエフェクト | GPU | 高スループットが必要 |

### Tip 3: Windows での AI開発環境構築

```bash
# 1. 基本環境
winget install Python.Python.3.12
winget install Git.Git
winget install Ollama.Ollama

# 2. CUDA (NVIDIA GPU の場合)
# https://developer.nvidia.com/cuda-downloads からインストール

# 3. Python AI パッケージ
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-directml  # DirectML (GPU + NPU)
pip install transformers accelerate
pip install llama-cpp-python  # ローカルLLM

# 4. Ollama モデルダウンロード
ollama pull llama3.1:8b
ollama pull codellama:7b-instruct
ollama pull nomic-embed-text  # 埋め込みモデル

# 5. 動作確認
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
curl http://localhost:11434/api/tags  # Ollamaモデル一覧
```

---

## 7. アンチパターン

### アンチパターン 1: NPU TOPSだけでAI性能を判断する

```
❌ 悪い例:
「50 TOPSのNPUだから何でも高速に動く」と思い込む
→ NPUはINT8推論に特化。FP32モデルはGPU/CPUにフォールバック

✅ 正しいアプローチ:
- NPU TOPS = INT8量子化モデル専用の指標
- LLM実行はGPU (VRAM) の方が重要
- 実際のベンチマーク（推論速度 tok/s）で判断
- モデルのNPU対応状況を事前確認
```

### アンチパターン 2: RAM不足で大規模LLMを実行しようとする

```
❌ 悪い例:
16GB RAMのPCで70Bパラメータモデルを実行しようとする
→ スワップ発生で推論速度が1/10以下に低下

✅ 正しいアプローチ（RAM別推奨モデル）:
- 8GB RAM  → Phi-3 Mini (3.8B, Q4) / Gemma 2 2B
- 16GB RAM → Llama 3.1 8B (Q4) / Mistral 7B (Q4)
- 32GB RAM → Llama 3.1 8B (Q8) / Mixtral 8x7B (Q4)
- 64GB RAM → Llama 3.1 70B (Q4) / Qwen 72B (Q4)
```

### アンチパターン 3: セキュリティを考慮せずにローカルLLMを公開する

```
❌ 悪い例:
Ollama API をファイアウォールなしでネットワークに公開
→ 外部からモデルの不正利用、プロンプトインジェクション攻撃

✅ 正しいアプローチ:
- Ollama はデフォルトで localhost のみにバインド
- 外部公開する場合はリバースプロキシ + 認証を設定
- レート制限を導入（1分あたりの最大リクエスト数）
- 入力のサニタイズとプロンプトインジェクション対策
- SSL/TLS を必ず有効化
```

### アンチパターン 4: ARM版Windowsの互換性問題を無視する

```
❌ 悪い例:
Snapdragon X Elite (ARM) PCでx86専用アプリを使おうとする
→ エミュレーション層による性能低下、一部アプリが動作しない

✅ 正しいアプローチ:
- ARM ネイティブ対応アプリを優先（Python, VS Code, Chrome等は対応済み）
- llama.cpp は ARM ビルドが利用可能
- Ollama は ARM Windows ネイティブ対応
- x86エミュレーションが必要な場合は性能低下を覚悟
- 購入前にターゲットアプリのARM対応状況を確認
```

---

## 8. FAQ

### Q1: Copilot+ PCとは何ですか？通常のPCとの違いは？

**A:** Copilot+ PCはMicrosoftが定義したAI PC規格で、以下の要件を満たす必要があります:
- NPU 40+ TOPS
- RAM 16GB以上
- SSD 256GB以上
- Windows 11 24H2以降

通常のPCとの違いは、Recall（画面履歴AI検索）、Live Captions（リアルタイム翻訳字幕）、Image Creator（オンデバイス画像生成）などのAI機能がNPUで高速動作する点です。

### Q2: ローカルLLMでどの程度の品質が得られますか？

**A:** 8Bパラメータ（Q4量子化）モデルの場合、簡単な質問応答、コード生成、要約タスクで実用的な品質が得られます。GPT-4やClaude 3.5と比較すると推論能力は劣りますが、プライバシー保護・オフライン利用・無料という利点があります。13B以上のモデルではかなり高品質になります。

### Q3: NPUとGPUのどちらをAI処理に使うべきですか？

**A:** タスクにより異なります:
- **NPU向き**: 常時実行のバックグラウンドAI（音声認識、カメラ補正、通知フィルタリング）。省電力が重要。
- **GPU向き**: LLM推論、画像生成、バッチ処理。VRAMとスループットが重要。
- **ハイブリッド**: NPUで前処理、GPUで本推論という組み合わせが最適。

### Q4: AI PCで画像生成（Stable Diffusion）は実用的ですか？

**A:** GPUの性能に依存します:
- **Snapdragon X Elite (Adreno)**: SD 1.5が15-30秒程度。SDXL はメモリ不足で困難。
- **Intel Arc (Xe2)**: OpenVINO最適化でSD 1.5が10-20秒。
- **AMD Radeon 890M**: ROCm対応でSD 1.5が10-25秒。
- **dGPU (RTX 4060+)**: SD 1.5が3-5秒、SDXLが10-15秒で実用的。
- NPUは画像生成には不向き（FP16/FP32が必要なため）。

### Q5: Macbook (Apple Silicon) とAI PCの比較は？

**A:** Apple Silicon (M3/M4) は統合メモリアーキテクチャで独自の強みがあります:
- **メモリ**: CPU/GPU/NPU共有で最大128GB。GPUメモリ不足が発生しない。
- **Neural Engine**: 最大38 TOPS。Core ML統合が優秀。
- **MLX**: Apple独自のAIフレームワークで最適化。
- **LLM実行**: 70Bモデルも64GBモデルで実行可能。
- Windows AI PCの利点はソフトウェアエコシステムの広さと低価格帯のラインナップ。

---

## 9. エッジケース分析

### エッジケース 1: 統合メモリとディスクリートGPUの競合

AI PCの多くはiGPUを内蔵しつつ、外付けeGPUやThunderbolt接続のdGPUも利用可能です。この場合、DirectML/ONNX Runtimeがどのデバイスを選択するかが問題になります。

```python
# 複数GPUが存在する場合のデバイス列挙と明示的選択
import onnxruntime as ort

# 利用可能な全ExecutionProviderを確認
providers = ort.get_available_providers()
print(f"利用可能: {providers}")
# 例: ['TensorrtExecutionProvider', 'CUDAExecutionProvider',
#       'DmlExecutionProvider', 'QNNExecutionProvider', 'CPUExecutionProvider']

# dGPU (CUDA) を優先し、なければ iGPU (DirectML)、最後にNPU
preferred_providers = [
    ('CUDAExecutionProvider', {'device_id': 0}),
    ('DmlExecutionProvider', {'device_id': 0}),  # iGPU
    ('QNNExecutionProvider', {}),                  # NPU
    ('CPUExecutionProvider', {}),
]

session = ort.InferenceSession("model.onnx", providers=preferred_providers)
active = session.get_providers()
print(f"実際に使用中: {active[0]}")
```

**注意点**: eGPU接続時にホットプラグ（動作中の抜き差し）を行うと、セッションがクラッシュします。eGPU使用時は必ず起動前に接続し、セッション終了後に取り外してください。

### エッジケース 2: NPUモデルの精度劣化と検出

INT8量子化でNPUに最適化したモデルは、FP32オリジナルと比較して精度が低下する場合があります。特に以下のケースで顕著です:

```
精度劣化が起きやすいケース:
├── 長文生成（500トークン以上）
│   → 累積誤差で出力品質が低下
├── 多言語混在テキスト
│   → 日本語・英語の切り替え時に劣化
├── 数値計算を含む推論
│   → 四則演算の精度がINT8では不足
└── 小さい物体の検出（画像AI）
    → 量子化で微細な特徴が失われる

精度検証スクリプト:
$ python -c "
from sklearn.metrics import accuracy_score
# FP32とINT8の出力を比較
fp32_results = run_model('model_fp32.onnx', test_data)
int8_results = run_model('model_int8.onnx', test_data)
acc = accuracy_score(fp32_results, int8_results)
print(f'INT8一致率: {acc:.4f}')  # 0.98以上なら許容範囲
"
```

### エッジケース 3: バッテリー駆動時のAI性能スロットリング

ノートPCのバッテリー駆動時、NPUとGPUの両方がパワースロットリングを受けます:

| 電源状態 | NPU性能 | GPU性能 | LLM推論速度 |
|---------|---------|---------|------------|
| AC電源接続 | 100% (45 TOPS) | 100% | 15 tok/s |
| バッテリー（高性能） | 80% (36 TOPS) | 70% | 10 tok/s |
| バッテリー（バランス） | 50% (22 TOPS) | 40% | 6 tok/s |
| バッテリー（省電力） | 30% (13 TOPS) | 20% | 3 tok/s |

```powershell
# Windows で電源プランをプログラムから確認・変更
powercfg /getactivescheme
# AI処理実行前に高性能モードへ切り替え
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
```

**ベストプラクティス**: バッテリー駆動時はモデルサイズを自動的に小さいものに切り替える adaptive loading パターンを実装しましょう。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| AI PCの定義 | NPU 40+ TOPS搭載、Copilot+ PC認定 |
| 主要NPU | Hexagon (Qualcomm), AI Boost (Intel), XDNA (AMD) |
| ローカルLLM | Ollama/LM Studio で7B〜13Bモデルが実用的 |
| 量子化 | INT8/INT4 (GGUF形式) でNPU/メモリ最適化 |
| 開発ツール | ONNX Runtime, DirectML, Windows ML, OpenVINO |
| 選定基準 | TOPSよりRAM容量と実測ベンチマークを重視 |
| セキュリティ | ローカルLLMの公開には認証・レート制限が必須 |
| ARM互換性 | Snapdragon X ではARM対応アプリを優先 |

---

## 次に読むべきガイド

- [GPUコンピューティング — NVIDIA RTX、CUDA](./01-gpu-computing.md)
- [エッジAI — Jetson、Coral、ONNX Runtime](./02-edge-ai.md)
- [DLフレームワーク — PyTorch、TensorFlow](../../ai-analysis-guide/docs/02-deep-learning/03-frameworks.md)

---

## 参考文献

1. **Microsoft** — "Copilot+ PCs Technical Specifications," microsoft.com, 2024
2. **Qualcomm** — "Snapdragon X Elite Compute Platform," qualcomm.com, 2024
3. **Intel** — "Intel Core Ultra Processors with Intel AI Boost," intel.com, 2024
4. **Ollama** — "Run Large Language Models Locally," ollama.com, 2024
5. **ONNX Runtime** — "QNN Execution Provider," onnxruntime.ai, 2024
6. **OpenVINO** — "Intel NPU Plugin," docs.openvino.ai, 2024
