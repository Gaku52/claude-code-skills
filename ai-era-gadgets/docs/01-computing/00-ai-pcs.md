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

### 1.1 NPU処理の流れ

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

---

## 4. アンチパターン

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

---

## 5. FAQ

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

---

## まとめ

| 項目 | ポイント |
|------|---------|
| AI PCの定義 | NPU 40+ TOPS搭載、Copilot+ PC認定 |
| 主要NPU | Hexagon (Qualcomm), AI Boost (Intel), XDNA (AMD) |
| ローカルLLM | Ollama/LM Studio で7B〜13Bモデルが実用的 |
| 量子化 | INT8/INT4 (GGUF形式) でNPU/メモリ最適化 |
| 開発ツール | ONNX Runtime, DirectML, Windows ML |
| 選定基準 | TOPSよりRAM容量と実測ベンチマークを重視 |

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
