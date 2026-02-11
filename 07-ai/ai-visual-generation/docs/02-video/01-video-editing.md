# 動画編集 -- AI編集ツール

> AIを活用した動画編集の自動化技術を、自動字幕生成・シーン検出・オブジェクト除去・音声分離まで実践的に解説し、編集ワークフローの効率を劇的に向上させる手法を提示する

## この章で学ぶこと

1. **AI 動画編集の基本機能** -- 自動字幕生成、シーン検出、無音部分カット、オブジェクト追跡
2. **主要ツールの比較** -- Runway、Descript、CapCut、DaVinci Resolve の AI 機能と使い分け
3. **プロダクションワークフロー** -- 素材取込から公開までの AI 活用パイプライン

---

## 1. AI 動画編集の全体像

### 1.1 ワークフロー

```
AI 動画編集パイプライン

  素材取込         粗編集            仕上げ          公開
  +----------+    +----------+     +----------+    +----------+
  | 自動文字  |    | AI シーン |     | AI カラー|    | AI サムネ|
  | 起こし    | -> | 検出・分割| --> | グレーディ| -> | イル生成 |
  | (Whisper) |    | 無音カット|     | ング     |    | リサイズ |
  +----------+    +----------+     +----------+    +----------+
  | AI 話者   |    | AI オブジェ|    | AI 音声  |    | AI 字幕  |
  | 分離      |    | クト除去  |     | ノイズ除去|    | 翻訳     |
  +----------+    +----------+     +----------+    +----------+
```

### 1.2 技術マップ

```
AI 動画編集 技術スタック

  音声処理
  ├── Whisper (OpenAI) --- 音声→テキスト変換
  ├── Demucs (Meta)   --- 音声分離 (BGM/ボーカル)
  └── RVC             --- 音声変換・クローニング

  映像処理
  ├── SAM (Meta)      --- 自動セグメンテーション
  ├── RIFE            --- フレーム補間 (スローモーション)
  ├── Real-ESRGAN     --- 超解像 (アップスケール)
  └── ProPainter      --- オブジェクト除去・修復

  テキスト処理
  ├── GPT-4           --- スクリプト生成・要約
  ├── DeepL/Google    --- 字幕翻訳
  └── ElevenLabs      --- AI ナレーション生成
```

---

## 2. 自動字幕生成

### 2.1 Whisper による実装

```python
# OpenAI Whisper で自動字幕生成
import whisper
import json

model = whisper.load_model("large-v3")

# 音声ファイルから文字起こし + タイムスタンプ
result = model.transcribe(
    "video.mp4",
    language="ja",
    task="transcribe",
    word_timestamps=True,      # 単語レベルのタイムスタンプ
    verbose=False,
)

# SRT 形式で出力
def to_srt(segments):
    srt_lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg['start'])
        end = format_timestamp(seg['end'])
        text = seg['text'].strip()
        srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(srt_lines)

def format_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

srt_content = to_srt(result['segments'])
with open("subtitles.srt", "w", encoding="utf-8") as f:
    f.write(srt_content)

print(f"字幕生成完了: {len(result['segments'])} セグメント")
```

### 2.2 話者分離 (Speaker Diarization)

```python
# pyannote.audio で話者分離
from pyannote.audio import Pipeline
import whisper

# 話者分離モデル
diarization = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="your-hf-token"
)

# 話者セグメントの取得
dia_result = diarization("interview.wav")

for turn, _, speaker in dia_result.itertracks(yield_label=True):
    print(f"[{turn.start:.1f}s - {turn.end:.1f}s] {speaker}")
    # [0.5s - 12.3s] SPEAKER_00
    # [12.8s - 25.1s] SPEAKER_01
```

---

## 3. シーン検出と自動カット

```python
# PySceneDetect でシーン検出
from scenedetect import detect, ContentDetector, split_video_ffmpeg

# シーン検出（コンテンツの変化を検出）
scene_list = detect("raw_footage.mp4", ContentDetector(threshold=27.0))

print(f"検出されたシーン: {len(scene_list)} 個")
for i, scene in enumerate(scene_list):
    print(f"  シーン {i+1}: {scene[0].get_timecode()} - {scene[1].get_timecode()}")

# シーンごとに動画を分割
split_video_ffmpeg("raw_footage.mp4", scene_list, output_dir="scenes/")
```

```python
# FFmpeg + Whisper で無音部分の自動カット
import subprocess
import json

def detect_silence(input_file, threshold=-30, duration=1.0):
    """無音区間を検出"""
    cmd = [
        'ffmpeg', '-i', input_file,
        '-af', f'silencedetect=noise={threshold}dB:d={duration}',
        '-f', 'null', '-'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # stderr からサイレンス区間を解析
    silences = parse_silence_output(result.stderr)
    return silences

def remove_silence(input_file, output_file, silences):
    """無音区間を除去した動画を生成"""
    # 音声がある区間のみを抽出して結合
    filter_complex = build_trim_filter(silences, get_duration(input_file))
    cmd = ['ffmpeg', '-i', input_file, '-filter_complex', filter_complex, output_file]
    subprocess.run(cmd)

# 使用例
silences = detect_silence("lecture.mp4", threshold=-35, duration=0.8)
remove_silence("lecture.mp4", "lecture_trimmed.mp4", silences)
```

---

## 4. AI 映像処理

### 4.1 超解像（アップスケール）

```python
# Real-ESRGAN で動画アップスケール
# コマンドライン実行
# realesrgan-ncnn-vulkan -i input.mp4 -o output.mp4 -n realesrgan-x4plus -s 4

# Python API
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(scale=4, model_path='weights/RealESRGAN_x4plus.pth', model=model)

# フレームごとに超解像
output, _ = upsampler.enhance(input_frame, outscale=4)
```

### 4.2 フレーム補間（スローモーション）

```bash
# RIFE でフレーム補間 (30fps → 120fps)
python inference_video.py \
  --video input_30fps.mp4 \
  --output output_120fps.mp4 \
  --exp 2 \  # 2^2 = 4倍のフレーム数
  --model rife-v4.6
```

---

## 5. 主要ツール比較

| 機能 | Runway | Descript | CapCut | DaVinci Resolve |
|------|:------:|:-------:|:------:|:--------------:|
| AI 字幕生成 | 対応 | 対応 (高精度) | 対応 | 対応 |
| テキストベース編集 | -- | 対応 (核心機能) | -- | -- |
| AI 背景除去 | Green Screen AI | -- | AI カットアウト | Magic Mask |
| オブジェクト除去 | Inpainting | -- | -- | Object Removal |
| AI カラー補正 | -- | -- | -- | AI Color Match |
| 音声ノイズ除去 | -- | Studio Sound | ノイズ除去 | Voice Isolation |
| 料金 | $12-76/月 | $24/月 | 無料/Pro $10/月 | 無料/Studio $295 |
| 対象 | クリエイター | ポッドキャスト・YouTube | SNS動画 | プロ映像制作 |

| ユースケース | 推奨ツール | 理由 |
|------------|-----------|------|
| YouTube 動画 | Descript | テキストベース編集で高速 |
| SNS ショート動画 | CapCut | 無料、テンプレート豊富 |
| 映画・CM 品質 | DaVinci Resolve | プロ仕様のカラー・音声ツール |
| 実験的 VFX | Runway | 最先端の AI 映像生成 |
| ポッドキャスト | Descript | 音声編集 + 動画化 |

---

## 6. アンチパターン

### アンチパターン 1: AI 字幕を無校正で公開

```
BAD:
  Whisper で字幕生成 → そのまま公開
  → 固有名詞の誤認識、句読点の不適切な位置
  → 視聴者の信頼を損なう

GOOD:
  1. Whisper で字幕ドラフト生成
  2. 固有名詞・専門用語を辞書登録
  3. 人間が最終校正（特に数字・固有名詞）
  4. タイミング調整（読みやすい表示時間）
```

### アンチパターン 2: 過度な AI 効果の適用

```
BAD:
  AI 超解像 + AI カラー + AI ノイズ除去 + AI 手ブレ補正
  → アーティファクト（AI の痕跡）が蓄積
  → 不自然な映像になる

GOOD:
  - 最も効果の高い1-2種類の処理に絞る
  - 元素材の品質を活かす
  - AI 処理前後の比較を必ず確認
  - エクスポート前にフル解像度でプレビュー
```

---

## 7. FAQ

### Q1. Whisper の精度を向上させるには？

**A.** (1) `large-v3` モデルを使用する（精度最高だが処理速度は遅い）。(2) 言語を明示的に指定する（`language="ja"`）。(3) 初期プロンプト（`initial_prompt`）で専門用語やコンテキストを与える。(4) ノイズの多い音声は事前に音声分離（Demucs）でボーカルを抽出してから処理する。

### Q2. 長尺動画の編集を効率化するには？

**A.** (1) Whisper で文字起こし → テキストベースで不要部分を特定。(2) Descript のテキストベース編集で、テキストの削除 = 動画の該当部分カット。(3) PySceneDetect でシーン分割 → 不要シーンを除外。(4) 無音部分の自動カット。これらの組み合わせで10時間の素材を1-2時間で粗編集できる。

### Q3. AI 動画編集の処理を高速化するには？

**A.** (1) GPU を活用する（NVIDIA CUDA 対応の GPU で Whisper は10倍高速化）。(2) プロキシ編集（低解像度で編集→最終書き出しで高解像度に切替）。(3) バッチ処理（複数動画を並列処理）。(4) クラウド GPU（Google Colab、Runway Cloud）を活用。ローカル処理にこだわらず、クラウドとのハイブリッド運用が現実的。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 自動字幕 | Whisper で生成 → 人間が校正 → SRT/VTT 出力 |
| シーン検出 | PySceneDetect で自動分割、無音カットで効率化 |
| 超解像 | Real-ESRGAN で低解像度素材を4K化 |
| 音声処理 | Demucs で音声分離、ノイズ除去 |
| ツール選定 | YouTube=Descript、SNS=CapCut、プロ=DaVinci Resolve |
| 品質管理 | AI 処理は最小限に、過度な適用はアーティファクトの原因 |

---

## 次に読むべきガイド

- [アニメーション](./02-animation.md) -- AI アニメーション生成技術
- [デザインツール](../01-image/03-design-tools.md) -- サムネイル・バナーの AI デザイン
- [倫理的考慮](../03-3d/03-ethical-considerations.md) -- AI 生成コンテンツの権利と倫理

---

## 参考文献

1. **Whisper (OpenAI)** -- https://github.com/openai/whisper -- 音声認識モデル
2. **Runway ML Documentation** -- https://docs.runwayml.com/ -- AI 映像編集ツール
3. **DaVinci Resolve Training** -- https://www.blackmagicdesign.com/products/davinciresolve/training
