# ポッドキャストツール — 自動文字起こし・要約・編集

> 音声コンテンツの制作・管理を AI で革新する。自動文字起こし、インテリジェント要約、AI アシスト編集の技術と実装を体系的に学ぶ。

---

## この章で学ぶこと

1. **自動文字起こし (ASR)** — Whisper をはじめとする最新 ASR エンジンの活用法と精度向上テクニック
2. **インテリジェント要約** — 長時間音声からチャプター分割・要約・ショーノートを自動生成する手法
3. **AI アシスト編集** — フィラー除去、無音検出、ノイズ除去、BGM ダッキングの自動化

---

## 1. ポッドキャスト制作パイプラインの全体像

### 1.1 AI 活用パイプライン

```
+----------+     +----------+     +-----------+     +-----------+
|  録音    | --> |  前処理   | --> |  文字起こし | --> |  後処理    |
|          |     |  ノイズ除去|     |  ASR      |     |  要約/編集 |
+----------+     +----------+     +-----------+     +-----------+
                                                          |
                      +-----------------------------------+
                      |
                      v
+----------+     +-----------+     +-----------+
|  公開    | <-- |  最終編集  | <-- |  AI支援    |
|  配信    |     |  マスタリング|    |  チャプター |
+----------+     +-----------+     +-----------+
```

### 1.2 従来ワークフローと AI 活用ワークフローの比較

| 工程 | 従来の手法 | AI 活用手法 | 時間削減 |
|------|-----------|------------|---------|
| 文字起こし | 手動 (音声の3〜5倍の時間) | Whisper 自動起こし | 90%+ |
| 要約/ショーノート | 手動作成 | LLM による自動生成 | 80%+ |
| フィラー除去 | 波形を目視で編集 | AI 検出 + 自動除去 | 70%+ |
| ノイズ除去 | DAW プラグイン手動調整 | AI ワンクリック | 60%+ |
| チャプター分割 | 手動タイムスタンプ | トピック検出 + 自動分割 | 85%+ |

---

## 2. 自動文字起こし

### 2.1 Whisper による高精度文字起こし

```python
# コード例 1: OpenAI Whisper で日本語ポッドキャストを文字起こしする
import whisper
import json

# モデルをロード (tiny/base/small/medium/large-v3)
model = whisper.load_model("large-v3")

# 文字起こし実行
result = model.transcribe(
    "podcast_episode_042.mp3",
    language="ja",           # 日本語を指定
    task="transcribe",       # "translate" で英訳も可能
    word_timestamps=True,    # 単語レベルのタイムスタンプ
    condition_on_previous_text=True,  # 文脈を考慮
    initial_prompt="ポッドキャスト「テックトーク」第42回。ゲスト: 田中太郎。"
)

# セグメントごとの結果
for segment in result["segments"]:
    start = segment["start"]
    end = segment["end"]
    text = segment["text"]
    print(f"[{start:.1f}s - {end:.1f}s] {text}")

# SRT字幕ファイルとして出力
def to_srt(segments):
    srt = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        srt.append(f"{i}\n{start} --> {end}\n{seg['text'].strip()}\n")
    return "\n".join(srt)

def format_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

with open("episode_042.srt", "w", encoding="utf-8") as f:
    f.write(to_srt(result["segments"]))
```

### 2.2 faster-whisper で高速推論

```python
# コード例 2: faster-whisper による高速文字起こし (CTranslate2 ベース)
from faster_whisper import WhisperModel

# INT8量子化モデルで高速化
model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="int8_float16"  # INT8量子化で2〜4倍高速
)

segments, info = model.transcribe(
    "podcast_episode_042.mp3",
    language="ja",
    beam_size=5,
    vad_filter=True,           # Voice Activity Detection で無音を除外
    vad_parameters=dict(
        min_silence_duration_ms=500,  # 500ms以上の無音を区切りに
        speech_pad_ms=200,
    ),
)

print(f"検出言語: {info.language} (確信度: {info.language_probability:.2f})")
print(f"音声全体の長さ: {info.duration:.1f}秒")

for segment in segments:
    print(f"[{segment.start:.1f}s -> {segment.end:.1f}s] {segment.text}")
```

### 2.3 話者分離 (Speaker Diarization)

```python
# コード例 3: pyannote.audio で話者分離 + Whisper 文字起こし
from pyannote.audio import Pipeline
import whisper
import torch

# 話者分離パイプライン
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="YOUR_HF_TOKEN"
)

# 話者分離を実行
diarization = diarization_pipeline("podcast_episode_042.wav")

# Whisper で文字起こし
whisper_model = whisper.load_model("large-v3")
transcription = whisper_model.transcribe(
    "podcast_episode_042.wav",
    language="ja",
    word_timestamps=True
)

# 話者分離結果と文字起こしを統合
def merge_diarization_transcription(diarization, segments):
    """話者ラベルとテキストを結合する"""
    result = []
    for segment in segments:
        mid_time = (segment["start"] + segment["end"]) / 2
        # 最も近い話者ラベルを見つける
        speaker = "Unknown"
        for turn, _, spk in diarization.itertracks(yield_label=True):
            if turn.start <= mid_time <= turn.end:
                speaker = spk
                break
        result.append({
            "speaker": speaker,
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })
    return result

merged = merge_diarization_transcription(
    diarization, transcription["segments"]
)
for entry in merged:
    print(f"[{entry['speaker']}] {entry['start']:.1f}s: {entry['text']}")
```

---

## 3. インテリジェント要約

### 3.1 LLM による自動要約とショーノート生成

```python
# コード例 4: GPT-4 / Claude でポッドキャスト要約を生成する
from openai import OpenAI

client = OpenAI()

def generate_show_notes(transcript: str, episode_title: str) -> str:
    """文字起こしからショーノートを自動生成する"""
    prompt = f"""以下はポッドキャスト「{episode_title}」の文字起こしです。
以下のフォーマットでショーノートを生成してください:

1. エピソード概要 (3〜5文)
2. 主要トピック (箇条書き、各トピックに簡単な説明)
3. チャプターマーク (タイムスタンプ付き)
4. キーワード/用語集
5. 関連リンク (言及されたツール、書籍、サービスなど)

文字起こし:
{transcript}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "あなたはポッドキャスト編集者です。"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        temperature=0.3  # 要約は低温度で安定させる
    )

    return response.choices[0].message.content
```

### 3.2 チャプター分割のアルゴリズム

```
+--------------------------------------------------------------+
|  音声ストリーム                                                |
|  =============================================>               |
|                                                                |
|  Step 1: 文字起こし + タイムスタンプ                           |
|  [0:00] "今日は..." [2:30] "次に..." [15:20] "最後に..."     |
|                                                                |
|  Step 2: テキストをウィンドウ分割                              |
|  [Win1: 0-5min] [Win2: 5-10min] [Win3: 10-15min] ...         |
|                                                                |
|  Step 3: 各ウィンドウの埋め込みベクトルを計算                  |
|  [Vec1] [Vec2] [Vec3] ...                                      |
|                                                                |
|  Step 4: 隣接ウィンドウのコサイン類似度を計算                  |
|  cos(V1,V2)=0.92  cos(V2,V3)=0.45  cos(V3,V4)=0.88          |
|                  ^^^^^^^^^^^^                                   |
|                  類似度が低い = トピック境界                     |
|                                                                |
|  Step 5: 境界にチャプターマークを生成                          |
|  Chapter 1: 0:00 - 10:00 "イントロダクション"                  |
|  Chapter 2: 10:00 - 25:00 "ゲストインタビュー"                 |
|  Chapter 3: 25:00 - 35:00 "Q&Aコーナー"                       |
+--------------------------------------------------------------+
```

---

## 4. AI アシスト編集

### 4.1 フィラー除去

```python
# コード例 5: フィラーワードを検出して除去する
import pydub
from pydub import AudioSegment
import re

def remove_fillers(transcript_segments, audio_path, filler_patterns=None):
    """
    フィラーワード（えーと、あの、まあ等）を検出し、音声から除去する。
    """
    if filler_patterns is None:
        filler_patterns = [
            r"えーと", r"あのー?", r"まあ?", r"そのー?",
            r"なんか", r"えー+", r"うーん",
        ]

    audio = AudioSegment.from_file(audio_path)
    combined_pattern = "|".join(filler_patterns)

    # フィラー区間を特定
    filler_regions = []
    for seg in transcript_segments:
        if re.fullmatch(combined_pattern, seg["text"].strip()):
            filler_regions.append((
                int(seg["start"] * 1000),  # ms
                int(seg["end"] * 1000)
            ))

    # フィラー区間を除去して連結
    if not filler_regions:
        return audio

    cleaned_parts = []
    prev_end = 0
    for start, end in sorted(filler_regions):
        if start > prev_end:
            cleaned_parts.append(audio[prev_end:start])
        prev_end = end
    cleaned_parts.append(audio[prev_end:])

    result = cleaned_parts[0]
    for part in cleaned_parts[1:]:
        # クロスフェードで自然な接続
        result = result.append(part, crossfade=50)

    return result
```

### 4.2 ノイズ除去ツール比較

| ツール | 手法 | リアルタイム | 品質 | コスト |
|--------|------|-------------|------|--------|
| RNNoise | RNNベースの音声強調 | ○ | 良 | 無料 (OSS) |
| Adobe Podcast Enhance | クラウドAI | x | 優 | 無料 (制限付き) |
| NVIDIA Broadcast | RTXベースAI | ○ | 優 | 無料 (GPU必要) |
| Dolby.io | クラウドAPI | x | 優 | 有料 |
| Auphonic | マルチバンドAI | x | 優 | フリーミアム |

---

## 5. アンチパターン

### アンチパターン 1: 「文字起こし結果を無検証で公開」

```
[誤り] Whisper の出力をそのまま字幕・記事として公開する

問題点:
- 固有名詞の誤認識（人名、製品名、技術用語）
- ハルシネーション（無音区間に存在しないテキストが生成される）
- 話者の混同（who said what が不正確）

[正解] 自動起こし → 人間レビュー → 公開 の3ステップ
  1. initial_prompt に固有名詞リストを渡して精度向上
  2. 信頼度スコアが低いセグメントをハイライト表示
  3. 人間が修正したデータをファインチューニングに活用
```

### アンチパターン 2: 「一括処理で全エピソードを同一設定」

```
[誤り] 収録環境・ゲスト・内容が異なるのに同じ設定で処理する

問題点:
- 静かな環境で収録した回にノイズ除去を強くかけると音質劣化
- ゲストの声質によって話者分離の精度が変わる
- 専門用語が多い回はプロンプト調整が必要

[正解] エピソードごとにメタデータを管理し、設定を調整する
  - 収録環境プロファイル (スタジオ/リモート/屋外)
  - ゲスト情報と声質プロファイル
  - 専門分野に応じた用語辞書
```

---

## 6. FAQ

### Q1: Whisper のモデルサイズはどれを選ぶべきですか？

**A:** 用途と環境に応じて選択します。

- **tiny / base**: リアルタイム用途、エッジデバイス。日本語精度は低め
- **small / medium**: バランス重視。GPU があれば medium を推奨
- **large-v3**: 最高精度。日本語の文字起こし精度が大幅に改善。faster-whisper + INT8 量子化で実用速度に

処理時間の目安（1時間の音声、GPU使用時）: tiny=1分、small=3分、medium=8分、large-v3=15分

### Q2: 話者分離の精度を上げるには？

**A:** 以下のアプローチが有効です。

1. **事前に話者数を指定**: `num_speakers=2` のように指定すると精度向上
2. **話者の声サンプルを提供**: 事前登録した声紋との照合で精度向上
3. **高品質な音声入力**: 各話者を別マイクで収録し、マルチチャンネルで処理
4. **後処理ルール**: 「ホストは常に最初に話す」などのヒューリスティクス

### Q3: 長時間（3時間超）のポッドキャストを効率的に処理するには？

**A:** 以下の戦略を推奨します。

1. **チャンク分割**: 30分ごとに分割して並列処理（前後5秒のオーバーラップを設ける）
2. **VAD 前処理**: 無音区間をスキップして処理時間を短縮
3. **段階的処理**: まず small モデルで粗い文字起こし → 重要区間のみ large-v3 で再処理
4. **ストリーミング API**: faster-whisper のストリーミングモードで逐次出力

---

## 7. まとめ

| 機能 | 技術 | 推奨ツール | 精度 |
|------|------|-----------|------|
| 文字起こし | ASR (Whisper) | faster-whisper, Whisper API | 95%+ (日本語) |
| 話者分離 | Speaker Diarization | pyannote.audio 3.1 | 90%+ |
| 要約生成 | LLM | GPT-4o, Claude | 高品質 |
| チャプター分割 | トピック検出 | 埋め込み + 境界検出 | 85%+ |
| フィラー除去 | ASR + ルール | Whisper + 正規表現 | 80%+ |
| ノイズ除去 | 音声強調 AI | RNNoise, Auphonic | 高品質 |

---

## 次に読むべきガイド

- [リアルタイム音声](../03-development/02-real-time-audio.md) — WebRTC、ストリーミング STT/TTS の実装
- [音声合成の基礎](../01-basics/01-tts-fundamentals.md) — TTS エンジンの選択と活用
- [LLM 比較ガイド](../../../llm-and-ai-comparison/docs/01-overview/00-landscape.md) — 要約に使う LLM の選定

---

## 参考文献

1. Radford, A. et al. (2023). "Robust Speech Recognition via Large-Scale Weak Supervision." *Proceedings of the 40th International Conference on Machine Learning (ICML 2023)*. OpenAI. https://arxiv.org/abs/2212.04356
2. Bredin, H. et al. (2023). "pyannote.audio 2.1: speaker diarization pipeline." *INTERSPEECH 2023*. https://doi.org/10.21437/Interspeech.2023-105
3. Park, T.J. et al. (2022). "A Review of Speaker Diarization: Recent Advances with Deep Learning." *Computer Speech & Language, 72*. https://doi.org/10.1016/j.csl.2021.101317
