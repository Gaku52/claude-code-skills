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

### 1.3 ポッドキャスト制作の品質チェックリスト

```python
# ポッドキャスト品質チェックパイプライン
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class QualityReport:
    """ポッドキャスト品質レポート"""
    loudness_lufs: float
    true_peak_dbtp: float
    noise_floor_db: float
    silence_ratio: float
    clipping_count: int
    dc_offset: float
    stereo_balance: float
    overall_grade: str

class PodcastQualityChecker:
    """ポッドキャスト品質自動チェッカー"""

    # 配信プラットフォーム別の推奨値
    PLATFORM_SPECS = {
        "apple_podcasts": {
            "lufs_target": -16.0,
            "lufs_tolerance": 1.0,
            "true_peak_max": -1.0,
            "sample_rate": 44100,
            "format": "AAC",
            "bitrate": "128kbps",
        },
        "spotify": {
            "lufs_target": -14.0,
            "lufs_tolerance": 2.0,
            "true_peak_max": -1.0,
            "sample_rate": 44100,
            "format": "OGG Vorbis",
            "bitrate": "96kbps",
        },
        "youtube": {
            "lufs_target": -14.0,
            "lufs_tolerance": 1.0,
            "true_peak_max": -1.0,
            "sample_rate": 48000,
            "format": "AAC",
            "bitrate": "192kbps",
        },
    }

    def check_audio(self, audio: np.ndarray, sr: int,
                     platform: str = "apple_podcasts") -> QualityReport:
        """音声品質を包括的にチェック"""
        spec = self.PLATFORM_SPECS[platform]

        loudness = self._measure_lufs(audio, sr)
        true_peak = self._measure_true_peak(audio)
        noise_floor = self._measure_noise_floor(audio, sr)
        silence_ratio = self._measure_silence_ratio(audio)
        clipping = self._count_clipping(audio)
        dc_offset = float(np.mean(audio))
        stereo_balance = self._check_stereo_balance(audio)

        # 総合評価
        grade = self._compute_grade(
            loudness, true_peak, noise_floor, clipping, spec
        )

        return QualityReport(
            loudness_lufs=loudness,
            true_peak_dbtp=true_peak,
            noise_floor_db=noise_floor,
            silence_ratio=silence_ratio,
            clipping_count=clipping,
            dc_offset=dc_offset,
            stereo_balance=stereo_balance,
            overall_grade=grade,
        )

    def _measure_lufs(self, audio, sr):
        """LUFS(ラウドネスユニット)を測定"""
        # ITU-R BS.1770 準拠の簡略版
        rms = np.sqrt(np.mean(audio ** 2))
        return 20 * np.log10(rms + 1e-10)

    def _measure_true_peak(self, audio):
        """True Peak(dBTP)を測定"""
        peak = np.max(np.abs(audio))
        return 20 * np.log10(peak + 1e-10)

    def _measure_noise_floor(self, audio, sr, frame_ms=50):
        """ノイズフロアを推定"""
        frame_size = int(sr * frame_ms / 1000)
        frames = [audio[i:i+frame_size] for i in range(0, len(audio)-frame_size, frame_size)]
        frame_energies = [20 * np.log10(np.sqrt(np.mean(f**2)) + 1e-10) for f in frames]
        # 最も静かな10%のフレームの平均 = ノイズフロア推定
        frame_energies.sort()
        bottom_10 = frame_energies[:max(1, len(frame_energies) // 10)]
        return np.mean(bottom_10)

    def _measure_silence_ratio(self, audio, threshold_db=-50):
        """無音区間の割合を計算"""
        threshold = 10 ** (threshold_db / 20)
        silent_samples = np.sum(np.abs(audio) < threshold)
        return silent_samples / len(audio)

    def _count_clipping(self, audio, threshold=0.99):
        """クリッピング箇所のカウント"""
        return int(np.sum(np.abs(audio) > threshold))

    def _check_stereo_balance(self, audio):
        """ステレオバランスチェック（モノラルなら0.0）"""
        if audio.ndim < 2:
            return 0.0
        left_rms = np.sqrt(np.mean(audio[0] ** 2))
        right_rms = np.sqrt(np.mean(audio[1] ** 2))
        if left_rms + right_rms == 0:
            return 0.0
        return (left_rms - right_rms) / (left_rms + right_rms)

    def _compute_grade(self, loudness, true_peak, noise_floor, clipping, spec):
        """総合評価グレードを算出"""
        issues = []
        if abs(loudness - spec["lufs_target"]) > spec["lufs_tolerance"]:
            issues.append("loudness")
        if true_peak > spec["true_peak_max"]:
            issues.append("true_peak")
        if noise_floor > -40:
            issues.append("noise")
        if clipping > 10:
            issues.append("clipping")

        if len(issues) == 0:
            return "A (配信品質)"
        elif len(issues) == 1:
            return f"B (要改善: {issues[0]})"
        else:
            return f"C (問題あり: {', '.join(issues)})"

# 使用例
checker = PodcastQualityChecker()
# report = checker.check_audio(audio_data, sr=44100, platform="apple_podcasts")
```

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

### 2.4 文字起こし精度向上テクニック

```python
# Whisper の精度を最大化するテクニック集

class WhisperAccuracyOptimizer:
    """Whisper文字起こし精度最適化"""

    def __init__(self, model_size="large-v3"):
        self.model = whisper.load_model(model_size)

    def transcribe_with_domain_prompt(
        self,
        audio_path: str,
        domain: str = "tech",
    ) -> dict:
        """ドメイン特化プロンプトで精度向上"""
        # ドメイン別の専門用語プロンプト
        domain_prompts = {
            "tech": (
                "テクノロジーポッドキャスト。API、Docker、Kubernetes、"
                "マイクロサービス、CI/CD、GitHub Actions、TypeScript、"
                "React、Next.js、AWS、GCP、Azure、LLM、GPT、Claude。"
            ),
            "medical": (
                "医療ポッドキャスト。患者、診断、治療、"
                "インスリン、コレステロール、血圧、MRI、CT、"
                "免疫療法、抗体、ワクチン。"
            ),
            "finance": (
                "金融ポッドキャスト。株式、債券、投資信託、"
                "日経平均、TOPIX、PER、PBR、ROE、配当利回り、"
                "マクロ経済、金融政策。"
            ),
            "gaming": (
                "ゲームポッドキャスト。PlayStation、Nintendo Switch、"
                "Steam、FPS、RPG、MMO、eスポーツ、ストリーミング、"
                "GPU、フレームレート。"
            ),
        }

        initial_prompt = domain_prompts.get(domain, "")

        result = self.model.transcribe(
            audio_path,
            language="ja",
            initial_prompt=initial_prompt,
            word_timestamps=True,
            condition_on_previous_text=True,
            # ハルシネーション抑制パラメータ
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )

        return result

    def post_process_transcript(self, segments: list) -> list:
        """文字起こし結果の後処理"""
        processed = []

        for seg in segments:
            text = seg["text"]

            # 1. 重複テキストの除去
            text = self._remove_repetitions(text)

            # 2. 句読点の正規化
            text = self._normalize_punctuation(text)

            # 3. 数値表現の統一
            text = self._normalize_numbers(text)

            processed.append({**seg, "text": text})

        return processed

    def _remove_repetitions(self, text: str) -> str:
        """Whisperのハルシネーションによる繰り返しを除去"""
        import re
        # 3回以上繰り返されるフレーズを検出
        pattern = r"(.{3,}?)\1{2,}"
        return re.sub(pattern, r"\1", text)

    def _normalize_punctuation(self, text: str) -> str:
        """句読点を統一"""
        replacements = {
            "．": "。",
            "，": "、",
            "  ": " ",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text.strip()

    def _normalize_numbers(self, text: str) -> str:
        """数値表現の統一（全角→半角）"""
        zen = "０１２３４５６７８９"
        han = "0123456789"
        for z, h in zip(zen, han):
            text = text.replace(z, h)
        return text

    def validate_transcript(self, segments: list) -> dict:
        """文字起こし品質のバリデーション"""
        total_segments = len(segments)
        low_confidence = []
        hallucination_suspects = []
        empty_segments = []

        for i, seg in enumerate(segments):
            # 信頼度チェック
            if seg.get("avg_logprob", 0) < -0.8:
                low_confidence.append(i)

            # ハルシネーション疑い（圧縮率が高い = 繰り返し）
            if seg.get("compression_ratio", 0) > 2.4:
                hallucination_suspects.append(i)

            # 空セグメント
            if not seg.get("text", "").strip():
                empty_segments.append(i)

        return {
            "total_segments": total_segments,
            "low_confidence_count": len(low_confidence),
            "low_confidence_segments": low_confidence,
            "hallucination_suspects": len(hallucination_suspects),
            "empty_segments": len(empty_segments),
            "quality_score": 1.0 - (
                (len(low_confidence) + len(hallucination_suspects))
                / max(total_segments, 1)
            ),
        }
```

### 2.5 バッチ処理パイプライン

```python
import os
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

class PodcastBatchProcessor:
    """複数エピソードの一括処理パイプライン"""

    def __init__(self, output_dir: str, model_size: str = "large-v3"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_size = model_size

    def process_episode(self, audio_path: str, metadata: dict = None) -> dict:
        """1エピソードを完全処理"""
        episode_name = Path(audio_path).stem
        episode_dir = self.output_dir / episode_name
        episode_dir.mkdir(exist_ok=True)

        results = {
            "episode": episode_name,
            "audio_path": audio_path,
            "processed_at": datetime.now().isoformat(),
        }

        # Step 1: 音声前処理
        preprocessed_path = self._preprocess(audio_path, episode_dir)
        results["preprocessed"] = str(preprocessed_path)

        # Step 2: 文字起こし
        transcript = self._transcribe(preprocessed_path)
        transcript_path = episode_dir / "transcript.json"
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)
        results["transcript_path"] = str(transcript_path)

        # Step 3: SRT字幕生成
        srt_path = episode_dir / "subtitles.srt"
        self._generate_srt(transcript["segments"], srt_path)
        results["srt_path"] = str(srt_path)

        # Step 4: テキスト全文出力
        full_text_path = episode_dir / "full_text.txt"
        full_text = " ".join(s["text"] for s in transcript["segments"])
        with open(full_text_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        results["full_text_path"] = str(full_text_path)

        return results

    def process_batch(self, audio_paths: list, max_workers: int = 2) -> list:
        """複数エピソードの並列処理"""
        results = []
        for path in audio_paths:
            try:
                result = self.process_episode(path)
                results.append(result)
                print(f"完了: {path}")
            except Exception as e:
                print(f"エラー: {path} - {e}")
                results.append({"audio_path": path, "error": str(e)})
        return results

    def _preprocess(self, audio_path, output_dir):
        """音声前処理（ノイズ除去・正規化）"""
        import subprocess
        output_path = output_dir / "preprocessed.wav"
        # ffmpeg で 16kHz モノラルに変換
        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-ar", "16000", "-ac", "1",
            "-acodec", "pcm_s16le",
            str(output_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path

    def _transcribe(self, audio_path):
        """Whisper文字起こし"""
        from faster_whisper import WhisperModel
        model = WhisperModel(self.model_size, device="cuda", compute_type="int8_float16")
        segments, info = model.transcribe(
            str(audio_path),
            language="ja",
            beam_size=5,
            vad_filter=True,
        )
        return {
            "language": info.language,
            "duration": info.duration,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "avg_logprob": s.avg_logprob,
                }
                for s in segments
            ],
        }

    def _generate_srt(self, segments, output_path):
        """SRT字幕ファイル生成"""
        lines = []
        for i, seg in enumerate(segments, 1):
            start = self._format_srt_time(seg["start"])
            end = self._format_srt_time(seg["end"])
            lines.append(f"{i}\n{start} --> {end}\n{seg['text'].strip()}\n")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _format_srt_time(self, seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
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

### 3.3 セマンティックチャプター分割の実装

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Optional

class SemanticChapterDetector:
    """セマンティック分析によるチャプター自動分割"""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        self.embedder = SentenceTransformer(model_name)

    def detect_chapters(
        self,
        segments: list,
        window_seconds: float = 120.0,
        similarity_threshold: float = 0.65,
        min_chapter_seconds: float = 180.0,
    ) -> list:
        """
        文字起こしセグメントからチャプター境界を自動検出

        Parameters:
            segments: Whisperの出力セグメント [{start, end, text}, ...]
            window_seconds: テキストウィンドウの幅（秒）
            similarity_threshold: 境界判定の閾値（低いほど多くの境界を検出）
            min_chapter_seconds: 最小チャプター長（秒）

        Returns:
            [{start, end, title, summary}, ...]
        """
        # Step 1: 時間ウィンドウごとにテキストを集約
        windows = self._create_windows(segments, window_seconds)

        if len(windows) < 2:
            return [{"start": 0, "end": segments[-1]["end"],
                     "title": "全体", "summary": ""}]

        # Step 2: 各ウィンドウの埋め込みベクトルを計算
        texts = [w["text"] for w in windows]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)

        # Step 3: 隣接ウィンドウ間のコサイン類似度を計算
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Step 4: 類似度が閾値以下の箇所を境界として検出
        boundaries = [0]  # 先頭は常に境界
        for i, sim in enumerate(similarities):
            if sim < similarity_threshold:
                boundary_time = windows[i + 1]["start"]
                # 最小チャプター長をチェック
                if boundary_time - boundaries[-1] >= min_chapter_seconds:
                    boundaries.append(boundary_time)

        # 末尾を追加
        end_time = segments[-1]["end"]
        if end_time not in boundaries:
            boundaries.append(end_time)

        # Step 5: チャプター情報を構築
        chapters = []
        for i in range(len(boundaries) - 1):
            chapter_segments = [
                s for s in segments
                if s["start"] >= boundaries[i] and s["end"] <= boundaries[i + 1]
            ]
            chapter_text = " ".join(s["text"] for s in chapter_segments)
            chapters.append({
                "start": boundaries[i],
                "end": boundaries[i + 1],
                "text": chapter_text[:500],  # 要約生成用に先頭500文字
            })

        return chapters

    def _create_windows(self, segments, window_seconds):
        """セグメントを時間ウィンドウに集約"""
        if not segments:
            return []
        windows = []
        current_text = ""
        window_start = segments[0]["start"]

        for seg in segments:
            if seg["start"] - window_start >= window_seconds and current_text:
                windows.append({
                    "start": window_start,
                    "end": seg["start"],
                    "text": current_text.strip(),
                })
                current_text = ""
                window_start = seg["start"]
            current_text += " " + seg["text"]

        if current_text:
            windows.append({
                "start": window_start,
                "end": segments[-1]["end"],
                "text": current_text.strip(),
            })

        return windows
```

### 3.4 マルチフォーマット出力

```python
class PodcastContentGenerator:
    """ポッドキャストコンテンツの多形式出力"""

    def __init__(self, llm_client):
        self.llm = llm_client

    def generate_all_formats(self, transcript: str, metadata: dict) -> dict:
        """全フォーマットのコンテンツを一括生成"""
        return {
            "show_notes": self._generate_show_notes(transcript, metadata),
            "blog_post": self._generate_blog_post(transcript, metadata),
            "social_posts": self._generate_social_posts(transcript, metadata),
            "newsletter": self._generate_newsletter(transcript, metadata),
            "search_keywords": self._extract_keywords(transcript),
        }

    def _generate_show_notes(self, transcript, metadata):
        """Apple Podcasts/Spotify向けショーノート"""
        prompt = f"""以下のポッドキャスト文字起こしからショーノートを生成してください。

タイトル: {metadata.get('title', '')}
ゲスト: {metadata.get('guest', '')}

フォーマット:
## 概要
(3-5文でエピソードの要約)

## トピック
- (主要トピック1)
- (主要トピック2)
...

## チャプター
- 00:00 (チャプター名)
...

## メンション
(言及されたツール、書籍、サービス)

文字起こし:
{transcript[:8000]}
"""
        return self._call_llm(prompt)

    def _generate_social_posts(self, transcript, metadata):
        """SNS投稿（Twitter/X、LinkedIn）を生成"""
        prompt = f"""以下のポッドキャスト文字起こしから、SNS投稿を3パターン生成してください。

タイトル: {metadata.get('title', '')}

1. Twitter/X用（280文字以内、ハッシュタグ付き）
2. LinkedIn用（500文字程度、ビジネス寄り）
3. Instagram用（キャッチーな引用 + 解説）

文字起こし:
{transcript[:4000]}
"""
        return self._call_llm(prompt)

    def _extract_keywords(self, transcript):
        """SEO向けキーワード抽出"""
        prompt = f"""以下の文字起こしから、検索エンジン最適化に役立つキーワードを20個抽出してください。
重要度順に並べ、各キーワードの出現回数も示してください。

文字起こし:
{transcript[:6000]}
"""
        return self._call_llm(prompt)

    def _generate_blog_post(self, transcript, metadata):
        """ブログ記事に変換"""
        prompt = f"""以下のポッドキャスト文字起こしを、読みやすいブログ記事に変換してください。

要件:
- 会話形式ではなく、記事形式に再構成
- 見出し（H2, H3）を適切に使用
- 重要な引用を「」で強調
- 1500-2500文字程度

タイトル: {metadata.get('title', '')}
文字起こし:
{transcript[:10000]}
"""
        return self._call_llm(prompt)

    def _generate_newsletter(self, transcript, metadata):
        """メールニュースレター用テキスト"""
        prompt = f"""以下のポッドキャストの内容を、メールニュースレター用に要約してください。

要件:
- 件名（開封率を高める魅力的なもの）
- リード文（50文字以内）
- 本文（3つのキーポイント）
- CTA（ポッドキャストを聴くへの誘導）

タイトル: {metadata.get('title', '')}
文字起こし:
{transcript[:6000]}
"""
        return self._call_llm(prompt)

    def _call_llm(self, prompt):
        """LLM API呼び出し（共通）"""
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "あなたはポッドキャストの編集・マーケティング担当です。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.4,
        )
        return response.choices[0].message.content
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

### 4.2 高度な音声編集パイプライン

```python
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import numpy as np

class PodcastEditor:
    """AIアシスト ポッドキャスト編集パイプライン"""

    def __init__(self, audio_path: str):
        self.audio = AudioSegment.from_file(audio_path)
        self.sample_rate = self.audio.frame_rate
        self.edit_log = []

    def remove_long_silences(self, threshold_db=-45, min_silence_ms=2000,
                              keep_ms=500):
        """長い無音区間を短縮"""
        from pydub.silence import detect_silence

        silences = detect_silence(
            self.audio,
            min_silence_len=min_silence_ms,
            silence_thresh=threshold_db,
        )

        if not silences:
            return self

        # 無音区間を keep_ms に短縮
        parts = []
        prev_end = 0
        removed_ms = 0

        for start, end in silences:
            silence_len = end - start
            if silence_len > min_silence_ms:
                parts.append(self.audio[prev_end:start + keep_ms // 2])
                removed_ms += (silence_len - keep_ms)
                prev_end = end - keep_ms // 2

        parts.append(self.audio[prev_end:])
        self.audio = sum(parts)
        self.edit_log.append(f"無音短縮: {removed_ms/1000:.1f}秒削減")
        return self

    def auto_level(self, target_dbfs=-16.0):
        """音量の自動レベリング（セクションごと）"""
        chunk_ms = 30000  # 30秒チャンク
        chunks = []

        for i in range(0, len(self.audio), chunk_ms):
            chunk = self.audio[i:i + chunk_ms]
            current_db = chunk.dBFS
            if current_db != float('-inf'):
                gain = target_dbfs - current_db
                # 極端なゲイン変更を制限
                gain = max(-12, min(12, gain))
                chunk = chunk + gain
            chunks.append(chunk)

        self.audio = sum(chunks)
        self.edit_log.append(f"自動レベリング: 目標 {target_dbfs} dBFS")
        return self

    def apply_podcast_eq(self):
        """ポッドキャスト向けEQ（ハイパスフィルタ + プレゼンス強調）"""
        # pydub では直接的なEQは限定的なので、
        # ffmpeg 連携で実装
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            self.audio.export(tmp_in.name, format="wav")
            tmp_in_path = tmp_in.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            tmp_out_path = tmp_out.name

        # ffmpeg のEQフィルタ
        # ハイパス80Hz + プレゼンス帯域(2-5kHz)ブースト
        eq_filter = (
            "highpass=f=80,"
            "equalizer=f=200:t=q:w=1.5:g=-2,"    # 低域の濁り除去
            "equalizer=f=3000:t=q:w=1.0:g=3,"     # プレゼンスブースト
            "equalizer=f=8000:t=q:w=1.5:g=1"       # エアバンド
        )

        cmd = [
            "ffmpeg", "-y", "-i", tmp_in_path,
            "-af", eq_filter,
            tmp_out_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        self.audio = AudioSegment.from_wav(tmp_out_path)
        self.edit_log.append("ポッドキャストEQ適用")

        # 一時ファイル削除
        os.unlink(tmp_in_path)
        os.unlink(tmp_out_path)
        return self

    def add_intro_outro(self, intro_path: str = None, outro_path: str = None,
                         crossfade_ms: int = 2000):
        """イントロ/アウトロの追加"""
        if intro_path:
            intro = AudioSegment.from_file(intro_path)
            # BGMのボリュームを下げる
            intro = intro - 6  # -6dB
            self.audio = intro.append(self.audio, crossfade=crossfade_ms)
            self.edit_log.append("イントロ追加")

        if outro_path:
            outro = AudioSegment.from_file(outro_path)
            outro = outro - 6
            self.audio = self.audio.append(outro, crossfade=crossfade_ms)
            self.edit_log.append("アウトロ追加")

        return self

    def export(self, output_path: str, format: str = "mp3",
               bitrate: str = "192k", tags: dict = None):
        """最終出力"""
        export_params = {"format": format}
        if format == "mp3":
            export_params["bitrate"] = bitrate
        if tags:
            export_params["tags"] = tags

        self.audio.export(output_path, **export_params)
        print(f"出力: {output_path}")
        print(f"長さ: {len(self.audio)/1000:.1f}秒")
        print(f"編集ログ: {', '.join(self.edit_log)}")
        return output_path
```

### 4.3 BGMダッキング

```python
class BGMDucker:
    """話者音声に合わせたBGM自動ダッキング"""

    def __init__(self, duck_db: float = -15.0, attack_ms: int = 200,
                 release_ms: int = 500):
        self.duck_db = duck_db
        self.attack_ms = attack_ms
        self.release_ms = release_ms

    def apply(self, voice_audio: AudioSegment,
              bgm_audio: AudioSegment) -> AudioSegment:
        """話者音声に合わせてBGMをダッキング"""

        # BGMを話者音声の長さに合わせてループ
        while len(bgm_audio) < len(voice_audio):
            bgm_audio = bgm_audio + bgm_audio
        bgm_audio = bgm_audio[:len(voice_audio)]

        # VADで話者区間を検出
        voice_regions = self._detect_voice_regions(voice_audio)

        # ダッキングカーブを生成
        duck_curve = self._create_duck_curve(
            len(voice_audio), voice_regions
        )

        # BGMにダッキングカーブを適用
        ducked_bgm = self._apply_curve(bgm_audio, duck_curve)

        # ミックス
        return voice_audio.overlay(ducked_bgm)

    def _detect_voice_regions(self, audio, chunk_ms=100, threshold_db=-35):
        """話者音声区間を検出"""
        regions = []
        for i in range(0, len(audio), chunk_ms):
            chunk = audio[i:i + chunk_ms]
            if chunk.dBFS > threshold_db:
                regions.append((i, i + chunk_ms))
        return self._merge_regions(regions, gap_ms=300)

    def _merge_regions(self, regions, gap_ms):
        """近接区間をマージ"""
        if not regions:
            return []
        merged = [regions[0]]
        for start, end in regions[1:]:
            if start - merged[-1][1] <= gap_ms:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        return merged

    def _create_duck_curve(self, total_ms, voice_regions):
        """ダッキングカーブ生成（0.0 = フルダック、1.0 = ノーダック）"""
        curve = np.ones(total_ms)
        duck_linear = 10 ** (self.duck_db / 20)

        for start, end in voice_regions:
            # アタック（フェードダウン）
            attack_start = max(0, start - self.attack_ms)
            for i in range(attack_start, start):
                progress = (i - attack_start) / self.attack_ms
                curve[i] = 1.0 - (1.0 - duck_linear) * progress

            # ダック区間
            curve[start:end] = duck_linear

            # リリース（フェードアップ）
            release_end = min(total_ms, end + self.release_ms)
            for i in range(end, release_end):
                progress = (i - end) / self.release_ms
                curve[i] = duck_linear + (1.0 - duck_linear) * progress

        return curve

    def _apply_curve(self, audio, curve):
        """ダッキングカーブをオーディオに適用"""
        samples = np.array(audio.get_array_of_samples(), dtype=np.float64)
        # カーブをサンプル単位に補間
        sample_curve = np.interp(
            np.linspace(0, len(curve), len(samples)),
            np.arange(len(curve)),
            curve,
        )
        ducked_samples = (samples * sample_curve).astype(np.int16)
        return audio._spawn(ducked_samples.tobytes())
```

### 4.4 ノイズ除去ツール比較

| ツール | 手法 | リアルタイム | 品質 | コスト |
|--------|------|-------------|------|--------|
| RNNoise | RNNベースの音声強調 | ○ | 良 | 無料 (OSS) |
| Adobe Podcast Enhance | クラウドAI | x | 優 | 無料 (制限付き) |
| NVIDIA Broadcast | RTXベースAI | ○ | 優 | 無料 (GPU必要) |
| Dolby.io | クラウドAPI | x | 優 | 有料 |
| Auphonic | マルチバンドAI | x | 優 | フリーミアム |
| Descript | AIトランスクリプション+編集 | x | 優 | 有料 |

---

## 5. ポッドキャストホスティングとRSSフィード

### 5.1 RSSフィード自動生成

```python
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from datetime import datetime
import hashlib

class PodcastRSSGenerator:
    """ポッドキャストRSSフィード自動生成"""

    def __init__(self, title: str, description: str, author: str,
                 website: str, image_url: str):
        self.title = title
        self.description = description
        self.author = author
        self.website = website
        self.image_url = image_url
        self.episodes = []

    def add_episode(self, title: str, description: str,
                     audio_url: str, duration_seconds: int,
                     pub_date: datetime, file_size_bytes: int,
                     episode_number: int = None,
                     season_number: int = None,
                     chapters: list = None):
        """エピソードを追加"""
        self.episodes.append({
            "title": title,
            "description": description,
            "audio_url": audio_url,
            "duration": duration_seconds,
            "pub_date": pub_date,
            "file_size": file_size_bytes,
            "episode_number": episode_number,
            "season_number": season_number,
            "guid": hashlib.md5(audio_url.encode()).hexdigest(),
            "chapters": chapters or [],
        })

    def generate_rss(self) -> str:
        """RSS 2.0 + iTunes拡張のXMLを生成"""
        rss = Element("rss", version="2.0")
        rss.set("xmlns:itunes", "http://www.itunes.com/dtds/podcast-1.0.dtd")
        rss.set("xmlns:podcast", "https://podcastindex.org/namespace/1.0")

        channel = SubElement(rss, "channel")
        SubElement(channel, "title").text = self.title
        SubElement(channel, "description").text = self.description
        SubElement(channel, "link").text = self.website
        SubElement(channel, "language").text = "ja"

        # iTunes拡張
        SubElement(channel, "itunes:author").text = self.author
        SubElement(channel, "itunes:summary").text = self.description
        image = SubElement(channel, "itunes:image")
        image.set("href", self.image_url)

        # エピソード
        for ep in sorted(self.episodes, key=lambda e: e["pub_date"], reverse=True):
            item = SubElement(channel, "item")
            SubElement(item, "title").text = ep["title"]
            SubElement(item, "description").text = ep["description"]

            enclosure = SubElement(item, "enclosure")
            enclosure.set("url", ep["audio_url"])
            enclosure.set("length", str(ep["file_size"]))
            enclosure.set("type", "audio/mpeg")

            SubElement(item, "guid", isPermaLink="false").text = ep["guid"]
            SubElement(item, "pubDate").text = ep["pub_date"].strftime(
                "%a, %d %b %Y %H:%M:%S +0900"
            )

            # 再生時間
            h = ep["duration"] // 3600
            m = (ep["duration"] % 3600) // 60
            s = ep["duration"] % 60
            SubElement(item, "itunes:duration").text = f"{h:02d}:{m:02d}:{s:02d}"

            if ep["episode_number"]:
                SubElement(item, "itunes:episode").text = str(ep["episode_number"])
            if ep["season_number"]:
                SubElement(item, "itunes:season").text = str(ep["season_number"])

        xml_str = tostring(rss, encoding="unicode")
        return parseString(xml_str).toprettyxml(indent="  ")
```

### 5.2 配信プラットフォーム自動投稿

```python
class PodcastDistributor:
    """ポッドキャスト配信自動化"""

    def __init__(self):
        self.platforms = {}

    def register_platform(self, name: str, api_client):
        """配信プラットフォームを登録"""
        self.platforms[name] = api_client

    def publish_episode(self, episode_data: dict) -> dict:
        """全プラットフォームにエピソードを配信"""
        results = {}
        for name, client in self.platforms.items():
            try:
                result = client.publish(episode_data)
                results[name] = {"status": "success", "url": result.get("url")}
                print(f"[{name}] 配信完了: {result.get('url')}")
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}
                print(f"[{name}] 配信失敗: {e}")
        return results

    def generate_episode_package(
        self,
        audio_path: str,
        transcript: str,
        metadata: dict,
    ) -> dict:
        """エピソード配信パッケージの一括生成"""
        content_gen = PodcastContentGenerator(self._get_llm_client())

        package = {
            "audio": audio_path,
            "metadata": metadata,
            "show_notes": content_gen._generate_show_notes(transcript, metadata),
            "social_posts": content_gen._generate_social_posts(transcript, metadata),
            "srt_subtitles": self._generate_srt(transcript),
            "vtt_subtitles": self._generate_vtt(transcript),
        }
        return package
```

---

## 6. トラブルシューティング

### 6.1 よくある問題と解決策

```
問題: Whisperがハルシネーション（存在しない音声のテキスト生成）を起こす
==================================================
原因:
- 無音区間が長い
- ノイズが多い音声
- condition_on_previous_text=True での誤った文脈引き継ぎ

解決策:
1. VADフィルタを有効化
   model.transcribe(audio, vad_filter=True)

2. ハルシネーション抑制パラメータの調整
   no_speech_threshold=0.6     # 無音判定を厳しく
   logprob_threshold=-1.0      # 低信頼度セグメントをスキップ
   compression_ratio_threshold=2.4  # 繰り返し検出

3. condition_on_previous_text=False に設定
   （精度は下がるが、ハルシネーションの連鎖を防止）

4. 音声前処理でノイズ除去を先に実行
==================================================

問題: 話者分離の精度が低い（話者が入れ替わる）
==================================================
原因:
- 話者の声質が似ている
- 重なり発話（オーバーラップ）が多い
- 音声品質が低い（リモート収録等）

解決策:
1. 話者数を事前に指定
   diarization(audio, num_speakers=2)

2. 各話者を別マイク・別チャンネルで収録
   （後処理不要の根本解決）

3. 話者の声紋（speaker embedding）を事前登録
   - 各話者の3-10秒のサンプルを用意
   - 事前にembeddingを抽出してリファレンスとして使用

4. リモート収録時は各話者がローカル録音
   （Riverside.fm、Zencastr等のサービスを利用）
==================================================

問題: 長時間音声（3時間超）の処理でメモリ不足
==================================================
原因:
- 音声データ全体をメモリにロード
- GPUメモリ不足

解決策:
1. チャンク分割処理
   - 30分ごとに分割（前後5秒オーバーラップ）
   - 各チャンクを個別に処理

2. faster-whisper + VADフィルタ
   - 無音区間をスキップしてメモリ削減

3. INT8量子化で必要VRAM削減
   compute_type="int8_float16"  # large-v3: 10GB → 5GB

4. CPU処理へのフォールバック
   device="cpu", compute_type="int8"  # 遅いが安定
==================================================
```

### 6.2 パフォーマンスチューニング

```python
# 処理速度の最適化テクニック

class PerformanceOptimizer:
    """ポッドキャスト処理のパフォーマンス最適化"""

    @staticmethod
    def benchmark_models(audio_path: str) -> dict:
        """モデルサイズごとの処理時間を測定"""
        import time
        from faster_whisper import WhisperModel

        results = {}
        for model_size in ["tiny", "base", "small", "medium", "large-v3"]:
            try:
                model = WhisperModel(model_size, device="cuda",
                                     compute_type="int8_float16")
                start = time.time()
                segments, info = model.transcribe(audio_path, language="ja")
                # セグメントを消費（ジェネレータなので）
                text = " ".join(s.text for s in segments)
                elapsed = time.time() - start

                results[model_size] = {
                    "time_seconds": elapsed,
                    "audio_duration": info.duration,
                    "rtf": elapsed / info.duration,  # Real-Time Factor
                    "text_length": len(text),
                }
                print(f"{model_size}: {elapsed:.1f}s (RTF: {elapsed/info.duration:.2f}x)")
            except Exception as e:
                results[model_size] = {"error": str(e)}

        return results

    @staticmethod
    def optimal_settings(audio_duration_minutes: int,
                          gpu_vram_gb: int,
                          quality_priority: str = "balanced") -> dict:
        """最適な設定を推奨"""
        settings = {
            "model_size": "large-v3",
            "compute_type": "float16",
            "beam_size": 5,
            "vad_filter": True,
        }

        # GPU VRAM に応じたモデル選択
        if gpu_vram_gb < 4:
            settings["model_size"] = "small"
            settings["compute_type"] = "int8"
        elif gpu_vram_gb < 8:
            settings["model_size"] = "medium"
            settings["compute_type"] = "int8_float16"
        elif gpu_vram_gb < 12:
            settings["model_size"] = "large-v3"
            settings["compute_type"] = "int8_float16"
        else:
            settings["model_size"] = "large-v3"
            settings["compute_type"] = "float16"

        # 品質優先度に応じた調整
        if quality_priority == "speed":
            settings["beam_size"] = 1
            if settings["model_size"] in ["large-v3", "medium"]:
                settings["model_size"] = "small"
        elif quality_priority == "quality":
            settings["beam_size"] = 10

        # 長時間音声の場合
        if audio_duration_minutes > 120:
            settings["vad_filter"] = True
            settings["chunk_processing"] = True
            settings["chunk_duration_minutes"] = 30

        return settings
```

---

## 7. アンチパターン

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

### アンチパターン 3: 「マスタリングなしで配信」

```
[誤り] 文字起こし・編集に注力し、マスタリングを省略する

問題点:
- プラットフォーム間で音量が統一されない
- リスナーが音量調整を頻繁に行う必要がある
- True Peak超過でクリッピングが発生

[正解] 配信前に必ずラウドネス正規化を実施
  1. ターゲットLUFSを設定（Apple Podcasts: -16, Spotify: -14）
  2. True Peakを-1.0dBTP以下に制限
  3. ノイズフロアが-50dB以下であることを確認
  4. 全エピソードで一貫したラウドネスを維持
```

---

## 8. ベストプラクティス

### 8.1 ポッドキャスト制作のベストプラクティス

```
収録段階:
==================================================
1. マイクの選定
   - USB: Blue Yeti, Rode NT-USB+ (手軽さ重視)
   - XLR: Shure SM7B, Rode PodMic (品質重視)
   - ラべリア: Rode Wireless GO II (リモート・屋外)

2. 録音環境の最適化
   - 吸音材の設置（反響を -10dB 以上低減）
   - エアコン・ファンの停止（ノイズフロア -50dB 以下目標）
   - ポップフィルター使用（破裂音除去）

3. 録音設定
   - サンプルレート: 44.1kHz または 48kHz
   - ビット深度: 24bit（ヘッドルーム確保）
   - ゲイン: ピーク -6dB 程度（クリッピング防止）
   - 各話者を別トラックで録音（後処理の柔軟性）

後処理段階:
==================================================
4. 標準後処理フロー
   Step 1: ノイズ除去（RNNoise or Adobe Enhance）
   Step 2: フィラー除去（AI検出 → 手動確認 → 削除）
   Step 3: EQ（ハイパス80Hz + プレゼンスブースト）
   Step 4: コンプレッション（-20dB threshold, ratio 3:1）
   Step 5: ラウドネス正規化（-16 LUFS for Apple Podcasts）
   Step 6: True Peakリミッティング（-1.0 dBTP）

5. 文字起こし・要約
   - faster-whisper large-v3 + VAD でベースライン
   - ドメイン特化プロンプトで精度向上
   - LLM でショーノート・チャプター自動生成
   - 必ず人間レビューを挟む

配信段階:
==================================================
6. ファイルフォーマット
   - MP3: 128-192kbps CBR（最も互換性が高い）
   - AAC: 128kbps（Apple推奨）
   - ID3タグ: タイトル、アーティスト、アートワーク必須

7. メタデータの最適化
   - タイトル: 検索しやすいキーワードを含む
   - 説明文: 最初の2文が検索結果に表示される
   - チャプターマーク: Apple Podcasts で対応
   - 文字起こし: SEO効果 + アクセシビリティ向上
```

---

## 9. FAQ

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

### Q4: ポッドキャストの音質を手軽に改善する最も効果的な方法は？

**A:** コストパフォーマンスの高い順に、(1) Adobe Podcast Enhance（無料、ウェブ上でワンクリック）、(2) ハイパスフィルタ（80Hz以下カット）、(3) ラウドネス正規化（-16 LUFS）。この3つだけで、聴感上の品質は大幅に改善します。特にAdobe Podcast Enhanceは、ノイズ除去・残響除去・EQを自動で行い、スタジオ品質に近づけてくれます。

### Q5: AI文字起こしの結果を効率的にレビューするには？

**A:** 全文を読むのではなく、以下の効率的レビュー手法を推奨します。(1) 信頼度スコアが低いセグメントのみをレビュー対象にする（avg_logprob < -0.8）。(2) 固有名詞リストを事前に作成し、誤認識がないか一括検索。(3) Descriptなどのテキストベース編集ツールを使い、テキストを修正すると音声も連動して編集される。(4) 修正データを蓄積し、定期的にファインチューニングに活用。

### Q6: ポッドキャストの収益化にAIをどう活用できますか？

**A:** AIを収益化の複数段階で活用できます。(1) 文字起こしからSEO最適化されたブログ記事を自動生成し、検索流入を増加。(2) ソーシャルメディア投稿の自動生成で認知度向上。(3) チャプター分割と要約で聴取体験を改善し、リスナー維持率を向上。(4) 多言語翻訳（Whisperの翻訳機能 + LLM）でグローバル展開。(5) 有料メンバー向けに完全な文字起こしや拡張ショーノートを提供。

---

## 10. まとめ

| 機能 | 技術 | 推奨ツール | 精度 |
|------|------|-----------|------|
| 文字起こし | ASR (Whisper) | faster-whisper, Whisper API | 95%+ (日本語) |
| 話者分離 | Speaker Diarization | pyannote.audio 3.1 | 90%+ |
| 要約生成 | LLM | GPT-4o, Claude | 高品質 |
| チャプター分割 | トピック検出 | 埋め込み + 境界検出 | 85%+ |
| フィラー除去 | ASR + ルール | Whisper + 正規表現 | 80%+ |
| ノイズ除去 | 音声強調 AI | RNNoise, Auphonic | 高品質 |
| BGMダッキング | VAD + ゲイン制御 | pydub + カスタム | 90%+ |
| 品質チェック | 信号分析 | pyloudnorm, カスタム | 自動化 |

---

## 次に読むべきガイド

- [リアルタイム音声](../03-development/02-real-time-audio.md) — WebRTC、ストリーミング STT/TTS の実装
- [音声エフェクト](../01-music/02-audio-effects.md) — AI EQ、ノイズ除去、マスタリング
- [STT技術](../00-fundamentals/03-stt-technologies.md) — Whisper、Google Speech、Azure Speech の詳細

---

## 参考文献

1. Radford, A. et al. (2023). "Robust Speech Recognition via Large-Scale Weak Supervision." *Proceedings of the 40th International Conference on Machine Learning (ICML 2023)*. OpenAI. https://arxiv.org/abs/2212.04356
2. Bredin, H. et al. (2023). "pyannote.audio 2.1: speaker diarization pipeline." *INTERSPEECH 2023*. https://doi.org/10.21437/Interspeech.2023-105
3. Park, T.J. et al. (2022). "A Review of Speaker Diarization: Recent Advances with Deep Learning." *Computer Speech & Language, 72*. https://doi.org/10.1016/j.csl.2021.101317
4. ITU-R BS.1770-5 (2023). "Algorithms to measure audio programme loudness and true-peak audio level" — ラウドネス測定の国際規格
5. Apple (2024). "Apple Podcasts for Creators: Audio Requirements" — Apple Podcasts配信のオーディオ要件仕様
