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

### 1.3 AI 動画編集の進化タイムライン

```
2019  基礎技術の成熟
  │     └─ 自動字幕（YouTube 自動字幕の精度向上）
  │
2020  Descript 登場
  │     └─ テキストベース動画編集の革新
  │
2021  Whisper (OpenAI)
  │     └─ 多言語高精度文字起こしの民主化
  │
2022  Runway Gen-1 / SAM
  │     └─ AI 映像効果とセグメンテーション
  │
2023  AI 動画編集の統合化
  │     └─ CapCut AI、DaVinci Resolve AI 機能拡充
  │
2024  エンドツーエンド AI 編集
  │     └─ プロンプトベースの編集指示
  │   ProPainter / Track Anything
  │     └─ 動画内オブジェクト除去の高品質化
  │
2025  マルチモーダル編集
        └─ テキスト指示で動画全体を自動編集
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

### 2.3 Whisper + 話者分離の統合パイプライン

```python
# 話者分離付き文字起こしの完全パイプライン
import whisper
from pyannote.audio import Pipeline
from dataclasses import dataclass

@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str
    speaker: str

class SpeakerAwareTranscriber:
    """話者情報付き文字起こし"""

    def __init__(self, whisper_model: str = "large-v3", hf_token: str = ""):
        self.whisper = whisper.load_model(whisper_model)
        self.diarization = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )

    def transcribe_with_speakers(
        self,
        audio_path: str,
        language: str = "ja",
        max_speakers: int = None,
    ) -> list[TranscriptionSegment]:
        """話者分離付き文字起こし"""

        # 1. Whisper で文字起こし
        whisper_result = self.whisper.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
        )

        # 2. pyannote で話者分離
        dia_params = {}
        if max_speakers:
            dia_params["max_speakers"] = max_speakers
        dia_result = self.diarization(audio_path, **dia_params)

        # 3. 話者情報と文字起こしを統合
        segments = []
        for seg in whisper_result["segments"]:
            # セグメントの中間時刻で話者を判定
            mid_time = (seg["start"] + seg["end"]) / 2
            speaker = self._get_speaker_at_time(dia_result, mid_time)

            segments.append(TranscriptionSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
                speaker=speaker,
            ))

        return segments

    def _get_speaker_at_time(self, dia_result, time: float) -> str:
        """指定時刻の話者を取得"""
        for turn, _, speaker in dia_result.itertracks(yield_label=True):
            if turn.start <= time <= turn.end:
                return speaker
        return "UNKNOWN"

    def export_to_srt(
        self,
        segments: list[TranscriptionSegment],
        output_path: str,
        include_speaker: bool = True,
    ):
        """SRT 形式でエクスポート"""
        lines = []
        for i, seg in enumerate(segments, 1):
            start = self._format_time(seg.start)
            end = self._format_time(seg.end)
            text = f"[{seg.speaker}] {seg.text}" if include_speaker else seg.text
            lines.append(f"{i}\n{start} --> {end}\n{text}\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _format_time(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# 使用例
transcriber = SpeakerAwareTranscriber(
    whisper_model="large-v3",
    hf_token="your-hf-token",
)

segments = transcriber.transcribe_with_speakers(
    "interview.wav",
    language="ja",
    max_speakers=2,
)

transcriber.export_to_srt(segments, "interview_subtitles.srt")

# 出力例:
# 1
# 00:00:01,200 --> 00:00:05,800
# [SPEAKER_00] 今日はAI動画編集についてお話しします
#
# 2
# 00:00:06,100 --> 00:00:10,500
# [SPEAKER_01] はい、よろしくお願いします
```

### 2.4 字幕のスタイリングと自動翻訳

```python
# 字幕の自動翻訳とスタイリング

from dataclasses import dataclass
from typing import Optional

@dataclass
class SubtitleStyle:
    """字幕のスタイル設定"""
    font_family: str = "Noto Sans JP"
    font_size: int = 48
    font_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline_width: int = 3
    background_color: Optional[str] = None  # None=透明
    position: str = "bottom_center"  # top, bottom, center
    max_chars_per_line: int = 20

class SubtitleProcessor:
    """字幕の後処理・翻訳"""

    def auto_split_long_lines(
        self, text: str, max_chars: int = 20
    ) -> str:
        """長い字幕テキストを改行で分割"""
        if len(text) <= max_chars:
            return text

        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line + word) > max_chars and current_line:
                lines.append(current_line.strip())
                current_line = word + " "
            else:
                current_line += word + " "

        if current_line.strip():
            lines.append(current_line.strip())

        return "\n".join(lines)

    def translate_subtitles(
        self,
        segments: list,
        target_lang: str = "en",
        service: str = "deepl",
    ) -> list:
        """字幕を翻訳"""
        if service == "deepl":
            return self._translate_with_deepl(segments, target_lang)
        elif service == "gpt4":
            return self._translate_with_gpt4(segments, target_lang)
        return segments

    def _translate_with_gpt4(self, segments: list, target_lang: str) -> list:
        """GPT-4 を使った高品質翻訳（文脈考慮）"""
        from openai import OpenAI
        client = OpenAI()

        # 全文を一括翻訳（文脈を維持するため）
        all_texts = [seg.text for seg in segments]
        numbered_texts = "\n".join(
            f"{i}: {text}" for i, text in enumerate(all_texts)
        )

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": f"Translate the following numbered lines to {target_lang}. "
                               f"Keep the numbering. Maintain natural conversation flow.",
                },
                {"role": "user", "content": numbered_texts},
            ],
        )

        # 翻訳結果をパース
        translated_lines = response.choices[0].message.content.strip().split("\n")
        for i, seg in enumerate(segments):
            if i < len(translated_lines):
                # "0: translated text" → "translated text"
                parts = translated_lines[i].split(": ", 1)
                seg.text = parts[1] if len(parts) > 1 else parts[0]

        return segments

    def generate_ass_file(
        self,
        segments: list,
        style: SubtitleStyle,
        output_path: str,
    ):
        """ASS (Advanced SubStation Alpha) 形式で出力"""
        header = f"""[Script Info]
Title: AI Generated Subtitles
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV
Style: Default,{style.font_family},{style.font_size},&H00FFFFFF,&H00000000,&H00000000,0,0,1,{style.outline_width},0,2,10,10,40

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        events = []
        for seg in segments:
            start = self._to_ass_time(seg.start)
            end = self._to_ass_time(seg.end)
            text = seg.text.replace("\n", "\\N")
            events.append(
                f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}"
            )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header + "\n".join(events))

    def _to_ass_time(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int((seconds % 1) * 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
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

### 3.1 高度なシーン検出パイプライン

```python
# 複数の検出手法を組み合わせた高精度シーン検出

from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector, ThresholdDetector, AdaptiveDetector

class AdvancedSceneDetector:
    """複数手法によるシーン検出"""

    def __init__(self):
        self.detectors = {
            "content": ContentDetector(threshold=27.0),
            "adaptive": AdaptiveDetector(
                adaptive_threshold=3.0,
                min_scene_len=15,  # 最低15フレーム
            ),
        }

    def detect_scenes(
        self,
        video_path: str,
        method: str = "adaptive",
        min_scene_duration: float = 1.0,
    ) -> list:
        """シーン検出を実行"""
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(self.detectors[method])
        scene_manager.detect_scenes(video)

        scene_list = scene_manager.get_scene_list()

        # 最小シーン長でフィルタリング
        fps = video.frame_rate
        min_frames = int(min_scene_duration * fps)
        filtered = [
            scene for scene in scene_list
            if (scene[1] - scene[0]).get_frames() >= min_frames
        ]

        return filtered

    def classify_scenes(self, video_path: str, scenes: list) -> list:
        """
        各シーンを分類

        分類カテゴリ:
        - dialogue: 人物の会話シーン
        - action: 動きの多いシーン
        - transition: トランジション
        - static: 静的なシーン（スライド、テロップ等）
        """
        classified = []
        for scene in scenes:
            # フレームの動き量を計算
            motion = self._calculate_motion(video_path, scene)
            # 音声解析
            has_speech = self._detect_speech(video_path, scene)

            if has_speech and motion < 0.3:
                category = "dialogue"
            elif motion > 0.7:
                category = "action"
            elif motion < 0.1:
                category = "static"
            else:
                category = "general"

            classified.append({
                "start": scene[0].get_timecode(),
                "end": scene[1].get_timecode(),
                "category": category,
                "motion_score": motion,
                "has_speech": has_speech,
            })

        return classified

    def auto_highlight(
        self,
        video_path: str,
        target_duration: float = 60.0,
        priority: list = None,
    ) -> list:
        """
        自動ハイライト生成

        動きの多いシーンと会話シーンを優先的に選択し、
        指定時間に収まるようにシーンを選択する
        """
        scenes = self.detect_scenes(video_path)
        classified = self.classify_scenes(video_path, scenes)

        if priority is None:
            priority = ["action", "dialogue", "general", "static"]

        # 優先度に基づいてソート
        scored = []
        for scene in classified:
            priority_score = (
                len(priority) - priority.index(scene["category"])
                if scene["category"] in priority else 0
            )
            scored.append((priority_score, scene))

        scored.sort(reverse=True, key=lambda x: x[0])

        # 目標時間に収まるようにシーンを選択
        selected = []
        total_duration = 0
        for _, scene in scored:
            # シーン長を計算（簡易）
            scene_duration = 3.0  # 仮の値
            if total_duration + scene_duration <= target_duration:
                selected.append(scene)
                total_duration += scene_duration

        return selected

    def _calculate_motion(self, video_path, scene):
        """シーンの動き量を計算"""
        return 0.5  # 実際にはフレーム差分で計算

    def _detect_speech(self, video_path, scene):
        """シーンに音声があるか検出"""
        return True  # 実際にはVADで検出
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

### 4.3 オブジェクト除去（ProPainter）

```python
# ProPainter による動画内オブジェクト除去

import cv2
import numpy as np
from pathlib import Path

class VideoObjectRemover:
    """動画からオブジェクトを除去"""

    def __init__(self, model_path: str = "propainter_weights"):
        self.model = self._load_model(model_path)

    def remove_object(
        self,
        video_path: str,
        mask_dir: str,
        output_path: str,
        flow_completion: bool = True,
    ):
        """
        動画からオブジェクトを除去

        video_path: 入力動画
        mask_dir: フレームごとのマスク画像ディレクトリ
          白=除去対象、黒=保持
        output_path: 出力動画
        flow_completion: オプティカルフロー補完を使用
        """
        # 1. 動画をフレームに分解
        frames = self._extract_frames(video_path)
        masks = self._load_masks(mask_dir, len(frames))

        # 2. オプティカルフローの計算
        if flow_completion:
            flows_forward = self._compute_flow(frames, direction="forward")
            flows_backward = self._compute_flow(frames, direction="backward")
            # マスク領域のフロー補完
            flows_forward = self._complete_flow(flows_forward, masks)
            flows_backward = self._complete_flow(flows_backward, masks)

        # 3. 時空間注意機構によるインペインティング
        inpainted_frames = self.model.inpaint(
            frames=frames,
            masks=masks,
            flows_f=flows_forward if flow_completion else None,
            flows_b=flows_backward if flow_completion else None,
        )

        # 4. 動画として書き出し
        self._write_video(inpainted_frames, output_path, fps=30)

    def track_and_remove(
        self,
        video_path: str,
        initial_mask: str,
        output_path: str,
    ):
        """
        最初のフレームのマスクからオブジェクトを追跡して除去
        SAM + Track Anything を組み合わせ
        """
        # 1. SAM でセグメンテーション
        # 2. 後続フレームで追跡
        # 3. 全フレームのマスクを生成
        # 4. ProPainter で除去
        pass

    def _extract_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def _load_masks(self, mask_dir, num_frames):
        masks = []
        for i in range(num_frames):
            mask_path = Path(mask_dir) / f"mask_{i:04d}.png"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                masks.append(mask)
            else:
                masks.append(np.zeros_like(masks[0]) if masks else None)
        return masks

    def _write_video(self, frames, output_path, fps=30):
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps, (w, h)
        )
        for frame in frames:
            writer.write(frame)
        writer.release()

    def _compute_flow(self, frames, direction):
        """RAFT によるオプティカルフロー計算"""
        pass

    def _complete_flow(self, flows, masks):
        """マスク領域のフロー補完"""
        pass

    def _load_model(self, model_path):
        """ProPainter モデルのロード"""
        pass
```

### 4.4 AI 音声処理

```python
# Demucs による音声分離

class AudioSeparator:
    """AI による音声分離"""

    def separate(
        self,
        audio_path: str,
        output_dir: str,
        model: str = "htdemucs_ft",
    ) -> dict:
        """
        音声をステムに分離

        出力ステム:
        - vocals: ボーカル/音声
        - drums: ドラム
        - bass: ベース
        - other: その他の楽器
        """
        import subprocess

        cmd = [
            "python", "-m", "demucs",
            "--model", model,
            "--out", output_dir,
            audio_path,
        ]
        subprocess.run(cmd, check=True)

        stem_dir = Path(output_dir) / model / Path(audio_path).stem
        return {
            "vocals": str(stem_dir / "vocals.wav"),
            "drums": str(stem_dir / "drums.wav"),
            "bass": str(stem_dir / "bass.wav"),
            "other": str(stem_dir / "other.wav"),
        }

    def remove_background_music(
        self,
        video_path: str,
        output_path: str,
    ):
        """動画から BGM を除去し、音声のみ残す"""
        import subprocess

        # 1. 音声を抽出
        subprocess.run([
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "temp_audio.wav"
        ])

        # 2. 音声分離
        stems = self.separate("temp_audio.wav", "./separated")

        # 3. ボーカルのみで動画を再構成
        subprocess.run([
            "ffmpeg",
            "-i", video_path,
            "-i", stems["vocals"],
            "-c:v", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            output_path,
        ])

    def enhance_voice(
        self,
        audio_path: str,
        output_path: str,
    ):
        """音声のノイズ除去と品質向上"""
        # 1. 音声分離でボーカルを抽出
        # 2. ノイズ除去（spectral gating）
        # 3. EQ調整（音声帯域を強調）
        # 4. ラウドネス正規化
        pass
```

---

## 5. 自動編集ワークフロー

### 5.1 YouTube 動画の自動編集パイプライン

```python
# YouTube 動画の自動編集パイプライン

class YouTubeAutoEditor:
    """YouTube 動画の自動編集"""

    def __init__(self):
        self.whisper_model = whisper.load_model("large-v3")
        self.scene_detector = AdvancedSceneDetector()

    def auto_edit(
        self,
        raw_video: str,
        output_video: str,
        config: dict = None,
    ):
        """
        自動編集フルパイプライン

        config:
          silence_threshold: 無音カットの閾値 (dB)
          silence_duration: 無音判定の最小長 (秒)
          padding: カット前後の余白 (秒)
          target_duration: 目標動画長 (秒, None=制限なし)
          add_subtitles: 字幕を追加するか
          subtitle_lang: 字幕言語
        """
        if config is None:
            config = {
                "silence_threshold": -35,
                "silence_duration": 0.8,
                "padding": 0.1,
                "target_duration": None,
                "add_subtitles": True,
                "subtitle_lang": "ja",
            }

        print("Step 1: 文字起こし...")
        transcription = self.whisper_model.transcribe(
            raw_video,
            language=config["subtitle_lang"],
            word_timestamps=True,
        )

        print("Step 2: 無音区間検出...")
        silences = detect_silence(
            raw_video,
            threshold=config["silence_threshold"],
            duration=config["silence_duration"],
        )

        print("Step 3: シーン検出...")
        scenes = self.scene_detector.detect_scenes(raw_video)

        print("Step 4: 編集ポイントの決定...")
        edit_points = self._merge_edit_points(
            silences=silences,
            scenes=scenes,
            transcription=transcription,
            padding=config["padding"],
        )

        print("Step 5: 動画のカット & 結合...")
        self._apply_edits(raw_video, edit_points, output_video)

        if config["add_subtitles"]:
            print("Step 6: 字幕の追加...")
            srt_path = output_video.replace(".mp4", ".srt")
            self._generate_subtitles(transcription, srt_path)

        print(f"自動編集完了: {output_video}")

    def _merge_edit_points(self, silences, scenes, transcription, padding):
        """無音カット、シーン検出、文字起こしを統合して編集ポイントを決定"""
        # 音声がある区間を保持
        keep_regions = []
        # 無音区間の逆（音声がある区間）を計算
        # シーン境界で自然なカットポイントを選択
        # 言葉の途中でのカットを避ける
        return keep_regions

    def _apply_edits(self, input_video, edit_points, output_video):
        """FFmpeg で編集を適用"""
        pass

    def _generate_subtitles(self, transcription, output_path):
        """字幕ファイルを生成"""
        srt = to_srt(transcription["segments"])
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt)
```

### 5.2 バッチ処理パイプライン

```python
# 複数動画の一括処理

import concurrent.futures
from pathlib import Path

class BatchVideoProcessor:
    """複数動画のバッチ処理"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def process_batch(
        self,
        input_dir: str,
        output_dir: str,
        operations: list,
    ):
        """
        ディレクトリ内の全動画を処理

        operations: 適用する処理のリスト
          例: ["transcribe", "remove_silence", "upscale", "subtitle"]
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        videos = list(Path(input_dir).glob("*.mp4"))

        print(f"処理対象: {len(videos)} 本の動画")

        results = {}
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = {}
            for video in videos:
                output = Path(output_dir) / video.name
                future = executor.submit(
                    self._process_single, str(video), str(output), operations
                )
                futures[future] = video.name

            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    results[name] = {"status": "success", **result}
                    print(f"  完了: {name}")
                except Exception as e:
                    results[name] = {"status": "error", "error": str(e)}
                    print(f"  失敗: {name} - {e}")

        return results

    def _process_single(self, input_path, output_path, operations):
        """単一動画の処理"""
        result = {}
        for op in operations:
            if op == "transcribe":
                result["transcription"] = self._transcribe(input_path)
            elif op == "remove_silence":
                input_path = self._remove_silence(input_path)
            elif op == "upscale":
                input_path = self._upscale(input_path)
            elif op == "subtitle":
                self._add_subtitle(input_path, output_path)
        return result

    def _transcribe(self, path):
        pass

    def _remove_silence(self, path):
        return path

    def _upscale(self, path):
        return path

    def _add_subtitle(self, input_path, output_path):
        pass
```

---

## 6. 主要ツール比較

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
| 教育コンテンツ | Descript + CapCut | 字幕重視、コスト効率 |
| 企業プレゼン | CapCut / Canva | テンプレート活用、簡単操作 |

### ツール別 AI 機能詳細比較

| AI 機能 | 品質 | 速度 | コスト | 推奨ツール |
|---------|:----:|:----:|:-----:|-----------|
| 自動字幕 (日本語) | ★★★★★ | 高速 | 低 | Whisper (ローカル) |
| 自動字幕 (多言語) | ★★★★☆ | 高速 | 中 | Descript |
| テキストベース編集 | ★★★★★ | 即時 | 中 | Descript |
| オブジェクト除去 | ★★★★☆ | 中速 | 高 | Runway |
| 背景除去 | ★★★★☆ | 高速 | 低 | CapCut |
| 音声分離 | ★★★★★ | 中速 | 低 | Demucs (ローカル) |
| カラーグレーディング | ★★★★★ | 即時 | 中 | DaVinci Resolve |
| フレーム補間 | ★★★★☆ | 低速 | 低 | RIFE (ローカル) |
| 超解像 | ★★★★☆ | 低速 | 低 | Real-ESRGAN (ローカル) |

---

## 7. アンチパターン

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

### アンチパターン 3: 無音カットの過剰適用

```
BAD:
  全ての無音を機械的にカット
  → 話の「間」が失われ、聞き取りにくい
  → 息つく暇のない不自然なテンポ

GOOD:
  - 意図的な「間」は残す（2秒以下の無音は保持）
  - カット前後にパディング（0.1-0.2秒）を追加
  - 話者の交代時は少し長めの間を残す
  - テーマ転換時のポーズを尊重
```

### アンチパターン 4: Whisper のモデルサイズを考慮しない

```
BAD:
  全ての文字起こしに large-v3 を使用
  → 短い動画でも処理に時間がかかる
  → GPU リソースの無駄遣い

GOOD:
  - ドラフト段階: base / medium で高速処理
  - 最終版: large-v3 で高精度処理
  - リアルタイム用: tiny / base（CPU動作可能）
  - 言語が明確: language パラメータを指定して精度向上
```

---

## 8. パフォーマンス最適化ガイド

### 8.1 処理速度の最適化

```
処理パイプラインの最適化テクニック:

1. プロキシ編集
   - 元素材: 4K / 60fps
   - 編集用プロキシ: 720p / 30fps
   - 最終出力時に元素材で再レンダリング
   - → 編集速度が 4-8 倍高速化

2. GPU パイプライン
   - Whisper: GPU で 10 倍高速化
   - Real-ESRGAN: GPU 必須、タイル処理で VRAM 節約
   - RIFE: GPU で リアルタイム処理可能
   - → CPU 比較で全体 5-10 倍高速化

3. 並列処理
   - 音声処理と映像処理を並列実行
   - フレーム単位の処理はマルチプロセス化
   - 複数動画のバッチ処理
   - → スループット 2-4 倍向上

4. キャッシュ戦略
   - 文字起こし結果をキャッシュ
   - シーン検出結果をキャッシュ
   - 中間フレームを一時ファイルに保存
   - → 再処理時間を 90% 削減
```

### 8.2 品質管理チェックリスト

```
動画品質チェックリスト:

映像:
  □ 解像度が出力要件を満たしている
  □ フレームレートが一定（ドロップフレームなし）
  □ カラーグレーディングが統一されている
  □ AI アーティファクトが目立たない
  □ トランジションが自然

音声:
  □ 音量レベルが統一されている（LUFS 基準）
  □ ノイズが除去されている
  □ BGM と音声のバランスが適切
  □ 話者の声がクリア

字幕:
  □ 誤字・脱字がない
  □ タイミングが音声と一致している
  □ 固有名詞が正しい
  □ 改行位置が適切
  □ フォントサイズと色が読みやすい
```

---

## 9. FAQ

### Q1. Whisper の精度を向上させるには？

**A.** (1) `large-v3` モデルを使用する（精度最高だが処理速度は遅い）。(2) 言語を明示的に指定する（`language="ja"`）。(3) 初期プロンプト（`initial_prompt`）で専門用語やコンテキストを与える。(4) ノイズの多い音声は事前に音声分離（Demucs）でボーカルを抽出してから処理する。

### Q2. 長尺動画の編集を効率化するには？

**A.** (1) Whisper で文字起こし → テキストベースで不要部分を特定。(2) Descript のテキストベース編集で、テキストの削除 = 動画の該当部分カット。(3) PySceneDetect でシーン分割 → 不要シーンを除外。(4) 無音部分の自動カット。これらの組み合わせで10時間の素材を1-2時間で粗編集できる。

### Q3. AI 動画編集の処理を高速化するには？

**A.** (1) GPU を活用する（NVIDIA CUDA 対応の GPU で Whisper は10倍高速化）。(2) プロキシ編集（低解像度で編集→最終書き出しで高解像度に切替）。(3) バッチ処理（複数動画を並列処理）。(4) クラウド GPU（Google Colab、Runway Cloud）を活用。ローカル処理にこだわらず、クラウドとのハイブリッド運用が現実的。

### Q4. Descript のテキストベース編集の仕組みは？

**A.** Descript は動画の音声を文字起こしし、テキストとタイムラインを完全に同期させる。テキストエディタで文章を削除すると、対応する動画の区間も自動的にカットされる。逆に、テキストの並べ替えで動画の構成を変更できる。「um」「えーと」といったフィラーワードの自動検出・削除機能もある。テキストベースで大まかな編集を行い、タイムラインで微調整する2段階ワークフローが効率的。

### Q5. AI で動画の BGM を自動生成できるか？

**A.** 可能。(1) **Suno AI**: テキストプロンプトから楽曲を生成。動画の雰囲気に合わせた BGM を作成できる。(2) **Udio**: 高品質な音楽生成。(3) **Stable Audio**: Stability AI の音楽生成モデル。(4) **Mubert**: API 連携可能な BGM 生成サービス。いずれも商用利用にはライセンス確認が必要。動画の長さに合わせた尺調整や、シーン転換に合わせた曲調変化は、現時点では人間の調整が必要。

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
| 自動化 | 無音カット + 字幕生成の自動パイプラインが最も投資効果高い |
| 最適化 | GPU 活用 + プロキシ編集 + バッチ処理で大幅高速化 |

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
4. **ProPainter** -- Zhou et al. (ICCV 2023) -- 動画インペインティング
5. **Demucs** -- Rouard et al. (2023) -- 音声分離モデル
6. **PySceneDetect** -- https://www.scenedetect.com/ -- シーン検出ライブラリ
