# 音楽生成 — Suno、Udio、MusicGen

> AIによる音楽生成技術の仕組み、主要サービスの比較、プロンプトエンジニアリングを解説する

## この章で学ぶこと

1. AI音楽生成の技術的アーキテクチャ（コーデック言語モデル、拡散モデル）
2. 主要サービス（Suno、Udio、MusicGen）の特徴と使い分け
3. 効果的なプロンプトの書き方と音楽生成ワークフロー

---

## 1. AI音楽生成の技術基盤

### 1.1 音楽生成モデルのアーキテクチャ分類

```
AI音楽生成の主要アーキテクチャ
==================================================

1. コーデック言語モデル型（MusicGen, MusicLM）
   テキスト → LM → 音声コーデック → 波形
   ┌──────┐   ┌──────────┐   ┌─────────┐   ┌──────┐
   │Prompt│──→│Language  │──→│Codec    │──→│Audio │
   │      │   │Model     │   │Decoder  │   │Wave  │
   └──────┘   │(Transformer)│ │(Encodec)│   └──────┘
              └──────────┘   └─────────┘

2. 拡散モデル型（Stable Audio, Riffusion）
   テキスト → 潜在空間で拡散 → デコード → 波形
   ┌──────┐   ┌──────────┐   ┌─────────┐   ┌──────┐
   │Prompt│──→│Diffusion │──→│VAE      │──→│Audio │
   │      │   │(UNet)    │   │Decoder  │   │Wave  │
   └──────┘   └──────────┘   └─────────┘   └──────┘

3. ハイブリッド型（Suno, Udio）
   テキスト + 歌詞 → 複合モデル → 音楽（ボーカル+伴奏）
   ┌──────────┐   ┌───────────────┐   ┌──────┐
   │Prompt    │──→│Multi-stage    │──→│Full  │
   │+ Lyrics  │   │Generation     │   │Song  │
   └──────────┘   │(独自アーキ)    │   └──────┘
                  └───────────────┘
==================================================
```

### 1.2 Encodecによる音声トークン化

```python
# Encodecの概念: 音声をトークン列に変換

class EncodecConcept:
    """
    Encodec（Meta）: ニューラル音声コーデック
    - 音声波形 → 離散トークン列 → 音声波形
    - 複数のコードブック（残差量子化）
    - 帯域幅に応じた品質制御
    """

    def __init__(self):
        self.n_codebooks = 8        # コードブック数
        self.codebook_size = 1024   # 各コードブックの語彙サイズ
        self.frame_rate = 75        # 75 Hz（1秒 = 75フレーム）

    def encode(self, audio_waveform):
        """音声 → トークン列"""
        # Step 1: エンコーダ（CNN）で特徴抽出
        features = self.encoder(audio_waveform)
        # Step 2: 残差量子化（RVQ）で離散化
        # 1つ目のコードブック: 粗い表現
        # 2つ目以降: 前の量子化誤差を補正
        codes = self.quantizer(features)  # shape: (n_codebooks, n_frames)
        return codes  # 例: (8, 75) for 1秒の音声

    def decode(self, codes):
        """トークン列 → 音声"""
        features = self.dequantizer(codes)
        audio = self.decoder(features)
        return audio

# MusicGenでの利用
# テキスト → Transformer LM → Encodecコード → Encodecデコーダ → 音声
```

### 1.3 テキスト-音楽アライメントの仕組み

```python
# CLAP (Contrastive Language-Audio Pretraining) による
# テキストと音楽の意味的対応付け

class CLAPConcept:
    """
    CLAP: テキストと音声を共通の埋め込み空間にマッピング
    - 画像のCLIPと同様のアーキテクチャ
    - テキストエンコーダ + オーディオエンコーダ
    - 対照学習で同じ意味の組を近づける
    """

    def __init__(self):
        self.text_encoder = None  # BERT/RoBERTa ベース
        self.audio_encoder = None  # HTS-AT / HTSAT ベース
        self.embedding_dim = 512

    def compute_similarity(self, text: str, audio_path: str) -> float:
        """テキストと音声の類似度を計算"""
        text_embedding = self.text_encoder(text)  # (512,)
        audio_embedding = self.audio_encoder(audio_path)  # (512,)

        # コサイン類似度
        similarity = np.dot(text_embedding, audio_embedding) / (
            np.linalg.norm(text_embedding) * np.linalg.norm(audio_embedding)
        )
        return similarity

    def rank_generations(self, prompt: str, audio_paths: list) -> list:
        """生成された複数の音楽をプロンプトとの一致度でランキング"""
        scores = []
        for path in audio_paths:
            score = self.compute_similarity(prompt, path)
            scores.append((score, path))
        return sorted(scores, reverse=True)

# 使用例: 生成候補のランキング
# clap = CLAPConcept()
# ranked = clap.rank_generations(
#     "upbeat electronic dance music",
#     ["gen_1.wav", "gen_2.wav", "gen_3.wav"]
# )
```

### 1.4 条件付き生成の詳細メカニズム

```
条件付き音楽生成のメカニズム
==================================================

1. Classifier-Free Guidance (CFG)
   ┌────────────────────────────────────────┐
   │ output = (1 + w) * cond_output         │
   │          - w * uncond_output            │
   │                                        │
   │ w = CFGスケール（大きいほどプロンプト忠実）│
   │ cond_output = プロンプト条件付き出力     │
   │ uncond_output = 無条件出力              │
   └────────────────────────────────────────┘

   CFGスケールの影響:
   - w = 1.0: 弱い条件付け（多様性高い）
   - w = 3.0: 標準（バランス）
   - w = 5.0+: 強い条件付け（プロンプト忠実だが多様性低い）

2. メロディ条件付き生成
   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │Reference │──→│Chromagram│──→│Cross-    │──→ 出力
   │Melody    │   │Extraction│   │Attention │
   └──────────┘   └──────────┘   │with LM   │
                                 └──────────┘
   * クロマグラム: 12半音のエネルギー分布
   * メロディの輪郭を保持しつつ新しい音楽を生成

3. オーディオ条件付き生成（AudioGen的アプローチ）
   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │Reference │──→│Audio     │──→│Concat    │──→ 出力
   │Audio     │   │Encoder   │   │Prompting │
   └──────────┘   └──────────┘   └──────────┘
   * 参照音声のスタイルを引き継いで生成
==================================================
```

---

## 2. 主要サービスの詳細

### 2.1 MusicGen（Meta）

```python
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# MusicGen モデルのロード
model = MusicGen.get_pretrained("facebook/musicgen-large")
model.set_generation_params(
    duration=30,          # 生成時間（秒）
    top_k=250,           # Top-K サンプリング
    top_p=0.0,           # Top-P サンプリング（0=無効）
    temperature=1.0,     # 温度パラメータ
    cfg_coef=3.0,        # Classifier-Free Guidance 係数
)

# テキストからの音楽生成
descriptions = [
    "upbeat electronic dance music with heavy bass and synth leads, 128 BPM",
    "gentle acoustic guitar fingerpicking with soft piano, relaxing ambient",
    "epic orchestral soundtrack with dramatic strings and brass, cinematic",
]

wav = model.generate(descriptions)
# wav shape: (3, 1, sample_rate * duration)

# 保存
for i, one_wav in enumerate(wav):
    audio_write(
        f"output_{i}",
        one_wav.cpu(),
        model.sample_rate,
        strategy="loudness",
        loudness_compressor=True,
    )

# メロディ条件付き生成
import torchaudio

melody_waveform, sr = torchaudio.load("melody_reference.wav")
wav = model.generate_with_chroma(
    descriptions=["jazz piano improvisation over the given melody"],
    melody_wavs=melody_waveform,
    melody_sample_rate=sr,
)
```

### 2.2 MusicGen の高度な使い方

```python
import torch
from audiocraft.models import MusicGen, MultiBandDiffusion
from audiocraft.data.audio import audio_write

class MusicGenAdvanced:
    """MusicGenの高度な活用パターン"""

    def __init__(self, model_size="large"):
        """
        model_size options:
        - "small": 300M params, 低VRAM、高速
        - "medium": 1.5B params, バランス
        - "large": 3.3B params, 最高品質
        - "melody": メロディ条件付き対応
        - "stereo-*": ステレオ出力対応
        """
        self.model = MusicGen.get_pretrained(f"facebook/musicgen-{model_size}")
        self.mbd = None  # MultiBandDiffusion（高品質デコーダ）

    def generate_with_continuation(
        self,
        prompt: str,
        audio_prefix: str,
        total_duration: float = 60.0,
        overlap: float = 5.0,
    ) -> torch.Tensor:
        """
        音声の続きを生成（長尺対応）
        MusicGenの30秒制限を超えて長い楽曲を生成
        """
        import torchaudio

        prefix_wav, sr = torchaudio.load(audio_prefix)
        if sr != self.model.sample_rate:
            prefix_wav = torchaudio.functional.resample(
                prefix_wav, sr, self.model.sample_rate
            )

        segments = [prefix_wav]
        generated_duration = prefix_wav.shape[-1] / self.model.sample_rate

        while generated_duration < total_duration:
            # 直前のオーバーラップ部分を入力として続きを生成
            overlap_samples = int(overlap * self.model.sample_rate)
            context = segments[-1][:, -overlap_samples:]

            self.model.set_generation_params(
                duration=min(30, total_duration - generated_duration + overlap),
                cfg_coef=3.0,
            )

            continuation = self.model.generate_continuation(
                prompt=context.unsqueeze(0),
                prompt_sample_rate=self.model.sample_rate,
                descriptions=[prompt],
            )

            # オーバーラップ部分をクロスフェードで結合
            new_segment = continuation[0][:, overlap_samples:]
            segments.append(new_segment)
            generated_duration += new_segment.shape[-1] / self.model.sample_rate

        return torch.cat(segments, dim=-1)

    def generate_variations(
        self,
        prompt: str,
        n_variations: int = 5,
        temperature_range: tuple = (0.7, 1.3),
    ) -> list:
        """
        同じプロンプトから異なるバリエーションを生成
        温度パラメータを変えて多様性を制御
        """
        variations = []
        temps = torch.linspace(
            temperature_range[0], temperature_range[1], n_variations
        )

        for temp in temps:
            self.model.set_generation_params(
                duration=15,
                temperature=float(temp),
                top_k=250,
                cfg_coef=3.0,
            )
            wav = self.model.generate([prompt])
            variations.append({
                "audio": wav[0],
                "temperature": float(temp),
            })

        return variations

    def batch_generate_with_styles(
        self,
        base_theme: str,
        styles: list,
    ) -> dict:
        """
        同じテーマを異なるスタイルで一括生成
        例: "夏の海" を Jazz, Lo-fi, Rock 等で生成
        """
        prompts = [f"{style} music about {base_theme}" for style in styles]

        self.model.set_generation_params(duration=30, cfg_coef=3.0)
        wavs = self.model.generate(prompts)

        results = {}
        for i, style in enumerate(styles):
            audio_write(
                f"{base_theme}_{style}",
                wavs[i].cpu(),
                self.model.sample_rate,
                strategy="loudness",
            )
            results[style] = wavs[i]

        return results

# 使用例
gen = MusicGenAdvanced("large")

# 長尺生成（60秒）
long_track = gen.generate_with_continuation(
    prompt="ambient electronic with evolving textures",
    audio_prefix="seed_audio.wav",
    total_duration=60.0,
)

# 5バリエーション生成
variations = gen.generate_variations(
    prompt="upbeat J-Pop with catchy synth melody",
    n_variations=5,
)
```

### 2.3 Suno APIの利用

```python
import requests
import time

class SunoClient:
    """Suno AI 音楽生成クライアント（非公式API概念）"""

    BASE_URL = "https://api.suno.ai/v1"

    def __init__(self, api_key: str):
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def generate_song(
        self,
        prompt: str,
        lyrics: str = None,
        style: str = None,
        title: str = None,
        instrumental: bool = False,
    ) -> dict:
        """楽曲を生成"""
        payload = {
            "prompt": prompt,
            "make_instrumental": instrumental,
        }
        if lyrics:
            payload["lyrics"] = lyrics
        if style:
            payload["style"] = style
        if title:
            payload["title"] = title

        response = requests.post(
            f"{self.BASE_URL}/generate",
            json=payload,
            headers=self.headers,
        )
        return response.json()

    def wait_for_completion(self, task_id: str, timeout: int = 300) -> dict:
        """生成完了を待機"""
        start = time.time()
        while time.time() - start < timeout:
            status = self.check_status(task_id)
            if status["state"] == "completed":
                return status
            elif status["state"] == "failed":
                raise Exception(f"生成失敗: {status.get('error')}")
            time.sleep(5)
        raise TimeoutError("生成タイムアウト")

# 使用例
client = SunoClient("your-api-key")

# ボーカル入り楽曲
result = client.generate_song(
    prompt="明るいポップソング、夏の海辺をテーマに",
    lyrics="""
[Verse 1]
波の音が聞こえてくる
青い空の下で
風が優しく吹いている
夏が始まる

[Chorus]
走り出そう、海へ
輝く太陽の下で
今日は最高の一日
忘れられない夏
""",
    style="J-Pop, upbeat, summer vibes",
    title="夏の海辺",
)
```

### 2.4 プロンプトエンジニアリング

```python
# 音楽生成プロンプトの構成要素

music_prompt_template = {
    "ジャンル": "electronic, jazz, classical, rock, J-Pop, lo-fi, ambient",
    "ムード": "upbeat, melancholic, energetic, relaxing, dramatic, ethereal",
    "テンポ": "slow (60-80 BPM), medium (90-120 BPM), fast (130-160 BPM)",
    "楽器": "piano, guitar, synth, strings, drums, bass, brass, flute",
    "構成": "intro, verse, chorus, bridge, outro, build-up, drop",
    "品質修飾": "professional, studio quality, high fidelity, warm tone",
    "参考": "in the style of ..., reminiscent of ..., inspired by ...",
}

# 効果的なプロンプト例
effective_prompts = [
    # ジャンル + ムード + 楽器 + テンポ + 品質
    "Lo-fi hip hop beat with jazzy piano chords, mellow saxophone, "
    "vinyl crackle, and soft drum loops. Relaxing study music at 85 BPM. "
    "Warm and nostalgic tone.",

    # シーン描写型
    "Soundtrack for walking through a neon-lit Tokyo street at night. "
    "Synthwave with Japanese city pop influences. Electric guitar, "
    "retro synthesizers, and a groovy bassline. 110 BPM.",

    # 感情表現型
    "A bittersweet farewell song. Gentle acoustic guitar fingerpicking "
    "with a delicate female vocal melody. Gradually building with soft "
    "strings joining midway. Emotional and cinematic.",
]
```

### 2.5 プロンプト最適化の体系的手法

```python
class MusicPromptOptimizer:
    """音楽生成プロンプトの体系的最適化"""

    # プロンプト構成要素のテンプレート
    PROMPT_COMPONENTS = {
        "genre": {
            "weight": "高",
            "examples": [
                "electronic", "jazz", "classical", "rock", "hip hop",
                "ambient", "folk", "R&B", "metal", "country",
                "J-Pop", "K-Pop", "bossa nova", "reggae", "funk",
            ],
            "tip": "複数ジャンルの組み合わせで独自性を出す",
        },
        "mood": {
            "weight": "高",
            "examples": [
                "upbeat", "melancholic", "energetic", "peaceful",
                "dark", "ethereal", "nostalgic", "triumphant",
                "mysterious", "playful", "tense", "dreamy",
            ],
            "tip": "感情の変化（例: 'starting calm, building to triumphant'）が効果的",
        },
        "instruments": {
            "weight": "中",
            "examples": [
                "acoustic guitar", "electric piano", "synthesizer pads",
                "orchestral strings", "808 bass", "jazz drums",
                "flute", "saxophone", "choir", "harp",
            ],
            "tip": "具体的な楽器名を指定するほど精度が上がる",
        },
        "tempo": {
            "weight": "中",
            "examples": [
                "slow (60-80 BPM)", "moderate (90-110 BPM)",
                "upbeat (120-140 BPM)", "fast (150-170 BPM)",
            ],
            "tip": "BPM数値を明示すると安定する",
        },
        "production": {
            "weight": "低",
            "examples": [
                "studio quality", "lo-fi aesthetic", "vinyl warmth",
                "crisp modern production", "raw and organic",
                "heavily reverbed", "dry and intimate",
            ],
            "tip": "プロダクションスタイルで雰囲気を制御",
        },
    }

    def build_prompt(
        self,
        genre: str,
        mood: str,
        instruments: list = None,
        tempo: str = None,
        production: str = None,
        scene: str = None,
        negative: str = None,
    ) -> str:
        """構造化されたプロンプトを構築"""
        parts = [genre]

        if mood:
            parts.append(f"{mood} atmosphere")
        if instruments:
            parts.append(f"featuring {', '.join(instruments)}")
        if tempo:
            parts.append(f"at {tempo}")
        if production:
            parts.append(f"with {production} production")
        if scene:
            parts.append(f"evoking {scene}")

        prompt = ", ".join(parts) + "."

        if negative:
            prompt += f" Avoid: {negative}."

        return prompt

    def generate_variations(self, base_prompt: str, n: int = 5) -> list:
        """プロンプトのバリエーションを生成"""
        variations = []
        modifiers = [
            "with more emphasis on rhythm",
            "with a darker, moodier tone",
            "with brighter, more uplifting energy",
            "with minimal arrangement",
            "with rich, layered arrangement",
            "with retro vintage feel",
            "with modern futuristic production",
        ]
        import random
        for _ in range(n):
            modifier = random.choice(modifiers)
            variations.append(f"{base_prompt} {modifier}")
        return variations

    def evaluate_prompt_quality(self, prompt: str) -> dict:
        """プロンプトの品質を評価"""
        issues = []
        suggestions = []

        # 長さチェック
        word_count = len(prompt.split())
        if word_count < 5:
            issues.append("プロンプトが短すぎる（5語未満）")
            suggestions.append("ジャンル、ムード、楽器、テンポを追加")
        elif word_count > 50:
            issues.append("プロンプトが長すぎる可能性（50語超）")
            suggestions.append("最も重要な要素に絞る")

        # 必須要素チェック
        has_genre = any(g in prompt.lower() for g in ["rock", "jazz", "pop",
                        "electronic", "classical", "ambient", "hip hop"])
        has_mood = any(m in prompt.lower() for m in ["upbeat", "calm", "dark",
                       "energetic", "relaxing", "dramatic"])
        has_instrument = any(i in prompt.lower() for i in ["guitar", "piano",
                            "drums", "synth", "bass", "strings"])

        if not has_genre:
            suggestions.append("ジャンルを明示的に追加")
        if not has_mood:
            suggestions.append("ムード/雰囲気を追加")
        if not has_instrument:
            suggestions.append("主要楽器を指定")

        score = 10
        score -= len(issues) * 2
        score -= (3 - sum([has_genre, has_mood, has_instrument])) * 1.5

        return {
            "score": max(0, min(10, score)),
            "issues": issues,
            "suggestions": suggestions,
            "has_genre": has_genre,
            "has_mood": has_mood,
            "has_instrument": has_instrument,
            "word_count": word_count,
        }

# 使用例
optimizer = MusicPromptOptimizer()

prompt = optimizer.build_prompt(
    genre="Lo-fi hip hop",
    mood="nostalgic and cozy",
    instruments=["jazzy piano", "vinyl crackle", "soft drums"],
    tempo="85 BPM",
    production="warm lo-fi",
    scene="studying in a rainy cafe",
)
print(prompt)
# "Lo-fi hip hop, nostalgic and cozy atmosphere, featuring jazzy piano,
#  vinyl crackle, soft drums, at 85 BPM, with warm lo-fi production,
#  evoking studying in a rainy cafe."

quality = optimizer.evaluate_prompt_quality(prompt)
print(f"品質スコア: {quality['score']}/10")
```

### 2.6 歌詞構造のガイドライン

```python
# Suno / Udio 向け歌詞フォーマットガイド

class LyricsFormatter:
    """AI音楽生成向け歌詞フォーマッタ"""

    # 歌詞構造タグ
    STRUCTURE_TAGS = {
        "[Intro]": "楽器のみのイントロ",
        "[Verse]": "Aメロ（物語の展開部分）",
        "[Verse 1]": "Aメロ1番",
        "[Verse 2]": "Aメロ2番",
        "[Pre-Chorus]": "Bメロ（サビへの助走）",
        "[Chorus]": "サビ（最も印象的な部分）",
        "[Bridge]": "ブリッジ（転調・展開部分）",
        "[Outro]": "アウトロ",
        "[Instrumental]": "楽器のみの間奏",
        "[Hook]": "フック（キャッチーな繰り返し）",
        "[Breakdown]": "ブレイクダウン（音を落とす部分）",
        "[Drop]": "ドロップ（EDMの盛り上がり）",
        "[Rap]": "ラップパート",
        "[Spoken]": "語り/台詞パート",
    }

    @staticmethod
    def create_song_structure(style: str = "pop") -> str:
        """ジャンル別の推奨構造テンプレート"""
        structures = {
            "pop": "[Intro]\n[Verse 1]\n[Pre-Chorus]\n[Chorus]\n"
                   "[Verse 2]\n[Pre-Chorus]\n[Chorus]\n[Bridge]\n[Chorus]\n[Outro]",
            "rock": "[Intro]\n[Verse 1]\n[Chorus]\n[Verse 2]\n"
                    "[Chorus]\n[Instrumental]\n[Chorus]\n[Outro]",
            "edm": "[Intro]\n[Breakdown]\n[Drop]\n[Breakdown]\n"
                   "[Drop]\n[Outro]",
            "ballad": "[Intro]\n[Verse 1]\n[Verse 2]\n[Chorus]\n"
                      "[Verse 3]\n[Chorus]\n[Bridge]\n[Chorus]\n[Outro]",
            "rap": "[Intro]\n[Verse 1]\n[Hook]\n[Verse 2]\n"
                   "[Hook]\n[Bridge]\n[Verse 3]\n[Hook]\n[Outro]",
        }
        return structures.get(style, structures["pop"])

    @staticmethod
    def format_lyrics(raw_lyrics: str, style: str = "pop") -> str:
        """生テキストの歌詞を構造化フォーマットに変換"""
        lines = [l.strip() for l in raw_lyrics.strip().split("\n") if l.strip()]

        if len(lines) < 4:
            return f"[Verse]\n{raw_lyrics}"

        formatted = []
        chunk_size = 4  # 4行ずつをセクションに
        sections = ["Verse 1", "Chorus", "Verse 2", "Chorus", "Bridge", "Chorus"]

        for i, section in enumerate(sections):
            start = i * chunk_size
            end = start + chunk_size
            if start >= len(lines):
                break
            section_lines = lines[start:end]
            formatted.append(f"[{section}]")
            formatted.extend(section_lines)
            formatted.append("")

        return "\n".join(formatted)

# 使用例
formatter = LyricsFormatter()
structure = formatter.create_song_structure("pop")
print(structure)
```

---

## 3. 音楽生成ワークフロー

### 3.1 プロダクション向けワークフロー

```
AI音楽制作ワークフロー
==================================================

Phase 1: 構想・プロンプト設計
    │
    ├── ジャンル、ムード、テンポの決定
    ├── リファレンス楽曲の選定
    └── プロンプト作成（複数バリエーション）
    │
    ▼
Phase 2: 生成・選定
    │
    ├── 複数バリエーションを生成（5-10候補）
    ├── ベスト候補の選定
    └── 必要に応じてプロンプト調整・再生成
    │
    ▼
Phase 3: 後処理・編集
    │
    ├── ステム分離（ボーカル/伴奏）
    ├── EQ・コンプレッション調整
    ├── 不要部分のカット・構成変更
    └── マスタリング
    │
    ▼
Phase 4: 統合・仕上げ
    │
    ├── 他の音源との統合
    ├── 最終ミックス
    └── 各フォーマットでのエクスポート
==================================================
```

### 3.2 自動化されたワークフロー実装

```python
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class MusicGenerationConfig:
    """音楽生成の設定"""
    prompt: str
    duration: float = 30.0
    n_candidates: int = 5
    temperature: float = 1.0
    cfg_coef: float = 3.0
    output_dir: str = "./output"
    target_lufs: float = -14.0

class AutoMusicPipeline:
    """自動音楽制作パイプライン"""

    def __init__(self, model_name: str = "facebook/musicgen-large"):
        from audiocraft.models import MusicGen
        self.model = MusicGen.get_pretrained(model_name)

    def run(self, config: MusicGenerationConfig) -> dict:
        """完全自動パイプラインの実行"""
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: 複数候補を生成
        candidates = self._generate_candidates(config)

        # Phase 2: 品質スコアリング
        scored = self._score_candidates(candidates, config.prompt)

        # Phase 3: ベスト候補の後処理
        best = scored[0]
        processed = self._post_process(best["audio"], config)

        # Phase 4: エクスポート
        output_path = output_dir / "final_output.wav"
        self._export(processed, output_path)

        return {
            "output_path": str(output_path),
            "n_candidates": len(candidates),
            "best_score": best["score"],
            "all_scores": [s["score"] for s in scored],
        }

    def _generate_candidates(self, config):
        """複数候補の生成"""
        self.model.set_generation_params(
            duration=config.duration,
            temperature=config.temperature,
            cfg_coef=config.cfg_coef,
        )

        candidates = []
        for i in range(config.n_candidates):
            wav = self.model.generate([config.prompt])
            candidates.append(wav[0])

        return candidates

    def _score_candidates(self, candidates, prompt):
        """品質スコアリング"""
        import numpy as np

        scored = []
        for i, audio in enumerate(candidates):
            # 簡易スコアリング: RMS、ダイナミックレンジ、静寂比率
            samples = audio.cpu().numpy().flatten()
            rms = np.sqrt(np.mean(samples ** 2))
            dynamic_range = np.max(np.abs(samples)) / (rms + 1e-10)
            silence_ratio = np.mean(np.abs(samples) < 0.01)

            score = (
                0.4 * min(rms * 10, 1.0) +          # 適度な音量
                0.3 * min(dynamic_range / 10, 1.0) +  # ダイナミックレンジ
                0.3 * (1.0 - silence_ratio)           # 無音が少ない
            )

            scored.append({"audio": audio, "score": score, "index": i})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    def _post_process(self, audio, config):
        """後処理（正規化、EQ等）"""
        import numpy as np

        samples = audio.cpu().numpy()

        # ピーク正規化
        peak = np.max(np.abs(samples))
        if peak > 0:
            samples = samples * (0.95 / peak)

        # DCオフセット除去
        samples = samples - np.mean(samples)

        return samples

    def _export(self, audio, output_path):
        """ファイル出力"""
        import soundfile as sf
        import numpy as np

        if audio.ndim > 1:
            audio = audio.squeeze()
        sf.write(str(output_path), audio, 32000)

# 使用例
pipeline = AutoMusicPipeline()
result = pipeline.run(MusicGenerationConfig(
    prompt="Cinematic orchestral music with emotional strings, "
           "building from soft piano to full orchestra, 90 BPM",
    duration=30.0,
    n_candidates=5,
))
print(f"出力: {result['output_path']}")
print(f"ベストスコア: {result['best_score']:.3f}")
```

---

## 4. ユースケース別実装

### 4.1 動画BGM自動生成

```python
class VideoBackgroundMusicGenerator:
    """動画コンテンツ向けBGM自動生成"""

    # シーンタイプ別のプロンプトテンプレート
    SCENE_PROMPTS = {
        "intro": "corporate intro music, professional, confident, "
                 "modern electronic with clean synths, 110 BPM, 10 seconds",
        "presentation": "soft background music for presentation, "
                        "minimal piano and ambient pads, non-intrusive, "
                        "professional corporate, 90 BPM",
        "action": "high energy action music, driving drums, "
                  "electric guitar riffs, intense and exciting, 140 BPM",
        "emotional": "emotional cinematic music, gentle piano with "
                     "warm strings, touching and heartfelt, 70 BPM",
        "celebration": "upbeat celebration music, happy and bright, "
                       "acoustic guitar with claps and tambourine, 120 BPM",
        "tutorial": "light and friendly tutorial background music, "
                    "ukulele with soft percussion, positive and engaging, "
                    "100 BPM",
        "outro": "gentle outro music, fading out, calm and conclusive, "
                 "soft piano with ambient textures, 80 BPM",
    }

    def generate_for_scenes(self, scenes: list) -> dict:
        """シーンリストに基づいてBGMを一括生成"""
        results = {}
        for scene in scenes:
            prompt = self.SCENE_PROMPTS.get(
                scene["type"],
                self.SCENE_PROMPTS["presentation"]
            )
            # duration指定を追加
            if "duration" in scene:
                prompt += f", {scene['duration']} seconds"

            results[scene["name"]] = {
                "prompt": prompt,
                "type": scene["type"],
            }
        return results

    def generate_loopable(self, prompt: str, duration: float = 30.0) -> str:
        """ループ可能なBGMを生成"""
        loop_prompt = prompt + ". Seamless loop, consistent energy level."
        # 実際にはここでモデルを呼び出す
        return loop_prompt

# 使用例
bgm_gen = VideoBackgroundMusicGenerator()
scenes = [
    {"name": "intro", "type": "intro", "duration": 10},
    {"name": "main_content", "type": "tutorial", "duration": 120},
    {"name": "ending", "type": "outro", "duration": 15},
]
bgm_plan = bgm_gen.generate_for_scenes(scenes)
```

### 4.2 ゲーム音楽の動的生成

```python
class GameMusicEngine:
    """ゲーム向け動的音楽生成エンジン"""

    # ゲーム状態に対応する音楽パラメータ
    GAME_STATES = {
        "exploration": {
            "prompt": "peaceful exploration music, fantasy RPG, "
                      "gentle flute and harp, ambient nature sounds",
            "energy": 0.3,
            "tempo": "70 BPM",
        },
        "combat": {
            "prompt": "intense battle music, epic orchestral, "
                      "pounding drums, aggressive brass, urgent strings",
            "energy": 0.9,
            "tempo": "150 BPM",
        },
        "boss_fight": {
            "prompt": "epic boss battle music, heavy metal meets orchestra, "
                      "choir chanting, double bass drums, distorted guitars",
            "energy": 1.0,
            "tempo": "170 BPM",
        },
        "town": {
            "prompt": "medieval town music, cheerful and bustling, "
                      "acoustic guitar, fiddle, accordion, tavern atmosphere",
            "energy": 0.5,
            "tempo": "110 BPM",
        },
        "dungeon": {
            "prompt": "dark dungeon ambient music, eerie and mysterious, "
                      "deep drones, distant echoes, subtle percussion",
            "energy": 0.4,
            "tempo": "60 BPM",
        },
        "victory": {
            "prompt": "triumphant victory fanfare, heroic brass, "
                      "celebratory orchestral, rising melody",
            "energy": 0.8,
            "tempo": "120 BPM",
        },
    }

    def get_music_for_state(self, game_state: str,
                             intensity: float = 1.0) -> dict:
        """ゲーム状態に応じた音楽パラメータを取得"""
        state_config = self.GAME_STATES.get(
            game_state, self.GAME_STATES["exploration"]
        )

        # 強度に応じてプロンプトを調整
        prompt = state_config["prompt"]
        if intensity > 0.7:
            prompt += " More intense and dramatic."
        elif intensity < 0.3:
            prompt += " More subdued and calm."

        return {
            "prompt": prompt,
            "energy": state_config["energy"] * intensity,
            "tempo": state_config["tempo"],
        }

    def create_transition(self, from_state: str, to_state: str,
                           duration_seconds: float = 5.0) -> str:
        """状態遷移時のトランジション音楽プロンプト"""
        from_config = self.GAME_STATES.get(from_state, {})
        to_config = self.GAME_STATES.get(to_state, {})

        return (
            f"Musical transition from {from_config.get('prompt', '')} "
            f"to {to_config.get('prompt', '')}, "
            f"smooth crossfade, {duration_seconds} seconds"
        )
```

---

## 5. 比較表

### 5.1 主要音楽生成サービス比較

| 項目 | Suno | Udio | MusicGen | Stable Audio |
|------|------|------|----------|-------------|
| 種別 | SaaS | SaaS | OSS | SaaS/OSS |
| ボーカル生成 | 対応 | 対応 | 非対応 | 非対応 |
| 歌詞入力 | 対応 | 対応 | 非対応 | 非対応 |
| 最大長 | ~4分 | ~2分 | 30秒(標準) | 190秒 |
| 品質 | 高い | 高い | 中〜高 | 中〜高 |
| カスタマイズ | プロンプト | プロンプト | コード制御 | プロンプト |
| 商用利用 | 有料プラン | 有料プラン | MIT License | 条件付き |
| ローカル実行 | 不可 | 不可 | 可能 | Open版可能 |
| GPU要件 | 不要 | 不要 | 16GB+ VRAM | 8GB+ VRAM |
| API | あり | あり | Python | あり |

### 5.2 用途別推奨サービス

| ユースケース | 推奨 | 理由 |
|-------------|------|------|
| ボーカル入り楽曲 | Suno / Udio | 歌声生成に対応 |
| BGM/インスト | MusicGen / Stable Audio | 楽器音の品質が高い |
| プロトタイプ | Suno | 簡単操作、高品質 |
| 研究開発 | MusicGen | OSS、カスタマイズ可能 |
| ゲーム音楽 | Stable Audio | ループ音楽に対応 |
| 動画BGM | Suno / Stable Audio | 長さ・ムード制御が容易 |
| 商用利用（低コスト）| MusicGen | MIT License、無料 |
| ブランドサウンド | MusicGen + ファインチューニング | 独自スタイルの学習 |

### 5.3 モデルサイズ別性能比較（MusicGen）

| モデル | パラメータ | VRAM | 生成速度(30秒) | 品質 | 推奨用途 |
|--------|-----------|------|---------------|------|---------|
| small | 300M | ~4GB | ~5秒 | 中 | プロトタイプ、テスト |
| medium | 1.5B | ~8GB | ~15秒 | 中〜高 | バランス重視 |
| large | 3.3B | ~16GB | ~30秒 | 高 | 本番品質 |
| melody | 1.5B | ~8GB | ~15秒 | 高(メロディ) | メロディ条件付き |
| stereo-small | 300M | ~4GB | ~8秒 | 中 | ステレオ出力 |
| stereo-large | 3.3B | ~16GB | ~40秒 | 高 | 高品質ステレオ |

---

## 6. トラブルシューティング

### 6.1 よくある問題と解決策

```
問題: 生成された音楽にノイズやアーティファクトが含まれる
==================================================
原因:
- 温度パラメータが高すぎる
- CFG係数が不適切
- モデルサイズが小さい

解決策:
1. temperature を 0.8-1.0 の範囲に設定
2. cfg_coef を 3.0-5.0 の範囲で調整
3. large モデルを使用
4. MultiBandDiffusion (MBD) デコーダで品質向上:
   mbd = MultiBandDiffusion.get_mbd_musicgen()
   wav = mbd.tokens_to_wav(tokens)
==================================================

問題: プロンプトに一致しない音楽が生成される
==================================================
原因:
- プロンプトが曖昧
- CFG係数が低すぎる
- モデルの理解範囲外の指示

解決策:
1. プロンプトを具体化（ジャンル+楽器+テンポ+ムード）
2. cfg_coef を 4.0-6.0 に上げる
3. 英語プロンプトを使用（学習データの大部分が英語）
4. 段階的に条件を追加して生成結果を確認
==================================================

問題: MusicGenでGPUメモリ不足 (OOM) が発生する
==================================================
原因:
- モデルサイズが大きい
- 生成時間が長い
- バッチサイズが大きい

解決策:
1. 小さいモデルに切り替え: musicgen-small (300M)
2. duration を短く（15秒以下）
3. バッチサイズを1に
4. float16で実行: model.to(torch.float16)
5. CPU処理（遅いが安定）: model.to("cpu")
==================================================
```

---

## 7. アンチパターン

### 7.1 アンチパターン: 曖昧なプロンプト

```python
# BAD: 曖昧すぎるプロンプト
bad_prompts = [
    "いい感じの曲を作って",          # 何が「いい感じ」か不明
    "音楽",                         # 情報量ゼロ
    "かっこいいロック",              # 具体性不足
]

# GOOD: 具体的で構造化されたプロンプト
good_prompts = [
    # ジャンル + テンポ + 楽器 + ムード + 品質
    "Energetic J-Rock with distorted electric guitars, driving drums at 150 BPM, "
    "powerful bass riffs, and anthemic chorus melodies. "
    "Stadium rock energy with modern production quality.",

    # シーン + 詳細 + 技術指定
    "Ambient electronic music for a sci-fi movie scene. "
    "Deep sub-bass drones, ethereal pad synths, glitchy percussion elements, "
    "and distant vocal textures. Dark and mysterious atmosphere. "
    "Slow tempo around 70 BPM with evolving textures.",
]
```

### 7.2 アンチパターン: 生成物の無加工使用

```python
# BAD: AI生成音楽をそのまま使用
def bad_workflow(prompt):
    audio = music_gen.generate(prompt)
    publish(audio)  # 品質のバラつき、著作権リスク

# GOOD: 後処理パイプラインを通す
def good_workflow(prompt, n_candidates=5):
    # 1. 複数候補生成
    candidates = [music_gen.generate(prompt) for _ in range(n_candidates)]

    # 2. 品質スコアリング（自動 + 手動）
    scored = []
    for i, audio in enumerate(candidates):
        score = auto_quality_score(audio)  # CLAP Score等
        scored.append((score, i, audio))
    scored.sort(reverse=True)

    # 3. ベスト候補を選択
    best_audio = scored[0][2]

    # 4. 後処理
    processed = apply_effects(best_audio, [
        ("normalize", {"target_db": -14}),
        ("eq", {"low_cut": 30, "high_cut": 18000}),
        ("compress", {"threshold": -20, "ratio": 3}),
    ])

    # 5. 最終確認
    return processed
```

### 7.3 アンチパターン: 著作権の確認を怠る

```python
# BAD: 著作権状況を確認せずに商用利用
def bad_commercial_use(service, prompt):
    audio = service.generate(prompt)
    sell(audio)  # ライセンス確認なし

# GOOD: サービスごとのライセンスを確認
def good_commercial_use(service, prompt):
    # サービス別ライセンスチェック
    license_check = {
        "suno_free": {"commercial": False, "credit_required": True},
        "suno_pro": {"commercial": True, "credit_required": False},
        "musicgen": {"commercial": True, "license": "MIT"},
        "stable_audio_free": {"commercial": False},
        "stable_audio_pro": {"commercial": True, "terms": "check website"},
    }

    plan = license_check.get(service.plan)
    if not plan or not plan.get("commercial"):
        raise ValueError(
            f"商用利用不可: {service.plan}。有料プランにアップグレードしてください。"
        )

    audio = service.generate(prompt)

    # メタデータにAI生成であることを記録
    metadata = {
        "generator": service.name,
        "prompt": prompt,
        "license": plan.get("license", "proprietary"),
        "ai_generated": True,
        "generation_date": datetime.now().isoformat(),
    }

    return audio, metadata
```

---

## 8. ベストプラクティス

### 8.1 音楽生成のベストプラクティス

```
プロンプト設計:
==================================================
1. 英語で記述する（学習データの大部分が英語）
2. ジャンル→ムード→楽器→テンポの順で記述
3. 具体的なBPM値を指定する
4. ネガティブプロンプト（避けたい要素）も活用
5. 参照楽曲のスタイルを具体的に描写

品質管理:
==================================================
1. 必ず複数候補（5-10）を生成して選定
2. 自動品質スコアリング + 人間の聴感評価を組み合わせる
3. CLAP Scoreでプロンプト一致度を定量評価
4. ラウドネス正規化（-14 LUFS）を最終出力に適用
5. 最低限のEQ処理（ローカット80Hz、ハイカット18kHz）

ワークフロー:
==================================================
1. プロンプト反復（粗い指定→微調整→最終版）
2. 生成→ステム分離→個別編集→再合成のフロー
3. AIのガイド提案結果を人間が最終判断
4. 生成ログ（プロンプト、パラメータ、スコア）を記録
5. 成功パターンのプロンプトをテンプレート化
```

---

## 9. FAQ

### Q1: AI生成音楽の著作権はどうなりますか？

法律はまだ発展途上ですが、2025-2026年時点の一般的な見解は次のとおりです。(1) AI生成物自体の著作権: 多くの法域で、AIが自律的に生成したものには著作権が発生しないとされています。(2) プロンプト作成者の権利: 十分な創作的寄与があれば、一部権利が認められる可能性があります。(3) 各サービスの利用規約: Sunoの有料プランでは商用利用が許可され、生成物の使用権がユーザーに付与されます。MusicGenはMITライセンスですが、学習データに関する議論は継続中です。商用利用時は必ず利用規約を確認してください。

### Q2: MusicGenのファインチューニングは可能ですか？

可能です。Meta公式のaudiocraftリポジトリにファインチューニングのためのスクリプトが含まれています。手順は、(1) 学習データの準備（音声ファイル + テキスト説明のペア）、(2) audiocraft_trainer の設定、(3) 既存のチェックポイントからの学習再開。ただし、large モデルのファインチューニングには32GB以上のVRAMが必要です。小規模データでの学習はsmallモデルから始めることを推奨します。

### Q3: 生成音楽の品質を自動評価する方法はありますか？

主な評価指標として、(1) FAD（Frechet Audio Distance）: 生成音楽と参照音楽の分布間距離、(2) CLAP Score: テキストと音声の意味的一致度、(3) KL Divergence: ジャンル/楽器分類の分布差。ただし、音楽の品質は主観的な要素が大きく、自動指標だけでは不十分です。実用的には、自動スコアでの粗い選別 + 人間による聴感評価の組み合わせが最も効果的です。

### Q4: 生成音楽を商用コンテンツ（YouTube動画、広告等）で使用する際の注意点は？

(1) 使用するサービスの商用利用ライセンスを確認する（Suno Proプラン等）。(2) Content IDシステムでの誤検出に備え、生成証明（プロンプト、生成日時、サービス名）を保持する。(3) 他の楽曲と酷似した出力がないか聴感確認する。(4) AI生成であることの開示義務がある場合は遵守する。(5) 定期的にサービスの利用規約の更新を確認する。

### Q5: 長い楽曲（3分以上）を生成するにはどうすればよいですか？

MusicGenの標準は30秒ですが、以下の方法で長尺化が可能です。(1) Continuation機能: 生成した音声の末尾数秒を入力として続きを生成する連鎖方式。(2) Suno/Udio: 最大4分の楽曲を一度に生成可能。(3) セクション結合: Intro→Verse→Chorusを個別に生成し、クロスフェードで結合。(4) ループ生成: 30秒のループ素材を生成し、DAWで構成を組み立てる。品質面ではSuno/Udieが最も安定した長尺出力を実現しています。

### Q6: AI音楽生成の計算コストを最適化するには？

(1) 小さいモデルから始める: musicgen-smallで十分な品質が得られる場合も多い。(2) バッチ生成: 複数プロンプトを一度に処理してGPU効率を上げる。(3) キャッシュ: 同じプロンプトの結果をキャッシュし、再生成を避ける。(4) INT8量子化: モデルを量子化してVRAM使用量を削減。(5) スポットインスタンス: クラウドGPU（AWS, GCP）のスポットインスタンスでコスト削減。1曲あたりの生成コストは、GPU使用（$0.5-2/時）× 生成時間で算出できます。

---

## まとめ

| 項目 | 要点 |
|------|------|
| 技術基盤 | コーデック言語モデル型と拡散モデル型の2大アプローチ |
| Suno/Udio | ボーカル+歌詞対応。商用プランあり |
| MusicGen | Meta OSS。コード制御可能、ファインチューニング対応 |
| プロンプト | ジャンル+ムード+テンポ+楽器+品質修飾を具体的に |
| ワークフロー | 複数生成→選定→後処理→仕上げの4段階 |
| 著作権 | サービス利用規約を確認。法整備は進行中 |
| 品質管理 | CLAP Score + 聴感評価の組み合わせ |
| 長尺化 | Continuation機能 or Suno/Udio の直接生成 |

## 次に読むべきガイド

- [01-stem-separation.md](./01-stem-separation.md) — ステム分離（Demucs、LALAL.AI）
- [02-audio-effects.md](./02-audio-effects.md) — AI音声エフェクト
- [03-midi-ai.md](./03-midi-ai.md) — MIDI×AI（自動作曲、コード進行生成）

## 参考文献

1. Copet, J., et al. (2023). "Simple and Controllable Music Generation" — MusicGen論文。テキスト条件付き音楽生成のベースライン
2. Agostinelli, A., et al. (2023). "MusicLM: Generating Music From Text" — Google MusicLM。テキストからの高品質音楽生成
3. Evans, Z., et al. (2024). "Stable Audio: Fast Timing-Conditioned Latent Audio Diffusion" — Stable Audio。潜在拡散モデルベースの音声生成
4. Défossez, A., et al. (2023). "High Fidelity Neural Audio Compression" — Encodec論文。音楽生成の基盤となるニューラル音声コーデック
5. Wu, Y., et al. (2023). "Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation" — CLAP論文。テキスト-音声の対照学習
