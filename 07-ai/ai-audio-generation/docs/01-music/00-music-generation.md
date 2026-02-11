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

### 2.2 Suno APIの利用

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

### 2.3 プロンプトエンジニアリング

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

---

## 4. 比較表

### 4.1 主要音楽生成サービス比較

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

### 4.2 用途別推奨サービス

| ユースケース | 推奨 | 理由 |
|-------------|------|------|
| ボーカル入り楽曲 | Suno / Udio | 歌声生成に対応 |
| BGM/インスト | MusicGen / Stable Audio | 楽器音の品質が高い |
| プロトタイプ | Suno | 簡単操作、高品質 |
| 研究開発 | MusicGen | OSS、カスタマイズ可能 |
| ゲーム音楽 | Stable Audio | ループ音楽に対応 |
| 動画BGM | Suno / Stable Audio | 長さ・ムード制御が容易 |
| 商用利用（低コスト）| MusicGen | MIT License、無料 |

---

## 5. アンチパターン

### 5.1 アンチパターン: 曖昧なプロンプト

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

### 5.2 アンチパターン: 生成物の無加工使用

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

---

## 6. FAQ

### Q1: AI生成音楽の著作権はどうなりますか？

法律はまだ発展途上ですが、2025-2026年時点の一般的な見解は次のとおりです。(1) AI生成物自体の著作権: 多くの法域で、AIが自律的に生成したものには著作権が発生しないとされています。(2) プロンプト作成者の権利: 十分な創作的寄与があれば、一部権利が認められる可能性があります。(3) 各サービスの利用規約: Sunoの有料プランでは商用利用が許可され、生成物の使用権がユーザーに付与されます。MusicGenはMITライセンスですが、学習データに関する議論は継続中です。商用利用時は必ず利用規約を確認してください。

### Q2: MusicGenのファインチューニングは可能ですか？

可能です。Meta公式のaudiocraftリポジトリにファインチューニングのためのスクリプトが含まれています。手順は、(1) 学習データの準備（音声ファイル + テキスト説明のペア）、(2) audiocraft_trainer の設定、(3) 既存のチェックポイントからの学習再開。ただし、large モデルのファインチューニングには32GB以上のVRAMが必要です。小規模データでの学習はsmallモデルから始めることを推奨します。

### Q3: 生成音楽の品質を自動評価する方法はありますか？

主な評価指標として、(1) FAD（Frechet Audio Distance）: 生成音楽と参照音楽の分布間距離、(2) CLAP Score: テキストと音声の意味的一致度、(3) KL Divergence: ジャンル/楽器分類の分布差。ただし、音楽の品質は主観的な要素が大きく、自動指標だけでは不十分です。実用的には、自動スコアでの粗い選別 + 人間による聴感評価の組み合わせが最も効果的です。

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

## 次に読むべきガイド

- [01-stem-separation.md](./01-stem-separation.md) — ステム分離（Demucs、LALAL.AI）
- [02-audio-effects.md](./02-audio-effects.md) — AI音声エフェクト
- [03-midi-ai.md](./03-midi-ai.md) — MIDI×AI（自動作曲、コード進行生成）

## 参考文献

1. Copet, J., et al. (2023). "Simple and Controllable Music Generation" — MusicGen論文。テキスト条件付き音楽生成のベースライン
2. Agostinelli, A., et al. (2023). "MusicLM: Generating Music From Text" — Google MusicLM。テキストからの高品質音楽生成
3. Evans, Z., et al. (2024). "Stable Audio: Fast Timing-Conditioned Latent Audio Diffusion" — Stable Audio。潜在拡散モデルベースの音声生成
4. Défossez, A., et al. (2023). "High Fidelity Neural Audio Compression" — Encodec論文。音楽生成の基盤となるニューラル音声コーデック
