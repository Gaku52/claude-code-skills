# MIDI×AI — 自動作曲、アレンジ、コード進行生成

> AIを活用したMIDI音楽制作（自動作曲、コード進行生成、アレンジ）の技術と実践を解説する

## この章で学ぶこと

1. MIDIデータの基礎知識とAI処理のためのデータ表現
2. AI自動作曲・コード進行生成の主要手法とモデル
3. DAW連携とプロダクションワークフローへの統合

---

## 1. MIDIの基礎

### 1.1 MIDIデータ構造

```
MIDIメッセージの構造
==================================================

MIDIイベントの基本単位:
┌──────────┬──────────┬──────────┬──────────┐
│  Delta   │ Status   │  Data 1  │  Data 2  │
│  Time    │  Byte    │  (Note)  │(Velocity)│
│ (時間差) │(メッセージ種別)│(0-127)│ (0-127) │
└──────────┴──────────┴──────────┴──────────┘

Note On:  0x90 | channel, note_number, velocity
Note Off: 0x80 | channel, note_number, velocity

ピアノロール表現:
  MIDI Note Number
  ↑
  72│        ■■■■
  71│
  69│  ■■■■■■■■          ■■■■
  67│          ■■■■■■
  65│
  64│■■■■■■
  60│                ■■■■■■■■■■
    └──────────────────────────→ Time (ticks)

ノート番号と音名の対応:
  C4(ド)=60, D4(レ)=62, E4(ミ)=64, F4(ファ)=65
  G4(ソ)=67, A4(ラ)=69, B4(シ)=71, C5(ド)=72
==================================================
```

### 1.2 MIDIデータのプログラミング

```python
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

def create_chord_progression():
    """コード進行をMIDIで生成"""
    mid = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    mid.tracks.append(track)

    # テンポ設定（BPM 120）
    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(120)))

    # C-Am-F-G のコード進行
    chords = [
        {"name": "C",  "notes": [60, 64, 67], "duration": 480 * 2},  # 2拍
        {"name": "Am", "notes": [57, 60, 64], "duration": 480 * 2},
        {"name": "F",  "notes": [53, 57, 60], "duration": 480 * 2},
        {"name": "G",  "notes": [55, 59, 62], "duration": 480 * 2},
    ]

    for chord in chords:
        # Note On（全音同時）
        for i, note in enumerate(chord["notes"]):
            track.append(Message('note_on', note=note, velocity=80, time=0))

        # Note Off（duration 後）
        for i, note in enumerate(chord["notes"]):
            time = chord["duration"] if i == 0 else 0
            track.append(Message('note_off', note=note, velocity=0, time=time))

    mid.save('chord_progression.mid')
    return mid

# MIDI → トークン列（AI入力用）
def midi_to_tokens(midi_file: str) -> list:
    """MIDIをAI処理用のトークン列に変換"""
    mid = MidiFile(midi_file)
    tokens = []

    for msg in mid.tracks[0]:
        if msg.type == 'note_on' and msg.velocity > 0:
            tokens.append(f"NOTE_ON_{msg.note}")
            tokens.append(f"VELOCITY_{msg.velocity // 8}")  # 0-15に量子化
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            tokens.append(f"NOTE_OFF_{msg.note}")
        if msg.time > 0:
            # 時間を量子化
            time_token = min(msg.time // 120, 15)  # 0-15に量子化
            tokens.append(f"TIME_SHIFT_{time_token}")

    return tokens
```

---

## 2. AI自動作曲

### 2.1 MIDIトークナイザ

```python
# MidiTok: MIDI専用トークナイザライブラリ

from miditok import REMI, TokenizerConfig
from pathlib import Path

# トークナイザの設定
config = TokenizerConfig(
    num_velocities=32,      # ベロシティの量子化数
    use_chords=True,        # コード検出を有効化
    use_tempos=True,        # テンポ変化を含める
    use_time_signatures=True,
    nb_tempos=32,           # テンポの量子化数
    tempo_range=(40, 250),
)

# REMI トークナイザの作成
tokenizer = REMI(config)

# MIDIファイルをトークン化
tokens = tokenizer("song.mid")
print(f"トークン数: {len(tokens.ids)}")
print(f"最初の10トークン: {tokens.tokens[:10]}")
# 例: ['Bar_None', 'Position_0', 'Chord_C:maj', 'Pitch_60', 'Velocity_80', ...]

# トークンからMIDI復元
reconstructed_midi = tokenizer.decode(tokens)
reconstructed_midi.dump_midi("reconstructed.mid")
```

### 2.2 Transformer による作曲

```python
# Transformerベースの自動作曲モデル（概念実装）

import torch
import torch.nn as nn

class MusicTransformer(nn.Module):
    """MIDI自動作曲用 Transformer"""

    def __init__(
        self,
        vocab_size: int = 512,    # トークン語彙サイズ
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, n_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)

        embeddings = self.embedding(x) + self.pos_encoding(positions)

        # Causal mask（未来のトークンを見ない）
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(x.device)

        memory = torch.zeros_like(embeddings)  # Decoder-only
        output = self.transformer(embeddings, memory, tgt_mask=mask)
        logits = self.output_proj(output)
        return logits

    def generate(self, seed_tokens, max_length=512, temperature=0.8, top_k=40):
        """自動作曲（トークン生成）"""
        self.eval()
        generated = seed_tokens.clone()

        with torch.no_grad():
            for _ in range(max_length):
                logits = self.forward(generated)
                next_logits = logits[:, -1, :] / temperature

                # Top-K フィルタリング
                top_k_logits, top_k_indices = torch.topk(next_logits, top_k)
                probs = torch.softmax(top_k_logits, dim=-1)
                idx = torch.multinomial(probs, 1)
                next_token = top_k_indices.gather(-1, idx)

                generated = torch.cat([generated, next_token], dim=1)

        return generated
```

### 2.3 コード進行生成

```python
import random

class ChordProgressionGenerator:
    """音楽理論ベース + AI のコード進行生成"""

    # ダイアトニックコード（Cメジャー）
    DIATONIC_CHORDS = {
        "I":   {"root": "C",  "type": "maj", "notes": [0, 4, 7]},
        "ii":  {"root": "Dm", "type": "min", "notes": [2, 5, 9]},
        "iii": {"root": "Em", "type": "min", "notes": [4, 7, 11]},
        "IV":  {"root": "F",  "type": "maj", "notes": [5, 9, 0]},
        "V":   {"root": "G",  "type": "maj", "notes": [7, 11, 2]},
        "vi":  {"root": "Am", "type": "min", "notes": [9, 0, 4]},
        "vii": {"root": "Bdim","type": "dim", "notes": [11, 2, 5]},
    }

    # 一般的なコード進行パターン
    COMMON_PROGRESSIONS = {
        "ポップ定番":     ["I", "V", "vi", "IV"],
        "カノン進行":     ["I", "V", "vi", "iii", "IV", "I", "IV", "V"],
        "小室進行":       ["vi", "IV", "V", "I"],
        "王道バラード":   ["I", "vi", "IV", "V"],
        "ジャズ定番":     ["ii", "V", "I", "I"],
        "ブルース":       ["I", "I", "IV", "I", "V", "IV", "I", "V"],
    }

    def generate(self, style: str = "ポップ定番", key: str = "C",
                 bars: int = 8) -> list:
        """コード進行を生成"""
        base_progression = self.COMMON_PROGRESSIONS.get(style)
        if not base_progression:
            base_progression = random.choice(list(self.COMMON_PROGRESSIONS.values()))

        # barsに合わせて繰り返しまたはバリエーション生成
        progression = []
        while len(progression) < bars:
            progression.extend(base_progression)
        progression = progression[:bars]

        # キー変換
        transposed = self._transpose(progression, key)
        return transposed

    def _transpose(self, progression, target_key):
        """キー変換（簡略版）"""
        key_offsets = {
            "C": 0, "C#": 1, "D": 2, "Eb": 3, "E": 4, "F": 5,
            "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11,
        }
        offset = key_offsets.get(target_key, 0)
        # 実際にはノート番号にオフセットを加算
        return [(chord, offset) for chord in progression]

# 使用例
gen = ChordProgressionGenerator()
chords = gen.generate(style="小室進行", key="A", bars=8)
print(f"生成されたコード進行: {chords}")
```

---

## 3. DAW連携

### 3.1 DAW連携アーキテクチャ

```
AI × DAW ワークフロー
==================================================

┌────────────────────────────────────────┐
│              DAW (Ableton / Logic)      │
│                                        │
│  ┌──────┐  ┌──────┐  ┌──────┐        │
│  │Track1│  │Track2│  │Track3│  ...    │
│  │Piano │  │Bass  │  │Drums │        │
│  └──┬───┘  └──┬───┘  └──┬───┘        │
│     │         │         │             │
│     └────┬────┘────┬────┘             │
│          │         │                  │
│     ┌────▼─────┐   │                  │
│     │MIDI Out  │   │                  │
│     │(VST/AU)  │   │                  │
│     └────┬─────┘   │                  │
└──────────┼─────────┼──────────────────┘
           │         │
    ┌──────▼─────────▼──────┐
    │    AI 作曲エンジン     │
    │                       │
    │  Input: MIDI + Config  │
    │  ├─ コード進行提案      │
    │  ├─ メロディ生成        │
    │  ├─ ベースライン生成    │
    │  ├─ ドラムパターン生成  │
    │  └─ アレンジ提案       │
    │                       │
    │  Output: MIDI          │
    └───────────────────────┘
==================================================
```

### 3.2 ドラムパターン生成

```python
import numpy as np

class DrumPatternGenerator:
    """AIドラムパターン生成器"""

    # General MIDI ドラムマップ（抜粋）
    GM_DRUMS = {
        "kick":     36,
        "snare":    38,
        "hihat_c":  42,  # Closed Hi-Hat
        "hihat_o":  46,  # Open Hi-Hat
        "crash":    49,
        "ride":     51,
        "tom_high": 48,
        "tom_mid":  45,
        "tom_low":  41,
    }

    # 基本パターンテンプレート（16分音符グリッド）
    PATTERNS = {
        "basic_rock": {
            "kick":    [1,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0],
            "snare":   [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
            "hihat_c": [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
        },
        "four_on_floor": {
            "kick":    [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0],
            "snare":   [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
            "hihat_c": [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
            "hihat_o": [0,0,0,0, 0,0,0,1, 0,0,0,0, 0,0,0,1],
        },
    }

    def generate(self, style="basic_rock", bars=4, humanize=True):
        """ドラムパターンを生成"""
        base = self.PATTERNS.get(style, self.PATTERNS["basic_rock"])

        pattern = {}
        for instrument, beat in base.items():
            full_pattern = beat * bars
            if humanize:
                full_pattern = self._humanize(full_pattern)
            pattern[instrument] = full_pattern

        return pattern

    def _humanize(self, pattern, timing_var=10, velocity_var=15):
        """人間らしさを付与（タイミング/ベロシティのゆらぎ）"""
        humanized = []
        for hit in pattern:
            if hit:
                velocity = max(40, min(127, 80 + np.random.randint(-velocity_var, velocity_var)))
                timing_offset = np.random.randint(-timing_var, timing_var)
                humanized.append({"hit": True, "velocity": velocity, "offset": timing_offset})
            else:
                humanized.append({"hit": False})
        return humanized

    def to_midi(self, pattern, bpm=120, ticks_per_beat=480):
        """パターンをMIDIに変換"""
        from mido import MidiFile, MidiTrack, Message, MetaMessage

        mid = MidiFile(ticks_per_beat=ticks_per_beat)
        track = MidiTrack()
        mid.tracks.append(track)
        track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm)))

        tick_per_16th = ticks_per_beat // 4

        for instrument, beats in pattern.items():
            note = self.GM_DRUMS[instrument]
            for i, beat in enumerate(beats):
                if isinstance(beat, dict) and beat["hit"]:
                    time = i * tick_per_16th
                    track.append(Message('note_on', note=note, velocity=beat["velocity"], time=0))
                    track.append(Message('note_off', note=note, time=tick_per_16th // 2))

        return mid
```

---

## 4. 比較表

### 4.1 AI作曲ツール比較

| 項目 | Magenta | MuseNet | AIVA | Amper/Shutterstock | MusicTransformer |
|------|---------|---------|------|-------------------|-----------------|
| 種別 | OSS | API(終了) | SaaS | SaaS | 研究 |
| MIDI出力 | 対応 | 非対応 | 対応 | 限定的 | 対応 |
| ジャンル | 多様 | 多様 | クラシック中心 | 多様 | 多様 |
| インタラクティブ | 対応 | 非対応 | 一部 | 非対応 | 非対応 |
| カスタマイズ | 高い | 低い | 中程度 | 低い | 高い |
| 商用利用 | Apache 2.0 | - | 有料プラン | 有料プラン | 研究用 |

### 4.2 コード進行生成手法の比較

| 手法 | 音楽理論知識 | 創造性 | 制御性 | 実装コスト |
|------|-------------|--------|--------|-----------|
| ルールベース | 必須 | 低い | 最高 | 低い |
| マルコフ連鎖 | 不要（学習） | 中程度 | 中程度 | 低い |
| LSTM/GRU | 不要（学習） | 高い | 低い | 中程度 |
| Transformer | 不要（学習） | 最高 | 低い | 高い |
| ルール+AI混合 | 一部必要 | 高い | 高い | 中程度 |

---

## 5. アンチパターン

### 5.1 アンチパターン: 音楽理論の完全無視

```python
# BAD: 完全ランダムなノート生成
def bad_melody_generation(length=32):
    notes = [random.randint(0, 127) for _ in range(length)]
    velocities = [random.randint(0, 127) for _ in range(length)]
    return notes, velocities  # 不協和音だらけ

# GOOD: スケール制約付き生成
def good_melody_generation(length=32, key="C", scale="major"):
    scales = {
        "major":     [0, 2, 4, 5, 7, 9, 11],
        "minor":     [0, 2, 3, 5, 7, 8, 10],
        "pentatonic": [0, 2, 4, 7, 9],
    }
    key_offset = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

    scale_notes = scales[scale]
    offset = key_offset[key]

    # 使用可能なノート（複数オクターブ）
    available_notes = []
    for octave in range(3, 6):  # C3-C6
        for degree in scale_notes:
            note = octave * 12 + degree + offset
            if 48 <= note <= 84:
                available_notes.append(note)

    # メロディ生成（隣接音への進行を優先）
    melody = [random.choice(available_notes)]
    for _ in range(length - 1):
        current = melody[-1]
        # 隣接する3音から選択（ステップワイズモーション）
        nearby = [n for n in available_notes if abs(n - current) <= 4]
        melody.append(random.choice(nearby))

    return melody
```

### 5.2 アンチパターン: クオンタイズの機械的適用

```python
# BAD: 全ノートを完全にクオンタイズ
def bad_quantize(midi_events, grid=480):
    for event in midi_events:
        event.time = round(event.time / grid) * grid
    return midi_events  # 機械的でグルーブが失われる

# GOOD: グルーブ保持クオンタイズ
def good_quantize(midi_events, grid=480, strength=0.75, swing=0.0):
    """
    Parameters:
        grid: クオンタイズグリッド（ticks）
        strength: 0.0（なし）〜 1.0（完全）
        swing: 0.0（ストレート）〜 1.0（フルスイング）
    """
    for event in midi_events:
        # 最寄りのグリッドポイント
        nearest = round(event.time / grid) * grid

        # スウィング適用（偶数拍のみずらす）
        if swing > 0 and (nearest // grid) % 2 == 1:
            nearest += int(grid * swing * 0.33)

        # strength に応じて部分的にクオンタイズ
        event.time = int(event.time + (nearest - event.time) * strength)

    return midi_events
```

---

## 6. FAQ

### Q1: AI作曲で生成されたMIDIデータの著作権はどうなりますか？

MIDIデータ自体は著作物として保護される可能性がありますが、AI生成物の著作権帰属は法的にグレーゾーンです。多くのAI作曲サービス（AIVA、Amper等）は有料プランで商用利用権を付与しています。OSSモデル（Magenta等）で生成した場合、学習データの著作権問題が残ります。安全策として、(1) AI生成を出発点に人間が大幅に編集する、(2) 商用利用を明示的に許可するサービスを使う、(3) 生成プロセスを記録しておく、が推奨されます。

### Q2: AIで生成したコード進行をDAWで使うにはどうすればよいですか？

最も簡単な方法は、(1) Python等でMIDIファイルを出力、(2) DAWにMIDIファイルをドラッグ&ドロップ。リアルタイム連携には、(1) 仮想MIDIポート（macOS: IAC Driver、Windows: loopMIDI）経由で送信、(2) Ableton LiveのMax for Live デバイス内でAIモデルを実行、(3) OSCプロトコルでDAWとAIエンジン間を通信。MidiTokやpretty_midiライブラリを使うと、AI出力をDAWフレンドリーなMIDIファイルに変換しやすくなります。

### Q3: メロディ生成AIの品質を向上させるコツは？

効果的な手法は5つあります。(1) スケール制約: 使用可能なノートをスケール内に限定。(2) コンテキスト長の拡大: より長い文脈を考慮するモデル（Transformer）を使用。(3) コード条件付き生成: コード進行に沿ったメロディ生成。(4) 後処理ルール: 連続する同音の制限、跳躍の制限、フレーズ終止の処理。(5) 温度パラメータの調整: 低い値（0.6-0.8）でより「安全な」メロディ、高い値（1.0-1.2）でより実験的なメロディ。

---

## まとめ

| 項目 | 要点 |
|------|------|
| MIDIデータ | ノート番号(0-127) + ベロシティ + タイミングの3要素 |
| トークン化 | REMI、Compound Word等でMIDIをLM入力に変換 |
| AI作曲 | Transformer ベースが主流。コンテキスト長が品質に直結 |
| コード進行 | 音楽理論ルール + AI のハイブリッドが実用的 |
| DAW連携 | MIDIファイル出力 or 仮想MIDIポートで接続 |
| 品質向上 | スケール制約 + コード条件付き + 適切な温度設定 |

## 次に読むべきガイド

- [00-music-generation.md](./00-music-generation.md) — 音楽生成（Suno、MusicGen）
- [02-audio-effects.md](./02-audio-effects.md) — 音声エフェクト
- [../03-development/01-audio-processing.md](../03-development/01-audio-processing.md) — 音声処理ライブラリ

## 参考文献

1. Huang, C.Z.A., et al. (2018). "Music Transformer: Generating Music with Long-Term Structure" — Music Transformer論文。相対位置エンコーディングによる長期構造の生成
2. Fraternali, D., et al. (2023). "MidiTok: A Python package for MIDI file tokenization" — MidiTok論文。MIDI トークン化ライブラリ
3. Roberts, A., et al. (2018). "A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music Generation" — MusicVAE論文。階層的潜在変数による音楽生成
