# MIDI×AI — 自動作曲、アレンジ、コード進行生成

> AIを活用したMIDI音楽制作（自動作曲、コード進行生成、アレンジ）の技術と実践を解説する

## この章で学ぶこと

1. MIDIデータの基礎知識とAI処理のためのデータ表現
2. AI自動作曲・コード進行生成の主要手法とモデル
3. DAW連携とプロダクションワークフローへの統合
4. メロディ生成・ベースライン生成・ドラムパターン生成の実装
5. MIDIデータの前処理・後処理テクニック
6. 実務で使えるAI作曲パイプラインの構築

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

### 1.2 MIDIファイルフォーマットの詳細

```
MIDIファイル構造:
==================================================

SMF (Standard MIDI File) フォーマット:

■ Format 0: 全トラックを1つにまとめたフォーマット
  ┌────────────┬────────────────────────┐
  │ Header     │ Track 0                │
  │ MThd       │ (全チャンネルのデータ)   │
  └────────────┴────────────────────────┘

■ Format 1: マルチトラック（同期再生）
  ┌────────────┬──────────┬──────────┬──────────┐
  │ Header     │ Track 0  │ Track 1  │ Track 2  │
  │ MThd       │(テンポ等)│(メロディ)│(ベース)  │
  └────────────┴──────────┴──────────┴──────────┘

■ Format 2: マルチトラック（独立再生、稀に使用）

Header Chunk (MThd):
  4D 54 68 64  = "MThd"
  00 00 00 06  = チャンクサイズ（常に6）
  00 01        = フォーマット（0/1/2）
  00 03        = トラック数
  01 E0        = 分解能（480 ticks/beat）

Track Chunk (MTrk):
  4D 54 72 6B  = "MTrk"
  xx xx xx xx  = チャンクサイズ
  [イベントデータ...]

MIDIチャンネルメッセージ一覧:
  0x80-0x8F  Note Off          (2 data bytes)
  0x90-0x9F  Note On           (2 data bytes)
  0xA0-0xAF  Polyphonic Aftertouch (2 data bytes)
  0xB0-0xBF  Control Change    (2 data bytes)
  0xC0-0xCF  Program Change    (1 data byte)
  0xD0-0xDF  Channel Aftertouch (1 data byte)
  0xE0-0xEF  Pitch Bend        (2 data bytes)

主要コントロールチェンジ番号:
  CC#1   = モジュレーション
  CC#7   = ボリューム
  CC#10  = パン
  CC#11  = エクスプレッション
  CC#64  = サステインペダル
  CC#91  = リバーブ
  CC#93  = コーラス
==================================================
```

### 1.3 MIDIデータのプログラミング

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

### 1.4 pretty_midi によるMIDI操作

```python
import pretty_midi
import numpy as np

class MIDIProcessor:
    """pretty_midiを使った高度なMIDI操作"""

    def __init__(self):
        self.pm = None

    def load(self, filepath: str):
        """MIDIファイルの読み込みと解析"""
        self.pm = pretty_midi.PrettyMIDI(filepath)
        return self.analyze()

    def analyze(self) -> dict:
        """MIDIファイルの詳細解析"""
        info = {
            "tempo_changes": self.pm.get_tempo_changes(),
            "time_signatures": [],
            "key_signatures": [],
            "instruments": [],
            "total_duration": self.pm.get_end_time(),
            "total_notes": 0,
        }

        for ts in self.pm.time_signature_changes:
            info["time_signatures"].append({
                "numerator": ts.numerator,
                "denominator": ts.denominator,
                "time": ts.time
            })

        for ks in self.pm.key_signature_changes:
            info["key_signatures"].append({
                "key_number": ks.key_number,
                "time": ks.time
            })

        for inst in self.pm.instruments:
            inst_info = {
                "name": inst.name,
                "program": inst.program,
                "is_drum": inst.is_drum,
                "note_count": len(inst.notes),
                "pitch_range": (
                    min(n.pitch for n in inst.notes) if inst.notes else 0,
                    max(n.pitch for n in inst.notes) if inst.notes else 0,
                ),
                "velocity_range": (
                    min(n.velocity for n in inst.notes) if inst.notes else 0,
                    max(n.velocity for n in inst.notes) if inst.notes else 0,
                ),
            }
            info["instruments"].append(inst_info)
            info["total_notes"] += len(inst.notes)

        return info

    def extract_piano_roll(self, fs: int = 100) -> np.ndarray:
        """ピアノロール行列の抽出（AI入力用）"""
        # shape: (128, time_steps)
        piano_roll = self.pm.get_piano_roll(fs=fs)
        return piano_roll

    def extract_chroma(self, fs: int = 100) -> np.ndarray:
        """クロマグラム抽出（コード検出用）"""
        # shape: (12, time_steps)
        chroma = self.pm.get_chroma(fs=fs)
        return chroma

    def transpose(self, semitones: int) -> pretty_midi.PrettyMIDI:
        """全体を移調"""
        for inst in self.pm.instruments:
            for note in inst.notes:
                note.pitch = max(0, min(127, note.pitch + semitones))
        return self.pm

    def change_tempo(self, factor: float) -> pretty_midi.PrettyMIDI:
        """テンポ変更（factor=2.0で倍速）"""
        for inst in self.pm.instruments:
            for note in inst.notes:
                note.start /= factor
                note.end /= factor
        return self.pm

    def split_by_instrument(self) -> dict:
        """楽器別にMIDIを分割"""
        result = {}
        for inst in self.pm.instruments:
            new_midi = pretty_midi.PrettyMIDI()
            new_inst = pretty_midi.Instrument(
                program=inst.program,
                is_drum=inst.is_drum,
                name=inst.name
            )
            new_inst.notes = inst.notes.copy()
            new_midi.instruments.append(new_inst)
            result[inst.name] = new_midi
        return result

    def merge_midis(self, midi_files: list) -> pretty_midi.PrettyMIDI:
        """複数のMIDIファイルをマージ"""
        merged = pretty_midi.PrettyMIDI()
        for filepath in midi_files:
            pm = pretty_midi.PrettyMIDI(filepath)
            for inst in pm.instruments:
                merged.instruments.append(inst)
        return merged

    def quantize_notes(self, grid_size: float = 0.125,
                       strength: float = 0.8):
        """ノートのクオンタイズ（grid_size秒単位）"""
        for inst in self.pm.instruments:
            for note in inst.notes:
                nearest_start = round(note.start / grid_size) * grid_size
                nearest_end = round(note.end / grid_size) * grid_size
                note.start += (nearest_start - note.start) * strength
                note.end += (nearest_end - note.end) * strength
                # 最小デュレーション保証
                if note.end - note.start < grid_size * 0.5:
                    note.end = note.start + grid_size * 0.5
        return self.pm

    def extract_melody(self, track_index: int = 0) -> list:
        """指定トラックからメロディ（最高音）を抽出"""
        inst = self.pm.instruments[track_index]
        # 時間でソート
        sorted_notes = sorted(inst.notes, key=lambda n: n.start)

        # 重複するノートのうち最高音だけを残す
        melody = []
        current_time = -1
        for note in sorted_notes:
            if abs(note.start - current_time) < 0.01:
                if note.pitch > melody[-1].pitch:
                    melody[-1] = note
            else:
                melody.append(note)
                current_time = note.start

        return melody
```

### 1.5 MIDIデータの数値表現とAI前処理

```python
import numpy as np
from typing import List, Tuple

class MIDIFeatureExtractor:
    """MIDI特徴量抽出（AI学習データ前処理用）"""

    def __init__(self, resolution: int = 480):
        self.resolution = resolution  # ticks per beat

    def notes_to_matrix(self, notes: list,
                        duration_beats: int = 32) -> np.ndarray:
        """ノートリストを行列表現に変換

        出力: (128, time_steps) の行列
        - 行: MIDIノート番号 (0-127)
        - 列: 時間ステップ（16分音符単位）
        - 値: ベロシティ (0-127)
        """
        steps_per_beat = 4  # 16分音符
        total_steps = duration_beats * steps_per_beat
        matrix = np.zeros((128, total_steps), dtype=np.float32)

        for note in notes:
            start_step = int(note["start"] * steps_per_beat)
            end_step = int(note["end"] * steps_per_beat)
            pitch = note["pitch"]
            velocity = note["velocity"]

            if 0 <= pitch <= 127:
                start_step = max(0, min(start_step, total_steps - 1))
                end_step = max(start_step + 1, min(end_step, total_steps))
                matrix[pitch, start_step:end_step] = velocity / 127.0

        return matrix

    def matrix_to_notes(self, matrix: np.ndarray,
                        threshold: float = 0.1) -> list:
        """行列表現からノートリストに復元"""
        notes = []
        steps_per_beat = 4

        for pitch in range(128):
            in_note = False
            start = 0
            for step in range(matrix.shape[1]):
                if matrix[pitch, step] > threshold and not in_note:
                    in_note = True
                    start = step
                elif (matrix[pitch, step] <= threshold or
                      step == matrix.shape[1] - 1) and in_note:
                    in_note = False
                    velocity = int(np.max(
                        matrix[pitch, start:step + 1]) * 127)
                    notes.append({
                        "pitch": pitch,
                        "start": start / steps_per_beat,
                        "end": step / steps_per_beat,
                        "velocity": velocity,
                    })

        return sorted(notes, key=lambda n: n["start"])

    def extract_rhythm_pattern(self, notes: list,
                                beats: int = 4) -> np.ndarray:
        """リズムパターンの抽出（16分音符グリッド）"""
        steps = beats * 4
        pattern = np.zeros(steps, dtype=np.float32)

        for note in notes:
            step = int(note["start"] * 4) % steps
            pattern[step] = max(pattern[step],
                                note["velocity"] / 127.0)

        return pattern

    def compute_pitch_histogram(self, notes: list) -> np.ndarray:
        """ピッチクラスヒストグラム（12次元、コード検出用）"""
        histogram = np.zeros(12, dtype=np.float32)
        for note in notes:
            pitch_class = note["pitch"] % 12
            duration = note["end"] - note["start"]
            histogram[pitch_class] += duration

        # 正規化
        total = np.sum(histogram)
        if total > 0:
            histogram /= total

        return histogram

    def compute_interval_histogram(self, notes: list) -> np.ndarray:
        """音程ヒストグラム（メロディ特徴量）"""
        sorted_notes = sorted(notes, key=lambda n: n["start"])
        intervals = np.zeros(25, dtype=np.float32)  # -12 to +12

        for i in range(1, len(sorted_notes)):
            interval = sorted_notes[i]["pitch"] - sorted_notes[i-1]["pitch"]
            interval = max(-12, min(12, interval))
            intervals[interval + 12] += 1

        total = np.sum(intervals)
        if total > 0:
            intervals /= total
        return intervals

    def compute_velocity_statistics(self, notes: list) -> dict:
        """ベロシティの統計量"""
        velocities = [n["velocity"] for n in notes]
        if not velocities:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        return {
            "mean": np.mean(velocities),
            "std": np.std(velocities),
            "min": np.min(velocities),
            "max": np.max(velocities),
        }

    def compute_note_density(self, notes: list,
                              window_beats: float = 1.0) -> list:
        """ノート密度の時系列（拍単位）"""
        if not notes:
            return []
        max_time = max(n["end"] for n in notes)
        windows = int(np.ceil(max_time / window_beats))
        density = []

        for w in range(windows):
            start = w * window_beats
            end = (w + 1) * window_beats
            count = sum(1 for n in notes
                        if n["start"] < end and n["end"] > start)
            density.append(count)

        return density
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

### 2.2 各種トークン化手法の比較と実装

```python
from miditok import REMI, TSD, MIDILike, Structured, CPWord

class TokenizerComparison:
    """各トークン化手法の比較実装"""

    def __init__(self, config: TokenizerConfig):
        self.config = config

    def compare_tokenizers(self, midi_path: str) -> dict:
        """全トークナイザで同一MIDIをトークン化し比較"""
        tokenizers = {
            "REMI": REMI(self.config),
            "TSD": TSD(self.config),
            "MIDILike": MIDILike(self.config),
            "Structured": Structured(self.config),
            "CPWord": CPWord(self.config),
        }

        results = {}
        for name, tok in tokenizers.items():
            tokens = tok(midi_path)
            results[name] = {
                "token_count": len(tokens.ids),
                "vocab_size": len(tok),
                "tokens_sample": tokens.tokens[:20],
                "description": self._get_description(name),
            }

        return results

    def _get_description(self, name: str) -> str:
        descriptions = {
            "REMI": "位置ベース。Bar/Position/Pitch/Velocity/Duration。"
                    "最も直感的で広く使われる。",
            "TSD": "Time Shift + Duration。相対的な時間表現。"
                   "シーケンスが短くなる傾向。",
            "MIDILike": "生MIDIメッセージに近い表現。"
                        "Note On/Off を明示的に表現。",
            "Structured": "トラック/小節/位置を階層的に表現。"
                          "マルチトラック向き。",
            "CPWord": "Compound Word。複数属性を1トークンに圧縮。"
                      "シーケンス長を大幅に削減。",
        }
        return descriptions.get(name, "")


# トークン化の具体例
"""
REMI トークン列の例（Cメジャーコード、4分音符）:

  Bar_0                    ← 小節0の開始
  Position_0               ← 拍頭（位置0）
  Chord_C:maj              ← コード検出結果
  Pitch_60                 ← ノートC4
  Velocity_80              ← ベロシティ
  Duration_2.0             ← 2拍（4分音符x2）
  Pitch_64                 ← ノートE4
  Velocity_80
  Duration_2.0
  Pitch_67                 ← ノートG4
  Velocity_80
  Duration_2.0

TSD トークン列の例（同じ内容）:

  Pitch_60
  Velocity_80
  Duration_480             ← ticks
  TimeShift_0              ← 同時発音
  Pitch_64
  Velocity_80
  Duration_480
  TimeShift_0
  Pitch_67
  Velocity_80
  Duration_480
  TimeShift_960            ← 次のイベントまでの時間
"""
```

### 2.3 Transformer による作曲

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

### 2.4 相対位置エンコーディングによる Music Transformer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RelativeAttention(nn.Module):
    """相対位置エンコーディング付きSelf-Attention

    Music Transformer (Huang et al., 2018) の核心技術。
    絶対位置ではなく、ノート間の相対的な距離を考慮することで
    長期的な構造（反復、変奏）を捉える。
    """

    def __init__(self, d_model: int, n_heads: int,
                 max_relative_position: int = 512):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.max_relative_position = max_relative_position

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # 相対位置埋め込み
        self.relative_embeddings = nn.Embedding(
            2 * max_relative_position + 1, self.d_k
        )

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Q, K, V の計算
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k)

        Q = Q.transpose(1, 2)  # (batch, heads, seq, d_k)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # コンテンツベースのattention
        content_score = torch.matmul(Q, K.transpose(-2, -1))

        # 相対位置ベースのattention
        positions = torch.arange(seq_len, device=x.device)
        relative_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_pos = relative_pos.clamp(
            -self.max_relative_position,
            self.max_relative_position
        ) + self.max_relative_position

        rel_embeddings = self.relative_embeddings(relative_pos)
        position_score = torch.einsum(
            'bhqd,qkd->bhqk', Q, rel_embeddings
        )

        # 統合スコア
        scores = (content_score + position_score) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        return self.W_o(output)


class MusicTransformerV2(nn.Module):
    """相対位置エンコーディング付き Music Transformer"""

    def __init__(self, vocab_size=512, d_model=512,
                 n_heads=8, n_layers=6, max_seq_len=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': RelativeAttention(d_model, n_heads),
                'norm1': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(0.1),
                ),
                'norm2': nn.LayerNorm(d_model),
            })
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.shape[1]
        h = self.embedding(x)

        # Causal mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1
        ).bool()

        for layer in self.layers:
            # Self-Attention with relative position
            h_norm = layer['norm1'](h)
            h = h + layer['attention'](h_norm, mask=mask)
            # Feed-Forward
            h_norm = layer['norm2'](h)
            h = h + layer['ffn'](h_norm)

        return self.output_proj(h)
```

### 2.5 学習パイプライン

```python
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class MIDIDataset(Dataset):
    """MIDIトークンデータセット"""

    def __init__(self, data_dir: str, tokenizer, max_seq_len: int = 1024):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.sequences = []

        midi_files = list(Path(data_dir).glob("**/*.mid")) + \
                     list(Path(data_dir).glob("**/*.midi"))

        for midi_file in midi_files:
            try:
                tokens = tokenizer(str(midi_file))
                ids = tokens.ids
                # 固定長シーケンスに分割
                for i in range(0, len(ids) - max_seq_len, max_seq_len // 2):
                    seq = ids[i:i + max_seq_len + 1]
                    if len(seq) == max_seq_len + 1:
                        self.sequences.append(seq)
            except Exception as e:
                print(f"スキップ: {midi_file} - {e}")

        print(f"読み込み完了: {len(self.sequences)} シーケンス "
              f"({len(midi_files)} MIDIファイル)")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y


class MusicTrainer:
    """Music Transformer の学習"""

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-4, weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader: DataLoader) -> float:
        """1エポックの学習"""
        self.model.train()
        total_loss = 0
        total_batches = 0

        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x)
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            self.optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            if batch_idx % 100 == 0:
                avg_loss = total_loss / total_batches
                print(f"  Batch {batch_idx}: loss={avg_loss:.4f}")

        self.scheduler.step()
        return total_loss / total_batches

    def train(self, data_dir: str, epochs: int = 50,
              batch_size: int = 16, save_dir: str = "checkpoints"):
        """学習ループ"""
        dataset = MIDIDataset(data_dir, self.tokenizer)
        dataloader = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=True, num_workers=4
        )

        Path(save_dir).mkdir(exist_ok=True)
        best_loss = float("inf")

        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataloader)
            print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(
                    self.model.state_dict(),
                    f"{save_dir}/best_model.pt"
                )
                print(f"  ベストモデル保存 (loss={best_loss:.4f})")

            # 定期的にサンプル生成
            if (epoch + 1) % 10 == 0:
                self._generate_sample(epoch + 1, save_dir)

    def _generate_sample(self, epoch: int, save_dir: str):
        """学習中のサンプル生成"""
        self.model.eval()
        seed = torch.randint(0, 100, (1, 16)).to(self.device)
        with torch.no_grad():
            generated = self.model.generate(
                seed, max_length=256,
                temperature=0.9, top_k=50
            )
        tokens_list = generated[0].cpu().tolist()
        midi = self.tokenizer.decode(tokens_list)
        midi.dump_midi(f"{save_dir}/sample_epoch{epoch}.mid")
        print(f"  サンプル生成: sample_epoch{epoch}.mid")
```

### 2.6 コード進行生成

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
        "逆循環":         ["I", "vi", "ii", "V"],
        "レディオヘッド": ["I", "iii", "vi", "IV"],
        "ネオソウル":     ["ii", "V", "I", "vi"],
        "ボサノバ":       ["I", "vi", "ii", "V"],
    }

    # テンションコード定義
    TENSION_CHORDS = {
        "Imaj7":   {"notes": [0, 4, 7, 11]},
        "ii7":     {"notes": [2, 5, 9, 0]},
        "iii7":    {"notes": [4, 7, 11, 2]},
        "IVmaj7":  {"notes": [5, 9, 0, 4]},
        "V7":      {"notes": [7, 11, 2, 5]},
        "vi7":     {"notes": [9, 0, 4, 7]},
        "Vaug":    {"notes": [7, 11, 3]},
        "bVII":    {"notes": [10, 2, 5]},
        "IV#dim":  {"notes": [6, 9, 0]},
    }

    def generate(self, style: str = "ポップ定番", key: str = "C",
                 bars: int = 8, use_tensions: bool = False) -> list:
        """コード進行を生成"""
        base_progression = self.COMMON_PROGRESSIONS.get(style)
        if not base_progression:
            base_progression = random.choice(
                list(self.COMMON_PROGRESSIONS.values()))

        # barsに合わせて繰り返しまたはバリエーション生成
        progression = []
        while len(progression) < bars:
            progression.extend(base_progression)
        progression = progression[:bars]

        # テンション付加
        if use_tensions:
            progression = self._add_tensions(progression)

        # キー変換
        transposed = self._transpose(progression, key)
        return transposed

    def _add_tensions(self, progression: list) -> list:
        """確率的にテンションコードを付加"""
        tension_map = {
            "I": "Imaj7", "ii": "ii7", "iii": "iii7",
            "IV": "IVmaj7", "V": "V7", "vi": "vi7",
        }
        result = []
        for chord in progression:
            if chord in tension_map and random.random() < 0.5:
                result.append(tension_map[chord])
            else:
                result.append(chord)
        return result

    def _transpose(self, progression, target_key):
        """キー変換"""
        key_offsets = {
            "C": 0, "C#": 1, "D": 2, "Eb": 3, "E": 4, "F": 5,
            "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11,
        }
        offset = key_offsets.get(target_key, 0)
        return [(chord, offset) for chord in progression]

    def generate_with_markov(self, length: int = 8,
                              start: str = "I") -> list:
        """マルコフ連鎖によるコード進行生成"""
        # 遷移確率行列（音楽理論ベース）
        transitions = {
            "I":   {"ii": 0.15, "iii": 0.05, "IV": 0.30,
                    "V": 0.30, "vi": 0.15, "vii": 0.05},
            "ii":  {"V": 0.50, "vii": 0.10, "I": 0.10,
                    "iii": 0.10, "IV": 0.10, "vi": 0.10},
            "iii": {"vi": 0.30, "IV": 0.30, "ii": 0.20,
                    "I": 0.10, "V": 0.10},
            "IV":  {"V": 0.35, "I": 0.25, "ii": 0.15,
                    "vi": 0.15, "vii": 0.10},
            "V":   {"I": 0.50, "vi": 0.25, "IV": 0.10,
                    "iii": 0.10, "ii": 0.05},
            "vi":  {"IV": 0.30, "ii": 0.25, "V": 0.20,
                    "I": 0.15, "iii": 0.10},
            "vii": {"I": 0.50, "iii": 0.20, "vi": 0.15,
                    "IV": 0.10, "V": 0.05},
        }

        progression = [start]
        current = start

        for _ in range(length - 1):
            probs = transitions[current]
            chords = list(probs.keys())
            weights = list(probs.values())
            next_chord = random.choices(chords, weights=weights, k=1)[0]
            progression.append(next_chord)
            current = next_chord

        return progression

    def to_midi(self, progression: list, key: str = "C",
                bpm: int = 120, beats_per_chord: int = 4) -> 'MidiFile':
        """コード進行をMIDIに変換"""
        from mido import MidiFile, MidiTrack, Message, MetaMessage
        import mido

        key_offsets = {
            "C": 0, "C#": 1, "D": 2, "Eb": 3, "E": 4, "F": 5,
            "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11,
        }
        offset = key_offsets.get(key, 0)
        ticks_per_beat = 480

        mid = MidiFile(ticks_per_beat=ticks_per_beat)
        track = MidiTrack()
        mid.tracks.append(track)
        track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm)))

        for chord_name in progression:
            if chord_name in self.DIATONIC_CHORDS:
                chord_data = self.DIATONIC_CHORDS[chord_name]
            elif chord_name in self.TENSION_CHORDS:
                chord_data = self.TENSION_CHORDS[chord_name]
            else:
                continue

            notes = [(n + offset) % 12 + 60 for n in chord_data["notes"]]
            duration = ticks_per_beat * beats_per_chord

            for i, note in enumerate(notes):
                track.append(Message(
                    'note_on', note=note, velocity=80, time=0
                ))

            for i, note in enumerate(notes):
                time = duration if i == 0 else 0
                track.append(Message(
                    'note_off', note=note, velocity=0, time=time
                ))

        return mid


# 使用例
gen = ChordProgressionGenerator()

# パターンベース生成
chords = gen.generate(style="小室進行", key="A", bars=8)
print(f"パターン生成: {chords}")

# マルコフ連鎖生成
markov_chords = gen.generate_with_markov(length=8, start="I")
print(f"マルコフ生成: {markov_chords}")
```

---

## 3. メロディ生成

### 3.1 条件付きメロディ生成

```python
import numpy as np
import random

class MelodyGenerator:
    """コード進行に基づくメロディ生成"""

    SCALES = {
        "major":      [0, 2, 4, 5, 7, 9, 11],
        "minor":      [0, 2, 3, 5, 7, 8, 10],
        "dorian":     [0, 2, 3, 5, 7, 9, 10],
        "mixolydian": [0, 2, 4, 5, 7, 9, 10],
        "pentatonic": [0, 2, 4, 7, 9],
        "blues":      [0, 3, 5, 6, 7, 10],
    }

    # コードトーン重み（コード構成音を優先）
    CHORD_TONE_WEIGHT = 0.6
    SCALE_TONE_WEIGHT = 0.3
    PASSING_TONE_WEIGHT = 0.1

    def __init__(self, key: str = "C", scale: str = "major",
                 octave_range: tuple = (4, 6)):
        self.key = key
        self.scale = scale
        self.octave_range = octave_range
        self.key_offset = {
            "C": 0, "C#": 1, "D": 2, "Eb": 3, "E": 4, "F": 5,
            "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11,
        }.get(key, 0)

    def generate(self, chord_progression: list,
                 notes_per_chord: int = 8,
                 style: str = "stepwise") -> list:
        """コード進行に沿ったメロディを生成"""
        melody = []
        prev_pitch = 60 + self.key_offset  # 開始音

        for chord in chord_progression:
            chord_notes = self._get_chord_tones(chord)
            scale_notes = self._get_available_notes()

            for i in range(notes_per_chord):
                if style == "stepwise":
                    pitch = self._stepwise_motion(
                        prev_pitch, chord_notes, scale_notes
                    )
                elif style == "arpeggiated":
                    pitch = self._arpeggio_motion(
                        prev_pitch, chord_notes, i
                    )
                elif style == "mixed":
                    if random.random() < 0.3:
                        pitch = self._arpeggio_motion(
                            prev_pitch, chord_notes, i
                        )
                    else:
                        pitch = self._stepwise_motion(
                            prev_pitch, chord_notes, scale_notes
                        )
                else:
                    pitch = random.choice(scale_notes)

                velocity = self._generate_velocity(i, notes_per_chord)
                duration = self._generate_duration(i, notes_per_chord)

                melody.append({
                    "pitch": pitch,
                    "velocity": velocity,
                    "duration": duration,
                    "chord": chord,
                })
                prev_pitch = pitch

        return melody

    def _stepwise_motion(self, prev_pitch: int,
                          chord_notes: list,
                          scale_notes: list) -> int:
        """隣接音進行（ステップワイズモーション）"""
        # 近隣のスケール内ノートを候補に
        candidates = []
        weights = []

        for note in scale_notes:
            distance = abs(note - prev_pitch)
            if distance <= 7:  # 5度以内
                candidates.append(note)
                # コードトーンは重み高い
                if note % 12 in [n % 12 for n in chord_notes]:
                    weight = self.CHORD_TONE_WEIGHT
                else:
                    weight = self.SCALE_TONE_WEIGHT
                # 近い音ほど重み高い
                weight *= max(0.1, 1.0 - distance / 7.0)
                weights.append(weight)

        if not candidates:
            return prev_pitch

        total = sum(weights)
        weights = [w / total for w in weights]
        return random.choices(candidates, weights=weights, k=1)[0]

    def _arpeggio_motion(self, prev_pitch: int,
                          chord_notes: list, index: int) -> int:
        """アルペジオ進行"""
        if not chord_notes:
            return prev_pitch
        target = chord_notes[index % len(chord_notes)]
        # 最も近いオクターブを選択
        best_pitch = target
        best_distance = abs(target - prev_pitch)
        for octave_shift in [-12, 0, 12]:
            candidate = target + octave_shift
            if (self.octave_range[0] * 12 <= candidate <=
                    self.octave_range[1] * 12):
                distance = abs(candidate - prev_pitch)
                if distance < best_distance:
                    best_pitch = candidate
                    best_distance = distance
        return best_pitch

    def _get_chord_tones(self, chord: str) -> list:
        """コード構成音をMIDIノート番号で返す"""
        chord_intervals = {
            "I": [0, 4, 7], "ii": [2, 5, 9], "iii": [4, 7, 11],
            "IV": [5, 9, 0], "V": [7, 11, 2], "vi": [9, 0, 4],
            "vii": [11, 2, 5],
        }
        intervals = chord_intervals.get(chord, [0, 4, 7])
        notes = []
        for octave in range(self.octave_range[0], self.octave_range[1] + 1):
            for interval in intervals:
                note = octave * 12 + (interval + self.key_offset) % 12
                notes.append(note)
        return notes

    def _get_available_notes(self) -> list:
        """使用可能なスケール内ノートを取得"""
        scale = self.SCALES[self.scale]
        notes = []
        for octave in range(self.octave_range[0], self.octave_range[1] + 1):
            for degree in scale:
                note = octave * 12 + (degree + self.key_offset) % 12
                notes.append(note)
        return sorted(notes)

    def _generate_velocity(self, position: int, total: int) -> int:
        """位置に応じたベロシティ生成"""
        # 拍頭は強め、裏拍は弱め
        if position % 4 == 0:
            base = 90
        elif position % 2 == 0:
            base = 75
        else:
            base = 65
        variation = random.randint(-8, 8)
        return max(40, min(127, base + variation))

    def _generate_duration(self, position: int, total: int) -> float:
        """位置に応じたデュレーション生成（拍単位）"""
        durations = [0.25, 0.5, 0.5, 0.25, 0.5, 0.25, 0.5, 0.25]
        return durations[position % len(durations)]
```

### 3.2 VAEによるメロディ生成（MusicVAE風）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MelodyVAE(nn.Module):
    """VAEベースのメロディ生成モデル

    MusicVAE (Roberts et al., 2018) からインスパイアされた実装。
    潜在空間での補間によりメロディのモーフィングが可能。
    """

    def __init__(self, input_dim=128, hidden_dim=256,
                 latent_dim=64, seq_len=32):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # Encoder (Bidirectional LSTM)
        self.encoder = nn.LSTM(
            input_dim, hidden_dim, num_layers=2,
            batch_first=True, bidirectional=True
        )
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        # Decoder (LSTM with teacher forcing)
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_dim + hidden_dim, hidden_dim,
            num_layers=2, batch_first=True
        )
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """入力メロディを潜在空間にエンコード"""
        _, (h, _) = self.encoder(x)
        # 双方向の最終隠れ状態を結合
        h = torch.cat([h[-2], h[-1]], dim=-1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """再パラメータ化トリック"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, target=None):
        """潜在変数からメロディを復元"""
        batch_size = z.shape[0]
        decoder_init = self.decoder_input(z).unsqueeze(0).repeat(2, 1, 1)
        h = (decoder_init, torch.zeros_like(decoder_init))

        outputs = []
        current_input = torch.zeros(batch_size, 1, 128).to(z.device)

        for t in range(self.seq_len):
            z_expanded = z.unsqueeze(1)
            decoder_in = torch.cat([current_input, z_expanded], dim=-1)
            output, h = self.decoder(decoder_in, h)
            note_logits = self.output_proj(output)
            outputs.append(note_logits)

            if target is not None and self.training:
                current_input = target[:, t:t+1, :]
            else:
                current_input = torch.softmax(note_logits, dim=-1)

        return torch.cat(outputs, dim=1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, target=x)
        return recon, mu, logvar

    def interpolate(self, melody_a, melody_b, steps=8):
        """2つのメロディ間の潜在空間補間"""
        self.eval()
        with torch.no_grad():
            mu_a, _ = self.encode(melody_a.unsqueeze(0))
            mu_b, _ = self.encode(melody_b.unsqueeze(0))

            interpolated = []
            for alpha in np.linspace(0, 1, steps):
                z = mu_a * (1 - alpha) + mu_b * alpha
                melody = self.decode(z)
                interpolated.append(melody.squeeze(0))

        return interpolated

    def sample(self, n_samples=1, temperature=1.0):
        """潜在空間からランダムサンプリング"""
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim) * temperature
            melodies = self.decode(z)
        return melodies
```

---

## 4. DAW連携

### 4.1 DAW連携アーキテクチャ

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

### 4.2 仮想MIDIポートによるリアルタイム連携

```python
import mido
import time
import threading

class RealtimeMIDIBridge:
    """DAWとAIエンジンのリアルタイムMIDI連携"""

    def __init__(self, input_port_name: str = None,
                 output_port_name: str = None):
        """
        macOS: IAC Driver を使用
        Windows: loopMIDI を使用
        Linux: ALSA仮想ポートを使用
        """
        if input_port_name is None:
            available = mido.get_input_names()
            print(f"利用可能な入力ポート: {available}")
            input_port_name = available[0] if available else None

        if output_port_name is None:
            available = mido.get_output_names()
            print(f"利用可能な出力ポート: {available}")
            output_port_name = available[0] if available else None

        self.input_port = mido.open_input(input_port_name)
        self.output_port = mido.open_output(output_port_name)
        self.running = False
        self.note_buffer = []
        self.callback = None

    def set_callback(self, callback):
        """MIDI入力時のコールバック設定"""
        self.callback = callback

    def start(self):
        """リアルタイム処理開始"""
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop)
        self.thread.daemon = True
        self.thread.start()
        print("MIDI Bridge 開始")

    def stop(self):
        """停止"""
        self.running = False
        self.thread.join()
        self.input_port.close()
        self.output_port.close()
        print("MIDI Bridge 停止")

    def _listen_loop(self):
        """MIDIメッセージ受信ループ"""
        while self.running:
            for msg in self.input_port.iter_pending():
                if msg.type == 'note_on' and msg.velocity > 0:
                    self.note_buffer.append(msg)

                    # バッファが一定量たまったらAI処理
                    if len(self.note_buffer) >= 4:
                        if self.callback:
                            response = self.callback(
                                self.note_buffer.copy()
                            )
                            self._send_response(response)
                        self.note_buffer.clear()

            time.sleep(0.001)  # 1ms間隔

    def _send_response(self, midi_messages: list):
        """AI生成結果をDAWに送信"""
        for msg in midi_messages:
            self.output_port.send(msg)

    def send_note(self, note: int, velocity: int = 80,
                  channel: int = 0, duration: float = 0.5):
        """単一ノートの送信"""
        on = mido.Message('note_on', note=note,
                          velocity=velocity, channel=channel)
        off = mido.Message('note_off', note=note,
                           velocity=0, channel=channel)
        self.output_port.send(on)
        time.sleep(duration)
        self.output_port.send(off)

    def send_chord(self, notes: list, velocity: int = 80,
                   channel: int = 0, duration: float = 1.0):
        """コードの送信"""
        for note in notes:
            on = mido.Message('note_on', note=note,
                              velocity=velocity, channel=channel)
            self.output_port.send(on)

        time.sleep(duration)

        for note in notes:
            off = mido.Message('note_off', note=note,
                               velocity=0, channel=channel)
            self.output_port.send(off)


# 使用例: リアルタイムハーモナイズ
def harmonize_callback(input_notes: list) -> list:
    """入力ノートに対してハーモニーを生成"""
    response = []
    for msg in input_notes:
        # 3度上と5度上のハーモニーを追加
        harmony_3rd = mido.Message(
            'note_on', note=min(127, msg.note + 4),
            velocity=int(msg.velocity * 0.7), channel=1
        )
        harmony_5th = mido.Message(
            'note_on', note=min(127, msg.note + 7),
            velocity=int(msg.velocity * 0.6), channel=1
        )
        response.extend([harmony_3rd, harmony_5th])
    return response
```

### 4.3 ドラムパターン生成

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
        "clap":     39,
        "rimshot":  37,
        "cowbell":  56,
        "tambourine": 54,
        "shaker":   70,
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
        "hip_hop": {
            "kick":    [1,0,0,0, 0,0,1,0, 0,0,1,0, 0,0,0,0],
            "snare":   [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
            "hihat_c": [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
        },
        "bossa_nova": {
            "kick":    [1,0,0,1, 0,0,1,0, 0,1,0,0, 1,0,0,0],
            "rimshot": [0,0,1,0, 0,1,0,0, 1,0,0,1, 0,0,1,0],
            "hihat_c": [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
        },
        "shuffle": {
            "kick":    [1,0,0,0, 0,0,1,0, 1,0,0,0, 0,0,0,0],
            "snare":   [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
            "hihat_c": [1,0,1,1, 0,1,1,0, 1,1,0,1, 1,0,1,0],
        },
        "drum_and_bass": {
            "kick":    [1,0,0,0, 0,0,1,0, 0,1,0,0, 0,0,0,0],
            "snare":   [0,0,0,0, 1,0,0,1, 0,0,0,0, 1,0,0,0],
            "hihat_c": [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
            "ride":    [0,0,0,0, 0,0,0,0, 0,0,1,0, 0,0,0,0],
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

    def generate_variation(self, base_style: str = "basic_rock",
                            variation_amount: float = 0.2) -> dict:
        """ベースパターンにバリエーションを加える"""
        base = self.PATTERNS.get(base_style, self.PATTERNS["basic_rock"])
        varied = {}

        for instrument, beat in base.items():
            new_beat = beat.copy()
            for i in range(len(new_beat)):
                if random.random() < variation_amount:
                    new_beat[i] = 1 - new_beat[i]  # トグル
            varied[instrument] = new_beat

        return varied

    def generate_fill(self, length_16ths: int = 16) -> dict:
        """ドラムフィルの生成"""
        fill = {
            "snare": [0] * length_16ths,
            "tom_high": [0] * length_16ths,
            "tom_mid": [0] * length_16ths,
            "tom_low": [0] * length_16ths,
            "crash": [0] * length_16ths,
        }

        # フィルパターン生成（徐々に密度を上げる）
        for i in range(length_16ths):
            density = (i + 1) / length_16ths
            if random.random() < density * 0.8:
                # 高→低のタム回し
                if i < length_16ths * 0.33:
                    fill["tom_high"][i] = 1
                elif i < length_16ths * 0.66:
                    fill["tom_mid"][i] = 1
                else:
                    if random.random() < 0.5:
                        fill["tom_low"][i] = 1
                    else:
                        fill["snare"][i] = 1

        # 最後にクラッシュ
        fill["crash"][-1] = 1

        return fill

    def _humanize(self, pattern, timing_var=10, velocity_var=15):
        """人間らしさを付与（タイミング/ベロシティのゆらぎ）"""
        humanized = []
        for hit in pattern:
            if hit:
                velocity = max(40, min(127, 80 + np.random.randint(
                    -velocity_var, velocity_var)))
                timing_offset = np.random.randint(
                    -timing_var, timing_var)
                humanized.append({
                    "hit": True,
                    "velocity": velocity,
                    "offset": timing_offset
                })
            else:
                humanized.append({"hit": False})
        return humanized

    def to_midi(self, pattern, bpm=120, ticks_per_beat=480):
        """パターンをMIDIに変換"""
        from mido import MidiFile, MidiTrack, Message, MetaMessage
        import mido

        mid = MidiFile(ticks_per_beat=ticks_per_beat)
        track = MidiTrack()
        mid.tracks.append(track)
        track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm)))

        tick_per_16th = ticks_per_beat // 4

        # 全楽器のイベントを時系列にソート
        events = []
        for instrument, beats in pattern.items():
            note = self.GM_DRUMS[instrument]
            for i, beat in enumerate(beats):
                if isinstance(beat, dict) and beat["hit"]:
                    time_ticks = i * tick_per_16th + beat.get("offset", 0)
                    events.append({
                        "time": max(0, time_ticks),
                        "note": note,
                        "velocity": beat["velocity"],
                    })
                elif isinstance(beat, int) and beat == 1:
                    events.append({
                        "time": i * tick_per_16th,
                        "note": note,
                        "velocity": 80,
                    })

        events.sort(key=lambda e: e["time"])

        prev_time = 0
        for event in events:
            delta = event["time"] - prev_time
            track.append(Message(
                'note_on', note=event["note"],
                velocity=event["velocity"],
                time=max(0, delta), channel=9  # GM Drum Channel
            ))
            track.append(Message(
                'note_off', note=event["note"],
                velocity=0,
                time=tick_per_16th // 2, channel=9
            ))
            prev_time = event["time"] + tick_per_16th // 2

        return mid
```

### 4.4 ベースライン生成

```python
class BasslineGenerator:
    """コード進行に基づくベースライン自動生成"""

    STYLES = {
        "root_notes": "ルート音のみ（シンプル）",
        "walking": "ウォーキングベース（ジャズ）",
        "syncopated": "シンコペーション（ファンク）",
        "octave": "オクターブ奏法（ロック）",
        "arpeggiated": "アルペジオ（ポップ）",
    }

    def __init__(self, key: str = "C"):
        self.key = key
        self.key_offset = {
            "C": 0, "C#": 1, "D": 2, "Eb": 3, "E": 4, "F": 5,
            "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11,
        }.get(key, 0)

    def generate(self, chord_progression: list,
                 style: str = "root_notes",
                 beats_per_chord: int = 4) -> list:
        """ベースラインを生成"""
        bassline = []

        for chord in chord_progression:
            root = self._get_root_note(chord)
            chord_tones = self._get_chord_tones(chord)

            if style == "root_notes":
                notes = self._root_note_pattern(
                    root, beats_per_chord)
            elif style == "walking":
                notes = self._walking_pattern(
                    root, chord_tones, beats_per_chord)
            elif style == "syncopated":
                notes = self._syncopated_pattern(
                    root, chord_tones, beats_per_chord)
            elif style == "octave":
                notes = self._octave_pattern(
                    root, beats_per_chord)
            elif style == "arpeggiated":
                notes = self._arpeggiated_pattern(
                    root, chord_tones, beats_per_chord)
            else:
                notes = self._root_note_pattern(
                    root, beats_per_chord)

            bassline.extend(notes)

        return bassline

    def _get_root_note(self, chord: str) -> int:
        """コードのルート音をMIDIノート番号で取得（ベース音域）"""
        chord_roots = {
            "I": 0, "ii": 2, "iii": 4, "IV": 5,
            "V": 7, "vi": 9, "vii": 11,
        }
        interval = chord_roots.get(chord, 0)
        # ベース音域: C2-C4 (36-60)
        return 36 + (interval + self.key_offset) % 12

    def _get_chord_tones(self, chord: str) -> list:
        """コードトーンをベース音域で取得"""
        root = self._get_root_note(chord)
        if chord in ["I", "IV", "V"]:
            return [root, root + 4, root + 7]  # メジャー
        elif chord in ["ii", "iii", "vi"]:
            return [root, root + 3, root + 7]  # マイナー
        else:
            return [root, root + 3, root + 6]  # ディミニッシュ

    def _root_note_pattern(self, root: int,
                            beats: int) -> list:
        """ルート音パターン"""
        notes = []
        for i in range(beats * 2):  # 8分音符
            if i % 2 == 0:
                notes.append({
                    "pitch": root,
                    "velocity": 90 if i % 4 == 0 else 70,
                    "duration": 0.5,
                })
            else:
                notes.append({
                    "pitch": 0,  # 休符
                    "velocity": 0,
                    "duration": 0.5,
                })
        return notes

    def _walking_pattern(self, root: int,
                          chord_tones: list,
                          beats: int) -> list:
        """ウォーキングベースパターン"""
        notes = []
        scale = [0, 2, 3, 5, 7, 9, 10]  # ミクソリディアン的

        for i in range(beats):
            if i == 0:
                pitch = root
            elif i == beats - 1:
                # 次のコードへのアプローチノート
                pitch = root + random.choice([-1, 1, -2, 2])
            else:
                degree = random.choice(scale)
                pitch = root + degree
                if pitch > root + 12:
                    pitch -= 12

            notes.append({
                "pitch": pitch,
                "velocity": 80 + random.randint(-10, 10),
                "duration": 1.0,
            })

        return notes

    def _syncopated_pattern(self, root: int,
                             chord_tones: list,
                             beats: int) -> list:
        """シンコペーションパターン"""
        # 16分音符グリッド
        grid = [0] * (beats * 4)
        # シンコペーション配置
        syncopation = [1,0,0,1, 0,0,1,0, 0,1,0,0, 1,0,0,1]

        notes = []
        for i in range(len(syncopation[:beats*4])):
            if syncopation[i]:
                ct = random.choice(chord_tones)
                notes.append({
                    "pitch": ct,
                    "velocity": 85 + random.randint(-10, 10),
                    "duration": 0.25,
                })
            else:
                notes.append({
                    "pitch": 0,
                    "velocity": 0,
                    "duration": 0.25,
                })
        return notes

    def _octave_pattern(self, root: int, beats: int) -> list:
        """オクターブ奏法パターン"""
        notes = []
        for i in range(beats * 2):
            pitch = root if i % 2 == 0 else root + 12
            notes.append({
                "pitch": pitch,
                "velocity": 90 if i % 4 == 0 else 75,
                "duration": 0.5,
            })
        return notes

    def _arpeggiated_pattern(self, root: int,
                              chord_tones: list,
                              beats: int) -> list:
        """アルペジオパターン"""
        notes = []
        for i in range(beats * 2):
            pitch = chord_tones[i % len(chord_tones)]
            notes.append({
                "pitch": pitch,
                "velocity": 80,
                "duration": 0.5,
            })
        return notes
```

---

## 5. AI作曲パイプライン統合

### 5.1 エンドツーエンドの作曲システム

```python
class AICompositionPipeline:
    """AI作曲パイプライン統合クラス"""

    def __init__(self, key="C", scale="major", bpm=120):
        self.key = key
        self.scale = scale
        self.bpm = bpm
        self.chord_gen = ChordProgressionGenerator()
        self.melody_gen = MelodyGenerator(key=key, scale=scale)
        self.bass_gen = BasslineGenerator(key=key)
        self.drum_gen = DrumPatternGenerator()

    def compose(self, bars: int = 16,
                chord_style: str = "ポップ定番",
                melody_style: str = "mixed",
                bass_style: str = "root_notes",
                drum_style: str = "basic_rock") -> dict:
        """楽曲全体を自動生成"""

        # 1. コード進行生成
        chord_progression = self.chord_gen.generate(
            style=chord_style, key=self.key, bars=bars
        )
        chord_names = [c[0] for c in chord_progression]

        # 2. メロディ生成
        melody = self.melody_gen.generate(
            chord_progression=chord_names,
            notes_per_chord=8,
            style=melody_style
        )

        # 3. ベースライン生成
        bassline = self.bass_gen.generate(
            chord_progression=chord_names,
            style=bass_style,
            beats_per_chord=4
        )

        # 4. ドラムパターン生成
        drums = self.drum_gen.generate(
            style=drum_style, bars=bars, humanize=True
        )

        return {
            "chords": chord_progression,
            "melody": melody,
            "bassline": bassline,
            "drums": drums,
            "metadata": {
                "key": self.key,
                "scale": self.scale,
                "bpm": self.bpm,
                "bars": bars,
            }
        }

    def export_midi(self, composition: dict,
                    output_path: str = "composition.mid"):
        """作曲結果をマルチトラックMIDIに出力"""
        from mido import MidiFile, MidiTrack, Message, MetaMessage
        import mido

        mid = MidiFile(ticks_per_beat=480)
        tpb = 480

        # テンポトラック
        tempo_track = MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_track.append(MetaMessage(
            'set_tempo', tempo=mido.bpm2tempo(self.bpm)
        ))
        tempo_track.append(MetaMessage(
            'track_name', name='Tempo', time=0
        ))

        # メロディトラック
        melody_track = MidiTrack()
        mid.tracks.append(melody_track)
        melody_track.append(MetaMessage(
            'track_name', name='Melody', time=0
        ))
        melody_track.append(Message(
            'program_change', program=0, channel=0, time=0
        ))  # Piano

        prev_time = 0
        for note in composition["melody"]:
            start_ticks = int(note.get("start", 0) * tpb)
            duration_ticks = int(note["duration"] * tpb)
            delta = max(0, start_ticks - prev_time)

            melody_track.append(Message(
                'note_on', note=note["pitch"],
                velocity=note["velocity"],
                time=delta, channel=0
            ))
            melody_track.append(Message(
                'note_off', note=note["pitch"],
                velocity=0, time=duration_ticks, channel=0
            ))
            prev_time = start_ticks + duration_ticks

        # ベーストラック
        bass_track = MidiTrack()
        mid.tracks.append(bass_track)
        bass_track.append(MetaMessage(
            'track_name', name='Bass', time=0
        ))
        bass_track.append(Message(
            'program_change', program=33, channel=1, time=0
        ))  # Fingered Bass

        for note in composition["bassline"]:
            if note["pitch"] > 0:
                duration_ticks = int(note["duration"] * tpb)
                bass_track.append(Message(
                    'note_on', note=note["pitch"],
                    velocity=note["velocity"],
                    time=0, channel=1
                ))
                bass_track.append(Message(
                    'note_off', note=note["pitch"],
                    velocity=0, time=duration_ticks, channel=1
                ))
            else:
                rest_ticks = int(note["duration"] * tpb)
                bass_track.append(Message(
                    'note_on', note=60, velocity=0,
                    time=rest_ticks, channel=1
                ))

        mid.save(output_path)
        print(f"MIDIファイル出力: {output_path}")
        return mid


# 使用例
pipeline = AICompositionPipeline(key="C", scale="major", bpm=120)
composition = pipeline.compose(
    bars=16,
    chord_style="カノン進行",
    melody_style="mixed",
    bass_style="walking",
    drum_style="basic_rock"
)
pipeline.export_midi(composition, "my_song.mid")
```

---

## 6. 比較表

### 6.1 AI作曲ツール比較

| 項目 | Magenta | MuseNet | AIVA | Amper/Shutterstock | MusicTransformer | Suno |
|------|---------|---------|------|-------------------|-----------------|------|
| 種別 | OSS | API(終了) | SaaS | SaaS | 研究 | SaaS |
| MIDI出力 | 対応 | 非対応 | 対応 | 限定的 | 対応 | 非対応 |
| ジャンル | 多様 | 多様 | クラシック中心 | 多様 | 多様 | 多様 |
| インタラクティブ | 対応 | 非対応 | 一部 | 非対応 | 非対応 | 一部 |
| カスタマイズ | 高い | 低い | 中程度 | 低い | 高い | 低い |
| 商用利用 | Apache 2.0 | - | 有料プラン | 有料プラン | 研究用 | 有料プラン |
| API提供 | Python | - | REST | REST | - | REST |

### 6.2 コード進行生成手法の比較

| 手法 | 音楽理論知識 | 創造性 | 制御性 | 実装コスト | 学習データ量 |
|------|-------------|--------|--------|-----------|------------|
| ルールベース | 必須 | 低い | 最高 | 低い | 不要 |
| マルコフ連鎖 | 不要（学習） | 中程度 | 中程度 | 低い | 少量 |
| LSTM/GRU | 不要（学習） | 高い | 低い | 中程度 | 中量 |
| Transformer | 不要（学習） | 最高 | 低い | 高い | 大量 |
| ルール+AI混合 | 一部必要 | 高い | 高い | 中程度 | 中量 |
| VAE | 不要（学習） | 高い | 中〜高 | 高い | 大量 |
| GAN | 不要（学習） | 高い | 低い | 最高 | 大量 |
| Diffusion | 不要（学習） | 最高 | 中程度 | 最高 | 大量 |

### 6.3 MIDIトークン化手法の比較

| 手法 | シーケンス長 | 情報保持 | マルチトラック | 実装難易度 | 代表ライブラリ |
|------|------------|---------|-------------|-----------|-------------|
| REMI | 長い | 高い | 限定的 | 低い | MidiTok |
| TSD | 中程度 | 高い | 限定的 | 低い | MidiTok |
| MIDILike | 最長 | 最高 | 対応 | 最低 | MidiTok |
| Structured | 中程度 | 高い | 対応 | 中程度 | MidiTok |
| CPWord | 最短 | 高い | 限定的 | 高い | MidiTok |
| Octuple | 短い | 高い | 対応 | 高い | MidiTok |

---

## 7. アンチパターン

### 7.1 アンチパターン: 音楽理論の完全無視

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

### 7.2 アンチパターン: クオンタイズの機械的適用

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

### 7.3 アンチパターン: 過大なコンテキスト長

```python
# BAD: メモリ不足を招く超長シーケンス
def bad_training():
    model = MusicTransformer(max_seq_len=16384)  # 長すぎる
    # → メモリ不足、学習が極端に遅い
    # → Attention の計算量は O(n^2)

# GOOD: 適切なコンテキスト長と分割戦略
def good_training():
    model = MusicTransformer(max_seq_len=2048)  # 適切な長さ

    # 長い楽曲は重複ありで分割
    def split_with_overlap(tokens, max_len=2048, overlap=256):
        segments = []
        for i in range(0, len(tokens) - max_len, max_len - overlap):
            segments.append(tokens[i:i + max_len])
        return segments
```

### 7.4 アンチパターン: 学習データの偏り

```python
# BAD: 特定ジャンルの学習データのみ
def bad_dataset():
    # クラシック音楽のみ10,000曲
    # → ポップスを生成しようとしても全てクラシック風になる

# GOOD: バランスの取れたデータセット構築
def good_dataset():
    dataset_config = {
        "genres": {
            "pop": 3000,
            "rock": 2000,
            "jazz": 2000,
            "classical": 1500,
            "electronic": 1500,
        },
        "total": 10000,
        "augmentation": {
            "transpose": True,      # 全12キーに移調
            "tempo_variation": True, # テンポを±20%変化
            "velocity_scaling": True, # ベロシティスケール
        },
        "filtering": {
            "min_notes": 50,        # 最低ノート数
            "max_notes": 10000,     # 最大ノート数
            "min_duration": 30,     # 最低30秒
            "max_duration": 600,    # 最大10分
        }
    }
    return dataset_config
```

---

## 8. FAQ

### Q1: AI作曲で生成されたMIDIデータの著作権はどうなりますか？

MIDIデータ自体は著作物として保護される可能性がありますが、AI生成物の著作権帰属は法的にグレーゾーンです。多くのAI作曲サービス（AIVA、Amper等）は有料プランで商用利用権を付与しています。OSSモデル（Magenta等）で生成した場合、学習データの著作権問題が残ります。安全策として、(1) AI生成を出発点に人間が大幅に編集する、(2) 商用利用を明示的に許可するサービスを使う、(3) 生成プロセスを記録しておく、が推奨されます。

### Q2: AIで生成したコード進行をDAWで使うにはどうすればよいですか？

最も簡単な方法は、(1) Python等でMIDIファイルを出力、(2) DAWにMIDIファイルをドラッグ&ドロップ。リアルタイム連携には、(1) 仮想MIDIポート（macOS: IAC Driver、Windows: loopMIDI）経由で送信、(2) Ableton LiveのMax for Live デバイス内でAIモデルを実行、(3) OSCプロトコルでDAWとAIエンジン間を通信。MidiTokやpretty_midiライブラリを使うと、AI出力をDAWフレンドリーなMIDIファイルに変換しやすくなります。

### Q3: メロディ生成AIの品質を向上させるコツは？

効果的な手法は5つあります。(1) スケール制約: 使用可能なノートをスケール内に限定。(2) コンテキスト長の拡大: より長い文脈を考慮するモデル（Transformer）を使用。(3) コード条件付き生成: コード進行に沿ったメロディ生成。(4) 後処理ルール: 連続する同音の制限、跳躍の制限、フレーズ終止の処理。(5) 温度パラメータの調整: 低い値（0.6-0.8）でより「安全な」メロディ、高い値（1.0-1.2）でより実験的なメロディ。

### Q4: MIDIデータの前処理で気をつけるべき点は？

MIDI前処理の重要ポイントは以下の通りです。(1) 分解能の統一: 異なるソースのMIDIファイルは分解能（ticks_per_beat）が異なるため、480に統一するのが標準的。(2) チャンネルの整理: 不要なチャンネル（GM System Exclusive等）を除去。(3) ノートの正規化: ベロシティ範囲の正規化（例: 20-120を0-127に線形マッピング）。(4) 重複ノートの除去: Note Onが連続してNote Offがない異常データの修正。(5) テンポ情報の処理: テンポ変化を含むMIDIは相対時間に変換してから処理する。

### Q5: リアルタイムでAI作曲を使うには遅延をどう解決しますか？

リアルタイム推論の遅延対策は3段階あります。(1) モデルの軽量化: 蒸留モデルやONNXランタイムで推論速度を10倍以上改善可能。(2) バッファリング戦略: 4拍分のバッファを設けて先読み生成。ユーザーが演奏中に次の4拍を生成する。(3) キャッシュとプリコンピュート: よく使うコード進行パターンの結果を事前計算してキャッシュしておく。GPU搭載のPCであれば、512トークンの生成が100ms以下で完了するため、実用的なリアルタイム性が確保できます。

### Q6: 学習データに適したMIDIデータセットはどこで入手できますか？

主要なMIDIデータセットは以下の通りです。(1) Lakh MIDI Dataset: 約17万曲の大規模データセット。ジャンルが多様で研究用途に最適。(2) MAESTRO: Google Magentaが公開するピアノ演奏データセット。高品質なパフォーマンスMIDI。(3) GiantMIDI-Piano: 約1万曲のクラシックピアノ曲。(4) ADL Piano MIDI: 約11,000曲のポップス/ロックのピアノMIDI。(5) MusicNet: アノテーション付きの音楽データセット。ライセンスは各データセットごとに異なるため、商用利用時は必ず確認してください。

---

## まとめ

| 項目 | 要点 |
|------|------|
| MIDIデータ | ノート番号(0-127) + ベロシティ + タイミングの3要素 |
| トークン化 | REMI、Compound Word等でMIDIをLM入力に変換 |
| AI作曲 | Transformer ベースが主流。コンテキスト長が品質に直結 |
| コード進行 | 音楽理論ルール + AI のハイブリッドが実用的 |
| メロディ生成 | コード条件付き+スケール制約で品質確保 |
| ベースライン | スタイル別テンプレート + AI変奏が効率的 |
| ドラムパターン | パターンDB + ヒューマナイズで自然なグルーブ |
| DAW連携 | MIDIファイル出力 or 仮想MIDIポートで接続 |
| 品質向上 | スケール制約 + コード条件付き + 適切な温度設定 |
| 学習パイプライン | データ前処理→トークン化→学習→サンプル生成の自動化 |

## 次に読むべきガイド

- [00-music-generation.md](./00-music-generation.md) — 音楽生成（Suno、MusicGen）
- [02-audio-effects.md](./02-audio-effects.md) — 音声エフェクト
- [../03-development/01-audio-processing.md](../03-development/01-audio-processing.md) — 音声処理ライブラリ

## 参考文献

1. Huang, C.Z.A., et al. (2018). "Music Transformer: Generating Music with Long-Term Structure" — Music Transformer論文。相対位置エンコーディングによる長期構造の生成
2. Fraternali, D., et al. (2023). "MidiTok: A Python package for MIDI file tokenization" — MidiTok論文。MIDI トークン化ライブラリ
3. Roberts, A., et al. (2018). "A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music Generation" — MusicVAE論文。階層的潜在変数による音楽生成
4. Dong, H.W., et al. (2018). "MuseGAN: Multi-track Sequential Generative Adversarial Networks for Symbolic Music Generation and Accompaniment" — マルチトラックGANによる音楽生成
5. Hawthorne, C., et al. (2019). "Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset" — MAESTROデータセットとピアノ音楽生成
6. Lakh MIDI Dataset — https://colinraffel.com/projects/lmd/ — 大規模MIDI研究用データセット
