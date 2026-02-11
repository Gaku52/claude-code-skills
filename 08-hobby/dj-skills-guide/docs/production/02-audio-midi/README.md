# Audio/MIDI基礎

Ableton Liveの心臓部を完全マスター。AudioとMIDIを自在に操り、思い通りの音楽制作を実現します。

## このセクションで学ぶこと

- Audio基礎（波形、サンプルレート、ビット深度）
- MIDI基礎（ノート、ベロシティ、CC）
- Clip編集テクニック
- Warp機能完全理解
- Quantize活用法
- 録音テクニック
- オートメーション

---

## なぜAudio/MIDIが重要なのか

**制作の基礎:**

```
Audio:
録音した音
サンプル
ボーカル、ギター等

MIDI:
音階情報
シンセをコントロール
編集が自由

両方使いこなす:
= プロの制作

できること:

Audio:
ボーカル録音
ドラムサンプル配置
Warpでテンポ合わせ

MIDI:
ベースライン打ち込み
コード進行作成
メロディ作成

組み合わせ:
Audio ドラム
+ MIDI ベース
+ MIDI シンセ
= 完成した曲

統計:

プロの曲:
Audio 40% + MIDI 60%
または
Audio 60% + MIDI 40%

両方必須:
どちらかだけでは不完全
```

---

## 学習の順序

**このセクションの推奨学習パス:**

```
Week 1: Audio基礎
└─ audio-basics.md
   波形、サンプルレート理解

Week 2: MIDI基礎
└─ midi-basics.md
   ノート、ベロシティ理解

Week 3: 編集テクニック
└─ clip-editing.md
└─ warp-modes.md
   Clip編集とWarp

Week 4: 応用
└─ quantization.md
└─ recording-techniques.md
└─ automation.md
   Quantize、録音、オートメーション

合計: 1ヶ月でAudio/MIDI完全習得

次のステップ:
production/03-instruments/
→ 音源を学ぶ
```

---

## 各ファイルの内容

### 1. Audio基礎 (audio-basics.md)

```
内容:
- 波形とは
- サンプルレート（44.1kHz）
- ビット深度（16bit, 24bit）
- Audio Clip の構造
- ファイル形式（WAV, MP3, AIFF）

学習時間: 2-3時間
実践: サンプルをインポート、再生
```

### 2. MIDI基礎 (midi-basics.md)

```
内容:
- MIDIとは何か
- ノート（C3, D#4等）
- ベロシティ（0-127）
- MIDI CC（コントロールチェンジ）
- MIDI Clip の構造
- ピアノロール

学習時間: 2-3時間
実践: MIDIノート打ち込み
```

### 3. Clip編集 (clip-editing.md)

```
内容:
- Clip の基本操作
- カット、コピー、ペースト
- ループ設定
- トランスポーズ
- リバース
- Fade In/Out
- Consolidate

学習時間: 3-4時間
実践: 8小節ループを自由に編集
```

### 4. Warp機能 (warp-modes.md)

```
内容:
- Warpとは（BPM同期）
- Warp Mode 6種類
  - Beats, Tones, Texture, Re-Pitch, Complex, Complex Pro
- Warp Marker配置
- タイミング修正
- ボーカルWarp

学習時間: 3-4時間
実践: 異なるBPMのサンプルを同期
```

### 5. Quantize（クオンタイズ） (quantization.md)

```
内容:
- Quantizeとは（グリッド補正）
- Quantize設定（1/4, 1/8, 1/16）
- Swing（グルーヴ感）
- Groove Pool
- ヒューマナイズ

学習時間: 2-3時間
実践: MIDIノートをQuantize、Swing適用
```

### 6. 録音テクニック (recording-techniques.md)

```
内容:
- Audio録音（ボーカル、ギター）
- MIDI録音（リアルタイム演奏）
- オーバーダビング
- パンチイン/アウト
- ループ録音
- 複数テイク
- Comping（ベストテイク選択）

学習時間: 3-4時間
実践: マイクで録音、MIDIキーボードで演奏録音
```

### 7. オートメーション (automation.md)

```
内容:
- オートメーションとは
- エンベロープ描画
- Record Automation
- オートメーションモード（Touch, Latch, Write）
- MIDI Learn
- パラメーターマッピング
- LFO的な使い方

学習時間: 3-4時間
実践: フィルター開閉、音量フェード、パンニング
```

---

## Audio vs MIDI

**2つの違いを理解:**

```
Audio:
┌────────────────┐
│ 波形 ~~~~/\~~~ │ ← 音そのもの
└────────────────┘

特徴:
- 録音した音
- サンプル
- 編集に制限

メリット:
- リアルな音
- すぐ使える

デメリット:
- 音程変更が難しい
- 編集の自由度低い

用途:
- ドラムサンプル
- ボーカル録音
- ギター録音
- 環境音

MIDI:
┌────────────────┐
│ ■ ■ ■ ■ ■ ■  │ ← 音階情報
└────────────────┘

特徴:
- 音の指示書
- 音源が必要
- 自由に編集

メリット:
- 音程変更自由
- 編集が簡単
- 軽い（KB単位）

デメリット:
- 音源必要
- 打ち込み感

用途:
- ベースライン
- コード進行
- メロディ
- シンセパート

組み合わせ:

Techno トラック:
- Kick: Audio（サンプル）
- Bass: MIDI（シンセ）
- Hi-Hat: Audio（サンプル）
- Synth: MIDI（Wavetable）
- Vocal: Audio（録音）

両方使う:
= 最強
```

---

## このセクションで作れるもの

**1ヶ月後のあなた:**

```
作れるもの:

8小節ループ:
- Audio ドラム（Warp済み）
- MIDI ベース（Quantize済み）
- MIDI シンセ（オートメーション付き）

完成度:
プロと同じレベルの
音質とタイミング

できること:

Audio:
- サンプルをWarp
- 完璧にBPM同期
- タイミング修正

MIDI:
- ピアノロールで打ち込み
- Quantizeで補正
- ベロシティ調整

録音:
- ボーカル録音
- MIDIキーボード録音
- 複数テイクからベスト選択

オートメーション:
- フィルター開閉
- 音量フェード
- エフェクトパラメーター変化

次のステップ:
音源セクションで
さらに深く
```

---

## 必要な機材

**最小構成:**

```
必須:

Ableton Live Standard:
¥49,800

PC/Mac:
既に持っている

推奨:

MIDIキーボード:
AKAI MPK Mini MK3: ¥13,000
→ MIDI入力が楽

オーディオインターフェイス:
Focusrite Scarlett Solo: ¥14,000
→ 録音用

マイク（録音する場合）:
SHURE SM58: ¥12,000
→ ボーカル録音

任意:

DDJ-FLX4:
MIDIコントローラーとして

初期投資:
Ableton + MIDI Keyboard
= 約¥62,800

録音もする:
+ Scarlett Solo + SM58
= 約¥88,800
```

---

## よくある質問

### Q1: AudioとMIDI、どっちが重要？

**A:** 両方必須

```
Audio:
ドラム、ボーカル、効果音
→ 不可欠

MIDI:
ベース、シンセ、メロディ
→ これも不可欠

比率:

Techno: Audio 30% / MIDI 70%
House: Audio 40% / MIDI 60%
Hip Hop: Audio 70% / MIDI 30%

結論:
両方マスター必須
```

### Q2: Warpは必須？

**A:** 必須

```
なぜ:

サンプルのBPM:
元のBPMと違う
→ Warpで合わせる

例:
プロジェクト: 128 BPM
サンプル: 140 BPM
→ Warpで128に

使わないと:
ズレる
使えない

結論:
Warp = Ableton最強機能
必ずマスター
```

### Q3: Quantizeすると機械的？

**A:** 設定次第

```
100% Quantize:
完璧にグリッド
→ 機械的

50% Quantize:
半分だけ補正
→ 人間らしさ残る

Swing:
タイミングをずらす
→ グルーヴ感

推奨:

Kick: 100%
Bass: 100%
Hi-Hat: 75%
Synth: 50%

メリハリ:
重要な音は100%
装飾的な音は50-75%
```

---

## 学習のコツ

**効率的に習得:**

```
実践重視:

理論: 20%
実践: 80%

読むだけではダメ:
必ず手を動かす

毎日触る:

30分でOK:
継続が大事

週3-5日:
1ヶ月で習得

小さく始める:

4小節ループ:
完璧に作る

8小節に拡大:
徐々に複雑化

間違いを恐れない:

失敗OK:
Cmd+Z で戻る

実験:
色々試す

保存:
定期的にCmd+S

コミュニティ:

Discord/Reddit:
分からないことを聞く

フィードバック:
他の人に聴いてもらう
```

---

## チェックリスト

### 準備

```
□ Ableton Live インストール済み
□ MIDIキーボード接続（推奨）
□ オーディオIF接続（録音する場合）
□ 学習時間確保（週5-7時間）
```

### 学習

```
□ audio-basics.md 完了
□ midi-basics.md 完了
□ clip-editing.md 完了
□ warp-modes.md 完了
□ quantization.md 完了
□ recording-techniques.md 完了
□ automation.md 完了
```

### 実践

```
□ 8小節ループ完成（Audio + MIDI）
□ サンプルをWarp
□ MIDIノートをQuantize
□ オートメーション描画
□ Audio/MIDI録音
```

---

## 参考リンク

### 公式

- [Ableton Live Manual - Audio Clips](https://www.ableton.com/en/manual/audio-clips-tempo-and-warping/)
- [Ableton Live Manual - MIDI Clips](https://www.ableton.com/en/manual/working-with-instruments-and-effects/)
- [Ableton Live Manual - Automation](https://www.ableton.com/en/manual/automation-and-editing-envelopes/)

### YouTube

- [Ableton Official Tutorials](https://www.youtube.com/c/AbletonOfficial)
- [You Suck at Producing - Warp](https://www.youtube.com/watch?v=...)
- [Seed to Stage - MIDI Basics](https://www.youtube.com/...)

---

**準備はできましたか？** [Audio基礎](./audio-basics.md) から始めましょう！
