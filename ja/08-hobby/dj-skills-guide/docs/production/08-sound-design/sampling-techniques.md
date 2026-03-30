# Sampling Techniques（サンプリング技術）

Ableton Live 12のSamplerとSimplerを使ったサンプリング技術を完全マスター。ワンショット加工、ボーカルチョップ、テクスチャサウンドデザインで、オリジナリティ溢れる音を作ります。

## この章で学ぶこと

- SamplerとSimplerの違いと使い分け
- ワンショット加工（ドラム、FX）
- ボーカルチョップテクニック
- テクスチャサウンドデザイン
- Loop Point設定テクニック
- Reverse/Stretch技術
- 実践的な練習方法


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [Modulation Techniques（モジュレーション技術）](./modulation-techniques.md) の内容を理解していること

---

## なぜSampling Techniquesが重要なのか

**オリジナリティの源:**

```
シンセのみ:

制約:
波形は限定的
誰でも同じ波形
プリセットに依存

結果:
似た音になる
個性が出にくい
差別化困難

サンプリング追加:

自由:
あらゆる音源
レコード、フィールド録音、声
無限の可能性

結果:
完全オリジナル
強い個性
差別化容易

プロの使用率:

Hip Hop: 90%サンプリング
House: 50%サンプリング
Techno: 30%サンプリング
IDM: 70%サンプリング

理由:
オリジナリティ
個性
ビンテージ感
```

**サンプリングの独自性:**

```
できること:

1. ワンショット加工:
   - ドラムサンプル加工
   - 完全に別の音に変換

2. ボーカルチョップ:
   - ボーカルをスライス
   - リズム楽器化

3. テクスチャ:
   - 環境音、ノイズ
   - Ambient、Pad化

4. ビンテージ感:
   - レコードサンプル
   - Lo-Fi効果

シンセでは不可能:
人間の声
レコードのノイズ
自然の音
偶然の音
```

---

## SamplerとSimplerの違い

### 基本比較

```
Simpler:

特徴:
シンプル
1サンプル = 1インスタンス
ワンショット、シンプルなループ

用途:
ドラム
簡単なサンプル再生
ワンショット加工

CPU: 軽い

使用頻度: 60%

Sampler:

特徴:
複雑
マルチサンプル対応
Zone設定、複雑なループ

用途:
楽器作成
複数サンプル
ボーカルチョップ

CPU: 重い

使用頻度: 40%
```

### 使い分け

```
Simplerを使う場合:

- 1つのサンプルのみ
- ドラムワンショット
- 簡単な加工
- CPU節約

例:
キックサンプル加工
スネアサンプル Pitch Down
FX サンプル Reverse

Samplerを使う場合:

- 複数サンプル
- ボーカルチョップ
- 楽器作成
- 複雑なMapping

例:
ボーカル8スライス → 8鍵盤
ドラムキット（Kick、Snare、HH...）
マルチサンプル楽器
```

---

## Simplerインターフェース完全理解

### セクション構成

```
上部:
┌─────────────────────────────────────┐
│  Sample Display                     │
│  [波形表示、Start/End/Loop設定]     │
└─────────────────────────────────────┘

中央:
┌─────────────────────────────────────┐
│  Filter    Amp    Pitch    LFO     │
└─────────────────────────────────────┘

下部:
┌─────────────────────────────────────┐
│  Effects  (Saturator, Filter等)    │
└─────────────────────────────────────┘
```

### 再生モード

```
Classic Mode:

特徴:
サンプルを通常再生
ピッチシフト
ループ対応

用途:
楽器音
ワンショット
一般的なサンプル

1-Shot Mode:

特徴:
最後まで再生
ループなし
鍵盤を離してもStop しない

用途:
ドラムワンショット
FX

Slice Mode:

特徴:
サンプルを自動スライス
各スライスを鍵盤に配置

用途:
ドラムループ
ボーカルチョップ
```

### 主要パラメータ

```
Sample Tab:

Start: サンプル開始位置
End: サンプル終了位置
Loop: ループのOn/Off
Fade: フェードイン/アウト
Snap: グリッドにスナップ

Filter Tab:

Type: LP、HP、BP、Notch
Cutoff: 20 Hz - 20 kHz
Resonance: 0-100%
Envelope: Filter Envelope Amount

Pitch/Osc Tab:

Transpose: -48 〜 +48 semitones
Detune: -50 〜 +50 cent
Spread: ステレオ拡張

LFO Tab:

Waveform: Sine、Triangle、Square等
Rate: 0.01-40 Hz、Sync可能
Destination: Pitch、Filter、Volume、Pan
```

---

## 実践1: ワンショット加工（ドラムサンプル）

**目標:** 既存のキックサンプルを完全に別の音に変える

### Step 1: サンプル準備（5分）

```
サンプル入手:

方法1: Abletonライブラリ
Browser → Drums → Kicks
任意のKickサンプルを選択

方法2: Splice等
外部サンプルパック
Kickサンプルをダウンロード

方法3: 自分で録音
キックドラムを録音
WAV形式で保存

推奨:
まずはAbletonライブラリから
```

### Step 2: Simpler追加（3分）

```
手順:

1. 新規MIDIトラック作成
2. Simplerをドラッグ&ドロップ
3. Kickサンプルを Simpler にドラッグ
4. 波形が表示される

初期設定:
Mode: Classic
ピッチ: C3（標準）
```

### Step 3: Pitch加工（10分）

```
Transpose:

元の音: C3（標準）

-12 semitones（1オクターブ下）:
→ 低く、太いキック
用途: Sub Kick、Dubstep

-24 semitones（2オクターブ下）:
→ 超低域、轟音
用途: 808 Kick風

+12 semitones（1オクターブ上）:
→ 高く、パンチのあるキック
用途: Techno、Hardstyle

推奨:
-12 semitones（低域補強）

Detune:

0 cent: クリーン
+20 cent: わずかに高く、独特
-30 cent: わずかに低く、太い

推奨:
-10 cent（微妙に太く）
```

### Step 4: Filter加工（10分）

```
Low Pass Filter:

Cutoff: 5000 Hz → 300 Hz
→ 高域削除、超低域キック

Resonance: 30%
→ Cutoff周波数強調

Filter Envelope:
A: 0 ms
D: 50 ms
S: 0%
R: 0 ms
Amount: +30
→ アタック時にFilter開く

High Pass Filter:

Cutoff: 100 Hz → 500 Hz
→ 低域削除、クリック音のみ

用途:
サイドチェイン用キック
Technoのクリック音
```

### Step 5: Amp Envelope加工（8分）

```
Amp Envelope:

短いキック:
A: 0 ms
D: 30 ms
S: 0%
R: 0 ms
→ タイトなキック

長いキック:
A: 0 ms
D: 200 ms
S: 0%
R: 100 ms
→ 808風ロングキック

パンチのあるキック:
A: 0 ms
D: 80 ms
S: 0%
R: 20 ms

推奨:
Decay 50-100 ms
```

### Step 6: エフェクト追加（12分）

```
エフェクトチェイン:

1. Saturator:
   Drive: 6 dB
   Curve: Warm
   理由: 倍音追加、太い音

2. EQ Eight:
   High Pass: 30 Hz
   Bell: 60 Hz、+4 dB、Q 1.0
   Bell: 100 Hz、-2 dB、Q 0.8
   理由: Sub強調、濁り削除

3. Compressor:
   Ratio: 8:1
   Threshold: -10 dB
   Attack: 0 ms
   Release: 50 ms
   Gain: +3 dB
   理由: パンチ強化

4. Erosion（オプション）:
   Mode: Wide Noise
   Frequency: 5 kHz
   Amount: 10%
   理由: 高域ノイズ、Lo-Fi感

5. Utility:
   Gain: 調整
   Width: 0%（Mono）
   理由: キックはMono必須
```

### Step 7: 複数バージョン作成（12分）

```
バージョン1: Sub Kick
Transpose: -12 semitones
Filter: LP 200 Hz
Decay: 150 ms

バージョン2: Click Kick
Transpose: 0 semitones
Filter: HP 500 Hz
Decay: 20 ms

バージョン3: 808 Kick
Transpose: -18 semitones
Filter: LP 300 Hz
Decay: 300 ms
Pitch Envelope: -12 semitones、50 ms

バージョン4: Lo-Fi Kick
Transpose: -5 semitones
Erosion: 20%
Redux: Bit Depth 8 bit
```

### 完成基準

```
□ 元のサンプルと完全に別の音
□ Transpose、Filter、Envelope調整
□ 4種類以上のバージョン作成
□ 用途明確（Sub、Click等）
□ 保存してライブラリ化

所要時間: 60分
```

---

## 実践2: ボーカルチョップテクニック

**目標:** ボーカルサンプルをスライスしてリズム楽器にする

### Step 1: ボーカルサンプル準備（5分）

```
サンプル入手:

方法1: Spliceボーカルパック
Vocal Phrase（2-4小節）
Dry（Reverb なし）推奨

方法2: Looperman
無料ボーカルサンプル

方法3: 自分で録音
1フレーズ歌う
2-4小節

推奨:
Splice（高品質、商用利用可能）
```

### Step 2: Simplerに読み込み（5分）

```
手順:

1. 新規MIDIトラック作成
2. ボーカルサンプルを Simpler にドラッグ
3. Mode: Slice に変更

Slice設定:

Slice By: Transient
→ アタックを自動検出

Sensitivity: 50%
→ スライス数調整

Playback Mode: Mono
→ 前の音を停止

確認:
8-16スライスが理想
```

### Step 3: スライス調整（10分）

```
スライス編集:

1. 波形表示でスライスマーカー確認
2. 不要なスライス削除
3. 手動でスライス追加

マーカー移動:
ドラッグして位置調整
アタック直前に配置

スライス結合:
不要なマーカー削除
→ 2つのスライスが1つに

目標:
8スライス（C3-G3に配置）
各スライスが明瞭
```

### Step 4: 鍵盤Mapping（8分）

```
Mapping:

C3: スライス1（最初の音節）
D3: スライス2
E3: スライス3
F3: スライス4
G3: スライス5
A3: スライス6
B3: スライス7
C4: スライス8（最後の音節）

確認:
各鍵盤で正しいスライス再生
アタックが揃っている
```

### Step 5: Envelope調整（10分）

```
Amp Envelope:

A: 0 ms
D: 100 ms
S: 0%
R: 50 ms

理由:
短いDecay → パーカッシブ
Release 50 ms → 自然な余韻

Filter Envelope:

Amount: +20
A: 0 ms
D: 150 ms
S: 0%
R: 50 ms

理由:
アタック時にFilter開く
→ 明瞭な子音
```

### Step 6: エフェクト追加（15分）

```
エフェクトチェイン:

1. EQ Eight:
   High Pass: 200 Hz
   Peak: 3000 Hz、+2 dB、Q 1.5
   理由: 低域削除、存在感

2. Compressor:
   Ratio: 4:1
   Threshold: -15 dB
   Attack: 5 ms
   Release: 80 ms
   Gain: +4 dB
   理由: レベル均一化

3. Saturator:
   Drive: 3 dB
   Curve: Soft Curve
   理由: 温かみ

4. Reverb:
   Type: Small Room
   Decay: 0.8s
   Dry/Wet: 15%
   理由: 自然な空間感

5. Delay:
   Time: 1/16
   Feedback: 20%
   Dry/Wet: 10%
   理由: 微妙な反復

6. Utility:
   Width: 50%
   理由: ややMono、中央定位
```

### Step 7: MIDIパターン作成（15分）

```
パターン例（16小節）:

Bar 1-2:
C3 ── ── E3 ── ── ── G3 ──
リズムパターン

Bar 3-4:
D3 ── F3 ── ── E3 ── C3 ──
変化

Bar 5-6:
C3 E3 ── G3 ── F3 ── ── ──
密集

Bar 7-8:
C3 ── ── ── E3 ── ── ── ──
スパース

Bar 9-16: 繰り返しとバリエーション

ベロシティ: 80-120（変化をつける）
Length: 1/16（短く）

確認:
□ リズムが気持ち良い
□ グルーヴがある
□ ボーカル感が残っている
```

### 完成基準

```
□ 8スライス作成
□ 鍵盤にMapping
□ Envelope調整
□ エフェクトチェイン構築
□ MIDIパターン作成
□ リズム楽器として機能

所要時間: 70分
```

---

## 実践3: テクスチャサウンドデザイン

**目標:** 環境音をAmbient Pad、テクスチャサウンドに変換

### Step 1: 環境音サンプル準備（10分）

```
サンプル入手:

方法1: フィールド録音
スマホで録音
雨、風、街の音、海の音
30秒-1分

方法2: Freesound.org
無料環境音サンプル
CC0ライセンス

方法3: Abletonライブラリ
Browser → Samples → Textures

推奨サンプル:
雨音
風音
ビニール袋の音
ラジオノイズ
```

### Step 2: Samplerに読み込み（5分）

```
Simplerではなく Sampler を使用

理由:
複雑なLoop Point設定
より高度な加工

手順:
1. 新規MIDIトラック
2. Samplerをドラッグ
3. 環境音サンプルを Zone Editorにドラッグ
```

### Step 3: Loop Point設定（15分）

```
Loop Mode:

Forward Loop:
サンプルを繰り返し再生

Ping Pong Loop:
前後に再生、滑らか

Loop設定:

Start: サンプル中の滑らかな部分
End: Startから1-2秒後
Crossfade: 200 ms

コツ:
波形のゼロクロス点を選ぶ
→ クリックノイズ防止

Fade設定:
In: 50 ms
Out: 50 ms
→ 滑らかな開始/終了
```

### Step 4: Pitch/Time処理（12分）

```
Pitch Down:

Transpose: -24 semitones（2オクターブ下）
→ 非常に遅い、深い音

Detune: -10 cent
→ さらに低く

Time Stretch:

Sample Tab → Stretch
Mode: Texture
Ratio: 50%（半分の速度）

理由:
環境音が Drone化
Ambient感

Grain設定:

Spray: 30%
Flux: 40%
→ ランダムな粒子感
```

### Step 5: Filter深掘り（12分）

```
Filter:

Type: Low Pass 12 dB
Cutoff: 800 Hz
Resonance: 5%

Filter Envelope:
A: 2000 ms
D: 3000 ms
S: 60%
R: 4000 ms
Amount: +15

理由:
ゆっくりFilter開く
→ Ambient特有の立ち上がり

Filter LFO:

Destination: Cutoff
Waveform: Sine
Rate: 0.1 Hz（ゆっくり）
Depth: 20%

理由:
音色がゆっくり変化
→ 動きのあるテクスチャ
```

### Step 6: Modulation設定（15分）

```
LFO 1（Volume変化）:

Destination: Amp Volume
Waveform: Triangle
Rate: 0.08 Hz
Depth: 12%

理由:
呼吸感、波のような変化

LFO 2（Pan変化）:

Destination: Pan
Waveform: Sine
Rate: 0.15 Hz
Depth: 60%

理由:
音が左右に動く
→ ステレオ感

LFO 3（Pitch変化）:

Destination: Pitch
Waveform: Random S&H
Rate: 0.05 Hz
Depth: 8%

理由:
微妙なピッチ揺らぎ
→ 有機的な感じ
```

### Step 7: エフェクトチェイン（20分）

```
エフェクトチェイン:

1. EQ Eight:
   High Pass: 250 Hz
   Low Pass: 6000 Hz
   理由: 中域のみ、濁り削除

2. Grain Delay:
   Spray: 50%
   Frequency: 2 kHz
   Time: 1/4
   Dry/Wet: 35%
   理由: 粒子感、複雑な反復

3. Reverb:
   Type: Large Hall
   Decay: 8s（非常に長い）
   Damping: 2000 Hz
   Dry/Wet: 55%
   理由: 大きな空間感

4. Phaser:
   Rate: 0.12 Hz
   Amount: 40%
   Dry/Wet: 25%
   理由: 位相変化、動き

5. Chorus:
   Rate: 0.3 Hz
   Amount: 35%
   Dry/Wet: 30%
   理由: 広がり

6. Auto Filter:
   Cutoff: LFO制御
   Resonance: 8%
   理由: さらなる音色変化

7. Erosion:
   Mode: Sine
   Frequency: 8 kHz
   Amount: 5%
   理由: 微妙なノイズ

8. EQ Eight（2個目）:
   Peak: 400 Hz、+1.5 dB、Q 0.5
   Peak: 2500 Hz、+1 dB、Q 0.8
   理由: 存在感

9. Utility:
   Width: 180%
   Gain: -4 dB
   理由: 大きなステレオ拡張
```

### Step 8: 演奏（10分）

```
演奏方法:

コード:
Am7（A2、C3、E3、G3）
長く伸ばす（8-16拍）

Velocity: 70-90（弱め）

確認:
□ ゆっくり立ち上がる
□ 広がりがある
□ 動きがある
□ Ambient感
□ テクスチャとして機能

用途:
Ambient楽曲
Intro/Outro
空間を埋める
```

### 完成基準

```
□ 環境音がPad/Textureに変換
□ Loop Point設定
□ Time Stretch 50%
□ LFO 3個設定
□ Reverb 8s以上
□ Stereo Width 150%以上
□ Ambient感

所要時間: 100分
```

---

## Reverse（逆再生）テクニック

### 基本設定

```
Simpler/Sampler:

Reverse Button: On
→ サンプルが逆再生される

用途:

1. Reverse Cymbal:
   シンバルを逆再生
   → ビルドアップ効果

2. Reverse Vocal:
   ボーカルを逆再生
   → 幽玄な効果

3. Reverse Reverb:
   音を録音 → Reverb追加 → Reverse
   → リバースリバーブ
```

### 実践例

```
Reverse Crash（クラッシュシンバル）:

1. Crashサンプルを Simpler に読み込み
2. Reverse: On
3. Amp Envelope:
   A: 0 ms
   D: 1000 ms
   S: 0%
   R: 0 ms
4. 用途: ビルドアップ、Riser

Reverse Snare:

1. Snareサンプルを Simpler に読み込み
2. Reverse: On
3. Transpose: +5 semitones
4. Filter: HP 500 Hz
5. 用途: Fillパターン

Reverse Pad:

1. Padサンプル（長い音）を読み込み
2. Reverse: On
3. Reverb: Decay 5s、Dry/Wet 60%
4. 用途: Ambient、Intro
```

---

## Stretch（時間伸縮）テクニック

### Warp Mode種類

```
Beats Mode:

用途: ドラムループ
特徴: Transientを保持
推奨: BPM変更時

Tones Mode:

用途: メロディ、ハーモニー
特徴: ピッチ保持、高品質
推奨: ボーカル、楽器

Texture Mode:

用途: Ambient、テクスチャ
特徴: Grain効果
推奨: Pad、環境音

Re-Pitch Mode:

用途: ビンテージ効果
特徴: ピッチとテンポ連動
推奨: Lo-Fi、テープ感

Complex/Complex Pro Mode:

用途: 高品質
特徴: 最高品質、CPU重い
推奨: マスタリング、重要な音
```

### 実践例

```
Slow Down（テンポ遅く）:

1. ドラムループを Audio Track に配置
2. Warp: On
3. BPM: 140 → 70（半分）
4. Warp Mode: Beats
5. 用途: Lo-Fi、Chill Hop

Speed Up（テンポ速く）:

1. ボーカルサンプルを配置
2. Warp: On
3. BPM: 120 → 160
4. Warp Mode: Complex
5. 用途: Uptempo、Jungle

Extreme Stretch:

1. 短いサンプル（1秒）を配置
2. Warp: On
3. 10秒に引き伸ばす
4. Warp Mode: Texture
5. Grain設定調整
6. 用途: Drone、Ambient
```

---

## よくある質問（FAQ）

**Q1: SimplerとSampler、どちらを使うべきですか？**

```
A: 用途で選ぶ

Simpler:
使用ケース:
- ドラムワンショット 90%
- 簡単なサンプル再生 80%
- ボーカルチョップ（8スライス以下）60%

Sampler:
使用ケース:
- 複雑なボーカルチョップ 80%
- 楽器作成（マルチサンプル）100%
- 高度なLoop Point設定 90%

推奨:
まずSimpler（80%のケース）
必要ならSampler（20%のケース）
```

**Q2: ボーカルサンプルの著作権は大丈夫ですか？**

```
A: ライセンス確認必須

安全な方法:

1. Splice、Loopmasters:
   商用利用可能
   サブスクリプション

2. Freesound.org:
   CC0、CC BYライセンス
   帰属表示（CC BY）

3. 自分で録音:
   完全に自由
   オリジナリティ最高

4. ボーカルパック:
   Royalty-Free表記確認

危険な方法:
YouTubeから抽出 → NG
Spotifyから録音 → NG
CDから抽出 → NG（サンプリング処理必要）
```

**Q3: サンプルが濁って聞こえます**

```
A: EQ、Filter で整理

手順:

1. EQ Eight:
   High Pass: 不要な低域削除
   例: ボーカル → 200 Hz
       ドラム → 30 Hz

2. Filter:
   Cutoff調整
   高域削りすぎ → Cutoff上げる

3. Saturation削減:
   Drive下げる
   倍音過多が濁りの原因

4. Reverb削減:
   Dry/Wet下げる
   空間過多が濁りの原因

5. スペクトラム確認:
   EQ Eightで視覚的確認
   濁りの周波数を特定
```

**Q4: Loop Pointでクリックノイズが出ます**

```
A: ゼロクロス点を選ぶ

手順:

1. 波形拡大:
   Loop Start/End を拡大表示

2. ゼロクロス点探し:
   波形が中央線を横切る点
   ここでLoopするとクリックなし

3. Crossfade設定:
   Crossfade: 100-300 ms
   滑らかな接続

4. Fade設定:
   Loop Fade: On
   自動フェード

5. サンプル選択:
   滑らかなループに向くサンプル選び
```

**Q5: Reverseサンプルのタイミングが合いません**

```
A: 逆算して配置

手順:

1. サンプル長確認:
   例: 2秒

2. 配置位置:
   目標: Bar 4の頭で終わる
   → Bar 3の3拍目から開始

3. 計算:
   サンプル長 = 2秒
   BPM 120 = 1拍 0.5秒
   2秒 = 4拍前から開始

4. Nudge:
   微調整で完璧に合わせる

5. Fade Out:
   最後にFade Out追加
   → 自然な終わり
```

**Q6: サンプルをさらに加工するコツは？**

```
A: レイヤリング + 予想外のエフェクト

コツ:

1. レイヤリング:
   元のサンプル
   + Pitch Down版
   + Reverse版
   → 3層で厚み

2. 予想外のエフェクト:
   Erosion、Redux
   → Lo-Fi、デジタル感

3. Grain Delay:
   粒子感
   → 独特の質感

4. Frequency Shifter:
   ピッチシフト（非ハーモニック）
   → 金属音、ロボット声

5. Vocoder:
   他の音源でモジュレート
   → 完全に別の音

6. 極端な設定:
   Reverb 100%
   Delay Feedback 90%
   → 偶然の発見
```

---

## 練習方法

### Week 1: ワンショット加工マスター

```
Day 1-2: Kick加工
1. 5種類のKickサンプル
2. 各サンプルを4バージョン作成
3. Sub、Click、808、Lo-Fi

Day 3-4: Snare加工
1. 5種類のSnareサンプル
2. Transpose、Filter、Envelope調整
3. 20個のSnareバリエーション作成

Day 5-6: FX加工
1. Crash、Riseサンプル
2. Reverse、Stretch
3. ビルドアップ用FX作成

Day 7: ドラムキット作成
1. 加工したサンプルでキット作成
2. Drum Rackに配置
3. オリジナルドラムキット完成

目標:
ワンショット加工を20分で完了
```

### Week 2: ボーカルチョップマスター

```
Day 1-2: 基本チョップ
1. 3種類のボーカルサンプル
2. 8スライスに分割
3. MIDIパターン作成

Day 3-4: 複雑なパターン
1. 16スライスに分割
2. 複雑なMIDIパターン
3. ベロシティ変化

Day 5-6: エフェクト実験
1. Reverb、Delay調整
2. Grain Delay追加
3. 5種類のエフェクトチェイン

Day 7: 楽曲制作
1. ボーカルチョップをメインに
2. House、Hip Hopトラック作成
3. 完成

目標:
ボーカルチョップを40分で完了
```

### Week 3: テクスチャサウンドマスター

```
Day 1-2: 環境音録音
1. フィールド録音（雨、風、街）
2. 10種類の環境音収集
3. サンプルライブラリ作成

Day 3-4: Pad化
1. 環境音をSamplerに読み込み
2. Loop Point設定
3. Ambient Pad作成

Day 5-6: LFO深掘り
1. LFO 3個設定
2. Volume、Pan、Pitch変化
3. 動きのあるテクスチャ

Day 7: Ambient楽曲制作
1. テクスチャサウンドをメインに
2. 3-5個レイヤリング
3. Ambient曲完成

目標:
テクスチャサウンドを60分で完了
```

### Week 4: 総合練習

```
Day 1-2: Reverse技術
1. 10種類のサンプルReverse
2. Riser、Crash作成
3. ビルドアップパターン

Day 3-4: Stretch技術
1. Extreme Stretch（10倍）
2. Drone作成
3. Ambient Base

Day 5-6: レイヤリング
1. ワンショット3層
2. ボーカルチョップ + Pad
3. 複雑な音色作成

Day 7: オリジナル楽曲制作
1. サンプリングのみで楽曲制作
2. シンセなし
3. 完成、公開

目標:
サンプリング技術を自在に使える
```


---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

```
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

```python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```

---

## 実務での適用シナリオ

### シナリオ1: スタートアップでのMVP開発

**状況:** 限られたリソースで素早くプロダクトをリリースする必要がある

**アプローチ:**
- シンプルなアーキテクチャを選択
- 必要最小限の機能に集中
- 自動テストはクリティカルパスのみ
- モニタリングは早期から導入

**学んだ教訓:**
- 完璧を求めすぎない（YAGNI原則）
- ユーザーフィードバックを早期に取得
- 技術的負債は意識的に管理する

### シナリオ2: レガシーシステムのモダナイゼーション

**状況:** 10年以上運用されているシステムを段階的に刷新する

**アプローチ:**
- Strangler Fig パターンで段階的に移行
- 既存のテストがない場合はCharacterization Testを先に作成
- APIゲートウェイで新旧システムを共存
- データ移行は段階的に実施

| フェーズ | 作業内容 | 期間目安 | リスク |
|---------|---------|---------|--------|
| 1. 調査 | 現状分析、依存関係の把握 | 2-4週間 | 低 |
| 2. 基盤 | CI/CD構築、テスト環境 | 4-6週間 | 低 |
| 3. 移行開始 | 周辺機能から順次移行 | 3-6ヶ月 | 中 |
| 4. コア移行 | 中核機能の移行 | 6-12ヶ月 | 高 |
| 5. 完了 | 旧システム廃止 | 2-4週間 | 中 |

### シナリオ3: 大規模チームでの開発

**状況:** 50人以上のエンジニアが同一プロダクトを開発する

**アプローチ:**
- ドメイン駆動設計で境界を明確化
- チームごとにオーナーシップを設定
- 共通ライブラリはInner Source方式で管理
- APIファーストで設計し、チーム間の依存を最小化

```python
# チーム間のAPI契約定義
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class APIContract:
    """チーム間のAPI契約"""
    endpoint: str
    method: str
    owner_team: str
    consumers: List[str]
    sla_ms: int  # レスポンスタイムSLA
    priority: Priority

    def validate_sla(self, actual_ms: int) -> bool:
        """SLA準拠の確認"""
        return actual_ms <= self.sla_ms

    def to_openapi(self) -> dict:
        """OpenAPI形式で出力"""
        return {
            'path': self.endpoint,
            'method': self.method,
            'x-owner': self.owner_team,
            'x-consumers': self.consumers,
            'x-sla-ms': self.sla_ms
        }

# 使用例
contracts = [
    APIContract(
        endpoint="/api/v1/users",
        method="GET",
        owner_team="user-team",
        consumers=["order-team", "notification-team"],
        sla_ms=200,
        priority=Priority.HIGH
    ),
    APIContract(
        endpoint="/api/v1/orders",
        method="POST",
        owner_team="order-team",
        consumers=["payment-team", "inventory-team"],
        sla_ms=500,
        priority=Priority.CRITICAL
    )
]
```

### シナリオ4: パフォーマンスクリティカルなシステム

**状況:** ミリ秒単位のレスポンスが求められるシステム

**最適化ポイント:**
1. キャッシュ戦略（L1: インメモリ、L2: Redis、L3: CDN）
2. 非同期処理の活用
3. コネクションプーリング
4. クエリ最適化とインデックス設計

| 最適化手法 | 効果 | 実装コスト | 適用場面 |
|-----------|------|-----------|---------|
| インメモリキャッシュ | 高 | 低 | 頻繁にアクセスされるデータ |
| CDN | 高 | 低 | 静的コンテンツ |
| 非同期処理 | 中 | 中 | I/O待ちが多い処理 |
| DB最適化 | 高 | 高 | クエリが遅い場合 |
| コード最適化 | 低-中 | 高 | CPU律速の場合 |
---

## まとめ

サンプリング技術は、オリジナリティの源です。

**重要ポイント:**

1. **Simpler**: ワンショット、簡単な加工（80%）
2. **Sampler**: ボーカルチョップ、複雑な加工（20%）
3. **ワンショット**: Pitch、Filter、Envelope で別の音に
4. **ボーカルチョップ**: 8スライス、リズム楽器化
5. **テクスチャ**: 環境音をPad化、Ambient
6. **Reverse**: ビルドアップ、Riser
7. **Stretch**: テンポ変更、Drone

**学習順序:**
1. ワンショット加工（Week 1）
2. ボーカルチョップ（Week 2）
3. テクスチャサウンド（Week 3）
4. Reverse/Stretch（Week 4）

**オリジナリティ:**
- 自分で録音（最高）
- ライセンス確認（必須）
- 50%以上加工（オリジナル化）

**次のステップ:** [Layering（レイヤリング）](./layering.md) へ進む

---

## 関連ファイル

- **[Wavetable Sound Design](./wavetable-sound-design.md)** - Wavetable活用
- **[Layering](./layering.md)** - レイヤリング技術
- **[Genre Sounds](./genre-sounds.md)** - ジャンル別サウンド
- **[03-instruments/simpler-sampler.md](../03-instruments/)** - Simpler/Sampler詳細

---

**サンプリングで、あなただけの音を作りましょう！**

---

## サンプリングの法的側面と権利処理

### 著作権の基本構造

```
音楽著作権の二重構造:

1. 著作権（楽曲そのもの）:
   - 作曲者・作詞者が保有
   - メロディ、歌詞、ハーモニーが対象
   - JASRACやNexToneが管理（日本の場合）
   - 保護期間: 著作者の死後70年

2. 著作隣接権（原盤権）:
   - レコード会社・実演家が保有
   - 録音された音源そのものが対象
   - マスター音源の使用許諾が必要
   - 保護期間: 発行後70年

サンプリングで必要な許諾:

ケース1: レコードからサンプリング
→ 著作権 + 原盤権 の両方が必要

ケース2: 自分で演奏して録音
→ 著作権のみ必要（原盤権は自分）

ケース3: パブリックドメイン楽曲を自分で録音
→ どちらも不要（完全自由）

ケース4: ロイヤリティフリーサンプルパック
→ ライセンス条件に従えばOK
```

### クリアランス（権利処理）の手順

```
Step 1: 権利者の特定

楽曲の権利者:
- JASRACデータベース検索（日本曲）
  → https://www.jasrac.or.jp/
- ASCAP / BMI（アメリカ曲）
- PRS（イギリス曲）

原盤の権利者:
- レコード会社（メジャー）
- インディーレーベル
- アーティスト本人（自主制作の場合）

Step 2: 使用許諾申請

申請に含める情報:
- サンプリング元の楽曲名・アーティスト名
- 使用する部分（何秒〜何秒）
- 使用方法（ループ、ワンショット等）
- リリース予定の楽曲情報
- 配信プラットフォーム
- 商用 / 非商用

Step 3: 交渉

一般的な条件:
- ロイヤリティ（売上の数%）
- 一時金（固定額）
- クレジット表記
- サンプリング元のライター登録

相場（目安）:
- インディー: $500〜$5,000
- メジャー: $5,000〜$50,000+
- 有名曲: $50,000〜$500,000+

Step 4: 契約締結

確認事項:
- 使用範囲（全世界 / 地域限定）
- 使用期間（永久 / 期間限定）
- 独占 / 非独占
- 改変の可否
- サブライセンスの可否
```

### 安全なサンプリングソースの選び方

```
リスクレベル別分類:

【リスクなし（推奨）】

1. 自分で録音した音:
   - フィールドレコーディング
   - 自分の声・楽器演奏
   - 環境音・生活音
   → 完全に自由

2. パブリックドメイン:
   - 著作権が切れた楽曲
   - クラシック音楽（録音は別）
   - 1953年以前の録音（アメリカ）
   → 録音自体の権利に注意

3. CC0ライセンス素材:
   - Freesound.org（CC0フィルター）
   - Wikimedia Commons
   → 帰属表示も不要

【リスク低（条件付きOK）】

4. ロイヤリティフリーサンプルパック:
   - Splice
   - Loopmasters
   - KSHMR Sample Pack
   → ライセンス条件を確認

5. Creative Commons（CC BY等）:
   - Freesound.org（CC BY）
   - クレジット表記必須
   → 商用利用可否を確認

【リスク高（許諾必要）】

6. 既存の楽曲:
   - レコード、CD、配信音源
   - 必ずクリアランス必要
   → 無断使用は著作権侵害

7. 映画・TV・ゲームの音声:
   - 複数の権利者が存在
   - クリアランスが複雑
   → 専門家に相談推奨
```

### 「認識できないほど加工」の神話

```
よくある誤解:

「元の音がわからないほど加工すれば大丈夫」
→ 法的には必ずしも安全ではない

判例:

1. Bridgeport Music v. Dimension Films (2005):
   - 2秒のギターリフを Pitch Down して使用
   - 裁判所: サンプリングは許諾が必要
   - 「Get a license or do not sample」

2. VMG Salsoul v. Ciccone (2016):
   - 0.23秒のホーンサンプル
   - 裁判所: 一般リスナーが識別できないなら合法
   - 判断が分かれるケース

実務的な指針:

安全な加工レベル:
- 元の音が完全に認識不能
- メロディ・リズムパターンが残っていない
- 音色のみを参考にした再制作

危険な加工レベル:
- メロディが認識できる
- 特徴的なリズムパターンが残る
- 有名なフレーズの断片

最善の対策:
- ロイヤリティフリー素材を使用
- 自分で録音・演奏
- 必要ならクリアランス取得
```

---

## フィールドレコーディング実践

### 機材と準備

```
録音機材レベル別:

【初心者向け（予算0〜1万円）】

スマートフォン:
- iPhone / Android内蔵マイク
- 録音アプリ: Voice Memos、AudioShare
- 品質: 十分にサンプリング素材として使用可能
- コツ: 風防として手で覆う

【中級者向け（予算1〜5万円）】

ポータブルレコーダー:
- ZOOM H1n / H4n Pro
- TASCAM DR-05X / DR-40X
- 品質: 24bit/96kHz対応
- メリット: XY ステレオマイク内蔵

【上級者向け（予算5万円〜）】

専用マイク + レコーダー:
- DPA 4060（ラベリア）
- Sennheiser MKH 8040（指向性）
- Sound Devices MixPre-3 II
- 品質: プロフェッショナル

録音フォーマット:
- WAV 24bit/48kHz（推奨最低限）
- WAV 24bit/96kHz（推奨）
- MP3は避ける（非可逆圧縮）
```

### 録音テクニックとスポット

```
録音場所と素材の種類:

【自然環境】

雨音:
- 場所: 屋根の下、窓際
- コツ: 異なる強度で複数テイク
- 用途: Ambient Pad、ノイズレイヤー
- 録音時間: 2-5分

波の音:
- 場所: 海岸、湖畔
- コツ: 風防必須、波打ち際から3-5m
- 用途: Ambient、Riser、Whoosh
- 録音時間: 5-10分

風音:
- 場所: 高台、開けた場所
- コツ: 風防なしだとクリップする
- 用途: ノイズ、Sweep、テクスチャ
- 録音時間: 3-5分

虫の声:
- 場所: 公園、森（夜間）
- コツ: 静かな時間帯に
- 用途: Ambient、リズムパターン
- 録音時間: 3-5分

【都市環境】

電車:
- 場所: 駅ホーム、車内
- 素材: ドア開閉音、アナウンス、走行音
- 用途: リズム素材、テクスチャ
- 注意: 録音禁止区域に注意

工事現場:
- 場所: 建設現場の外側
- 素材: 金属音、衝撃音、重機音
- 用途: パーカッション、インダストリアル
- 注意: 安全な距離を保つ

商店街・市場:
- 場所: 商店街、マーケット
- 素材: 人声のざわめき、呼び込み
- 用途: Ambient、ボーカルテクスチャ
- 注意: プライバシーへの配慮

【室内環境】

キッチン:
- 素材: 水音、食器の音、調理音
- 用途: パーカッション、ワンショット
- 例: フライパンの音 → メタリックヒット

浴室:
- 素材: 水滴、反響音、シャワー
- 用途: Ambient、ドリップ、リバーブ
- 特徴: 自然なリバーブ空間

日用品:
- 素材: ペンのクリック、鍵の音、ジッパー
- 用途: ハイハット代替、パーカッション
- コツ: マイクに近づけて録音
```

### フィールドレコーディングの編集ワークフロー

```
Step 1: 素材の取り込みと整理（10分）

1. レコーダーからPCに転送
2. フォルダ分類:
   field-recordings/
   ├── nature/
   │   ├── rain_20260301_001.wav
   │   ├── ocean_20260302_001.wav
   │   └── wind_20260303_001.wav
   ├── urban/
   │   ├── train_20260304_001.wav
   │   └── market_20260305_001.wav
   └── indoor/
       ├── kitchen_20260306_001.wav
       └── objects_20260307_001.wav

3. 命名規則: [カテゴリ]_[日付]_[連番].wav

Step 2: ノイズ除去と基本編集（15分）

Ableton Liveでの作業:
1. Audio Trackに読み込み
2. 不要部分をカット
3. Fade In / Fade Out 追加
4. ノーマライズ（-1 dBFS目安）

ノイズ除去（必要な場合）:
- EQ Eight: 超低域カット（30 Hz以下）
- Gate: 無音部分のノイズ除去
- 注意: 過度なノイズ除去は質感を失う

Step 3: サンプル化（20分）

ワンショット化:
- 特徴的な瞬間を切り出し
- 長さ: 0.1-2秒
- Fade Out: 50-200 ms

ループ化:
- 繰り返し可能な部分を選択
- 長さ: 1-8小節分
- ループポイントをゼロクロスに設定

テクスチャ化:
- 長い素材（5-30秒）
- 最小限の編集
- Simplerで後から加工
```

---

## リサンプリング技法

### リサンプリングとは

```
定義:
自分が作った音を再びサンプリングして加工すること

通常のサンプリング:
外部の音源 → サンプル → 加工 → 完成

リサンプリング:
自分の音 → 録音 → 再サンプリング → 加工 → 完成
（何度でも繰り返せる）

メリット:
1. 著作権問題なし（自分の音）
2. 無限の加工可能性
3. 偶然の発見（Happy Accident）
4. CPU負荷軽減（エフェクトをBounce）
5. ユニークなサウンド

使用されるジャンル:
- Dubstep / Bass Music: リサンプリングが核心技術
- Drum & Bass: ベースのリサンプリング
- IDM / Experimental: 音のマニピュレーション
- Lo-Fi: テクスチャ追加
```

### リサンプリングのワークフロー

```
方法1: Ableton LiveのResample入力

手順:
1. Audio Trackの入力を「Resampling」に設定
2. 録音したい音を再生
3. Record ボタンで録音
4. マスター出力がそのまま録音される

用途:
- 複数トラックのBounce
- エフェクト込みの録音
- リアルタイムのツマミ操作を録音

方法2: Freeze & Flatten

手順:
1. トラックを右クリック → Freeze Track
2. 右クリック → Flatten
3. MIDIトラックがAudioトラックに変換
4. エフェクト処理が確定

用途:
- CPU負荷軽減
- 確定的なBounce
- 後から再加工

方法3: Export & Re-import

手順:
1. 範囲を選択（Ctrl/Cmd + Shift + R）
2. WAVでExport
3. 新しいトラックにDrag & Drop
4. Simplerに読み込んで加工

用途:
- 最高品質のBounce
- 別プロジェクトへの持ち込み
- サンプルライブラリ化
```

### リサンプリング実践: ベース音の進化

```
Round 1: 元のベース音作成

1. Operator（FMシンセ）で基本ベース
   - Oscillator A: Sine
   - Oscillator B: Saw、Ratio 2
   - Filter: LP 800 Hz
   - 録音: 4小節

Round 2: 1回目のリサンプリング

1. Round 1を Simpler に読み込み
2. Transpose: -5 semitones
3. Saturator: Drive 12 dB
4. Reverb: Decay 2s、Dry/Wet 30%
5. 録音: 4小節

Round 3: 2回目のリサンプリング

1. Round 2を Simpler に読み込み
2. Warp Mode: Texture
3. Grain Size: 小さく
4. Frequency Shifter: +200 Hz
5. Erosion: 15%
6. 録音: 4小節

Round 4: 3回目のリサンプリング

1. Round 3を Simpler に読み込み
2. Reverse: On
3. Stretch: 200%
4. Auto Filter: LFO制御
5. Grain Delay: Spray 60%
6. 録音: 4小節

結果:
- 元のシンプルなベースが完全に別の音に
- 各Roundの音を比較して最適なものを選択
- 偶然の発見を活かす
```

---

## サンプルマングリング

### サンプルマングリングとは

```
定義:
サンプルを極端に加工して原形をとどめない音にする技術

通常のサンプリング:
サンプルの特徴を活かす
→ 元の音が認識できる範囲

サンプルマングリング:
サンプルを破壊的に加工
→ 元の音が完全に認識不能
→ 全く新しい音が生まれる

使用場面:
- 実験的なサウンドデザイン
- テクスチャ作成
- グリッチ / ノイズ作成
- ユニークなパーカッション作成
```

### マングリング手法一覧

```
手法1: Extreme Pitch Shift

方法:
- Transpose: -48 semitones（4オクターブ下）
- または: +48 semitones（4オクターブ上）

結果:
- 低: ドローン、重低音、ランブル
- 高: キラキラ、グリッチ、針金の音

実践:
1. 人の声を -36 semitones
   → 地鳴りのようなドローン
2. ドアの閉まる音を +24 semitones
   → 金属的なクリック音

手法2: Extreme Time Stretch

方法:
- Warp Mode: Texture
- 元の長さの 1000%〜10000% に伸ばす
- Grain Size: 最小値

結果:
- 元の音のテクスチャだけが残る
- 美しいAmbientドローン
- グレイン（粒子）が聴こえる

実践:
1. 1秒のスネアを30秒に伸ばす
   → 金属的なドローン
2. 2秒のボーカルを2分に伸ばす
   → エーテリアルなパッド

手法3: Bit Crusher（Redux）

方法:
- Ableton Redux エフェクト
- Bit Depth: 1-4 bit
- Sample Rate: 500-2000 Hz

結果:
- デジタルな歪み
- レトロゲーム風
- ノイズの追加

実践:
1. ピアノを 4 bit / 1000 Hz
   → チップチューンピアノ
2. ボーカルを 2 bit / 500 Hz
   → ロボットボイス

手法4: Granular Processing

方法:
- Grain Delay使用
- Spray: 80-100%
- Frequency: ランダム
- Pitch: ランダム

結果:
- 粒子状の散乱音
- テクスチャの分解
- 予測不能なパターン

手法5: Feedback Loop

方法:
1. Delay: Feedback 95%（100%未満厳守）
2. エフェクトチェインにFilter、Saturator
3. 録音しながらパラメータを操作

結果:
- 自己増殖する音
- カオス的な発展
- 予測不能な音響

注意:
- Feedback 100%以上は危険（無限増幅）
- Limiter必須
- モニター音量に注意

手法6: Convolution（たたみ込み）

方法:
- Convolution Reverb使用
- IRファイル代わりにサンプルを読み込み
- 通常: 部屋のIR → 残響
- マングリング: 任意の音 → 未知の結果

実践:
1. IRとしてドラムループを使用
   → 入力音にドラムパターンの特性が付加
2. IRとしてボーカルを使用
   → 入力音にフォルマントが付加
```

---

## グラニュラーシンセシス詳説

### グラニュラーシンセシスの原理

```
基本概念:

音をごく短い粒子（Grain）に分解し、
それらを再構成して新しい音を作る技術

Grain（粒子）:
- 長さ: 1ms〜100ms程度
- 各Grainは元のサンプルの断片
- Grainの重なり方で音色が決まる

パラメータ:

1. Grain Size（粒子サイズ）:
   1-10 ms: ノイズ的、テクスチャ
   10-50 ms: 金属的、鈴のような音
   50-100 ms: 元の音が認識できる
   100 ms以上: ほぼ元の音

2. Grain Density（密度）:
   低密度: パラパラ、離散的
   中密度: テクスチャ、流れ
   高密度: 滑らか、連続的

3. Grain Position（位置）:
   固定: 同じ部分を繰り返し
   スキャン: ゆっくり移動
   ランダム: 予測不能

4. Pitch Randomization:
   0%: 元のピッチ
   低: 微妙な揺らぎ
   高: カオス的

5. Spray / Scatter:
   時間軸上のランダム配置
   → テクスチャの複雑さ
```

### Ableton Liveでのグラニュラー処理

```
方法1: Simpler（Textureモード）

設定:
1. サンプルを読み込み
2. Warp: On
3. Mode: Texture（Complex Pro横のドロップダウン）

調整:
- Grain Size: Flux で制御
- Spray: 時間的ランダムさ
- コツ: MIDIで演奏しながら調整

方法2: Grain Delay

設定:
1. Audio Track にGrain Delayを追加
2. Spray: 粒子の散乱度
3. Frequency: Grain のピッチ
4. Pitch: ピッチシフト量
5. Random Pitch: ランダムピッチ

用途:
- リアルタイム処理
- テクスチャ付加
- 空間的な広がり

方法3: Max for Live Granular

推奨デバイス:
- Granulator II（Robert Henke作）
  → 最も高品質なグラニュラーシンセ
  → Ableton Suite付属

- Grain Scanner
  → シンプルなグラニュラー
  → 初心者向け

Granulator IIの主要パラメータ:
- Position: サンプル内の読み取り位置
- Grain Size: 1-500 ms
- Spray: 時間的ランダムさ
- Pitch: ±48 semitones
- Random: 各パラメータのランダム量
- Voices: 同時Grain数（1-32）
```

### グラニュラーシンセシス実践

```
実践1: ボイスからAmbient Pad

素材: 自分の声（あー、と5秒間）

手順:
1. Granulator IIに読み込み
2. Grain Size: 80 ms
3. Spray: 30%
4. Position: LFOでゆっくりスキャン
5. Pitch Random: 10%
6. Voices: 8
7. Reverb: Decay 6s、Dry/Wet 50%
8. コードを演奏（C-E-G-B）

結果: 人間の声の温かみを持つ幻想的なパッド

実践2: 金属音からリズムパターン

素材: 鍵の束を振る音（2秒）

手順:
1. Simpler Texture モードに読み込み
2. Grain Size: 小さく（Flux低）
3. Spray: 50%
4. Filter: Band Pass 2000 Hz
5. Gate: リズムパターンのSidechain
6. Delay: 1/16、Feedback 30%

結果: 金属的でリズミカルなパーカッション

実践3: 環境音からドローン

素材: 雨音（30秒）

手順:
1. Granulator IIに読み込み
2. Grain Size: 200 ms
3. Spray: 60%
4. Position: 固定（最も滑らかな部分）
5. Pitch: -12 semitones
6. Voices: 16
7. Filter LFO: Sine、0.05 Hz
8. Reverb: Decay 12s

結果: 深く包み込むようなドローン
```

---

## Ableton Simplerの高度な活用

### Warp機能の深掘り

```
SimplerのWarp設定:

Classic Mode + Warp:
- Warp On → ピッチ変更時にテンポ維持
- Warp Off → ピッチとテンポが連動（ビンテージ的）

Warp Modeの使い分け:

Beats:
- ドラムループに最適
- Transient Mode:
  1/4 → 4分音符ごと
  1/8 → 8分音符ごと
  1/16 → 16分音符ごと
  Transient → 自動検出

Tones:
- メロディ、単音楽器に最適
- Grain Size: 自動調整
- 品質: Beatsより高い

Texture:
- パッド、環境音に最適
- Grain Size: 手動設定可能
- Flux: グレインのランダム度

Complex:
- ポリフォニック音源に最適
- CPU: 重い
- 品質: 最高（Proはさらに高い）
```

### Multi-Sample的な使い方

```
Simplerで疑似マルチサンプル:

通常: 1サンプルを全鍵盤にマッピング
問題: 高音域/低音域で不自然

解決策: Instrument Rack活用

手順:
1. Instrument Rackを作成
2. Chain 1: Simpler（低音域サンプル）
   Key Zone: C1-B2
3. Chain 2: Simpler（中音域サンプル）
   Key Zone: C3-B4
4. Chain 3: Simpler（高音域サンプル）
   Key Zone: C5-B6

各ChainのSimpler設定:
- Transpose: 各音域に合わせて調整
- Filter: 音域に合わせたEQ的処理
- Volume: レベル合わせ

メリット:
- 各音域で自然な音色
- Simplerの軽さを活かせる
- 柔軟なカスタマイズ
```

### Simplerのスライスモード応用

```
スライスモードの高度設定:

Slice By オプション:

1. Transient:
   - 自動でアタック検出
   - Sensitivity で細かさ調整
   - ドラムループに最適

2. Beat:
   - 拍子ベースでスライス
   - 1/4、1/8、1/16 等
   - 正確なリズムに最適

3. Region:
   - 等間隔でスライス
   - 2、4、8、16、32 分割
   - 均等な分割に最適

4. Manual:
   - 手動でマーカー設置
   - 最も柔軟
   - 不規則な素材に最適

Playback設定:

Mono:
- 前の音を止めて新しい音を再生
- パーカッシブな使用に最適

Poly:
- 複数の音を同時再生
- パッド的な使用に最適

Trigger / Gate:
- Trigger: ノートオンで最後まで再生
- Gate: ノートオフで停止
```

---

## サンプルパック制作

### サンプルパックの構成

```
プロフェッショナルなサンプルパック構成:

[Pack Name]/
├── Drums/
│   ├── Kicks/        (20-30サンプル)
│   ├── Snares/       (20-30サンプル)
│   ├── Hi-Hats/      (15-25サンプル)
│   ├── Claps/        (10-15サンプル)
│   ├── Percussion/   (15-25サンプル)
│   └── Loops/        (10-20ループ)
├── Bass/
│   ├── One-Shots/    (10-20サンプル)
│   └── Loops/        (10-15ループ)
├── Synths/
│   ├── Leads/        (10-15サンプル)
│   ├── Pads/         (10-15サンプル)
│   └── Stabs/        (10-15サンプル)
├── Vocals/
│   ├── Chops/        (20-30サンプル)
│   ├── Phrases/      (10-15サンプル)
│   └── Ad-Libs/      (15-20サンプル)
├── FX/
│   ├── Risers/       (10-15サンプル)
│   ├── Downlifters/  (10-15サンプル)
│   ├── Impacts/      (10-15サンプル)
│   └── Textures/     (10-15サンプル)
├── Loops/
│   ├── Full/         (10-20ループ)
│   └── Stems/        (各ループのステム)
└── README.txt

合計: 200-350サンプル（標準的なパック）
```

### サンプルの品質基準

```
技術仕様:

フォーマット: WAV 24bit/44.1kHz（最低限）
推奨: WAV 24bit/48kHz
ワンショット長: 0.1秒〜5秒
ループ長: 2小節、4小節、8小節
ピーク: -1 dBFS〜-3 dBFS
ノイズフロア: -60 dBFS以下

命名規則:
[BPM]_[Key]_[カテゴリ]_[番号]_[説明].wav
例: 128_Cm_Kick_01_Punchy.wav
例: 125_Am_VocalChop_03_Ethereal.wav

品質チェックリスト:
□ クリック/ポップノイズなし
□ 適切なFade In/Out
□ ノーマライズ済み
□ DCオフセットなし
□ 無音部分が適切（先頭0-10ms、末尾50-200ms）
□ ステレオ/モノが適切
□ BPM/キー情報が正確
□ ファイル名が命名規則に従っている
```

### 配布と販売

```
配布プラットフォーム:

無料配布:
1. Bandcamp（"name your price"設定）
2. SoundCloud（説明欄にリンク）
3. 自分のウェブサイト
4. Reddit（r/drumkits等）

有料販売:
1. Splice（最大のプラットフォーム）
   - 審査あり
   - レベニューシェア
   - 個別サンプル販売

2. Loopmasters
   - 審査あり
   - パック単位販売
   - プロモーション支援

3. Bandcamp
   - 審査なし
   - 自由な価格設定
   - 直接販売

4. Gumroad
   - 簡単な設定
   - デジタル配信に最適
   - 手数料が低い

ライセンス設定:
- Royalty-Free（推奨）
- 商用利用可（推奨）
- 再配布禁止（推奨）
- クレジット不要（推奨）
```

---

## ジャンル別サンプリング戦略

### Hip Hop / Lo-Fi

```
特徴的なサンプリング手法:

ソウル/ファンクからのサンプリング:
- ビンテージレコードの質感
- コード進行をループ
- Pitch Down（-3〜-5 semitones）で重厚に

チョップ手法:
1. フレーズを小節単位でスライス
2. 順番を入れ替えて新しい展開
3. キー合わせのためにTranspose

Lo-Fi加工チェイン:
1. Vinyl Distortion（Abletonプラグイン）
   Tracing Model: On
   Pinch: 10%
   → レコードノイズ

2. EQ Eight
   High Cut: 8000 Hz
   Low Cut: 80 Hz
   → 帯域制限でビンテージ感

3. Redux
   Bit Depth: 12 bit
   Sample Rate: 22050 Hz
   → デジタル劣化

4. Saturator
   Drive: 5 dB
   Curve: Analog Clip
   → テープサチュレーション

5. Auto Filter
   LFO Rate: 2 Hz
   Amount: 微量
   → Wow/Flutter（テープ揺れ）

ドラムサンプリング:
- SP-1200風: 12bit サンプリング
- MPC風: パッドの叩き方でベロシティ
- Chop & Flip: ブレイクビーツの再構成
```

### House / Disco

```
特徴的なサンプリング手法:

ディスコからのサンプリング:
- ストリングスのフレーズ
- ベースライン
- ボーカルフック

手法:
1. 4小節のループを作成
2. BPMを120-128に調整
3. Warp Mode: Complex Pro

ボーカルチョップ（House向け）:
- 短いフレーズ（1-2小節）
- リバーブ感を活かす
- Filter Sweep: LP → Open

クラシックHouseサンプル加工:
1. EQ: 中域を少しブースト（1-3 kHz）
2. Compressor: 軽い圧縮
3. Tape Saturation: 温かみ
4. Stereo Width: 120%
5. Reverb: Plate、Decay 1.5s

909/808ドラムのリサンプリング:
- クラシックドラムマシンの音を録音
- Saturatorで倍音追加
- Transient Shaperでアタック強調
- レイヤリングで独自性
```

### Techno / Industrial

```
特徴的なサンプリング手法:

工業音・金属音:
- 工場の音、機械音
- 金属を叩く音
- 電子ノイズ

加工手法:
1. Extreme Pitch Shift
   -24〜+24 semitones
   → 原形をとどめない

2. Distortion Chain
   Overdrive → Saturator → Erosion
   → 過激な歪み

3. Resonator
   周波数: キーに合わせる
   Decay: 短め
   → メロディック・パーカッション

4. Frequency Shifter
   Fine: 50-500 Hz
   → 非調和的な倍音

テクノ向けリズム処理:
- Gate: 16分音符パターン
- Sidechain: キックに同期
- Delay: Ping Pong、1/16
- リバーブ: 短いDecay（0.3-0.8s）

インダストリアルテクスチャ:
1. フィールドレコーディング素材
2. Grain Delay: Spray 70%
3. Auto Pan: 高速
4. Phaser: Rate 0.5 Hz
5. Erosion: Wide Noise 20%
```

### Drum & Bass / Jungle

```
特徴的なサンプリング手法:

ブレイクビーツの解体と再構成:

クラシックブレイク:
- Amen Break
- Think Break
- Apache Break
- Funky Drummer

チョップ手法:
1. ブレイクを16-32スライスに分割
2. 各ヒットを個別に加工
   - キック: Sub追加、Compress
   - スネア: リバーブ、Distortion
   - ハイハット: HP Filter、Pitch Up
3. 新しい順番で再配置
4. BPM: 170-180に高速化

タイムストレッチ活用:
- 元のBPM（110-130）→ 170-180
- Warp Mode: Beats（Transient保持）
- アーティファクトを味として活かす

ベースのリサンプリング:
1. シンセベースを作成
2. Resampling録音
3. Pitch Down -12
4. Saturator: Drive 15 dB
5. EQ: Sub（30-60 Hz）ブースト
6. 再度Resampling
7. スライスしてリズム的に使用

Reese Bass作成:
1. 2つのSaw波（微妙にDetune）
2. LP Filter: Cutoff 500 Hz
3. Filter LFO: ゆっくり動かす
4. Distortion: 軽め
5. Chorus: 微量
→ うねるようなベース
```

---

## 実践演習: 総合サンプリングプロジェクト

### 演習1: 「キッチンビート」チャレンジ

```
目的: キッチンの音だけでビートを作る

素材収集（30分）:
1. まな板を叩く音 → キック
2. フライパンを叩く音 → スネア
3. グラスをスプーンで叩く音 → ハイハット
4. 水道の水音 → ハイハットオープン
5. 冷蔵庫の扉を閉める音 → クラップ
6. 食器を重ねる音 → パーカッション
7. 電子レンジのビープ音 → シンセヒット
8. やかんの蒸気音 → ノイズ/FX

加工手順:
1. 各素材をSimpler に読み込み
2. キック: Pitch Down -12、LP Filter 200 Hz
3. スネア: Pitch Down -5、Reverb
4. ハイハット: HP Filter 5000 Hz、短いDecay
5. Drum Rack に配置
6. MIDIパターン作成（4小節）
7. EQ、Compressorで仕上げ

評価基準:
□ ビートとして機能する
□ 各素材が適切な役割を果たす
□ グルーヴがある
□ 元の音が認識できない加工レベル
□ 4小節以上のパターン

所要時間: 90分
```

### 演習2: 「ボイスオーケストラ」チャレンジ

```
目的: 自分の声だけで楽曲を作る

素材収集（20分）:
1. 「あー」持続音 → パッド
2. 「ん」ハミング → ベース
3. 舌打ち音 → ハイハット
4. 「ぱ」破裂音 → キック
5. 「つ」破擦音 → スネア
6. 口笛 → リード
7. 息を吸う音 → ライザー
8. 歌のフレーズ → ボーカルチョップ

加工手順:
1. パッド: 「あー」をGranulator IIで処理
   Grain Size: 100 ms、Voices: 12、Reverb 60%
2. ベース: 「ん」を-24 semitones、Saturator
3. ドラム: 各素材をDrum Rackに配置して加工
4. リード: 口笛をOctave Up、Chorus
5. ライザー: 息をReverse、長いReverb
6. ボーカル: フレーズを8スライス

楽曲構成:
- Intro: パッドのみ（8小節）
- Verse: パッド + ドラム + ベース（16小節）
- Build: ライザー追加（4小節）
- Drop: 全要素（16小節）
- Outro: パッドのフェードアウト（8小節）

評価基準:
□ 楽曲として成立する
□ 5つ以上の音色が使われている
□ 展開がある（Intro→Verse→Drop等）
□ 元が声だとわからない加工
□ 2分以上の長さ

所要時間: 120分
```

### 演習3: 「リサンプリング10回」チャレンジ

```
目的: 1つの音を10回リサンプリングして変化を追跡

開始素材: 任意の1つの音（推奨: ピアノの単音）

ルール:
- 毎回異なるエフェクト/加工を使用
- 各Roundを録音して保存
- 最終結果と元の音を比較

Round 1: Pitch Shift（-7 semitones）
Round 2: Reverb（Decay 5s、Wet 80%）
Round 3: Reverse
Round 4: Time Stretch 200%（Texture Mode）
Round 5: Grain Delay（Spray 50%）
Round 6: Frequency Shifter（+150 Hz）
Round 7: Erosion（Noise 30%）
Round 8: Chorus + Flanger
Round 9: Auto Filter（LFO Sweep）
Round 10: Bit Crusher（6 bit）

記録シート:
Round | 使用エフェクト | 音の印象 | 評価(1-5)
1     |               |          |
2     |               |          |
...   |               |          |
10    |               |          |

分析ポイント:
- どのRoundで元の音から離れたか
- 最も劇的な変化はどのRoundか
- 好みの音はどのRoundか
- 偶然の発見はあったか

所要時間: 60分
```

---

## トラブルシューティング

### サンプリング時の一般的な問題と解決策

```
問題1: サンプルにクリック/ポップノイズが出る

原因:
- サンプルの開始/終了点が波形の途中
- ループポイントが不適切
- DCオフセット

解決策:
1. Fade In/Out を追加（1-10 ms）
2. ゼロクロスポイントに開始/終了を設定
3. Utility → DC Filter: On
4. ループの場合: Crossfade を増やす（100-300 ms）

問題2: Pitch Shiftすると音質が劣化する

原因:
- 大幅なPitch Shiftによるアーティファクト
- 不適切なWarp Mode

解決策:
1. Complex Pro Modeを使用（最高品質）
2. 少しずつShift（±12以内推奨）
3. 大幅なShiftの場合: リサンプリングで段階的に
4. EQで不要なアーティファクトを除去
5. Warp Off で Pitch-Tempo連動を活かす

問題3: サンプルのテンポがプロジェクトに合わない

原因:
- サンプルのBPMが異なる
- Warp設定が不適切

解決策:
1. サンプルのWarp: On
2. 正しいBPMを設定:
   - 右クリック → 「○○ BPMに設定」
   - マニュアルでWarpマーカーを調整
3. Warp Mode を素材に合わせて選択
4. 半分/倍のBPMも試す

問題4: ボーカルチョップのタイミングがずれる

原因:
- スライスポイントが不正確
- Quantize設定がOff

解決策:
1. Sensitivity を調整して適切なスライス数に
2. 手動でスライスマーカーを修正
3. MIDIノートをQuantize（1/16推奨）
4. Simplerの各スライスのStart位置を微調整
5. Groove Pool でスイング追加

問題5: サンプルのキーが楽曲に合わない

原因:
- サンプルのキーが不明
- 不適切なTranspose値

解決策:
1. Spectrum / Tuner で元のキーを確認
2. キー検出ツール使用:
   - Mixed In Key
   - Keyfinder（無料）
3. Transpose で半音ずつ調整
4. Detune で微調整（±50 cent）
5. どうしても合わない場合: サンプルを加工して
   調性をぼかす（Reverb、Delay等）

問題6: CPU負荷が高すぎる

原因:
- 多数のSampler/Simplerインスタンス
- Complex Pro Warp Mode
- 重いエフェクトチェイン

解決策:
1. Freeze & Flatten で Audio化
2. Warp Mode: Complex Pro → Beats に変更
3. Simplerを使用（Samplerより軽い）
4. エフェクトをBounce（Resample）
5. Sample Rate: 48kHz → 44.1kHz
6. Buffer Size: 256 → 512 → 1024

問題7: リバーブ/ディレイの残響がループ時に途切れる

原因:
- クリップ境界でエフェクトのテールがカット

解決策:
1. Return Track にリバーブ/ディレイを配置
2. クリップの末尾を少し延長
3. Arrangement View で重なりを持たせる
4. Consolidate（統合）してからFade
5. Fade Out をリバーブのDecay以上に設定

問題8: サンプルが位相的に問題がある

原因:
- レイヤリング時の位相打ち消し
- ステレオサンプルのMono再生

解決策:
1. Utility → Phase Invert で片方反転して確認
2. 問題のある周波数をEQで処理
3. サンプルの開始位置を微調整（位相合わせ）
4. Mono再生で確認（位相問題が顕著に出る）
5. Mid/Sideプロセッシングで分離
```

### パフォーマンス最適化のヒント

```
プロジェクトの整理:

1. 不要なサンプルを削除
   - File → Collect All and Save
   - 未使用サンプルを除外

2. サンプルの最適化
   - 無駄に長いサンプルをトリム
   - 高解像度すぎるサンプルをダウンサンプル
   - ステレオ不要ならMono化（容量半分）

3. エフェクトの整理
   - 不要なエフェクトを削除/Bypass
   - 確定したエフェクトはBounce
   - Return Track活用（共有リバーブ）

4. メモリ管理
   - RAMモード vs HDDモード
   - 大きなサンプルはHDDモード
   - 頻繁に使うサンプルはRAMモード

ライブパフォーマンス向け:
- 全サンプルをFlatten
- エフェクトを最小限に
- Buffer Size: 256以下
- 不要なプラグインを削除
- テスト: 本番環境で事前確認
```

---

## 高度なサンプリングTips

### クリエイティブなサンプリングソース

```
意外なサンプリングソース:

1. ラジオの周波数間ノイズ:
   - AMラジオの局間ノイズ
   - 用途: テクスチャ、Ambient
   - 加工: Grain Delay + Reverb

2. 電化製品のモーター音:
   - 冷蔵庫、洗濯機、扇風機
   - 用途: ドローン、ベースの素材
   - 加工: Pitch Down + Filter

3. 楽器の非伝統的な演奏:
   - ギターのボディを叩く
   - ピアノの弦を直接弾く
   - ドラムのリムショット
   - 用途: パーカッション

4. テキスタイルの音:
   - ジーンズをこする音
   - 布を裂く音
   - ジッパーの音
   - 用途: ノイズ、ハイハット

5. ガラス・陶器:
   - グラスの縁を指でこする
   - 陶器を叩く
   - ガラスが割れる音（注意して）
   - 用途: ベル、パーカッション

6. 紙・段ボール:
   - 紙をくしゃくしゃにする
   - 段ボールを破る
   - 本のページをめくる
   - 用途: ノイズ、テクスチャ

7. 体の音:
   - 手拍子（様々なバリエーション）
   - 指スナップ
   - 足踏み
   - ボディパーカッション
   - 用途: オーガニックパーカッション

8. デジタルエラー音:
   - データ破損したファイルの再生
   - 画像ファイルを音声として読み込む
   - 意図的なバッファオーバーフロー
   - 用途: グリッチ、実験音楽
   - 注意: 音量に注意
```

### サンプルのキャラクター付け

```
ビンテージ感を加える:

1. テープシミュレーション:
   - Saturator（Analog Clip）
   - Auto Filter（微量のWow/Flutter）
   - EQ: High Shelf -3 dB @ 8 kHz
   - Noise: 微量のテープヒス追加

2. レコード感を加える:
   - Vinyl Distortion（Ableton）
   - Crackle: 10-20%
   - Pinch: 5-10%
   - EQ: 帯域制限（100 Hz〜8 kHz）

3. ラジオ感を加える:
   - Band Pass Filter: 500-3000 Hz
   - Distortion: 軽め
   - Bit Crusher: 12 bit
   - Mono化

モダン感を加える:

1. クリスタルクリア:
   - EQ: 精密なカット
   - Compressor: クリーンな圧縮
   - Stereo Width: 広め
   - Air Band: 12 kHz ブースト

2. ハイパーポップ風:
   - Pitch Shift: +12（オクターブ上）
   - Formant Shift: 上げる
   - Auto-Tune的処理
   - 極端なCompression

3. フューチャーベース風:
   - サイドチェインCompression
   - スーパーソー的なレイヤリング
   - LFO Volume（1/4 Triplet）
   - Wide Stereo
```

---

## チェックリスト: サンプリングマスターへの道

```
レベル1: 基礎（1-2週間）
□ SimplerとSamplerの違いを理解
□ ワンショット加工が15分でできる
□ 基本的なボーカルチョップができる
□ Reverseサンプルが作れる
□ 著作権の基本を理解

レベル2: 中級（3-4週間）
□ テクスチャサウンドデザインができる
□ フィールドレコーディングを実践
□ リサンプリングの基本ワークフロー習得
□ ジャンルに合ったサンプリング手法を選択
□ サンプルのキャラクター付けができる

レベル3: 上級（2-3ヶ月）
□ グラニュラーシンセシスを活用
□ サンプルマングリング技術を習得
□ 複雑なエフェクトチェインを設計
□ オリジナルサンプルパックを制作
□ ライブでのサンプリング操作が可能

レベル4: マスター（6ヶ月〜）
□ あらゆる素材から楽曲制作可能
□ 独自のサンプリングワークフロー確立
□ 他者にサンプリング技術を教えられる
□ サンプルパックの販売・配布
□ サンプリングが自身のアーティスト性の核

継続的な成長のために:
- 毎日15分でも新しい音をサンプリング
- 月に1つサンプルパックを作る
- 他のプロデューサーのサンプリング手法を研究
- 意図的に制約を設ける（キッチンの音だけ等）
- 失敗を恐れず実験する
```

---

## 次に読むべきガイド

- [Synthesis Basics（シンセシス基礎）](./synthesis-basics.md) - 次のトピックへ進む

---

## 参考文献

- [MDN Web Docs](https://developer.mozilla.org/) - Web技術のリファレンス
- [Wikipedia](https://ja.wikipedia.org/) - 技術概念の概要
