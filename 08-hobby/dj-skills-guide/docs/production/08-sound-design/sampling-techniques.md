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
