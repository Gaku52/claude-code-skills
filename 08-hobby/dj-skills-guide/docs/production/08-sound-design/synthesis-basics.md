# Synthesis Basics（シンセシス基礎）

シンセサイザーで音を作る基本原理を完全理解します。Oscillator、Filter、Envelope、LFOの4つの要素をマスターすれば、70%の音は作れるようになります。

## この章で学ぶこと

- シンセシスの基本原理（減算式合成）
- Oscillator（オシレーター）と波形の種類
- Filter（フィルター）と音色の削り方
- Envelope（エンベロープ）と時間的変化
- LFO（エルエフオー）と周期的変化
- 4つの要素を組み合わせた音作り
- Wavetableでの実践
- 基本パッチ10個の作成


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [Sampling Techniques（サンプリング技術）](./sampling-techniques.md) の内容を理解していること

---

## なぜSynthesis Basicsが重要なのか

**理解度の差:**

```
理解なし:
行動:
プリセットをランダムに選ぶ
パラメータの意味が分からない
調整できない

結果:
運任せ
時間浪費（2時間）
完成しない

理解あり:
行動:
目的の音をイメージ
必要なパラメータを調整
10分で完成

結果:
確実に作れる
時間効率
クリエイティブ

プロとアマの差:
アマ: パラメータ理解度 20%
プロ: パラメータ理解度 90%
```

**シンセシス理解の重要性:**

- **効率**: 10分で目的の音を作れる
- **創造性**: 無限の可能性を理解できる
- **応用**: どのシンセでも通用する知識
- **修正**: 問題を即座に解決できる

---

## シンセサイザーの基本構造

### 信号の流れ（Signal Flow）

```
1. Oscillator（音の源）
   ↓
   音を生成
   ↓
2. Filter（音色を削る）
   ↓
   倍音を削って音色を作る
   ↓
3. Amplifier（音量制御）
   ↓
   音量を調整
   ↓
4. Output（出力）

モジュレーション:
- Envelope → Filter、Amplifier
- LFO → Oscillator、Filter
```

### 減算式合成（Subtractive Synthesis）

**基本原理:**

```
原理:
倍音豊かな波形（Saw、Square）
   ↓
Filterで削る
   ↓
目的の音色

例:
Saw波（倍音多い）
   ↓
Low Pass Filter 500 Hz
   ↓
丸い音色
   ↓
ベース音

なぜ "減算式" か:
倍音を減らす（削る）ことで音を作る
→ 最も一般的なシンセシス方式
→ Wavetable、Analogはこの方式
```

**他のシンセシス方式（参考）:**

```
FM合成（Frequency Modulation）:
倍音を足す
→ Operator

加算合成（Additive Synthesis）:
Sine波を重ねる
→ 複雑

ウェーブテーブル合成:
波形を切り替える
→ Wavetable

サンプリング:
録音した音を加工
→ Sampler
```

---

## 1. Oscillator（オシレーター）

**定義**: 音の源。波形を生成する。

### 波形の種類と特徴

**1. Sine波（サイン波）**

```
形状:
     /\
    /  \
   /    \
  /      \___

特徴:
- 最も単純な波形
- 倍音なし（基音のみ）
- 柔らかい音
- 丸い音

用途:
- Sub Bass（50-100 Hz）
- ベル音（FM合成）
- テスト信号

倍音構造:
基音（1倍音）: 100%
2倍音: 0%
3倍音: 0%
→ 倍音なし
```

**2. Saw波（ノコギリ波）**

```
形状:
 /|  /|  /|
/ | / | / |
  |/  |/  |

特徴:
- 全倍音を含む
- 明るい音
- 鋭い音
- 倍音最多

用途:
- ベース（Techno）
- リード（全ジャンル）
- ブラス
- パッド

倍音構造:
基音: 100%
2倍音: 50%
3倍音: 33%
4倍音: 25%
5倍音: 20%
...
→ 全倍音を含む（最も明るい）
```

**3. Square波（矩形波）**

```
形状:
 ___   ___
|   | |   |
|   |_|   |_

特徴:
- 奇数倍音のみ
- Sawより柔らかい
- 中空的な音

用途:
- リード（8bit風）
- ベース（Deep House）
- パッド
- チップチューン

倍音構造:
基音: 100%
2倍音: 0%
3倍音: 33%
4倍音: 0%
5倍音: 20%
...
→ 奇数倍音のみ
```

**4. Triangle波（三角波）**

```
形状:
   /\
  /  \
 /    \
/      \

特徴:
- 奇数倍音のみ（Squareより少ない）
- 非常に柔らかい
- Sineに近い

用途:
- Sub Bass（Sineの代替）
- パッド
- ビンテージ音

倍音構造:
基音: 100%
2倍音: 0%
3倍音: 11%（Squareの1/3）
5倍音: 4%
→ 奇数倍音のみ（非常に少ない）
```

### 波形の選び方

```
目的の音 → 推奨波形

Sub Bass:
- Sine 80%
- Triangle 20%

Techno Bass:
- Saw 90%
- Square 10%

Deep House Bass:
- Square 60%
- Triangle 40%

リード:
- Saw 70%
- Square 30%

パッド:
- Triangle 40%
- Square 30%
- Saw 30%
```

### Oscillatorパラメータ

**1. Octave（オクターブ）**

```
定義: 音の高さ（1オクターブ = 周波数2倍）

範囲:
-3 octave（超低音）
-2 octave（低音）
-1 octave（低め）
0 octave（標準）
+1 octave（高め）
+2 octave（高音）
+3 octave（超高音）

用途:
Sub Bass: -1 octave
ベース: 0 octave
リード: 0 octave
パッド: 0 octave
ハイハット: +3 octave
```

**2. Detune（デチューン）**

```
定義: 微妙なピッチのズレ（±50 cent = ±半音）

範囲:
0 cent（ズレなし）
±5 cent（わずかに太い）
±10 cent（コーラス効果）
±20 cent（ハモリ感）
±50 cent（半音）

用途:
2つのOSCをDetune:
OSC 1: Detune 0
OSC 2: Detune +7 cent
→ 音が太くなる、コーラス効果

理由:
微妙にズレると波が干渉
→ 音が揺れる（ビート）
→ アナログ感、温かみ
```

**3. Volume（音量）**

```
定義: Oscillatorの音量バランス

使用例:
OSC 1（Saw）: Volume 100%（メイン）
OSC 2（Square）: Volume 50%（ブレンド）
Sub（Sine）: Volume 30%（低域補強）

バランス:
Main OSC: 100%
2nd OSC: 50-80%
Sub: 20-40%
```

---

## 2. Filter（フィルター）

**定義**: 特定の周波数を削る（減衰させる）

### Filterの種類

**1. Low Pass Filter（ローパスフィルター）**

```
動作:
Cutoff周波数より高い周波数を削る
→ 低い周波数だけ通す（Pass）

図:
|
|    ←音量
|___
|    \
|     \___
|__________|_________
    ↑      ←周波数
  Cutoff

用途:
- ベース（200-500 Hz以下を通す）
- パッド（1500 Hz以下を通す）
- 温かい音色

最も一般的なFilter（使用頻度80%）
```

**2. High Pass Filter（ハイパスフィルター）**

```
動作:
Cutoff周波数より低い周波数を削る
→ 高い周波数だけ通す

図:
|           ___
|          /
|        _/
|       /
|______/
|___|________
    ↑
  Cutoff

用途:
- 低域削除（30-50 Hz以下を削る）
- リード（200 Hz以下を削る）
- 音の整理

ミキシングで必須（使用頻度70%）
```

**3. Band Pass Filter（バンドパスフィルター）**

```
動作:
Cutoff周波数の周辺だけ通す
→ 帯域を通す

図:
|
|      /\
|     /  \
|    /    \
|___/      \___
|___|_____|___
    ↓    ↑
   低   高

用途:
- 電話音（300-3000 Hz）
- ラジオ音
- エフェクト

特殊用途（使用頻度10%）
```

**4. Notch Filter（ノッチフィルター）**

```
動作:
Cutoff周波数だけ削る
→ バンドパスの逆

用途:
- 特定周波数の削除
- ハウリング除去

特殊用途（使用頻度5%）
```

### Filterパラメータ

**1. Cutoff（カットオフ周波数）**

```
定義: どこから削るか

範囲: 20 Hz - 20000 Hz

設定例:
Sub Bass:
Cutoff 150 Hz
→ 150 Hz以上を削る
→ 深い低音のみ

Techno Bass:
Cutoff 500 Hz（開始）
Cutoff 2000 Hz（ピーク）
→ オートメーションで動かす

リード:
Cutoff 3000 Hz
→ 3000 Hz以上を削る
→ 柔らかめのリード

パッド:
Cutoff 1500 Hz
→ 非常に柔らかい

最も重要なパラメータ（サウンドの80%を決める）
```

**2. Resonance（レゾナンス）**

```
定義: Cutoff周波数を強調

範囲: 0-100%

動作:
Cutoff周波数を山型に強調
→ ピーキング

図:
|
|      /\  ←Resonance 70%
|     /  \
|    /    \___
|___/
|___|________
    ↑
  Cutoff

効果:
0%: 強調なし（滑らか）
20%: わずかに強調
50%: 中程度の強調（「ピコピコ」）
70%: 強い強調（TB-303風）
90%+: 自己発振（ホイッスル音）

用途:
Acid Bass: Resonance 60-80%
クリーンベース: Resonance 0-20%
リード: Resonance 10-30%
```

**3. Slope（スロープ、dB/oct）**

```
定義: 削る急峻さ

種類:
12 dB/oct: 緩やか（柔らかい）
24 dB/oct: 急峻（鋭い）
48 dB/oct: 非常に急峻

選び方:
ベース: 24 dB/oct（低域をしっかり削る）
リード: 12 dB/oct（柔らかく削る）
パッド: 12 dB/oct

Wavetableでは:
デフォルト 24 dB/oct（十分）
```

---

## 3. Envelope（エンベロープ）

**定義**: 時間的な変化を制御する

### ADSR Envelope

```
構成要素:
A = Attack（立ち上がり時間）
D = Decay（減衰時間）
S = Sustain（持続レベル）
R = Release（余韻時間）

図:
    ___Peak
   /|   \
  / |    \_____Sustain
 /  |          \
/   |           \___
A  D     S      R

タイムライン:
1. 鍵盤を押す
2. Attack: 0 → Peak（100%）
3. Decay: Peak → Sustain レベル
4. Sustain: 鍵盤を押している間
5. 鍵盤を離す
6. Release: Sustain → 0
```

### パラメータ詳細

**1. Attack（アタック）**

```
定義: 鍵盤を押してから最大音量に達するまでの時間

範囲: 0 ms - 5000 ms

設定例:
0 ms: 即座に最大音量（ベース、ドラム）
10 ms: わずかに柔らかい（リード）
100 ms: ゆっくり立ち上がる（パッド）
500 ms: 非常にゆっくり（Ambient）
2000 ms: 超ゆっくり（Drone）

用途別:
Sub Bass: 0 ms（即座に）
Techno Bass: 0-5 ms
リード: 10-50 ms
パッド: 500-2000 ms
```

**2. Decay（ディケイ）**

```
定義: Peak から Sustain レベルまで減衰する時間

範囲: 0 ms - 5000 ms

設定例:
0 ms: 即座にSustainレベル
100 ms: 短い減衰
500 ms: 中程度の減衰
2000 ms: 長い減衰

用途別:
Pluck音（ピアノ、ギター）: Decay 200-500 ms
パッド: Decay 1000 ms
持続音（オルガン）: Decay 0 ms（すぐSustain）
```

**3. Sustain（サステイン）**

```
定義: 鍵盤を押している間の音量レベル

範囲: 0% - 100%

設定例:
100%: 鍵盤を押している間ずっと最大音量（オルガン）
70%: わずかに減衰（リード）
50%: 中程度の減衰
0%: すぐに消える（Pluck、ドラム）

用途別:
オルガン、パッド: 100%
リード: 70-90%
ベース: 80-100%
Pluck、ドラム: 0%
```

**4. Release（リリース）**

```
定義: 鍵盤を離してから音が消えるまでの時間

範囲: 0 ms - 10000 ms

設定例:
10 ms: 即座に消える（パーカッシブ）
100 ms: 短い余韻（ベース）
500 ms: 中程度の余韻（リード）
2000 ms: 長い余韻（パッド）
5000 ms: 超長い余韻（Ambient）

用途別:
ベース: 10-100 ms（短い）
リード: 200-500 ms
パッド: 2000-5000 ms（長い）
```

### Envelopeの種類

**1. Amp Envelope（音量エンベロープ）**

```
対象: 音量

必須: Yes（全ての音に必須）

設定例:
Sub Bass:
A: 0 ms
D: 50 ms
S: 100%
R: 10 ms

パッド:
A: 800 ms
D: 1000 ms
S: 80%
R: 3000 ms
```

**2. Filter Envelope（フィルターエンベロープ）**

```
対象: Filter Cutoff

効果: 音色の時間的変化

設定例:
Pluck音:
A: 0 ms
D: 300 ms
S: 0%
R: 100 ms
Envelope Amount: +50

動作:
鍵盤を押す → Cutoff 開く（明るい）
300 ms後 → Cutoff 閉じる（暗い）
→ ギター、ピアノのような減衰
```

**3. Pitch Envelope（ピッチエンベロープ）**

```
対象: Pitch

効果: ピッチ変化

設定例:
キック:
A: 0 ms
D: 50 ms
S: 0%
R: 0 ms
Envelope Amount: +12 semitones

動作:
鍵盤を押す → Pitch +12 semitones（1オクターブ上）
50 ms後 → Pitch 0（元に戻る）
→ ドゥーン（キック音）
```

---

## 4. LFO（Low Frequency Oscillator）

**定義**: 周期的な変化を作るオシレーター

### LFOの基本

```
特徴:
- 非常に低い周波数（0.1-20 Hz）
- 音として聞こえない
- パラメータを周期的に変化させる

用途:
- ビブラート（Pitch変化）
- トレモロ（Volume変化）
- ワウワウ（Filter変化）
- オートパン（Pan変化）
```

### LFOパラメータ

**1. Rate（レート、速度）**

```
定義: LFOの周期の速さ

範囲: 0.01 Hz - 40 Hz

設定例:
0.2 Hz: 非常にゆっくり（5秒/周期）
1 Hz: ゆっくり（1秒/周期）
5 Hz: 中速（ビブラート）
10 Hz: 速い（トレモロ）
20 Hz: 非常に速い

Sync to Tempo:
1/4: 4分音符ごと
1/8: 8分音符ごと
1/16: 16分音符ごと
→ テンポに同期（重要）
```

**2. Depth（デプス、深さ）**

```
定義: 変化の大きさ

範囲: 0% - 100%

設定例:
0%: 変化なし
5%: わずかな変化（ビブラート）
20%: 中程度の変化
50%: 大きな変化
100%: 最大変化

バランス:
ビブラート: Depth 5-10%
ワウワウ: Depth 30-60%
トレモロ: Depth 20-40%
```

**3. Waveform（波形）**

```
種類:
1. Sine: 滑らか
2. Triangle: やや角ばった
3. Square: カクカク（オン/オフ）
4. Saw Up: 上昇のこぎり
5. Saw Down: 下降のこぎり
6. Random (S&H): ランダム

用途:
ビブラート: Sine
ワウワウ: Triangle、Sine
トレモロ: Square
ランダム効果: Random
```

### LFO応用例

**1. ビブラート（Pitch変化）**

```
設定:
Destination: OSC Pitch
Waveform: Sine
Rate: 5 Hz
Depth: 5%

効果:
ピッチが上下に揺れる
→ 歌声のビブラート
→ リード、パッドに使用
```

**2. ワウワウ（Filter変化）**

```
設定:
Destination: Filter Cutoff
Waveform: Triangle
Rate: 1/4（テンポ同期）
Depth: 40%

効果:
Filter Cutoffが周期的に開閉
→ ワウワウ音
→ Funkベース、Technoリード
```

**3. トレモロ（Volume変化）**

```
設定:
Destination: Amp Volume
Waveform: Sine
Rate: 8 Hz
Depth: 30%

効果:
音量が周期的に変化
→ トレモロ効果
→ パッド、FX
```

**4. オートパン（Pan変化）**

```
設定:
Destination: Pan
Waveform: Sine
Rate: 1/8（テンポ同期）
Depth: 80%

効果:
音が左右に動く
→ ステレオ効果
→ パッド、FX、リード
```

---

## 4つの要素を組み合わせた音作り

### 例1: Sub Bass

```
目標: 深く太い Sub Bass

Oscillator:
- Waveform: Sine
- Octave: 0
- Detune: 0

Filter:
- Type: Low Pass 24 dB
- Cutoff: 150 Hz
- Resonance: 0%

Amp Envelope:
- Attack: 0 ms
- Decay: 50 ms
- Sustain: 100%
- Release: 10 ms

LFO:
- なし

理由:
- Sine波: 倍音なし、純粋な低音
- LP 150 Hz: 150 Hz以上を完全カット
- Attack 0 ms: 即座に鳴る
- LFOなし: 安定した低音
```

### 例2: Techno Acid Bass

```
目標: TB-303風のアシッドベース

Oscillator:
- Waveform: Saw
- Octave: 0
- Detune: 0

Filter:
- Type: Low Pass 24 dB
- Cutoff: 500 Hz（開始点）
- Resonance: 70%

Filter Envelope:
- Attack: 0 ms
- Decay: 200 ms
- Sustain: 20%
- Release: 50 ms
- Envelope Amount: +50

Amp Envelope:
- Attack: 0 ms
- Decay: 100 ms
- Sustain: 80%
- Release: 20 ms

LFO:
- なし（Filter EnvelopeとCutoffオートメーションで動き）

理由:
- Saw波: 明るい音色
- Resonance 70%: ピコピコ音
- Filter Envelope: 音色が時間的に変化
- Cutoffオートメーション: 手動で動かす
```

### 例3: Ambient Pad

```
目標: 柔らかく広がりのあるパッド

Oscillator 1:
- Waveform: Triangle
- Octave: 0
- Volume: 100%

Oscillator 2:
- Waveform: Saw
- Octave: +1
- Detune: +12 cent
- Volume: 60%

Filter:
- Type: Low Pass 12 dB
- Cutoff: 1500 Hz
- Resonance: 10%

Amp Envelope:
- Attack: 1000 ms
- Decay: 1500 ms
- Sustain: 70%
- Release: 3000 ms

LFO 1:
- Destination: Filter Cutoff
- Waveform: Sine
- Rate: 0.3 Hz
- Depth: 15%

理由:
- 2 OSC Detune: 広がり
- Attack 1000 ms: ゆっくり立ち上がる
- Release 3000 ms: 長い余韻
- LFO → Filter: 音色がゆっくり変化
```

---

## Wavetableでの実践

### 基本パッチ1: Sub Bass

```
手順:
1. Wavetableを追加
2. OSC 1: Basic Shapes > Sine
3. OSC 2: Off
4. Filter: Low Pass 24 dB
5. Cutoff: 150 Hz
6. Amp Envelope:
   - A: 0, D: 50, S: 100%, R: 10
7. テスト: C1, F1, G1 を演奏

確認:
スペクトラムで 50-100 Hz が強い
150 Hz 以上はほぼなし
```

### 基本パッチ2: Techno Bass

```
手順:
1. Wavetableを追加
2. OSC 1: Basic Shapes > Saw
3. Filter: Low Pass 24 dB
4. Cutoff: 500 Hz
5. Resonance: 60%
6. Filter Envelope:
   - A: 0, D: 200, S: 20%, R: 50
   - Amount: +40
7. Amp Envelope:
   - A: 0, D: 100, S: 80%, R: 20
8. Macro 1 に Filter Cutoff をマッピング
9. テスト: C2 で16分音符パターン

確認:
Filter Cutoffを動かすと「ピコピコ」
Resonanceで独特の音色
```

### 基本パッチ3: Simple Lead

```
手順:
1. Wavetableを追加
2. OSC 1: Basic Shapes > Saw
3. OSC 2: Basic Shapes > Square
   - Detune: +7 cent
   - Volume: 60%
4. Unison: 4 voices, Detune 10%
5. Filter: Low Pass 12 dB
6. Cutoff: 2500 Hz
7. Resonance: 15%
8. Amp Envelope:
   - A: 10, D: 200, S: 70%, R: 300
9. LFO 1 → OSC 1 Pitch:
   - Waveform: Sine
   - Rate: 5 Hz
   - Depth: 5%

確認:
2 OSC Detuneで太い音
Unisonで広がり
LFOでビブラート
```

---

## よくある質問（FAQ）

**Q1: どの波形を使えばいいですか？**

```
A: 用途で選ぶ

迷ったら:
ベース: Saw（80%）、Square（20%）
リード: Saw（70%）、Square（30%）
パッド: Triangle（50%）、Saw（50%）
Sub: Sine（100%）

理由:
Saw: 倍音が多い → 明るい、削りやすい
Sine: 倍音なし → 純粋な低音
Square: 中空的 → Deep House風
Triangle: 柔らかい → パッド
```

**Q2: Filter Cutoffをどこに設定すればいいですか？**

```
A: 音域で決める

目安:
Sub Bass: 100-200 Hz
ベース: 300-800 Hz
リード: 1500-4000 Hz
パッド: 1000-2000 Hz

調整方法:
1. Cutoffを最大（20000 Hz）にする
2. ゆっくり下げる
3. 「これだ！」という点で止める
4. 微調整
```

**Q3: Envelopeの設定がうまくいきません**

```
A: 音の種類で設定を変える

短い音（ドラム、Pluck）:
A: 0 ms
D: 100-300 ms
S: 0%
R: 50 ms

持続音（ベース、リード）:
A: 0-50 ms
D: 100-500 ms
S: 70-100%
R: 100-500 ms

ゆっくりした音（パッド）:
A: 500-2000 ms
D: 1000 ms
S: 70%
R: 2000-5000 ms
```

**Q4: LFOはいつ使いますか？**

```
A: 動きが欲しい時

用途:
静的な音 → LFOなし
動きのある音 → LFO使用

例:
Sub Bass: LFOなし（安定が重要）
リード: LFO → Pitch（ビブラート）
パッド: LFO → Filter（ゆっくり変化）
FX: LFO → Pan（左右に動く）

初心者:
最初はLFOなしでOK
慣れたら追加
```

---

## 練習方法

### Week 1: 波形の理解

```
Day 1-2: 4つの波形を聴き比べ
1. Wavetable追加
2. 各波形でC3を演奏
3. 違いを体感

Day 3-4: Filterで削る
1. Saw波を選択
2. Cutoffを20000 Hz → 200 Hz
3. 音色の変化を聴く

Day 5-7: Resonanceの効果
1. Cutoff 500 Hz
2. Resonance 0% → 80%
3. 「ピコピコ」音を体感
```

### Week 2: Envelope練習

```
Day 1-2: Attack変化
1. Sub Bass作成
2. Attack 0 ms → 2000 ms
3. 立ち上がりの変化を聴く

Day 3-4: Release変化
1. Pluck音作成
2. Release 10 ms → 5000 ms
3. 余韻の変化を聴く

Day 5-7: Filter Envelope
1. Pluck音にFilter Envelope追加
2. Envelope Amount 0 → +60
3. 音色の時間的変化を聴く
```

### Week 3: LFO練習

```
Day 1-2: ビブラート
1. リード作成
2. LFO → Pitch
3. Rate、Depth調整

Day 3-4: ワウワウ
1. ベース作成
2. LFO → Filter Cutoff
3. Waveform変更（Sine、Triangle、Square）

Day 5-7: オートパン
1. パッド作成
2. LFO → Pan
3. Rate 1/8（テンポ同期）
```

### Week 4: 総合練習

```
Day 1-2: 基本パッチ10個作成
1. Sub Bass
2. Techno Bass
3. Deep House Bass
4. Simple Lead
5. Pluck
6. Simple Pad
7. FX (Riser)
8. FX (Impact)
9. Bass with LFO
10. Pad with LFO

Day 3-7: ジャンル別音作り
参考曲を選び、特定の音を再現
50%似たら成功
```

---

## まとめ

シンセシスの基礎は、4つの要素で構成されます。

**4つの要素:**

1. **Oscillator**: 音の源、波形選択
2. **Filter**: 音色を削る、Cutoff/Resonance
3. **Envelope**: 時間的変化、ADSR
4. **LFO**: 周期的変化、Rate/Depth

**重要ポイント:**

- **Saw波**: 最も汎用的、ベース・リードの基本
- **Filter Cutoff**: サウンドの80%を決める最重要パラメータ
- **Amp Envelope**: 全ての音に必須
- **LFO**: 動きのある音を作る

**学習順序:**
1. 波形の違いを聴き比べ（Sine、Saw、Square、Triangle）
2. Filter Cutoffで音色を削る練習
3. Amp Envelopeで時間的変化を作る
4. LFOで周期的変化を追加

**次のステップ:** [Wavetable Sound Design](./wavetable-sound-design.md) へ進む

---

## 関連ファイル

- **[Wavetable Sound Design](./wavetable-sound-design.md)** - Wavetableでの実践的音作り
- **[03-instruments/wavetable.md](../03-instruments/wavetable.md)** - Wavetable詳細
- **[00-fundamentals/audio-basics.md](../../00-fundamentals/audio-basics.md)** - 音響基礎

---

**シンセシスの基礎をマスターして、自分だけの音を作りましょう！** 🎹

---

## 発展編: シンセシス方式の詳細解説

### 減算合成（Subtractive Synthesis）の深堀り

減算合成は最も歴史が長く、直感的に理解しやすいシンセシス方式です。1960年代のMoog SynthesizerやARP 2600に始まり、現在もほとんどのソフトウェアシンセの基盤となっています。

```
減算合成の歴史と代表機種:

1960年代:
- Moog Modular（1964）: 世界初の商業的シンセサイザー
- ARP 2600（1971）: セミモジュラーの名機
- Minimoog Model D（1970）: ポータブルシンセの革命

1970-80年代:
- Sequential Circuits Prophet-5（1978）: 初のポリフォニックプログラマブルシンセ
- Roland Jupiter-8（1981）: アナログポリの最高峰
- Oberheim OB-Xa（1981）: 豊かなパッドサウンド
- Roland SH-101（1982）: ベースマシンの定番
- Roland TB-303（1981）: アシッドベースの原点

1990年代以降（ソフトウェア）:
- Native Instruments Massive（2007）: ウェーブテーブル + 減算
- u-he Diva（2011）: アナログモデリングの最高峰
- Arturia V Collection: クラシックシンセの再現
- Ableton Wavetable（2018）: 減算合成の現代版

現在のDAW内蔵シンセ:
- Ableton Analog: 純粋な減算合成
- Logic Pro Retro Synth: マルチ方式
- FL Studio 3xOsc: 入門向け減算
```

**減算合成のシグナルフローの詳細:**

```
完全なシグナルフロー:

OSC 1 ──┐
         ├── Mixer ── Filter 1 ── Filter 2 ── AMP ── FX ── Output
OSC 2 ──┤                 ↑           ↑        ↑
         │            Filter Env   LFO 2    Amp Env
OSC 3 ──┤
         │
Noise ───┘

モジュレーション:
LFO 1 → OSC Pitch（ビブラート）
LFO 2 → Filter Cutoff（ワウワウ）
LFO 3 → AMP Volume（トレモロ）
Pitch Env → OSC Pitch
Filter Env → Filter Cutoff
Amp Env → AMP Volume
Mod Wheel → Filter Cutoff / LFO Depth
Velocity → Filter Cutoff / AMP Volume

ルーティング可能なパラメータ:
- OSC: Pitch、Waveform Position、PWM、Level
- Filter: Cutoff、Resonance、Drive
- AMP: Volume、Pan
- FX: Wet/Dry、パラメータ各種
```

**フィルタードライブ（Filter Drive）の効果:**

```
Filter Drive とは:
フィルターの入力にゲインを加え、
倍音を付加するディストーション効果

Drive量による変化:
0%:   クリーン、変化なし
10%:  わずかなサチュレーション、温かみ
30%:  明確なオーバードライブ、中域が太くなる
50%:  強いディストーション、攻撃的
80%:  極端な歪み、ノイジー
100%: 完全にクリップ、破壊的

Drive の種類:
- Soft Clip: 滑らかなサチュレーション（アナログ風）
- Hard Clip: シャープなクリッピング（デジタル風）
- Fold: 波形を折り返す（独特の倍音）
- Asymmetric: 非対称クリッピング（真空管風）

実践的な使い方:
Techno Bass:
  Drive 20-40%、Filter Cutoff 低め
  → 太く攻撃的なベースに

Acid Bass:
  Drive 50-70%、Resonance 60%+
  → TB-303風のスクリーミングベースに

Warm Pad:
  Drive 5-10%、Soft Clip
  → アナログ風の温かさを追加
```

### 加算合成（Additive Synthesis）の詳細

加算合成は減算合成とは正反対のアプローチをとります。純粋なSine波を何百本も重ね合わせることで、どんな音色でも理論的に再現可能です。

```
加算合成の基本原理:

フーリエの定理:
「あらゆる周期波形は、Sine波の重ね合わせで表現できる」

例: Saw波の再構成
基音（f）       : sin(2πft)           × 1.000
第2倍音（2f）   : sin(2π × 2f × t)   × 0.500
第3倍音（3f）   : sin(2π × 3f × t)   × 0.333
第4倍音（4f）   : sin(2π × 4f × t)   × 0.250
第5倍音（5f）   : sin(2π × 5f × t)   × 0.200
第6倍音（6f）   : sin(2π × 6f × t)   × 0.167
...
第n倍音（nf）   : sin(2π × nf × t)   × 1/n

Sine波の数と精度:
  8本:  大まかなSaw波（まだギザギザ）
 16本:  かなり近い（ほとんどのケースで十分）
 32本:  ほぼ完璧
 64本:  聴覚的に区別不可能
256本:  理論的に完璧

例: Square波の再構成（奇数倍音のみ）
基音（f）       : sin(2πft)           × 1.000
第3倍音（3f）   : sin(2π × 3f × t)   × 0.333
第5倍音（5f）   : sin(2π × 5f × t)   × 0.200
第7倍音（7f）   : sin(2π × 7f × t)   × 0.143
第9倍音（9f）   : sin(2π × 9f × t)   × 0.111
→ 偶数倍音（2, 4, 6...）は振幅0
```

**加算合成の代表的なシンセサイザー:**

```
ハードウェア:
- Kawai K5（1987）: 初期の加算合成シンセ
- Technos Acxel（1988）: 32パーシャル
- Kurzweil K150（1986）: 加算合成エンジン搭載

ソフトウェア:
- Razor（Native Instruments）: リアルタイム加算合成
- Harmor（Image-Line）: 加算+リシンセシス
- Alchemy（Logic Pro内蔵）: 加算+スペクトラル
- Loom II（AIR Music）: 加算合成の入門に最適

加算合成の長所:
+ 理論的にどんな音色でも再現可能
+ 各倍音を個別にコントロール可能
+ スペクトラルモーフィングが自然
+ リシンセシス（サンプルの分析・再合成）が可能

加算合成の短所:
- CPU負荷が高い
- パラメータが膨大
- 直感的な操作が難しい
- リアルタイム制御が複雑
```

**加算合成での音作りアプローチ:**

```
Step 1: 基音を設定
  倍音 1（基音）を最大振幅に設定
  → ピッチの基準

Step 2: 倍音構造を決定
  明るい音: 高次倍音を多く
  暗い音: 高次倍音を少なく
  中空的な音: 偶数倍音を減らす
  金属的な音: 非整数倍音を追加

Step 3: 各倍音にエンベロープを設定
  実際の楽器のように、各倍音が異なる時間変化を持つ
  例: ピアノ
    低次倍音: ゆっくり減衰
    高次倍音: 速く減衰
    → 弾いた瞬間は明るく、徐々に暗くなる

Step 4: 時間的変化を追加
  スペクトラルエンベロープ:
    アタック時: 全倍音が急速に立ち上がる
    ディケイ時: 高次倍音から順に減衰
    サステイン時: 低次倍音のみ残る
    リリース時: 残った倍音がゆっくり消える
```

### FM合成（Frequency Modulation Synthesis）の詳細

FM合成は1973年にスタンフォード大学のJohn Chowning教授が発見した合成方式です。Yamaha DX7（1983）で一世を風靡しました。

```
FM合成の基本原理:

キャリア（Carrier）:
  実際に聞こえる音を出すオシレーター

モジュレータ（Modulator）:
  キャリアの周波数を変調するオシレーター

数式:
  y(t) = A × sin(2π × fc × t + I × sin(2π × fm × t))

  fc: キャリア周波数
  fm: モジュレータ周波数
  I: モジュレーションインデックス（変調の深さ）
  A: 振幅

モジュレーションインデックスによる変化:
  I = 0:   純粋なSine波（変調なし）
  I = 1:   わずかな倍音追加（柔らかいベル音）
  I = 3:   中程度の倍音（エレピ風）
  I = 5:   豊かな倍音（ブラス風）
  I = 10:  非常に複雑な倍音（金属的）
  I = 20+: ノイジーな音（SFX）

キャリアとモジュレータの周波数比:
  1:1  → 全倍音（Saw波に似る）
  1:2  → 偶数倍音強調（クラリネット風）
  1:3  → 奇数倍音パターン
  1:4  → 高次倍音に特徴
  2:1  → オクターブ上の成分
  3:2  → 5度の関係
  1:√2 → 非整数比、金属的な音

非整数比の効果:
  整数比: 調和的、楽器的な音
  非整数比: 非調和的、金属的、ベル音
  例: 1:1.41（√2）→ ゴーンという鐘の音
```

**FM合成のアルゴリズム（DX7の例）:**

```
DX7は6つのオペレーター（OP）と32のアルゴリズム:

代表的なアルゴリズム:

Algorithm 1（直列）:
  OP6 → OP5 → OP4 → OP3 → OP2 → OP1 → Output
  → 非常に複雑な倍音構造

Algorithm 5（並列+直列）:
  OP6 → OP5 ─┐
  OP4 → OP3 ─┼→ Output
  OP2 → OP1 ─┘
  → 3つの独立したFMペア

Algorithm 32（全並列）:
  OP6 ─┐
  OP5 ─┤
  OP4 ─┤
  OP3 ─┼→ Output
  OP2 ─┤
  OP1 ─┘
  → 6つのSine波の加算合成

フィードバック:
  オペレーターが自分自身を変調
  OP → OP（自己参照）
  → フィードバック量でSine波→Saw波に変化
  → ノイズ生成にも使用

現代のFMシンセ:
- Ableton Operator: 4オペレーター、11アルゴリズム
- Native Instruments FM8: 6オペレーター、自由ルーティング
- Dexed（無料）: DX7完全互換
- Arturia DX7 V: DX7モデリング + 拡張機能
```

**FM合成での実践的な音作り:**

```
エレクトリックピアノ（DX7 E.Piano風）:

Operator設定:
  OP1（Carrier）: Ratio 1.00、Level 99
  OP2（Modulator）: Ratio 1.00、Level 70

  OP2 Envelope:
    Rate 1: 99（即座にピーク）
    Rate 2: 60（中速で減衰）
    Rate 3: 30（ゆっくり減衰）
    Rate 4: 10（リリース）
    Level 1: 99
    Level 2: 75
    Level 3: 50
    Level 4: 0

  結果: アタック時に明るく、徐々に暗くなる
  → Velocity感度を追加すると、強く弾くほど明るくなる

ベル音:
  OP1（Carrier）: Ratio 1.00
  OP2（Modulator）: Ratio 3.50（非整数比）
  Mod Index: 高め
  → 非調和倍音でベルのような金属的な響き

FM Bass:
  OP1（Carrier）: Ratio 1.00
  OP2（Modulator）: Ratio 1.00
  Mod Index: Envelope制御
    アタック時: 高い（明るい）
    減衰後: 低い（暗い）
  → パンチのあるベースサウンド
```

### ウェーブテーブル合成（Wavetable Synthesis）の詳細

ウェーブテーブル合成は、複数の波形を格納したテーブルを持ち、テーブル内の位置を移動することで音色を連続的に変化させる方式です。

```
ウェーブテーブルの構造:

ウェーブテーブル = 複数の波形フレームの集合

例: 256フレームのウェーブテーブル
  Frame 0:   Sine波
  Frame 64:  Sine + 少し倍音
  Frame 128: Saw波に近い
  Frame 192: 複雑な波形
  Frame 255: ノイズに近い

Position（テーブルポジション）:
  0%: Frame 0（Sine波）
  25%: Frame 64
  50%: Frame 128（Saw波に近い）
  75%: Frame 192
  100%: Frame 255（ノイズに近い）

補間:
  Frame間は滑らかに補間される
  → 連続的な音色変化が可能

特徴:
+ 減算合成ではできない音色変化が可能
+ テーブルポジションをLFO/Envelopeで動かせる
+ 複雑な倍音構造を簡単に実現
+ 視覚的に波形を確認できる
```

**ウェーブテーブルの作成方法:**

```
方法1: プリセットウェーブテーブル
  DAW/シンセに内蔵されたテーブルを使用
  → 最も簡単、初心者向け

方法2: 波形の描画
  Serum、Wavetable等のエディタで手描き
  → カスタム波形を作成

方法3: サンプルからの変換
  録音した音をウェーブテーブルに変換
  スペクトラル分析 → フレーム化
  → ユニークな音色を作成

方法4: 数式による生成
  数学的な波形を計算して生成
  例: スーパーフォーミュラ、フラクタル波形
  → 実験的な音色

方法5: リシンセシス
  既存の音をスペクトラル分析し、
  ウェーブテーブルとして再構成
  → 既存の音を新しい方法で変形

Serum でのウェーブテーブル作成:
1. OSC パネルの鉛筆アイコンをクリック
2. Wavetable Editor を開く
3. Import > Audio File で音声ファイルを読み込み
4. または手描きで波形を作成
5. フレーム数を設定（通常64-256）
6. モーフィングモードを選択
7. Export でテーブルを保存
```

**ウェーブテーブルのモジュレーション:**

```
テーブルポジションの動的制御:

LFO → Table Position:
  Rate: 0.5 Hz（ゆっくり）
  Depth: 50%
  → 音色がゆっくり変化するパッド向き

Envelope → Table Position:
  A: 0 ms、D: 500 ms、S: 30%、R: 200 ms
  Amount: +60
  → アタック時に明るく、減衰後に暗くなる

Macro → Table Position:
  Macroノブでリアルタイム制御
  → ライブパフォーマンスに最適

Velocity → Table Position:
  強く弾く → ポジション高い（明るい音色）
  弱く弾く → ポジション低い（暗い音色）
  → 演奏表現力が向上

Auto → Table Position:
  テンポ同期で自動変化
  1/4拍ごとに位置が変わる
  → リズミカルな音色変化
```

### グラニュラー合成（Granular Synthesis）の詳細

グラニュラー合成は、音声サンプルを微小な「粒（グレイン）」に分解し、それらを再構成することで新しい音を生み出す方式です。

```
グラニュラー合成の基本概念:

グレイン（Grain）とは:
  音声サンプルの極めて短い断片
  通常 1ms - 100ms の長さ

  各グレインのパラメータ:
  - Position（サンプル内の読み取り位置）
  - Size（グレインの長さ）
  - Pitch（ピッチ）
  - Pan（パンニング）
  - Amplitude（音量）
  - Envelope（窓関数）

窓関数（Window Function）:
  各グレインにかけるエンベロープ形状

  Hann窓:     /‾\      滑らかな立ち上がり・減衰
  Hamming窓:  /‾‾\     Hannに似るがわずかに異なる
  Gaussian窓: /‾\      ベル型、最も滑らか
  Triangle窓: /\       三角形、シンプル
  Rectangle窓: |‾|     矩形、クリック音が出やすい

  推奨: Hann窓またはGaussian窓（クリック音が少ない）

グレインサイズの効果:
  1-5 ms:   非常に短い → ノイジーな質感
  5-20 ms:  短い → ざらついた質感、テクスチャ的
  20-50 ms: 中程度 → 原音の特徴が残る
  50-100 ms: 長い → 原音がほぼ聞こえる
  100+ ms:  非常に長い → サンプリングに近い

グレイン密度（Density）:
  密度 = 同時に鳴るグレインの数
  1-5:    疎ら → ぽつぽつした音
  10-20:  中程度 → テクスチャ的
  50-100: 密 → 連続的な音
  200+:   非常に密 → 滑らかなドローン
```

**グラニュラー合成の代表的なシンセサイザー:**

```
ソフトウェア:
- Granulator II（Ableton Max for Live、無料）
- Quanta（Audio Damage）: モダンなグラニュラーシンセ
- Padshop（Steinberg）: グラニュラー + スペクトラル
- Grain（Native Instruments Reaktor）
- Clouds（Mutable Instruments → VCV Rack）
- Ribs（無料、Max/MSP）

ハードウェア:
- Mutable Instruments Clouds: Eurorackの定番グラニュラー
- Tasty Chips GR-1: 専用グラニュラーシンセ
- 1010music Blackbox: サンプラー + グラニュラー

グラニュラー合成の特徴的な用途:
1. タイムストレッチ:
   ピッチを変えずに速度を変更
   → DAWのワープ機能の基盤技術

2. ピッチシフト:
   速度を変えずにピッチを変更
   → ボーカルのキー変更に使用

3. テクスチャ生成:
   日常音をドローン/パッドに変換
   → Ambient音楽に最適

4. フリーズ効果:
   特定の瞬間を無限に持続
   → サウンドスケープ、DJ効果

5. スクラブ:
   再生位置を手動/自動で移動
   → 実験的サウンドデザイン
```

### 物理モデリング合成（Physical Modeling Synthesis）の詳細

物理モデリングは、楽器の物理的な振動メカニズムを数学的にシミュレートする方式です。

```
物理モデリングの基本原理:

実際の楽器の仕組みを数式でモデル化:

弦楽器モデル:
  1. 励起（Excitation）: 弦を弾く/弓で擦る
  2. 振動体（Resonator）: 弦の振動
  3. 減衰（Damping）: 空気抵抗、内部摩擦
  4. 共鳴体（Body）: ボディの共鳴

管楽器モデル:
  1. 励起: 息を吹き込む/リード振動
  2. 管の共鳴: 管内の定在波
  3. 放射: 管の開口部からの音の放射
  4. フィードバック: 管→リードの相互作用

打楽器モデル:
  1. 励起: マレットで叩く
  2. 膜/板の振動: 2次元振動
  3. 減衰: 材質に依存
  4. 共鳴体: ボディ/シェル

Karplus-Strong アルゴリズム:
  弦楽器の最もシンプルなモデル

  手順:
  1. ノイズバースト（励起）を生成
  2. ディレイラインに送る
  3. ディレイ出力をローパスフィルターに通す
  4. フィルター出力をディレイ入力にフィードバック
  5. 繰り返し → 弦の振動をシミュレート

  パラメータ:
  - ディレイ長 = 1 / 基本周波数（ピッチを決定）
  - フィルターのカットオフ = 減衰の速さ
  - ノイズの長さ = アタックの特性
  - フィードバック量 = サステインの長さ
```

**物理モデリングの代表的なシンセサイザー:**

```
ソフトウェア:
- Ableton Collision: 打楽器物理モデリング
- Ableton Tension: 弦楽器物理モデリング
- Ableton Electric: エレクトリックピアノモデリング
- Applied Acoustics Chromaphone 3: マルチ物理モデリング
- Applied Acoustics Strum GS-2: ギター物理モデリング
- Modartt Pianoteq: ピアノ物理モデリングの最高峰
- Audio Modeling SWAM: 管楽器/弦楽器

ハードウェア:
- Yamaha VL1（1994）: 初の商用物理モデリングシンセ
- Korg Z1（1997）: マルチモデリングエンジン
- Roland V-Piano: ピアノ物理モデリング

利点:
+ 非常にリアルな楽器音
+ パラメータが物理的に意味を持つ
+ CPU効率が良い（サンプルベースに比べ）
+ 演奏表現力が高い

制約:
- 既存楽器の模倣が主な用途
- 新しい音色の発見がやや難しい
- パラメータの相互関係が複雑
- 設計に高度な数学知識が必要
```

---

## 発展編: フィルター設計の深堀り

### フィルターの数学的背景

```
フィルターの基本特性:

周波数応答（Frequency Response）:
  入力周波数に対する出力振幅の関係

  Low Pass Filter（理想的）:
  ゲイン
  |
  |  1.0 ___________
  |                 |
  |                 |
  |  0.0            |_________
  |__________________|_________ 周波数
                   Cutoff

  実際のフィルター:
  ゲイン
  |
  |  1.0 ___________
  |                 \
  |                  \
  |  0.0              \_______
  |__________________|_________ 周波数
                   Cutoff

  → 理想的な垂直カットは不可能
  → スロープ（傾斜）で表現

スロープ（Slope）:
  単位: dB/octave（dB/oct）

  6 dB/oct（1-pole）:
    非常に緩やか、ほとんど削れない
    → ティルトEQ的な用途

  12 dB/oct（2-pole）:
    緩やか、柔らかいフィルタリング
    → パッド、ストリングス向き

  18 dB/oct（3-pole）:
    TB-303のフィルター
    → 独特のアシッドサウンド

  24 dB/oct（4-pole）:
    急峻、しっかりカット
    → Moog式フィルター、ベース向き

  36 dB/oct（6-pole）:
    非常に急峻
    → デジタルシンセで利用可能

  48 dB/oct（8-pole）:
    ほぼ壁のようなカット
    → 特殊用途
```

### フィルターの種類（拡張）

```
State Variable Filter（SVF）:
  1つのフィルター回路から LP、HP、BP、Notch を同時出力
  → 多くのシンセで採用
  → モーフィング（LP↔BP↔HP）が可能

Ladder Filter（ラダーフィルター）:
  Moog が発明した4-pole（24 dB/oct）フィルター
  特徴: 温かみのある音、低域が太い
  自己発振時に純粋なSine波を生成
  → クラシックなアナログサウンド

Diode Ladder Filter:
  Roland TB-303 で使用された 3-pole（18 dB/oct）
  特徴: Moog ラダーとは異なる独特のキャラクター
  → アシッドベースの要

SEM Filter:
  Oberheim SEM で使用された 2-pole（12 dB/oct）SVF
  特徴: 開放的、明るい音色
  → パッド、リード向き

Comb Filter（コムフィルター）:
  短いディレイを使ったフィードバックフィルター
  特徴: 周期的な山と谷ができる
  → フランジャー効果、金属的な音色
  → 物理モデリングの基盤技術

Formant Filter:
  母音（あ、い、う、え、お）の共鳴特性を模倣
  → ボコーダー的な効果
  → トーキングベース、ボーカルシンセ
```

### フィルターキートラッキング

```
Filter Key Tracking とは:
  演奏するノートの高さに合わせてCutoffを自動調整

なぜ必要か:
  Cutoff固定の場合:
    低いノート → フィルター開きすぎ（明るすぎ）
    高いノート → フィルター閉じすぎ（暗すぎ）

  Key Tracking有効:
    低いノート → Cutoff低い（適切な明るさ）
    高いノート → Cutoff高い（適切な明るさ）
    → 全音域で一貫した音色

設定:
  0%:   Key Trackingなし（Cutoff固定）
  50%:  半分追従（やや追従）
  100%: 完全追従（ノートとCutoffが1:1）
  150%: 過剰追従（高音ほどより開く）

用途:
  ベース: 0-30%（低域を均一に）
  リード: 50-100%（全域で明るさ統一）
  パッド: 30-50%
  アシッド: 0%（固定Cutoffで個性を出す）
```

---

## 発展編: ADSR応用テクニック

### マルチステージエンベロープ

```
ADSR の限界:
  4段階の変化しかできない
  → 複雑な音色変化には不十分

マルチステージエンベロープ（MSEG）:
  自由な数のステージを持つエンベロープ

  例: 6ステージエンベロープ
  レベル
  |
  |     /\
  |    /  \    /‾‾‾‾\
  |   /    \  /      \
  |  /      \/        \___
  | /                      \
  |/                        \___
  |________________________________ 時間
  S1  S2  S3   S4    S5   S6

  S1: Initial Attack（急速な立ち上がり）
  S2: Overshoot（ピークを超える）
  S3: Dip（一度下がる）
  S4: Secondary Peak（二度目のピーク）
  S5: Sustain Phase（持続）
  S6: Release（減衰）

Serum MSEG の活用:
  最大16ポイントのエンベロープ
  カーブ形状を各区間で変更可能
  テンポ同期対応
  → リズミカルな音色変化
  → サイドチェイン風のダッキング
  → 複雑なフィルタースウィープ
```

### エンベロープのカーブ形状

```
リニア（Linear）カーブ:
  |     /
  |    /
  |   /
  |  /
  | /
  |/________
  → 均一な速度で変化
  → デジタル的、正確

エクスポネンシャル（Exponential）カーブ:
  |           ___
  |         _/
  |       _/
  |     _/
  |   _/
  |__/________
  → 最初は遅く、後から速く
  → 自然な立ち上がり（パッド向き）

ログリズミック（Logarithmic）カーブ:
  |    ___
  |   /
  |  /
  | /
  |/
  |__________
  → 最初は速く、後から遅く
  → パーカッシブなアタック向き

S字カーブ:
  |        ___
  |      _/
  |    _/
  |   /
  |__/
  |__________
  → 中間が最も速い
  → 滑らかな遷移

カーブの選択基準:
  Attack:
    ベース/ドラム: ログリズミック（即座に立ち上がる）
    パッド: エクスポネンシャル（ゆっくり立ち上がる）
    リード: リニア（均一な立ち上がり）

  Decay:
    Pluck: エクスポネンシャル（自然な減衰）
    ベース: ログリズミック
    パッド: リニア

  Release:
    短い余韻: ログリズミック
    長い余韻: エクスポネンシャル
    標準: リニア
```

### Velocity（ベロシティ）との連携

```
Velocity = 鍵盤を押す強さ（0-127）

Velocity → Amp:
  強く弾く → 大きい音
  弱く弾く → 小さい音
  → 基本中の基本、ほぼ全ての場合で設定

Velocity → Filter Cutoff:
  強く弾く → Cutoff 開く（明るい音）
  弱く弾く → Cutoff 閉じる（暗い音）
  → ピアノ的な表現、非常に有効

Velocity → Envelope Amount:
  強く弾く → エンベロープの効き大
  弱く弾く → エンベロープの効き小
  → ダイナミクスに応じた音色変化

Velocity → Attack Time:
  強く弾く → 短いアタック
  弱く弾く → 長いアタック
  → ストリングスやパッドで有効

Velocity → Wavetable Position:
  強く弾く → 明るい波形位置
  弱く弾く → 暗い波形位置
  → ウェーブテーブルシンセで特に有効

設定の目安:
  Velocity Sensitivity: 0-100%
  0%: Velocity無視（オルガン的）
  50%: 中程度の反応
  100%: 最大反応（ピアノ的）

  通常は60-80%に設定（自然な反応）
```

---

## 発展編: ユニゾン・デチューンとボイシング

### ユニゾン（Unison）の詳細

```
ユニゾンとは:
  1つのノートに対して複数のボイス（声）を重ねる機能
  → 音が太く、広がりのあるサウンドに

ボイス数による変化:
  1 Voice:   モノフォニック、細い音
  2 Voices:  わずかに太い
  4 Voices:  明確に太い（最も一般的）
  8 Voices:  非常に太い（スーパーソウ）
  16 Voices: 極端に太い（壁のような音）
  32 Voices: 密集したサウンド

デチューン（Detune）量の目安:
  0%:    全ボイスが同じピッチ（効果なし）
  5%:    微妙な揺れ（コーラス的）
  10%:   はっきりした厚み（EDMリード）
  20%:   強いデチューン（トランスリード）
  30%:   非常に強い（ハードスタイル）
  50%+:  極端（特殊効果）

スプレッド（Stereo Spread）:
  各ボイスのパン位置を左右に分散
  0%: 全ボイスがセンター
  50%: 適度な広がり
  100%: 左右いっぱいに広がる
  → ヘッドフォンで聴くと立体的に

実践設定例:

EDM スーパーソウリード:
  Voices: 8
  Detune: 15%
  Spread: 80%
  → 大きく広がる太いリード

Trance Lead:
  Voices: 4
  Detune: 20%
  Spread: 60%
  → クラシックなトランスリード

Techno Stab:
  Voices: 2
  Detune: 5%
  Spread: 40%
  → タイトなスタブサウンド

Ambient Pad:
  Voices: 16
  Detune: 8%
  Spread: 100%
  → 壁のように広がるパッド
```

### ボイシング（Voicing）モード

```
ポリフォニック（Polyphonic）:
  複数のノートを同時に演奏可能
  コード演奏に必要
  一般的なボイス数: 8-64
  → パッド、コード、ほとんどの用途

モノフォニック（Monophonic）:
  1つのノートしか鳴らない
  → ベースライン、リード

  レガート（Legato）モード:
    次のノートが押されている間に新しいノートを押すと、
    エンベロープがリトリガーされない
    → 滑らかなフレーズ、ポルタメント対応

  リトリガー（Retrigger）モード:
    新しいノートごとにエンベロープがリスタート
    → パーカッシブなベースライン

ポルタメント / グライド:
  ノート間のピッチを滑らかに接続

  Time: 0-2000 ms
  0 ms:   ピッチ変化なし（通常）
  50 ms:  わずかなスライド
  200 ms: 明確なスライド（303ベース風）
  500 ms: ゆっくりとしたグライド（リード）

  モード:
  - Always: 常にグライド
  - Legato: レガート演奏時のみグライド

  形状:
  - Linear: 直線的なグライド
  - Exponential: 最初は速く、後から遅い

ノートプライオリティ:
  モノフォニック時、複数ノートが重なった場合の優先順位
  - Last Note: 最後に押したノート（最も一般的）
  - High Note: 最高音ノート
  - Low Note: 最低音ノート
```

---

## 発展編: シンセルーティングとモジュレーションマトリクス

### モジュレーションマトリクスの概念

```
モジュレーションマトリクスとは:
  任意のソース（LFO、Envelope等）から
  任意のデスティネーション（Cutoff、Pitch等）への
  接続を自由に設定できるシステム

基本構造:
  Source（ソース）  →  Amount（量）  →  Destination（行き先）

代表的なソース:
  - LFO 1, 2, 3...
  - Envelope 1, 2, 3...
  - Velocity（打鍵の強さ）
  - Aftertouch（打鍵後の圧力）
  - Mod Wheel（モジュレーションホイール）
  - Pitch Bend
  - Key Track（鍵盤の位置）
  - Random（ランダム値）
  - Macro 1-8
  - MIDI CC 各種

代表的なデスティネーション:
  - OSC 1/2 Pitch
  - OSC 1/2 Level
  - OSC 1/2 Wavetable Position
  - Filter Cutoff
  - Filter Resonance
  - AMP Level
  - Pan
  - LFO Rate / Depth
  - FX Send / パラメータ
  - Unison Detune
  - Noise Level

Amount（モジュレーション量）:
  -100% ～ +100%
  正の値: ソースが増加 → デスティネーションも増加
  負の値: ソースが増加 → デスティネーションは減少
  0%: モジュレーションなし
```

### 実践的なモジュレーション設定

```
設定例1: エクスプレッシブリード
  Mod Matrix:
  1. LFO 1 → OSC Pitch      Amount: +5%  （ビブラート）
  2. Mod Wheel → LFO 1 Depth Amount: +100% （Modで深さ制御）
  3. Velocity → Filter Cutoff Amount: +40%  （強弱で音色変化）
  4. Aftertouch → Filter Cutoff Amount: +20% （押し込みで音色変化）
  5. Key Track → Filter Cutoff Amount: +60%  （高音ほど明るく）

  効果:
  - Mod Wheelを上げるとビブラートが深くなる
  - 強く弾くと音が明るくなる
  - 鍵盤を押し込むとさらに明るくなる
  → ライブ演奏に最適な設定

設定例2: 進化するパッド
  Mod Matrix:
  1. LFO 1 → Filter Cutoff    Amount: +30%  （ゆっくりフィルター変化）
  2. LFO 2 → Wavetable Pos    Amount: +50%  （波形変化）
  3. LFO 3 → Pan              Amount: +60%  （左右移動）
  4. Envelope 2 → Filter Cutoff Amount: +40% （時間的変化）
  5. LFO 1 → LFO 2 Rate       Amount: +20%  （LFOの速度が変化）

  LFO設定:
  LFO 1: Sine、0.1 Hz（非常にゆっくり）
  LFO 2: Triangle、0.3 Hz
  LFO 3: Sine、0.2 Hz

  効果:
  - 音色がゆっくりと複雑に変化し続ける
  - 同じノートを弾いても常に異なるニュアンス
  → Ambient、チルアウトに最適

設定例3: リズミカルなベース
  Mod Matrix:
  1. LFO 1 → Filter Cutoff    Amount: +60%
  2. LFO 1 → AMP Level        Amount: +30%
  3. Velocity → Filter Env Amt Amount: +50%

  LFO設定:
  LFO 1: Square（カクカク）、1/16 テンポ同期

  効果:
  - 16分音符でフィルターとボリュームが変化
  - シーケンサー的なベースパターン
  → Techno、House のベースラインに
```

### Macro（マクロ）コントロール

```
Macroとは:
  1つのノブで複数のパラメータを同時に制御する機能
  ライブパフォーマンスや直感的な操作に最適

Ableton Wavetable の Macro 設定:
  Macro 1-8 に自由にパラメータをアサイン可能

設定例:

Macro 1「Brightness（明るさ）」:
  → Filter Cutoff: +80%
  → Wavetable Position: +40%
  → Filter Resonance: +20%

  効果: ノブを回すだけで音が明→暗に変化

Macro 2「Movement（動き）」:
  → LFO 1 Depth: +100%
  → LFO 2 Depth: +60%
  → Chorus Mix: +40%

  効果: ノブを上げると音に動きが加わる

Macro 3「Fatness（太さ）」:
  → Unison Voices: 1→8
  → Unison Detune: +50%
  → Sub OSC Level: +80%
  → Drive: +30%

  効果: ノブを回すと音がどんどん太くなる

Macro 4「Space（空間）」:
  → Reverb Send: +100%
  → Delay Send: +60%
  → Stereo Spread: +80%
  → High Cut: -30%

  効果: ノブを上げると音が遠くに広がる

MIDIコントローラーへのマッピング:
  Macro → MIDI CC → 物理ノブ
  → ハードウェアノブでリアルタイム制御
  → DJパフォーマンス中の音色変化に最適
```

---

## 発展編: 音作りの体系的アプローチ

### トップダウンアプローチ

```
Step 1: リファレンスを選ぶ
  目標の音を明確にする

  方法:
  - 好きな曲から特定の音を選ぶ
  - 言葉で表現する（太い、明るい、暗い、鋭い等）
  - 似た音のプリセットを見つける

Step 2: 音を分析する
  周波数帯域:
    低域が強い？ → ベース系
    中域が強い？ → リード系
    高域が強い？ → FX系

  時間的変化:
    アタックは速い？遅い？
    減衰はある？持続する？
    余韻は短い？長い？

  音色の特徴:
    明るい？暗い？
    鋭い？丸い？
    太い？細い？
    動きがある？静的？

Step 3: 要素を選択する
  分析結果から各要素を決定:

  OSC:
    明るい → Saw波
    暗い → Triangle波/Sine波
    中空的 → Square波
    複雑 → Wavetable/FM

  Filter:
    暗い → LP Cutoff低め
    明るい → LP Cutoff高め/HPF
    狭い → BP

  Envelope:
    パーカッシブ → 短いDecay、Sustain 0%
    持続的 → 長いSustain
    ゆっくり → 長いAttack

  LFO:
    動きあり → LFO追加
    静的 → LFOなし

Step 4: 作成と調整
  1. 基本パッチを作成（OSC + Filter + Amp Env）
  2. 大まかに調整
  3. Filter Envelopeを追加
  4. LFOを追加
  5. エフェクトを追加
  6. 微調整

Step 5: レファレンスと比較
  A/B比較:
  - レファレンスと自分の音を交互に聴く
  - 違いを特定
  - パラメータを調整
  - 80%似たら成功
```

### ボトムアップアプローチ（実験的音作り）

```
目的の音がない場合の探索方法:

方法1: ランダムパラメータ
  1. Init Patch（初期状態）から開始
  2. ランダムにパラメータを変更
  3. 面白い音が見つかったら保存
  4. そこから微調整

方法2: プリセットの分解
  1. 好きなプリセットを選ぶ
  2. 各パラメータを確認
  3. なぜその音になるか理解
  4. パラメータを変更して変化を観察
  5. 新しい方向に発展させる

方法3: 制約付き実験
  ルールを設定して実験:
  - OSC 1つだけで面白い音を作る
  - Filter Cutoff固定、他で変化させる
  - LFO 3つを同時使用
  - Envelope Amount を極端に設定

方法4: サウンドモーフィング
  1. 2つの異なるプリセットのパラメータをメモ
  2. 中間値で新しいパッチを作成
  3. 予想外の結果が面白い音になることが多い

方法5: レイヤリング
  複数のシンセ/パッチを重ねる:
  Layer 1: 低域担当（Sub Bass）
  Layer 2: 中域担当（主な音色）
  Layer 3: 高域担当（倍音、キャラクター）
  Layer 4: テクスチャ（ノイズ、グラニュラー）
  → 各レイヤーはシンプルでも合成結果は複雑に
```

### ジャンル別の音作り指針

```
Techno:
  Bass:
    OSC: Saw / Square
    Filter: LP 24dB, Cutoff低め, Resonance高め
    Envelope: 短いDecay、低いSustain
    Drive: 中程度
    → 暗く、パンチのあるベース

  Lead:
    OSC: Saw × 2（Detune）
    Filter: LP/BP、Cutoff中程度
    LFO → Filter: テンポ同期
    → 反復的、催眠的なリード

  Pad:
    OSC: Wavetable、Position動的
    Filter: LP、Cutoff低め
    Release長め
    Reverb多め
    → 暗く深い空間

House / Deep House:
  Bass:
    OSC: Square / Triangle
    Filter: LP 12dB、Cutoff中程度
    Envelope: 中程度のDecay
    → 丸く温かいベース

  Chord:
    OSC: Saw × 2
    Unison: 4 voices
    Filter: LP、Cutoff中〜高
    → 太いコードスタブ

  Pad:
    OSC: Triangle + Saw（低音量）
    Filter: LP、Cutoff低め
    Attack長め、Release長め
    → 温かく包み込むパッド

Drum & Bass / Dubstep:
  Bass:
    OSC: Wavetable / FM
    Filter: LP/BP、Cutoff動的
    LFO → Wavetable Position: 激しく
    Drive: 高め
    → 激しく動くベース（ワブルベース）

  Reese Bass:
    OSC: Saw × 2（大きくDetune）
    Filter: LP、Cutoff低め
    → うなるようなベース（Reese）

Trance:
  Lead:
    OSC: Saw × 2
    Unison: 8 voices、Detune大
    Filter: LP、Cutoff高め
    → 巨大なスーパーソウリード

  Pluck:
    OSC: Saw
    Filter: LP、短いFilter Envelope
    Reverb + Delay
    → キラキラしたプラック音

Ambient / Chill:
  Pad:
    OSC: Wavetable / Granular
    Filter: LP、非常に低いCutoff
    Attack: 2000ms+
    Release: 5000ms+
    LFO: 複数、非常にゆっくり
    Reverb: 大量
    → 広大で浮遊感のあるパッド

  Texture:
    Granular合成 + Reverb
    → 環境音的テクスチャ
```

---

## 発展編: 実践パッチ作成演習

### 演習1: クラシックTB-303アシッドベース

```
目標: Roland TB-303風のスクリーミングアシッドベース

手順:
1. シンセ: Wavetable（または Analog）

2. OSC設定:
   OSC 1: Saw波
   OSC 2: Off
   Octave: 0（ベース音域）

3. Filter設定:
   Type: Low Pass 18dB（可能なら）/ 24dB
   Cutoff: 300 Hz
   Resonance: 75%（高め）
   Drive: 30%

4. Filter Envelope:
   Attack: 0 ms
   Decay: 150 ms
   Sustain: 0%
   Release: 50 ms
   Envelope Amount: +60

5. Amp Envelope:
   Attack: 0 ms
   Decay: 200 ms
   Sustain: 60%
   Release: 30 ms

6. グライド設定:
   Portamento: On
   Time: 80 ms
   Mode: Legato

7. オートメーション:
   Cutoff を MIDI CC またはオートメーションで動かす
   → 「ウニョウニョ」するアシッドサウンド

確認ポイント:
- Resonanceを上げると「ピコピコ」音が強くなるか
- Cutoffを動かすと音色が劇的に変化するか
- グライドが滑らかに繋がるか
- 16分音符パターンでグルーヴが出るか
```

### 演習2: マッシブスーパーソウリード

```
目標: EDM/Trance で使われる巨大なリードサウンド

手順:
1. シンセ: Wavetable（または Serum）

2. OSC 1設定:
   Waveform: Saw
   Octave: 0

3. OSC 2設定:
   Waveform: Saw
   Octave: +1
   Detune: +7 cent

4. Unison設定:
   Voices: 8
   Detune: 15%
   Spread: 80%

5. Filter設定:
   Type: Low Pass 12dB
   Cutoff: 3500 Hz
   Resonance: 10%
   Key Track: 80%

6. Amp Envelope:
   Attack: 5 ms
   Decay: 300 ms
   Sustain: 75%
   Release: 400 ms

7. LFO設定:
   LFO 1 → OSC 1 Pitch
   Waveform: Sine
   Rate: 5.5 Hz
   Depth: 3%（微妙なビブラート）

8. エフェクト:
   Reverb: 20% Mix、Large Hall
   Delay: 1/8 dotted、15% Mix
   Chorus: 軽め

確認ポイント:
- ユニゾンで音が大きく広がっているか
- ステレオ感が十分か（ヘッドフォンで確認）
- コードを弾いても破綻しないか
- ビブラートが自然か
```

### 演習3: ディープダブテクノパッド

```
目標: 深く暗い、空間的なパッドサウンド

手順:
1. シンセ: Wavetable

2. OSC 1設定:
   Waveform: Triangle
   Octave: 0
   Volume: 100%

3. OSC 2設定:
   Waveform: Wavetable（Digital系）
   Position: 30%
   Octave: +1
   Detune: +15 cent
   Volume: 40%

4. Filter設定:
   Type: Low Pass 12dB
   Cutoff: 800 Hz
   Resonance: 15%

5. Amp Envelope:
   Attack: 1500 ms
   Decay: 2000 ms
   Sustain: 65%
   Release: 4000 ms

6. Filter Envelope:
   Attack: 500 ms
   Decay: 3000 ms
   Sustain: 30%
   Release: 2000 ms
   Amount: +35

7. LFO設定:
   LFO 1 → Filter Cutoff
   Waveform: Sine
   Rate: 0.15 Hz
   Depth: 25%

   LFO 2 → Wavetable Position（OSC 2）
   Waveform: Triangle
   Rate: 0.08 Hz
   Depth: 40%

   LFO 3 → Pan
   Waveform: Sine
   Rate: 0.2 Hz
   Depth: 30%

8. エフェクト:
   Reverb: 50% Mix、Large Plate
   Delay: 1/4 + Feedback 40%
   EQ: High Cut 6000 Hz

確認ポイント:
- 音がゆっくり立ち上がるか（1.5秒）
- 鍵盤を離した後も4秒間余韻があるか
- 音色がゆっくりと変化し続けるか
- 空間的な広がりがあるか
- Dub Techno の曲に合いそうか
```

### 演習4: FMベル音

```
目標: Yamaha DX7風のクリスタルベル

手順（Ableton Operator使用）:

1. アルゴリズム: Algorithm 1（直列、A→B→C→D）
   または OP2 → OP1 のシンプルな2オペレーター構成

2. Operator A（Carrier）:
   Ratio: 1.00（基音）
   Level: 90
   Envelope: A: 0、D: 3000 ms、S: 0%、R: 2000 ms

3. Operator B（Modulator）:
   Ratio: 3.50（非整数比 → ベル的な響き）
   Level: 70
   Envelope: A: 0、D: 1500 ms、S: 0%、R: 1000 ms

4. Operator C, D: Off（シンプルに保つ）

5. グローバル設定:
   Tone: 60%
   Time: 80%

6. エフェクト:
   Reverb: 40% Mix、Hall
   → 残響でベル感を強調

Wavetableで擬似的に再現する場合:
1. OSC 1: Sine波
2. OSC 2: Sine波、Ratio を非整数に
3. FM Amount を適度に設定
4. Filter: なし（バイパス）
5. Amp Env: A: 0、D: 2000、S: 0%、R: 1500

確認ポイント:
- 金属的な倍音が聞こえるか
- アタック時に明るく、徐々に暗くなるか
- ピッチによって倍音構造が変わるか
- Ratioを変えると音色がどう変わるか実験
```

### 演習5: グラニュラーアンビエントテクスチャ

```
目標: 環境音からドローン/テクスチャを生成

手順（Granulator II 使用）:

1. サンプル準備:
   任意の音声ファイルを用意
   推奨: 自然音、楽器音、街の音など
   長さ: 5秒以上

2. Granulator II にサンプルをドロップ

3. 基本パラメータ:
   Grain Size: 50 ms
   Spray（Position Randomization）: 30%
   Density: 30 grains/sec

4. ピッチ設定:
   Pitch: 0 st（オリジナルピッチ）
   Pitch Randomization: ±5 st
   → 各グレインのピッチがランダムに変化

5. 再生位置:
   File Position: 手動で設定
   Scan Speed: 非常に遅い（0.01×）
   → サンプル内をゆっくり移動

6. エンベロープ:
   Attack: 2000 ms
   Release: 5000 ms
   → 非常にゆっくりした立ち上がりと余韻

7. エフェクト:
   Reverb: 60% Mix
   Delay: Ping-pong、30% Mix
   → 広大な空間に配置

8. LFO 自動化:
   LFO → File Position: ゆっくり移動
   LFO → Grain Size: わずかに変化
   → 常に変化し続けるテクスチャ

確認ポイント:
- 原音の面影が残りつつ新しい音になっているか
- 音がぷつぷつ途切れていないか（密度調整）
- 長時間聴いても飽きない変化があるか
- Ambient の曲に使えそうか
```

### 演習6: Reese Bass（リースベース）

```
目標: Drum & Bass で多用されるうなるようなベース

手順:
1. シンセ: Wavetable / Serum / Massive

2. OSC 1設定:
   Waveform: Saw
   Octave: 0

3. OSC 2設定:
   Waveform: Saw
   Octave: 0
   Detune: +25 cent（大きめのデチューン）

4. Unison:
   Voices: 2（最小限）
   Detune: 5%（わずか）

5. Filter設定:
   Type: Low Pass 24dB
   Cutoff: 600 Hz
   Resonance: 20%

6. Amp Envelope:
   Attack: 0 ms
   Decay: 0 ms
   Sustain: 100%
   Release: 50 ms

7. LFO設定:
   LFO 1 → Filter Cutoff
   Waveform: Sine / Triangle
   Rate: 0.5 Hz（ゆっくり）
   Depth: 40%

8. ポストプロセッシング:
   Distortion: OTT または Saturator
   EQ: 200-500 Hz ブースト
   Multiband Compression
   → 音圧と存在感を追加

9. レイヤリング:
   Sub Layer: Sine波、-1 Octave
   → 低域の土台を追加

確認ポイント:
- 2つのSaw波の干渉で「うなり」が聞こえるか
- Cutoffを動かすと音色が変化するか
- 低域が十分に出ているか
- Drum & Bass のドラムパターンと合わせて確認
```

---

## トラブルシューティング

### よくある問題と解決策

```
問題1: 音が細い、薄い
原因と対策:
  → OSC を 2つ使う（Detune付き）
  → Unison Voices を増やす（4-8）
  → Sub OSC を追加（Sine、-1 Oct）
  → Saturation / Drive を軽くかける
  → 低域EQブースト

問題2: 音がこもる、暗すぎる
原因と対策:
  → Filter Cutoff を上げる
  → Resonance を少し上げる
  → High Pass Filter で低域を削る
  → Wavetable Position を明るい方向に
  → Drive で倍音を追加

問題3: 音が耳に痛い、キンキンする
原因と対策:
  → Filter Cutoff を下げる
  → Resonance を下げる
  → 12dB/oct に変更（24dB から）
  → High Shelf EQ でカット
  → Saturation の種類を Soft Clip に

問題4: 音に動きがない、退屈
原因と対策:
  → LFO を追加（Filter Cutoff へ）
  → Wavetable Position を LFO で動かす
  → Filter Envelope を追加
  → Chorus / Flanger エフェクトを追加
  → Macro でパラメータをまとめて動かす

問題5: 音が歪む、クリップする
原因と対策:
  → OSC の Volume を下げる
  → Unison Voices を減らす
  → Drive / Distortion を下げる
  → Utility で -3dB 下げる
  → Limiter を挿す（応急処置）

問題6: ミックスで音が埋もれる
原因と対策:
  → EQ で他の楽器と被る帯域をカット
  → Saturation で倍音を追加（存在感）
  → Sidechain Compression
  → Mid/Side EQ で定位を調整
  → 音色自体を見直す（Cutoff、Resonance）

問題7: CPU負荷が高い
原因と対策:
  → Unison Voices を減らす
  → OSC の数を減らす
  → オーバーサンプリングを下げる
  → Freeze / Flatten（バウンス）する
  → エフェクトを最適化
```

### パラメータのクイックリファレンス

```
用途別推奨設定一覧:

═══════════════════════════════════════════════
   パラメータ      Sub Bass  Techno Bass  Lead    Pad
═══════════════════════════════════════════════
OSC波形          Sine     Saw        Saw     Triangle
OSC数            1        1          2       2
Unison Voices    1        1          4-8     2-4
Detune           0%       0%         10-15%  5-10%
Filter Type      LP 24    LP 24      LP 12   LP 12
Cutoff           150Hz    500Hz      3kHz    1.5kHz
Resonance        0%       40-70%     10-20%  10-15%
Attack           0ms      0ms        5-10ms  500-2000ms
Decay            50ms     200ms      300ms   1500ms
Sustain          100%     60-80%     70%     65%
Release          10ms     30ms       400ms   3000-5000ms
LFO              なし     Filter     Pitch   Filter+Pan
═══════════════════════════════════════════════

エフェクトチェーン推奨:
  Bass:  EQ(HP 30Hz) → Saturation → Compression → EQ
  Lead:  Chorus → Delay → Reverb → EQ
  Pad:   Chorus → Reverb(大) → Delay → EQ(HC)
  Pluck: Reverb → Delay → EQ
  FX:    各種エフェクト → Reverb → Limiter
```

---

## 更なる学習リソース

### 推奨学習パス

```
レベル1: 基礎（本章の内容）
  期間: 1-2ヶ月
  目標:
  - 4要素（OSC/Filter/Env/LFO）を理解
  - 基本パッチ10個を暗記
  - 好きなプリセットの構造を分析できる

レベル2: 中級
  期間: 3-6ヶ月
  目標:
  - FM合成の基本を理解
  - ウェーブテーブルの自作
  - モジュレーションマトリクスを活用
  - ジャンル別の音作りが可能

レベル3: 上級
  期間: 6-12ヶ月
  目標:
  - グラニュラー合成を活用
  - 物理モデリングの理解
  - リシンセシス（音の分析・再合成）
  - 独自のサウンドアイデンティティ確立

レベル4: マスター
  期間: 1年以上
  目標:
  - どんな音でも再現できる
  - 新しい音色を即座に作れる
  - 複数のシンセを組み合わせたレイヤリング
  - サウンドデザインを教えられる
```

### おすすめのシンセサイザー（学習用）

```
無料シンセ:
1. Vital（Matt Tytel）: Serum に匹敵する機能、無料版あり
2. Dexed: DX7 エミュレーション、FM合成学習に最適
3. Surge XT: オープンソース、多機能
4. Odin 2: 減算合成の学習に最適
5. Helm: シンプルで分かりやすいUI

有料シンセ（投資価値あり）:
1. Xfer Serum: 業界標準ウェーブテーブルシンセ
2. Native Instruments Massive X: EDM定番
3. u-he Diva: アナログモデリング最高峰
4. Arturia Pigments: マルチエンジン、教育的UI
5. Spectrasonics Omnisphere: 最大級の音源

DAW内蔵シンセ（追加費用なし）:
- Ableton: Wavetable、Operator、Analog、Drift
- Logic Pro: Alchemy、ES2、Retro Synth
- FL Studio: Sytrus、Harmor、3xOsc
- Bitwig: Polymer、Phase-4、Grid
```

---

## 関連ファイル

- **[Wavetable Sound Design](./wavetable-sound-design.md)** - Wavetableでの実践的音作り
- **[03-instruments/wavetable.md](../03-instruments/wavetable.md)** - Wavetable詳細
- **[00-fundamentals/audio-basics.md](../../00-fundamentals/audio-basics.md)** - 音響基礎

---

**シンセシスの基礎をマスターして、自分だけの音を作りましょう！** 🎹

---

## 次に読むべきガイド

- [Wavetable Sound Design（Wavetableサウンドデザイン）](./wavetable-sound-design.md) - 次のトピックへ進む

---

## 参考文献

- [MDN Web Docs](https://developer.mozilla.org/) - Web技術のリファレンス
- [Wikipedia](https://ja.wikipedia.org/) - 技術概念の概要
