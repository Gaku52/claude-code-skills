# パッド設計

**空間を満たすアンビエントサウンドを完全マスター**

パッドは楽曲の「背景」であり、空間を満たし、深みと雰囲気を作り出す重要な要素です。リードやベースが前面に出る一方で、パッドは楽曲全体を支え、感情的な基盤を提供します。このガイドでは、Analogを使った温かいパッド、レイヤリング、ステレオ幅の活用まで、プロレベルのパッド設計を完全マスターします。

---

## この章で学ぶこと

- ✅ Analogでの温かいパッド作成
- ✅ レイヤリング（3-4音色を重ねる）
- ✅ ステレオ幅80-100%の活用
- ✅ Reverb/Chorus活用
- ✅ アンビエントパッド vs リズミックパッド
- ✅ ジャンル別パッド音色
- ✅ ミキシングでのスペース作り

**学習時間**: 3-5時間
**難易度**: ★★★☆☆ 中級

---

## なぜパッドが重要なのか

### DJの視点から

**DJとして**:
- パッドのブレイクダウンで**観客が一体感**を感じる
- 空間が満たされることで**楽曲が完成**される
- パッドの有無で**楽曲の深み**が変わる

**プロデューサーとして**:
- 楽曲に**感情的な基盤**を提供
- リード/ベースを**引き立てる**背景
- 空間を完全にコントロールできる

### プロの意見

> "パッドがなければ楽曲は空っぽに聞こえる。空間を満たすことが重要。"
> — **Eric Prydz**

> "Tranceではパッドが感情の90%を決める。リードは10%に過ぎない。"
> — **Above & Beyond**

> "Deep Houseはパッドが全て。温かく、包み込むようなパッドが必須。"
> — **Kerri Chandler**

### 数字で見る重要性

| ジャンル | パッド重要度 | 制作時間配分 | 使用頻度 |
|---------|------------|-----------|---------|
| **Trance** | ★★★★★ (100%) | 3-4時間 / 20時間 | 95% |
| **Progressive House** | ★★★★★ (100%) | 3-4時間 / 20時間 | 90% |
| **Deep House** | ★★★★★ (100%) | 3-4時間 / 20時間 | 95% |
| **Ambient/Chillout** | ★★★★★ (100%) | 8-10時間 / 20時間 | 100% |
| **Techno** | ★★★☆☆ (60%) | 1-2時間 / 20時間 | 50% |

---

## 1. パッドの種類

### 1.1 アンビエントパッド（Ambient Pad）

**特徴**:
- 常に鳴り続ける
- コード進行に沿う
- ステレオ幅100%
- 長いリリースタイム（2-4秒）

**音色イメージ**:
```
"Aaaahhhhh..." - 広がる、包み込む
```

**使用ジャンル**:
- Trance 90%
- Progressive House 80%
- Ambient 100%

### 1.2 リズミックパッド（Rhythmic Pad）

**特徴**:
- リズムに合わせて鳴る
- 短いリリース（500-1000 ms）
- Offbeat、シンコペーション
- ビートと連動

**音色イメージ**:
```
"Wah-wah-wah" - リズミック、動的
```

**使用ジャンル**:
- Progressive House 70%
- Tech House 60%
- Melodic Techno 80%

### 1.3 ストリングパッド（String Pad）

**特徴**:
- オーケストラ的
- 複数の音を重ねる
- クラシカル、エモーショナル
- Saw波ベース

**音色イメージ**:
```
バイオリンセクションのような厚み
```

**使用ジャンル**:
- Trance 60%
- Orchestral House 80%

### 1.4 ダークパッド（Dark Pad）

**特徴**:
- 低音域中心
- フィルター Cutoff 低め
- 不安、緊張感
- Minor/Phrygianスケール

**音色イメージ**:
```
"Oooooh..." - ダーク、ミステリアス
```

**使用ジャンル**:
- Techno 70%
- Dark Progressive 80%
- Horror/Suspense 100%

---

## 2. Analogでの作成（ステップバイステップ）

### Step 1: 新規トラック作成

```
1. Cmd+Shift+T（新規MIDIトラック）
2. Browser → Instruments → Analog
3. ドラッグ&ドロップ
```

### Step 2: Oscillator設定

**Oscillator 1（メイン）**:
```
Waveform: Saw
Shape: 0%（標準）
Octave: 0
Semi: 0
Detune: 0
Level: 0.00 dB
```

**Oscillator 2（Detune用）**:
```
Waveform: Saw
Shape: 0%
Octave: 0
Semi: 0
Detune: +10 cents（わずかにずらす）
Level: 0.00 dB
```

**Noise**:
```
Color: Pink（柔らかい）
Level: -40 dB（わずかに混ぜる）

効果: 温かみ、アナログ感
```

### Step 3: Filter設定

**Filter 1（Low Pass）**:
```
Type: Low Pass 24 dB（Moogスタイル）
Cutoff: 1500 Hz（暗めの音色）
Resonance: 10-15%（わずかに）
```

**Filter Envelope**:
```
Attack: 500 ms（ゆっくり）
Decay: 1000 ms
Sustain: 70%
Release: 2000 ms（長い）
Envelope Amount: +20%

効果:
  - ゆっくりとフィルターが開く
  - 自然な音色変化
```

### Step 4: Amp Envelope

```
Attack: 500 ms（ゆっくりフェードイン）
Decay: 0 ms
Sustain: 100%
Release: 2000-4000 ms（非常に長い余韻）

→ パッドの定番設定
```

### Step 5: Unison

```
Unison: 4 voices
Detune: 20-30%
Pan: 100%（ステレオ幅最大）

効果:
  - 厚み
  - ステレオ幅広がり
```

### Step 6: Global設定

```
Voices: 8-12（ポリフォニー、コード対応）
Glide: 0 ms（ポルタメント無効）
```

---

## 3. レイヤリング

### 3.1 なぜレイヤリングが必要か

**1音色の限界**:
```
単一パッド:
  - 音域が限定される
  - 周波数帯域が狭い
  - 単調に聞こえる

レイヤリング（3-4音色）:
  - 全音域カバー
  - 周波数帯域が広い
  - 厚み、深み
```

### 3.2 3層構成

**Layer 1: Low Pad（100-500 Hz）**
```
音源: Analog
Oscillator: Saw
Filter Cutoff: 800 Hz
EQ: High Pass 100 Hz、Low Pass 500 Hz
音量: -6 dB
ステレオ幅: 60%（Mono寄り）

役割: 低域の温かみ
```

**Layer 2: Mid Pad（500 Hz - 2 kHz）**
```
音源: Wavetable
Oscillator: Saw
Filter Cutoff: 2000 Hz
EQ: High Pass 500 Hz、Low Pass 2 kHz
音量: -3 dB（メイン）
ステレオ幅: 80%

役割: 中域の厚み、主役
```

**Layer 3: High Pad（2 kHz以上）**
```
音源: Wavetable
Oscillator: Sine/Triangle（柔らかい）
Filter: Off
EQ: High Pass 2 kHz
音量: -12 dB
ステレオ幅: 100%

役割: 高域の輝き、エアリー
```

### 3.3 同じMIDIを使用

```
1つのMIDIトラック:
  - C Major: C3 - E3 - G3

3つの音源トラック:
  - Layer 1: C3 - E3 - G3
  - Layer 2: C3 - E3 - G3
  - Layer 3: C4 - E4 - G4（1オクターブ上）

MIDI Routing:
  - Track 1（MIDI）→ Track 2, 3, 4（Audio）
  - または各トラックに同じMIDIをコピー
```

### 3.4 周波数分離EQ

**完全分離でクリアなサウンド**:

**Layer 1 EQ**:
```
EQ Eight:
  1. High Pass 100 Hz
  2. Low Pass 500 Hz (12 dB/oct)

→ 100-500 Hz のみ
```

**Layer 2 EQ**:
```
EQ Eight:
  1. High Pass 500 Hz (12 dB/oct)
  2. Low Pass 2 kHz (12 dB/oct)

→ 500 Hz - 2 kHz のみ
```

**Layer 3 EQ**:
```
EQ Eight:
  1. High Pass 2 kHz (12 dB/oct)

→ 2 kHz以上のみ
```

**確認**:
```
Spectrum Analyzer:
  - 3層が完全に分離
  - 重複がほぼゼロ

→ マスキングなし、クリア
```

---

## 4. ジャンル別パッド

### 4.1 Trance Epic Pad

**目標**: 感動的、広がり、エピック

**Analog設定**:
```
Oscillator 1: Saw
Oscillator 2: Saw (Detune +12 cents)
Noise: Pink (-40 dB)

Filter: Low Pass 2000 Hz
Resonance: 15%

Unison: 6 voices
Detune: 35%
Pan: 100%

Amp Envelope:
  - Attack: 1000 ms（非常にゆっくり）
  - Release: 4000 ms（非常に長い）
```

**エフェクトチェイン**:
```
1. EQ Eight:
   - High Pass 200 Hz
   - Boost 1-2 kHz +2 dB（温かみ）
   - Air 10 kHz +2 dB

2. Chorus:
   - Rate: 0.5 Hz
   - Depth: 40%
   - Dry/Wet: 50%

3. Send A (Reverb):
   - Hall 4.0s（非常に長い）
   - Pre-Delay: 50 ms
   - Send Level: 50%

4. Utility:
   - Width: 100%（最大ステレオ幅）
```

### 4.2 Deep House Warm Pad

**目標**: 温かい、包み込む、ジャジー

**Analog設定**:
```
Oscillator 1: Saw
Oscillator 2: Square (Detune +7 cents)
Noise: Pink (-35 dB)

Filter: Low Pass 1200 Hz（暗め）
Resonance: 20%（わずかに）

Amp Envelope:
  - Attack: 300 ms
  - Release: 2000 ms
```

**エフェクトチェイン**:
```
1. EQ Eight:
   - High Pass 150 Hz
   - Boost 400 Hz +2 dB（温かみ）
   - Cut 2 kHz -2 dB（柔らかく）

2. Saturator:
   - Drive: 4 dB
   - Curve: Warm
   - Dry/Wet: 30%

3. Send A (Reverb):
   - Room 2.0s（短め）
   - Send Level: 30%

4. Send B (Chorus):
   - Rate: 0.3 Hz
   - Depth: 50%
   - Send Level: 40%

5. Utility:
   - Width: 90%
```

### 4.3 Progressive House Rhythmic Pad

**目標**: リズミック、動的、グルーヴィー

**Wavetable設定**:
```
Oscillator 1: Saw
Unison: 4
Detune: 25%
Stereo: 70%

Filter: Low Pass 2500 Hz
Envelope Amount: +40%

Filter Envelope:
  - Attack: 10 ms（速い）
  - Decay: 400 ms（短い）
  - Sustain: 30%
  - Release: 500 ms

Amp Envelope:
  - Attack: 10 ms
  - Release: 500 ms（短め）
```

**MIDIパターン**:
```
Offbeat:
Kick:  |X---|X---|X---|X---|
Pad:   |-C--|-F--|-G--|-C--|
        Kickの直後
```

**エフェクトチェイン**:
```
1. Compressor (Sidechain to Kick):
   - Ratio: 6:1
   - Threshold: -20 dB
   - Attack: 10 ms
   - Release: 150 ms

2. Send A (Delay):
   - 1/8 Dotted
   - Feedback: 25%
   - Send Level: 20%

3. Utility:
   - Width: 70%
```

### 4.4 Techno Dark Pad

**目標**: ダーク、ミステリアス、緊張感

**Analog設定**:
```
Oscillator 1: Saw
Oscillator 2: Square (Detune -7 cents)

Filter: Low Pass 600 Hz（非常に暗い）
Resonance: 25%

LFO → Filter Cutoff:
  - Rate: 1/4（遅い）
  - Amount: 20%
  - Waveform: Sine

→ ゆっくりとした「wah」効果

Amp Envelope:
  - Attack: 200 ms
  - Release: 1500 ms
```

**エフェクトチェイン**:
```
1. EQ Eight:
   - High Pass 80 Hz
   - Cut 1 kHz以上 -4 dB（ダーク）

2. Saturator:
   - Drive: 8 dB（強め）
   - Curve: A-Shape

3. Send A (Reverb):
   - Dark Hall 3.0s
   - Send Level: 40%

4. Utility:
   - Width: 80%
```

---

## 5. ステレオ幅の活用

### 5.1 なぜステレオ幅が重要か

**Mono（Width 0%）**:
```
特徴:
  - 中央に集中
  - パンチがある
  - 明瞭

用途: ベース、キック、ボーカル
```

**Stereo（Width 100%）**:
```
特徴:
  - 左右に広がる
  - 空間を満たす
  - 包み込む

用途: パッド、アンビエント、FX
```

### 5.2 トラック別最適幅

```
Kick:         0% (Mono必須)
Bass:         0% (120 Hz以下Mono)
Snare/Clap:   20-30%
Hi-Hat:       50-70%
Lead:         60-80%
Pad:          80-100%（最も広い）
Ambient FX:   100%
```

### 5.3 Utility設定

**パッドの最適設定**:
```
Utility:
  - Width: 90-100%
  - Bass Mono: On (100 Hz)

効果:
  - 高域: ステレオ100%
  - 低域: Mono（位相問題回避）
```

**確認方法**:
```
Correlation Meter:
  +1.0: 完全Mono
  +0.5〜+0.8: 適度なステレオ（パッド理想）
  0.0: 無相関
  -1.0: 逆位相（NG）

パッド目標: +0.5〜+0.7
```

### 5.4 Mono互換性テスト

**なぜ必要か**:
```
クラブシステム:
  - 一部のスピーカーはMono
  - Stereoパッドが消える可能性

スマホ/ラジオ:
  - Mono再生が多い
```

**テスト方法**:
```
1. Utility: Width 0%（Mono化）
2. パッドが聞こえるか確認
3. 聞こえない場合:
   - Width 90% → 70%に下げる
   - または Mid/Side EQ で Mid を強化
```

---

## 6. Reverb/Chorus

### 6.1 Reverb（空間の深さ）

**パッドに最適なReverb**:
```
Return Track A: Reverb

Type: Hall
Size: Large
Decay Time: 3.0-4.0s（長い）
Pre-Delay: 30-50 ms
Diffusion: 70-80%
High Cut: 8 kHz（明るすぎない）
Low Cut: 200 Hz（低域は Dry）
Dry/Wet: 100%（Return Trackなので）

Pad Send A: 40-60%（多め）
```

**ジャンル別**:
```
Trance: Hall 4.0s（非常に長い）
Progressive House: Hall 2.5s
Deep House: Room 2.0s（温かい）
Techno: Dark Hall 3.0s
```

### 6.2 Chorus（厚み）

**パッドに最適なChorus**:
```
Return Track B: Chorus

Rate: 0.3-0.5 Hz（遅い）
Depth: 40-60%
Delay 1: 7 ms
Delay 2: 21 ms
Feedback: 10%
Dry/Wet: 100%

Pad Send B: 30-50%
```

**効果**:
- 音が揺らぐ
- 厚み、温かみ
- アナログ感

### 6.3 エフェクトチェイン順序

**パッド完全エフェクトチェイン**:
```
1. EQ Eight（前処理）:
   - High Pass 150-200 Hz
   - Boost/Cut

2. Saturator（倍音、温かみ）:
   - Drive: 3-5 dB
   - Curve: Warm

3. Compressor（ダイナミクス均一化）:
   - Ratio: 2:1（軽め）
   - Threshold: -20 dB
   - Attack: 30 ms
   - Release: 500 ms

4. Utility（ステレオ幅）:
   - Width: 90-100%
   - Bass Mono: On

5. Send A - Reverb（空間）

6. Send B - Chorus（厚み）
```

---

## 7. ミキシング

### 7.1 EQ処理

**パッドの基本EQ**:
```
EQ Eight:

1. High Pass 150-200 Hz
   （ベース領域を完全に避ける）

2. Low-Mid Cut 250-400 Hz -2 to -3 dB
   （マッディネス除去）

3. Presence Boost/Cut 1-2 kHz ±2 dB
   （リードとの関係で調整）

4. Air Boost 8-12 kHz +1 to +2 dB
   （輝き）
```

**他のトラックとのスペース作り**:
```
リードがある場合:
  - Lead: 2-4 kHz +3 dB
  - Pad: 2-4 kHz -3 dB

リードがない場合:
  - Pad: 2-4 kHz +2 dB（前に出す）
```

### 7.2 音量バランス

**ミックス内での適切な音量**:
```
Kick: -6 dB (Peak)
Bass: -9 dB
Pad: -12 to -15 dB（背景）
Lead: -9 to -6 dB

→ パッドは控えめが基本
```

**セクション別**:
```
Intro/Breakdown: Pad -9 dB（主役）
Drop: Pad -15 dB（背景）
```

### 7.3 サイドチェイン（オプション）

**Kickとの共存**:
```
Pad Track:
→ Compressor (Sidechain: Kick)

設定:
  - Ratio: 3:1（軽め）
  - Threshold: -25 dB
  - Attack: 30 ms（遅め）
  - Release: 300 ms

→ わずかにKickとの空間を作る
```

**使用頻度**:
```
Progressive House: 70%（必須レベル）
Trance: 30%（オプション）
Deep House: 10%（ほぼ不要）
```

---

## 8. 練習方法

### 初級（Week 1-2）

**Week 1: プリセット使用**
```
Day 1-2: Analog プリセット "Warm Pad" で曲作り
Day 3-4: Wavetable プリセット "Lush Pad" で曲作り
Day 5-7: 5つのパッドプリセットを試す
```

**Week 2: 簡単な改変**
```
Day 1-2: Filter Cutoff調整（明るさ）
Day 3-4: Attack/Release調整（フェード）
Day 5-7: Reverb Send調整（空間）
```

---

### 中級（Week 3-4）

**Week 3: ゼロから作成**
```
Day 1-2: Analog基本パッド作成
Day 3-4: エフェクトチェイン完成
Day 5-7: ステレオ幅、Reverb最適化
```

**Week 4: レイヤリング**
```
Day 1-3: 3層パッド作成
Day 4-5: 周波数分離EQ
Day 6-7: ミキシング完成
```

---

### 上級（Week 5-8）

**Week 5-6: ジャンル別パッド**
```
Day 1-2: Trance Epic Pad
Day 3-4: Deep House Warm Pad
Day 5-6: Progressive Rhythmic Pad
Day 7: Techno Dark Pad
```

**Week 7-8: フル楽曲制作**
```
1. パッドを核にした楽曲
2. Intro → Breakdown → Drop
3. パッドの役割を最大化
4. SoundCloudアップロード
```

---

## 9. よくある失敗と対処法

### 失敗1: パッドが目立ちすぎる

**対処法**:
```
1. 音量: -15 dB（控えめ）
2. EQ: 2-4 kHz -3 dB（リードを際立たせる）
3. Attack: 500 ms→1000 ms（ゆっくり）
4. ステレオ幅: 100%→80%
```

---

### 失敗2: パッドが薄い

**対処法**:
```
1. レイヤリング（3音色）
2. Unison: 4→6 voices
3. Detune: 20%→35%
4. Chorus Send: 40%
```

---

### 失敗3: 低域がぼやける

**対処法**:
```
1. EQ: High Pass 200 Hz（ベース領域完全カット）
2. Utility: Bass Mono On (150 Hz)
3. Filter Cutoff: 1500 Hz（暗めに）
```

---

### 失敗4: Mono互換性が悪い

**対処法**:
```
1. Width: 100%→80%
2. Mid/Side EQ: Mid +2 dB
3. Unison Detune: 50%→30%
4. Mono化テストで確認
```

---

## まとめ

### パッド設計の核心

1. **Analog**: 温かい、アナログ的
2. **レイヤリング**: 3層で全音域カバー
3. **ステレオ幅**: 80-100%（最も広い）
4. **Reverb**: Hall 3.0s以上
5. **音量**: -12 to -15 dB（背景）

### DJから制作者へ

**DJスキル**:
- ブレイクダウンでのパッドの重要性を体感
- 空間が満たされる感覚を理解

**プロデューサーとして**:
- 空間を完全にコントロール
- 感情的な基盤を提供
- 楽曲を完成させる

### 次のステップ

1. **[カウンターメロディ](./counter-melody.md)**: メロディを補完
2. **[ミキシング](../05-mixing/)**: パッドを完璧に仕上げる
3. **[アレンジメント](../06-arrangement/)**: パッドの使い分け

---

**次のステップ**: [カウンターメロディ](./counter-melody.md) へ進む

---

**🎵 空間を満たすパッドを完全マスターして、感動的な楽曲を作りましょう！**
