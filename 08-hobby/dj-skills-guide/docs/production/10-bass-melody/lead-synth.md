# リードシンセ

**楽曲の主役となる音色を完全マスター**

リードシンセは楽曲の「顔」であり、メロディを演奏する主役の音色です。Tranceのエピックなスーパーソー、Technoのアシッドリード、Progressive Houseの感動的なプラックまで、リードシンセの音色デザインが楽曲の個性を決定づけます。このガイドでは、Wavetableを使ったプロレベルのリードシンセ作成を完全マスターします。

---

## この章で学ぶこと

- ✅ Wavetableでのリード音色設計
- ✅ Supersawテクニック（Unison 8、Detune 30-50%）
- ✅ フィルターエンベロープ
- ✅ LFOモジュレーション
- ✅ ジャンル別リード音色（Trance、Techno、House、Progressive）
- ✅ エフェクトチェイン（EQ→Comp→Reverb→Delay）
- ✅ プリセット改変テクニック

**学習時間**: 4-6時間
**難易度**: ★★★☆☆ 中級

---

## なぜリードシンセが重要なのか

### DJの視点から

**DJとして**:
- リードシンセのドロップで**フロアが爆発**する瞬間を体感
- 音色の違いで**ジャンルが識別**できる
- エピックなリードで**観客の感情が高まる**

**プロデューサーとして**:
- その「魔法の瞬間」を**自分で作り出せる**
- 音色デザインで**楽曲の個性**を確立
- Wavetableだけで**無限のリード音色**を作成

### プロの意見

> "スーパーソーリードがなければTrancejゃない。Unisonで8-16ボイス、Detune 40%が黄金律。"
> — **Armin van Buuren**

> "リードシンセは楽曲の顔。音色が弱ければ、メロディがどんなに良くても埋もれる。"
> — **Deadmau5**

> "Progressive Houseではプラックリード。アタック10ms、リリース300msで完璧。"
> — **Eric Prydz**

### 数字で見る重要性

| ジャンル | リード重要度 | 制作時間配分 | 使用頻度 |
|---------|------------|-----------|---------|
| **Trance** | ★★★★★ (100%) | 4-5時間 / 20時間 | 90% |
| **Progressive House** | ★★★★★ (100%) | 3-4時間 / 20時間 | 80% |
| **Future Bass** | ★★★★★ (100%) | 4-5時間 / 20時間 | 95% |
| **Techno** | ★★★☆☆ (60%) | 1-2時間 / 20時間 | 40% |
| **Deep House** | ★★★☆☆ (60%) | 2-3時間 / 20時間 | 50% |

---

## 1. リードシンセの種類

### 1.1 Supersaw Lead（Trance、Future Bass）

**特徴**:
- 複数のSaw波を重ねる（Unison）
- わずかにDetuneして厚みを作る
- ステレオ幅が広い
- エピック、感動的

**音色イメージ**:
```
"Waaaaaaah" - 厚い、広がる、パワフル
```

**使用ジャンル**:
- Trance 90%
- Progressive House 70%
- Future Bass 80%
- Big Room House 60%

### 1.2 Pluck Lead（Progressive House）

**特徴**:
- 短いアタック（10-30 ms）
- 短いリリース（200-500 ms）
- 歯切れが良い
- リズミック

**音色イメージ**:
```
"Plink plink plink" - 明瞭、弾む
```

**使用ジャンル**:
- Progressive House 80%
- Melodic Techno 60%
- Tech House 50%

### 1.3 Acid Lead（Techno）

**特徴**:
- 303スタイル
- フィルターのレゾナンス高め
- モジュレーション激しい
- アグレッシブ

**音色イメージ**:
```
"Beeeow beeeow" - 鋭い、酸っぱい
```

**使用ジャンル**:
- Acid Techno 90%
- Tech House 40%
- Electro House 30%

### 1.4 Stab Lead（House）

**特徴**:
- コード全体を短く鳴らす
- アタック即座（0 ms）
- リリース短い（100-200 ms）
- パンチがある

**音色イメージ**:
```
"Stab! Stab!" - 突き刺すような
```

**使用ジャンル**:
- House 60%
- Disco 70%
- Funky House 80%

---

## 2. Wavetableでの作成（Supersaw）

### Step 1: 新規トラック作成

```
1. Cmd+Shift+T（新規MIDIトラック）
2. Browser → Instruments → Wavetable
3. ドラッグ&ドロップ
```

### Step 2: Oscillator設定

**Oscillator 1（メイン）**:
```
Category: Basic Shapes
Wavetable: Saw
Position: 0.00（完全なSaw波）
Level: 0.00 dB
```

**Oscillator 2（Detune用）**:
```
Category: Basic Shapes
Wavetable: Saw
Position: 0.00
Level: 0.00 dB
Detune: +7 cents（わずかにずらす）
```

**Sub Oscillator**:
```
Off（またはLevel -12 dB、低域補強用）
```

### Step 3: Unison設定（最重要）

**Oscillator 1 Unison**:
```
Unison: 8 voices
Detune: 40%
Stereo: 70%

効果:
  - 1音が8音に分裂
  - わずかにDetuneして厚み
  - ステレオ幅70%で広がり
```

**Oscillator 2 Unison**:
```
Unison: 8 voices
Detune: 35%（Osc 1と少し違う）
Stereo: 60%
```

**結果**:
```
合計16ボイス（Osc 1: 8 + Osc 2: 8）
→ 超厚いSupersaw
```

### Step 4: Filter設定

**Filter 1（Low Pass）**:
```
Type: Low Pass (Clean)
Cutoff: 3000 Hz（初期値）
Resonance: 10-20%
```

**Filter Envelope**:
```
Attack: 10 ms
Decay: 500 ms
Sustain: 60%
Release: 300 ms
Envelope Amount: +30%

効果:
  - アタック時にフィルターが開く
  - 徐々に閉じて落ち着く
```

### Step 5: Amp Envelope

```
Attack: 10 ms（わずかなフェードイン）
Decay: 0 ms
Sustain: 100%
Release: 500 ms（長めの余韻）
```

### Step 6: Global設定

```
Voices: 8（ポリフォニー、和音対応）
Glide: 0 ms（ポルタメント無効）
```

---

## 3. ジャンル別リードシンセ

### 3.1 Trance Supersaw

**目標**: エピック、感動的、ステレオ幅広い

**Wavetable設定**:
```
Oscillator 1:
  - Wavetable: Saw
  - Unison: 8
  - Detune: 45%
  - Stereo: 80%

Oscillator 2:
  - Wavetable: Saw
  - Detune: +10 cents
  - Unison: 8
  - Detune: 40%
  - Stereo: 70%

Filter:
  - Cutoff: 4000 Hz
  - Resonance: 20%
  - Envelope Amount: +40%

Amp Envelope:
  - Attack: 20 ms
  - Release: 800 ms（長い余韻）
```

**エフェクトチェイン**:
```
1. EQ Eight:
   - High Pass 200 Hz
   - Boost 4 kHz +3 dB（明瞭さ）
   - Air 12 kHz +2 dB（輝き）

2. Compressor:
   - Ratio: 3:1
   - Threshold: -12 dB
   - Attack: 10 ms
   - Release: 100 ms

3. Send A (Reverb):
   - Hall 3.0s
   - Send Level: 30%

4. Send B (Delay):
   - 1/8 Dotted
   - Feedback: 30%
   - Send Level: 20%

5. Utility:
   - Width: 100%（最大ステレオ幅）
```

### 3.2 Progressive House Pluck

**目標**: 歯切れ良い、リズミック、明瞭

**Wavetable設定**:
```
Oscillator 1:
  - Wavetable: Saw
  - Unison: 4（Tranceより少ない）
  - Detune: 30%
  - Stereo: 60%

Oscillator 2:
  - Off

Filter:
  - Cutoff: 2000 Hz
  - Resonance: 15%
  - Envelope Amount: +50%（大きめ）

Filter Envelope:
  - Attack: 0 ms（即座）
  - Decay: 300 ms（短い）
  - Sustain: 10%（ほぼゼロ）
  - Release: 100 ms（短い）

Amp Envelope:
  - Attack: 10 ms
  - Decay: 0 ms
  - Sustain: 100%
  - Release: 300 ms（Tranceより短い）
```

**エフェクトチェイン**:
```
1. EQ Eight:
   - High Pass 300 Hz
   - Boost 2 kHz +2 dB

2. Compressor (Sidechain to Kick):
   - Ratio: 6:1
   - Threshold: -20 dB
   - Attack: 10 ms
   - Release: 150 ms

3. Send A (Reverb):
   - Hall 2.0s
   - Send Level: 15%（控えめ）

4. Utility:
   - Width: 70%
```

### 3.3 Techno Acid Lead

**目標**: アグレッシブ、フィルターモジュレーション激しい

**Operator設定（FM合成）**:
```
Algorithm: 2
Operator A (Carrier):
  - Waveform: Saw
  - Coarse: 1.00
  - Level: 0.00 dB

Operator B (Modulator):
  - Waveform: Saw
  - Coarse: 1.00
  - Fine: 0.02（わずかにDetune）
  - Level: 30-60%（モジュレーション量）

Filter:
  - Type: Low Pass 12 dB
  - Cutoff: 500-2000 Hz（LFOで変化）
  - Resonance: 70%（高め、303スタイル）

Filter LFO:
  - Rate: 1/16（速い）
  - Amount: 80%
  - Waveform: Saw Up

または:

  - Rate: 1/8 Triplet
  - Amount: 60%
```

**エフェクトチェイン**:
```
1. Saturator:
   - Drive: 6 dB
   - Curve: A-Shape

2. EQ Eight:
   - High Pass 150 Hz
   - Boost 800 Hz +2 dB（酸味）

3. Send A (Reverb):
   - Room 1.0s（短い）
   - Send Level: 10%

4. Auto Filter (追加モジュレーション):
   - LFO Rate: 1/4
   - Amount: 30%
```

### 3.4 Future Bass Vocal Lead

**目標**: 人間的、感情的、ボーカル風

**Wavetable設定**:
```
Oscillator 1:
  - Wavetable: Vocal Formants（人間の声）
  - Position: 0.30（適度なフォルマント）
  - Unison: 8
  - Detune: 35%
  - Stereo: 80%

Oscillator 2:
  - Wavetable: Saw
  - Level: -6 dB（補助）
  - Unison: 4
  - Detune: 25%

Filter:
  - Cutoff: 3000 Hz
  - Resonance: 25%
  - Envelope Amount: +35%

LFO 1 → Filter Cutoff:
  - Rate: 1/2（遅い）
  - Amount: 20%
  - Waveform: Sine

→ ゆっくりとした「wah wah」効果
```

**エフェクトチェイン**:
```
1. Vocoder（オプション、より人間的に）:
   - Carrier: Lead Synth
   - Modulator: Vocal Sample

2. EQ Eight:
   - Boost 2-4 kHz +3 dB（人間の声の周波数）

3. Compressor:
   - Ratio: 4:1
   - Threshold: -15 dB

4. Send A (Reverb):
   - Hall 2.5s
   - Send Level: 35%

5. Send B (Chorus):
   - Rate: 0.5 Hz
   - Depth: 40%
   - Send Level: 25%（温かみ）
```

---

## 4. 高度なテクニック

### 4.1 LFOモジュレーション

**Filter Cutoff LFO（Wah効果）**:
```
LFO Settings:
  - Rate: 1/4（ビートに同期）
  - Waveform: Sine（滑らか）
  - Amount: 40%

効果:
  - フィルターが開閉
  - "Wah wah wah" サウンド
  - リズミック
```

**Pitch LFO（ビブラート）**:
```
LFO Settings:
  - Rate: 5 Hz（速い）
  - Waveform: Sine
  - Amount: 5%（わずか）

効果:
  - 音程が微妙に揺れる
  - 人間的、生命感
```

**Amp LFO（トレモロ）**:
```
LFO Settings:
  - Rate: 1/8（ビートに同期）
  - Waveform: Square（オン/オフ）
  - Amount: 50%

効果:
  - 音量が周期的に変化
  - ゲート効果
```

### 4.2 Macro Knob設定

**8つのMacro Knobで即座にコントロール**:

```
Macro 1: Filter Cutoff (300 - 8000 Hz)
  → 明るさ調整

Macro 2: Resonance (0 - 50%)
  → 音色の鋭さ

Macro 3: Unison Detune (0 - 100%)
  → 厚み調整

Macro 4: Stereo Width (0 - 100%)
  → 広がり調整

Macro 5: Amp Attack (0 - 100 ms)
  → アタックの鋭さ

Macro 6: Amp Release (100 - 2000 ms)
  → 余韻の長さ

Macro 7: LFO Rate (1/16 - 1 Bar)
  → モジュレーション速度

Macro 8: Reverb Send (0 - 50%)
  → 空間の深さ
```

**使い方**:
```
1. Wavetableのパラメーターを右クリック
2. "Map to Macro 1"
3. Range設定（Min/Max）
4. Macro Knobを回すだけで調整
```

### 4.3 レイヤリング

**複数のリードを重ねる**:

**Layer 1: メインリード**
```
音域: C4 - C5
Wavetable Supersaw
Unison 8
ステレオ幅: 80%
音量: 0 dB
```

**Layer 2: サブリード（1オクターブ下）**
```
音域: C3 - C4
同じMIDI、1オクターブ下
音量: -6 dB
ステレオ幅: 60%

効果: 厚み、低域補強
```

**Layer 3: ハイリード（1オクターブ上）**
```
音域: C5 - C6
音量: -12 dB
ステレオ幅: 100%

効果: 輝き、エアリー
```

**ミキシング**:
```
EQで周波数分離:
  - Layer 1: 500 Hz - 4 kHz（メイン）
  - Layer 2: 100 - 1000 Hz（Low Cut 500 Hz）
  - Layer 3: 2 kHz以上（High Pass 2 kHz）

→ お互いに干渉しない
```

---

## 5. エフェクトチェイン

### 5.1 完全エフェクトチェイン（Trance Lead）

**順序が重要**:

```
1. EQ Eight（前処理）:
   - High Pass 200 Hz
   - Low-Mid Cut 400 Hz -2 dB
   - Presence Boost 4 kHz +3 dB
   - Air Boost 12 kHz +2 dB

2. Compressor:
   - Ratio: 3:1
   - Threshold: -12 dB
   - Attack: 10 ms
   - Release: 100 ms
   - Makeup Gain: +3 dB

3. Saturator（倍音追加）:
   - Drive: 3 dB
   - Curve: Warm
   - Dry/Wet: 30%

4. Utility（ステレオ幅）:
   - Width: 100%

5. Send A - Reverb:
   - Hall 3.0s
   - Pre-Delay: 30 ms
   - Dry/Wet: 100%
   - Send Level: 30%

6. Send B - Delay:
   - Time: 1/8 Dotted
   - Feedback: 30%
   - Dry/Wet: 100%
   - Send Level: 20%
```

### 5.2 サイドチェイン（必須）

**Kickとの共存**:
```
Lead Track:
→ Compressor (2つ目、専用)
→ Audio From: 1-Kick

設定:
  - Ratio: 4:1
  - Threshold: -20 dB
  - Attack: 10 ms
  - Release: 150 ms

効果:
  - Kickが鳴る瞬間、リードが下がる
  - グルーヴ、ダンサビリティ
  - Progressive House必須
```

---

## 6. ミキシング

### 6.1 EQ処理

**リードシンセの基本EQ**:
```
EQ Eight:

1. High Pass 200-300 Hz
   （ベース領域回避）

2. Low-Mid Cut 400-600 Hz -2 to -3 dB
   （マッディネス除去）

3. Presence Boost 2-4 kHz +2 to +4 dB
   （明瞭さ、前に出す）

4. Air Boost 10-14 kHz +1 to +2 dB
   （輝き、エアリー）
```

**他のトラックとのスペース作り**:
```
Lead: 2-4 kHz +3 dB
Pad: 2-4 kHz -3 dB

→ お互いに干渉しない
```

### 6.2 ステレオ幅

**リードの最適幅**:
```
Trance Lead: 80-100%（広い）
Pluck Lead: 60-70%（中央寄り）
Mono Lead: 0%（完全中央、ボーカル風）

Utility:
  - Width: 80%

確認:
  - Mono互換性チェック
  - Correlation Meter: +0.3以上
```

### 6.3 音量バランス

**ミックス内での適切な音量**:
```
Kick: -6 dB (Peak)
Bass: -9 dB
Pad: -12 dB
Lead: -9 to -6 dB（ジャンルによる）

Trance: -6 dB（リードが主役）
Progressive House: -9 dB（控えめ）
Techno: -12 dB（背景）
```

---

## 7. プリセット改変テクニック

### 7.1 プリセットから始める

**Wavetable Factory Presets**:
```
Browser → Sounds → Wavetable
→ "Lead" フォルダ

推奨プリセット:
  - "Classic Lead"
  - "Supersaw Lead"
  - "Pluck Lead"
  - "Acid Lead"
```

### 7.2 改変の手順

**Step 1: Unisonを調整**
```
プリセット: Unison 4
自分: Unison 8
Detune: 30% → 45%

→ より厚く、広がる
```

**Step 2: Filterを調整**
```
Cutoff: 2000 Hz → 3500 Hz（明るく）
Resonance: 10% → 25%（鋭く）
```

**Step 3: Envelopeを調整**
```
Amp Attack: 50 ms → 10 ms（速く）
Amp Release: 200 ms → 500 ms（長く）
```

**Step 4: エフェクトを追加**
```
プリセットのエフェクト → 削除
自分のエフェクトチェイン → 適用
```

**Step 5: 保存**
```
Save Preset As:
  - "Supersaw Lead - My Style"
  - Favorites に追加
```

---

## 8. 練習方法

### 初級（Week 1-2）

**Week 1: プリセット使用**
```
Day 1-2: プリセット "Classic Lead" で曲作り
Day 3-4: プリセット "Pluck Lead" で曲作り
Day 5-7: 5つのプリセットを試す
```

**Week 2: 簡単な改変**
```
Day 1-2: Unison、Detune調整
Day 3-4: Filter Cutoff調整
Day 5-7: Envelope調整
```

---

### 中級（Week 3-4）

**Week 3: ゼロからSupersaw作成**
```
Day 1-2: Oscillator設定、Unison
Day 3-4: Filter、Envelope
Day 5-7: エフェクトチェイン完成
```

**Week 4: ジャンル別リード**
```
Day 1-2: Trance Supersaw
Day 3-4: Progressive Pluck
Day 5-7: Techno Acid
```

---

### 上級（Week 5-8）

**Week 5-6: リファレンストラック分析**
```
1. プロの楽曲のリード音色を耳で分析
2. Wavetableで完全再現
3. プリセット保存
4. 自分の楽曲に使用
```

**Week 7-8: オリジナル音色開発**
```
1. 完全オリジナルリード作成
2. Macro Knob設定
3. 10パターン保存
4. フル楽曲制作
```

---

## 9. よくある失敗と対処法

### 失敗1: リードが埋もれる

**対処法**:
```
1. EQ: 2-4 kHz +3 dB
2. Compressor: Threshold -15 dB
3. ステレオ幅: 80%
4. 他の楽器: 2-4 kHz -2 dB
```

---

### 失敗2: 音が薄い

**対処法**:
```
1. Unison: 4 → 8
2. Detune: 20% → 40%
3. Oscillator 2を追加（Detune +7 cents）
4. レイヤリング（1オクターブ下も追加）
```

---

### 失敗3: 音が汚い、ざらつく

**対処法**:
```
1. Unison Detune下げる: 60% → 30%
2. Resonance下げる: 40% → 15%
3. EQ: High Pass 200 Hz（低域ノイズ除去）
4. Filter Cutoff下げる: 5000 → 3000 Hz
```

---

### 失敗4: ステレオ幅が広すぎてMono互換性NG

**対処法**:
```
1. Utility Width: 100% → 70%
2. Unison Stereo: 80% → 60%
3. Mono互換性テスト（Utilityでmono化）
4. Correlation Meter: +0.3以上確認
```

---

## まとめ

### リードシンセの核心

1. **Supersaw**: Unison 8、Detune 40%が基本
2. **Filter Envelope**: Attack 10ms、Decay 500ms
3. **エフェクトチェイン**: EQ→Comp→Reverb→Delay
4. **ステレオ幅**: 60-80%（ジャンルによる）
5. **サイドチェイン**: Kickとの共存必須

### DJから制作者へ

**DJスキル**:
- フロアで効いたリード音色を分析
- ジャンル別の音色特性を理解
- エピックな瞬間を体感

**プロデューサーとして**:
- その「魔法」を自分で作り出せる
- 音色デザインで個性を確立
- 無限のバリエーションを創造

### 次のステップ

1. **[パッド](./pads.md)**: リードを補完する背景音
2. **[ミキシング](../05-mixing/)**: リードを完璧に仕上げる
3. **[サウンドデザイン](../08-sound-design/)**: 高度な音色設計

---

**次のステップ**: [パッド](./pads.md) へ進む

---

**楽曲の主役となるリードシンセを完全マスターして、エピックなトラックを作りましょう！**

---

## 10. リードシンセの音作り詳細ガイド

### 10.1 オシレーター選択の基礎理論

リードシンセの音作りで最初に決定すべきは「どの波形から出発するか」です。波形選択が音色の方向性を根本的に決定します。

**基本波形の特性比較**:

| 波形 | 倍音構成 | 音色の印象 | 適するジャンル |
|------|---------|-----------|-------------|
| **Sawtooth（鋸歯波）** | 全倍音を含む | 明るく、存在感がある | Trance, Progressive, Future Bass |
| **Square（矩形波）** | 奇数倍音のみ | 中空的、太い | Retro, Synthwave, Chiptune |
| **Pulse（パルス波）** | PWMで倍音が変化 | 動的、生命感 | Techno, Minimal |
| **Triangle（三角波）** | 少ない倍音 | 暗く、柔らかい | Ambient, Chill |
| **Sine（正弦波）** | 基音のみ | 純粋、ピュア | Sub Bass, FM合成のキャリア |
| **Noise（ノイズ）** | 全周波数 | ざらつき、テクスチャ | エフェクト、アクセント |

**Wavetableスキャニングの活用**:
```
Wavetable Position のオートメーション:

Static（固定）:
  - Position: 0.00 → 特定の音色を維持
  - 安定感のあるリード向け

Envelope Scan（エンベロープスキャン）:
  - Attack時に Position 0.00 → 0.50
  - Decay時に 0.50 → 0.20
  - アタック時だけ明るくなる効果

LFO Scan（LFOスキャン）:
  - Rate: 1/2（遅め）
  - Amount: 30%
  - 常に音色が変化し続ける
  - パッド的なリードに最適

Random Scan（ランダム）:
  - S&H（Sample & Hold）LFO
  - Rate: 1/16
  - Amount: 15%
  - 予測不能な音色変化
  - Glitch系リードに最適
```

**カスタムWavetableの読み込み**:
```
Serum:
  1. Wavetable Editor → メニュー → Import
  2. WAVファイルをドラッグ&ドロップ
  3. 256サンプル x Nフレーム に自動分割
  4. WT Position でスキャン

Vital:
  1. OSC → Wavetable → ファイルアイコン
  2. WAV / フレーズをドラッグ
  3. Audio to WT 変換で自動マッピング

Wavetable（Ableton）:
  1. Oscillator → 右クリック → "Load Wavetable"
  2. .wav ファイルを選択
  3. Frame数を指定
```

### 10.2 倍音構造の理解と操作

**フーリエ級数とシンセサイザー**:

音は基本周波数（ファンダメンタル）とその整数倍の周波数（倍音/ハーモニクス）で構成されます。リードシンセの音色は倍音の構成比率で決まります。

```
基音: f（例: 440 Hz = A4）
第2倍音: 2f（880 Hz）
第3倍音: 3f（1320 Hz）
第4倍音: 4f（1760 Hz）
...
第N倍音: Nf

鋸歯波の倍音:
  全ての整数倍音を含む
  振幅は 1/n に比例
  1: 100%, 2: 50%, 3: 33%, 4: 25%...

矩形波の倍音:
  奇数倍音のみ
  振幅は 1/n に比例
  1: 100%, 3: 33%, 5: 20%, 7: 14%...
```

**倍音操作テクニック**:
```
1. サブトラクティブ（減算）:
   - ローパスフィルターで上の倍音をカット
   - Cutoff下げる → 暗い音
   - Cutoff上げる → 明るい音

2. FM（周波数変調）:
   - モジュレーターの周波数比で倍音を追加
   - Ratio 1:1 → 全倍音強調
   - Ratio 1:2 → オクターブ上の倍音追加
   - Ratio 1:3 → 5度上の倍音追加（金属的）

3. ウェーブシェーピング:
   - Saturator/Distortion で倍音を生成
   - Soft Clip → 奇数倍音（温かい歪み）
   - Hard Clip → 奇数+偶数倍音（アグレッシブ）
   - Tube → 偶数倍音（真空管の温かみ）

4. リングモジュレーション:
   - 2つの信号の積（掛け算）
   - 和と差の周波数が生まれる
   - 非整数倍音が発生 → 金属的、ベル的
```

### 10.3 フィルタータイプ詳細

**ローパスフィルター（LP）の種類と使い分け**:
```
12 dB/oct（2 Pole）:
  - 穏やかなカット
  - 自然なサウンド
  - アコースティック系リードに適する
  - Ableton: "Clean"

24 dB/oct（4 Pole）:
  - 急峻なカット
  - Moog系の力強いサウンド
  - Supersaw、Bass Lead に適する
  - Ableton: "OSR"（Over Sampled Resonance）

36 dB/oct（6 Pole）:
  - 非常に急峻
  - 特定周波数を狙い撃ち
  - 効果的なフィルタースイープ
  - Serum / Vital で使用可能
```

**ハイパスフィルター（HP）の活用**:
```
リードシンセでのHP使用:
  - ベースとの住み分け
  - Cutoff: 150-400 Hz（ジャンルによる）

Trance Lead: HP 150-200 Hz
  → ベースラインに空間を確保

Progressive Pluck: HP 250-350 Hz
  → 短い音なので高めでも自然

Acid Lead: HP 100-150 Hz
  → 低域の太さを残す
```

**バンドパスフィルター（BP）の活用**:
```
電話ボイス風リード:
  - BP Center: 1200 Hz
  - Width: Narrow
  - Resonance: 40%
  → Lo-Fiな質感、ブレイクダウンで効果的

ラジオ風リード:
  - BP Center: 2000 Hz
  - Width: Medium
  - Resonance: 20%
  → 中域に集中した存在感
```

**コムフィルター**:
```
Flanger風の櫛形フィルター:
  - Delay Time: 1-10 ms
  - Feedback: 40-80%
  - フランジャー的な金属感
  - リードに独特のカラーを追加

短いディレイのフィードバック:
  - Delay Time: 2 ms → ピッチ: 約500 Hz に共振
  - Delay Time: 5 ms → ピッチ: 約200 Hz に共振
  - フィードバック量で効果の強さを調整
```

---

## 11. Supersawの完全設計ガイド

### 11.1 Supersawの物語と進化

Supersawは1996年にRoland JP-8000で誕生した「Super Saw」が起源です。7基のデチューンされた鋸歯波を一つのオシレーターとして鳴らすこの革新は、Trance、Eurodance、そして現代のEDM全般に不可欠なサウンドとなりました。

**歴史的な進化**:
```
1996: Roland JP-8000
  - 7ボイス固定のSuper Saw
  - ハードウェアの制約下でも圧倒的な厚み

2000年代前半: ソフトシンセの登場
  - Sylenth1: 最大8ユニゾンボイス
  - Nexus: プリセットベースのSupersaw

2010年代: 高機能ソフトシンセ
  - Serum: 最大16ユニゾン、Wavetable対応
  - Massive: 柔軟なルーティング

2020年代: 次世代シンセ
  - Vital: 最大16ユニゾン、無料で高機能
  - Phase Plant: モジュラー設計で無限の可能性
```

### 11.2 ユニゾンボイス数による音色変化

**ボイス数ごとの特徴**:
```
2 voices:
  - 基本的なデチューン
  - クリーンな厚み
  - CPU負荷: 最小
  - 用途: サブリード、アクセント

4 voices:
  - バランスの良い厚み
  - ステレオ感も確保
  - CPU負荷: 低
  - 用途: Pluckリード、House系

8 voices:
  - プロフェッショナルな厚み
  - 十分なステレオ幅
  - CPU負荷: 中
  - 用途: Trance、Progressive House
  - 最もよく使われるセッティング

16 voices:
  - 極限の厚み
  - 壁のようなサウンド
  - CPU負荷: 高
  - 用途: Epic Trance、Big Room、Anthem

32 voices（Vital等）:
  - 過剰な厚み（意図的な使用）
  - パッド的な質感
  - CPU負荷: 非常に高
  - 用途: 特殊効果、テクスチャ
```

### 11.3 デチューン量の詳細調整

**デチューン量による音色変化マップ**:
```
0-10%: ほぼ位相効果
  - コーラスライクな揺らぎ
  - クリーンなリード向け

10-25%: 軽いデチューン
  - 明瞭さを保ちつつ厚みを追加
  - Melodic Techno、Deep House向け

25-40%: 標準デチューン
  - バランスの取れた厚み
  - ほとんどのジャンルで使用
  - Trance: 35-40% が黄金律

40-60%: 強デチューン
  - 非常に厚く、やや曖昧な音程感
  - Epic Trance、Big Room向け
  - ピッチの正確性は犠牲に

60-100%: 極端なデチューン
  - もはやピッチは不明瞭
  - テクスチャ、パッド向け
  - 特殊効果として使用
```

**デチューンカーブの種類**:
```
Linear（線形）:
  - 均等にデチューン
  - バランスの良い厚み
  - Serum デフォルト

Exponential（指数）:
  - 中心付近に多くの声
  - 外側は広がる
  - よりナチュラルな印象

Random（ランダム）:
  - 毎回わずかに異なるデチューン
  - アナログシンセ風の有機的な揺らぎ
  - Diva、Repro で特に効果的
```

### 11.4 ステレオ分布の制御

```
Stereo Spread の設定:

0%（Mono）:
  - 全ボイスが中央
  - モノリードとして使用
  - サブバスとの組み合わせに最適

30-50%（Narrow）:
  - 適度な広がり
  - 中央の存在感を維持
  - ヴァースのリード向け

60-80%（Standard）:
  - プロフェッショナルな広がり
  - メインドロップのリード標準
  - Mono互換性も確保

90-100%（Wide）:
  - 最大限の広がり
  - パッド的な使い方
  - Mono互換性の確認必須

テクニック: Mid/Side処理
  - Mid（中央）: Mono Saw + 軽いDetune
  - Side（左右）: Wide Supersaw
  → 中央の存在感 + 左右の広がり を両立
```

---

## 12. Pluckリードの完全設計

### 12.1 Pluckサウンドの基本原理

Pluck（撥弦）サウンドは、物理的な弦を弾く動作を模倣します。急速なアタックと自然な減衰が特徴です。

**エンベロープの核心**:
```
理想的なPluckエンベロープ:

Filter Envelope:
  Attack:  0-5 ms   （瞬時にフィルターが開く）
  Decay:   150-400 ms（自然に閉じる）
  Sustain: 0-15%    （ほぼゼロ）
  Release: 50-200 ms（短い余韻）
  Amount:  +40-70%  （大きな変化量）

Amp Envelope:
  Attack:  0-10 ms  （瞬時に音が出る）
  Decay:   200-600 ms（自然に減衰）
  Sustain: 0-20%    （ほぼゼロ）
  Release: 100-300 ms（短い余韻）

ポイント:
  - FilterとAmpの Decay を異なる値にする
  - Filter Decay < Amp Decay で自然な音色変化
  - Filter が先に閉じ、音色が暗くなってから音量が減衰
```

### 12.2 ジャンル別Pluck設定

**Progressive House Pluck（Eric Prydz風）**:
```
Oscillator:
  - Saw波
  - Unison: 2-4
  - Detune: 15-25%

Filter:
  - LP 24dB
  - Cutoff: 1500 Hz
  - Resonance: 10%
  - Env Amount: +55%

Filter Envelope:
  A: 0ms  D: 250ms  S: 5%  R: 150ms

Amp Envelope:
  A: 5ms  D: 400ms  S: 10%  R: 200ms

Effects:
  - Reverb: Hall 2.0s, Send 20%
  - Delay: 1/8 Dotted, Feedback 25%, Send 15%
  - Sidechain: Kick に対して 4:1
```

**Melodic Techno Pluck（Stephan Bodzin風）**:
```
Oscillator:
  - Saw + Square ブレンド（70:30）
  - Unison: 2
  - Detune: 10%

Filter:
  - LP 12dB（穏やかなカット）
  - Cutoff: 2000 Hz
  - Resonance: 25%
  - Env Amount: +40%

Filter Envelope:
  A: 0ms  D: 350ms  S: 10%  R: 200ms

Amp Envelope:
  A: 0ms  D: 500ms  S: 15%  R: 250ms

Effects:
  - Saturator: Drive 2dB, Warm
  - Reverb: Room 1.5s, Send 25%
  - Delay: 1/4, Feedback 35%, Send 20%
```

**Future Bass Pluck（Flume風）**:
```
Oscillator:
  - Wavetable: Digital/Harmonic系
  - Unison: 6-8
  - Detune: 30%
  - Stereo: 80%

Filter:
  - LP 24dB
  - Cutoff: 800 Hz
  - Resonance: 15%
  - Env Amount: +65%

Filter Envelope:
  A: 0ms  D: 200ms  S: 0%  R: 100ms

Amp Envelope:
  A: 0ms  D: 300ms  S: 0%  R: 150ms

LFO → Volume:
  - Rate: 1/8
  - Amount: 100%
  - Shape: Square
  → ゲート/チョップ効果

Effects:
  - OTT（マルチバンドコンプ）: Amount 40%
  - Reverb: Plate 1.5s, Send 30%
  - Chorus: Rate 1Hz, Depth 30%
```

### 12.3 Pluckの音程パターンとシーケンス

```
基本的なPluckパターン（Progressive House）:

パターン1: アルペジオ上昇
  C4 → E4 → G4 → C5 | C4 → E4 → G4 → C5
  1/16ノート、ベロシティ変化あり

パターン2: シンコペーション
  C4 . . E4 | . G4 . . | C5 . . G4 | . E4 . .
  （. = 休符）

パターン3: オクターブジャンプ
  C3 C4 | E3 E4 | G3 G4 | C4 C5

パターン4: コード分散
  Am: A3 C4 E4 A4
  F:  F3 A3 C4 F4
  C:  C3 E3 G3 C4
  G:  G3 B3 D4 G4
```

---

## 13. アルペジオリードの設計

### 13.1 アルペジエーターの基本設定

**Ableton内蔵アルペジエーター**:
```
MIDI Effects → Arpeggiator をドラッグ

基本設定:
  Style: Up（上昇）
  Rate: 1/16（16分音符）
  Gate: 80%（音の長さ）
  Steps: 4
  Octave Range: 2（2オクターブ）

スタイル一覧:
  Up:       C→E→G→C5→E5→G5
  Down:     G5→E5→C5→G→E→C
  Up-Down:  C→E→G→C5→G→E
  Random:   ランダム順
  Converge: 外から内へ
  Diverge:  内から外へ
  Pinky Up: 最高音を繰り返しつつ上昇
  Thumb Up: 最低音を繰り返しつつ上昇
```

### 13.2 アルペジオリードの音色設計

**Trance Arp Lead**:
```
シンセ設定:
  Oscillator: Saw
  Unison: 4（Supersawより控えめ）
  Detune: 25%

Filter:
  LP Cutoff: 3500 Hz
  Resonance: 15%
  Envelope Amount: +30%

Amp Envelope:
  A: 5ms  D: 0ms  S: 100%  R: 150ms
  （Gate長で音の長さを制御）

アルペジエーター:
  Style: Up-Down
  Rate: 1/16
  Gate: 70%
  Octave: 2

Effects:
  - Delay: 1/16 Dotted, Feedback 40%
  → アルペジオが連鎖して複雑なリズムに
  - Reverb: Hall 2.5s, Send 25%
  - Ping Pong Delay: 1/8, Feedback 30%
  → 左右に飛び交うアルペジオ
```

**Techno Arp Lead**:
```
シンセ設定:
  Oscillator: Square + Saw（50:50）
  Unison: 2
  Detune: 10%

Filter:
  LP Cutoff: 1500 Hz
  Resonance: 40%（高め）
  LFO → Cutoff:
    Rate: 1/4
    Amount: 50%

Amp Envelope:
  A: 0ms  D: 100ms  S: 60%  R: 50ms

アルペジエーター:
  Style: Random
  Rate: 1/32（非常に速い）
  Gate: 50%（短い）
  Octave: 3

Effects:
  - Distortion: Drive 4dB
  - Auto Filter: LFO Rate 1/2
  - Reverb: Room 0.8s, Send 15%
```

### 13.3 手動アルペジオプログラミング

**MIDIで手動作成するメリット**:
```
自動アルペジエーターの限界:
  - パターンが機械的
  - ベロシティ変化が単調
  - タイミングのゆらぎがない

手動プログラミングの利点:
  - 意図的なアクセント配置
  - ベロシティのヒューマナイズ
  - タイミングのスウィング
  - 休符の戦略的配置
  - ノートレングスの変化

手順:
  1. アルペジエーターで基本パターン生成
  2. MIDIクリップに「Capture to MIDI」で書き出し
  3. アルペジエーターを無効化
  4. MIDIノートを手動で微調整
  5. ベロシティのランダマイズ（1-5%）
  6. タイミングのヒューマナイズ（5-15ms）
```

---

## 14. 高度なレイヤリングテクニック

### 14.1 周波数帯域別レイヤリング

単なるオクターブ重ねではなく、周波数帯域を明確に分離した専門的なレイヤリング手法です。

**5レイヤーシステム**:
```
Layer 1: Sub Layer（サブ低域）
  周波数: 40-150 Hz
  波形: Sine（正弦波）
  オクターブ: -2（メインの2オクターブ下）
  音量: -18 dB
  処理: Mono、Saturation軽め
  目的: 低域の土台、クラブでの体感

Layer 2: Body Layer（ボディ）
  周波数: 150-800 Hz
  波形: Saw、Unison 2
  オクターブ: -1
  音量: -9 dB
  処理: Mono〜Narrow Stereo
  目的: 音の太さ、存在感の根幹

Layer 3: Main Lead（メインリード）
  周波数: 800-4000 Hz
  波形: Supersaw、Unison 8
  オクターブ: 0（基準）
  音量: 0 dB
  処理: Wide Stereo
  目的: メロディの主役、明瞭さ

Layer 4: Presence Layer（プレゼンス）
  周波数: 4000-10000 Hz
  波形: Bright Saw または Digital Wavetable
  オクターブ: +1
  音量: -12 dB
  処理: Very Wide Stereo
  目的: 明瞭さの補強、空気感

Layer 5: Air/Noise Layer（エアー）
  周波数: 8000-20000 Hz
  波形: Noise + Filter（HP 8kHz）
  音量: -24 dB
  処理: Wide、リバーブ多め
  目的: 超高域のシズル感、空間的な広がり
```

**レイヤー間のEQ処理**:
```
各レイヤーにEQ Eightを挿入:

Layer 1 EQ:
  - LP 150 Hz（上はカット）

Layer 2 EQ:
  - HP 150 Hz / LP 800 Hz（帯域限定）

Layer 3 EQ:
  - HP 300 Hz / LP 5000 Hz（メイン帯域）

Layer 4 EQ:
  - HP 3000 Hz / LP 12000 Hz

Layer 5 EQ:
  - HP 8000 Hz

重要: 隣接レイヤーのクロスオーバーは
  12dB/oct以上のスロープで分離
```

### 14.2 テクスチャレイヤリング

**ノイズレイヤーの追加**:
```
目的: デジタルシンセの無機質さに有機的なテクスチャを追加

手順:
  1. 新規MIDIトラック作成
  2. Simpler / Sampler を挿入
  3. ノイズサンプルをロード:
     - White Noise（全帯域）
     - Pink Noise（低域強調）
     - ヴィンテージテープヒス
     - ヴァイナルクラックル

  4. フィルター設定:
     - HP: 4000-8000 Hz
     - LP: 12000-16000 Hz
     → 必要な帯域だけ通す

  5. Amp Envelope:
     メインリードと同じエンベロープ
     → リードと連動して鳴る

  6. 音量: -20 to -30 dB
     → 聞こえるか聞こえないか程度

効果:
  - アナログ的な温かみ
  - サウンドの「生きている」感
  - マスキング効果で他の音との馴染みが向上
```

**フォルマントレイヤーの追加**:
```
人間の声のような響きをリードに追加:

手順:
  1. Wavetable の Oscillator を追加設定
  2. Wavetable Category: "Vocal" / "Formant"
  3. Position を LFO でスキャン
     → "ah" → "ee" → "oh" のような変化

  4. 音量: -12 to -18 dB（メインリードより小さく）
  5. HP Filter: 1000 Hz（低域をカット）

Vocoder を使う方法:
  1. Vocoder をインサート
  2. Carrier: リードシンセ
  3. Modulator: ボーカルサンプル / フォルマントシンセ
  4. Bands: 20-40
  5. Depth: 60-80%
  → リードシンセが声のように変形
```

### 14.3 パラレル処理によるレイヤリング

```
同一の音源を複数チャンネルに分岐して異なる処理を施す方法:

構成:
  Source: メインリードシンセ（1トラック）
  ↓
  Bus A（Dry）: 未処理の原音
  Bus B（Distorted）: ディストーション処理
  Bus C（Filtered）: フィルター処理
  Bus D（Reverbed）: リバーブ処理
  ↓
  Group トラックでまとめる

Ableton での実装:
  1. リードトラックを Group に入れる
  2. Audio Effect Rack を作成
  3. Chain を4つ作成
  4. 各 Chain に異なるエフェクトを配置
  5. Chain Volume で各チェインのバランス調整

Chain A（Dry）:
  - エフェクトなし
  - Volume: 0 dB

Chain B（Saturated）:
  - Saturator: Drive 8dB, Soft Sine
  - EQ: HP 500 Hz, LP 8000 Hz
  - Volume: -9 dB

Chain C（Pitch Shifted）:
  - Pitch: +12 semitones（1オクターブ上）
  - EQ: HP 4000 Hz
  - Volume: -15 dB

Chain D（Ambient）:
  - Reverb: Hall 4.0s, Dry/Wet 100%
  - EQ: HP 2000 Hz
  - Volume: -12 dB

利点:
  - 原音を保持したまま追加の質感を得る
  - 各チェインの音量を独立してコントロール
  - オートメーションで動的に変化させられる
```

---

## 15. エフェクトチェーン詳細設計

### 15.1 インサートエフェクトの順序理論

エフェクトの接続順序は音質に決定的な影響を与えます。

**推奨エフェクトチェイン（詳細版）**:
```
Signal Flow（信号の流れ）:

1. ゲイン調整（Utility / Trim）
   → 入力レベルの最適化
   → 後段のエフェクトに適切なレベルで入力

2. EQ（前処理）
   → 不要な周波数の除去
   → HP Filter、共振除去
   → 後段のエフェクトに「きれいな信号」を渡す

3. コンプレッション
   → ダイナミクスの制御
   → Ratio: 2:1〜4:1
   → 一定の音量感を確保

4. サチュレーション / ディストーション
   → 倍音の追加
   → 存在感、温かみの付与
   → コンプ後に配置で安定した歪み

5. EQ（後処理 / トーンシェーピング）
   → 歪みで生じた不要倍音の処理
   → プレゼンスブースト
   → 最終的な音色調整

6. モジュレーション系
   → Chorus、Flanger、Phaser
   → ステレオ感、動きの追加

7. ステレオ幅処理（Utility / Wider）
   → 最終的なステレオイメージ
   → Mid/Side バランス

8. 空間系（Send経由が推奨）
   → Reverb、Delay
   → Send で処理することで原音を維持
```

### 15.2 ディストーション/サチュレーション詳細

**種類別の特性**:
```
Soft Clipping（ソフトクリッピング）:
  - 穏やかな歪み
  - 偶数倍音が多い
  - 温かみのある音色
  - 設定: Saturator → Curve: Soft Sine
  - Drive: 2-6 dB
  - 用途: Trance Lead、Progressive

Hard Clipping（ハードクリッピング）:
  - アグレッシブな歪み
  - 奇数+偶数倍音
  - エッジの効いた音色
  - 設定: Saturator → Curve: Hard Curve
  - Drive: 4-10 dB
  - 用途: Techno、Electro

Tube Saturation（真空管）:
  - 偶数倍音優位
  - 非常に温かい
  - コンプレッション効果も付与
  - 設定: Saturator → Curve: A-Shape
  - Drive: 3-8 dB
  - 用途: アナログ風リード

Bitcrusher（ビットクラッシュ）:
  - デジタルの粗さ
  - ローファイ感
  - Redux: Bit Depth 8-12
  - Downsample: 4-16x
  - 用途: Lo-Fi リード、Glitch

Waveshaper（ウェーブシェーパー）:
  - カスタム歪みカーブ
  - 完全にユニークな倍音
  - Serum / Vital 内蔵
  - 用途: 実験的リードサウンド

Multiband Distortion（マルチバンド歪み）:
  - 帯域別に異なる歪み
  - 低域はクリーンに保ちつつ中高域だけ歪ませる
  - 設定例:
    Low（〜300Hz）: Drive 0dB（クリーン）
    Mid（300-3000Hz）: Drive 4dB
    High（3000Hz〜）: Drive 2dB
```

### 15.3 モジュレーションエフェクト詳細

**Chorus（コーラス）**:
```
仕組み: 原音 + 短いディレイ（LFOでモジュレーション）
  → わずかなピッチ変動で厚みを追加

リードシンセ向け設定:
  Rate: 0.3-1.0 Hz（遅め）
  Depth: 20-40%
  Feedback: 0-20%
  Dry/Wet: 20-40%

用途:
  - Supersawにさらなる厚みを追加
  - 単音リードにステレオ感を付与
  - クリーンなリードに温かみを追加
```

**Phaser（フェイザー）**:
```
仕組み: オールパスフィルターで特定周波数の位相を回転
  → ジェット音のような掃引効果

リードシンセ向け設定:
  Rate: 0.1-0.5 Hz（非常に遅い）
  Depth: 40-60%
  Feedback: 30-60%
  Poles: 4-8（多いほど効果が強い）
  Dry/Wet: 30-50%

用途:
  - ブレイクダウンでの音色変化
  - Psytrance系のサイケデリックなリード
  - ロングノートに動きを追加
```

**Flanger（フランジャー）**:
```
仕組み: 原音 + 非常に短いディレイ（LFOでスイープ）
  → 金属的な櫛形フィルター効果

リードシンセ向け設定:
  Rate: 0.05-0.3 Hz（超遅い）
  Depth: 50-80%
  Feedback: 40-70%
  Dry/Wet: 30-50%

用途:
  - ジェットエンジン的なスイープ
  - Techno系のインダストリアルなリード
  - ブレイクダウンの演出
```

### 15.4 空間系エフェクトの詳細設計

**リバーブの種類と使い分け**:
```
Room（ルーム）:
  サイズ: 小〜中
  Decay: 0.5-1.5s
  Pre-Delay: 5-15ms
  用途: Techno、タイトなリード
  特徴: 密集した反射、近い空間

Hall（ホール）:
  サイズ: 大
  Decay: 1.5-4.0s
  Pre-Delay: 20-40ms
  用途: Trance、Epic Lead
  特徴: 広大な空間、長い余韻

Plate（プレート）:
  サイズ: 中
  Decay: 1.0-2.5s
  Pre-Delay: 0-10ms
  用途: Progressive House、Pluck
  特徴: 明るい、拡散が早い

Chamber（チャンバー）:
  サイズ: 小〜中
  Decay: 0.8-2.0s
  Pre-Delay: 10-20ms
  用途: ボーカルリード、フォルマント系
  特徴: 自然な反射、均一

Shimmer（シマー）:
  Decay: 3.0-8.0s
  Pitch Shift: +12 semitones
  用途: Ambient、Chill
  特徴: ピッチシフトされた残響、幻想的

Spring（スプリング）:
  Decay: 0.5-1.5s
  用途: Dub Techno、Lo-Fi
  特徴: バネの振動、独特の金属感
```

**ディレイの種類と使い分け**:
```
Sync Delay（同期ディレイ）:
  Time: 1/8、1/4、1/16
  Feedback: 20-40%
  用途: リズミックな反復

Dotted Delay（付点ディレイ）:
  Time: 1/8 Dotted
  Feedback: 25-35%
  用途: Trance定番、U2のThe Edge的な効果
  特徴: BPMとわずかにずれた反復で独特のグルーヴ

Triplet Delay（3連ディレイ）:
  Time: 1/8 Triplet
  Feedback: 20-30%
  用途: ワルツ的な揺らぎ、Liquid DnB

Ping Pong Delay:
  Time: 1/8 or 1/16
  Feedback: 30-50%
  Width: 100%
  用途: ステレオフィールドの活用
  特徴: 左右に交互に反復

Multi-Tap Delay:
  Tap 1: 1/16, Pan -30%, Volume -3dB
  Tap 2: 1/8, Pan +50%, Volume -6dB
  Tap 3: 1/4, Pan -70%, Volume -9dB
  Tap 4: 1/2, Pan +30%, Volume -12dB
  用途: 複雑な空間デザイン
  特徴: 複数の反復が異なるタイミング・位置で発生

Tape Delay:
  Wow/Flutter: 微量
  Saturation: On
  Feedback: 30-50%（自己発振注意）
  用途: アナログ感のあるリード
  特徴: 反復ごとに音質が劣化する温かい減衰
```

---

## 16. ジャンル別リードサウンド完全ガイド

### 16.1 Psytrance Lead

```
特徴:
  - 非常にアグレッシブ
  - フィルターモジュレーション激しい
  - 1/16以上の高速パターン
  - レゾナンスが高い

Oscillator:
  Osc 1: Saw、Unison 4、Detune 20%
  Osc 2: Square、Detune +5 cents
  Sub: Off

Filter:
  Type: LP 24dB
  Cutoff: 800 Hz
  Resonance: 60%（非常に高い）

  LFO → Cutoff:
    Rate: 1/16
    Amount: 70%
    Shape: Saw Up
    → 高速フィルタースイープ

  Envelope → Cutoff:
    Amount: +50%
    A: 0ms  D: 100ms  S: 20%  R: 50ms

Effects Chain:
  1. Distortion: Drive 6dB
  2. EQ: Boost 1-3kHz +4dB
  3. Flanger: Rate 0.1Hz, Depth 60%
  4. Delay: 1/16 Dotted, FB 40%
  5. Reverb: Room 1.0s, Send 15%

特記事項:
  - BPM 140-150 が標準
  - Gate Trance のような高速パターンが多い
  - Acid 303 ラインとの組み合わせ
```

### 16.2 Synthwave / Retrowave Lead

```
特徴:
  - 80年代シンセサイザーを模倣
  - 太いユニゾン
  - アナログ感のある揺らぎ
  - Chorus/Delayが重要

Oscillator:
  Osc 1: Saw、Unison 6、Detune 35%
  Osc 2: Square（PWM）、Pulse Width: 40%
  Sub: Sine -1oct、-12dB

Filter:
  Type: LP 24dB（Moog風）
  Cutoff: 2500 Hz
  Resonance: 20%

  LFO → Cutoff:
    Rate: 0.3 Hz
    Amount: 15%
    Shape: Sine（遅い揺らぎ）

Amp:
  A: 30ms（わずかにソフト）
  D: 0ms  S: 100%  R: 400ms

LFO → Pitch（ビブラート）:
  Rate: 5.5 Hz
  Amount: 8 cents
  Delay: 500ms（ノートオン後0.5秒で開始）

Effects Chain:
  1. Chorus: Rate 0.5Hz, Depth 40%, Dry/Wet 35%
     → Juno-60風の厚み
  2. EQ: Boost 3kHz +2dB, Air 10kHz +1dB
  3. Saturator: Tube, Drive 3dB
  4. Delay: 1/4, Feedback 35%, Tape風
  5. Reverb: Plate 2.0s, Send 30%

参考アーティスト:
  - The Midnight
  - Kavinsky
  - Perturbator
```

### 16.3 Dubstep Growl Lead

```
特徴:
  - ウォブルベース的なリード
  - LFOモジュレーション極端
  - FM合成やWavetable多用
  - 非常に加工度が高い

Oscillator:
  Osc 1: Wavetable "Growl" / "Digital"
  Osc 2: Saw、FM Amount: 40%
  Unison: 4-8、Detune: 20%

Filter:
  Type: LP 12dB
  Cutoff: 1000 Hz
  Resonance: 35%

  LFO 1 → Cutoff:
    Rate: 1/4（ウォブル）
    Amount: 60%
    Shape: Sine

  LFO 2 → Wavetable Position:
    Rate: 1/8
    Amount: 40%
    Shape: Saw

Modulation Matrix:
  Macro 1 → LFO 1 Rate: 1/16 〜 1/1
  Macro 2 → Filter Cutoff: 200 〜 4000 Hz
  Macro 3 → FM Amount: 0 〜 80%
  Macro 4 → Distortion Drive: 0 〜 12 dB

Effects Chain:
  1. OTT（Multiband Comp）: Amount 50%
  2. Distortion: Hard Clip, Drive 6dB
  3. EQ: HP 100Hz, Boost 800Hz +3dB
  4. Frequency Shifter: 微量（+2-5Hz）
  5. Reverb: Room 0.5s, Send 10%
```

### 16.4 Drum and Bass Lead

```
特徴:
  - Reese Bass をリードとして使用
  - デチューンされた鋸歯波ペア
  - フィルターオートメーション重要
  - BPM 170-180

Oscillator:
  Osc 1: Saw
  Osc 2: Saw、Detune +8 cents
  Osc 3: Saw、Detune -5 cents
  Sub: Sine -1oct、-6dB

Filter:
  Type: LP 24dB
  Cutoff: 変動（オートメーション）
  Resonance: 20-35%

Amp Envelope:
  A: 0ms  D: 0ms  S: 100%  R: 100ms

特殊テクニック - Reeseの作り方:
  1. 2-3基のSaw波をわずかにデチューン
  2. Unison は使わない（手動デチューン）
  3. LFO → Osc 2 Pitch: 1-3 cents の揺らぎ
  4. Filter Cutoff をパターンに合わせてオートメーション
  5. Distortion でエッジを追加

Effects:
  1. Saturator: Warm, Drive 4dB
  2. EQ: HP 60Hz, Notch at 200Hz -3dB
  3. Phaser: Rate 0.1Hz, Subtle
  4. Compressor: 4:1, Fast Attack
```

### 16.5 Lo-Fi / Chill Lead

```
特徴:
  - ローファイ質感
  - テープサチュレーション
  - ピッチの不安定さ
  - ノスタルジック

Oscillator:
  Osc 1: Triangle または Sine
  Osc 2: Off
  Sub: Off

Filter:
  Type: LP 12dB
  Cutoff: 2000-3000 Hz
  Resonance: 5%（低い）

Amp Envelope:
  A: 50ms（ソフト）  D: 0ms  S: 100%  R: 600ms

LFO → Pitch:
  Rate: 4 Hz
  Amount: 3 cents（微量）
  → 古いレコードのような不安定さ

LFO → Volume:
  Rate: 0.1 Hz
  Amount: 5%
  → わずかな音量変動

Effects:
  1. Redux: Bit Depth 12, Downsample 2x
     → 微量のデジタル粗さ
  2. Saturator: Tube, Drive 2dB
     → テープの温かみ
  3. EQ: LP Shelf 8kHz -4dB
     → 高域をロールオフ
  4. Chorus: Rate 0.2Hz, Depth 15%
  5. Reverb: Room 1.5s, Send 20%
  6. Utility: Width 50%（あまり広げない）
```

---

## 17. オートメーション活用ガイド

### 17.1 フィルターオートメーション

**ブレイクダウン → ドロップ のフィルタースイープ**:
```
ブレイクダウン（8小節）:
  小節1: Cutoff 4000 Hz（通常）
  小節2: Cutoff 3000 Hz（徐々に閉じる）
  小節3: Cutoff 2000 Hz
  小節4: Cutoff 1500 Hz
  小節5: Cutoff 1000 Hz
  小節6: Cutoff 500 Hz
  小節7: Cutoff 300 Hz（もごもご）
  小節8: Cutoff 200 Hz → ドロップ直前に急上昇

ドロップ（Beat 1）:
  Cutoff: 200 Hz → 6000 Hz（瞬時に開放）
  → カタルシス、エネルギーの解放

オートメーションカーブ:
  閉じる方向: リニア（直線的に下降）
  開く方向: 指数的（一気に開放）
```

**小節単位のフィルターリズム**:
```
4小節パターン:

小節1: Cutoff 3000Hz（通常）
小節2: Cutoff 4000Hz（少し明るく）
小節3: Cutoff 2500Hz（少し暗く）
小節4: Cutoff 5000Hz（最も明るく）
→ 4小節周期で音色が変化

8ビートパターン:
  Beat 1: 3000Hz
  Beat 2: 2000Hz
  Beat 3: 3500Hz
  Beat 4: 1500Hz
  Beat 5: 4000Hz
  Beat 6: 2000Hz
  Beat 7: 4500Hz
  Beat 8: 1000Hz → リセット
```

### 17.2 ピッチオートメーション

**ピッチベンド効果**:
```
ドロップ前のライザー:
  開始: -24 semitones（2オクターブ下）
  8小節かけて: 0 semitones まで上昇
  最後の1拍: +2 semitones（オーバーシュート）
  ドロップ: 0 semitones に戻る

ダイブ効果:
  通常: 0 semitones
  特定のノート終わり: -12 semitones（急降下）
  復帰: 0 semitones
  → Dubstep、Trap で多用

ビブラートの深さオートメーション:
  ノート開始: ビブラート 0%
  0.5秒後: ビブラート 10%
  1.0秒後: ビブラート 30%
  → ロングノートに表情を追加
```

### 17.3 ステレオ幅オートメーション

```
セクション別のステレオ幅変化:

Intro:
  Width: 40%（控えめ）
  → 空間に余裕を持たせる

Verse:
  Width: 60%（やや広い）
  → リードが登場、適度な存在感

Build-Up:
  Width: 60% → 20%（徐々に狭める）
  → エネルギーを中央に集約
  → ドロップでの解放感を演出

Drop:
  Width: 100%（最大）
  → 一気に広がる爽快感
  → カタルシスの瞬間

Breakdown:
  Width: 100% → 50%（徐々に狭める）
  → 再び集約、次のドロップへの準備
```

### 17.4 エフェクトパラメーターのオートメーション

**リバーブのオートメーション**:
```
通常パート:
  Reverb Send: 20%（控えめ）
  Decay: 2.0s

ブレイクダウン:
  Reverb Send: 20% → 50%（徐々に増加）
  Decay: 2.0s → 4.0s（長くなる）
  → 空間が広がっていく

ドロップ直前（最後の1拍）:
  Reverb Send: 50% → 0%（急にカット）
  → 一瞬の静寂、インパクト

ドロップ:
  Reverb Send: 25%（標準に戻す）
  Decay: 2.0s
```

**ディレイのオートメーション**:
```
フレーズ終わりのディレイスロー:

通常: Delay Time 1/8 Dotted（テンポ同期）
フレーズの最後のノート:
  → Delay Time を Sync Off にして手動で伸ばす
  → Time: 200ms → 800ms（ゆっくり広がる）
  → Feedback: 30% → 60%
  → 次のフレーズ開始前に Feedback: 0%

効果:
  - フレーズの終わりが余韻で飾られる
  - 次のフレーズとの間に自然な「橋」
```

### 17.5 マクロオートメーション

```
1つのMacro Knobで複数パラメーターを同時制御:

Macro "INTENSITY"（Macro 1）:
  0% → 100% で以下が同時変化:

  Filter Cutoff:  1500Hz → 5000Hz
  Resonance:      10%    → 35%
  Unison Detune:  20%    → 50%
  Reverb Send:    15%    → 35%
  Delay Send:     10%    → 25%
  Distortion:     0dB    → 4dB
  Stereo Width:   60%    → 90%

使い方:
  Verse: Macro = 30%
  Chorus: Macro = 70%
  Drop: Macro = 100%
  Breakdown: Macro = 20%

→ 1つのノブを動かすだけでリード全体の印象が変化
→ DJミックス中のようなリアルタイム操作が可能
→ ライブパフォーマンスでMIDIコントローラーにマッピング
```

---

## 18. 実践パッチレシピ集

### 18.1 Epic Trance Anthem Lead

```
目標: 2000年代Tranceの象徴的なスーパーソーリード
参考: Above & Beyond - Sun & Moon

=== シンセ設定（Serum推奨） ===

Osc A:
  Waveform: Default (Saw)
  Unison: 7
  Detune: 0.35
  Blend: 0.50
  Phase: Random
  Level: 100%

Osc B:
  Waveform: Default (Saw)
  Unison: 7
  Detune: 0.30
  Blend: 0.50
  Phase: Random
  Coarse: +12（1オクターブ上）
  Level: 60%（-4dB）

Sub:
  Direct Out（フィルター前）
  Level: 30%
  Octave: -1

Noise:
  Type: White
  Level: 5%
  LP: 12kHz
  → わずかなエアー感

Filter:
  Type: MG Low 24（Moog風LP）
  Cutoff: 75 Hz（MIDI note値）
  Resonance: 15%
  Drive: 0%

  Env 1 → Cutoff: +45
  A: 5ms  D: 800ms  S: 40%  R: 500ms

Amp Envelope（Env 2）:
  A: 15ms  D: 0ms  S: 100%  R: 700ms

=== エフェクト ===

FX1: Hyper / Dimension（ステレオ拡張）
  Mix: 40%  Rate: 0.2Hz

FX2: Compressor
  Threshold: -10dB  Ratio: 3:1  Attack: 10ms  Release: 100ms

FX3: EQ
  HP: 180Hz  Boost: 3.5kHz +3dB  Air: 12kHz +2dB

FX4: Reverb（Insert）
  Size: 70%  Decay: 2.0s  Mix: 15%  Pre-Delay: 25ms

=== DAW側エフェクト ===

Send A: Hall Reverb 3.5s → Send 30%
Send B: 1/8 Dotted Delay, FB 30% → Send 20%
Sidechain: Kick、4:1、Attack 5ms、Release 150ms
```

### 18.2 Melodic Techno Stab

```
目標: Tale Of Us / Afterlife スタイルの暗いスタブ
参考: Tale Of Us - Another Earth

=== シンセ設定（Diva推奨） ===

Osc 1:
  Model: Jup-8 Saw
  Level: 100%

Osc 2:
  Model: Jup-8 Square
  Pulse Width: 60%
  Level: 80%
  Detune: +3 cents

Osc 3:
  Model: Jup-8 Saw
  Octave: -1
  Level: 50%

Filter:
  Model: Ladder（Moog風）
  Cutoff: 800 Hz
  Resonance: 25%
  Env Amount: +60%
  Key Track: 50%

Filter Envelope:
  A: 0ms  D: 200ms  S: 5%  R: 100ms

Amp Envelope:
  A: 2ms  D: 350ms  S: 10%  R: 150ms

=== エフェクト ===

1. Saturator: Tube, Drive 3dB
2. EQ: HP 200Hz, Cut 400Hz -2dB, Boost 2kHz +2dB
3. Compressor: 3:1, Threshold -15dB
4. Send → Reverb: Hall 2.5s, Dark, Pre-Delay 30ms
5. Send → Delay: 1/4, Feedback 35%, HP 500Hz
6. Sidechain: Kick, 5:1, Attack 0ms, Release 200ms

ノート:
  - マイナーキー（Am, Dm, Em が多い）
  - ベロシティ: 80-110（変化をつける）
  - シンコペーションを多用
```

### 18.3 Future Bass Supersaw Chord

```
目標: Illenium / Said The Sky スタイルの感動的コードリード
参考: Illenium - Fractures

=== シンセ設定（Serum推奨） ===

Osc A:
  Waveform: Analog_BD_Saw
  Unison: 7
  Detune: 0.40
  Blend: 0.50
  Level: 100%

Osc B:
  Waveform: Digital → "Square Formant"
  Unison: 5
  Detune: 0.30
  Blend: 0.40
  Coarse: +12
  Level: 40%

Sub: Sine, -1oct, Level 25%

Filter:
  Type: MG Low 18
  Cutoff: 60（低め）
  Resonance: 10%
  Env 1 → Cutoff: +55（大きな開き）

Filter Envelope:
  A: 0ms  D: 400ms  S: 15%  R: 200ms

Amp Envelope:
  A: 5ms  D: 600ms  S: 25%  R: 300ms

LFO 1 → Volume:
  Rate: 1/8
  Shape: Square
  Amount: 100%（Gate: 完全なオン/オフ）
  → サイドチェーン風のチョップ効果

=== エフェクト ===

FX1: OTT（Multiband Compression）
  Amount: 45%（かけすぎ注意）

FX2: Distortion: Tube, Drive 2dB, Mix 30%

FX3: EQ
  HP: 250Hz, Boost 4kHz +3dB

FX4: Reverb
  Plate, Decay 1.8s, Mix 20%

=== DAW側 ===

Send A: Shimmer Reverb → Send 25%
Send B: 1/8 Ping Pong Delay → Send 15%

演奏テクニック:
  - 7thコード、9thコードを多用
  - ボイシング: Close（密集）が基本
  - LFOゲートでリズミックな切り刻み
  - ドロップで LFO Gate OFF → サステインに切り替え
```

### 18.4 Minimal Techno Blip Lead

```
目標: Richie Hawtin / Plastikman スタイルのミニマルリード
参考: Plastikman - Spastik

=== シンセ設定（Operator推奨） ===

Operator A（Carrier）:
  Waveform: Sine
  Coarse: 1
  Level: 100%

Operator B（Modulator）:
  Waveform: Sine
  Coarse: 3（3倍音）
  Level: 20-50%（オートメーション）

Operator C:
  Off

Operator D:
  Off

Algorithm: 1（B → A）

Filter:
  HP: 200 Hz
  LP: 4000 Hz
  Resonance: 15%

Amp Envelope:
  A: 0ms  D: 50-150ms  S: 0%  R: 30ms
  → 非常に短い、ブリップ的

=== エフェクト ===

1. EQ: HP 300Hz（余分な低域除去）
2. Saturator: Soft Sine, Drive 2dB
3. Delay: 1/16, Feedback 45%, HP 500Hz, LP 8000Hz
   → 反復が連なってリズムパターンを形成
4. Reverb: Room 0.5s, Send 10%（最小限）

テクニック:
  - ノートは1-3音（C, D, E程度）のみ
  - リズムが主役（音程変化は最小限）
  - ベロシティの変化で表情をつける
  - Operator B Level のオートメーションで倍音変化
```

### 18.5 Ambient / Chill Pad-Lead

```
目標: Tycho / Boards of Canada スタイルの浮遊リード
参考: Tycho - A Walk

=== シンセ設定（Vital推奨） ===

Osc 1:
  Waveform: Analog → Triangle
  Unison: 3
  Detune: 0.08（わずか）
  Level: 100%

Osc 2:
  Waveform: Wavetable → "Warm Digital"
  WT Position: 0.35
  Unison: 2
  Detune: 0.05
  Level: 70%

Osc 3:
  Waveform: Noise → Pink
  Level: 8%（わずかなテクスチャ）

Filter:
  Type: Analog LP 12dB
  Cutoff: 2200 Hz
  Resonance: 5%（最小限）

LFO 1 → WT Position:
  Rate: 0.08 Hz（非常に遅い）
  Amount: 0.30
  → ゆっくりと音色が変化

LFO 2 → Pitch:
  Rate: 4.5 Hz
  Amount: 4 cents（微量ビブラート）
  Delay: 1.0s（1秒後から開始）

LFO 3 → Filter Cutoff:
  Rate: 0.15 Hz
  Amount: 800 Hz
  → 呼吸するようなフィルター変化

Amp Envelope:
  A: 200ms（ゆっくり立ち上がり）
  D: 0ms
  S: 100%
  R: 2000ms（長い余韻）

=== エフェクト ===

1. Chorus: Rate 0.3Hz, Depth 25%, Mix 30%
2. Phaser: Rate 0.05Hz, Depth 30%, Mix 20%
3. EQ: Gentle LP Shelf 8kHz -3dB（高域をロールオフ）
4. Tape Delay: Time 1/4, Feedback 40%, Wow 5%, Mix 25%
5. Reverb: Hall 5.0s, Pre-Delay 50ms, Mix 35%
6. Utility: Width 70%

ノート:
  - ペンタトニックスケール多用
  - ロングノート（2-4小節）
  - レガートプレイ（音が重なる）
  - ベロシティは均一（80前後）
```

---

## 19. トラブルシューティング詳細

### 19.1 CPU負荷の最適化

```
問題: リードシンセがCPUを過度に消費する

原因と対策:

1. Unison ボイス数が多すぎる
   対策: 16 → 8 に減らす（聴感上の差は小さい）
   節約: CPU 30-50% 削減

2. オーバーサンプリング
   対策: 制作中は 1x、バウンス時に 2x-4x
   節約: CPU 50-75% 削減

3. エフェクトの重複
   対策: Send/Return を活用（1つのリバーブを共有）
   節約: CPU 20-40% 削減

4. リアルタイム処理
   対策: フリーズ（Freeze）機能を活用
   Cmd+Shift+F（Ableton: Freeze Track）
   → トラックをオーディオにレンダリング
   → CPU負荷ゼロ

5. バッファサイズ
   制作時: 512-1024 samples（安定性重視）
   録音時: 128-256 samples（低レイテンシー）
   ミックス時: 2048 samples（最大安定性）
```

### 19.2 位相の問題

```
問題: レイヤーしたリードが薄く聞こえる

原因: 位相キャンセレーション
  → 複数の波形が打ち消し合う

診断:
  1. Correlation Meter を確認
  2. 値が 0 以下 → 位相問題あり
  3. Mono で聴いて音量が著しく下がる → 問題あり

対策:
  1. Utility で位相反転（Phase Invert）を試す
     → 片方のレイヤーの位相を180度回転

  2. わずかなディレイ追加
     → Simple Delay: 0.5-2ms
     → 位相関係を変化させる

  3. EQ で帯域分離を徹底
     → レイヤー間の重複周波数を排除

  4. デチューン量を微調整
     → 完全にゼロでなく 1-3 cents ずらす

  5. ステレオ幅の差別化
     → Layer 1: 60%, Layer 2: 30%
     → 空間的に分離
```

### 19.3 ミックスでの周波数衝突

```
問題: リードがボーカルやパッドと周波数衝突

リードシンセの主要周波数帯域:
  基音: 250-1000 Hz（メロディの音域による）
  プレゼンス: 2-5 kHz
  エアー: 8-12 kHz

衝突しやすい要素:
  ボーカル: 200 Hz - 4 kHz
  パッド: 300 Hz - 6 kHz
  ギター: 200 Hz - 5 kHz

解決テクニック:

1. ダイナミックEQ
   → リード 3kHz にダイナミックバンド
   → ボーカルが鳴っている時だけ自動的にカット
   → ボーカルがない時はリードが全帯域で鳴る

2. サイドチェインEQ
   → FabFilter Pro-Q 3 のサイドチェイン機能
   → ボーカルの信号をトリガーに使用
   → リードの該当帯域を自動ダッキング

3. ステレオ配置の差別化
   → リード: Wide（80%）
   → ボーカル: Center（Mono）
   → パッド: Very Wide（100%）
   → 空間的な住み分け

4. アレンジメントの工夫
   → ボーカルパートではリードを控えめに
   → インストパートでリードを前面に
   → コール&レスポンス的な配置
```

---

## 20. プロフェッショナルワークフロー

### 20.1 音色設計のワークフロー

```
Step 1: リファレンスの選定（10分）
  - 目標とする楽曲を3-5曲選ぶ
  - リード音色の特徴をメモ
  - 周波数帯域、ステレオ幅、エフェクト量を分析

Step 2: 基本波形の選択（5分）
  - リファレンスに最も近い波形を選択
  - Saw / Square / Wavetable / FM

Step 3: Oscillator設定（15分）
  - ユニゾン数、デチューン量
  - 複数オシレーターの組み合わせ
  - ここで70%の音色が決まる

Step 4: フィルター設計（10分）
  - タイプ、カットオフ、レゾナンス
  - エンベロープ設定
  - LFOモジュレーション

Step 5: エフェクト適用（15分）
  - EQ、コンプ、サチュレーション
  - 空間系の調整
  - リファレンスと比較

Step 6: 楽曲コンテキストでの確認（15分）
  - 他のトラックと同時に鳴らす
  - 周波数衝突の確認
  - ステレオ幅の確認
  - サイドチェインの設定

Step 7: 微調整とプリセット保存（10分）
  - 最終的な微調整
  - プリセットとして保存
  - 名前と説明を記録

合計: 約80分 で1つのプロレベルリード音色が完成
```

### 20.2 プリセット管理システム

```
フォルダ構造:

User Presets/
├── Lead/
│   ├── Supersaw/
│   │   ├── Epic_Trance_Lead_v1.fxp
│   │   ├── Progressive_Saw_Lead_v2.fxp
│   │   └── BigRoom_Anthem_v1.fxp
│   ├── Pluck/
│   │   ├── ProgHouse_Pluck_v3.fxp
│   │   ├── MelodicTechno_Stab_v1.fxp
│   │   └── FutureBass_Pluck_v2.fxp
│   ├── Acid/
│   │   ├── 303_Classic_v1.fxp
│   │   └── AcidTechno_Reso_v1.fxp
│   ├── Vocal/
│   │   ├── Formant_Lead_v1.fxp
│   │   └── Vocoder_Saw_v1.fxp
│   └── Experimental/
│       ├── GlitchLead_v1.fxp
│       └── GranularLead_v1.fxp

命名規則:
  [ジャンル]_[タイプ]_[特徴]_v[バージョン]
  例: Trance_Supersaw_EpicWide_v3

メタデータ（メモ帳や Notion で管理）:
  - 作成日
  - 使用シンセ
  - 適するジャンル
  - BPM範囲
  - キー（メジャー/マイナー）
  - 使用楽曲リスト
```

### 20.3 リファレンストラック分析法

```
プロの音色を分析する手順:

1. スペクトラムアナライザーで確認
   - SPAN（無料）またはPro-Q 3
   - リードが鳴っている部分をソロ化（難しい場合はイントロ/アウトロ）
   - ピーク周波数を特定
   - 帯域幅を確認

2. ステレオイメージの分析
   - Correlation Meter
   - Mid/Side 分離で確認
   - 中央に何があるか、左右に何があるか

3. エンベロープの観察
   - アタックの速さ
   - サステインの有無
   - リリースの長さ
   - 波形表示で確認

4. エフェクトの推定
   - リバーブの長さ（テール）
   - ディレイのタイミング
   - ディストーションの有無
   - コーラス/フランジャーの痕跡

5. 再現と比較
   - A/B切り替えで原曲と比較
   - 音量を揃えて比較（ラウドネスの違いに注意）
   - 1つずつパラメーターを近づける
```

---

## 21. 最終チェックリスト

### リードシンセ完成度チェック

```
音色設計:
  [ ] 波形の選択は適切か
  [ ] ユニゾン数とデチューンは最適か
  [ ] フィルター設定は意図通りか
  [ ] エンベロープは音楽的か
  [ ] LFOモジュレーションは効果的か

ミキシング:
  [ ] EQで不要帯域をカットしたか
  [ ] コンプレッションは適度か
  [ ] ステレオ幅は適切か（Mono互換性確認）
  [ ] サイドチェインは設定済みか
  [ ] 他のトラックとの周波数衝突はないか

エフェクト:
  [ ] リバーブの量は適切か（多すぎないか）
  [ ] ディレイのタイミングはBPMに合っているか
  [ ] サチュレーションは音を改善しているか
  [ ] エフェクトの順序は正しいか

パフォーマンス:
  [ ] CPU負荷は許容範囲内か
  [ ] レイテンシーは問題ないか
  [ ] プリセットとして保存したか
  [ ] オートメーションは音楽的に機能しているか

楽曲コンテキスト:
  [ ] ジャンルに適した音色か
  [ ] メロディが明瞭に聞こえるか
  [ ] 感情的なインパクトがあるか
  [ ] DJがプレイしたくなるサウンドか
```

---

**次のステップ**: [パッド](./pads.md) へ進む

---

**楽曲の主役となるリードシンセを完全マスターして、エピックなトラックを作りましょう！**
