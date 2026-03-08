# Operator

FM合成の力。ベル、ブラス、金属的サウンドなど、Wavetableでは作れない複雑な音色をマスターします。

## この章で学ぶこと

- FM合成の基礎
- Operatorの構造（4オシレーター）
- アルゴリズム理解
- オシレーター設定
- エンベロープ活用
- FMベル音色の作り方
- ブラス音色の作り方
- 金属的ベースの作り方

---

## なぜOperatorが重要なのか

**Wavetableにない音色:**

```
音色の特徴:

Wavetable:
温かい
アナログ的
ベース、パッド

Operator:
明るい
金属的
ベル、ブラス

使い分け:

ベース → Wavetable
パッド → Wavetable
ドラム → Drum Rack

ベル → Operator (唯一の選択肢)
ブラス → Operator
金属音 → Operator

プロの使用率:

Wavetable: 70%
Operator: 15%
その他: 15%

用途:

メロディ要素:
ベル
チャイム

アクセント:
金属的ヒット
FXサウンド

特殊ベース:
FM Bass
歪んだベース

歴史:

FM合成:
1980年代 Yamaha DX7
伝説的
```

---

## FM合成とは

**周波数変調:**

### 基本原理

```
通常の合成 (Wavetable):

Oscillator → Filter → 音

FM合成:

Oscillator A (Carrier)
    ↑
    変調
    ↑
Oscillator B (Modulator)

= 複雑な倍音

用語:

Carrier (キャリア):
聴こえる音

Modulator (モジュレーター):
Carrierを変調
聴こえない

変調量:
大きい = 明るい、金属的
小さい = シンプル

効果:

豊富な倍音:
ベル音
金属音
ブラス
```

### Operator vs Wavetable

```
Wavetable:
波形選択 → Filter
シンプル
直感的

Operator:
4つのオシレーター
相互変調
複雑
パワフル

向き不向き:

Wavetable ○:
ベース
パッド
ストリングス

Operator ○:
ベル
ブラス
エレピ
金属音
```

---

## Operator概要 - Ableton Live標準FMシンセサイザー

### Operatorの歴史と背景

Ableton LiveのOperatorは、Yamaha DX7に代表される1980年代のFM合成技術を、現代のDAW環境で使いやすく再構築したシンセサイザーです。DX7は6オペレーター構成でしたが、Operatorは4オペレーター構成に最適化されており、より直感的な操作性を実現しています。

```
FM合成の歴史年表:

1967年: John Chowning（スタンフォード大学）がFM合成を発見
1973年: Yamaha がFM合成のライセンスを取得
1983年: Yamaha DX7 発売 - FM合成が世界的に普及
    - 6オペレーター / 32アルゴリズム
    - 約20万台以上販売
    - 80年代ポップスの象徴的サウンド
1985年: DX7II 発売 - デュアルトーン対応
1987年: TX81Z 発売 - 4オペレーター構成
2001年: Native Instruments FM7 - ソフトウェアFMシンセ
2004年: Ableton Live 4 - Operator初搭載
    - 4オペレーター / 11アルゴリズム
    - Live統合のシンプルUI
2009年: Ableton Live 8 - Operator大幅アップデート
    - フィルタータイプ追加
    - UIリニューアル
現在: Ableton Live 12 - さらに進化
    - MPE対応
    - 高品質フィルター
    - 拡張波形

Operatorの設計思想:
- DX7の複雑さを排除しつつ本質を保持
- 4オペレーター = 十分な表現力 + 操作しやすさ
- Ableton Liveのワークフローに完全統合
- オートメーション・マッピングが容易
```

### Operatorが得意とするサウンド領域

```
非常に得意（Operator推奨）:
├── ベルサウンド全般
│   ├── チャーチベル（教会の鐘）
│   ├── チューブラーベル
│   ├── ガムランベル
│   └── クリスタルベル
├── エレクトリックピアノ
│   ├── DX7 Eピアノ（Rhodesの代替ではない独自の音色）
│   ├── FM Eピアノ（クリスタルクリーン）
│   └── Wurlitzer風FM
├── ブラスサウンド
│   ├── シンセブラス
│   ├── FMホルン
│   └── ブラスセクション
├── マレット系
│   ├── マリンバ
│   ├── ビブラフォン
│   └── シロフォン
└── メタリック/インダストリアル
    ├── 金属的パーカッション
    ├── インダストリアルヒット
    └── ノイズテクスチャ

得意（良い結果が出せる）:
├── FMベース
│   ├── ファンキーFMベース
│   ├── ウォブルベース
│   └── サブベース + FM倍音
├── リード
│   ├── シンセリード
│   ├── オルガン風リード
│   └── ホイッスル音色
└── 効果音/SFX
    ├── レーザー音
    ├── アラーム音
    └── サイレン

あまり得意でない（Wavetable/Analogの方が良い）:
├── 温かいパッド → Wavetable推奨
├── アナログベース → Analog推奨
├── ストリングス → Wavetable推奨
└── ボーカル系 → Wavetable推奨
```

---

## FM合成の基礎理論 - 数学的理解

### 周波数変調の数学

FM合成を深く理解するには、その数学的背景を知ることが重要です。

```
■ 基本的なFM合成の数式

出力 = A × sin(2π × fc × t + I × sin(2π × fm × t))

fc = キャリア周波数（聴こえる基本音の周波数）
fm = モジュレーター周波数（変調する周波数）
A = 振幅（音量）
I = モジュレーションインデックス（変調の深さ）
t = 時間

■ モジュレーションインデックス（I）の影響

I = 0: 純粋なサイン波（倍音なし）
    → 最もシンプル、フルートのような音

I = 1: 軽い倍音
    → 少し明るさが加わる
    → 第1〜第2倍音が出現

I = 2-3: 中程度の倍音
    → ベルらしさが出始める
    → 第1〜第5倍音程度

I = 5-10: 豊かな倍音
    → 明るく金属的
    → ブラスやオルガン的

I = 10以上: 非常に複雑
    → ノイズ的な成分増加
    → メタリック、インダストリアル

■ OperatorにおけるI値の対応

Modulatorの Level パラメーター ≈ モジュレーションインデックス
Level 0% → I ≈ 0（変調なし）
Level 50% → I ≈ 2-3（中程度）
Level 100% → I ≈ 5-10（豊かな倍音）

■ C:M比率（Carrier対Modulator比率）

C:M = 1:1 → 整数倍音のみ → ノコギリ波的
C:M = 1:2 → 奇数・偶数倍音 → クラリネット的
C:M = 1:3 → 第3倍音強調 → ベル的
C:M = 1:4 → 第4倍音強調 → 明るいベル
C:M = 1:0.5 → サブハーモニクス → 太いベース

非整数比率:
C:M = 1:1.41 → 金属的、ベル
C:M = 1:2.76 → ガムラン的
C:M = 1:3.14(π) → 複雑な金属音
C:M = 1:7.13 → ノイズ的パーカッション
```

### サイドバンドの理論

```
■ FM合成で発生する周波数成分（サイドバンド）

キャリア周波数 fc と モジュレーター周波数 fm の場合:

発生する周波数:
fc（基本音）
fc + fm（上側第1サイドバンド）
fc - fm（下側第1サイドバンド）
fc + 2fm（上側第2サイドバンド）
fc - 2fm（下側第2サイドバンド）
fc + 3fm（上側第3サイドバンド）
fc - 3fm（下側第3サイドバンド）
...

■ 具体例: fc = 440Hz, fm = 880Hz (C:M = 1:2)

fc = 440 Hz（基本音 A4）
440 + 880 = 1320 Hz（上側第1）
440 - 880 = -440 Hz → 位相反転で 440 Hz に折り返し
440 + 1760 = 2200 Hz（上側第2）
440 - 1760 = -1320 Hz → 1320 Hz に折り返し

→ 結果: 440, 1320, 2200, 3080... Hz
→ 整数倍音 = 調和的な音

■ 具体例: fc = 440Hz, fm = 619Hz (C:M = 1:1.41)

fc = 440 Hz
440 + 619 = 1059 Hz
440 - 619 = -179 Hz → 179 Hz に折り返し
440 + 1238 = 1678 Hz
440 - 1238 = -798 Hz → 798 Hz に折り返し

→ 結果: 179, 440, 798, 1059, 1678... Hz
→ 非整数倍音 = 不協和 = ベル的/金属的

■ 折り返し（フォールディング）

負の周波数は位相が反転して正の周波数として聴こえる
これがFM合成特有の複雑さを生む

低い fm の場合:
多くの成分が折り返す → 低域が豊か → ベース向き

高い fm の場合:
サイドバンドが広がる → 高域が豊か → ベル/金属向き
```

### 倍音構造の視覚化

```
■ サイン波（FM合成なし、I = 0）

周波数 →
│
│ ██
│ ██
│ ██
│ ██
│ ██
│ ██
└──────────────────────
  f0

■ 軽いFM（I = 1、C:M = 1:2）

周波数 →
│
│ ████
│ ████  ██
│ ████  ██  █
│ ████  ██  █
└──────────────────────
  f0   2f  3f

■ 強いFM（I = 5、C:M = 1:2）

周波数 →
│
│ ████
│ ████ ███
│ ████ ███ ███
│ ████ ███ ███ ██ ██ █ █
│ ████ ███ ███ ██ ██ █ █ ░ ░
└──────────────────────────────
  f0  2f  3f  4f 5f 6f 7f 8f 9f

■ ベル音（I = 3、C:M = 1:1.41 非整数比）

周波数 →
│
│ ████
│ ████    ███
│ ████ ██ ███  ██
│ ████ ██ ███  ██  ██ █ █
└──────────────────────────────
  f0  ?f  ?f  ?f  ?f  (非整数倍音が散在)
```

---

## 4オシレーター構成の詳細

### Operatorの構造

**4オシレーターシステム:**

### 全体像

```
┌─────────────────────────────┐
│ Algorithm (アルゴリズム)     │
│ オシレーター接続方法          │
└─────────────────────────────┘
           │
    ┌──────┴──────┐
    │             │
Oscillator     Oscillator
A (Carrier)    B (Modulator)
    │             │
Oscillator     Oscillator
C (Modulator)  D (Modulator)
    │
    ▼
┌─────────────────────────────┐
│ Filter (オプション)          │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Output                      │
└─────────────────────────────┘

4つのオシレーター:

A, B, C, D:
それぞれ独立

役割:
Carrier または Modulator

接続:
Algorithmで決定
```

### 各オシレーターの詳細パラメーター

```
■ オシレーター共通パラメーター

1. On/Off トグル
   - 各オシレーターを個別にオン/オフ
   - オフにするとCPU節約

2. Waveform（波形選択）
   - Sine（サイン波）★デフォルト・推奨
     → 最も純粋なFM合成結果
     → DX7もサイン波ベース
   - Saw（ノコギリ波）
     → 元から倍音が豊か
     → FM変調でさらに複雑に
   - Square（矩形波）
     → 奇数倍音のみ
     → クラリネット的な質感
   - Triangle（三角波）
     → サイン波に近いがわずかに倍音あり
     → ソフトなFM結果
   - Noise（ノイズ）
     → パーカッション、SFXに
     → ハイハットのモジュレーターに最適
   - User（ユーザー波形）
     → カスタム波形読み込み可能

3. Coarse（粗調整）
   - 0.5x〜48x の範囲
   - 整数値: 整数倍音 → 調和的
   - 非整数値: 非整数倍音 → 非調和的（ベル、金属）
   - 代表的な設定:
     0.5x: 1オクターブ下
     1x: 基本音（ユニゾン）
     2x: 1オクターブ上
     3x: 1オクターブ+5度上
     4x: 2オクターブ上
     5x: 2オクターブ+長3度上
     7x: 2オクターブ+短7度上

4. Fine（微調整）
   - -100〜+100 cent
   - デチューン効果
   - コーラス的な厚み
   - ±5〜10 cent: 温かさ追加
   - ±20〜50 cent: 明確なデチューン
   - ±100 cent: 半音のずれ

5. Level（レベル）
   - 0〜100%
   - Carrierの場合: 出力音量
   - Modulatorの場合: 変調深度（=モジュレーションインデックス）

6. Fixed（固定周波数モード）
   - Off: 通常モード（鍵盤追従）
   - On: 周波数がHz指定（鍵盤に追従しない）
   - 用途: ドラム、SFX、ノイズ系

7. Feedback（フィードバック）
   - 自分自身を変調
   - 0%: 純粋なサイン波
   - 低: 少しノコギリ波的
   - 中: ノコギリ波的
   - 高: ノイズ的
   - OSC Dのみ使用可能（制限）
```

### インターフェイス

```
上部:
┌─────────────────────────────┐
│ Algorithm Selector          │
│ 11種類から選択              │
└─────────────────────────────┘

中央:
┌──────┬──────┬──────┬──────┐
│ OSC A│ OSC B│ OSC C│ OSC D│
│      │      │      │      │
│ Freq │ Freq │ Freq │ Freq │
│ Level│ Level│ Level│ Level│
└──────┴──────┴──────┴──────┘

下部:
┌─────────────────────────────┐
│ Envelope × 4 (各OSC用)      │
│ ADSR表示                    │
└─────────────────────────────┘

右側:
┌─────────────────────────────┐
│ Global Controls             │
│ Volume, Pan, Transpose      │
└─────────────────────────────┘
```

---

## Algorithm（アルゴリズム）完全解説

**接続パターン:**

### 全11アルゴリズム詳細

```
■ Algorithm 1（基本FM）
D→C→B→A → [OUT]
直列接続（4段）

特徴:
- 最もFM的な音色
- 非常に複雑な倍音構造
- 変調が連鎖するため少しの調整で大きく変化
用途: 複雑なベル、金属音、実験的サウンド
難易度: ★★★★★（最も制御が難しい）

■ Algorithm 2（3段 + 並列）
C→B→A → [OUT]
D → [OUT]

特徴:
- A はBから変調（Bは Cから変調）
- D は独立したキャリア
- 2つの音色を同時に出力
用途: ベル + サブトーン、レイヤードサウンド
難易度: ★★★☆☆

■ Algorithm 3（2段+2段 並列）
B→A → [OUT]
D→C → [OUT]

特徴:
- 2つの独立したFMペア
- それぞれ別の音色を作れる
- ミックスバランスで音色調整
用途: デュアルトーンベル、厚みのあるパッド
難易度: ★★★☆☆

■ Algorithm 4（2段+1+1 並列）
B→A → [OUT]
C → [OUT]
D → [OUT]

特徴:
- 1つのFMペア + 2つの独立キャリア
- 3つの音源を同時に使用
用途: リッチなアンビエント、複合テクスチャ
難易度: ★★☆☆☆

■ Algorithm 5（全並列）
A → [OUT]
B → [OUT]
C → [OUT]
D → [OUT]

特徴:
- 全オシレーターが独立キャリア
- FM変調なし（アディティブ合成）
- 各オシレーターの波形・周波数・音量で音色構成
用途: オルガン、アディティブ合成、コード
難易度: ★☆☆☆☆（最も簡単）

■ Algorithm 6（フィードバック直列）
D(FB)→C→B→A → [OUT]
（Dにフィードバックあり）

特徴:
- Algorithm 1 + Dのフィードバック
- フィードバックでDがノコギリ波的に
- より豊かな変調源
用途: 複雑なリード、アグレッシブサウンド
難易度: ★★★★★

■ Algorithm 7（ツインモジュレーター）
C→A → [OUT]
D→B → [OUT]
（ただしBもAを変調）

特徴:
- Cが A を変調、D が B を変調
- B が A をさらに変調
- 複雑な変調経路
用途: リッチなブラス、複雑なテクスチャ
難易度: ★★★★☆

■ Algorithm 8（デュアルモジュレーター）
B→A → [OUT]
C→A → [OUT]（BとCが両方Aを変調）
D → [独立]

特徴:
- Aが2つのモジュレーターから変調される
- 異なる倍音特性の合成
- ブラスに最適
用途: ブラス、リード、複雑な倍音
難易度: ★★★☆☆

■ Algorithm 9（クロスモジュレーション）
A ←→ B（相互変調）
C → [OUT]
D → [OUT]

特徴:
- AとBが互いを変調
- カオス的な結果になりやすい
- 実験的サウンド
用途: グリッチ、ノイズ、実験的SFX
難易度: ★★★★★

■ Algorithm 10（リングモジュレーション的）
B→A → [OUT]
D→C → [OUT]
（AとCが互いを変調）

特徴:
- 2組のFMペアが相互作用
- リングモジュレーション的な効果
用途: メタリックサウンド、ベル、SFX
難易度: ★★★★☆

■ Algorithm 11（フィードバック並列）
A(FB) → [OUT]
B(FB) → [OUT]
C(FB) → [OUT]
D(FB) → [OUT]

特徴:
- 全オシレーターにフィードバック
- 各オシレーターが自己変調
- フィードバック量でサイン波〜ノコギリ波
用途: オルガン、アディティブ+倍音制御
難易度: ★★☆☆☆
```

### アルゴリズム選択ガイド

```
■ 音色別推奨アルゴリズム

ベルサウンド:
  第1候補: Algorithm 1（最もベル的）
  第2候補: Algorithm 2（ベル + サブ）
  第3候補: Algorithm 3（デュアルベル）

ブラスサウンド:
  第1候補: Algorithm 8（デュアルモジュレーター）
  第2候補: Algorithm 7（ツインモジュレーター）
  第3候補: Algorithm 2（3段変調）

エレクトリックピアノ:
  第1候補: Algorithm 1（クラシックDX EP）
  第2候補: Algorithm 2（EP + サブ）
  第3候補: Algorithm 8（リッチEP）

ベース:
  第1候補: Algorithm 1（FMベース）
  第2候補: Algorithm 8（厚いベース）
  第3候補: Algorithm 5（アディティブベース）

パッド/アンビエント:
  第1候補: Algorithm 5（アディティブ）
  第2候補: Algorithm 4（レイヤード）
  第3候補: Algorithm 3（デュアルFMパッド）

パーカッション/SFX:
  第1候補: Algorithm 9（カオス的）
  第2候補: Algorithm 10（メタリック）
  第3候補: Algorithm 1（FM打楽器）

オルガン:
  第1候補: Algorithm 5（アディティブ）
  第2候補: Algorithm 11（FB並列）
  第3候補: Algorithm 4（3トーン）
```

---

## エンベロープ詳細解説

**各OSCに個別:**

### 4つのEnvelope

```
特徴:
各オシレーターが独立したADSR

ベル音:

OSC A Envelope:
Attack: 0 ms
Decay: 2000 ms (長い)
Sustain: 0%
Release: 1000 ms

→ 余韻が長い

OSC B Envelope:
Attack: 0 ms
Decay: 500 ms (短い)
Sustain: 20%
Release: 300 ms

→ アタックが明るい

効果:
時間とともに音色変化
リアルなベル

ブラス:

全OSC共通:
Attack: 100-300 ms
徐々に明るく

Decay: 500 ms
Sustain: 70%
Release: 300 ms

効果:
ブレスの感じ
```

### エンベロープの役割と重要性

```
■ CarrierエンベロープとModulatorエンベロープの違い

Carrierのエンベロープ:
→ 音量の時間変化を制御
→ 音が聞こえる/聞こえないを決定
→ 通常のシンセのAmp Envelopeと同じ

Modulatorのエンベロープ:
→ 変調量（倍音の豊かさ）の時間変化を制御
→ 音色の時間変化を決定
→ FMシンセ最大の武器

例: ベルサウンド
- Carrier: 長いDecay（余韻）
- Modulator: 短いDecay（アタックだけ明るい）
→ 「カーン」: 最初明るく、徐々に暗くなる = 自然なベル

例: ブラスサウンド
- Carrier: 速いAttack、高いSustain
- Modulator: 遅いAttack
→ 最初暗く、徐々に明るくなる = ブレスの感じ

■ エンベロープの各パラメーター詳細

Attack Time:
- 0 ms: 即座に最大（打楽器、ベル）
- 1-50 ms: パーカッシブだが若干ソフト
- 50-200 ms: ブラス的な立ち上がり
- 200-500 ms: パッド的、ストリングス的
- 500 ms以上: 非常にゆっくり、アンビエント

Attack Level:
- 通常は100%（最大）
- 低く設定: アタックが控えめ

Decay Time:
- 0-100 ms: 非常に短い、パーカッシブ
- 100-500 ms: スタッカート的
- 500-2000 ms: 自然な減衰
- 2000-5000 ms: 長い余韻（ベル）
- 5000 ms以上: 非常に長い余韻

Sustain Level:
- 0%: 完全に減衰（ベル、マリンバ）
- 20-40%: わずかに持続
- 50-70%: 中程度（ブラス）
- 80-100%: 強い持続（オルガン、パッド）

Release Time:
- 0-50 ms: 即座にカット
- 50-200 ms: 短い余韻
- 200-500 ms: 自然な余韻
- 500-2000 ms: 長い余韻
- 2000 ms以上: 非常に長い（アンビエント）

■ エンベロープループモード

Operatorのエンベロープには特殊なモード:
- None: 通常のADSR（デフォルト）
- Loop: Attack→Decay区間をループ
- Beat: テンポ同期ループ
- Sync: テンポ同期

Loopモード活用:
- LFO的な効果
- リズミックな音色変化
- パルス的なテクスチャ
```

### Oscillator設定

**4つのオシレーター:**

### Frequency（周波数）

```
Coarse:
倍音比率
1x, 2x, 3x...

Fine:
微調整
-100 〜 +100 cent

Fixed:
固定周波数
Hz指定

ベル音設定:

OSC A (Carrier):
Coarse: 1x
基音

OSC B (Modulator):
Coarse: 4x
4倍音
明るい音

OSC C:
Coarse: 2.5x
非整数倍
金属的

ブラス設定:

OSC A:
Coarse: 1x

OSC B:
Coarse: 1x
Fine: +5 cent
わずかにデチューン

OSC C:
Coarse: 2x
```

### Level（レベル）

```
機能:
オシレーターの音量

Carrier:
聴こえる音量

Modulator:
変調量
= 倍音の豊かさ

ベル音:

OSC A (Carrier):
Level: 100%

OSC B (Modulator):
Level: 50-80%
適度な変調

ブラス:

OSC A:
Level: 100%

OSC B, C:
Level: 30-50%
控えめ
```

---

## Oscillator設定

**4つのオシレーター:**

### Frequency（周波数）

```
Coarse:
倍音比率
1x, 2x, 3x...

Fine:
微調整
-100 〜 +100 cent

Fixed:
固定周波数
Hz指定

ベル音設定:

OSC A (Carrier):
Coarse: 1x
基音

OSC B (Modulator):
Coarse: 4x
4倍音
明るい音

OSC C:
Coarse: 2.5x
非整数倍
金属的

ブラス設定:

OSC A:
Coarse: 1x

OSC B:
Coarse: 1x
Fine: +5 cent
わずかにデチューン

OSC C:
Coarse: 2x
```

### Level（レベル）

```
機能:
オシレーターの音量

Carrier:
聴こえる音量

Modulator:
変調量
= 倍音の豊かさ

ベル音:

OSC A (Carrier):
Level: 100%

OSC B (Modulator):
Level: 50-80%
適度な変調

ブラス:

OSC A:
Level: 100%

OSC B, C:
Level: 30-50%
控えめ
```

---

## Envelope（エンベロープ）

**各OSCに個別:**

### 4つのEnvelope

```
特徴:
各オシレーターが独立したADSR

ベル音:

OSC A Envelope:
Attack: 0 ms
Decay: 2000 ms (長い)
Sustain: 0%
Release: 1000 ms

→ 余韻が長い

OSC B Envelope:
Attack: 0 ms
Decay: 500 ms (短い)
Sustain: 20%
Release: 300 ms

→ アタックが明るい

効果:
時間とともに音色変化
リアルなベル

ブラス:

全OSC共通:
Attack: 100-300 ms
徐々に明るく

Decay: 500 ms
Sustain: 70%
Release: 300 ms

効果:
ブレスの感じ
```

---

## Filter（フィルター）

**オプション:**

### Filter設定

```
Operator のFilter:
必須ではない

理由:
FM自体が音色制御

使用する場合:

Type:
Lowpass

Cutoff:
2000-5000 Hz
高め

Resonance:
0-20%
控えめ

用途:
さらに暗くしたい
高域カット

ベル音:
Filterなし (推奨)

ブラス:
Cutoff 3000 Hz
わずかに暗く
```

---

## 実践: FM Bellの作り方

**10分で完成:**

### Step 1: 初期化 (1分)

```
1. 新規MIDIトラック

2. Operator挿入:
   Browser > Instruments > Operator

3. Init:
   右クリック > Initialize

4. 確認:
   C4ノート入力
```

### Step 2: Algorithm選択 (1分)

```
Algorithm:
Algorithm 1

理由:
シンプルなFM
A ← B

表示:
A: Carrier
B: Modulator
C, D: Off
```

### Step 3: Oscillator設定 (4分)

```
OSC A (Carrier):

Frequency:
Coarse: 1x (基音)

Level:
100%

OSC B (Modulator):

Frequency:
Coarse: 4x (4倍音)

Level:
70%

効果:
明るい金属的音

OSC C:

Frequency:
Coarse: 2.5x

Level:
30%

On: ☑

効果:
非整数倍 = より金属的

OSC D:

Off (使わない)
```

### Step 4: Envelope設定 (3分)

```
OSC A Envelope:

Attack: 0 ms
Decay: 3000 ms (長い余韻)
Sustain: 0%
Release: 2000 ms

OSC B Envelope:

Attack: 0 ms
Decay: 800 ms (短め)
Sustain: 10%
Release: 500 ms

効果:
アタック明るい
徐々に暗くなる

OSC C Envelope:

Attack: 0 ms
Decay: 1500 ms
Sustain: 0%
Release: 1000 ms
```

### Step 5: 仕上げ (1分)

```
Global:

Transpose:
+12 st (1オクターブ上)
ベルは高音域

Time:
100%
Envelope速度

Volume:
-3 dB

確認:
C5, E5, G5 コード
美しいベル音

保存:
"My FM Bell"
```

---

## 実践: Brassの作り方

**12分で完成:**

### Step 1: Algorithm (2分)

```
Algorithm:
Algorithm 8

構造:
A (Carrier)
↑ ↑
B  C (両方がAを変調)

効果:
複雑な倍音
ブラスらしい
```

### Step 2: Oscillator (5分)

```
OSC A:

Coarse: 1x
Level: 100%

OSC B:

Coarse: 1x
Fine: +5 cent
Level: 40%

効果:
わずかなデチューン
厚み

OSC C:

Coarse: 2x (2倍音)
Level: 35%

効果:
明るさ

OSC D:

Off
```

### Step 3: Envelope (4分)

```
全OSC共通設定:

Attack: 200 ms
徐々に明るく
ブレスの感じ

Decay: 400 ms

Sustain: 75%

Release: 250 ms

OSC B のみ:
Attack: 150 ms
わずかに速く
```

### Step 4: Filter (1分)

```
Filter:

Type: Lowpass

Cutoff: 3500 Hz

Resonance: 15%

効果:
わずかに暗く
アナログ的
```

### Step 5: 仕上げ

```
Global:

Volume: -6 dB
Transpose: 0

確認:
C3 - C4 範囲
ブラスセクション

保存:
"My Brass"
```

---

## フィルターセクション詳細

OperatorにはFM合成後の信号をさらに加工するためのフィルターセクションが搭載されています。FM合成自体が強力な倍音制御を持つため、フィルターは補助的な役割ですが、適切に使えばサウンドの幅が大きく広がります。

### フィルタータイプ一覧

```
■ Lowpass（ローパス）フィルター
- 指定周波数以上をカット
- FM音色を暗く、温かくする
- 最も使用頻度が高い

  LP 12dB/oct: 穏やかなカット
  LP 24dB/oct: 急峻なカット（推奨）

  用途:
  - FM音色が明るすぎる場合の調整
  - アナログシンセ的な質感付加
  - ベースサウンドの低域強調

■ Highpass（ハイパス）フィルター
- 指定周波数以下をカット
- 低域をカットして透明感を出す

  HP 12dB/oct: 穏やかなカット
  HP 24dB/oct: 急峻なカット

  用途:
  - パーカッション的アタックの強調
  - SFXのハイファイ化
  - ミックス中の棲み分け

■ Bandpass（バンドパス）フィルター
- 指定周波数帯のみを通す
- 電話的、ラジオ的な質感

  BP 12dB: 穏やか
  BP 24dB: ナローバンド

  用途:
  - ボコーダー的効果
  - 遠くの音の演出
  - 特定帯域の強調

■ Notch（ノッチ）フィルター
- 指定周波数帯をカット
- フェイザー的な効果

  用途:
  - 特定周波数のカット
  - フェイザー効果
  - 共鳴の除去

■ Morphフィルター
- LP → BP → HP をシームレスに変化
- Morphノブでフィルタータイプを連続変化
- オートメーションに最適
```

### フィルターエンベロープ

```
■ フィルター専用エンベロープ

OperatorにはOSCエンベロープとは別にフィルターエンベロープが装備:

Attack: フィルターが開く速度
Decay: フィルターが閉じるまでの時間
Sustain: 保持するフィルターの開き具合
Release: ノートオフ後のフィルター変化

■ フィルターエンベロープの活用例

ワウ効果:
- Attack: 0ms
- Decay: 300-800ms
- Sustain: 30%
- Release: 200ms
- Cutoff: 800Hz（低め）
- Envelope Amount: +60%
→ 「ワウ」と開閉するサウンド

プラック効果:
- Attack: 0ms
- Decay: 100-200ms
- Sustain: 0%
- Release: 50ms
- Cutoff: 500Hz
- Envelope Amount: +80%
→ シャープなプラック音

スウェル効果:
- Attack: 500-2000ms
- Decay: 0ms
- Sustain: 100%
- Release: 1000ms
- Cutoff: 1000Hz
- Envelope Amount: +50%
→ 徐々に明るくなる
```

---

## LFO（Low Frequency Oscillator）詳細

OperatorのLFOは音色に周期的な変化を加える重要なモジュレーションソースです。

### LFOの基本パラメーター

```
■ LFO波形タイプ

Sine（サイン波）:
- 滑らかな周期変化
- ビブラート、トレモロに最適
- 最も自然な変化

Triangle（三角波）:
- サインよりやや角のある変化
- ビブラートの代替

Square（矩形波）:
- オン/オフの切り替え
- トリル効果
- ゲートシーケンス的

Saw Up（ノコギリ波・上昇）:
- 徐々に上昇→急降下
- フィルタースイープ
- テクノ的な変化

Saw Down（ノコギリ波・下降）:
- 急上昇→徐々に下降
- 逆方向のスイープ

Sample & Hold（サンプル＆ホールド）:
- ランダムな値をステップ状に保持
- グリッチ的効果
- ランダムアルペジオ風

Noise（ノイズ）:
- 完全にランダム
- 揺らぎの追加

■ LFO Rate（速度）

Freeモード:
- 0.01Hz〜30Hz
- テンポに依存しない
- ビブラート: 4-7Hz が最適
- トレモロ: 2-10Hz
- スロー変調: 0.1-1Hz

Syncモード:
- テンポ同期
- 1/1, 1/2, 1/4, 1/8, 1/16 etc.
- 楽曲のグルーブに合った変化
- トランス的フィルタースイープに最適

■ LFO Amount（深さ）

0%: 効果なし
1-10%: 微妙な揺らぎ（推奨開始値）
10-30%: 明確な効果
30-60%: 強い変化
60-100%: 極端な効果

■ LFO Destination（変調先）

Operatorで指定可能な変調先:
- OSC A/B/C/D Level → トレモロ/AM効果
- OSC A/B/C/D Pitch → ビブラート
- Filter Cutoff → ワウワウ効果
- Filter Resonance → レゾナンス揺らぎ
- Global Pitch → 全体ビブラート

■ LFO Fade In

- LFO効果が徐々に増加
- 0ms: 即座にフル効果
- 100-500ms: 自然な立ち上がり
- ビブラートの「後から加わる」効果に最適
```

### LFO実践レシピ

```
■ 自然なビブラート

Wave: Sine
Rate: 5.5 Hz（Free）
Amount: 8%
Destination: OSC A Pitch
Fade In: 300ms
→ 管楽器的な自然なビブラート

■ テクノ的フィルタースイープ

Wave: Saw Up
Rate: 1/2（Sync）
Amount: 60%
Destination: Filter Cutoff
→ ハーフノートごとのフィルター変化

■ トレモロ効果

Wave: Sine
Rate: 7 Hz
Amount: 30%
Destination: OSC A Level
→ 音量の周期的変化

■ グリッチ的ピッチ変化

Wave: Sample & Hold
Rate: 1/16（Sync）
Amount: 15%
Destination: Global Pitch
→ 16分音符ごとにランダムなピッチ変化

■ ゆっくりした音色変化

Wave: Triangle
Rate: 0.2 Hz
Amount: 40%
Destination: OSC B Level（Modulator）
→ FM変調量が5秒周期で変化、動きのあるサウンド
```

---

## グローバルパラメーター

### Globalセクションの全パラメーター

```
■ Volume（ボリューム）
- 全体の出力音量
- -inf〜+6 dB
- 推奨: -6〜-3 dB（ヘッドルーム確保）

■ Pan（パン）
- 左右の定位
- -50（L）〜 0（C）〜 +50（R）
- 通常は中央（0）

■ Transpose（トランスポーズ）
- 全体のピッチを半音単位で変更
- -48〜+48 semitones
- ベル: +12（1オクターブ上）推奨
- ベース: 0 または -12

■ Spread（スプレッド）
- ステレオ幅
- 0%: モノラル
- 50%: 中程度のステレオ
- 100%: ワイドステレオ
- 注意: ベースには控えめに

■ Time（タイム）
- 全エンベロープの速度を一括変更
- 50%: エンベロープが2倍速
- 100%: 通常速度
- 200%: エンベロープが半分の速度
- ベロシティ対応にも使える

■ Tone（トーン）
- 全体的な明るさ調整
- 低い値: 暗い（ローパス的効果）
- 高い値: 明るい
- FM音色が明るすぎる場合の簡易調整

■ Velocity（ベロシティ感度）
- Volume Velocity: 音量のベロシティ応答
  0%: ベロシティ無視
  100%: 最大応答

- Filter Velocity: フィルターのベロシティ応答
  強く弾く → フィルターが開く

- Time Velocity: エンベロープ速度のベロシティ応答
  強く弾く → エンベロープが速くなる
  弱く弾く → エンベロープが遅くなる

■ Pitch Bend Range
- ピッチベンドの範囲設定
- 1〜24 semitones
- 2 semitones: 標準
- 12 semitones: 1オクターブ（特殊効果向け）

■ Glide（グライド/ポルタメント）
- ノート間のピッチスライド
- Time: スライドにかかる時間
- Mode:
  Off: グライドなし
  Glide: 常にスライド
  Portamento: レガートプレイ時のみスライド
```

---

## 実践サウンドデザイン: ベルサウンドバリエーション

### チャーチベル（教会の鐘）

```
■ 設定

Algorithm: 1（D→C→B→A）

OSC A (Carrier):
  Waveform: Sine
  Coarse: 1x
  Level: 100%
  Envelope: A=0, D=5000ms, S=0%, R=3000ms

OSC B (Modulator):
  Waveform: Sine
  Coarse: 3.5x（非整数 → 金属的）
  Level: 85%
  Envelope: A=0, D=2000ms, S=10%, R=1500ms

OSC C (Modulator):
  Waveform: Sine
  Coarse: 7.1x（高い非整数比）
  Level: 40%
  Envelope: A=0, D=800ms, S=0%, R=500ms

OSC D (Modulator):
  Waveform: Sine
  Coarse: 11.3x
  Level: 25%
  Envelope: A=0, D=400ms, S=0%, R=200ms

Global:
  Transpose: +12
  Spread: 30%
  Volume: -6 dB

特徴:
- 複数の非整数倍音による豊かな金属的響き
- 各モジュレーターのDecayが段階的に短い
  → アタックは複雑、余韻はシンプル
- 教会の鐘のような荘厳な響き
```

### クリスタルベル

```
■ 設定

Algorithm: 3（B→A + D→C 並列）

OSC A (Carrier):
  Waveform: Sine
  Coarse: 1x
  Level: 80%
  Envelope: A=0, D=3000ms, S=0%, R=2000ms

OSC B (Modulator):
  Waveform: Sine
  Coarse: 5x（高い整数倍 → 明るい）
  Level: 60%
  Envelope: A=0, D=1000ms, S=0%, R=500ms

OSC C (Carrier):
  Waveform: Sine
  Coarse: 4x
  Level: 40%
  Envelope: A=0, D=2000ms, S=0%, R=1500ms

OSC D (Modulator):
  Waveform: Sine
  Coarse: 9x
  Level: 30%
  Envelope: A=0, D=600ms, S=0%, R=300ms

Global:
  Transpose: +24（2オクターブ上）
  Spread: 50%
  Volume: -9 dB

特徴:
- 2つの独立FMペアによる複雑な倍音
- 高いトランスポーズで繊細な質感
- アンビエント、チルアウトに最適
```

### メタリックサウンド

```
■ インダストリアルヒット

Algorithm: 9（クロスモジュレーション）

OSC A:
  Waveform: Sine
  Coarse: 1x
  Level: 100%
  Envelope: A=0, D=800ms, S=0%, R=400ms

OSC B:
  Waveform: Sine
  Coarse: 1.41x（√2、非常に不協和）
  Level: 100%（最大変調）
  Envelope: A=0, D=600ms, S=0%, R=300ms

OSC C:
  Waveform: Noise
  Coarse: -
  Level: 30%
  Envelope: A=0, D=100ms, S=0%, R=50ms

OSC D:
  Off

Filter:
  Type: Bandpass
  Cutoff: 2000Hz
  Resonance: 40%

特徴:
- 相互変調による非常に複雑な倍音
- ノイズバーストのアタック
- バンドパスで特定帯域を強調
- テクノ、インダストリアルに最適
```

---

## 実践サウンドデザイン: FMベース

### クラシックFMベース（DXベース）

```
■ 設定

Algorithm: 1（D→C→B→A）

OSC A (Carrier):
  Waveform: Sine
  Coarse: 1x
  Level: 100%
  Envelope: A=0, D=300ms, S=80%, R=100ms

OSC B (Modulator):
  Waveform: Sine
  Coarse: 1x（ユニゾン）
  Level: 55%
  Envelope: A=0, D=200ms, S=30%, R=80ms

OSC C:
  Off

OSC D:
  Off

Filter:
  Type: LP 24dB
  Cutoff: 2500Hz
  Resonance: 15%
  Envelope: A=0, D=150ms, S=20%, R=80ms
  Env Amount: +40%

Global:
  Transpose: 0
  Volume: -3 dB
  Glide: 50ms（Portamento）

特徴:
- シンプルな1:1比率のFM
- パンチのあるアタック
- フィルターエンベロープでさらにダイナミック
- ファンク、ハウスに最適

■ ベロシティ設定
Volume Velocity: 40%
Filter Velocity: 60%
→ 強く弾くと明るく大きな音
```

### ウォブルFMベース

```
■ 設定

Algorithm: 8（B→A, C→A デュアルモジュレーター）

OSC A (Carrier):
  Waveform: Sine
  Coarse: 0.5x（1オクターブ下 = サブ）
  Level: 100%
  Envelope: A=0, D=0, S=100%, R=100ms

OSC B (Modulator):
  Waveform: Sine
  Coarse: 1x
  Level: 70%
  Envelope: A=0, D=0, S=100%, R=100ms

OSC C (Modulator):
  Waveform: Sine
  Coarse: 3x
  Level: 50%
  Envelope: A=0, D=0, S=100%, R=100ms

OSC D:
  Off

LFO:
  Wave: Sine
  Rate: 1/4 (Sync)
  Amount: 70%
  Destination: OSC B Level + OSC C Level

Filter:
  Type: LP 24dB
  Cutoff: 1500Hz
  Resonance: 30%

特徴:
- LFOが変調量を周期的に変化
- 4分音符ごとのウォブル効果
- ダブステップ、ドラムンベースに最適
- LFO Rateを変えてグルーブ調整
```

---

## 実践サウンドデザイン: パッドデザイン

### FMアンビエントパッド

```
■ 設定

Algorithm: 5（全並列 = アディティブ合成）

OSC A:
  Waveform: Sine
  Coarse: 1x
  Level: 80%
  Envelope: A=2000ms, D=0, S=100%, R=3000ms

OSC B:
  Waveform: Sine
  Coarse: 2x
  Fine: +3 cent
  Level: 40%
  Envelope: A=3000ms, D=0, S=100%, R=4000ms

OSC C:
  Waveform: Sine
  Coarse: 3x
  Fine: -5 cent
  Level: 25%
  Envelope: A=4000ms, D=0, S=80%, R=5000ms

OSC D:
  Waveform: Sine
  Coarse: 5x
  Fine: +7 cent
  Level: 15%
  Envelope: A=5000ms, D=1000, S=60%, R=6000ms

LFO:
  Wave: Sine
  Rate: 0.15Hz（非常に遅い）
  Amount: 20%
  Destination: OSC B Level + OSC D Level

Global:
  Spread: 80%
  Volume: -6 dB

特徴:
- アディティブ合成で倍音を個別制御
- 各オシレーターのAttackをずらして展開感
- 微妙なデチューンでコーラス効果
- 遅いLFOで生きている感じ
- アンビエント、チルアウトに最適
```

### ダークFMパッド

```
■ 設定

Algorithm: 3（B→A + D→C 並列）

OSC A (Carrier):
  Waveform: Sine
  Coarse: 1x
  Level: 90%
  Envelope: A=3000ms, D=2000ms, S=70%, R=4000ms

OSC B (Modulator):
  Waveform: Sine
  Coarse: 2.1x（非整数 → 不穏）
  Level: 35%
  Envelope: A=4000ms, D=3000ms, S=20%, R=2000ms

OSC C (Carrier):
  Waveform: Sine
  Coarse: 0.5x（1オクターブ下）
  Level: 60%
  Envelope: A=2000ms, D=0, S=100%, R=5000ms

OSC D (Modulator):
  Waveform: Sine
  Coarse: 3.7x（非整数）
  Level: 20%
  Envelope: A=5000ms, D=4000ms, S=10%, R=3000ms

Filter:
  Type: LP 24dB
  Cutoff: 1200Hz
  Resonance: 20%
  Envelope: A=3000ms, D=0, S=100%, R=2000ms
  Env Amount: +30%

Global:
  Spread: 60%
  Tone: -20
  Volume: -6 dB

特徴:
- 非整数比のモジュレーターで不穏な倍音
- 低いフィルターカットオフでダークな質感
- モジュレーターのAttackが遅い = 徐々に不穏に
- ダークアンビエント、ホラーに最適
```

---

## よくある質問

### Q1: Algorithmの選び方は？

**A:** 音色による

```
シンプルな音:
Algorithm 1, 2
Carrier 1つ

複雑な音:
Algorithm 7, 8
複数Modulator

実験:

1. Algorithm 1から開始

2. 順に試す

3. 耳で判断

4. 気に入ったもの選択

推奨:

ベル: 1, 2, 3
ブラス: 7, 8
エレピ: 4, 6
```

### Q2: Modulatorのレベルは？

**A:** 控えめから

```
変調量:

少ない (20-30%):
シンプル
暗い

中程度 (50-60%):
適度な倍音
推奨

多い (80-100%):
非常に明るい
金属的
ノイズ的

調整方法:

1. 0% から開始

2. 徐々に上げる

3. 明るすぎる手前で停止

ベル: 60-80%
ブラス: 30-50%
```

### Q3: Wavetableと使い分けは？

**A:** 音色次第

```
Wavetable使用:

ベース
パッド
ストリングス
温かい音

Operator使用:

ベル
ブラス
エレピ
金属音
FX

迷ったら:

1. Wavetableから試す

2. 物足りない

3. Operator試す

4. どちらか選択

両方使う:

Wavetable: メイン
Operator: アクセント
```

---

## まとめ

### Operator基礎

```
□ FM合成 = 周波数変調
□ 4オシレーター
□ Algorithm = 接続方法
□ Carrier = 聴こえる音
□ Modulator = 倍音追加
```

### 音色別設定

```
ベル:
Algorithm 1, OSC B Level 70%

ブラス:
Algorithm 8, Attack 200ms

金属音:
非整数倍Coarse (2.5x, 3.7x等)
```

### 重要ポイント

```
□ Algorithm 1から始める
□ Modulatorは控えめ
□ Envelope各OSC個別
□ プリセット活用推奨
□ Wavetableと使い分け
```

---

**次は:** [Analog](./analog.md) - アナログシンセの温かみ
