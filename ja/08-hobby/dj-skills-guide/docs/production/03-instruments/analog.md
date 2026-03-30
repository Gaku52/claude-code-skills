# Analog

アナログシンセの温かみ。クラシックな減算式合成で、ビンテージサウンドとモダンなTechnoサウンドを両立します。

## この章で学ぶこと

- アナログ合成の原理
- Analogの構造
- 2つのオシレーター活用
- Moogスタイルフィルター
- ノイズジェネレーター
- Unisonとデチューン
- アナログパッドの作り方
- ビンテージリードの作り方


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解

---

## なぜAnalogが重要なのか

**温かみのある音色:**

```
音色の特徴:

Wavetable:
モダン
デジタル
クリア

Operator:
明るい
金属的

Analog:
温かい
アナログ的
ビンテージ

用途:

Analog推奨:

パッド:
温かみ
広がり

リード:
ビンテージ感
Moog的

ベース:
太い
アナログ感

使用率:

Wavetable: 70%
Operator: 15%
Analog: 10%
その他: 5%

理由:
特定の用途
温かみが欲しい時

歴史:

アナログシンセ:
1970-80年代
Moog, Prophet-5, Juno-60

Analogのモデル:
これらをエミュレート
```

---

## Ableton Live Analogの概要と設計思想

### Analogが生まれた背景

Ableton LiveのAnalogは、1960年代から1980年代にかけて黄金期を迎えたアナログシンセサイザーの動作原理をソフトウェアで忠実に再現したバーチャルアナログシンセサイザーです。Bob Moogが1964年に発表したモジュラーシンセサイザーに始まり、Minimoog、ARP 2600、Sequential Circuits Prophet-5、Roland Juno-60、Oberheim OB-Xaといった歴史的名機の設計思想とサウンドキャラクターを、デジタル信号処理（DSP）技術で再現しています。

```
歴史的アナログシンセの系譜とAnalogの位置づけ:

1964年 - Moog Modular:
  ┗ モジュラーシンセの始祖
  ┗ 電圧制御（CV/Gate）方式
  ┗ Moogラダーフィルターの発明

1970年 - Minimoog Model D:
  ┗ 世界初の市販ポータブルシンセ
  ┗ 3 VCO + ラダーフィルター
  ┗ Analogのフィルターモデルの原型

1971年 - ARP 2600:
  ┗ セミモジュラー設計
  ┗ 豊富なパッチング
  ┗ 教育用途にも広く使用

1978年 - Sequential Circuits Prophet-5:
  ┗ 世界初のプログラマブルポリシンセ
  ┗ 5ボイスポリフォニー
  ┗ パッチメモリー機能

1982年 - Roland Juno-60:
  ┗ DCO（デジタル制御オシレーター）
  ┗ 内蔵コーラスの伝説的サウンド
  ┗ Analogのコーラスモデルの参考

1983年 - Yamaha DX7:
  ┗ FM合成の登場 → アナログ衰退期へ
  ┗ しかしアナログの温かみは失われず

2000年代 - バーチャルアナログの隆盛:
  ┗ DSP技術の進歩
  ┗ アナログ回路のモデリング
  ┗ Ableton Analog登場（Live 8〜）

Analogの設計目標:
  ┗ ビンテージアナログの温かみを再現
  ┗ 直感的なインターフェース
  ┗ CPU効率の良いモデリング
  ┗ Live環境との完全統合
```

### ハードウェアアナログとの比較

```
Analog（ソフトウェア）の利点:

完全リコール:
  セッション保存で全パラメータ再現
  プロジェクト間でのプリセット共有
  バージョン管理が可能

ポリフォニー制限なし:
  ハードウェアは4〜8ボイスが一般的
  Analogは実質無制限（CPU依存）
  複雑なコードボイシングが可能

安定性:
  チューニングのドリフトなし
  温度変化の影響なし
  経年劣化なし

オートメーション:
  Liveの全パラメータをオートメーション可能
  MIDIマッピング自由自在
  Push/Launchpadとの統合

コスト:
  ビンテージMoog: 100万円〜
  Prophet-5 (Rev4): 約50万円
  Analog: Liveに無料付属

ハードウェアの利点（Analogにないもの）:

真のアナログ回路の不完全さ:
  微細なピッチドリフト
  コンポーネントの個体差
  温度による音色変化
  → これらが「温かみ」の一部

触覚フィードバック:
  物理ノブの操作感
  即座のパラメータアクセス

回路飽和の自然さ:
  アナログ回路固有の歪み特性
  Analogはモデリングで近似

結論:
  制作環境ではAnalog（ソフト）が圧倒的に便利
  ライブ演奏ではハードウェアの体験も価値がある
  両方を理解することで音作りの幅が広がる
```

### Analogのアーキテクチャ詳細

```
シグナルフロー完全図:

┌─────────────────────────────────────────────────┐
│                  VOICE STRUCTURE                  │
├─────────────────────────────────────────────────┤
│                                                   │
│  ┌──────────┐    ┌──────────┐                    │
│  │  OSC 1   │    │  OSC 2   │                    │
│  │ Saw/Sq/  │    │ Saw/Sq/  │                    │
│  │ Tri/Sine │    │ Tri/Sine │                    │
│  │ PW/Sub   │    │ PW/Sub   │                    │
│  └────┬─────┘    └────┬─────┘                    │
│       │               │                          │
│       │  ┌──────────┐ │                          │
│       │  │  NOISE   │ │                          │
│       │  │ Wht/Pink │ │                          │
│       │  └────┬─────┘ │                          │
│       │       │       │                          │
│       ▼       ▼       ▼                          │
│  ┌────────────────────────┐                      │
│  │       MIXER            │                      │
│  │  OSC1 + Noise + OSC2   │                      │
│  │  Level / Balance       │                      │
│  └───────────┬────────────┘                      │
│              │                                    │
│              ▼                                    │
│  ┌────────────────────────┐  ┌──────────────┐   │
│  │     FILTER 1           │  │  FILTER ENV  │   │
│  │  LP/BP/HP/Notch        │←─│  ADSR        │   │
│  │  12dB/24dB             │  │  Amount      │   │
│  │  Cutoff/Reso/Drive     │  └──────────────┘   │
│  └───────────┬────────────┘                      │
│              │                                    │
│              ▼                                    │
│  ┌────────────────────────┐  ┌──────────────┐   │
│  │     AMPLIFIER          │  │   AMP ENV    │   │
│  │  Level                 │←─│   ADSR       │   │
│  │  Pan                   │  └──────────────┘   │
│  └───────────┬────────────┘                      │
│              │                                    │
│              ▼                                    │
│  ┌────────────────────────┐                      │
│  │     LFO 1 / LFO 2     │──→ OSC/Filter/Amp   │
│  │  Rate/Shape/Amount     │                      │
│  └────────────────────────┘                      │
│                                                   │
├─────────────────────────────────────────────────┤
│                 GLOBAL SECTION                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐    │
│  │ UNISON   │ │ GLIDE    │ │   OUTPUT     │    │
│  │ Voices   │ │ Time     │ │   Volume     │    │
│  │ Detune   │ │ Mode     │ │   Pan        │    │
│  └──────────┘ └──────────┘ └──────────────┘    │
└─────────────────────────────────────────────────┘
```

---

## 減算式合成とは

**Subtractive Synthesis:**

### 基本原理

```
シンプルな流れ:

Oscillator:
豊富な倍音を生成
Saw, Square波

↓

Filter:
不要な周波数をカット
削って音色作る

↓

Amplifier:
音量調整

= 減算式

Wavetableとの違い:

Wavetable:
複雑な波形
デジタル処理

Analog:
シンプルな波形
フィルターで削る
アナログ的

メリット:

シンプル:
理解しやすい

温かい:
アナログ感

クラシック:
ビンテージサウンド
```

### 減算合成の物理学的背景

アナログシンセサイザーの音作りを深く理解するには、音の物理的性質を把握することが重要です。全ての楽器音は基本周波数（ファンダメンタル）とその整数倍の周波数成分（倍音/ハーモニクス）の組み合わせで構成されています。

```
倍音構造の理解:

基本波形と倍音の関係:

Sine（正弦波）:
  倍音なし（基本波のみ）
  最もシンプルな音
  f = 基本周波数のみ
  用途: サブベース、テスト信号

Saw（ノコギリ波）:
  全ての整数倍の倍音を含む
  倍音の振幅 = 1/n（nは倍音の次数）
  第1倍音: 1.0（基本波）
  第2倍音: 0.5
  第3倍音: 0.33
  第4倍音: 0.25
  第5倍音: 0.2
  ...
  → 非常にリッチで明るい音
  → 減算合成に最適（削りがいがある）

Square（矩形波）:
  奇数次の倍音のみ含む
  倍音の振幅 = 1/n（奇数次のみ）
  第1倍音: 1.0
  第3倍音: 0.33
  第5倍音: 0.2
  第7倍音: 0.14
  → 偶数倍音がないため「ホロー」な音色
  → クラリネット、管楽器的

Triangle（三角波）:
  奇数次の倍音のみ（Squareと同じ）
  倍音の振幅 = 1/n^2（急速に減衰）
  第1倍音: 1.0
  第3倍音: 0.11
  第5倍音: 0.04
  → 非常にソフトで暗い音
  → フルート的、サブオシレーター向き

フーリエ合成の原理:
  あらゆる周期的な波形は
  正弦波の足し合わせで表現可能
  → 減算合成はその逆のアプローチ
  → リッチな倍音から不要なものを「引く」
```

```
フィルタースロープの物理的意味:

フィルターの急峻さ（dB/oct）:

6dB/oct（1-pole）:
  ┃
  ┃━━━━━┓
  ┃      ┗━━━━━━━━━
  ┃         緩やかな傾斜
  ┗━━━━━━━━━━━━━━━━━
  → 非常にソフトなフィルタリング
  → 自然な音色変化

12dB/oct（2-pole）:
  ┃
  ┃━━━━━┓
  ┃      ┗━━━━━━
  ┃            ┗━━━━
  ┗━━━━━━━━━━━━━━━━━
  → 中程度の傾斜
  → Oberheim系の音色

24dB/oct（4-pole）:
  ┃
  ┃━━━━━┓
  ┃      ┃━━━
  ┃         ┗━━━
  ┗━━━━━━━━━━━━━━━━━
  → 急峻なカット
  → Moogラダーフィルターの代名詞
  → Analogのデフォルト

カットオフ周波数での減衰量:
  6dB/oct:  カットオフの1オクターブ上で -6dB
  12dB/oct: カットオフの1オクターブ上で -12dB
  24dB/oct: カットオフの1オクターブ上で -24dB

実用上の違い:
  6dB:  ハイカットとして自然な効果
  12dB: ワウペダル的な動き
  24dB: ドラマチックなフィルタースイープ
        ベースラインの太さ
        Analogが得意とする領域
```

### 他の合成方式との詳細比較

```
シンセシス方式の比較表:

┌─────────────┬───────────────┬──────────────┬──────────────┐
│ 方式        │ 原理          │ 得意な音色   │ 代表機種     │
├─────────────┼───────────────┼──────────────┼──────────────┤
│ 減算合成    │ 倍音をフィルタ │ パッド       │ Moog         │
│ Subtractive │ で削る        │ ベース       │ Prophet-5    │
│             │               │ リード       │ Juno-60      │
│             │               │              │ → Analog     │
├─────────────┼───────────────┼──────────────┼──────────────┤
│ FM合成      │ 周波数変調    │ ベル音       │ Yamaha DX7   │
│ Frequency   │ キャリア+     │ エレピ       │ → Operator   │
│ Modulation  │ モジュレータ  │ 金属音       │              │
├─────────────┼───────────────┼──────────────┼──────────────┤
│ Wavetable   │ 波形テーブルの │ モダンベース │ PPG Wave     │
│             │ モーフィング  │ テクスチャ   │ → Wavetable  │
├─────────────┼───────────────┼──────────────┼──────────────┤
│ 加算合成    │ 正弦波の積み上│ オルガン     │ Hammond      │
│ Additive    │ げ            │ ベル         │ Kawai K5     │
├─────────────┼───────────────┼──────────────┼──────────────┤
│ グラニュラー│ 微小サンプルの│ テクスチャ   │ → Granulator │
│ Granular    │ 再構成        │ アンビエント │              │
├─────────────┼───────────────┼──────────────┼──────────────┤
│ 物理モデリン│ 楽器の物理    │ 弦楽器      │ Yamaha VL1   │
│ グ Physical │ シミュレーション│ 管楽器      │ → Tension    │
└─────────────┴───────────────┴──────────────┴──────────────┘

減算合成が選ばれる理由:
  1. 直感的 — ノブを回せば音が変わる
  2. 予測可能 — フィルターの動きが理解しやすい
  3. 音楽的 — 自然に心地よい音色が得られる
  4. 歴史的 — 50年以上の蓄積されたノウハウ
```

---

## Analogの構造

**2オシレーターシステム:**

### 全体像

```
┌────────────────────────────┐
│ Oscillator 1               │
│ Saw, Square, Triangle      │
└──────────┬─────────────────┘
           │
┌──────────▼─────────────────┐
│ Oscillator 2               │
│ Saw, Square, Triangle      │
└──────────┬─────────────────┘
           │
┌──────────▼─────────────────┐
│ Noise Generator            │
│ White, Pink                │
└──────────┬─────────────────┘
           │ Mix
           ▼
┌────────────────────────────┐
│ Filter (Moogスタイル)       │
│ 24dB Lowpass               │
└──────────┬─────────────────┘
           │
┌──────────▼─────────────────┐
│ Amplifier                  │
└──────────┬─────────────────┘
           │
┌──────────▼─────────────────┐
│ Effects                    │
│ Chorus等                   │
└────────────────────────────┘

制御:

Filter Envelope:
時間変化

Amp Envelope:
音量変化

LFO:
周期的変化
```

### インターフェイス

```
左側:
┌────────────────────────────┐
│ OSC 1, OSC 2               │
│ 波形選択、Tune             │
└────────────────────────────┘

中央:
┌────────────────────────────┐
│ FILTER                     │
│ Cutoff, Resonance, Type    │
└────────────────────────────┘

右側:
┌────────────────────────────┐
│ AMP, GLOBAL                │
│ Volume, Unison             │
└────────────────────────────┘

下部:
┌────────────────────────────┐
│ Envelopes, LFO             │
│ Filter Env, Amp Env        │
└────────────────────────────┘
```

---

## Oscillator（オシレーター）

**2つのオシレーター:**

### Wave（波形）

```
3種類の基本波形:

Saw (ノコギリ波):
明るい
豊富な倍音
ベース、リード向き

Square (矩形波):
中間的
ホロー
クラリネット的

Triangle (三角波):
暗い
シンプル
フルート的

パルス幅 (Square波):

Width調整:
細い = 明るい
太い = 暗い

PWM (Pulse Width Modulation):
LFOで自動変調
動きのある音

推奨:

ベース:
Saw + Square

パッド:
Saw + Triangle

リード:
Saw + Saw (Detune)
```

### Tune

```
Semi (半音):
-24 〜 +24 st

Detune:
-50 〜 +50 cent

OSC 2設定:

パッド:
Semi: +12 (1オクターブ上)
Detune: +7 cent

リード:
Semi: 0
Detune: +10 cent
= 厚み

ベース:
Semi: -12 (1オクターブ下)
Sub Bass層
```

### Level

```
OSC 1:
メイン
100%

OSC 2:
サブ
50-80%

バランス:

両方同じ:
厚い

OSC1優先:
クリア

推奨:

OSC 1: 100%
OSC 2: 70%
```

### オシレーター詳細パラメータ解説

各パラメータの音楽的な効果と、具体的な設定指針を深掘りします。

```
オシレーター波形の詳細な音色特性:

Saw（ノコギリ波）の活用法:
  ┃
  ┃ /│ /│ /│
  ┃/  │/  │/  │
  ┃
  特性:
  - 全倍音を含む最もリッチな波形
  - ブラス、ストリングス的な響き
  - フィルターの効果が最も分かりやすい
  - Analogで最も使用頻度が高い

  ジャンル別活用:
    Techno: メインオシレーターとして定番
    House: ベースラインの基本波形
    Trance: リードシンセの必須波形
    Ambient: フィルターで削ってパッドに

Square（矩形波）の活用法:
  ┃
  ┃ ┌─┐ ┌─┐ ┌─
  ┃ │ │ │ │ │
  ┃─┘ └─┘ └─┘
  ┃
  特性:
  - 奇数倍音のみで「ホロー」な音色
  - パルス幅で音色が劇的に変化
  - 8bitゲーム音のベースにもなる

  パルス幅（Pulse Width）の効果:
    50%（真のSquare）:
      最もホローな音色
      クラリネット的
    25%:
      やや鼻にかかった音
      オーボエ的
    10%:
      非常に細く鋭い音
      ストリング的な倍音
    PWM（自動変調）:
      LFOで幅を変調
      常に変化する豊かな音色
      ストリングアンサンブル的

Triangle（三角波）の活用法:
  ┃
  ┃ /\   /\
  ┃/  \ /  \
  ┃    V    V
  ┃
  特性:
  - 少ない倍音でソフトな音色
  - サブオシレーターとして最適
  - フルート、ビブラフォン的

  活用シーン:
    サブベース補強:
      OSC2をTriangleに設定
      Semi: -12（1オクターブ下）
      低域の安定した土台を提供
    パッドの柔らかさ:
      OSC1: Saw, OSC2: Triangle
      Triangleの割合を上げるほどソフトに
    ベルサウンド:
      OSC1: Triangle, OSC2: Triangle
      異なるチューニングで倍音を生成
```

```
2オシレーターの組み合わせマトリクス:

OSC1 Saw + OSC2 Saw:
  Detune: +5〜+15 cent
  → 定番の厚い音色
  → コーラス効果
  → パッド、リードに最適

OSC1 Saw + OSC2 Square:
  Semi: 0, Detune: +7
  → 太いベースサウンド
  → Sawの明るさ + Squareの芯
  → Technoベースの定番

OSC1 Saw + OSC2 Triangle:
  Semi: -12（1oct下）
  → サブベース付きリード
  → 低域の安定感
  → Deep Houseに最適

OSC1 Square + OSC2 Square:
  Width違い（50% + 30%）
  → ビンテージシンセサウンド
  → PWMの掛け合い
  → 80sシンセポップ風

OSC1 Square + OSC2 Triangle:
  Semi: +7（5度上）
  → パワーコード的な響き
  → チップチューン風味
  → ゲームミュージック

OSC1 Triangle + OSC2 Triangle:
  Semi: 0, Detune微小
  → 極めてソフトな音色
  → アンビエントベル
  → ヒーリングサウンド

推奨組み合わせランキング:
  1位: Saw + Saw（Detune） — 汎用性最高
  2位: Saw + Square — ベースに最強
  3位: Saw + Triangle — 深みのあるパッド
  4位: Square + Square — ビンテージ感
```

---

## Filter（フィルター）

**Moogスタイル:**

### Filter Type

```
24dB Lowpass:
Moog風
温かい
太い

12dB Lowpass:
軽い
明るい

Bandpass:
中域のみ
電話的

Highpass:
低域カット

推奨:

ほぼ全て:
24dB Lowpass

理由:
Analogの特徴
温かみ
```

### Cutoff & Resonance

```
Cutoff:

パッド:
800-1500 Hz
柔らかい

リード:
2000-4000 Hz
明るい

ベース:
500-1000 Hz
太い

Resonance:

0-30%:
自然

30-60%:
強調
Moog的

60-100%:
自己発振
特殊効果

推奨値:

パッド: 20%
リード: 40%
ベース: 25%
```

### Drive

```
機能:
アナログ歪み
温かみ

範囲:
0-100%

効果:
倍音付加
サチュレーション

推奨:

パッド: 10-20%
リード: 30-40%
ベース: 20-30%

注意:
上げすぎると歪む
```

### フィルターセクション詳細解説

フィルターはAnalogの音色を決定づける最も重要なセクションです。Moogラダーフィルターのモデリングを中心に、各フィルタータイプの特性と活用法を詳しく解説します。

```
フィルタータイプ別の詳細特性:

24dB Low Pass（4-pole Ladder Filter）:
  モデル元: Moog Minimoog ラダーフィルター
  特性:
    - 急峻なカットオフスロープ
    - レゾナンスを上げるとローエンドが減少（Moog特有）
    - 自己発振時に正弦波に近い音を生成
    - 太い低域とスムーズなカットオフが特徴
  最適用途:
    - ファットなベースライン
    - フィルタースイープ（エンベロープ/LFOで動かす）
    - ウォームパッド
    - クラシックなアナログリード
  セッティング例（Moog Bass）:
    Cutoff: 400-800 Hz
    Resonance: 30-45%
    Drive: 20-35%
    Filter Env Amount: +60%

12dB Low Pass（2-pole）:
  モデル元: Oberheim SEM フィルター
  特性:
    - 穏やかなカットオフスロープ
    - レゾナンスを上げても低域が保たれる
    - より明るく開放的なサウンド
    - Sawの高域が残りやすい
  最適用途:
    - ブライトなパッド
    - 透明感のあるリード
    - フィルターの効きを緩やかにしたい時
  セッティング例（Bright Pad）:
    Cutoff: 1500-2500 Hz
    Resonance: 15-25%
    Drive: 10%
    Filter Env Amount: +25%

Band Pass:
  特性:
    - カットオフ周辺の帯域のみ通過
    - 低域と高域の両方がカットされる
    - 電話・ラジオ的な音色
    - レゾナンスで帯域幅が狭くなる
  最適用途:
    - ボーカル的なシンセサウンド
    - ワウペダル効果
    - LoFiテクスチャ
    - フィルタースイープの特殊効果
  セッティング例（Vocal Synth）:
    Cutoff: 800-2000 Hz（スイープ）
    Resonance: 50-70%
    Drive: 15%
    LFOでCutoffをモジュレーション

High Pass:
  特性:
    - カットオフ以下の低域をカット
    - 音を薄く、明るくする
    - ミックスの中で空間を作る
  最適用途:
    - ベースとの被りを避ける
    - ライザーエフェクト
    - テクスチャ素材
    - パッドの低域処理
  セッティング例（Riser FX）:
    Cutoff: 100Hz → 8000Hz（オートメーション）
    Resonance: 40%
    Drive: 25%
    上昇するフィルタースイープ

Notch（ノッチ/バンドリジェクト）:
  特性:
    - 特定の帯域のみカット
    - フェイザー的な効果
    - LFOで動かすとスウィープ感
  セッティング例:
    Cutoff: 1000-3000 Hz（LFOでスイープ）
    Resonance: 30%
    Rate: 0.5-2 Hz
```

```
レゾナンスの深い理解:

レゾナンスとは何か:
  カットオフ周波数付近の帯域を強調するフィードバック機構
  フィルター出力の一部を入力にフィードバック

レゾナンス値による音色変化:

  0%:  フラットなフィルタリング
       自然な音色
       └── パッド、サブベースに

  20%: わずかな強調
       音に存在感が出る
       └── ウォームベース、クラシックパッド

  40%: はっきりとしたピーク
       フィルタースイープが際立つ
       └── Acid Bass、Moogリード

  60%: 強い共鳴
       自己発振の手前
       └── スクリーミングリード、FXサウンド

  80%: 自己発振寸前
       ピッチ感のあるフィルター
       └── 特殊効果、トランジション

  100%: 完全な自己発振
        フィルターが正弦波を生成
        └── パーカッション、キック合成

自己発振の活用:
  フィルターのレゾナンスを最大にすると
  フィルター自体がオシレーターとして機能
  → キックドラムのアタック部分の合成
  → 特殊なパーカッション音
  → 実験的サウンドデザイン
```

```
Filter Keytracking（キートラッキング）:

概要:
  演奏するノートの高さに応じて
  フィルターのカットオフ周波数を追従させる機能

設定値の意味:
  0%:   キートラッキングOFF
        全てのノートで同じカットオフ
        → ベースに適切（低い音が暗く、高い音が明るくなるのを防ぐ）

  50%:  半分追従
        → パッドに適切（自然な変化）

  100%: 完全追従
        ノートと同じだけカットオフが移動
        → リードに適切（全音域で一貫した音色）

  >100%: 過剰追従
         高い音ほどより明るくなる
         → 特殊効果

推奨設定:
  ベース: 0-30%
  パッド: 40-60%
  リード: 80-100%
  FX:    任意
```

---

## アンプリファイアとエンベロープの詳細

### アンプエンベロープの音楽的意味

エンベロープ（ADSR）は音の時間的変化を制御するパラメータです。楽器の発音特性を模倣し、表現力豊かなサウンドを作るために不可欠です。

```
ADSR各パラメータの詳細:

Attack（アタック）:
  キーを押してから最大音量に達するまでの時間

  0-5 ms:    瞬間的な立ち上がり
             ピアノ、パーカッション、プラック
             → ベース、リードの定番

  5-50 ms:   わずかな「ふわっ」とした立ち上がり
             → ブラス、ギター的ニュアンス

  50-500 ms: 緩やかな立ち上がり
             → ストリングス、ボウイング的

  500ms-3s:  ゆっくりとしたフェードイン
             → パッド、アンビエント

  3s以上:    非常にゆっくり
             → ドローン、テクスチャ

Decay（ディケイ）:
  最大音量からSustainレベルに降下する時間

  0-50 ms:   パーカッシブ
             → プラック、キーボード

  50-300 ms: ピアノ的な減衰
             → ベルサウンド

  300ms-1s:  中程度の減衰
             → リード、ブラス

  1s以上:    緩やかな減衰
             → パッド系

Sustain（サステイン）:
  キーを押し続けている間の音量レベル（%）

  0%:        キーを押し続けても音は消える
             → パーカッシブサウンド、プラック

  30-50%:    音量が下がって持続
             → クラビネット、ブラス的

  60-80%:    ほぼ一定レベルで持続
             → パッド、ストリングス

  100%:      減衰なしで持続
             → オルガン、ドローン

Release（リリース）:
  キーを離してから音が消えるまでの時間

  0-50 ms:   即座に消える
             → タイトなベース、パーカッション

  50-300 ms: 自然な消え方
             → ほとんどの楽器

  300ms-2s:  余韻が残る
             → パッド、リバーブ的効果

  2s以上:    長い余韻
             → アンビエント、ドローン
```

```
楽器別エンベロープ設定の一覧:

ピアノ的:
  A: 0ms  D: 800ms  S: 20%  R: 500ms
  → 瞬間的な立ち上がりと自然な減衰

オルガン的:
  A: 5ms  D: 0ms  S: 100%  R: 50ms
  → キーに忠実な発音と消音

ストリングス的:
  A: 800ms  D: 500ms  S: 70%  R: 1500ms
  → ボウイングの表現

ブラス的:
  A: 50ms  D: 200ms  S: 80%  R: 100ms
  → わずかな立ち上がりとソリッドな持続

プラック的:
  A: 0ms  D: 200ms  S: 0%  R: 150ms
  → 短いDecayでパーカッシブ

キック的:
  A: 0ms  D: 150ms  S: 0%  R: 50ms
  → 極端に短いDecay

ビンテージパッド的:
  A: 2000ms  D: 1500ms  S: 60%  R: 3000ms
  → ゆっくりとした全てのパラメータ
```

---

## Noise Generator

**ノイズ追加:**

### Noise Type

```
White Noise:
全周波数均等
「シャー」

Pink Noise:
低域多い
自然

用途:

パッド:
Pink Noise 5-10%
温かみ

リード:
White Noise 2-5%
存在感

Hi-Hat:
White Noise 100%
ノイズ楽器

設定:

Level:
0-15%
控えめ

Color:
White / Pink

Filter:
ノイズもフィルター通る
```

---

## Unison & Detune

**厚みの秘密:**

### Unison

```
機能:
音を複製
わずかにデチューン

Voices:
1 (Off)
2-8 (複製数)

Detune:
0-100%
ズレ具合

推奨設定:

パッド:
Voices: 4-6
Detune: 30-50%
非常に厚い

リード:
Voices: 2-3
Detune: 20-30%
適度に

ベース:
Voices: 2
Detune: 10-15%
わずかに

注意:

CPU負荷:
Voices多い = 重い

モノラル:
Unisonで広がり
```

---

## Envelope

**時間変化:**

### Filter Envelope

```
Amount:
-100% 〜 +100%

パッド設定:

Attack: 1000 ms
ゆっくり開く

Decay: 1500 ms

Sustain: 60%

Release: 2000 ms
長い余韻

Amount: +40%
適度に

リード設定:

Attack: 50 ms
速い

Decay: 400 ms

Sustain: 40%

Release: 300 ms

Amount: +60%
しっかり
```

### Amp Envelope

```
パッド:

Attack: 1500 ms
ふわっと

Decay: 1000 ms

Sustain: 80%

Release: 2500 ms
長い

リード:

Attack: 5 ms
即座

Decay: 300 ms

Sustain: 70%

Release: 200 ms
```

---

## 実践: Analog Padの作り方

**12分で完成:**

### Step 1: 初期化 (1分)

```
1. 新規MIDIトラック

2. Analog挿入:
   Browser > Instruments > Analog

3. Init

4. 確認
```

### Step 2: Oscillator (4分)

```
OSC 1:

Wave: Saw
Semi: 0
Detune: 0
Level: 100%

OSC 2:

Wave: Saw
Semi: +12 (1オクターブ上)
Detune: +7 cent
Level: 80%

OSC 1+2 Mix:

Sub:
Level: 10%
低域補強

Noise:
Type: Pink
Level: 8%
温かみ
```

### Step 3: Filter (3分)

```
Type:
24dB Lowpass

Cutoff:
1200 Hz
柔らかい

Resonance:
22%

Drive:
18%
わずかに温かく
```

### Step 4: Unison (2分)

```
Voices:
6
非常に厚い

Detune:
45%
広がり

Amount:
100%
```

### Step 5: Envelope (2分)

```
Filter Envelope:

Attack: 1200 ms
Decay: 1800 ms
Sustain: 65%
Release: 2500 ms
Amount: +35%

Amp Envelope:

Attack: 1500 ms
Decay: 1200 ms
Sustain: 85%
Release: 3000 ms

効果:
ゆっくり包み込む
```

### Step 6: 仕上げ

```
Global:

Volume: -6 dB

確認:
Cメジャーコード (C4, E4, G4)
温かく広がる

プリセット保存:
"My Warm Pad"

完成:
美しいアナログパッド
```

---

## 実践: Vintage Leadの作り方

**10分で完成:**

### Step 1: Oscillator (4分)

```
OSC 1:

Wave: Saw
Semi: 0
Level: 100%

OSC 2:

Wave: Saw
Semi: 0
Detune: +12 cent
Level: 90%

効果:
厚み、デチューン

Sub:
Level: 15%

Noise:
Level: 3%
わずかに
```

### Step 2: Filter (3分)

```
Type:
24dB Lowpass

Cutoff:
3000 Hz
明るめ

Resonance:
38%
Moog的

Drive:
32%
歪み
```

### Step 3: Envelope (2分)

```
Filter Envelope:

Attack: 60 ms
Decay: 500 ms
Sustain: 45%
Release: 350 ms
Amount: +55%

Amp Envelope:

Attack: 8 ms
Decay: 350 ms
Sustain: 75%
Release: 250 ms
```

### Step 4: Unison (1分)

```
Voices: 3

Detune: 25%

効果:
適度な厚み
```

### Step 5: 仕上げ

```
確認:
C4 - C5 メロディ
ビンテージ感

保存:
"My Vintage Lead"
```

---

## LFO（Low Frequency Oscillator）詳細設定

### LFOの基本概念

LFO（低周波オシレーター）は人間の可聴域以下の周波数で動作するオシレーターです。直接音として聞こえるのではなく、他のパラメータを周期的に変調（モジュレーション）するために使用します。ビブラート、トレモロ、ワウワウ、パンニングなど、音に動きと生命感を与える不可欠な要素です。

```
LFOの基本パラメータ:

Rate（レート/スピード）:
  変調の速さを設定

  0.1 Hz:  10秒で1周期
           → ゆっくりとしたうねり
           → パッドの微妙な変化
           → アンビエントテクスチャ

  0.5 Hz:  2秒で1周期
           → 明確なうねり
           → パッドのフィルタースウィープ

  1-3 Hz:  標準的なビブラート速度
           → ビブラート（ピッチ変調）
           → トレモロ（音量変調）
           → ワウワウ（フィルター変調）

  4-8 Hz:  速いビブラート
           → 激しいビブラート効果
           → ロトスピーカー的回転感

  10-20 Hz: 可聴域に近づく
            → バズ、グリッチ的効果
            → サイドバンド生成（FM的）

  Sync（テンポ同期）:
    1/1:   1小節で1周期
    1/2:   半小節で1周期
    1/4:   1拍で1周期（最もよく使う）
    1/8:   8分音符で1周期
    1/16:  16分音符で1周期
    → テンポに追従して変化
    → トランス系ゲート効果に必須

Shape（波形）:
  Sine:     滑らかな変調
            → ビブラート、トレモロの定番
            → 自然で心地よい

  Triangle: Sineに近いが頂点が鋭い
            → やや直線的な変化
            → フィルタースイープ

  Square:   ON/OFFの切り替え
            → トレモロ（振幅変調）
            → ゲート効果
            → トランスゲート

  Saw Up:   ゆっくり上昇→急降下
            → フィルターの「開いて→閉じる」
            → シーケンサー的効果

  Saw Down: 急上昇→ゆっくり降下
            → フィルターの「パッと開いて→閉じる」
            → パーカッシブな動き

  Random:   ランダムな値
            → サンプル&ホールド（S&H）
            → 電子的・ロボット的な音色変化
            → R2-D2的サウンド
```

```
LFOのモジュレーション先と音楽的効果:

LFO → Pitch（ピッチ）:
  少量（Amount: 5-15%）:
    → ビブラート効果
    → 歌声のような揺らぎ
    → Rate: 4-6 Hz が自然

  大量（Amount: 50-100%）:
    → サイレン効果
    → 警告音的サウンド
    → DJ向けトランジションFX

LFO → Filter Cutoff:
  少量（Amount: 10-25%）:
    → 穏やかなフィルターうねり
    → パッドに生命感を付加
    → アンビエントの定番テクニック

  中量（Amount: 30-60%）:
    → ワウワウ効果
    → Acid的なフィルタースイープ
    → ダブステップ的ウォブル

  大量（Amount: 70-100%）:
    → ドラマチックなスイープ
    → オートワウ全開
    → テンポ同期でリズミカルに

LFO → Amplitude（音量）:
  少量:
    → トレモロ効果
    → 微妙なボリュームの揺れ
    → ギターアンプ的

  大量 + Square波:
    → ゲート効果（トランスゲート）
    → ゲーティッドパッド
    → Side-chain的な効果をLFOで実現

LFO → Pulse Width:
  任意のAmount:
    → PWM（パルス幅変調）
    → ストリングアンサンブル的な豊かさ
    → Juno-60の代名詞サウンド
    → Rate: 0.5-2 Hz がクラシック

LFO → Pan（パン）:
  中量:
    → オートパン効果
    → ステレオ空間での音の移動
    → サイケデリックな広がり
    → Rate: 1/4〜1/8が音楽的
```

```
LFO設定レシピ集:

【ビブラートリード】
  LFO Shape: Sine
  LFO Rate: 5.5 Hz
  Destination: Pitch
  Amount: 8%
  → クラシックなリードビブラート

【トランスゲートパッド】
  LFO Shape: Square
  LFO Rate: 1/8（テンポ同期）
  Destination: Amplitude
  Amount: 100%
  → リズミカルなゲート効果

【アシッドワウベース】
  LFO Shape: Saw Down
  LFO Rate: 1/16（テンポ同期）
  Destination: Filter Cutoff
  Amount: 50%
  → 303的なアシッドサウンド

【ドリフティングパッド】
  LFO Shape: Sine
  LFO Rate: 0.15 Hz
  Destination: Filter Cutoff + Pitch
  Amount: Filter 20%, Pitch 3%
  → ゆっくりとした有機的な変化

【ロボットボイス（S&H）】
  LFO Shape: Random
  LFO Rate: 8 Hz
  Destination: Filter Cutoff
  Amount: 60%
  → 電子的なランダム変化

【ストリングアンサンブル】
  LFO Shape: Triangle
  LFO Rate: 0.8 Hz
  Destination: Pulse Width
  Amount: 40%
  → Juno-60的なPWMストリングス
```

---

## モジュレーションマトリクス活用法

### モジュレーションの基本概念

モジュレーション（変調）とは、あるパラメータ（ソース）の値を使って別のパラメータ（デスティネーション）を自動的に変化させる仕組みです。Analogでは主にエンベロープとLFOがモジュレーションソースとして機能します。

```
Analogのモジュレーションソースとデスティネーション:

モジュレーションソース（変調元）:
  ┌─────────────────────────────────┐
  │ Filter Envelope (ADSR)          │
  │ Amp Envelope (ADSR)             │
  │ LFO 1                          │
  │ LFO 2                          │
  │ Velocity（鍵盤の打鍵の強さ）    │
  │ Key（鍵盤の位置/ノート番号）    │
  │ Aftertouch（鍵盤の押し込み）    │
  │ Mod Wheel（モジュレーションホイール）│
  └─────────────────────────────────┘

モジュレーションデスティネーション（変調先）:
  ┌─────────────────────────────────┐
  │ OSC Pitch（ピッチ）              │
  │ OSC Pulse Width（パルス幅）      │
  │ OSC Level（オシレーターレベル）   │
  │ Filter Cutoff（カットオフ周波数） │
  │ Filter Resonance（レゾナンス）    │
  │ Amplifier Level（音量）           │
  │ Amplifier Pan（パン）             │
  │ LFO Rate（LFO速度）              │
  └─────────────────────────────────┘
```

```
ベロシティの活用:

Velocity → Filter Cutoff:
  効果: 強く弾くほどフィルターが開く
  Amount: +30-50%
  → 表現力のあるベースライン
  → アクセントをつけたリード

Velocity → Amp Level:
  効果: 強く弾くほど音量が大きい
  Amount: +50-80%
  → ダイナミクスのある演奏
  → パッドのベロシティレイヤー

Velocity → Filter Envelope Amount:
  効果: 強く弾くほどフィルターエンベロープが深くかかる
  Amount: +40-60%
  → ファンキーなクラビネットサウンド
  → ベロシティ感応型ベース

Velocity → Attack Time:
  効果: 強く弾くほどアタックが速い
  Amount: 負の値（-30%）
  → ソフトに弾くとパッド的
  → 強く弾くとパーカッシブ

推奨ベロシティ設定:
  ベース: Filter Cutoff +40%, Amp +60%
  パッド: Amp +30%（控えめ）
  リード: Filter Cutoff +50%, Amp +70%
  プラック: Filter Cutoff +60%, Amp +80%
```

```
Aftertouch（アフタータッチ）の活用:

鍵盤を押した後にさらに押し込むことで変調をかける:

Aftertouch → Filter Cutoff:
  押し込むほどフィルターが開く
  → 表現力豊かなソロ演奏
  → Amount: +30-50%

Aftertouch → LFO Amount:
  押し込むほどビブラートが深くなる
  → クラシックなシンセリード表現
  → Amount: +40-60%

Aftertouch → Volume:
  押し込むほど音量が上がる
  → クレッシェンド効果
  → Amount: +20-30%

Mod Wheel（モジュレーションホイール）の活用:

Mod Wheel → LFO Amount:
  ホイールでビブラートの深さを制御
  → ライブ演奏の定番
  → リアルタイムで表現力をコントロール

Mod Wheel → Filter Cutoff:
  ホイールでフィルターを開閉
  → DJセット中のフィルタースイープ
  → パフォーマンス向け

Mod Wheel → Noise Level:
  ホイールでノイズ量を制御
  → ブレス的な表現
  → フルート、管楽器的ニュアンス
```

---

## ジャンル別サウンドデザイン完全ガイド

### Technoサウンドデザイン

Technoではシンプルかつパワフルなアナログサウンドが求められます。Analogはその用途に最適です。

```
Techno向けAnalog設定:

【ダークTechnoベース】
  OSC1: Saw, Semi: 0, Level: 100%
  OSC2: Square, Semi: 0, Detune: +5, Level: 70%
  Noise: Off
  Filter: 24dB LP, Cutoff: 600Hz, Reso: 35%, Drive: 30%
  Filter Env: A:0 D:400ms S:20% R:200ms, Amount: +50%
  Amp Env: A:0 D:300ms S:60% R:150ms
  Unison: Off（モノフォニック推奨）
  Glide: On, Time: 50ms
  → 暗く太いベースライン
  → 16分音符パターンで効果的

【Technoスタブ】
  OSC1: Saw, Level: 100%
  OSC2: Saw, Detune: +8, Level: 85%
  Filter: 24dB LP, Cutoff: 2000Hz, Reso: 25%, Drive: 20%
  Filter Env: A:0 D:150ms S:0% R:100ms, Amount: +45%
  Amp Env: A:0 D:200ms S:0% R:80ms
  Unison: 2 Voices, Detune: 15%
  → パンチのあるスタブサウンド
  → オフビートで使用

【ミニマルTechnoパッド】
  OSC1: Triangle, Level: 100%
  OSC2: Saw, Semi: +12, Level: 40%
  Noise: Pink, 5%
  Filter: 24dB LP, Cutoff: 1000Hz, Reso: 15%
  LFO → Cutoff: Sine, Rate: 0.2Hz, Amount: 15%
  Amp Env: A:3000ms D:2000ms S:50% R:4000ms
  Unison: 4 Voices, Detune: 35%
  → 静かに漂うバックグラウンドパッド
  → ミニマルの空間演出

【インダストリアルリード】
  OSC1: Square, Width: 30%, Level: 100%
  OSC2: Saw, Detune: +15, Level: 90%
  Noise: White, 8%
  Filter: 24dB LP, Cutoff: 2500Hz, Reso: 50%, Drive: 45%
  Filter Env: A:5ms D:600ms S:30% R:400ms, Amount: +65%
  Amp Env: A:2ms D:500ms S:65% R:300ms
  Unison: 3 Voices, Detune: 25%
  → 攻撃的で存在感のあるリード
  → ブレイク部分のメロディに
```

### House / Deep Houseサウンドデザイン

```
House向けAnalog設定:

【Classic House Chord（Juno的）】
  OSC1: Saw, Level: 100%
  OSC2: Square, Width: 45%, Semi: 0, Level: 65%
  LFO → Pulse Width: Sine, Rate: 0.7Hz, Amount: 30%
  Filter: 12dB LP, Cutoff: 2200Hz, Reso: 20%
  Filter Env: A:20ms D:800ms S:50% R:600ms, Amount: +30%
  Amp Env: A:10ms D:600ms S:70% R:500ms
  Unison: 4 Voices, Detune: 25%
  → Juno-60的な暖かいコードサウンド
  → 白玉やスタッカートで使用

【Deep Houseベースライン】
  OSC1: Saw, Semi: 0, Level: 100%
  OSC2: Triangle, Semi: -12, Level: 60%
  Filter: 24dB LP, Cutoff: 800Hz, Reso: 20%, Drive: 15%
  Filter Env: A:0 D:300ms S:30% R:200ms, Amount: +35%
  Amp Env: A:2ms D:200ms S:50% R:150ms
  Unison: Off
  Glide: On, Time: 80ms
  → 温かく深いベースライン
  → グルーヴィなパターンに最適

【ローファイハウスパッド】
  OSC1: Saw, Level: 80%
  OSC2: Saw, Semi: +7, Detune: +12, Level: 60%
  Noise: Pink, 12%
  Filter: 24dB LP, Cutoff: 900Hz, Reso: 30%, Drive: 25%
  LFO → Cutoff: Triangle, Rate: 0.3Hz, Amount: 10%
  Amp Env: A:2000ms D:1500ms S:65% R:3000ms
  Unison: 6 Voices, Detune: 40%
  → ヴィンテージ感のあるローファイパッド
  → フィルターハウスの背景に

【ディスコ風ベース（ファンキー）】
  OSC1: Square, Width: 40%, Level: 100%
  OSC2: Saw, Semi: -12, Level: 50%
  Filter: 24dB LP, Cutoff: 1200Hz, Reso: 35%, Drive: 20%
  Filter Env: A:0 D:250ms S:25% R:150ms, Amount: +55%
  Amp Env: A:0 D:150ms S:40% R:100ms
  Velocity → Filter Cutoff: +50%
  → ファンキーなベースライン
  → 16分のシンコペーションで活きる
```

### Trance / Psytranceサウンドデザイン

```
Trance向けAnalog設定:

【クラシックトランスリード（Supersaw的）】
  OSC1: Saw, Level: 100%
  OSC2: Saw, Detune: +15, Level: 95%
  Noise: White, 3%
  Filter: 24dB LP, Cutoff: 3500Hz, Reso: 30%, Drive: 15%
  Filter Env: A:10ms D:600ms S:50% R:400ms, Amount: +45%
  Amp Env: A:5ms D:400ms S:80% R:300ms
  Unison: 6-8 Voices, Detune: 40%
  LFO → Pitch: Sine, Rate: 5.5Hz, Amount: 5%（ビブラート）
  → 壮大なトランスリード
  → メインメロディに

【トランスゲートパッド】
  OSC1: Saw, Level: 100%
  OSC2: Saw, Semi: +12, Detune: +7, Level: 75%
  Filter: 24dB LP, Cutoff: 1500Hz, Reso: 20%
  LFO → Amp: Square, Rate: 1/8（テンポ同期）, Amount: 90%
  Amp Env: A:500ms D:500ms S:80% R:1000ms
  Unison: 4 Voices, Detune: 30%
  → リズミカルなゲートパッド
  → トランスのビルドアップに

【Psytranceベース】
  OSC1: Saw, Level: 100%
  OSC2: Square, Width: 35%, Level: 80%
  Filter: 24dB LP, Cutoff: 700Hz, Reso: 40%, Drive: 35%
  Filter Env: A:0 D:200ms S:15% R:100ms, Amount: +70%
  Amp Env: A:0 D:180ms S:20% R:80ms
  LFO → Cutoff: Saw Down, Rate: 1/16, Amount: 35%
  Unison: Off（モノフォニック）
  → 強烈なサイケデリックベース
  → 16分の繰り返しパターンで

【アップリフティングパッド】
  OSC1: Triangle, Level: 100%
  OSC2: Saw, Semi: +12, Level: 50%
  Noise: Pink, 8%
  Filter: 12dB LP, Cutoff: 1800Hz, Reso: 15%
  LFO → Cutoff: Sine, Rate: 0.1Hz, Amount: 20%
  Amp Env: A:3000ms D:2000ms S:70% R:5000ms
  Unison: 6 Voices, Detune: 35%
  → 壮大で浮遊感のあるパッド
  → ブレイクダウンの背景に
```

### Ambient / Electronicaサウンドデザイン

```
Ambient向けAnalog設定:

【ドリフティングドローン】
  OSC1: Triangle, Level: 100%
  OSC2: Triangle, Semi: +7, Detune: +3, Level: 70%
  Noise: Pink, 15%
  Filter: 24dB LP, Cutoff: 600Hz, Reso: 10%
  LFO1 → Cutoff: Sine, Rate: 0.05Hz, Amount: 25%
  LFO2 → Pitch: Sine, Rate: 0.08Hz, Amount: 2%
  Amp Env: A:8000ms D:5000ms S:60% R:10000ms
  Unison: 8 Voices, Detune: 50%
  → 非常にゆっくりと変化するドローン
  → アンビエントインスタレーション向け

【クリスタルベル】
  OSC1: Triangle, Level: 100%
  OSC2: Triangle, Semi: +19（12+7 = オクターブ+5度）, Level: 40%
  Filter: 12dB LP, Cutoff: 4000Hz, Reso: 25%
  Filter Env: A:0 D:2000ms S:10% R:3000ms, Amount: +30%
  Amp Env: A:0 D:3000ms S:5% R:5000ms
  Unison: 2 Voices, Detune: 5%
  → 透明感のあるベルサウンド
  → アンビエントのアクセントに

【テクスチャレイヤー】
  OSC1: Saw, Level: 60%
  OSC2: Square, Width: 20%, Semi: +5, Level: 40%
  Noise: White, 25%
  Filter: Band Pass, Cutoff: 1500Hz, Reso: 40%
  LFO → Cutoff: Random, Rate: 3Hz, Amount: 30%
  LFO → Pan: Sine, Rate: 0.3Hz, Amount: 60%
  Amp Env: A:5000ms D:3000ms S:40% R:8000ms
  Unison: 4 Voices, Detune: 45%
  → 有機的に変化するテクスチャ
  → バックグラウンドレイヤーに
```

---

## ベースサウンドの極意

### ベースサウンドデザインの基本原則

Analogはアナログベースの再現に非常に優れています。Moogベース、Juno的ベース、モダンなエレクトロベースまで幅広くカバーできます。

```
ベースサウンドの基本原則:

1. モノフォニックが基本:
   ベースは単音で演奏するのが一般的
   → Analogの設定でMonoモードを選択
   → Glide（ポルタメント）で滑らかな音程移動

2. 低域を太くする工夫:
   OSC2をSub（-12 Semi）として使用
   → Triangle波が最もクリーンなSub
   → Saw波のSubは存在感があるが濁りやすい

3. フィルターが命:
   カットオフの設定がベースの「キャラクター」を決める
   → 400Hz以下: サブベース、ディープ
   → 400-800Hz: スタンダードなアナログベース
   → 800-1500Hz: ブライトなファンキーベース

4. エンベロープでアタック感を制御:
   Filter Envの設定がベースの「噛み付き」を決める
   → Amount高い: アタックが明るく弾ける
   → Amount低い: 均一で暗いベース

5. Driveで質感を付加:
   15-30%程度のDriveで「アナログらしい」太さ
   → 上げすぎると音が潰れるので注意
```

```
ベースサウンド比較チャート:

【Moog風 ファットベース】
  太さ:      ★★★★★
  明るさ:    ★★★☆☆
  パンチ:    ★★★★☆
  サブ感:    ★★★★★
  設定ポイント: 24dB LP + 高Drive + Saw波

【Juno風 バウンシーベース】
  太さ:      ★★★☆☆
  明るさ:    ★★★★☆
  パンチ:    ★★★★★
  サブ感:    ★★★☆☆
  設定ポイント: Square波 + 短いFilter Env + 低Resonance

【303風 アシッドベース】
  太さ:      ★★★★☆
  明るさ:    ★★★★★（スイープ時）
  パンチ:    ★★★☆☆
  サブ感:    ★★☆☆☆
  設定ポイント: Saw波 + 高Resonance + 高Filter Env Amount

【Sub Bass（サブベース）】
  太さ:      ★★★★★
  明るさ:    ★☆☆☆☆
  パンチ:    ★★☆☆☆
  サブ感:    ★★★★★
  設定ポイント: Triangle/Sine波 + Cutoff極低 + No Unison
```

---

## パッドサウンドの深掘り

### パッド作りの哲学

パッドサウンドはトラックの「空気感」や「雰囲気」を作る要素です。Analogの温かみのあるサウンドはパッド制作に理想的で、デジタルシンセでは得られない有機的な質感を生み出せます。

```
パッドデザインの5つの柱:

1. ゆっくりしたエンベロープ:
   Attack: 500ms-5000ms（ゆっくりフェードイン）
   Release: 1000ms-10000ms（長い余韻）
   → 急激な変化を避け、空間を満たす

2. デチューンによる厚み:
   OSC2のDetuneで微妙なピッチ差を作る
   Unisonで複数ボイスを重ねる
   → 単調さを避け、有機的な動きを生む

3. フィルターの穏やかな動き:
   LFOでCutoffをゆっくり変調
   Filter Envで時間変化を付与
   → 「呼吸」しているような生命感

4. ノイズの隠し味:
   Pink Noise 3-10%で空気感を追加
   → 音の密度が増し、より自然に

5. ステレオの広がり:
   Unisonのステレオスプレッド
   LFOによるオートパン
   → 空間いっぱいに広がるサウンド

パッドの音域と役割:

  低域パッド（C2-C3）:
    → ベースの補強
    → ウォームなベッド
    → Cutoff: 400-800Hz
    → 注意: ベースとの被りに注意

  中域パッド（C3-C5）:
    → メインの雰囲気作り
    → コード感の演出
    → Cutoff: 800-2000Hz
    → 最も汎用性が高い

  高域パッド（C5-C7）:
    → シマー、きらめき
    → 空間の高い部分を埋める
    → Cutoff: 2000-5000Hz
    → Noiseを多めに混ぜても良い
```

```
パッドバリエーション一覧:

【Warm Analog Pad（定番）】
  キャラクター: 温かく包み込む
  OSC: Saw + Saw（+12, Detune +7）
  Filter: 24dB LP, Cutoff 1200Hz
  LFO: Cutoff Sine 0.2Hz, Amount 15%
  Unison: 6 Voices, Detune 40%
  用途: ほぼ全てのジャンルで使える定番

【Glass Pad（ガラス的）】
  キャラクター: 透明で煌めく
  OSC: Triangle + Triangle（+19, Detune +5）
  Filter: 12dB LP, Cutoff 3000Hz
  LFO: Cutoff Sine 0.1Hz, Amount 10%
  Unison: 4 Voices, Detune 20%
  用途: アンビエント、チルアウト

【Dark Drone Pad（暗いドローン）】
  キャラクター: 重く、深く、暗い
  OSC: Saw + Square（Width 25%, -12）
  Filter: 24dB LP, Cutoff 500Hz, Reso 30%
  LFO: Cutoff Sine 0.05Hz, Amount 20%
  Unison: 8 Voices, Detune 50%
  用途: ダークアンビエント、ホラー、インダストリアル

【PWM String Pad（弦楽器的）】
  キャラクター: ストリングアンサンブル的
  OSC: Square（Width 50%）+ Square（Width 35%, +12）
  LFO → PW: Triangle, Rate 0.7Hz, Amount 35%
  Filter: 12dB LP, Cutoff 2500Hz
  Unison: 4 Voices, Detune 25%
  用途: 80sリバイバル、シンセウェイブ

【Breathing Pad（呼吸するパッド）】
  キャラクター: ゆっくり呼吸しているような
  OSC: Saw + Triangle（+12）
  Filter: 24dB LP, Cutoff 1000Hz
  LFO → Cutoff: Sine, Rate 0.15Hz, Amount 35%
  LFO → Amp: Sine, Rate 0.1Hz, Amount 15%
  Unison: 6 Voices, Detune 35%
  用途: メディテーション、ヒーリング、アンビエント
```

---

## よくある質問

### Q1: WavetableとAnalogどっち？

**A:** 用途次第

```
Wavetable推奨:

モダンな音:
Techno Bass
EDM Lead

クリア:
はっきり

Analog推奨:

温かい音:
パッド
ビンテージLead

アナログ感:
Moog的

推奨:

ベース → Wavetable
パッド → Analog
リード → どちらでも
```

### Q2: Unisonは常にOn？

**A:** 音色次第

```
Unison On推奨:

パッド:
Voices 4-6
必須

厚いリード:
Voices 2-3

Unison Off:

クリーンなベース:
単音
Unisonなし

CPU節約:
重い時Off

推奨:

パッド: 必須On
リード: お好みで
ベース: 基本Off
```

### Q3: 24dB vs 12dB Filter？

**A:** ほぼ24dB

```
24dB:
急峻
Moog的
温かい
推奨

12dB:
緩やか
明るい

使い分け:

99%: 24dB
特殊: 12dB

理由:
24dBがAnalogの特徴
```

---

## リードサウンドとプラックの作り方

### リードシンセの音作り理論

リードシンセは楽曲のメロディを担う最も目立つ要素です。Analogの温かみあるキャラクターは、ビンテージからモダンまで多彩なリードサウンドを実現します。

```
リードサウンドの設計原則:

1. 明瞭なアタック:
   Amp Env Attack: 0-10ms
   → メロディの音程が即座に認識される
   → 遅いアタックはパッドに近づく

2. フィルターエンベロープで「噛みつき」:
   Filter Env Amount: +40-70%
   Filter Env Decay: 200-600ms
   → アタック時に明るく、その後暗くなる
   → Moogリード特有のキャラクター

3. 適度なレゾナンス:
   Resonance: 30-50%
   → カットオフ周辺が強調されて存在感UP
   → 高すぎるとキンキンして耳障り

4. Unisonで厚みを出す:
   2-4 Voices, Detune 15-30%
   → 単音でも力強い存在感
   → 多すぎるとリードの輪郭がぼやける

5. ビブラートで生命感:
   LFO → Pitch: Sine, 4-6Hz, Amount 3-8%
   → 人の歌声に近い揺らぎ
   → Mod Wheelで制御するとライブ感UP
```

```
リードサウンドバリエーション:

【ファットMoogリード】
  OSC1: Saw, Level: 100%
  OSC2: Saw, Detune: +8, Level: 90%
  Sub: Triangle, -12, Level: 20%
  Filter: 24dB LP, Cutoff: 2500Hz, Reso: 40%, Drive: 35%
  Filter Env: A:5ms D:400ms S:40% R:300ms, Amount: +60%
  Amp Env: A:3ms D:300ms S:75% R:200ms
  Unison: 2 Voices, Detune: 12%
  LFO → Pitch: Sine, 5Hz, Amount: 5%
  → Minimoog的な太いリード
  → ソロメロディに最適

【シンセブラス】
  OSC1: Saw, Level: 100%
  OSC2: Square, Width: 40%, Semi: 0, Level: 75%
  Filter: 24dB LP, Cutoff: 1800Hz, Reso: 25%, Drive: 20%
  Filter Env: A:30ms D:300ms S:50% R:200ms, Amount: +50%
  Amp Env: A:20ms D:200ms S:80% R:150ms
  Unison: 4 Voices, Detune: 20%
  → ホーンセクション的なサウンド
  → コードスタブにも使える

【ナスティリード（攻撃的）】
  OSC1: Saw, Level: 100%
  OSC2: Square, Width: 25%, Detune: +12, Level: 95%
  Noise: White, 5%
  Filter: 24dB LP, Cutoff: 3500Hz, Reso: 55%, Drive: 50%
  Filter Env: A:0 D:500ms S:35% R:300ms, Amount: +70%
  Amp Env: A:0 D:400ms S:70% R:250ms
  Unison: 3 Voices, Detune: 20%
  → 歪みと攻撃性のあるリード
  → Techno/Industrial向け

【メランコリックリード（哀愁系）】
  OSC1: Triangle, Level: 100%
  OSC2: Saw, Semi: +12, Detune: +5, Level: 50%
  Filter: 24dB LP, Cutoff: 1500Hz, Reso: 20%, Drive: 10%
  Filter Env: A:30ms D:800ms S:45% R:600ms, Amount: +30%
  Amp Env: A:15ms D:500ms S:70% R:400ms
  LFO → Pitch: Sine, 5Hz, Amount: 6%（遅延ビブラート推奨）
  Unison: 2 Voices, Detune: 8%
  → 柔らかく切ない音色
  → メロディックテクノ、プログレッシブハウスに
```

### プラックサウンドの作り方

プラック（Pluck）は短いDecayとゼロSustainで作る、弦を弾いたような鋭い音色です。アルペジオやシーケンスに欠かせません。

```
プラックサウンドの基本構造:

重要ポイント:
  Sustain = 0% が基本（Amp Env, Filter Env両方）
  → キーを押し続けても音は減衰する
  → 弦楽器の「弾く」動作を再現

  Decayの長さがキャラクターを決定:
    50-100ms: 極短プラック（チック音）
    100-300ms: 標準プラック（ギター的）
    300-600ms: ロングプラック（ハープ的）

【ベーシックプラック】
  OSC1: Saw, Level: 100%
  OSC2: Triangle, Semi: +12, Level: 40%
  Filter: 24dB LP, Cutoff: 800Hz, Reso: 20%
  Filter Env: A:0 D:200ms S:0% R:150ms, Amount: +55%
  Amp Env: A:0 D:250ms S:0% R:200ms
  → クリーンで使いやすいプラック
  → アルペジオの定番

【メタリックプラック】
  OSC1: Square, Width: 30%, Level: 100%
  OSC2: Saw, Semi: +7, Level: 60%
  Filter: 24dB LP, Cutoff: 1200Hz, Reso: 45%, Drive: 25%
  Filter Env: A:0 D:150ms S:0% R:100ms, Amount: +65%
  Amp Env: A:0 D:200ms S:0% R:150ms
  → 金属的な響きのプラック
  → テクノシーケンスに

【ソフトプラック（マリンバ的）】
  OSC1: Triangle, Level: 100%
  OSC2: Triangle, Semi: +12, Level: 30%
  Filter: 12dB LP, Cutoff: 2000Hz, Reso: 10%
  Filter Env: A:0 D:300ms S:0% R:250ms, Amount: +25%
  Amp Env: A:0 D:400ms S:0% R:350ms
  → 柔らかい木琴的なプラック
  → チルアウト、ローファイに

【シャーププラック（DJ用）】
  OSC1: Saw, Level: 100%
  OSC2: Saw, Detune: +10, Level: 90%
  Filter: 24dB LP, Cutoff: 1500Hz, Reso: 35%, Drive: 15%
  Filter Env: A:0 D:120ms S:0% R:80ms, Amount: +70%
  Amp Env: A:0 D:150ms S:0% R:100ms
  Unison: 2 Voices, Detune: 15%
  → 鋭くパンチのあるプラック
  → ドロップのアクセントに
```

---

## FXサウンドデザイン

### Analogを使ったエフェクトサウンド

Analogは楽音だけでなく、効果音やトランジションサウンドの制作にも活用できます。

```
FXサウンド レシピ集:

【ライザー（上昇音）】
  OSC1: Saw, Level: 100%
  OSC2: Saw, Detune: +20, Level: 80%
  Noise: White, 15%
  Filter: 24dB LP, Cutoff: オートメーション（200Hz → 8000Hz）
  Resonance: 45%
  Amp Env: A:4000ms D:0 S:100% R:500ms
  Unison: 6 Voices, Detune: 45%
  使い方:
    4-8小節かけてカットオフを上げる
    Resoが高いほどドラマチック
    ビルドアップの定番エフェクト

【ダウンスイープ（下降音）】
  OSC1: Saw, Level: 100%
  Noise: White, 20%
  Filter: 24dB LP, Cutoff: オートメーション（6000Hz → 200Hz）
  Resonance: 50%
  Drive: 30%
  使い方:
    ドロップ直後の1-2拍で急降下
    インパクトの強調

【ホワイトノイズスウィープ】
  OSC1: Off
  OSC2: Off
  Noise: White, 100%
  Filter: 24dB LP, Cutoff: オートメーション
  Resonance: 30%
  Amp Env: A:2000ms D:1000ms S:80% R:3000ms
  使い方:
    海の波のような効果
    トランジション、フィルイン

【レーザー効果音】
  OSC1: Saw, Level: 100%
  Filter: 24dB LP, Cutoff: 5000Hz, Reso: 80%
  Filter Env: A:0 D:100ms S:0% R:50ms, Amount: +90%
  Amp Env: A:0 D:150ms S:0% R:50ms
  Pitch Env: 非常に速いDecay（+24 Semi → 0）
  使い方:
    短い単発のアクセント
    SFX的な用途

【サイレン】
  OSC1: Saw, Level: 100%
  Filter: 24dB LP, Cutoff: 3000Hz, Reso: 30%
  LFO → Pitch: Sine, Rate: 2Hz, Amount: 50%
  Amp Env: A:500ms D:0 S:100% R:500ms
  使い方:
    警告音的な効果
    テクノのブレイクダウンに
```

---

## レイヤリングテクニック

### Analogを他のインストゥルメントと組み合わせる

単体のAnalogでは得られない音の厚みや複雑さを、レイヤリング（複数の音源を重ねる）で実現します。

```
レイヤリングの基本原則:

1. 各レイヤーの役割を明確にする:
   Layer 1 (Body):  メインの音色、中域担当
   Layer 2 (Top):   アタックや高域の明るさ
   Layer 3 (Sub):   低域の土台
   Layer 4 (Noise): 空気感、テクスチャ

2. 周波数帯域の棲み分け:
   各レイヤーが異なる帯域を担当
   → EQで被りを処理
   → フィルターのCutoffで住み分け

3. 音量バランス:
   メインレイヤー: 0 dB（基準）
   サブレイヤー: -3 〜 -6 dB
   テクスチャ: -6 〜 -12 dB

実践的なレイヤリング例:

【リッチなリードサウンド】
  Layer 1 (Analog):
    Saw + Saw, Unison 2, 中域担当
    Filter: 24dB LP, Cutoff 2500Hz
  Layer 2 (Wavetable):
    デジタルな高域のきらめき
    HighPass 2000Hz以上
  Layer 3 (Analog):
    Triangle -12 Semi, サブレイヤー
    LowPass 500Hz以下

【厚いパッドサウンド】
  Layer 1 (Analog):
    Warm Pad（メイン）
    Cutoff: 1200Hz, Unison 6
  Layer 2 (Analog):
    PWM String Pad（高域）
    Cutoff: 2500Hz, Unison 4
  Layer 3 (Operator):
    FM的なベル音（アクセント）
    高域でキラキラ

【パワフルなベースサウンド】
  Layer 1 (Analog):
    Saw + Square（メインベース）
    Cutoff: 800Hz, Mono
  Layer 2 (Analog):
    Triangle -12 Semi（サブベース）
    Cutoff: 200Hz, No Filter Env
  処理:
    Layer 2にサイドチェインコンプ
    キックとの棲み分け
```

---

## プリセット分析と改造テクニック

### 既存プリセットから学ぶ

Analogのファクトリープリセットは音作りの教科書です。プリセットのパラメータを分析することで、サウンドデザインの技法を学ぶことができます。

```
プリセット分析の手順:

Step 1: プリセットをロード
  → まず音を聴いて印象をメモ

Step 2: オシレーターセクションを確認
  → どの波形の組み合わせか
  → DetuneやSemiの設定は
  → Noiseの量は

Step 3: フィルター設定を確認
  → フィルタータイプは何か
  → Cutoff、Resonance、Driveの値
  → Keytrackingの設定

Step 4: エンベロープを確認
  → Filter EnvのADSRとAmount
  → Amp EnvのADSR
  → 全体的な時間軸の傾向

Step 5: LFOとモジュレーションを確認
  → LFOの行き先とAmount
  → ベロシティのマッピング

Step 6: パラメータを1つずつ変えてみる
  → 各パラメータの効果を体感
  → 極端に変えて影響を理解

プリセット改造のコツ:
  1. まずCutoffだけを動かす → 音色の明暗
  2. Resonanceを上下 → キャラクター変化
  3. Filter Env Amountを変更 → アタック感
  4. Unison Voicesを変更 → 厚み
  5. LFOのRateとAmount → 動きの変化
  → 少しずつ変えて「自分の音」にする
```

---

## パフォーマンスでのAnalog活用

### ライブセットとDJパフォーマンス

Analogはライブパフォーマンスにおいて、リアルタイムの音色操作で大きな効果を発揮します。

```
パフォーマンス向けパラメータマッピング:

Push / Launchpad での推奨マッピング:

  ノブ1: Filter Cutoff
         最も効果的なリアルタイム操作
         フィルタースイープの定番

  ノブ2: Filter Resonance
         カットオフと組み合わせて音色変化

  ノブ3: LFO Rate
         モジュレーション速度の変更
         遅い → 速いで盛り上げ

  ノブ4: LFO Amount
         モジュレーション深さの変更
         ドライ → ウェットへ

  ノブ5: Filter Env Amount
         アタック感の変化
         暗い → 明るい

  ノブ6: Unison Detune
         厚み/コーラスの変化
         タイト → ワイド

  ノブ7: Drive
         歪みの追加/除去
         クリーン → ダーティ

  ノブ8: Volume
         全体音量

パフォーマンス中のテクニック:

  フィルタースイープ:
    Cutoffを徐々に開閉
    → ビルドアップ/ブレイクダウン
    → 最も基本的かつ効果的

  レゾナンスピーク:
    Resoを急に上げてCutoffをスイープ
    → ドラマチックな効果
    → DJトランジションに

  LFO Rate加速:
    LFO Rateを遅い→速いへ
    → テンション上昇
    → ドロップ前のビルドに

  キルスイッチ的使用:
    Cutoffを急に閉じる → 開く
    → ブレイクの演出
    → オーディエンスの注意を引く
```

---

## CPU最適化とプロジェクト管理

### Analog使用時のパフォーマンス対策

Analogは比較的CPU効率の良いインストゥルメントですが、Unison Voicesの多用やポリフォニー数によってはCPU負荷が上がります。

```
CPU負荷の要因と対策:

負荷が高くなる要因:
  1. Unison Voices数が多い（6-8）
  2. ポリフォニー数が多い（コードを多重に）
  3. 複数トラックでAnalogを使用
  4. LFOの高速レート
  5. Filter Driveの高い値

最適化テクニック:

  Freezeを活用:
    完成したAnalogトラックはFreeze
    → CPU負荷ゼロに
    → 編集時のみUnfreeze

  Unison Voicesを減らす:
    パッド: 6→4でも十分な厚み
    リード: 3→2で問題なし
    ベース: 基本的にOff

  ポリフォニーを制限:
    Analog > Global > Voices
    ベース: 1（モノフォニック）
    リード: 4
    パッド: 8
    → 不必要に高いポリフォニーは避ける

  バウンス（Flatten）:
    Analogをオーディオに書き出し
    → 最も確実なCPU削減
    → ただし後から編集不可

  リサンプリング:
    Analogの出力を別トラックに録音
    → オーディオとして扱える
    → 元のAnalogはミュート/削除可能

CPU負荷の目安:
  Analog 1インスタンス（Unison Off）: 約1-2%
  Analog 1インスタンス（Unison 4）:   約3-5%
  Analog 1インスタンス（Unison 8）:   約5-8%
  → 10トラック以上使う場合はFreeze推奨
```

---

## 実践レシピ集: 10の定番サウンド

### 音作りクイックリファレンス

現場で即座に使える10の定番サウンドレシピをまとめます。各レシピは5分以内で作成可能です。

```
レシピ1: 【808風サブベース】
  OSC1: Triangle, Level: 100%
  OSC2: Off
  Noise: Off
  Filter: 24dB LP, Cutoff: 250Hz, Reso: 0%
  Filter Env: Off（Amount: 0%）
  Amp Env: A:0 D:1500ms S:0% R:200ms
  Unison: Off, Mono
  Glide: On, Time: 30ms
  → 808キック的なロングサブベース
  → トラップ、ヒップホップの定番

レシピ2: 【Reese Bass（リースベース）】
  OSC1: Saw, Level: 100%
  OSC2: Saw, Detune: +25, Level: 100%
  Filter: 24dB LP, Cutoff: 500Hz, Reso: 20%, Drive: 20%
  LFO → Cutoff: Sine, Rate: 0.3Hz, Amount: 25%
  Amp Env: A:0 D:0 S:100% R:100ms
  Unison: Off, Mono
  → ドラムンベース/ジャングルの定番ベース
  → デチューンの「うねり」がポイント

レシピ3: 【Hoover Sound（フーバー）】
  OSC1: Saw, Level: 100%
  OSC2: Saw, Semi: -12, Detune: +30, Level: 90%
  Noise: White, 5%
  Filter: 24dB LP, Cutoff: 1500Hz, Reso: 30%, Drive: 40%
  LFO → Pitch: Sine, Rate: 3Hz, Amount: 15%
  Amp Env: A:50ms D:0 S:100% R:300ms
  Unison: 4 Voices, Detune: 50%
  Glide: On, Time: 150ms
  → 90年代レイブの象徴的サウンド
  → 掃除機のような「ブーン」という音

レシピ4: 【チップチューンリード】
  OSC1: Square, Width: 50%, Level: 100%
  OSC2: Square, Width: 25%, Semi: +12, Level: 50%
  Filter: Off（全開）
  Amp Env: A:0 D:0 S:100% R:30ms
  Unison: Off
  → 8bitゲーム風のリード
  → ファミコン/ゲームボーイ的

レシピ5: 【Acid 303ベース】
  OSC1: Saw, Level: 100%
  OSC2: Off
  Filter: 24dB LP, Cutoff: 500Hz, Reso: 70%, Drive: 30%
  Filter Env: A:0 D:200ms S:0% R:100ms, Amount: +80%
  Amp Env: A:0 D:300ms S:0% R:50ms
  Unison: Off, Mono
  Glide: On, Time: 60ms
  → TB-303的なアシッドベース
  → スライドとアクセントで表現力UP

レシピ6: 【Prophet-5風パッド】
  OSC1: Saw, Level: 100%
  OSC2: Square, Width: 50%, Semi: +12, Level: 70%
  Noise: Pink, 5%
  Filter: 24dB LP, Cutoff: 1500Hz, Reso: 15%, Drive: 10%
  Filter Env: A:500ms D:1000ms S:50% R:1500ms, Amount: +25%
  Amp Env: A:1000ms D:800ms S:75% R:2000ms
  Unison: 4 Voices, Detune: 30%
  → Sequential Prophet-5的な温かいパッド
  → 70s/80sシンセサウンド

レシピ7: 【ディープダブコード】
  OSC1: Saw, Level: 90%
  OSC2: Triangle, Semi: +12, Level: 50%
  Noise: Pink, 8%
  Filter: 24dB LP, Cutoff: 1000Hz, Reso: 25%, Drive: 15%
  Filter Env: A:20ms D:500ms S:40% R:400ms, Amount: +35%
  Amp Env: A:10ms D:400ms S:60% R:500ms
  Unison: 4 Voices, Detune: 20%
  → ディレイとリバーブを後段に追加
  → ダブテクノの定番コードサウンド

レシピ8: 【シンセストリングス】
  OSC1: Square, Width: 50%, Level: 100%
  OSC2: Square, Width: 30%, Semi: +12, Level: 60%
  LFO → PW: Triangle, Rate: 0.8Hz, Amount: 35%
  Filter: 12dB LP, Cutoff: 2500Hz, Reso: 10%
  Amp Env: A:800ms D:500ms S:80% R:1000ms
  Unison: 6 Voices, Detune: 30%
  → Juno-60/Solina的なストリングス
  → コーラスエフェクトを追加で完璧に

レシピ9: 【パーカッシブシンセヒット】
  OSC1: Saw, Level: 100%
  OSC2: Square, Detune: +7, Level: 80%
  Noise: White, 10%
  Filter: 24dB LP, Cutoff: 2000Hz, Reso: 35%, Drive: 25%
  Filter Env: A:0 D:80ms S:0% R:50ms, Amount: +75%
  Amp Env: A:0 D:100ms S:0% R:50ms
  Unison: 4 Voices, Detune: 25%
  → 短くパンチのあるヒットサウンド
  → ダウンビートのアクセントに

レシピ10: 【アンビエントテクスチャ】
  OSC1: Triangle, Level: 70%
  OSC2: Saw, Semi: +19, Level: 30%
  Noise: Pink, 20%
  Filter: Band Pass, Cutoff: 1200Hz, Reso: 35%
  LFO1 → Cutoff: Random, Rate: 1.5Hz, Amount: 25%
  LFO2 → Pan: Sine, Rate: 0.2Hz, Amount: 50%
  Amp Env: A:5000ms D:3000ms S:50% R:8000ms
  Unison: 4 Voices, Detune: 40%
  → 有機的で環境音楽的なテクスチャ
  → Reverbを深くかけて使用
```

---

## Glide（グライド/ポルタメント）の活用

### 音程の滑らかな移動

Glide（ポルタメント）は、ノート間の音程移動を滑らかにする機能です。ベースラインやリードに表現力を加えます。

```
Glideの設定パラメータ:

Time（グライドタイム）:
  0ms:     OFF（即座にピッチ変更）
  10-30ms: 極短（微妙なスラー効果）
  30-80ms: 短め（ベースラインに最適）
  80-200ms: 中程度（リードのスライド）
  200ms+:  長い（効果音的）

Mode:
  Always:   全てのノート間でグライド
  Legato:   レガート（繋げて弾いた時だけ）
            → 最も音楽的な設定
            → スタッカートでは即座にピッチ変更

推奨設定:
  アシッドベース: Time 60ms, Legato
  Moogリード: Time 100ms, Legato
  エフェクト的: Time 300ms+, Always
  DJ用ベース: Time 40ms, Legato
```

---

## まとめ

### Analog基礎

```
□ 減算式合成
□ 2オシレーター + Noise
□ Moogスタイルフィルター
□ Unison で厚み
□ 温かみのある音色
```

### 音色別設定

```
パッド:
Saw + Saw, Cutoff 1200 Hz, Unison 6

リード:
Saw + Saw, Cutoff 3000 Hz, Unison 2-3

ベース:
Saw + Square, Cutoff 800 Hz, Unison Off
```

### 重要ポイント

```
□ OSC2 をDetune
□ Unison でパワー
□ 24dB Lowpass推奨
□ Noise わずかに
□ Drive で温かみ
```

---

**次は:** [Sampler](./sampler.md) - サンプルを自在に操る

---

## 次に読むべきガイド

- [Drum Rack](./drum-rack.md) - 次のトピックへ進む

---

## 参考文献

- [MDN Web Docs](https://developer.mozilla.org/) - Web技術のリファレンス
- [Wikipedia](https://ja.wikipedia.org/) - 技術概念の概要
