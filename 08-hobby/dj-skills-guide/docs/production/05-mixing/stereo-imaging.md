# Stereo Imaging

ステレオ空間で広がりを作ります。Width・Panning・120 Hz以下Mono化を完全マスターします。

## この章で学ぶこと

- Stereo vs Mono基礎
- Width調整（Utility）
- Panning戦略
- 120 Hz以下Mono必須
- Mid/Side処理
- ステレオ幅測定
- Mono互換性確認
- Haasエフェクトと擬似ステレオ
- コーラス/ディチューンによるワイドニング
- 周波数帯域別ステレオ処理
- ステレオイメージャープラグイン比較
- ジャンル別ステレオ戦略
- マスタリングでのステレオ処理
- 実践テクニック集

---

## なぜStereo Imagingが重要なのか

**広がりの秘訣:**

```
Stereo Imaging悪い:

特徴:
狭い
平坦
単調

Stereo Imaging良い:

特徴:
広い
立体的
没入感

プロとアマの差:

アマ:
全てCenter
狭い

プロ:
適切な配置
Wide

結果:
プロ: 広大、立体的
アマ: 狭い、平坦

重要ルール:

120 Hz以下:
必ずMono

理由:
低域Stereo
→ 位相問題
→ Mono再生で消える
```

### ステレオイメージングの歴史と進化

ステレオイメージングの技術は、録音技術の発展とともに進化してきました。この歴史を理解することで、現代のテクニックをより深く活用できます。

```
1930年代: ステレオ録音の発明
  Alan Blumlein がステレオ録音技術の特許を取得
  2チャンネル録音の基礎が確立

1950-60年代: ステレオレコードの普及
  ハードパンニングが主流
  Beatles などがステレオミックスを実験
  ドラムが片チャンネル、ボーカルがもう片方など極端な配置

1970年代: より自然なステレオミックスへ
  パンポットの普及
  中央配置とサイド配置のバランスが洗練
  ステレオマイキング技法の発達

1980年代: デジタル時代の幕開け
  デジタルリバーブによるステレオ空間の拡張
  エレクトロニックミュージックでのステレオ実験
  Lexicon 480L などの名機が登場

1990年代: DAW時代の到来
  プラグインによるステレオ処理が一般化
  Mid/Side処理のデジタル実装
  マスタリングでのステレオ幅コントロールが標準に

2000年代: プラグインの進化
  iZotope Ozone などの統合ツール登場
  マルチバンドステレオ処理の普及
  ラウドネス戦争とステレオ幅の拡大競争

2010年代以降: イマーシブオーディオ
  Dolby Atmos の登場
  空間オーディオ（Apple Spatial Audio）
  バイノーラル処理の一般化
  3Dオーディオへの拡張
```

### 人間の聴覚とステレオ知覚

ステレオイメージングを効果的に活用するには、人間がどのようにステレオ空間を知覚するかを理解することが重要です。

```
両耳間時間差（ITD: Interaural Time Difference）:

原理:
  音源が右側にある場合:
  → 右耳に先に到達
  → 左耳に遅れて到達
  → 最大約0.6-0.7 ms の時間差

知覚への影響:
  低周波数（~1500 Hz以下）: ITDが主な定位手がかり
  脳が時間差から方向を判断
  Haasエフェクトの基礎原理

両耳間レベル差（ILD: Interaural Level Difference）:

原理:
  頭部が音波を遮蔽（ヘッドシャドウ効果）
  反対側の耳に届く音は減衰
  高周波数ほど効果が大きい

知覚への影響:
  高周波数（~1500 Hz以上）: ILDが主な定位手がかり
  パンニングで音量差を付ける根拠

周波数と定位感の関係:
  ~200 Hz以下: 方向感ほぼなし → Mono化が合理的
  200-1500 Hz: ITDで大まかな方向感
  1500 Hz以上: ILD + ITDで精密な定位
  8000 Hz以上: ピンナ（耳介）による上下方向の手がかり

実践への応用:
  低域はCenterに → 知覚的に方向感がないため
  高域の要素はPanningが効果的 → ILDによる定位が明確
  中域はWidth調整で微妙な広がりを制御
```

---

## Stereo vs Mono基礎

**基本理解:**

### 定義

```
Mono:

チャンネル: 1つ
L = R
同じ音

特徴:
中央定位
タイト
パワフル

用途:
Kick・Bass・Vocal

Stereo:

チャンネル: 2つ
L ≠ R
違う音

特徴:
広がり
空間感

用途:
Pad・FX・Hi-Hat

Mono互換性:

問題:
Stereo信号
→ Mono再生
→ 位相キャンセル
→ 音量down/消える

解決:
120 Hz以下Mono化
必須
```

### ステレオ信号の成り立ちを深く理解する

ステレオイメージングを適切に行うには、ステレオ信号がどのように構成されているかを正確に理解する必要があります。

```
ステレオ信号の数学的表現:

Left Channel (L):
  L = Mid + Side

Right Channel (R):
  R = Mid - Side

Mid成分:
  Mid = (L + R) / 2
  → 両チャンネルに共通する成分
  → Mono再生時に残る成分
  → 中央に定位する音

Side成分:
  Side = (L - R) / 2
  → 両チャンネルの差分
  → Mono再生時に消える成分
  → 左右に広がる音

位相の重要性:

同位相（In Phase）:
  L と R が同じ波形 → Mid成分が大きい → Mono的
  Mono再生で音量が保たれる

逆位相（Out of Phase）:
  L と R が反転波形 → Side成分が大きい → Wide
  Mono再生で音量がキャンセルされる

部分的な位相差:
  自然な録音では複雑な位相関係
  プラグインでの処理で意図的に制御

実際のトラックでの例:

完全Mono信号:
  L = R → Side = 0 → Width 0%
  例: モノラルシンセ、センターボーカル

自然なStereo信号:
  L ≈ R（微妙な差）→ Side小さい → Width 20-40%
  例: ステレオマイク録音、軽いコーラス

Wide Stereo信号:
  L ≠ R（大きな差）→ Side大きい → Width 80-100%
  例: ステレオシンセパッド、ダブルトラッキング

過剰なStereo信号:
  L ≒ -R（ほぼ逆位相）→ Side >> Mid → Width 150%+
  例: 過度なステレオエンハンス → 位相問題の原因
```

### Mono互換性が問題になる再生環境の詳細

```
クラブ/ライブ会場:
  大規模PA:
    サブウーファーは通常Mono
    メインスピーカーも場所によってはMono的
    広い会場では位置による位相干渉が発生

  モニタースピーカー:
    DJブースのモニターはMono的配置が多い
    フロアモニターも多くがMono

  対策:
    低域は必ずMono化（120Hz以下）
    中域以上もMono確認を怠らない

スマートフォン/タブレット:
  内蔵スピーカー:
    物理的に近接 → ほぼMono再生
    Stereo効果はほとんど知覚されない
    位相キャンセルが顕著に現れる

  イヤホン/ヘッドホン:
    完全なStereo再生
    ただし片耳使用も多い → Mono的

Bluetooth スピーカー:
  多くがモノラル:
    単一ドライバーのスピーカーが多い
    ステレオペアリングは少数派

  対策:
    Mono再生でも楽曲が成立するミックスが必須

車内オーディオ:
  リスニングポジション:
    運転席は左右非対称
    助手席も同様
    後部座席はさらに複雑

  結果:
    理想的なステレオイメージは実現しにくい
    センター定位の要素が最も重要

ストリーミングプラットフォーム:
  エンコーディングの影響:
    MP3/AAC圧縮でステレオ情報が劣化
    Joint Stereoエンコーディング
    低ビットレートではMono的になる

  対策:
    過度なステレオ処理は圧縮で問題化
    適度なステレオ幅が安全
```

---

## Width調整

**Utility活用:**

### Widthパラメーター

```
Utility > Width:

0%:
完全Mono
L = R

100%:
元のStereo
デフォルト

105-120%:
広げる
Side成分強調

150%+:
過剰
位相問題

-100%:
位相反転
特殊効果

推奨設定:

Kick: 0% (Mono)
Bass: 0% (Mono)
Snare: 0-10%
Lead: 20-30%
Vocal: 0-10%
Pad: 80-100%
FX: 100-120%
Hi-Hat: 40-60%

ルール:
低域ほどMono
高域ほどWide
```

### Width調整の詳細テクニック

Width調整は単にパーセンテージを変えるだけではありません。楽曲のダイナミクスや展開に合わせて動的にコントロールすることで、よりプロフェッショナルなミックスが実現できます。

```
オートメーションによるWidth変化:

ビルドアップ:
  序盤: Width 100%（通常のステレオ）
  ビルドアップ開始: 徐々にWidth縮小
  ドロップ直前: Width 20-30%（ほぼMono）
  → 緊張感の増大、エネルギーの蓄積

ドロップ:
  ドロップ瞬間: Width 100-110%に急拡大
  → 解放感、インパクト
  → コントラストによる知覚的な広がりの強調

ブレイクダウン:
  Width 80-100%で空間的な広がり
  Pad・Atmosphereを広めに配置
  → 休息感、浮遊感

応用テクニック:
  Width LFO: Auto PanのRate極低速で微妙な揺れ
  サイドチェーン連動: Kickに合わせてPadのWidth変化
  フィルター連動: ハイパスが上がるとWidthも縮小

周波数帯域別のWidth管理:

Sub Bass (20-60 Hz):
  Width: 0% (絶対Mono)
  理由: 方向感なし、位相問題の最大リスク

Bass (60-200 Hz):
  Width: 0-10%
  理由: 基本的にMono、微妙な広がりは許容

Low Mid (200-500 Hz):
  Width: 20-40%
  理由: ボディ感を損なわない程度の広がり

Mid (500-2000 Hz):
  Width: 30-60%
  理由: 明瞭度を保ちつつ空間感

High Mid (2000-6000 Hz):
  Width: 50-80%
  理由: プレゼンス帯域、広がりが効果的

High (6000-20000 Hz):
  Width: 70-120%
  理由: エア感、きらびやかさ、広がりが最も効果的
```

### Bass Mono機能

```
Utility > Bass Mono:

機能:
特定周波数以下をMono化

設定:

On/Off: On
Freq: 120 Hz

効果:
120 Hz以下 → Mono
120 Hz以上 → Stereo維持

メリット:

位相安全:
低域問題なし

高域Stereo:
広がり維持

必須:

全トラック:
Bass Mono: On

Master:
Bass Mono: On (120 Hz)

理由:
業界標準
Mono互換性
```

---

## Panning戦略

**左右配置:**

### 基本原則

```
Center (0%):

配置:
Kick・Bass・Snare・Vocal・Lead

理由:
最重要要素
パワー・存在感

L/R (-100% / +100%):

配置:
Hi-Hat・Percussion・FX

理由:
装飾
広がり

推奨配置:

Kick: Center
Bass: Center
Snare: Center
Vocal: Center
Lead: Center

Hi-Hat: L -30% / R +30%
Shaker: L -50% / R +50%
Percussion 1: L -40%
Percussion 2: R +40%

FX 1: L -80%
FX 2: R +80%

Pad: Center (Width 80-100%)

注意:

過剰Pan:
バランス悪い

LRペア:
バランス取る
```

### パンニングの高度なテクニック

```
LCR Panning（Left-Center-Right パンニング）:

概念:
  3つのポジションのみ使用: L / C / R
  中間位置を使わない伝統的手法

メリット:
  明確な定位感
  各要素の分離が良い
  ミックスの決断が早い（3択のみ）

デメリット:
  自然さに欠ける場合がある
  要素が多いトラックでは配置が困難

適用ジャンル:
  ロック、パンクなど（ギターL/R + ボーカルC）
  エレクトロニックではあまり使わない

Spectral Panning（周波数依存パンニング）:

概念:
  周波数帯域ごとに異なるパン位置を設定
  低域はCenter、高域はWide

実装方法:
  マルチバンドスプリッターで帯域分割
  各帯域に個別のPan設定
  または EQ + Pan の組み合わせ

効果:
  より自然で豊かなステレオイメージ
  低域の安定性を保ちつつ高域の広がり

Auto Pan（自動パンニング）:

Ableton Auto Pan:
  Rate: テンポ同期 or Hz指定
  Amount: パンの振り幅
  Phase: L/R の位相オフセット
  Shape: 波形選択

活用例:
  Hi-Hat: 1/8 note同期、Amount 30-50%
  FX Sweep: 1/4 note同期、Amount 80-100%
  Shaker: 1/16 note同期、Amount 20-30%

注意:
  過度のAuto PanはMono互換性を損なう
  重要な要素には使わない
  あくまでも装飾的な効果として使用

Binaural Panning:

概念:
  ヘッドホン再生に最適化されたパンニング
  頭部伝達関数（HRTF）を使用
  より自然な3D定位を実現

ツール:
  dearVR MICRO（無料）
  Waves Nx
  Apple Spatial Audio対応ツール

注意:
  スピーカー再生との互換性に注意
  ヘッドホン専用ミックスとして使用
```

---

## 120 Hz以下Mono化

**必須処理:**

### なぜ必須？

```
理由1: 位相問題

低域Stereo:
L/R位相差

Mono再生:
位相キャンセル
音量down

例:
Bass Stereo
→ Mono再生
→ 消える

理由2: 音響物理

低域:
波長長い
方向感ない

結果:
Stereo効果薄い

理由3: 再生環境

クラブ:
Mono再生多い

車:
Mono的

スマホ:
Mono

必須:
Mono互換性

業界標準:

全プロ:
120 Hz以下Mono

例外:
ほぼなし
```

### 実装方法

```
方法1: Utility (推奨)

全トラック:

Utility挿入
Bass Mono: On
Freq: 120 Hz

方法2: Master Track

Master:

Utility挿入
Bass Mono: On
Freq: 120 Hz

一括処理

方法3: Mid/Side EQ

EQ Eight:
Mode: Mid/Side

Side:
Low Cut 120 Hz

効果:
120 Hz以下 → Mid (Mono)
120 Hz以上 → Side (Stereo)

推奨:
方法1 (各トラック)
最も確実
```

### Mono化の周波数設定の詳細ガイド

```
120 Hz: 業界標準
  最も一般的な設定
  多くのプロエンジニアが採用
  安全かつ効果的

ジャンル別の推奨カットオフ:

テクノ/ハウス:
  120-150 Hz
  キックとベースの低域がMonoで安定
  クラブ再生を最優先

トランス/EDM:
  100-120 Hz
  ベースラインの低域を確保
  サイドチェーンとの相性を考慮

ドラムンベース:
  80-100 Hz
  ベースの周波数帯域が広いため低めに設定
  Reese Bassのステレオ感を一部残す

アンビエント/チルアウト:
  150-200 Hz
  より広いMono帯域で安定感
  ヘッドホン再生が多いが安全策

ポップ/R&B:
  120 Hz
  標準設定で問題なし
  ボーカル中心のミックスに最適

ヒップホップ/トラップ:
  100-120 Hz
  808ベースの低域を考慮
  サブベースのパワーを確保

カットオフ周波数の選び方:

低すぎる場合（60-80 Hz）:
  リスク: 80-120 Hzのステレオ成分が残る
  → Mono再生で問題が生じる可能性
  → クラブ再生で不安定

高すぎる場合（200 Hz+）:
  リスク: 必要以上にMono化
  → ベースラインのステレオ感が失われる
  → ミックスが狭く感じる

最適な判断方法:
  1. 120 Hzで設定
  2. Mono再生で確認
  3. 問題があれば微調整
  4. A/B比較で最終決定
```

---

## Mid/Side処理

**高度な技術:**

### 概念

```
Mid:

定義:
L + R
中央成分
Mono

Side:

定義:
L - R
側面成分
Stereo情報

Mid/Side EQ:

機能:
Mid・Side別にEQ

用途:

Mid:
存在感調整
Vocal・Lead

Side:
広がり調整
Pad・FX

Ableton:

EQ Eight:
Mode: Mid/Side選択可
```

### 実践例

```
Pad (広げる):

EQ Eight:
Mode: Mid/Side

Mid:
通常通り

Side:
High Shelf +2 dB @ 5 kHz
広がり強調

Vocal (明瞭に):

Mid:
Peak +3 dB @ 3 kHz
明瞭度

Side:
通常通り

Master (低域Mono):

Side:
Low Cut 120 Hz
低域Mono化
```

### Mid/Side処理の詳細テクニック

Mid/Side処理はステレオイメージングの最も強力なツールの一つです。正しく使うことで、ステレオ幅、明瞭度、奥行きを精密にコントロールできます。

```
Mid/Side EQの高度な活用:

マスターバスでのM/S EQ:

  Mid チャンネル:
    Low Shelf +1-2 dB @ 80 Hz: ベースの存在感強化
    Peak +1-2 dB @ 3 kHz: ボーカル/リードの明瞭度
    High Shelf -1 dB @ 10 kHz: Sibilance抑制

  Side チャンネル:
    High Pass 120 Hz: 低域Mono化（必須）
    Peak +2-3 dB @ 8-12 kHz: Air/Shimmer追加
    Peak -2 dB @ 300-500 Hz: Mud除去

  効果:
    中央の要素がクリアに
    サイドの要素がきらびやかに
    低域が安定

個別トラックでのM/S処理:

  Synth Pad:
    Side: High Shelf +3 dB @ 5 kHz
    Side: Peak +2 dB @ 10 kHz
    → 高域の広がりが増大、空間感向上

  Drum Bus:
    Mid: Peak +2 dB @ 100 Hz（Kickの存在感）
    Side: Peak +2 dB @ 8 kHz（シンバルの広がり）
    → Kickはタイトに、シンバルはWideに

  Vocal Bus:
    Mid: Peak +2 dB @ 3 kHz（明瞭度）
    Mid: Peak +1 dB @ 200 Hz（温かみ）
    Side: High Shelf -3 dB @ 5 kHz（リバーブのサイド成分抑制）
    → ボーカルが中央でクリアに際立つ

Mid/Side コンプレッション:

  概念:
    Mid と Side に別々のコンプレッサーを適用
    ダイナミクスレベルでステレオ幅をコントロール

  実装（Ableton）:
    Audio Effect Rack → Chain でM/S分離
    各ChainにCompressor
    Mid: Ratio 2:1、Threshold -12 dB
    Side: Ratio 3:1、Threshold -18 dB

  効果:
    Midのダイナミクスが安定 → センター要素の一貫性
    Sideのダイナミクスが制御 → ステレオ幅の一貫性

  注意:
    過度な圧縮はステレオイメージを破壊
    微妙な設定が鍵（1-3 dB のGR程度）
```

---

## ステレオ幅測定

**視覚的確認:**

### 測定方法

```
Goniometer:

表示:
L/R相関

形:

縦線:
Mono

丸:
Wide Stereo

横線:
位相問題

推奨:
やや縦長の楕円

Correlation Meter:

値:

+1.0: 完全Mono
+0.5-0.8: 良い
0.0: Wide
-1.0: 位相反転 (問題)

目標:
+0.4-0.7

Ableton:

なし (標準)

無料:
Youlean Loudness Meter
Correlation表示あり

有料:
iZotope Ozone Imager
```

### ステレオ幅測定の詳細ガイド

```
Goniometer（Lissajous表示）の詳細な読み方:

表示パターンと意味:

  縦線（｜形）:
    完全Mono信号
    L = R
    Correlation = +1.0
    例: モノラルシンセ、センターボーカル

  やや太い縦線:
    ほぼMono、微妙なステレオ情報
    Correlation = +0.8-0.9
    例: 軽いリバーブがかかったボーカル

  縦長の楕円:
    適度なステレオ幅
    Correlation = +0.5-0.7
    例: 良いミックスの典型的な表示
    → 目標範囲

  円形:
    Wide Stereo
    Correlation = 0.0付近
    例: ステレオパッド、ワイドなリバーブ
    → やや広すぎる可能性

  横長の楕円:
    Sideが過大
    Correlation = マイナス値
    例: 位相問題あり
    → 修正が必要

  横線（ー形）:
    完全逆位相
    L = -R
    Correlation = -1.0
    → 深刻な位相問題、即座に修正

Vectorscope の活用:

  Vectorscope vs Goniometer:
    基本的に同じ原理
    表示方法が若干異なる
    どちらも L/R 相関を視覚化

  読み方のコツ:
    表示の「重心」がセンターにあるか確認
    左右の偏りがないか確認
    急な位相変化がないか確認（点滅的な動き）

Stereo Width Meter:

  LUFS メーターとの併用:
    Loudness と Stereo Width を同時に確認
    ラウドネスが高い部分のステレオ幅に注目
    ドロップ部分の Width が適切か確認

  推奨プラグイン:
    Youlean Loudness Meter 2（無料）: Correlation表示あり
    iZotope Insight 2: 包括的なメータリング
    NUGEN Visualizer: 詳細なステレオ分析
    Voxengo SPAN（無料）: スペクトラム + ステレオ
    Flux:: Stereo Tool: 精密なステレオ測定
```

---

## Mono互換性確認

**必須チェック:**

### 確認方法

```
方法1: Utility

Master:

Utility挿入
Width: 0%
Mono化

確認:

再生:
各トラック聴こえる？

問題:
消える・音量down

解決:
Bass Mono: On (120 Hz)

方法2: Headphone片耳

簡易:
片耳で聴く

確認:
バランス良い？

方法3: 実機

スマホ:
1スピーカー = Mono

車:
Mono的環境

確認:
実際の再生環境
```

### Mono互換性チェックの体系的アプローチ

```
チェックリスト（ミックス完了前に必ず実施）:

Step 1: 全体のMonoチェック
  Master に Utility を挿入
  Width を 0% に設定
  楽曲全体を再生
  確認ポイント:
    □ キックが消えていないか
    □ ベースが消えていないか
    □ ボーカル/リードが消えていないか
    □ 全体の音量が大幅に下がっていないか
    □ 特定の周波数帯域が消えていないか

Step 2: セクション別チェック
  イントロ → ビルドアップ → ドロップ → ブレイク
  各セクションでMono再生を確認
  セクション間の音量差が適切か

Step 3: トラック個別チェック
  Soloで各トラックをMono確認
  問題のあるトラックを特定
  必要に応じてWidth調整やBass Mono設定を見直し

Step 4: A/B比較
  Stereo再生とMono再生を交互に切り替え
  音量差が3 dB以内が理想
  6 dB以上の差がある場合は問題あり

許容範囲の目安:

  Stereo → Mono での音量変化:
    0-2 dB低下: 優秀（Mono互換性が高い）
    2-4 dB低下: 良好（一般的なレベル）
    4-6 dB低下: 要注意（一部修正推奨）
    6 dB以上低下: 問題あり（修正必須）

  各要素の確認基準:
    Kick: Mono再生で音量変化なし（0 dB）
    Bass: Mono再生で音量変化なし（0 dB）
    Vocal: -1 dB以内
    Lead: -2 dB以内
    Pad: -3 dB以内（多少の低下は許容）
    FX: -4 dB以内（装飾的要素は多少の変化OK）
```

---

## トラック別設定

**推奨値:**

### 詳細設定

```
Kick:

Width: 0% (完全Mono)
Pan: Center
Bass Mono: On (全帯域)

理由:
最もパワフル
中央定位

Bass:

Width: 0%
Pan: Center
Bass Mono: On

Snare/Clap:

Width: 0-10% (ほぼMono)
Pan: Center

Vocal:

Width: 0-10%
Pan: Center
存在感

Lead:

Width: 20-30%
Pan: Center
やや広がり

Pad:

Width: 80-100%
Pan: Center
広い

Hi-Hat:

Width: 40-60%
Pan: L/R Auto Pan
または
固定 L -30% / R +30%

Percussion:

Width: 60-80%
Pan: L/R振り分け

FX:

Width: 100-120%
Pan: L/R極端
```

---

## Haas Effect

**擬似ステレオ:**

### 原理

```
定義:

Mono信号:
L/R微妙に遅延差

結果:
Stereo感

設定:

Delay: 10-30 ms
片側のみ

効果:
広がり

注意:

Mono再生:
Comb Filter
音質劣化

推奨:
装飾のみ使用
Kick・Bass・Vocal不可
```

### Haas Effectの詳細な活用法

Haas Effect（先行音効果）は、人間の聴覚システムが最初に到達する音の方向を音源の位置として認識する現象を利用したテクニックです。正しく使えば非常に効果的なステレオ拡張が可能ですが、誤用すると深刻な位相問題を引き起こします。

```
Haas Effectの科学的背景:

先行音効果（Precedence Effect）:
  人間の脳は最初に到達した音の方向を「音源の位置」と認識
  1-30 msの遅延範囲で効果が発生
  30 ms以上ではエコーとして知覚される

遅延時間と知覚の関係:
  0-1 ms: ほとんど変化なし（位相的な変化のみ）
  1-5 ms: 微妙な広がり、音像の「太さ」が増す
  5-15 ms: 明確なステレオ広がり（最も効果的な範囲）
  15-30 ms: 広いステレオ感だがエコー感が出始める
  30-50 ms: ダブリング/スラップバック効果
  50 ms以上: 明確なエコー

コームフィルター効果:
  遅延信号がオリジナルと重なると干渉パターンが発生
  特定の周波数がブースト/カットされる
  Mono再生時に最も顕著（L+Rでコームフィルターが発生）
  → 金属的な音質変化、薄い音になる

Abletonでの実装方法:

方法1: Simple Delay を使用
  Simple Delay を挿入
  Link: Off（L/R独立設定）
  Left: 0 ms（または通常の値）
  Right: 10-20 ms（遅延を付加）
  Dry/Wet: 100%
  Feedback: 0%

方法2: Audio Effect Rack で精密制御
  Audio Effect Rack を作成
  Chain 1: Original（処理なし）Pan Center
  Chain 2: Simple Delay（10-20 ms）Pan Right
  Chain 2の音量を -3 dB程度に設定
  → より自然なHaas Effect

方法3: Utility + Delay の組み合わせ
  トラックを複製
  複製トラック: Delay 10-20 ms 追加
  複製トラック: Pan を反対側に設定
  両トラック: Utility > Bass Mono On（120 Hz）
  → 低域の安全性を確保しつつHaas Effect

推奨パラメーター（要素別）:

Hi-Hat:
  Delay: 5-10 ms
  効果: 微妙な広がり
  注意: 過度にするとグルーヴが崩れる

Synth Stab:
  Delay: 10-15 ms
  効果: パンチを保ちつつ広がり
  注意: アタックの明瞭度を確認

Pad:
  Delay: 15-25 ms
  効果: 広大な空間感
  注意: Mono互換性の低下に注意

Vocal（装飾的）:
  Delay: 8-12 ms
  効果: 存在感の増大
  注意: メインボーカルには非推奨、バックボーカルに適用

Mono互換性の確保策:

必須対策:
  1. Bass Mono: On（120 Hz）を必ず併用
  2. 遅延信号の音量を -2〜-6 dB に設定
  3. 遅延信号にハイパスフィルター（200-300 Hz）を適用
  4. Mono再生で必ず確認

応用テクニック:
  遅延信号にEQを適用:
    High Pass 200 Hz（低域のコームフィルター回避）
    High Shelf -3 dB @ 10 kHz（高域の金属感抑制）

  遅延時間のオートメーション:
    ビルドアップで遅延時間を増加 → 広がりの増大
    ドロップで遅延をゼロに → Monoのインパクト
```

---

## コーラス/ディチューンによるワイドニング

**シンセサイザーとエフェクトによるステレオ拡張:**

コーラスやディチューン（微細なピッチ変化）は、ステレオ幅を広げるための最も安全で効果的な手法の一つです。Haas Effectよりも位相問題が少なく、自然な広がりを作ることができます。

### コーラスエフェクトの原理と活用

```
コーラスの基本原理:

信号の流れ:
  入力信号 → LFOで変調されたディレイライン → 原音とミックス
  L: 原音 + LFO変調ディレイ（位相A）
  R: 原音 + LFO変調ディレイ（位相B）
  → L/Rの微妙な差異がステレオ感を生む

パラメーター:
  Rate: LFOの速度（0.1-5 Hz）
  Depth: 変調の深さ（ディレイ時間の変動幅）
  Delay: ベースとなるディレイ時間（5-30 ms）
  Feedback: フィードバック量
  Dry/Wet: ミックスバランス

Ableton Chorus-Ensemble:

設定ガイド:
  Mode: Chorus（Classic / Ensemble / Vibrato）

  Pad向け設定:
    Mode: Ensemble
    Rate: 0.3-0.8 Hz
    Depth: 40-60%
    Dry/Wet: 30-50%
    → 豊かで広がりのあるパッドサウンド

  Lead向け設定:
    Mode: Classic
    Rate: 1-2 Hz
    Depth: 20-30%
    Dry/Wet: 20-30%
    → 微妙な広がりと動き

  FX向け設定:
    Mode: Ensemble
    Rate: 0.5-1.5 Hz
    Depth: 60-80%
    Dry/Wet: 50-70%
    → 明確なステレオ広がり

注意事項:
  Kick/Bassには使用しない（低域の揺れが問題）
  過度なDepthは音程の揺れとして知覚される
  Mono互換性の確認は必須
```

### ディチューンによるステレオ拡張

```
ディチューン（Detune）の原理:

概念:
  同じ音を微妙にピッチを変えて複製
  L/Rに異なるピッチ変化を適用
  → ユニゾン効果 + ステレオ広がり

ピッチ変化量と効果:
  1-5 cent: 微妙な太さの追加、自然な広がり
  5-15 cent: 明確なデチューン感、SuperSawのような効果
  15-30 cent: 強いコーラス効果、激しいデチューン
  30 cent以上: 不協和音に近づく、特殊効果

シンセサイザーでの実装:

Serum:
  Oscillator Detune:
    UnisonVoices: 4-8
    Detune: 10-25%
    Blend: 調整
    → SuperSaw系サウンドの基本

  UnisonVoicesの影響:
    2 Voice: シンプルな広がり
    4 Voice: バランスの良い広がり
    8 Voice: 非常に厚いサウンド
    16 Voice: 壁のようなサウンド（CPU負荷注意）

Massive X:
  Voicing > Unison:
    Voices: 2-8
    Detune: 適度に設定
    Spread: ステレオ幅

Vital（無料シンセ）:
  Unison設定:
    Voices: 2-16
    Detune: 0.1-0.5
    Spread: ステレオ幅

Abletonプラグインでの実装:

Shifter（ピッチシフター）:
  Audio Effect Rack を作成
  Chain 1: Shifter +5 cent、Pan Left
  Chain 2: Shifter -5 cent、Pan Right
  Chain 3: Original（ドライ）、Pan Center
  → 3ボイスのデチューンステレオ効果

Soundtoys MicroShift:
  業界標準のデチューンプラグイン
  Style 1: 微妙なデチューン（ナチュラル）
  Style 2: 中程度のデチューン（バランス良好）
  Style 3: 強めのデチューン（80年代的）
  Mix: 30-60%
  Delay: 微調整
  → 非常に自然なステレオ広がり
```

### Unison/SuperSawサウンドのステレオ管理

```
SuperSawのステレオ処理:

問題点:
  ユニゾンボイスが多い → 自然にWide
  低域もWideになりがち → Mono互換性問題
  Mono再生で音が薄くなる

対策:
  1. Bass Mono: On（120 Hz）は必須
  2. ユニゾンボイスの低域をMono化
  3. Mid/Side EQで低域Sideをカット

推奨ワークフロー:
  1. シンセでUnison/Detuneを設定
  2. Utility > Bass Mono: On（120 Hz）
  3. EQ Eight（M/S Mode）> Side: HP 150 Hz
  4. Mono確認（Width 0%）
  5. 必要に応じて全体のWidthを微調整

レイヤリングとステレオ:
  低域レイヤー: Mono（Width 0%）
  中域レイヤー: やや広げる（Width 40-60%）
  高域レイヤー: Wide（Width 80-100%）
  → 周波数帯域ごとに適切なステレオ幅
```

---

## 周波数帯域別ステレオ処理

**マルチバンドアプローチによる精密なステレオコントロール:**

周波数帯域ごとに異なるステレオ幅を設定することで、低域の安定性と高域の広がりを同時に実現できます。これはプロフェッショナルなミックスとマスタリングで欠かせないテクニックです。

### マルチバンドステレオ処理の概念

```
なぜマルチバンドが必要か:

単一のWidth設定の限界:
  Width 0%: 全帯域Mono → 高域の広がりも失われる
  Width 100%: 全帯域Stereo → 低域に位相問題
  Width 120%: 全帯域Wide → 低域が不安定

マルチバンドの利点:
  低域: Mono（安定性確保）
  中域: 適度なStereo（明瞭度と空間感のバランス）
  高域: Wide（きらびやかさと広がり）
  → 各帯域に最適なステレオ幅を個別設定

実装方法（Ableton）:

方法1: Audio Effect Rack + マルチバンドスプリット

  Audio Effect Rack を作成
  3つのChainを作成:

  Chain 1: Low（~200 Hz）
    EQ Eight: Low Pass 200 Hz
    Utility: Width 0%（Mono）
    → 低域の安全性

  Chain 2: Mid（200-5000 Hz）
    EQ Eight: Band Pass 200-5000 Hz
    Utility: Width 60%
    → 適度な広がり

  Chain 3: High（5000 Hz~）
    EQ Eight: High Pass 5000 Hz
    Utility: Width 100-120%
    → 高域の広がり

方法2: Multiband Dynamics を応用

  Multiband Dynamics を挿入
  各バンドのOutput Gain で間接的に制御
  Mid/Side モードと併用

方法3: サードパーティプラグイン

  iZotope Ozone Imager:
    4バンドのステレオ幅コントロール
    視覚的なステレオフィールド表示
    バンドごとの独立したWidth設定

  Cableguys ShaperBox（Widthモジュール）:
    テンポ同期のステレオ幅変化
    LFOによる動的ステレオ制御

  Polyverse Wider（無料）:
    シンプルなステレオ拡張
    Mono互換性が比較的良好
```

### 帯域別ステレオ処理の実践ガイド

```
Sub Bass（20-60 Hz）:
  Width: 0%（完全Mono）
  処理: なし（そのまま）
  理由:
    方向感が全くない周波数帯域
    位相問題のリスクが最も高い
    Mono化しても聴感上の変化なし

Bass（60-200 Hz）:
  Width: 0-10%
  処理: Utility Bass Mono で確実にMono化
  理由:
    基本的に方向感なし
    クラブのサブウーファーはMono再生
    わずかなステレオ成分も問題の原因に

Low Mid（200-500 Hz）:
  Width: 20-40%
  処理: 控えめなステレオ幅
  理由:
    ボディ感に関わる帯域
    過度なWidthはミックスの「重心」を失う
    ルームレゾナンスが問題になりやすい

Mid（500-2000 Hz）:
  Width: 30-60%
  処理: 中程度のステレオ幅
  理由:
    ボーカル/リードの主要帯域
    明瞭度とスペース感のバランスが重要
    この帯域のWidth処理がミックスの印象を左右

Upper Mid（2000-6000 Hz）:
  Width: 50-80%
  処理: やや広めのステレオ幅
  理由:
    プレゼンス帯域
    広がりの効果が明確に感じられる
    ボーカルの明瞭度を保ちつつ空間感

High（6000-12000 Hz）:
  Width: 70-110%
  処理: 広いステレオ幅
  理由:
    エア感、ブリリアンス
    ステレオ効果が最も効果的な帯域
    シンバル、ハイハットの広がり

Air（12000-20000 Hz）:
  Width: 80-120%
  処理: 最も広いステレオ幅
  理由:
    空気感、きらびやかさ
    超高域はMono互換性への影響が少ない
    広げることで「空間の開放感」が増大
```

---

## よくある失敗

### 1. 低域Stereo

```
問題:
Mono再生で消える

原因:
120 Hz以下Stereo

解決:

全トラック:
Bass Mono: On (120 Hz)

確認:
Mono再生テスト

必須:
100%のプロジェクト
```

### 2. Width過剰

```
問題:
位相問題
不自然

原因:
Width 150%+

解決:

最大:
Width 120%

推奨:
80-100% (Pad・FX)

理由:
自然な広がり
```

### 3. Mono確認なし

```
問題:
Mono再生でバランス崩壊

原因:
確認忘れ

解決:

必須:
Mono確認

頻度:
ミックス完了前

方法:
Utility Width 0%
```

### 4. 全てWide

```
問題:
中央スカスカ
パワーなし

原因:
全トラックWide

解決:

Center:
Kick・Bass・Vocal・Lead

Wide:
Pad・FX・装飾のみ

バランス:
Center 60%
Wide 40%
```

### 5. ステレオエンハンサーの多用

```
問題:
  不自然な広がり
  位相が不安定
  Mono再生で大幅な音量低下
  ヘッドホンとスピーカーで全く異なる印象

原因:
  複数のステレオエンハンサーを重ねがけ
  過度なSide成分の増幅
  低域にもステレオエンハンスを適用

解決:
  ステレオエンハンサーは1つのバスに1つまで
  低域は必ず除外（Bass Mono併用）
  微妙な設定から始める（+10-20%程度）
  A/B比較で常に効果を確認
  Mono再生テストを忘れずに
```

### 6. パンニングの左右非対称

```
問題:
  ステレオイメージが片側に偏る
  バランスの悪いミックス
  ヘッドホン再生で違和感

原因:
  左側にパンした要素が多い
  右側の要素が少ない/音量が小さい
  パンニングのバランスを考慮していない

解決:
  左右のエネルギーバランスを確認
  Goniometerで偏りをチェック
  対称的なパン配置を心がける:
    L -40% のPercussion → R +40% のPercussionとペアに
    偏りがある場合は反対側にも要素を配置
  最終確認: Correlation Meterが左右均等か
```

---

## ステレオエンハンサー

**プラグイン活用:**

### 種類

```
1. Stereo Width:

機能:
Mid/Side調整

推奨:
iZotope Ozone Imager (無料)

2. Microshift:

機能:
わずかなDetune
Stereo感

推奨:
Soundtoys MicroShift

3. Doubler:

機能:
音声複製
Stereo配置

推奨:
Waves Doubler

注意:

使いすぎ:
不自然

推奨:
Ableton標準で十分
Utility + Chorus
```

---

## 実践ワークフロー

**30分で完成:**

### Step-by-Step

```
0-10分: Bass Mono設定

1. 全トラック Utility挿入
2. Bass Mono: On
3. Freq: 120 Hz
4. 確認

10-15分: Width設定

1. Kick・Bass: 0%
2. Snare: 0-10%
3. Lead: 20-30%
4. Pad: 80-100%
5. FX: 100-120%

15-20分: Panning

1. Hi-Hat: L/R
2. Percussion: L/R
3. FX: L/R極端
4. バランス確認

20-25分: 確認

1. Stereo再生
2. Mono再生 (Width 0%)
3. 比較

25-30分: 微調整

各設定微調整
完成
```

---

## ステレオイメージャープラグイン比較

**主要プラグインの特徴と使い分け:**

ステレオイメージングに特化したプラグインは数多くありますが、それぞれに特徴があります。目的に応じて適切なツールを選択することが重要です。

### 無料プラグイン

```
iZotope Ozone Imager（無料版）:
  特徴:
    直感的なUI
    ステレオフィールドの視覚表示
    Width コントロール
    Stereoize モード（Mono信号をStereo化）

  メリット:
    無料で高品質
    視覚的なフィードバック
    初心者にも使いやすい

  デメリット:
    マルチバンドは有料版のみ
    細かい設定は不可

  推奨用途:
    個別トラックのWidth調整
    ステレオフィールドの視覚確認
    初心者の学習用

Polyverse Wider（無料）:
  特徴:
    シンプルな1ノブ操作
    位相問題が比較的少ない独自アルゴリズム
    Mono互換性が良好

  メリット:
    操作が極めて簡単
    CPU負荷が低い
    Mono再生でも比較的安全

  デメリット:
    細かいコントロールがない
    マルチバンド非対応

  推奨用途:
    手軽なステレオ拡張
    Mono互換性を重視する場面

Voxengo MSED（無料）:
  特徴:
    Mid/Side エンコード/デコード
    Mid と Side の個別音量調整
    インラインモードとサイドチェーンモード

  メリット:
    Mid/Side処理の基本を学べる
    他のプラグインと組み合わせて使用可能

  推奨用途:
    Mid/Side処理の学習
    他プラグインとのM/S チェーン構築

A1StereoControl（無料）:
  特徴:
    Width コントロール
    Pan コントロール
    Correlation メーター
    Safe Bass（低域Mono化）

  メリット:
    オールインワンの基本機能
    Correlation確認が便利

  推奨用途:
    基本的なステレオ管理
    Correlation の日常的な確認
```

### 有料プラグイン

```
iZotope Ozone Imager（有料版 / Ozone内蔵）:
  特徴:
    4バンドのマルチバンドステレオ処理
    各バンド独立のWidth設定
    高精度なステレオフィールド表示
    Stereoize機能（位相安全なステレオ化）

  価格帯: Ozone Standard/Advanced に含まれる
  推奨用途: マスタリング、バス処理

Soundtoys MicroShift:
  特徴:
    3つのスタイル（異なるデチューンアルゴリズム）
    非常に自然なステレオ拡張
    ビンテージハードウェアのエミュレーション

  価格帯: 単体購入可
  推奨用途: ボーカル、シンセ、ギターのステレオ拡張

Waves S1 Stereo Imager:
  特徴:
    Width / Rotation / Asymmetry
    直感的な操作
    低CPU負荷

  価格帯: 比較的安価
  推奨用途: 基本的なWidth調整

bx_stereomaker（Brainworx）:
  特徴:
    Mono信号を自然にStereo化
    周波数帯域別のステレオ処理
    Bass Mono機能内蔵

  推奨用途: モノラル素材のステレオ化
```

---

## ジャンル別ステレオ戦略

**ジャンルの特性に合わせたステレオイメージング:**

音楽ジャンルによって求められるステレオイメージは大きく異なります。クラブミュージックはMono互換性が最優先、アンビエントは広大な空間感が重要というように、ジャンルの特性を理解した上でステレオ処理を行うことが重要です。

```
テクノ:
  特徴:
    ミニマルな構成、キックとベースが主役
    クラブ再生が前提

  ステレオ戦略:
    Kick: 完全Mono（Width 0%）
    Bass: 完全Mono（Width 0%）
    Hi-Hat: 軽くWidthを付ける（30-50%）
    Pad/Atmosphere: Wide（80-100%）
    FX: 極端にWide（100-120%）

  Bass Mono: 120-150 Hz
  全体のCorrelation: +0.5-0.7（Mono寄り）

  ポイント:
    低域のパワーとMono互換性を最優先
    装飾的要素でのみステレオ感を演出
    ビルドアップ→ドロップのWidth変化が効果的

ハウス:
  特徴:
    グルーヴィーなリズム、ボーカルサンプル多用
    クラブ再生が主だがヘッドホンリスナーも多い

  ステレオ戦略:
    Kick: 完全Mono
    Bass: 完全Mono
    Vocal: センター定位、微妙なWidth（0-15%）
    Piano/Chord: やや広め（50-70%）
    Strings/Pad: Wide（80-100%）
    Percussion: L/R に振り分け

  Bass Mono: 120 Hz
  全体のCorrelation: +0.4-0.6

トランス:
  特徴:
    エモーショナルなメロディ、壮大な展開
    SuperSawが多用される

  ステレオ戦略:
    Kick: 完全Mono
    Bass: 完全Mono
    SuperSaw Lead: Wide（70-90%）だがBass Mono必須
    Pad: Very Wide（90-110%）
    FX Sweep: 極端にWide
    Pluck: やや広め（40-60%）

  Bass Mono: 100-120 Hz
  全体のCorrelation: +0.3-0.5（やや広め許容）

  ポイント:
    ビルドアップでMono化 → ドロップで一気にWide
    SuperSawの低域は必ずMono化
    壮大さの演出にステレオ幅を積極活用

ドラムンベース:
  特徴:
    高速BPM（170-180）、複雑なベースライン
    Reese Bassのステレオ動きが重要

  ステレオ戦略:
    Kick: 完全Mono
    Sub Bass: 完全Mono
    Reese Bass（中高域成分）: Width 40-60%
    Snare: ほぼMono（0-10%）
    Break Beat: やや広め（30-50%）
    Pad/Atmosphere: Wide（80-100%）

  Bass Mono: 80-100 Hz（Reese Bassの特性に合わせて低め）
  全体のCorrelation: +0.4-0.6

アンビエント/チルアウト:
  特徴:
    広大な空間感が最重要
    ヘッドホン再生が多い

  ステレオ戦略:
    Pad: Very Wide（100-120%）
    Texture: Wide（80-110%）
    Drone: Wide（90-110%）
    Melodic Element: やや広め（50-70%）
    Sub Bass（ある場合）: Mono

  Bass Mono: 150-200 Hz（より広いMono帯域で安定）
  全体のCorrelation: +0.2-0.4（Wide許容）

  ポイント:
    空間感を最大限に活用
    ただしMono互換性も一定レベル確保
    リバーブのステレオ幅が重要な要素

ヒップホップ/トラップ:
  特徴:
    808ベースが主役、ボーカルが中心
    スマホ/イヤホン再生が非常に多い

  ステレオ戦略:
    808 Bass: 完全Mono
    Kick: 完全Mono
    Hi-Hat: L/R にパン（30-50%）
    Vocal: センター定位（Width 0-10%）
    Ad-lib: L/Rにパン
    Melody/Sample: やや広め（40-60%）
    FX: Wide（80-100%）

  Bass Mono: 100-120 Hz
  全体のCorrelation: +0.5-0.7

  ポイント:
    808のサブベースは絶対Mono
    ボーカルの存在感を最優先
    スマホ再生でもバランスが崩れないこと
```

---

## マスタリングでのステレオ処理

**マスタリング段階でのステレオイメージング最終調整:**

マスタリングは楽曲の最終仕上げの段階です。ここでのステレオ処理は微妙かつ慎重に行う必要があります。大幅な変更はミックスに戻って修正すべきですが、マスタリングでの微調整は楽曲のクオリティを確実に高めます。

```
マスタリングでのステレオ処理の原則:

基本ルール:
  1. 大幅な変更はしない（ミックスに戻る）
  2. 微調整のみ行う（+/- 10-15% 程度）
  3. 低域は必ずMono確認
  4. Mono互換性を最終確認

マスタリングチェーンでのステレオ処理の位置:
  1. EQ（整形）
  2. コンプレッション
  3. ステレオイメージング ← ここ
  4. リミッター
  ※ リミッターの前に配置するのが一般的

マスタリングでの具体的処理:

Mid/Side EQ:
  Side: High Pass 120 Hz（低域Mono化の最終確認）
  Side: High Shelf +1-2 dB @ 10 kHz（Air追加）
  Mid: Peak +0.5-1 dB @ 3 kHz（ボーカル明瞭度）

マルチバンドステレオ処理:
  Low（~200 Hz）: Width 0%（Mono）
  Low-Mid（200-2000 Hz）: Width 95-100%（微調整程度）
  High-Mid（2000-8000 Hz）: Width 100-105%（微拡張）
  High（8000 Hz~）: Width 105-115%（Air拡張）

注意:
  マスタリングでの過度なWidth拡張は避ける
  +/- 5-10% の微調整が目安
  必ずBefore/After で比較確認
  異なる再生環境でのチェックを忘れずに

ステレオ処理後の最終チェック:
  □ Mono再生で問題ないか
  □ Correlation が +0.3 以上か
  □ 左右のバランスが均等か
  □ ヘッドホンとスピーカーの両方で確認
  □ 低音量でもステレオバランスが保たれるか
```

---

## 実践テクニック集

**すぐに使えるステレオイメージングのTips:**

### テクニック1: Mono → Stereo トランジション

```
効果:
  イントロやブレイクでMono → ドロップでStereo
  知覚的なインパクトが増大

実装:
  Master Bus の Utility に Width オートメーション
  ビルドアップ: Width 30-40%（Mono寄り）
  ドロップの瞬間: Width 100-110%
  → 「広がった！」という解放感

応用:
  逆パターン（Stereo → Mono → Stereo）も効果的
  フィルター開閉と組み合わせると相乗効果
```

### テクニック2: リバーブのステレオ配置

```
原理:
  ドライ信号: Center（Mono）
  リバーブ: Wide（Stereo）
  → 中央の明瞭度を保ちつつ空間感

実装:
  センドリターンでリバーブを使用
  リバーブリターントラックの Width: 100-120%
  ドライ信号のトラック: Width 0-20%
  → 分離感のある空間表現

ポイント:
  Pre-Delay を50-100 ms に設定
  → ドライ信号とリバーブの時間的分離
  → センター要素の明瞭度がさらに向上
```

### テクニック3: レイヤーごとのステレオ管理

```
概念:
  同じ要素の周波数レイヤーごとにWidth を変える

実装例（SuperSaw）:
  Sub レイヤー（〜100 Hz）: Width 0%
  Body レイヤー（100-2000 Hz）: Width 50%
  Bright レイヤー（2000 Hz〜）: Width 100%
  → 低域はタイト、高域はWide

効果:
  Mono互換性を確保しつつ最大限の広がり
  各帯域の役割が明確に
```

### テクニック4: ダブルトラッキング

```
原理:
  同じフレーズを2回演奏/録音
  左右に配置
  → 自然なステレオ広がり

シンセでの実装:
  同じMIDIを2つのトラックにルーティング
  それぞれ微妙にパラメーターを変える:
    オシレーターの波形を微妙に変更
    フィルターのカットオフを微妙にずらす
    LFOのRateを微妙に異なる値に
  Track 1: Pan Left 50%
  Track 2: Pan Right 50%

ボーカルでの実装:
  同じパートを2テイク録音
  それぞれL/Rにパン
  人間の微妙な演奏差がナチュラルなステレオ感を生む
```

### テクニック5: FXによるステレオ演出

```
Ping-Pong Delay:
  テンポ同期のL/R交互ディレイ
  FXリターンに配置
  Dry/Wet: 20-40%
  → リズミカルなステレオ動き

Stereo Phaser/Flanger:
  L/R のLFO位相をオフセット
  Phase: 180度
  → 回転するようなステレオ効果

Auto Filter（ステレオモード）:
  L/R で異なるフィルターカーブ
  Phase offset: 90-180度
  → 動的なステレオフィルター効果
```

---

## まとめ

### Stereo Imaging

```
□ 120 Hz以下 必ずMono
□ Kick・Bass Width 0%
□ Pad・FX Width 80-120%
□ Mono互換性確認必須
□ Bass Mono: On (全トラック)
```

### Panning

```
Center: Kick・Bass・Vocal・Lead
L/R: Hi-Hat・Percussion・FX
バランス: Center 60%, Wide 40%
```

### 重要原則

```
□ 低域ほどMono
□ 高域ほどWide
□ Width過剰注意 (最大120%)
□ Mono確認必須
□ Correlation +0.4-0.7
```

### ステレオ処理の黄金ルール

```
1. 低域は必ずMono化（120 Hz以下）
2. 重要な要素（Kick, Bass, Vocal, Lead）はCenter
3. 装飾的要素（Pad, FX, Percussion）でステレオ感を演出
4. マルチバンドで帯域別に最適なWidth設定
5. Mono互換性チェックは全プロジェクトで必須
6. Goniometer/Correlation Meterで視覚的に確認
7. 複数の再生環境でテスト（ヘッドホン、スピーカー、スマホ）
8. ジャンルの特性に合わせたステレオ戦略を選択
9. マスタリングでの変更は微調整のみ
10. 常にBefore/After で比較確認する習慣を
```

---

**次は:** [Depth & Space](./depth-space.md) - Reverb・Delayで奥行きを作る
