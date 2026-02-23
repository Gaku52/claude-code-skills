# DJ用トラック制作

**DJセット用のオリジナルトラックとツールを作成する**

DJ用トラックの制作は、通常の楽曲制作とは異なる特別な配慮が必要です。32小節のIntro/Outro、ミックスポイントの設計、DJフレンドリーな構成など、実際のDJセットで使用することを前提とした制作テクニックを学びます。本ドキュメントでは、DJ用トラック制作の全工程を網羅的に解説し、初心者から上級者まで実践的に活用できるガイドを提供します。

---

## 目次

1. [この章で学ぶこと](#この章で学ぶこと)
2. [DJ用トラックの特徴](#dj用トラックの特徴)
3. [Intro/Outroの設計](#introoutroの設計)
4. [ミックスポイントの設計](#ミックスポイントの設計)
5. [ジャンル別トラック構成](#ジャンル別トラック構成)
6. [キックドラムの設計](#キックドラムの設計)
7. [ベースラインの設計](#ベースラインの設計)
8. [アレンジメント構成の詳細](#アレンジメント構成の詳細)
9. [アカペラ・インストの分離](#アカペラインストの分離)
10. [DJ用ループ・フィルイン](#dj用ループフィルイン)
11. [トランジションツール](#トランジションツール)
12. [ブートレグ・エディット](#ブートレグエディット)
13. [ミキシング・マスタリング（DJ用）](#ミキシングマスタリングdj用)
14. [キーとBPMの選択](#キーとbpmの選択)
15. [エクスポート設定](#エクスポート設定)
16. [メタデータの設定](#メタデータの設定)
17. [DJ用テンプレート作成](#dj用テンプレート作成)
18. [配布とプロモーション](#配布とプロモーション)
19. [よくある失敗と対処法](#よくある失敗と対処法)
20. [実践ワークフロー](#実践ワークフロー)
21. [まとめ](#まとめ)

---

## この章で学ぶこと

- DJ対応 Intro/Outroの作り方（32小節必須）
- ミックスポイントの設計
- アカペラ・インストゥルメンタルの分離
- DJ用ループ・フィルインの制作
- トランジションツールの作成
- ブートレグ・エディットの作り方
- ジャンル別のDJトラック設計パターン
- ミキシング・マスタリングのDJ特有の考慮事項
- メタデータとファイル管理の最適化
- DJ用テンプレートの効率的な構築

**学習時間**: 3-5時間
**難易度**: 中級

---

## DJ用トラックの特徴

### 通常のトラックとの違い

**通常のトラック（リスナー向け）**:
```
構成: Intro → Verse → Chorus → Verse → Chorus → Bridge → Chorus → Outro
Intro: 8-16小節（短い、すぐに楽曲の核心へ）
Outro: 4-8小節（フェードアウトが多い）
総尺: 3-4分（ストリーミング最適化）
目的: 単体で完結する聴取体験
特徴:
  - フックが早い段階で提示される
  - ボーカルが主役
  - コンパクトな構成
  - ラジオ/プレイリスト向け
```

**DJ用トラック（パフォーマンス向け）**:
```
構成: Intro → Buildup → Drop → Breakdown → Drop → Outro
Intro: 32小節（長い、ミックス専用の空間）
Outro: 32小節（次の曲へのスムーズな接続）
総尺: 5-8分（DJミックスに最適化）
目的: DJセットの中で他の曲と調和する
特徴:
  - ゆっくりと要素が追加される
  - 明確なミックスポイント
  - 正確なBPMグリッド
  - キックから始まりキックで終わる
  - エネルギーの起伏が大きい
```

### DJ用トラックの要件一覧

```
必須要件:
  1. 32小節以上のIntro（キックから開始）
  2. 32小節以上のOutro（キックで終了）
  3. 正確なBPMグリッド（グリッドのズレは致命的）
  4. 明確なキー（1つのスケール内で完結）
  5. クリーンな低域（他のトラックとの干渉回避）
  6. 適切なダイナミックレンジ
  7. 高品質なマスタリング（音圧は-1dB LUFS推奨）

推奨要件:
  1. インストゥルメンタルバージョンの用意
  2. アカペラバージョンの用意
  3. ステム（個別パーツ）の用意
  4. ループポイントの設計
  5. メタデータの正確な記入
  6. 複数のブレイクダウンポイント
```

### DJが求めるトラックの品質

```
音質面:
  - サンプルレート: 44.1kHz / 48kHz
  - ビット深度: 24bit（制作時）→ 16bit/24bit（配布）
  - フォーマット: WAV / AIFF（推奨）、FLAC（許容）
  - ラウドネス: -8 to -6 LUFS（クラブ向け）
  - ピーク: -1 dBTP（True Peak）

構成面:
  - 8小節単位の構成（フレーズ構造の明確さ）
  - 予測可能なアレンジメント（DJが先を読める）
  - エネルギーの起伏が明確
  - 低域のクリーンさ（ミックス時の干渉回避）

実用面:
  - ファイル名にBPMとキーを含める
  - ID3タグの正確な記入
  - BeatportやTraxsource向けの規格準拠
  - Rekordbox / Traktor / Seratoでの動作確認
```

---

## Intro/Outroの設計

### 32小節ルール

**なぜ32小節か**:
```
DJのミックス時間と小節数の関係:

最短ミックス: 16小節（約30秒 @ 128BPM）
  - カットミックスやスラムミックス
  - エネルギッシュなトランジション
  - 上級DJ向け

標準ミックス: 32小節（約1分 @ 128BPM）
  - ブレンドトランジション
  - EQスワップ
  - 最も一般的

長めのミックス: 64小節（約2分 @ 128BPM）
  - ディープなブレンド
  - プログレッシブスタイル
  - テック/ミニマル向け

32小節の内訳（4フレーズ）:
  フレーズ1 (Bar 1-8):
    - 前の曲がまだメインで流れている
    - 新曲のキックのみが重なり始める
    - DJはEQで低域を制御

  フレーズ2 (Bar 9-16):
    - 前の曲のフェードアウトが進む
    - 新曲にハイハットやパーカッションが加わる
    - リズムの移行が明確に

  フレーズ3 (Bar 17-24):
    - 前の曲はほぼフェードアウト完了
    - 新曲にベースラインが加わる
    - 低域の主導権が新曲に移行

  フレーズ4 (Bar 25-32):
    - 前の曲は完全にフェードアウト
    - 新曲にパッド、シンセ要素が追加
    - フルドロップへの準備完了
```

### Intro の要素追加順序

**Bar 1-8 (フレーズ1): キックのみ**
```
構成:
  Kick: 4つ打ち（毎拍）
  BPM: 一定（絶対にブレない）
  音量: Peak -6dB

設計のポイント:
  - キック以外の要素は入れない
  - DJが前の曲のキックとブレンドしやすい
  - 低域のクリーンさが最重要
  - テイルが短めのキックが好ましい
  - サイドチェイン用のキックと同じパターン

オプション:
  - 非常に小さいリバーブテイル（空間認識用）
  - ホワイトノイズの微かなレイヤー
  - ただし、これらはあくまでサブリミナルレベル
```

**Bar 9-16 (フレーズ2): キック + ハイハット/パーカッション**
```
追加要素:
  + Hi-Hat: 8分音符 or 16分音符
  + Ride: オプション
  + Percussion: シェイカー、タンバリン等

設計のポイント:
  - リズムのキャラクターが明確になるフレーズ
  - ハイハットのベロシティに変化をつける
  - グルーヴ感を確立する
  - まだ低域は追加しない（キックのみ）

バリエーション:
  パターンA（ストレート）: HH on every 8th
  パターンB（シャッフル）: HH with swing
  パターンC（オフビート）: HH on off-beats only
```

**Bar 17-24 (フレーズ3): + ベースライン**
```
追加要素:
  + Sub Bass: ルート音中心
  + Mid Bass: オプション（軽め）
  + Clap/Snare: オプション（2, 4拍目）

設計のポイント:
  - ベースラインは徐々に入る（フィルターオープン推奨）
  - 低域の帯域が埋まり始める
  - DJはここで前の曲の低域を完全にカットする
  - ベースのルート音は曲のキーを示す

テクニック:
  - ハイパスフィルターを Bar 17 から徐々に開く
  - Bar 17: HPF @ 200Hz → Bar 24: HPF @ 20Hz
  - 自然なベースの導入効果
```

**Bar 25-32 (フレーズ4): + パッド/シンセ/ボーカルティーザー**
```
追加要素:
  + Pad: コード進行の提示
  + Lead Teaser: メロディの断片
  + Vocal Snippet: ボーカルの一部（あれば）
  + FX: ライザー、スウィープ

設計のポイント:
  - フルドロップへの期待感を作る
  - メインのメロディやフックの「予告編」
  - エネルギーが上昇していく感覚
  - Bar 31-32 はライザー/FXでテンション最大
  - Bar 32 の最後にはスネアロールやリバースFX

バリエーション:
  控えめ: パッドのみ追加
  標準: パッド + リードティーザー
  派手: パッド + リード + ボーカル + FX
```

**Bar 33以降: メインセクション（ドロップ）**
```
全要素が同時に登場:
  - Full Drums（キック、スネア、ハイハット、パーカッション）
  - Full Bass（サブ + ミッド）
  - Full Synth（リード、パッド、アルペジオ）
  - Vocal（あれば）
  - FX（アクセント）

インパクトを最大化する方法:
  - Bar 32 の最後 1/4 拍で無音を作る（ギャップ）
  - Bar 33 の 1拍目にキック + ベース + 全要素
  - インパクトFX（ダウンリフター、サブドロップ）を重ねる
  - サイドチェインが効き始める
```

### Outro の要素削除順序

**Introの逆順で削除**:
```
メインセクション最終部:
  全要素が鳴っている状態

Bar 1-8 (Outro フレーズ1): パッド/リード削除
  - メロディ要素を外す
  - FXでフェードアウト効果
  - ボーカルの最後のフレーズ

Bar 9-16 (Outro フレーズ2): ベースライン削除
  - ローパスフィルターで徐々にカット
  - Bar 9: LPF @ 20kHz → Bar 16: LPF @ 200Hz
  - またはボリュームフェード

Bar 17-24 (Outro フレーズ3): ハイハット/パーカッション削除
  - リズム要素を簡素化
  - キックとミニマルなパーカッションのみ

Bar 25-32 (Outro フレーズ4): キックのみ
  - 最もミニマルな状態
  - 次の曲のIntroとブレンドしやすい
  - 最後のバーで自然にフェードアウト or カット

重要: Outroの最後の音が完全に消えること
  - リバーブテイルが残らない
  - ディレイのフィードバックが残らない
  - サブベースのリリースが残らない
```

### Introバリエーション

```
バリエーション1: ミニマルイントロ
  Bar 1-16: キックのみ
  Bar 17-32: キック + ハイハット
  Bar 33: ドロップ（一気に全要素）
  → テクノ、ミニマル向け

バリエーション2: グラデーションイントロ
  Bar 1-8: キック
  Bar 9-16: + ハイハット + パーカッション
  Bar 17-24: + ベース + クラップ
  Bar 25-32: + パッド + FX + ボーカルスニペット
  Bar 33: ドロップ
  → ハウス、プログレッシブ向け（最も一般的）

バリエーション3: アトモスフェリックイントロ
  Bar 1-16: アンビエントパッド + ノイズ
  Bar 17-24: + キック（フィルター付き）
  Bar 25-32: キック全開 + ベース導入
  Bar 33: ドロップ
  → アンビエントテクノ、ディープハウス向け

バリエーション4: ブレイクビーツイントロ
  Bar 1-8: ブレイクビーツパターン
  Bar 9-16: + ベース
  Bar 17-24: + シンセスタブ
  Bar 25-32: + ボーカル + FX
  Bar 33: 4つ打ちドロップ
  → ブレイクス、UKガレージ向け

バリエーション5: ビルドアップイントロ
  Bar 1-16: キック（ハーフタイム）
  Bar 17-24: キック（フルタイム）+ ハイハット
  Bar 25-28: + ベース + スネアロール開始
  Bar 29-32: ライザー + スネアロール（加速）
  Bar 33: ドロップ（最大インパクト）
  → EDM、ビッグルーム向け
```

---

## ミックスポイントの設計

### ループポイント

**Introのループポイント**:
```
DJがループできるセクションを意図的に設計:

ループポイント1: Bar 9-16 (キック+ハイハット)
  - 最も汎用的なループポイント
  - DJが好きなタイミングでミックス開始
  - Bar 9 と Bar 17 のサウンドが滑らかに接続

ループポイント2: Bar 1-8 (キックのみ)
  - 非常にクリーンなループ
  - 長いトランジションを取りたいDJ向け

ループポイント3: Bar 17-24 (キック+HH+ベース)
  - ベースラインのグルーヴを聴かせたい場合

設計上の注意:
  - ループの先頭と末尾で音が途切れないこと
  - リバーブやディレイのテイルがループ境界で不自然にならないこと
  - エネルギーレベルがループ内で一定であること
  - ループ内での微妙なバリエーション（2-4ループまで）

検証方法（Ableton）:
  1. Clip Loop: On
  2. Loop Start: 目的のBar
  3. Loop End: 8小節後
  4. 10回以上ループして不自然さがないか確認
```

**Outroのループポイント**:
```
ループポイント1: Bar 25-32 (キックのみ)
  - 次の曲のIntroと重ねやすい
  - 最もクリーンな状態

ループポイント2: Bar 17-24 (キック+HH)
  - もう少しリズム要素を残したループ

設計ポイント:
  - Outroのループは無限に繰り返しても自然であること
  - 次の曲のどのセクションとも干渉しないこと
```

### クオンタイズ（グリッド整列）

**DJ用トラックは完璧なグリッド必須**:
```
Warp設定（Ableton）:
  Warp: On
  BPM: 正確に設定（例: 128.00、小数点以下も確認）
  Warp Marker: 小節頭に正確に配置
  モード: Beats（ドラムトラック）/ Complex Pro（マスター）

正確なグリッドが必要な理由:
  1. RekordboxのBeatGrid:
     - 正確なBPMが検出される
     - Waveformのグリッドが一致
     - CDJのSync機能が正しく動作

  2. TraktorのBeatGrid:
     - 自動検出の精度向上
     - Beat Jumpが正確に動作
     - ループが正確に機能

  3. SeratoのBeatGrid:
     - Quantize機能が正しく動作
     - Hot CueのSnap精度向上

グリッドのチェック方法:
  1. DAWでメトロノームと同時再生
  2. 曲全体を通して聴く（特に後半にズレがないか）
  3. 最初と最後の小節でグリッド位置を確認
  4. Rekordbox / Traktor / Serato に取り込んでグリッド確認
  5. CDJ/DJコントローラーで実際にミックスしてテスト
```

### ミックスしやすいポイントの作り方

```
ミックスフレンドリーなセクション設計:

1. ブレイクダウンの長さ:
   最低16小節のブレイクダウンを1つ以上設ける
   → DJがこのセクションで次の曲を導入しやすい

2. エネルギーの谷:
   ドロップの間に明確なエネルギーの谷を作る
   → DJがエネルギーを制御しやすい

3. リズムの簡素化ポイント:
   トラック中盤にキックのみのセクションを2-4小節入れる
   → ミックスポイントとして利用可能

4. フィルタースウィープポイント:
   ブレイクダウン前後にフィルタースウィープを設計
   → DJのフィルターエフェクトと自然に融合

5. 8小節ルール:
   全てのセクション変化は8小節（1フレーズ）の倍数で発生
   → DJがフレーズを読みやすい
```

---

## ジャンル別トラック構成

### テクノ（Tech/Peak Time Techno）

```
BPM: 128-135
キー: マイナーキー推奨
総尺: 6-8分

構成:
  Intro (32 bars): Kick → +HH → +Percussion → +FX
  Build 1 (16 bars): +Bass + Synth Stab
  Drop 1 (32 bars): Full elements
  Breakdown (16-32 bars): Atmospheric + Percussion only
  Build 2 (8-16 bars): Riser + Snare Roll
  Drop 2 (32 bars): Full elements（+ additional layer）
  Outro (32 bars): Elements remove → Kick only

サウンド特徴:
  - タイトなキック（短いテイル、パンチ重視）
  - サブベース（ルート音、シンプル）
  - メタリックパーカッション
  - インダストリアルなテクスチャー
  - リバーブスペース（ダブテクノ要素）
  - アシッドライン（303系、オプション）

制作Tips:
  - キックとベースのサイドチェインは深め（-12dB以上）
  - ハイハットのベロシティバリエーション
  - パーカッションのパンニング
  - 空間系エフェクトを積極的に使用
  - ミニマルでも「動き」を感じさせる
```

### ハウス（Tech House / Deep House）

```
BPM: 120-128
キー: マイナー/メジャー両方
総尺: 5-7分

構成:
  Intro (32 bars): Kick → +HH → +Bass → +Pad
  Verse/Groove (32 bars): Main groove + Vocal snippet
  Build (8-16 bars): +FX, Filter sweep
  Drop/Chorus (32 bars): Full elements + Main hook
  Breakdown (16 bars): Vocal + Pad + Minimal drums
  Drop 2 (32 bars): Full elements
  Outro (32 bars): Elements remove → Kick only

サウンド特徴:
  - ラウンドなキック（909系）
  - ウォームなベースライン（グルーヴィー）
  - オープン/クローズドハイハット（16分シャッフル）
  - ボーカルチョップ/サンプル
  - コードスタブ（ファンキー）
  - パッド（空間的）

制作Tips:
  - グルーヴ/スウィングを14-18%程度
  - ベースラインにバリエーション（4-8小節パターン）
  - ボーカルサンプルの効果的な配置
  - 空間の「広さ」を意識（ステレオイメージ）
  - ミックスの「温かさ」（サチュレーション軽め）
```

### プログレッシブハウス / メロディックテクノ

```
BPM: 122-130
キー: マイナーキー推奨
総尺: 7-10分

構成:
  Intro (32-64 bars): Kick → +HH → +Bass → +Arpeggio
  Build 1 (32 bars): Gradual element addition
  Drop 1 (32 bars): Main melody + Full groove
  Breakdown 1 (32-64 bars): Atmospheric + Melody development
  Build 2 (16 bars): Riser + Percussion build
  Drop 2 (32-64 bars): Full elements + Additional layers
  Breakdown 2 (16 bars): Emotional peak
  Drop 3 (16-32 bars): Final statement
  Outro (32-64 bars): Gradual element removal

サウンド特徴:
  - 長いテイルのキック
  - 深いベースライン（進行感のある）
  - レイヤードアルペジオ
  - エモーショナルなパッド進行
  - 壮大なブレイクダウン
  - 繊細なパーカッション

制作Tips:
  - 長いアレンジメントでもストーリーを持たせる
  - 各セクションに新しい要素を追加して展開
  - ブレイクダウンでの「溜め」を大切に
  - オートメーションで流れるような展開
  - リバーブの質にこだわる
```

### ドラムンベース

```
BPM: 170-180
キー: マイナーキー推奨
総尺: 4-6分

構成:
  Intro (16-32 bars): Atmospheric + Half-time drums
  Build (8-16 bars): Snare roll + Riser
  Drop 1 (32 bars): Full DnB beat + Bass
  Breakdown (8-16 bars): Pad + Vocal/Melody
  Build 2 (8 bars): Riser
  Drop 2 (32 bars): Full elements + Variation
  Outro (16-32 bars): Beat simplification → Kick/Snare only

サウンド特徴:
  - パワフルなキック + スネア（2ステップ）
  - 高速ブレイクビーツ
  - ディープなサブベース（Reese系）
  - シャープなハイハット
  - アグレッシブなシンセ（ニューロファンク）
  or エモーショナルなパッド（リキッド）

制作Tips:
  - キックとスネアのバランスが最重要
  - ベースのサイドチェインはキック&スネア両方に
  - 高速ブレイクビーツのプログラミング
  - Introはハーフタイムでも可（DJが合わせやすい）
  - Outroでビートを徐々にシンプルに
```

### トランス

```
BPM: 136-142
キー: マイナーキー推奨
総尺: 7-10分

構成:
  Intro (32-64 bars): Kick → +Bass → +Percussion
  Build 1 (16-32 bars): +Arpeggio + Pad
  Drop 1 (32 bars): Main melody + Full elements
  Breakdown (32-64 bars): Melodic development + Vocal
  Build 2 (16-32 bars): Big riser + Snare roll
  Drop 2 (32-64 bars): Main melody + Extra layers
  Outro (32-64 bars): Element removal

サウンド特徴:
  - パンチのあるキック
  - ドライビングベースライン
  - シーケンスアルペジオ（16分音符）
  - 壮大なパッド進行
  - エモーショナルなリード/メロディ
  - ライザー/FXが豊富

制作Tips:
  - ブレイクダウンの美しさがトランスの核
  - メロディの感動的な展開
  - 段階的なエネルギーの上昇
  - ベースラインはシンプルだがドライビング
  - Introは64小節でも許容される
```

---

## キックドラムの設計

### DJ用キックの要件

```
DJ用キックドラムに求められる特性:

1. パンチ:
   - アタック（トランジェント）が明確
   - 2-5kHz帯域にクリック要素
   - ピーク: -6dB to -3dB

2. ボディ:
   - 60-120Hz帯域のウォームな響き
   - 適度なサスティン（50-200ms）
   - ジャンルに応じた太さ

3. サブ:
   - 30-60Hz帯域
   - クリーンなサイン波
   - 他のトラックのサブと干渉しないテイル

4. テイル:
   - 短め推奨（200-400ms）
   - ミックス時に次のキックと重ならない
   - ローパスフィルターでテイルの高域をカット

キックのフリケンシー配分:
  30-60Hz: サブ（基音）
  60-120Hz: ボディ（厚み）
  120-300Hz: ボクシーさ（カットが多い帯域）
  300-1kHz: プレゼンス（明瞭さ）
  1-5kHz: クリック/アタック
  5kHz+: エア（空気感）
```

### ジャンル別キック設計

```
テクノキック:
  基音: 45-55Hz
  テイル: 短め（150-250ms）
  キャラクター: パンチ重視、タイト
  処理: コンプレッション強め、EQカーブ

ハウスキック:
  基音: 50-60Hz
  テイル: 中程度（200-350ms）
  キャラクター: ラウンド、ウォーム（909系）
  処理: サチュレーション軽め、自然な響き

トランスキック:
  基音: 45-55Hz
  テイル: やや長め（250-400ms）
  キャラクター: パワフル、ドライビング
  処理: レイヤリング（アタック + サブ）

ドラムンベースキック:
  基音: 50-70Hz
  テイル: 短め（100-200ms）
  キャラクター: タイト、スナッピー
  処理: トランジェントシェイパー、パラレルコンプ
```

---

## ベースラインの設計

### DJ用ベースの要件

```
ベースラインの設計原則:

1. キーの明確さ:
   - ルート音が明確
   - DJがキーを判断しやすい
   - Mixed In Keyの検出精度向上

2. サブベースの管理:
   - モノラル（センター定位）
   - 30-80Hz帯域のクリーンさ
   - キックとの住み分け（サイドチェイン）

3. パターンの予測可能性:
   - 4小節または8小節のループパターン
   - 1小節目のダウンビートにルート音
   - DJがフレーズを読みやすい

4. ミックス時の親和性:
   - 他のトラックのベースと干渉しにくい
   - フィルターでのコントロールが容易
   - EQカットで自然に消える
```

### ジャンル別ベースパターン

```
テクノベース:
  パターン: ルート音の繰り返し（1/4 or 1/8）
  サウンド: サイン波 + 軽いサチュレーション
  範囲: 1オクターブ以内
  特徴: シンプル、ドライビング

ハウスベース:
  パターン: グルーヴィーなラインン（1/8 + シンコペーション）
  サウンド: 909系サブ + ミッドベースレイヤー
  範囲: 1-1.5オクターブ
  特徴: ファンキー、ウォーム

プログレッシブベース:
  パターン: 長いノート（1/2 or 1小節）
  サウンド: フィルタード・サブベース
  範囲: 半オクターブ以内
  特徴: ドリーミー、進行感

ドラムンベースベース:
  パターン: Reese系（うねり）
  サウンド: ディストーション + フィルターモジュレーション
  範囲: 1-2オクターブ
  特徴: アグレッシブ、ダーク
```

---

## アレンジメント構成の詳細

### 8小節ルール

```
全てのセクション変化は8小節単位で管理:

理由:
  - DJはフレーズ単位（8小節）で楽曲を認識する
  - ミックスのタイミングがフレーズ頭に合う
  - CDJのPhase Meterが正確に表示される
  - 予測可能なアレンジメント = DJフレンドリー

悪い例:
  Intro: 7小節 → Drop: 31小節 → Breakdown: 12小節
  → DJがフレーズを見失う

良い例:
  Intro: 32小節 → Drop: 32小節 → Breakdown: 16小節
  → 全て8の倍数

例外:
  - フィルイン（1-2小節の追加）は許容
  - ただし次のセクション頭は必ずグリッドに合わせる
```

### エネルギーカーブの設計

```
トラック全体のエネルギーの流れ:

エネルギーレベル（1-10）:

|10|                              ████
| 9|                         ████ ████
| 8|               ████████  ████ ████
| 7|          ████ ████████  ████ ████
| 6|     ████ ████ ████████  ████ ████
| 5|████ ████ ████           ████ ████
| 4|████ ████ ████           ████
| 3|████ ████                     ████
| 2|████                          ████
| 1|████                          ████
   |Intro|Build|Drop1|Brkdwn|Bld2|Drop2|Outro
    32bar 16bar 32bar 16bar  8bar 32bar 32bar

設計ポイント:
  - Introは低エネルギーからスタート（1-3）
  - 段階的にエネルギーを上げる
  - ドロップで最大エネルギー（8-10）
  - ブレイクダウンで一旦エネルギーを下げる（3-5）
  - 2回目のドロップは1回目と同等かそれ以上
  - Outroは徐々にエネルギーを下げる
```

### テンション管理

```
テンションの作り方:

ライザー（上昇FX）:
  - ホワイトノイズのフィルタースウィープ（LP → 開く）
  - ピッチの上昇（+12 semitones over 8 bars）
  - ドラムロールの加速（1/8 → 1/16 → 1/32）
  - リバーブのDecay増加

ダウンリフター（下降FX）:
  - ドロップ直前の「落とし」
  - ピッチの下降
  - フィルターの閉じ
  - サブドロップ（低周波のインパクト）

テンションの解放:
  - ドロップの1拍目で全テンション解放
  - インパクトFX + キック + ベース同時開始
  - 直前の0.5-1拍の無音が効果的
```

---

## アカペラ・インストの分離

### なぜ必要か

**DJ用途**:
```
Acapella（ボーカルのみ）:
  - マッシュアップ素材として
  - 他のトラックのインストに重ねる
  - DJセットのアクセント
  - ライブリミックスの素材

Instrumental（ボーカル抜き）:
  - ボーカルトラックの下に敷く
  - リミックスのベーストラック
  - BGM使用
  - 他のアカペラを重ねる

ステム（個別パーツ）:
  - Drums: キック、スネア、ハイハット、パーカッション
  - Bass: サブベース、ミッドベース
  - Synth: リード、パッド、アルペジオ
  - Vocal: リードボーカル、ハーモニー
  → 4ステム形式が業界標準（Native Instruments Stems）
```

### 制作段階での分離

**トラック構成の分離**:
```
DAWでのグループ構成:

Group 1: DRUMS
  Track 1: Kick
  Track 2: Snare / Clap
  Track 3: Hi-Hat / Cymbals
  Track 4: Percussion
  → Bus: Drum Bus

Group 2: BASS
  Track 5: Sub Bass
  Track 6: Mid Bass
  → Bus: Bass Bus

Group 3: SYNTH / MUSIC
  Track 7: Lead Synth
  Track 8: Pad
  Track 9: Arpeggio
  Track 10: FX / Textures
  → Bus: Synth Bus

Group 4: VOCALS
  Track 11: Lead Vocal
  Track 12: Backing Vocal
  Track 13: Vocal FX
  → Bus: Vocal Bus

Master Bus:
  ← Drum Bus + Bass Bus + Synth Bus + Vocal Bus
```

**エクスポート手順**:
```
1. フルミックス（Original Mix）:
   全グループ有効
   Export: "Artist - Track Name (Original Mix).wav"
   → メインの完成版

2. インストゥルメンタル版:
   Group 4（VOCALS）をミュート
   Export: "Artist - Track Name (Instrumental).wav"
   → ボーカル以外の全要素

3. アカペラ版:
   Group 1-3をミュート、Group 4のみ有効
   Export: "Artist - Track Name (Acapella).wav"
   → ボーカルのみ

4. ステム版（4ステム）:
   各グループを個別にエクスポート
   Export:
     "Artist - Track Name (Stem - Drums).wav"
     "Artist - Track Name (Stem - Bass).wav"
     "Artist - Track Name (Stem - Synth).wav"
     "Artist - Track Name (Stem - Vocals).wav"
   → 各パーツを個別に操作可能

5. DJツール版:
   特定の要素のみ（例: Drums + Bass only）
   Export: "Artist - Track Name (Dub Mix).wav"
   → ミニマルバージョン

エクスポート設定:
  Sample Rate: 44.1kHz（Beatport標準）/ 48kHz
  Bit Depth: 24bit（推奨）/ 16bit
  Format: WAV（推奨）/ AIFF
  Normalize: Off
  Dither: Triangular（24bit→16bit変換時）
```

### 後からの分離（AI分離）

```
制作段階で分離していない場合:

AIベースのステム分離ツール:
  1. Ableton 11.1+ 内蔵機能:
     - 楽曲をドラム、ベース、ボーカル、その他に分離
     - 品質は良好だが完全ではない
     - CPU負荷が高い

  2. iZotope RX:
     - Music Rebalance機能
     - 高品質な分離
     - プロフェッショナル向け

  3. LALAL.AI:
     - クラウドベース
     - ステム分離に特化
     - ボーカル分離の品質が高い

  4. Demucs（Meta/Facebook）:
     - オープンソース
     - 4ステム分離
     - 高品質

注意点:
  - AI分離は完璧ではない（アーティファクトが残る）
  - 制作段階での分離が最高品質
  - DJ用途では許容範囲の品質
  - マスタリング済み音源からの分離は品質が落ちる
```

---

## DJ用ループ・フィルイン

### ドラムループの制作

**シンプルなドラムループ（8-16小節）**:
```
基本構成:
  Kick: 4つ打ち（毎拍）
  Snare/Clap: 2拍目、4拍目
  Hi-Hat (Closed): 16分音符（ベロシティ変化）
  Hi-Hat (Open): 2小節に1回（アクセント）
  Percussion: コンガ、ボンゴ等（グルーヴ追加）

BPM別ループ:
  120 BPM Loop: ディープハウス向け
  124 BPM Loop: テックハウス向け
  128 BPM Loop: ハウス/テクノ向け
  132 BPM Loop: テクノ向け
  140 BPM Loop: トランス/ハードスタイル向け
  174 BPM Loop: ドラムンベース向け

エクスポート設定:
  長さ: 8小節 or 16小節
  テイル: なし（クリーンカット）
  フォーマット: WAV 44.1kHz/24bit
  ファイル名: "Drum_Loop_128BPM_Aminor_YourName.wav"
```

**バリエーションループ**:
```
ループA（基本）: 標準パターン 8小節
ループB（バリエーション）: 基本 + パーカッション追加 8小節
ループC（フィル入り）: 基本 + 最後の2小節にフィル 8小節
ループD（ミニマル）: キック + HH のみ 8小節

用途:
  - ループA: トランジションのベース
  - ループB: エネルギー追加
  - ループC: セクション変化の予告
  - ループD: ミックスポイント
```

### フィルインの制作

**スネアロール**:
```
4小節のスネアロール（ビルドアップ用）:

Bar 1: 通常パターン（1/4音符スネア）
Bar 2: 1/8音符スネア
Bar 3: 1/16音符スネア
Bar 4: 1/32音符スネア（ロール）

追加処理:
  - ピッチ: 徐々に上昇（+0 → +2 semitones）
  - リバーブ: 徐々に増加
  - ボリューム: 徐々に増加
  - パンニング: 狭 → 広

バリエーション:
  A: クラップロール（よりオープンなサウンド）
  B: ハイハットロール（シャープなサウンド）
  C: パーカッションロール（ボンゴ/コンガ）
```

**ライザー（Riser）**:
```
4-8小節のライザー制作:

方法1: ホワイトノイズライザー
  ソース: White Noise
  フィルター: LP Filter
    Bar 1: Cutoff 200 Hz
    Bar 8: Cutoff 12000 Hz
  ピッチ: +12 semitones over 8 bars
  ボリューム: -20dB → -6dB

方法2: シンセライザー
  ソース: Saw Wave
  フィルター: LP Filter（同上）
  ユニゾン: 4-8 voices
  デチューン: 徐々に広がる
  リバーブ: 増加

方法3: リバースシンバル
  ソース: Crash Cymbal を反転
  フェード: 徐々にフェードイン
  リバーブ: 長め

方法4: コンビネーション
  全方法を重ねて壮大なライザー
  8小節: 最大インパクト
```

**ダウンリフター（Impact/Drop FX）**:
```
ドロップ前後のインパクトFX:

サブドロップ:
  ソース: Sine Wave
  ピッチ: C1 → C0（1オクターブ下降、0.5秒）
  ボリューム: 高め（インパクト感）
  注意: サブ帯域なのでスピーカーに負荷

インパクトヒット:
  ソース: ノイズバースト + サブ + リバーブ
  長さ: 0.5-2秒
  リバーブ: Large Hall、Decay 3-5秒
  用途: ドロップの1拍目に配置

リバースリバーブ:
  ソース: 次のセクションの最初の音
  処理: リバーブ → 反転
  長さ: 2-4秒
  用途: ドロップ直前に配置（期待感）
```

---

## トランジションツール

### Filter Sweep ループ

**作り方**:
```
コード進行: Am-F-C-G（または任意のシンプル進行）
シンセ: Wavetable / Analog Pad
Auto Filter:
  Type: Low Pass
  LFO Rate: 1/4（BPM同期）
  LFO Amount: 80%
  Resonance: 20-30%
ループ長: 16小節
ボリューム: -6dB（ミックスで使いやすい）

用途:
  - 2曲の間に挟むトランジションツール
  - エネルギーレベルの調整
  - ブレイクダウン時の空間埋め
  - アンビエントレイヤー
```

### Ambient Pad

**作り方**:
```
シンセ: Analog / Wavetable Pad
  Oscillator: Saw + Sine
  Filter: LP Filter、Cutoff 2000Hz
  Envelope: Attack 2000ms, Release 4000ms
  Unison: 4 voices, Detune 15%
リバーブ: Hall、Decay 4.0-6.0s、Mix 60%
ディレイ: 1/4 Ping Pong、Feedback 30%
コード: 1つのコード（Cm等）を32小節維持
ボリューム: -9dB

用途:
  - ブレイクダウンの空間演出
  - アンビエントトランジション
  - エナジーのクールダウン
  - 静かなセクションの背景
```

### ドラムツール

```
トランジション専用ドラムツール:

ツール1: キックのみループ（32小節）
  128BPM、4つ打ちキックのみ
  → 最もシンプルなトランジションツール

ツール2: キック+ライド（16小節）
  キック + ライドシンバル
  → テクノトランジション向け

ツール3: パーカッションブレイク（8小節）
  コンガ、ボンゴ、シェイカーのみ（キックなし）
  → エスニック/トライバル要素の追加

ツール4: ブレイクビーツ（8小節）
  ファンキーなブレイクビーツパターン
  → ブレイクス/UKガレージ的トランジション

ツール5: ハーフタイムループ（16小節）
  キック+スネアのハーフタイムパターン
  → テンポ感の変化（ダブステップ的）
```

---

## ブートレグ・エディット

### 合法的な範囲

**ブートレグ（Bootleg）**:
```
定義: 非公式リミックス/エディット

合法的な使用:
  - DJセット内でのみ使用（非販売）
  - SoundCloudに「Free Download」で公開
  - Mixcloudにアップロード（ライセンス済み）
  - 原曲クレジット必須
  - プロモーション用（ライブ配信等）

グレーゾーン:
  - YouTubeへのアップロード（Content IDで検出される可能性）
  - SoundCloudでの公開（テイクダウン要請の可能性）

明確に違法:
  - 販売（Beatport、iTunes、Amazon等）
  - ストリーミング配信（Spotify、Apple Music等）
  - 広告収益を得る使用
  - 原曲クレジットなしの使用

リスク管理:
  - 原曲の権利者に連絡を取ることが理想
  - テイクダウン要請には速やかに従う
  - 販売目的では絶対に使用しない
  - 「Free Download / Not For Sale」を明記
```

### エディットの種類

```
1. BPM変更エディット:
   Original: 110 BPM (Hip Hop)
   Edit: 128 BPM (House)

   方法:
     Ableton Warp:
       Complex Pro（ボーカル含む場合）
       Beats（ドラムブレイクのみの場合）
       Segment BPM: 128.00

     注意: 大幅なBPM変更は音質劣化
     推奨範囲: ±20%以内

2. 構成変更エディット:
   Original:
     Intro → Verse → Chorus → Verse → Chorus → Outro
   Edit（DJ用）:
     32小節 Intro → Chorus → Breakdown → Chorus → 32小節 Outro

   方法:
     - DAWで楽曲をセクションごとにカット
     - 必要なセクションを並べ替え
     - セクション間のクロスフェード
     - 32小節のIntro/Outroを追加

3. マッシュアップエディット:
   曲A: ボーカル/アカペラ
   曲B: インストゥルメンタル/ビート

   方法:
     - 曲Aのボーカルをステム分離
     - 曲Bのインストを準備
     - BPMを統一（Warp）
     - キーを合わせる（必要に応じてTranspose）
     - EQで帯域を住み分け
     - ミックス&マスタリング

4. 拡張エディット:
   Original: 3分の曲
   Edit: 6分のDJバージョン

   方法:
     - Intro/Outroの追加
     - ブレイクダウンの追加
     - セクションの繰り返し
     - FXの追加

5. レデュースエディット:
   Original: 7分のDJトラック
   Edit: コアセクションのみ抽出

   方法:
     - 不要なセクションのカット
     - ドロップ/ハイライトのみ抽出
     - 短いIntro/Outroを付加
```

### エディット例の詳細

**BPM変更の実践**:
```
例: Hip Hop曲をHouse BPMに変更

Original: "Artist - Song" 95 BPM
Target: 124 BPM

手順:
  1. DAWにインポート
  2. Warp Mode: Complex Pro
  3. Project Tempo: 124 BPM
  4. Warpマーカーを全体に配置
  5. ボーカルのピッチ確認（不自然でないか）
  6. 必要に応じてPitch Correct

追加処理:
  - 4つ打ちキックの追加（サイドチェイン付き）
  - ベースラインの追加/変更
  - ハイハットの追加
  - 32小節Intro/Outroの作成

結果: Hip Hop曲がHouseセットで使用可能に
```

**構成変更の実践**:
```
例: ポップ曲をDJエディットに変換

Original構成:
  Intro (8bar) → Verse1 → Chorus1 → Verse2 → Chorus2 → Bridge → Chorus3 → Outro (4bar)

DJ Edit構成:
  DJ Intro (32bar, kick only → elements add)
  → Chorus1 (16bar)
  → Breakdown (8bar, vocal snippet + pad)
  → Build (8bar, riser + snare roll)
  → Chorus2 (16bar, full energy)
  → Breakdown 2 (16bar, bridge melody)
  → Chorus3 (16bar)
  → DJ Outro (32bar, elements remove → kick only)

手順:
  1. 原曲を分析（セクション、BPM、キー）
  2. 使いたいセクションを選択
  3. DAWで並べ替え
  4. クロスフェードでセクション接続
  5. DJ Intro/Outroを制作して追加
  6. 全体のエネルギーフローを確認
  7. ミックス&マスタリング
```

---

## ミキシング・マスタリング（DJ用）

### ミキシングの注意点

```
DJ用トラックのミキシング特有の考慮事項:

1. 低域のクリーンさ:
   - 30Hz以下をハイパスフィルターでカット
   - キックとベースのサイドチェイン処理
   - 低域はモノラル（Utility: Mono below 120Hz）
   - サブベースとキックの帯域を明確に分離

   理由: DJミキシング時に2曲の低域が重なるため
   クリーンな低域 = ミックス時の干渉が少ない

2. ダイナミクスの管理:
   - 過度なコンプレッションを避ける
   - ピークとRMSの差: 8-12dB推奨
   - トランジェント（キックのアタック）を保持
   - DJミキサーのヘッドルームを確保

   理由: DJミキサーでの追加処理に余裕が必要

3. ステレオイメージ:
   - 低域（120Hz以下）: モノ
   - 中域（120Hz-5kHz）: やや狭め
   - 高域（5kHz以上）: 広め
   - 過度なステレオ拡張は避ける

   理由: クラブのPAシステムではモノ再生の場合がある

4. EQバランス:
   - フラットに近いバランスを目指す
   - 特定の帯域が突出しない
   - リファレンストラックと比較

   理由: DJミキサーのEQで調整されることを前提
```

### マスタリングの設定

```
DJ用トラックのマスタリング:

チェーン例:
  1. EQ（リニアフェイズ）:
     - HPF: 30Hz
     - LPF: 18kHz（微かにカット）
     - 問題帯域の修正

  2. マルチバンドコンプレッサー:
     - Low (30-120Hz): Ratio 3:1, Threshold -18dB
     - Mid (120Hz-5kHz): Ratio 2:1, Threshold -12dB
     - High (5-18kHz): Ratio 2:1, Threshold -15dB
     - 低域のタイトさ確保

  3. ステレオイメージャー:
     - Low: Mono
     - Mid: 100%
     - High: 110-120%

  4. リミッター:
     - Ceiling: -1.0 dBTP（True Peak）
     - Target Loudness: -7 to -5 LUFS
     - Attack: Auto / Fast

ラウドネス目標:
  クラブ向け: -8 to -6 LUFS
  ストリーミング兼用: -14 LUFS（Spotify等の正規化対応）
  DJ用推奨: -7 LUFS（最大の互換性）

True Peak制限:
  -1.0 dBTP: 推奨（安全マージン）
  -0.3 dBTP: 最小限（コーデック変換考慮）
  0 dBTP: 非推奨（クリッピングリスク）
```

### マスタリングチェックリスト

```
マスタリング完了前のチェック:

音質チェック:
  □ リファレンストラックとの比較
  □ 複数のスピーカー/ヘッドフォンで確認
  □ モノ互換性チェック（Utility: Mono）
  □ 低域のクリーンさ確認
  □ 高域の歪みがないか確認
  □ 全体のバランス（スペクトラムアナライザー）

技術チェック:
  □ ラウドネス: -8 to -6 LUFS
  □ True Peak: -1.0 dBTP以下
  □ サンプルレート: 44.1kHz
  □ ビット深度: 24bit → 16bit（適切なディザリング）
  □ DC Offset がないか
  □ ファイルの先頭と末尾に無音がないか

DJチェック:
  □ Rekordboxに取り込んでBeatGrid確認
  □ 他のDJトラックとのミックステスト
  □ Intro/Outroの長さ確認（32小節以上）
  □ ループポイントの動作確認
```

---

## キーとBPMの選択

### ジャンル別BPM

```
ジャンル別の標準BPM範囲:

ディープハウス: 118-124 BPM
テックハウス: 124-128 BPM
ハウス: 120-128 BPM
プログレッシブハウス: 122-130 BPM
メロディックテクノ: 124-130 BPM
テクノ: 128-135 BPM
ピークタイムテクノ: 132-140 BPM
ハードテクノ: 140-155 BPM
トランス: 136-142 BPM
アップリフティングトランス: 138-142 BPM
サイトランス: 140-148 BPM
ダブステップ: 140 BPM（ハーフタイム 70）
ドラムンベース: 170-180 BPM
ジャングル: 160-170 BPM
ヒップホップ: 80-100 BPM
トラップ: 130-170 BPM（ハーフタイム 65-85）
UK ガレージ: 130-140 BPM
ブレイクス: 125-135 BPM
エレクトロ: 125-130 BPM
ハードスタイル: 150-160 BPM

BPM選択のアドバイス:
  - ジャンルの中央値付近が最も汎用的
  - 速すぎ/遅すぎは他の曲とミックスしにくい
  - 最近のトレンドではBPMがやや上昇傾向
  - 複数ジャンルで使いたい場合: 126-128 BPM が万能
```

### ハーモニックミキシング対応

**キー設定**:
```
ハーモニックミキシングを前提とした制作:

キーの明確さ:
  - 1つのスケール（調）内で完結させる
  - 調号変更（転調）は最小限に
  - ルート音が明確に認識できること
  - Mixed In Keyやkeyfinder で正確に検出されること

Camelot Wheel（カメロットホイール）:
  1A = Ab minor    1B = B major
  2A = Eb minor    2B = F# major
  3A = Bb minor    3B = Db major
  4A = F minor     4B = Ab major
  5A = C minor     5B = Eb major
  6A = G minor     6B = Bb major
  7A = D minor     7B = F major
  8A = A minor     8B = C major
  9A = E minor     9B = G major
  10A = B minor    10B = D major
  11A = F# minor   11B = A major
  12A = C# minor   12B = E major

制作時のキー選択アドバイス:
  - マイナーキー（A列）: エレクトロニックミュージックで最も一般的
  - Am (8A): 最もポピュラー、多くの曲とミックス可能
  - Cm (5A): テクノ/ハウスで頻出
  - Gm (6A): ファンキーなハウスに多い
  - Em (9A): メロディックな曲に適合
  - メジャーキー（B列）: ハッピー/アップリフティングな曲に

隣接キーとの互換性:
  8A (Am) と相性の良いキー:
    - 7A (Dm): -1ステップ
    - 9A (Em): +1ステップ
    - 8B (C major): パラレルキー
  → これらのキーの曲とスムーズにミックス可能
```

---

## エクスポート設定

### ファイルフォーマット

```
DJ用トラックのエクスポート設定:

マスターファイル（配布用）:
  Format: WAV
  Sample Rate: 44.1 kHz（Beatport標準）
  Bit Depth: 16 bit（CD品質）
  Dithering: Triangular（24bit → 16bit変換時）

高品質マスター（アーカイブ用）:
  Format: WAV
  Sample Rate: 48 kHz / 96 kHz
  Bit Depth: 24 bit
  Dithering: なし

ストリーミング用:
  Format: WAV → 各プラットフォームが変換
  Sample Rate: 44.1 kHz
  Bit Depth: 16 bit

プロモーション用:
  Format: MP3
  Bit Rate: 320 kbps CBR
  Sample Rate: 44.1 kHz

ファイル命名規則:
  "Artist_Name - Track_Title (Mix_Name).wav"
  例: "DJ_Gaku - Midnight_Drive (Original_Mix).wav"
  例: "DJ_Gaku - Midnight_Drive (Instrumental).wav"
  例: "DJ_Gaku - Midnight_Drive (Dub_Mix).wav"
```

---

## メタデータの設定

### ID3タグ

```
正確なメタデータの重要性:

必須タグ:
  Title: Track Name (Mix Name)
  Artist: Artist Name
  Album: Single Name / EP Name
  Genre: Tech House / Techno / etc.
  Year: 2024
  BPM: 128（正確に）
  Key: Am / 8A（Camelot表記推奨）
  Comment: "Original Mix" / "DJ Edit"

推奨タグ:
  Label: レーベル名
  Catalog Number: CAT001
  ISRC: 国際標準レコーディングコード
  Initial Key: Am（Open Key表記）
  Energy: 1-10（Rekordbox互換）

Rekordbox向けの注意:
  - BPMは小数点以下2桁まで（例: 128.00）
  - Keyは正確に（Am, Cm等の標準表記）
  - Artwork: 500x500px以上

タグ編集ツール:
  - Mp3tag（Windows）
  - Kid3（Mac/Windows/Linux）
  - MusicBrainz Picard
  - Rekordbox内蔵エディター
```

---

## DJ用テンプレート作成

### Ableton Liveテンプレート

```
DJ Tool Template の構成:

テンプレート名: "DJ Tool Template [BPM]"

Track構成:
  Track 1: Kick
    デバイス: Drum Rack（キックサンプル）
    EQ: HPF 30Hz, LPF 8kHz
    Compressor: Ratio 4:1, Fast Attack

  Track 2: Snare/Clap
    デバイス: Drum Rack（スネア/クラップサンプル）
    EQ: HPF 100Hz
    Transient Shaper: Attack +3dB

  Track 3: Hi-Hat/Cymbals
    デバイス: Drum Rack（HHサンプル）
    EQ: HPF 500Hz
    Pan: 微妙にオフセンター

  Track 4: Percussion
    デバイス: Drum Rack（パーカッションサンプル）
    Pan: ステレオに広げる

  Track 5: Sub Bass
    デバイス: Operator / Wavetable（Sine Wave）
    EQ: LPF 120Hz
    Utility: Mono
    Sidechain: Track 1 (Kick)

  Track 6: Mid Bass
    デバイス: Wavetable（Saw/Square）
    EQ: HPF 80Hz, LPF 5kHz
    Sidechain: Track 1 (Kick)

  Track 7: Pad/Synth
    デバイス: Wavetable（Pad Preset）
    Reverb: Hall 3.0s
    EQ: HPF 200Hz

  Track 8: Lead
    デバイス: Wavetable（Lead Preset）
    Delay: 1/8 Ping Pong
    EQ: HPF 300Hz

  Track 9: FX/Risers
    デバイス: Simpler（FXサンプル）
    Reverb: Large Room

  Track 10: Vocal（オプション）
    デバイス: なし（オーディオ用）
    EQ: HPF 80Hz, De-esser
    Compressor: Ratio 3:1

Return Tracks:
  Return A: Reverb（Reverb: Hall 2.5s, HPF 200Hz）
  Return B: Delay（Echo: 1/8, Feedback 40%）
  Return C: Filter（Auto Filter: LPF, Res 20%）

Master Track:
  EQ Eight: HPF 30Hz
  Glue Compressor: Ratio 2:1, Makeup +1dB
  Limiter: Ceiling -1.0dBTP

Tempo: [BPM]
Time Signature: 4/4
```

### テンプレートのバリエーション

```
BPM別テンプレート:
  DJ_Tool_Template_124BPM.als（テックハウス）
  DJ_Tool_Template_128BPM.als（ハウス/テクノ）
  DJ_Tool_Template_132BPM.als（テクノ）
  DJ_Tool_Template_140BPM.als（トランス）
  DJ_Tool_Template_174BPM.als（ドラムンベース）

ジャンル別テンプレート:
  DJ_Tool_TechHouse_Template.als
  DJ_Tool_Techno_Template.als
  DJ_Tool_Progressive_Template.als
  DJ_Tool_DnB_Template.als
  DJ_Tool_Trance_Template.als

目的別テンプレート:
  DJ_Edit_Template.als（既存曲のエディット用）
  DJ_Mashup_Template.als（マッシュアップ用）
  DJ_Loop_Template.als（ループ制作用）
  DJ_Transition_Template.als（トランジションツール用）
```

---

## 配布とプロモーション

### SoundCloud無料配布

```
SoundCloudでのDJツール配布:

アップロード設定:
  1. SoundCloudにログイン
  2. Upload → トラックをアップロード
  3. 設定:
     Title: "Artist - Track Name (DJ Edit) [Free Download]"
     Genre: 適切なジャンル
     Tags: #FreeDownload, #DJTool, #DJEdit, #Techno, #House 等
     Description:
       "Free Download for DJs!
       BPM: 128
       Key: A Minor (8A)
       Duration: 6:30
       Format: WAV 16bit/44.1kHz

       Use in your sets!
       Tag me @YourName if you play this

       Download link: [Hypeddit/Toneden link]

       Not for sale. Original elements by [Original Artist]."

     Download: Enable
     License: Creative Commons（オリジナルの場合）

  4. Artwork: 正方形（800x800px以上）
  5. Waveformの確認

プロモーション戦略:
  - 定期的に新しいエディット/ツールを公開
  - SNSでDJコミュニティにシェア
  - DJチャートに含める
  - リポストグループに参加
  - タグ付けでコミュニティ拡大
```

### Bandcamp

```
Bandcampでの配布:

設定:
  Price: $0（Free）または Name Your Price
  Format: WAV + MP3（自動変換）
  Album Art: 1400x1400px
  Description: BPM、キー、用途を記載
  Tags: ジャンル、BPM、用途

利点:
  - 高品質なWAVファイルのダウンロード
  - ファンとの直接的な繋がり
  - 収益化の可能性（Name Your Price）
  - プロフェッショナルなプレゼンテーション
```

### Beatport / Traxsource

```
商用リリース:

Beatport:
  - ディストリビューター経由（DistroKid, TuneCore, Amuse等）
  - WAV 44.1kHz/16bit
  - メタデータの正確な記入
  - Artwork: 1400x1400px（JPG）
  - ジャンル分類の正確さ

Traxsource:
  - ハウス/ファンク系に強い
  - WAV 44.1kHz/16bit
  - 同様のディストリビューター経由

リリース戦略:
  - シングル: 1-2曲（Original + Instrumental）
  - EP: 3-5曲（バリエーション）
  - プロモ: DJプール/プロモサービス経由
```

---

## よくある失敗と対処法

### 失敗1: Introが短すぎる

```
問題: Introが16小節以下
影響: DJがミックスする時間が足りない
対処法:
  - 最低32小節のIntroを確保
  - 理想は64小節（余裕のあるミックス）
  - キックのみのセクションから始める

防止策:
  テンプレートに32小節のIntroを予め設定しておく
```

### 失敗2: BPMがずれている

```
問題: グリッドが正確でない、BPMが微妙にずれている
影響: Sync機能が正しく動作しない、手動ミックスが困難
対処法:
  - DAWのWarp設定を正確に（0.01BPM単位）
  - メトロノームと全体を通して確認
  - グリッドが小節頭に正確に一致
  - Rekordboxで取り込んでBeatGrid確認
  - CDJで実際にミックスしてテスト

防止策:
  - MIDIで打ち込んだリズムは完璧なグリッド
  - オーディオサンプルはWarp確認必須
  - テンポオートメーションを使わない（DJ用）
```

### 失敗3: キーが不明瞭

```
問題: 複数のキーが混在、転調が多い
影響: ハーモニックミキシングが困難
対処法:
  - 1つのスケール内で完結させる
  - Mixed In Keyで分析して確認
  - メタデータにキーを正確に記載
  - ベースのルート音をキーと一致させる

防止策:
  - 制作前にキーを決めて一貫させる
  - 転調する場合は明確なセクション変化とセットで
```

### 失敗4: 低域が濁っている

```
問題: キックとベースが干渉、低域が不明瞭
影響: DJミックス時に低域が暴れる
対処法:
  - キックとベースのサイドチェイン処理
  - サブベースを30Hz以下でカット
  - 低域をモノラルに
  - EQで帯域を住み分け

防止策:
  - サイドチェインをテンプレートに組み込む
  - Utility: Mono below 120Hz を標準装備
  - リファレンストラックと低域を比較
```

### 失敗5: 音量が不均一

```
問題: セクション間で音量差が大きい
影響: DJのゲイン調整が頻繁に必要
対処法:
  - 各セクションのラウドネスを測定
  - マスタリングでダイナミクスを管理
  - トラック内のゲイン統一

防止策:
  - ミキシング段階でバス間のバランスを確認
  - LUFS メーターで常にモニタリング
  - リミッターで最大レベルを制御
```

### 失敗6: エフェクトテイルの残留

```
問題: セクション境界でリバーブ/ディレイが途切れる or 残る
影響: 不自然なサウンド、ミックス時の干渉
対処法:
  - リバーブ/ディレイのDecay/Feedbackを適切に設定
  - Outroの最後でエフェクトを完全にフェードアウト
  - Introの最初はドライサウンドから始める

防止策:
  - エフェクトのオートメーションをIntro/Outroに組み込む
  - エクスポート時にOutro後の余分な無音を含める
  - 最終的にトリミングで不要部分をカット
```

### 失敗7: ステレオ幅が広すぎる

```
問題: モノ再生時に位相キャンセルが発生
影響: クラブのPAシステム（モノサブ）で低域が消える
対処法:
  - Utility: Mono でモノ互換性チェック
  - 低域（120Hz以下）はモノに
  - 過度なステレオエンハンサーを避ける

防止策:
  - ミキシング時にモノチェックを習慣化
  - ステレオイメージャーは控えめに
  - 低域モノ化をテンプレートに含める
```

---

## 実践ワークフロー

### Step 1: テンプレート作成

```
新規プロジェクト: "DJ Tool Template 128 BPM"

Track構成:
  1. Kick
  2. Snare/Clap
  3. Hi-Hat
  4. Percussion
  5. Sub Bass
  6. Mid Bass
  7. Pad
  8. Lead
  9. FX
  10. Vocal（オプション）

Return Tracks:
  A. Reverb (Hall 2.5s)
  B. Delay (1/8 Ping Pong)
  C. Filter (Auto Filter LPF)

Master:
  EQ Eight + Glue Compressor + Limiter

Tempo: 128 BPM
Time Signature: 4/4

→ このテンプレートを保存して毎回使用
```

### Step 2: 32小節 Intro作成

```
Arrangement View:

Bar 1-8: Kick のみ
  → クリーン、タイト、一定

Bar 9-16: + Hi-Hat + Percussion
  → グルーヴの確立

Bar 17-24: + Sub Bass + Mid Bass
  → 低域の導入（HPF オートメーション）

Bar 25-32: + Pad + Lead Teaser + FX
  → ドロップへの期待感

Bar 33: ドロップ（全要素同時開始）
```

### Step 3: メインセクション制作

```
Bar 33-96: メインコンテンツ

Drop 1 (Bar 33-64): 32小節
  全要素がフル稼働
  メインメロディ/フック
  最大エネルギー

Breakdown (Bar 65-80): 16小節
  ドラム軽減、パッド主体
  ボーカル/メロディの展開
  エネルギーの谷

Build (Bar 81-88): 8小節
  ライザー、スネアロール
  フィルタースウィープ
  テンション上昇

Drop 2 (Bar 89-120): 32小節
  全要素 + 追加レイヤー
  最大エネルギー（Drop 1以上）
  バリエーション
```

### Step 4: 32小節 Outro作成

```
Bar 121-152: Outro

Bar 121-128: パッド/リード削除
  → メロディ要素フェードアウト

Bar 129-136: ベースライン削除
  → LPFオートメーションで自然に

Bar 137-144: ハイハット/パーカッション削除
  → リズム簡素化

Bar 145-152: キックのみ
  → 最もクリーンな状態

最後の1-2小節: 自然なフェードアウト or クリーンカット
```

### Step 5: エクスポートと検証

```
エクスポート:

1. Original Mix:
   "Artist - Track_Name (Original Mix).wav"
   44.1 kHz, 16bit, WAV

2. Instrumental:
   ボーカルトラックミュート
   "Artist - Track_Name (Instrumental).wav"

3. Acapella（あれば）:
   インストトラックミュート
   "Artist - Track_Name (Acapella).wav"

4. ステム:
   各バスを個別エクスポート
   "Artist - Track_Name (Stem - Drums).wav"
   "Artist - Track_Name (Stem - Bass).wav"
   "Artist - Track_Name (Stem - Synth).wav"
   "Artist - Track_Name (Stem - Vocals).wav"

検証:
  □ Rekordboxに取り込み → BeatGrid確認
  □ 他のトラックとミックステスト
  □ 複数スピーカーでの聴取テスト
  □ モノ互換性チェック
  □ メタデータの確認
  □ ファイル名の正確さ
```

---

## まとめ

### DJ用トラック制作の核心

1. **32小節 Intro/Outro**: DJミックスの基盤。省略不可。
2. **正確なBPMグリッド**: Warp設定の完璧さがプレイの質を決める。
3. **明確なキー**: ハーモニックミキシング対応で他の曲との調和を実現。
4. **ループポイント**: DJが自由にミックスできるセクション設計。
5. **アカペラ/インスト/ステム**: 多様な使用方法を提供する分離エクスポート。
6. **クリーンな低域**: ミックス時の干渉を最小限に。
7. **適切なラウドネス**: クラブのPAシステムで最適に再生される音圧。
8. **正確なメタデータ**: DJがトラックを正しく認識・管理できる情報。

### DJとして

- セットで使うトラックの構造を理解している
- ミックスポイントがどこにあるか知っている
- Intro/Outroの長さを把握している
- エネルギーの流れを読める

### プロデューサーとして

- DJフレンドリーなトラックを制作できる
- 自分のセットで使える独自の武器を作れる
- 他のDJにもプレイされるトラックを作れる
- エディット/ブートレグで幅を広げられる

### 次のステップ

1. [エディット・リミックス](./edits-remixes.md) - 既存曲を改変する詳細テクニック
2. [Ableton for DJing](./ableton-for-djing.md) - AbletonでDJセットを構築
3. [制作者のためのDJ知識](./production-for-djs.md) - DJ視点を制作に活かす

---

**DJセット用のオリジナルトラックを作成して、セットを唯一無二のものにしましょう！**
