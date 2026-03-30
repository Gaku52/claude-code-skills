# 録音テクニック

完璧な録音を。Audio・MIDI両方の録音方法、オーバーダビング、パンチイン/アウト、コンピングまで完全マスター。

## この章で学ぶこと

- Audio録音（マイク、ライン入力、DI接続）
- MIDI録音（キーボード、パッド、打ち込み）
- Recording Mode（Session vs Arrangement）
- オーバーダビング（重ね録り）
- パンチイン/アウト
- コンピング（テイク選択）
- レイテンシー対策
- モニタリング設定
- マルチトラック録音
- リサンプリング
- Capture MIDI機能
- 録音後の整理・命名規則


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [Quantize（クオンタイズ）](./quantization.md) の内容を理解していること

---

## なぜ録音テクニックが重要なのか

**制作の質を決める:**

```
プロとアマの差:

アマチュア:
録音方法知らない
何度も失敗
音質悪い

結果:
時間の無駄
やり直し多い
モチベーション低下

プロ:
録音設定完璧
1発で決める
高音質

結果:
効率的
クオリティ高い
制作が楽しい

統計:

録音時の設定:
後から直せない
= 最重要

良い録音:
ミックス楽
マスタリング楽
プラグイン効果的

悪い録音:
どんなに処理しても
救えない

格言:
"Garbage in, Garbage out"
→ 入力がゴミなら出力もゴミ
```

### 録音品質がもたらす影響

```
制作工程ごとの影響:

1. 録音 → ミックスへの影響:

良い録音:
EQ: 微調整のみ
コンプ: 軽めでOK
ノイズ処理: 不要
時間: 30分で完了

悪い録音:
EQ: 大幅カット必要
コンプ: 強くかける必要
ノイズ処理: DeNoise必須
時間: 3時間かかる

2. ミックス → マスタリングへの影響:

良いミックス:
ヘッドルーム十分
ダイナミクス自然
マスタリング簡単

悪いミックス:
ヘッドルーム不足
歪み発生
マスタリング困難

3. 全体の制作時間:

正しい録音テクニック:
録音: 1時間
ミックス: 2時間
マスタリング: 30分
合計: 3.5時間

間違った録音テクニック:
録音: 30分（適当）
ミックス: 8時間（修正地獄）
マスタリング: 2時間
合計: 10.5時間
→ 3倍の時間がかかる
```

### 録音の3原則

```
原則1: 適切なレベル

目標: ピーク -12 〜 -6 dB
理由: ヘッドルーム確保
    デジタルクリップ防止
    後処理の余裕

原則2: クリーンな信号

ノイズ最小化:
エアコンOff
ケーブル短く
インピーダンスマッチング

原則3: 正確なモニタリング

聴こえる音 = 録音される音
レイテンシー最小
フィードバックなし
```

---

## Recording Mode

**Session vs Arrangement:**

### Session View録音

```
特徴:

Clip単位:
各Slotに録音

ループ録音:
自動的にループ

リアルタイム:
ライブ演奏向き

用途:

ライブパフォーマンス:
即興

アイデア収集:
素早くCapture

ループ作成:
4-8小節パターン

操作:

1. トラック選択

2. 空のClip Slot選択

3. Session Record ボタン (●)
   または
   トラックのArm (●)

4. 演奏開始

5. 自動的にループ

6. 停止
```

### Session View録音の詳細設定

```
Clip Length設定:

デフォルト:
録音開始 → 停止まで

固定長の場合:
Preferences > Record Warp Launch:

Default Launch Quantization:
1 Bar / 2 Bars / 4 Bars / 8 Bars

効果:
次のClip Slotのトリガーに合わせて
自動的にClip長が決まる

Session録音のクオンタイゼーション:

Launch Quantization:
None: 即座にトリガー
1 Bar: 1小節単位で同期
4 Bars: 4小節単位で同期

推奨:
1 Bar（ライブ向け）
4 Bars（精密な録音向け）

Scene録音:

複数トラックを同時録音:
1. 複数トラックArm
2. Scene トリガー
3. 全トラック同時録音
4. 同時停止

活用:
バンド一発録り
ライブセッション録音
```

### Arrangement View録音

```
特徴:

時間軸:
Linear録音

精密:
タイミング正確

編集向き:
後処理しやすい

用途:

曲構成:
イントロ → ドロップ

重ね録り:
複数テイク

本番録音:
最終形

操作:

1. トラック選択

2. トラックArm (●)

3. 再生ヘッド配置

4. Arrangement Record (●)

5. 演奏

6. 停止
```

### Arrangement View録音の詳細

```
自動録音設定:

Preferences > Record Warp Launch:

Start Transport with Record:
☑ On
→ F9押すだけで再生+録音開始

Pre-Roll:
録音開始前のカウントイン

Exclusive Arm:
☑ On → 1トラックのみArm
☐ Off → 複数トラックArm可

マーカーを活用した録音:

1. 曲の構成にマーカー配置:
   Intro: 1小節目
   Verse: 9小節目
   Chorus: 25小節目
   Drop: 33小節目

2. マーカーにジャンプ:
   次のマーカー: 右矢印
   前のマーカー: 左矢印

3. マーカー位置から録音開始:
   再生ヘッドをマーカーに
   → F9

4. セクション単位で録音:
   効率的

Undo機能:

録音を間違えた場合:
Cmd+Z で即座にUndo
元の状態に戻る

複数回Undo可能:
Cmd+Z を何度も押せる
```

### Session vs Arrangement の使い分け

```
Session View が適切な場合:

1. アイデアスケッチ:
   思いついたメロディを即録音
   ループとして保存
   後でArrangementに配置

2. ライブパフォーマンス:
   リアルタイムで録音
   Clip単位で管理
   即座にトリガー可能

3. ドラムパターン作成:
   4小節ループ録音
   オーバーダブで重ねる
   バリエーション作成

4. ベースライン制作:
   コード進行に合わせて
   ループ録音
   テイク選択

Arrangement View が適切な場合:

1. 本番ボーカル録音:
   曲全体を通して歌う
   セクション単位で管理
   コンピング可能

2. ギターソロ録音:
   特定のセクション
   パンチイン/アウト
   複数テイク

3. 最終ミックスダウン:
   全パートの配置確認
   オートメーション
   最終確認

4. ナレーション/ポッドキャスト:
   時間軸に沿った録音
   編集が容易
   セクション管理
```

---

## Audio録音

**マイク・楽器を録る:**

### 録音準備

```
機材接続:

オーディオインターフェース:
DDJ-FLX4:
MICポート × 2

外部IF:
Focusrite Scarlett等

マイク:
XLR接続
+48V ファンタム電源 (コンデンサー)

ギター/ベース:
1/4" TRS ケーブル

Ableton設定:

Preferences > Audio:

Audio Device:
DDJ-FLX4 (または外部IF)

Sample Rate:
44.1 kHz (推奨)

Buffer Size:
128 samples (録音時)
512-1024 samples (ミックス時)

Input Config:
入力チャンネル確認
Mono / Stereo設定

トラック設定:

Audio Track作成:
Cmd+T

Input:
Ext. In → 1 (Mic 1)
または
Ext. In → 1/2 (Stereo)

Monitor:
In (常に聴こえる)
Auto (録音中のみ)

Arm:
● On (録音可能)
```

### マイクの種類と選び方

```
ダイナミックマイク:

特徴:
ファンタム電源不要
丈夫・耐久性高い
感度低め

代表機種:
Shure SM58 (ボーカル定番)
Shure SM57 (楽器定番)
Sennheiser e835

用途:
ライブボーカル
ギターアンプ
ドラム (スネア、タム)
大音量ソース

メリット:
周囲のノイズ拾いにくい
フィードバック起きにくい
予算に優しい

デメリット:
繊細な表現は苦手
高域のエアー感少なめ

コンデンサーマイク:

特徴:
ファンタム電源必要 (+48V)
感度高い
周波数レンジ広い

代表機種:
RODE NT1-A
Audio-Technica AT2020
AKG C214
Neumann U87 (高級)

用途:
スタジオボーカル
アコースティックギター
ストリングス
ドラムオーバーヘッド

メリット:
繊細な音を捉える
高域の空気感
プロフェッショナルな音

デメリット:
環境ノイズ拾いやすい
衝撃に弱い
価格高め

リボンマイク:

特徴:
ファンタム電源不要（危険な場合あり）
温かみのあるサウンド
双指向性

代表機種:
Royer R-121
AEA R84

用途:
ギターアンプ
ブラス
クラシック楽器

指向性パターン:

カーディオイド (単一指向性):
前方の音を拾う
最も一般的
ボーカル録音に最適

オムニ (全指向性):
全方向の音を拾う
部屋の響きも録音
アンビエント録音に

フィギュアエイト (双指向性):
前後の音を拾う
デュエット録音
M/S録音に

スーパーカーディオイド:
カーディオイドより狭い
より指向性が強い
ノイズの多い環境で有効
```

### マイキングテクニック

```
ボーカル録音:

マイクとの距離:
15-30 cm（拳1-2個分）

ポップガード:
マイクの5-10 cm手前
パ行・バ行の破裂音防止

リフレクションフィルター:
マイク背面に設置
部屋の反射音を低減

マイクの高さ:
口の高さに合わせる
やや見上げる角度

近接効果:
マイクに近づく → 低域増加
離れる → 自然な音

ギターアンプ録音:

オンマイク:
スピーカーコーン中心: 明るい音
スピーカーコーン端: 暗い音
距離: 2-5 cm

オフマイク:
距離: 30 cm - 1 m
部屋の響きが加わる

ダブルマイク:
SM57 (オンマイク) + コンデンサー (オフマイク)
2つの音をブレンド
位相に注意

アコースティックギター録音:

マイク位置:
12フレット付近: バランス良い
サウンドホール: 低域豊か（ブーミーになりやすい）
ブリッジ付近: 高域・アタック

推奨:
12フレットに向けて
20-30 cm の距離
やや下向き

ステレオ録音:
XY方式: 2本を90度クロス → モノ互換性良い
AB方式: 2本を離す → 広がりあるステレオ
ORTF方式: 17cm離して110度 → 自然なステレオ

ドラム録音の基本:

最小構成 (2マイク):
キック: ダイナミックマイク (AKG D112等)
オーバーヘッド: コンデンサーマイク × 1 (モノ)

3マイク:
キック + スネア + オーバーヘッド

4マイク:
キック + スネア + OH (ステレオ)

フル:
キック + スネア(Top/Bottom) + タム×3 + OH(ステレオ) + ハイハット + ルーム
→ 10本以上

注意:
位相問題に常に注意
3:1ルール（マイク間距離は音源距離の3倍以上）
```

### 録音実行

```
Step 1: レベル調整 (5分)

1. トラックArm: ●

2. Monitor: In

3. 演奏/歌う:
レベルメーター確認

4. ゲイン調整:
IF のGainつまみ

目標:
ピーク -12 〜 -6 dB
赤色(Clip)絶対ダメ

5. 確認:
ヘッドフォンで音質

Step 2: カウントイン設定 (2分)

Preferences > Record Warp Launch:

Count In: 1 Bar (推奨)
または
2 Bars, 4 Bars

Metronome:
☑ On

効果:
録音前に1小節カウント
準備できる

Step 3: 録音 (本番)

Session View:

1. Clip Slot選択

2. Cmd+F9 (Start Record)

3. カウントイン待つ

4. 演奏

5. Space (Stop)

Arrangement View:

1. 再生ヘッド配置

2. F9 (Arrangement Record)

3. カウントイン

4. 演奏

5. Space

Step 4: 確認

1. Arm: Off (誤録音防止)

2. 再生:
Space

3. 波形確認:
クリップなし
ノイズなし

4. やり直し:
Delete → 再録音
```

### 24bit vs 16bit録音

```
ビット深度の違い:

16bit:
ダイナミックレンジ: 96 dB
CD品質
最終マスター向け

24bit:
ダイナミックレンジ: 144 dB
録音品質
制作中は常にこれ

32bit float:
ダイナミックレンジ: 事実上無限
最新IF対応
クリップしない

推奨:
録音時: 24bit
ミックス時: 32bit float (DAW内部)
書き出し: 16bit / 24bit

設定:
Preferences > Record Warp Launch:
Bit Depth: 24

理由:
24bitの方がノイズフロア低い
ヘッドルーム大きい
後処理に有利
ファイルサイズは1.5倍だが問題なし
```

### サンプルレートの選択

```
一般的な選択肢:

44.1 kHz:
CD品質
音楽制作の標準
CPU負荷最小
推奨

48 kHz:
映像制作の標準
動画用音声はこれ
YouTube向け

96 kHz:
ハイレゾ録音
オーバーサンプリング効果
CPU負荷2倍

192 kHz:
超ハイレゾ
アーカイブ用
実用的メリット少ない

推奨:
音楽制作: 44.1 kHz
映像音声: 48 kHz
高品質アーカイブ: 96 kHz

注意:
サンプルレートは統一すること
プロジェクト途中で変更しない
高サンプルレート = バッファーサイズ実質半分
→ レイテンシーに影響
```

---

## MIDI録音

**キーボード・打ち込み:**

### MIDIキーボード接続

```
接続方法:

USB MIDI:
MIDIキーボード → Mac/PC
USB接続のみ

5pin MIDI:
キーボード → IF → Mac/PC
古い機材

DDJ-FLX4:
MIDI入力なし
→ 別途MIDIキーボード必要

Bluetooth MIDI:
ワイヤレス接続
レイテンシー注意

Ableton認識:

Preferences > Link Tempo MIDI:

MIDI Ports:

Input: Your MIDI Keyboard
Track: ☑ On
Remote: ☐ Off

トラック設定:

MIDI Track作成:
Cmd+T

音源読み込み:
Wavetable, Operator等

Input:
All Ins → All Channels

Monitor:
In (常に音が鳴る)

Arm:
● On
```

### MIDIキーボードの種類

```
ミニキーボード (25鍵):

代表機種:
Akai MPK Mini
Arturia MiniLab
Novation Launchkey Mini

メリット:
コンパクト
持ち運びやすい
安価

デメリット:
鍵盤小さい
演奏性限定的

用途:
トラベル制作
メロディ入力
ビートメイク

49鍵キーボード:

代表機種:
Arturia KeyLab 49
Novation Launchkey 49
Native Instruments Komplete Kontrol A49

メリット:
両手演奏可能
バランス良い
デスクに収まる

用途:
メインキーボード
シンセ演奏
コード入力

61鍵/88鍵:

代表機種:
Arturia KeyLab 61/88
Native Instruments S61/S88
Kawai VPC1 (88鍵)

メリット:
フルサイズ
ピアノ演奏可能
タッチ/アフタータッチ

用途:
ピアニスト
クラシック/ジャズ
リアルな演奏表現

パッドコントローラー:

代表機種:
Akai MPC
Native Instruments Maschine
Ableton Push

メリット:
ドラム打ち込み最適
フィンガードラミング
Clip制御

用途:
ビートメイク
サンプルトリガー
ライブパフォーマンス
```

### MIDI録音実行

```
リアルタイム録音:

1. トラックArm

2. Monitor: In

3. 音色確認:
鍵盤弾いて確認

4. F9 (Record)

5. カウントイン

6. 演奏:
鍵盤を弾く

7. Space (Stop)

結果:
MIDIノートが記録される

ステップ録音 (推奨):

1. 空のMIDI Clip作成

2. Clip View表示

3. ピアノロールで:
ダブルクリック入力

メリット:
タイミング完璧
Quantize不要

Recording Quantization:

Preferences > Record Warp Launch:

MIDI Record Quantization:
1/16 (推奨)
または
None

効果:
録音中にリアルタイムQuantize
ズレ自動修正

注意:
機械的になる可能性
```

### MIDI録音の高度なテクニック

```
Velocity（ベロシティ）の活用:

ベロシティとは:
鍵盤を押す強さ
0-127の値

録音時の注意:
強弱をつけて演奏
→ 表現力が増す

後から編集:
Clip View > Notes
ベロシティレーンで調整

一定にしたい場合:
Velocity MIDI Effect挿入
Out Hi/Lo を同じ値に

Aftertouch（アフタータッチ）:

鍵盤を押した後に更に押し込む
→ 連続的なコントロール

用途:
ビブラート
フィルター開閉
音量変化

対応キーボード必要:
全てのキーボードが対応ではない

CC（コントロールチェンジ）録音:

CC1: モジュレーションホイール
CC7: ボリューム
CC10: パン
CC64: サスティンペダル
CC74: フィルターカットオフ

録音方法:
1. MIDIトラックArm
2. 演奏しながらCCを動かす
3. オートメーションとして記録

後から編集:
Envelope > MIDI Ctrl
各CC番号のエンベロープ

Pitch Bend録音:

ピッチベンドホイール:
-8192 〜 +8191の値

録音方法:
リアルタイムで演奏
自動的に記録

編集:
Envelope > Pitch Bend

MPE (MIDI Polyphonic Expression):

対応コントローラー:
Roli Seaboard
Sensel Morph
Linnstrument

特徴:
ノートごとに独立した表現
Press: 圧力
Slide: 水平移動
Glide: ピッチベンド

Ableton対応:
Live 11以降
MPE対応音源で使用可能
```

### Capture MIDI機能

```
Capture MIDIとは:

Ableton Live 11+の機能:
録音ボタンを押さなくても
演奏した内容を遡って記録

仕組み:
Armされたトラックの入力を
常にバッファーに保存

使い方:

1. MIDIトラックArm
2. 自由に演奏（録音ボタン押さない）
3. 良いフレーズが弾けた！
4. Capture MIDIボタン (★) を押す
   またはショートカット

結果:
直前の演奏がClipとして保存

メリット:
プレッシャーなし
自然な演奏が録れる
「録音中」の緊張がない
偶然の名フレーズを逃さない

活用シーン:

ウォームアップ中:
練習しているうちに良いフレーズ
→ Captureで保存

即興セッション:
自由に弾いている時
→ Captureで記録

アイデアスケッチ:
曲のアイデアを探っている時
→ 良いのが来たらCapture

テンポ検出:
Capture時にテンポも自動検出
プロジェクトのテンポに反映
```

---

## オーバーダビング

**重ね録り:**

### オーバーダビングとは

```
定義:
既存の録音に追加録音

用途:

複数テイク:
ドラム → ベース → ギター

ハモリ:
メインボーカル + ハモリ

パート追加:
ストリングス追加

Session View:

同じClip Slot:
上書きされる (デフォルト)

別のClip Slot:
別トラックに録音

Arrangement View:

新規Clip:
自動的に作成

既存Clip上:
上書き (注意)
```

### オーバーダビング実行

```
方法1: 別トラック (推奨)

1. 既存トラック再生

2. 新規トラック作成

3. 新規トラックArm

4. 録音開始:
既存音を聴きながら

5. 完了

メリット:
元のテイク保存
安全

方法2: 同じトラック

Session View:

別のClip Slot使用:
縦に並ぶ

Arrangement View:

Preferences > Record Warp Launch:

Create Fades: ☑
Exclusive Arm: ☐

録音:
既存Clipの隣に

方法3: MIDI重ね録り

1. MIDI Clip選択

2. Arm + Record

3. 演奏:
既存ノートに追加

4. 結果:
ノートが追加される

注意:
上書きではなく追加
```

### MIDI オーバーダブの詳細

```
Session ViewでのMIDIオーバーダブ:

設定:
Session Record ボタン: ● (点灯)
Overdub ボタン: + (点灯)

操作:
1. 既存MIDI Clipを再生
2. Overdubボタン有効
3. 演奏:
   既存ノートに新しいノートが追加

活用例:

ドラムパターン構築:

ループ1周目: キック
ループ2周目: スネア追加
ループ3周目: ハイハット追加
ループ4周目: パーカッション追加

→ 段階的にパターン完成

コード重ね:

ループ1: ルート音
ループ2: 3rd追加
ループ3: 5th追加
ループ4: 7th追加

→ 複雑なコードも確実に

消去しながらオーバーダブ:

既存ノートを消したい場合:
同じノートを演奏 → 上書き消去
（MIDI Arrangement Overdub設定による）

Arrangement ViewでのMIDIオーバーダブ:

MIDI Arrangement Overdub:
○ On → 既存ノートに追加
○ Off → 既存ノートを置き換え

使い分け:
追加: ハモリ、パート追加
置換: 修正、やり直し
```

### Audio オーバーダブの注意点

```
モニターバランス:

問題:
既存トラックの音と
新しく録音する音のバランス

解決:
1. 既存トラック:
   フェーダーで適切な音量

2. 新規トラック:
   Monitor: In
   ヘッドフォンで確認

3. Cue/Solo:
   ヘッドフォンミックス

クリック（メトロノーム）:

問題:
マイクでクリック音を拾う

解決:
1. 密閉型ヘッドフォン使用
2. クリック音量を下げる
3. ヘッドフォンからの音漏れ確認

位相の問題:

問題:
同じ音源を複数マイクで録音
→ 位相が打ち消し合う

解決:
1. マイク位置の確認
2. Utility > Phase Invert
3. 波形を目視確認

クロストーク:

問題:
隣のマイクに音が漏れる

解決:
1. マイク間の距離確保
2. 指向性の活用
3. ゲーティング（後処理）
```

---

## パンチイン/アウト

**部分的な録音:**

### パンチイン/アウトとは

```
定義:
指定範囲のみ録音

用途:

ミス修正:
1箇所だけやり直し

部分差し替え:
サビだけ録り直し

効率化:
全体録音不要

Ableton での方法:

手動:
Recordボタン手動On/Off

Loop範囲:
Loop内のみ録音
```

### 手動パンチイン

```
Arrangement View:

Step 1: 準備

1. 再生ヘッドを配置:
録音開始位置の少し前

2. トラックArm

3. 再生開始:
Space

Step 2: パンチイン

1. 録音開始位置で:
F9 (Record)

2. 演奏

3. 録音終了位置で:
F9 (Record停止)

4. 完了

コツ:

準備時間:
4-8拍前から再生

タイミング:
F9を正確に押す

練習:
何度かリハーサル
```

### 自動パンチイン/アウト

```
Loop範囲を使った自動パンチ:

設定:

1. パンチイン位置にLoop Start設定
2. パンチアウト位置にLoop End設定
3. Loop: ☑ On

操作:

1. 再生ヘッドをLoop Start前に配置
   （4拍前など）

2. トラックArm

3. F9 (Record + Play)

4. Loop Startで自動的に録音開始

5. Loop Endで自動的に録音終了

6. Loop先頭に戻る

7. 納得いくまで繰り返し

メリット:
F9のタイミングを気にしない
何度でもやり直せる
特定セクションに集中できる

活用シーン:

難しいフレーズの録音:
同じ4小節を何度もループ
ベストテイクが録れるまで

ボーカルのサビ:
サビ部分だけLoop
集中して歌う

ギターソロ:
ソロセクションをLoop
何度もトライ

プロの手法:

ループ録音 → 複数テイク → コンピング
= 最も効率的なワークフロー
```

### Loop録音

```
設定:

1. Loop Range設定:
録音したい範囲

2. Loop: ☑ On

3. トラックArm

録音:

1. F9 (Record)

2. Loopが回る:
何度も録音可能

3. 納得いくまで:
ループ繰り返し

4. Stop

Session View では:

Clip Loop Lengthが範囲
自動的にループ録音

用途:

ドラムループ:
4小節を完璧に

ベースライン:
8小節録音

ボーカルフレーズ:
サビ部分
```

---

## コンピング

**ベストテイク選択:**

### コンピングとは

```
定義:
複数テイクから良い部分を選択

例:

テイク1: イントロ◎, サビ△
テイク2: イントロ△, サビ◎
テイク3: イントロ○, サビ○

→ コンピング:
イントロ:テイク1
サビ:テイク2

結果:
完璧な1トラック

Abletonでの方法:

Take Lanes (Live 11+):
複数テイクを管理

手動編集:
Cmd+E で分割・選択
```

### コンピング実行

```
方法1: Take Lanes (Live 11+)

1. トラックArm

2. Loop録音:
複数回演奏

3. Take Lanes表示:
トラック左の [▼]

4. 各テイク表示:
縦に並ぶ

5. 良い部分選択:
クリックして有効化

6. Flatten:
1つのClipに統合

方法2: 手動編集 (Live 10以前)

1. 複数テイク録音:
別々のClipに

2. 良い部分を探す:
各Clip再生

3. 必要部分を切り出し:
Cmd+E で分割

4. 1つのトラックに配置:
コピー&ペースト

5. Consolidate:
Cmd+J で統合

方法3: 別トラック比較

1. テイクごとに別トラック

2. Solo/Mute で比較:
S / M キー

3. ベスト選択

4. 不要トラック削除
```

### Take Lanesの詳細操作

```
Take Lanes表示:

1. トラックヘッダーの [▼] クリック
   → Take Lanesが展開

2. 各レーンに録音テイクが表示:
   Take 1: [=========]
   Take 2: [=========]
   Take 3: [=========]

テイク選択:

1. 聴きたいレーンをクリック:
   → そのレーンがアクティブに

2. 部分選択:
   レーン内をドラッグ
   → 選択範囲のみアクティブ

3. 複数レーンから選択:
   Take 1: [====]............
   Take 2: ......[====]......
   Take 3: ............[====]
   → ベスト部分を組み合わせ

クロスフェード:

テイク切り替わり部分:
自動的にクロスフェード適用
→ つなぎ目がスムーズ

手動調整:
クロスフェードのハンドルをドラッグ
→ 長さ調整可能

Flatten（統合）:

コンピング完了後:
右クリック > Flatten
→ 1つのClipに統合
→ Take Lanesが消える

注意:
Flatten前に確認
戻すにはCmd+Z

Tips:

1. 全テイク通して聴く:
   まず全体像を把握

2. セクション単位で選択:
   イントロ、Aメロ、Bメロ、サビ

3. つなぎ目を確認:
   テイク切り替わり部分を再生
   不自然さがないか

4. 必要なら微調整:
   Clip Editorでフェード調整
```

### コンピングのプロのテクニック

```
効率的なコンピングワークフロー:

1. 事前準備:
   - テイク数を決める（3-5テイク）
   - セクションごとに区切る
   - マーカーを配置

2. 録音フェーズ:
   - 全テイクを一気に録音
   - 休憩を入れすぎない
   - 声/指の状態を保つ

3. 選択フェーズ:
   - 全テイクを通して聴く
   - メモを取る（テイク2のサビ◎等）
   - 客観的に判断

4. 組み立てフェーズ:
   - ベスト部分を選択
   - つなぎ目を確認
   - クロスフェード調整

5. 最終確認:
   - 通して再生
   - 不自然な箇所がないか
   - Flatten

コンピングの判断基準:

ピッチ:
音程が正確か
ピッチ補正が少なくて済むか

タイミング:
リズムが正確か
グルーヴ感があるか

表現力:
感情が込められているか
ダイナミクスが適切か

音質:
ノイズがないか
歪みがないか
マイクポジションが安定か

一貫性:
テイク間の音色が統一されているか
つなぎ目が自然か
```

---

## モニタリング設定

**録音中の音を聴く:**

### Monitor設定

```
3つのモード:

In (常時モニタリング):

機能:
常に入力音が聴こえる

用途:
Audio録音
MIDI演奏

注意:
レイテンシー感じる

Auto (自動):

機能:
Arm時のみ聴こえる
再生中は聴こえない

用途:
ほとんどの場合

推奨:
これが標準

Off (モニタリングなし):

機能:
入力音聴こえない

用途:
再生専用トラック

設定場所:

トラック: In/Out セクション
Monitor: In / Auto / Off

推奨設定:

Audio録音: Auto
MIDI録音: In
再生のみ: Off
```

### Direct Monitoring

```
オーディオIF の機能:

Direct Monitoring:
IFで直接モニタリング
DAWを通さない

メリット:

レイテンシーゼロ:
遅延なし

CPU負荷なし:
IFで処理

デメリット:

エフェクト聴けない:
DAWのプラグイン無効

設定:

オーディオIF:
Direct Mon: On

Ableton:
Monitor: Off (重複防止)

または:

IF: Direct Mon Off
Ableton: Monitor In

推奨:

レイテンシー問題あり:
Direct Mon: On

問題なし:
Ableton Monitor: In
→ エフェクト使える
```

### Cue/Preview設定

```
ヘッドフォンミックス:

問題:
ボーカリストが自分の声と
カラオケのバランスを調整したい

解決:
Cue Output使用

設定:

1. Preferences > Audio:
   Output Config:
   1/2: Master
   3/4: Cue Out

2. Master欄:
   Cue Out: 3/4

3. ヘッドフォンを3/4に接続

4. Soloモード: Cue
   (PFL: Pre-Fader Listen)

使い方:

1. ボーカリストのヘッドフォン:
   Cue Outに接続

2. カラオケトラック:
   Soloボタンで送る

3. ボーカルトラック:
   Monitor: In

4. ボーカリスト:
   Cueでカラオケ + 自分の声を聴く

5. エンジニア:
   Masterで全体を聴く

独立した音量調整:
ボーカリストが快適に歌える
```

### モニタリングのトラブルシューティング

```
問題1: フィードバック（ハウリング）

症状:
キーン！という甲高い音

原因:
マイクがスピーカーの音を拾う
→ 無限ループ

解決:
1. Monitor: Off にする
2. ヘッドフォンを使用
3. スピーカー音量を下げる
4. マイクの指向性を活用

問題2: 二重に聴こえる

症状:
自分の声がエコーのように二重

原因:
Direct Monitoring + DAWモニタリング
両方ONになっている

解決:
どちらか一方をOff:
IF: Direct Mon Off → Ableton: In
IF: Direct Mon On → Ableton: Off

問題3: 音が聴こえない

チェックリスト:
1. トラックArm: ● 点灯？
2. Monitor: In (またはAuto)?
3. ヘッドフォン/スピーカー接続？
4. Master音量: 0以上？
5. トラックフェーダー: 0以上？
6. Input設定: 正しいチャンネル？
7. IF のゲイン: 0以上？

問題4: ノイズが聴こえる

チェックリスト:
1. ケーブル確認
2. IF のゲインが高すぎ？
3. USBハブ経由？→ 直接接続
4. 電源ノイズ？→ アース確認
5. 周辺機器のノイズ？→ 離す
```

---

## レイテンシー対策

**遅延を最小化:**

### レイテンシーとは

```
定義:
入力から出力までの遅延

原因:

ADコンバーター:
アナログ → デジタル変換

バッファーサイズ:
音声処理のバッファー

DAコンバーター:
デジタル → アナログ変換

体感:

10 ms以下: 感じない
10-20 ms: わずかに感じる
20-50 ms: 明確に感じる
50 ms以上: 演奏不可能

測定:

Preferences > Audio:

Overall Latency:
Input Latency + Output Latency
例: 5.8 ms + 5.8 ms = 11.6 ms
```

### バッファーサイズ調整

```
設定:

Preferences > Audio:

Buffer Size:
32 samples (最小)
64 samples (小)
128 samples (録音推奨)
256 samples (通常)
512 samples (ミックス)
1024 samples (負荷大時)
2048 samples (マスタリング)

計算:

レイテンシー (ms) = バッファーサイズ / サンプルレート × 1000

44.1kHz, 128 samples:
128 / 44100 × 1000 = 2.9 ms

44.1kHz, 512 samples:
512 / 44100 × 1000 = 11.6 ms

推奨設定:

録音時:
128 samples
= レイテンシー小

ミックス時:
512-1024 samples
= CPU余裕

注意:

小さすぎ:
CPU負荷高い
ノイズ・途切れ

大きすぎ:
レイテンシー大
演奏しにくい
```

### レイテンシー補正

```
Ableton の自動補正:

Delay Compensation:
Options > Delay Compensation: ☑ On

仕組み:
各トラックのレイテンシーを自動測定
全トラックを揃える
→ 位相ズレなし

注意:
録音時にOff推奨の場合あり
（リアルタイム演奏に影響）

Reduced Latency When Monitoring:
Options > Reduced Latency When Monitoring: ☑

効果:
Arm中のトラックのレイテンシーを最小化
プラグインの一部をバイパス

手動レイテンシー補正:

Track Delay:
各トラックに手動で遅延設定

正の値: トラックを遅らせる
負の値: トラックを早める

用途:
外部ハードウェアのレイテンシー補正
特定のプラグインの遅延補正

External Audio Effectデバイス:
Hardware Latency:
外部機器のラウンドトリップ遅延を設定
自動測定ボタンあり
```

### レイテンシーの種類と対策まとめ

```
1. バッファーレイテンシー:
原因: Buffer Size設定
対策: Buffer Sizeを小さく（128推奨）

2. プラグインレイテンシー:
原因: リニアフェーズEQ、Lookaheadコンプ等
対策: Reduced Latency When Monitoring

3. AD/DAレイテンシー:
原因: オーディオIFのコンバーター
対策: 高品質IF使用（0.5-2ms程度）

4. ドライバーレイテンシー:
原因: OSのオーディオドライバー
対策: CoreAudio(Mac)使用、ASIO(Win)使用

5. USBレイテンシー:
原因: USB接続の遅延
対策: USB 3.0以上、ハブ不使用

トータルレイテンシーの目安:

理想: 5ms以下
良好: 10ms以下
許容: 20ms以下
問題あり: 20ms以上

プロの現場:
3-5ms を目標
DirectMonitoring + 低Buffer
```

### Driver Errorハンドリング

```
問題:

バッファーサイズ小さい
+ CPU負荷高い
= Driver Error (CPU Overload)

対策:

1. バッファーサイズ増やす:
128 → 256

2. Freeze Track:
重いトラックをFreeze

3. プラグイン整理:
不要なものOff

4. Sample Rate確認:
96kHz → 44.1kHz

5. バックグラウンドアプリ:
Chrome等を終了

緊急時:

Preferences > CPU:

Multi-Core: ☑
Plug-In Load: 50%
```

### CPU負荷の管理

```
CPU Meterの見方:

Ableton右上のCPU表示:

Current: 現在のCPU使用率
Average: 平均CPU使用率

目安:
30%以下: 安全
50%前後: 注意
70%以上: 危険
90%以上: クラッシュリスク

CPU負荷を減らす方法:

1. Freeze Track:
右クリック > Freeze Track
→ トラックをオーディオに変換
→ CPU解放
→ 編集不可（Unfreezeで戻る）

2. Flatten:
Freeze後に右クリック > Flatten
→ 完全にオーディオ化
→ プラグイン削除
→ CPU大幅削減

3. プラグインの整理:
未使用プラグイン: Off
重いリバーブ: Send/Returnに1つだけ
リニアフェーズEQ: ミニマムフェーズに

4. サンプルレート下げ:
96kHz → 44.1kHz
CPU負荷半減

5. オーバーサンプリング:
Hi-Quality Mode: Off（録音時）
後でOn（ミックス時）

6. バックグラウンドプロセス:
Chrome: 終了
Time Machine: 一時停止
Spotlight: インデックス除外
Wi-Fi: 必要なければOff

7. 電源設定 (Mac):
システム環境設定 > バッテリー:
低電力モード: Off
```

---

## リサンプリング

**内部録音テクニック:**

### リサンプリングとは

```
定義:
Abletonの出力を別トラックに録音

仕組み:
Master Out → Audio Track に録音

用途:
エフェクト込みで録音
ライブ演奏のキャプチャ
DJミックスの録音
複数トラックを1つに

設定:

1. 新規Audio Track作成: Cmd+T

2. Input: "Resampling"

3. Monitor: Off (フィードバック防止)

4. Arm: ●

5. 録音開始: F9

6. 停止: Space

結果:
全トラックの出力がAudioとして記録
```

### リサンプリングの活用法

```
1. エフェクト処理の録音:

シナリオ:
複雑なエフェクトチェーン
→ リアルタイムで動かす
→ リサンプリングで録音

手順:
1. エフェクトを設定
2. リサンプリングトラック作成
3. 録音しながらエフェクトを操作
4. 結果をオーディオとして保存

メリット:
CPU解放
ユニークなサウンド
偶然の産物を記録

2. ライブDJミックスの録音:

手順:
1. DJミックスを設定
2. リサンプリングトラック
3. ミックスしながら録音
4. 完成したミックスをエクスポート

3. バウンス (Bounce in Place):

Abletonでの方法:
1. クリップ/範囲を選択
2. リサンプリングトラックで録音
3. 元トラックMute/Delete

用途:
VSTを少しずつオーディオ化
プロジェクトの軽量化

4. サウンドデザイン:

手順:
1. 複数のシンセを重ねる
2. フィルター・エフェクトを追加
3. リサンプリングで録音
4. 録音結果をさらに加工

= レイヤードサウンドの作成
```

### 特定トラックのリサンプリング

```
Master以外のリサンプリング:

方法1: Send/Return使用

1. Returnトラック作成
2. 録音したいトラックのSendを上げる
3. 新規Audio Trackを作成
4. Input: Return A (等)
5. 録音

方法2: グループ使用

1. 録音したいトラックをグループ化
2. 新規Audio Track作成
3. Input: グループトラック名
4. 録音

方法3: トラック間ルーティング

1. 新規Audio Track作成
2. Input: 特定のトラック名を選択
3. Post FX / Pre FX 選択可能
4. 録音

Post FX:
エフェクト後の音を録音

Pre FX:
エフェクト前の音を録音

Pre Mixer:
フェーダー・パン前の音を録音

活用:
個別トラックのバウンス
エフェクト処理済み音の保存
パラレルプロセスの録音
```

---

## マルチトラック録音

**複数トラック同時録音:**

### マルチトラック録音の設定

```
必要機材:

オーディオIF:
複数入力対応
(例: Focusrite Scarlett 18i20)

マイク:
必要な本数

ケーブル:
XLR × 本数分

設定:

1. Preferences > Audio:
   Input Config:
   全入力チャンネルを有効化

2. Audio Track作成:
   必要なトラック数

3. 各トラックの入力設定:
   Track 1: Ext. In 1 (Kick)
   Track 2: Ext. In 2 (Snare)
   Track 3: Ext. In 3 (Hi-Hat)
   Track 4: Ext. In 4/5 (OH L/R)

4. 各トラックArm:
   Exclusive Arm: ☐ Off
   → 複数トラック同時Arm可能

5. レベル調整:
   各チャンネル個別に

6. 録音開始:
   F9 → 全トラック同時録音
```

### バンド一発録りのワークフロー

```
準備 (30分):

1. 部屋の準備:
   アンプ配置
   仕切り（ゴボ）設置
   マイク配置

2. 接続:
   全マイク → IF
   チャンネル確認

3. レベル調整:
   各楽器を個別にチェック
   ゲイン設定
   ピーク確認

4. ヘッドフォンミックス:
   各メンバーのモニタリング
   Cue Out活用

録音 (本番):

1. テスト録音:
   1コーラス通し
   レベル最終確認
   問題箇所修正

2. 本番テイク:
   テイク1: 全体のグルーヴ重視
   テイク2: 細部の修正
   テイク3: ベストを目指す

3. 確認:
   各テイクを再生
   各トラック個別チェック

後処理:

1. テイク選択/コンピング
2. 各トラックの波形確認
3. 不要部分のカット
4. 位相チェック
5. Consolidate
```

---

## 録音後の整理

**効率的なワークフロー:**

### ファイル命名規則

```
推奨命名規則:

[曲名]_[パート]_[テイク番号]_[日付]

例:
MyTrack_Vocal_T01_20260222
MyTrack_Vocal_T02_20260222
MyTrack_Guitar_T01_20260222

Ableton内での管理:

Clip名変更:
Clip選択 → Cmd+R → 名前入力

トラック名変更:
トラックヘッダーダブルクリック

色分け:
右クリック > カラー変更

推奨カラースキーム:
赤: ドラム/パーカッション
青: ベース
緑: ギター/キーボード
黄: ボーカル
紫: シンセ
オレンジ: エフェクト/FX
```

### プロジェクト管理

```
フォルダ構造:

Music/
├── Projects/
│   ├── MyTrack/
│   │   ├── MyTrack.als
│   │   ├── Samples/
│   │   │   ├── Recorded/
│   │   │   ├── Imported/
│   │   │   └── Resampled/
│   │   └── Backup/

Collect All and Save:

File > Collect All and Save:
全ファイルをプロジェクトフォルダに集約
外部参照をなくす
バックアップ容易

定期バックアップ:
1. プロジェクト保存: Cmd+S
2. 別名保存: Cmd+Shift+S
   → バージョン管理
3. 外部バックアップ: 定期的に

バージョン管理:

MyTrack_v01.als (初期)
MyTrack_v02.als (ボーカル録音後)
MyTrack_v03.als (ミックス開始)
MyTrack_v04.als (最終版)

→ いつでも前のバージョンに戻れる
```

### 録音後のチェックリスト

```
□ 全テイクの確認
  - 波形にクリップがないか
  - ノイズが入っていないか
  - レベルが適切か

□ ベストテイクの選定
  - コンピング（必要なら）
  - テイクにメモを付ける

□ 不要テイクの整理
  - 使わないClipの削除
  - または非表示

□ クリップ名の整理
  - わかりやすい名前に
  - テイク番号

□ Consolidate
  - 編集済みClipを統合
  - Cmd+J

□ プロジェクト保存
  - Cmd+S
  - バージョン保存

□ バックアップ
  - Collect All and Save
  - 外部ドライブに保存
```

---

## 実践: ボーカル録音

**完璧な録音:**

### Step 1: 準備 (10分)

```
機材:

マイク:
コンデンサーマイク推奨
(RODE NT1-A, Audio-Technica AT2020等)

オーディオIF:
ファンタム電源 +48V

ポップガード:
パ行・バ行対策

ヘッドフォン:
密閉型

接続:

1. マイク → IF (XLR)

2. +48V: On

3. IF → Mac/PC (USB)

4. ヘッドフォン → IF

Ableton設定:

1. Audio Track作成

2. Input: Ext. In 1

3. Monitor: In

4. Arm: ●

5. レベル確認:
ゲイン調整
ピーク -12 〜 -6 dB
```

### Step 2: 録音環境の最適化

```
部屋の処理:

問題:
部屋の反射音 → 録音品質低下

対策:

1. リフレクションフィルター:
   マイク背面に設置
   反射音を吸収

2. 毛布/カーテン:
   壁からの反射を減らす
   窓を覆う

3. マットレス:
   即席の吸音材
   マイク周辺に配置

4. クローゼット:
   洋服が吸音材代わり
   狭い空間で録音

5. 隅を避ける:
   部屋の角は低域が溜まる
   部屋の中央付近が理想

ノイズ対策:

1. エアコン: Off
2. 冷蔵庫: 確認
3. PC ファン: 離す
4. スマホ: 機内モード
5. 窓: 閉める
6. 家電: 不要なものOff

ボーカリストの準備:

1. 水分補給: 常温の水
2. 喉のウォームアップ: 5-10分
3. 歌詞の確認: 暗記推奨
4. ヘッドフォンフィット: 片耳外しもOK
5. マイク距離: 15-30cm
6. ポップガード位置: マイクから5-10cm
```

### Step 3: 録音 (15分)

```
1. カウントイン設定:
   2 Bars

2. Metronome: On
   Click音量調整

3. リハーサル:
   1-2回通し練習

4. 本番録音:
   F9 → 歌う → Space

5. 確認:
   再生して聴く

6. 必要なら:
   テイク2, 3
```

### Step 4: コンピング (10分)

```
1. ベストテイク選択:
   各テイク再生

2. 良い部分切り出し:
   Cmd+E で分割

3. 配置:
   1つのトラックに

4. クロスフェード:
   つなぎ目を滑らかに

5. Consolidate:
   Cmd+J

6. 完成
```

### Step 5: ボーカル録音後の処理

```
即座に行う処理:

1. ノイズゲート:
   静かな部分のノイズ除去
   Gate: Threshold -40dB付近

2. Clip Gain調整:
   テイク間の音量差を均一化
   波形の大きさを揃える

3. ブレス処理:
   不要なブレス音をカット
   または音量を下げる

4. ポップ/クリック除去:
   破裂音があれば処理
   フェードイン追加

後で行う処理（ミックス段階）:

1. EQ:
   ハイパスフィルター: 80Hz以下カット
   プレゼンス: 2-5kHz ブースト
   エアー: 10kHz+ シェルフブースト

2. コンプレッション:
   Ratio: 3:1 〜 4:1
   Attack: 10-30ms
   Release: Auto

3. ディエッサー:
   サ行の刺さりを抑制
   5-8kHz付近

4. リバーブ:
   Send/Returnで
   適度な空間感
```

---

## 実践: 楽器録音

### ギターDI録音

```
DI (Direct Input) とは:
ギターを直接IFに接続
アンプなしで録音

メリット:
ノイズ少ない
後からアンプシミュで音作り
何度でもやり直し可能

手順:

1. 接続:
   ギター → IF (Hi-Z入力)

   Hi-Z (ハイインピーダンス):
   ギター/ベース用入力
   IFに切り替えスイッチあり

2. Ableton設定:
   Audio Track
   Input: Ext. In 1
   Monitor: In
   Arm: ●

3. レベル調整:
   クリーンで弾いてピーク確認
   歪みペダル使用時も確認

4. 録音:
   アンプシミュレーター挿入
   (Ableton内蔵 or 外部プラグイン)
   Guitar Rig, Amplitube, Helix Native等

5. 後処理:
   DI信号をそのまま保存
   アンプシミュは後から変更可能
   = Re-Amping的なワークフロー
```

### シンセサイザー録音

```
ハードウェアシンセの録音:

接続パターン:

1. Audio録音:
   シンセ Audio Out → IF Audio In
   → Audio Trackで録音

2. MIDI制御 + Audio録音:
   Mac → MIDI → シンセ → Audio → IF
   → MIDI Track + Audio Track

3. External Instrument使用:
   1つのトラックで完結
   MIDI送信 + Audio受信

手順 (方法3 推奨):

1. MIDI Track作成
2. External Instrumentデバイス挿入
3. MIDI To: シンセのMIDIポート
4. Audio From: IFの入力チャンネル
5. Hardware Latency: 自動測定

録音:
MIDI演奏 → シンセがAudioで返す
→ Freeze/Flattenでオーディオ化

Tips:
DI的にクリーンな信号で録音
後からAbleton内でエフェクト処理
→ 柔軟性最大
```

---

## トラブルシューティング

### よくある問題と解決法

### Q1: レイテンシーが気になる

**A:** バッファーサイズを小さく

```
現在:
512 samples
= 11.6 ms レイテンシー

解決:

1. Buffer Size:
   512 → 128
   = 2.9 ms

2. それでもダメ:
   Direct Monitoring: On
   (IFの機能)

3. Mac の場合:
   CoreAudio は優秀
   128 samplesで安定

注意:

CPU負荷:
小さいバッファー = 高負荷

トラック数多い:
Freeze使用
```

### Q2: 録音レベルが小さい

**A:** ゲインを上げる

```
確認:

1. IF のゲインつまみ:
   右に回す

2. 目標レベル:
   ピーク -12 〜 -6 dB
   黄色表示

3. Clip (赤) は絶対ダメ:
   歪む

4. Ableton のフェーダー:
   0 dB (真ん中)

ゲイン vs フェーダー:

ゲイン (IF):
録音レベル
こっちで調整

Fader (Ableton):
再生音量
後で調整

推奨:

録音時:
ゲインで適正レベル

ミックス時:
フェーダーで音量バランス
```

### Q3: MIDIが録音されない

**A:** Armとモニター確認

```
チェックリスト:

1. トラックArm:
   ● が赤く点灯

2. Monitor:
   In (常に音が鳴る)

3. Input:
   All Ins / All Channels

4. 音源:
   Instrument挿入済み

5. MIDIキーボード:
   認識されている
   Preferences > MIDI

6. 鍵盤弾く:
   音が鳴る?

鳴らない場合:

MIDIキーボード:
USB接続確認
電源On

Ableton:
MIDI Ports設定
Track: ☑

音源:
Wavetable等を挿入
```

### Q4: 録音にノイズが乗る

**A:** ノイズ源の特定と対策

```
ノイズの種類:

ハム (ブーン):
原因: 電源ノイズ、アース不良
対策:
- バランスケーブル使用
- グラウンドリフトスイッチ
- 電源タップ確認

ヒス (シー):
原因: ゲインの上げすぎ、安価なプリアンプ
対策:
- ゲイン適正化
- 高品質プリアンプ
- マイクを近づける

クリック/ポップ:
原因: デジタルエラー、ケーブル不良
対策:
- ケーブル交換
- Buffer Size変更
- USBハブ使用中止

RF干渉:
原因: スマホ、Wi-Fi
対策:
- スマホ機内モード
- Wi-Fiルーターから離す
- シールドケーブル

一般的な対策:
1. ケーブルは短く
2. バランス接続推奨
3. USB直接接続
4. 不要機器の電源Off
```

### Q5: 録音が途中で止まる

**A:** CPU/ディスクの問題

```
原因と対策:

CPU Overload:
1. Buffer Size増加
2. Freeze Track使用
3. プラグイン削減
4. サンプルレート確認

Disk Overload:
1. SSD使用推奨
2. 外付けHDDなら7200rpm以上
3. 他のアプリのディスクアクセス停止
4. デフラグ（HDDの場合）

メモリ不足:
1. 不要アプリ終了
2. サンプル読み込み削減
3. RAM増設を検討

接続切れ:
1. USBケーブル確認
2. ハブ不使用
3. IF のドライバー更新
```

---

## 録音のショートカット一覧

```
基本操作:

Space: 再生/停止
F9: Arrangement Record
Cmd+F9: Session Record
Tab: Session/Arrangement切替
Cmd+T: 新規トラック

録音関連:

Record: F9 (Arrangement)
Session Record: Cmd+F9
Overdub: +ボタン
Capture MIDI: ★ボタン
Count In: 設定から

編集関連:

Cmd+E: 分割
Cmd+J: Consolidate
Cmd+Z: Undo
Cmd+Shift+Z: Redo
Delete: 削除

トラック操作:

Arm: トラックの●ボタン
Solo: S キー
Mute: M キー (0キー)
Monitor切替: トラック設定

ナビゲーション:

左右矢印: 再生ヘッド移動
上下矢印: トラック選択
Home: プロジェクト先頭
End: プロジェクト末尾

Loop:

Cmd+L: Loop範囲設定
Loop On/Off: Loopボタン

Zoom:

Cmd+'+': ズームイン
Cmd+'-': ズームアウト
Z: 選択範囲にズーム

マーカー:

Set Locator: 右クリック > Set Locator
次のLocator: 右矢印
前のLocator: 左矢印
```

---

## プロの録音テクニック集

### テクニック1: セーフティトラック

```
概要:
メイン録音とは別に
-10dB低いレベルで同時録音

方法:
1. メイントラック: ゲイン正常
2. セーフティトラック: ゲイン -10dB

メリット:
メイントラックがクリップしても
セーフティトラックは無事

プロの現場:
ライブ録音で必須
予測不能な音量に対応
```

### テクニック2: ルーム録音

```
概要:
演奏者から離れた位置にマイクを配置
部屋の響きを録音

用途:
ドラムの空気感
ストリングスの広がり
アコースティック楽器

設定:
1. メインマイク: 近距離
2. ルームマイク: 2-5m離す
3. 別トラックに録音
4. ミックスでブレンド

注意:
位相に注意
ルームマイクを遅らせない
```

### テクニック3: Re-Amping

```
概要:
DI録音した信号を
後からアンプに通して録音

手順:
1. ギターDI: 最初にクリーンで録音
2. DI信号をアンプに送る
3. アンプの音をマイクで録音
4. 納得いくまで調整可能

Ableton内Re-Amp:
1. DI録音済みトラック
2. アンプシミュレーター挿入
3. 好みの音色に調整
4. Freeze/Flatten

メリット:
演奏は1回でOK
アンプの音は後から何度でも変更
```

### テクニック4: パラレルコンプレッション録音

```
概要:
ドラム等の録音時に
コンプ有り/無しを同時に録音

方法:
1. マイク信号を2つのトラックに
2. Track A: コンプなし（クリーン）
3. Track B: コンプあり（パンチ）
4. ミックスでブレンド

メリット:
クリーンな原音を保持
パンチのある音も確保
後からバランス調整可能
```


---

## 実践演習

### 演習1: 基本的な実装

以下の要件を満たすコードを実装してください。

**要件:**
- 入力データの検証を行うこと
- エラーハンドリングを適切に実装すること
- テストコードも作成すること

```python
# 演習1: 基本実装のテンプレート
class Exercise1:
    """基本的な実装パターンの演習"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """入力値の検証"""
        if value is None:
            raise ValueError("入力値がNoneです")
        return True

    def process(self, value):
        """データ処理のメインロジック"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """処理結果の取得"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# テスト
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "例外が発生するべき"
    except ValueError:
        pass

    print("全テスト合格!")

test_exercise1()
```

### 演習2: 応用パターン

基本実装を拡張して、以下の機能を追加してください。

```python
# 演習2: 応用パターン
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """応用パターンの演習"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """アイテムの追加（サイズ制限付き）"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """キーによる検索"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """キーによる削除"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """統計情報"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# テスト
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # サイズ制限
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("応用テスト全合格!")

test_advanced()
```

### 演習3: パフォーマンス最適化

以下のコードのパフォーマンスを改善してください。

```python
# 演習3: パフォーマンス最適化
import time
from functools import lru_cache

# 最適化前（O(n^2)）
def slow_search(data: list, target: int) -> int:
    """非効率な検索"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# 最適化後（O(n)）
def fast_search(data: list, target: int) -> tuple:
    """ハッシュマップを使った効率的な検索"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# ベンチマーク
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"非効率版: {slow_time:.4f}秒")
    print(f"効率版:   {fast_time:.6f}秒")
    print(f"高速化率: {slow_time/fast_time:.0f}倍")

benchmark()
```

**ポイント:**
- アルゴリズムの計算量を意識する
- 適切なデータ構造を選択する
- ベンチマークで効果を測定する

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

## チーム開発での活用

### コードレビューのチェックリスト

このトピックに関連するコードレビューで確認すべきポイント:

- [ ] 命名規則が一貫しているか
- [ ] エラーハンドリングが適切か
- [ ] テストカバレッジは十分か
- [ ] パフォーマンスへの影響はないか
- [ ] セキュリティ上の問題はないか
- [ ] ドキュメントは更新されているか

### ナレッジ共有のベストプラクティス

| 方法 | 頻度 | 対象 | 効果 |
|------|------|------|------|
| ペアプログラミング | 随時 | 複雑なタスク | 即時のフィードバック |
| テックトーク | 週1回 | チーム全体 | 知識の水平展開 |
| ADR (設計記録) | 都度 | 将来のメンバー | 意思決定の透明性 |
| 振り返り | 2週間ごと | チーム全体 | 継続的改善 |
| モブプログラミング | 月1回 | 重要な設計 | 合意形成 |

### 技術的負債の管理

```
優先度マトリクス:

        影響度 高
          │
    ┌─────┼─────┐
    │ 計画 │ 即座 │
    │ 的に │ に   │
    │ 対応 │ 対応 │
    ├─────┼─────┤
    │ 記録 │ 次の │
    │ のみ │ Sprint│
    │     │ で   │
    └─────┼─────┘
          │
        影響度 低
    発生頻度 低  発生頻度 高
```

---

## セキュリティの考慮事項

### 一般的な脆弱性と対策

| 脆弱性 | リスクレベル | 対策 | 検出方法 |
|--------|------------|------|---------|
| インジェクション攻撃 | 高 | 入力値のバリデーション・パラメータ化クエリ | SAST/DAST |
| 認証の不備 | 高 | 多要素認証・セッション管理の強化 | ペネトレーションテスト |
| 機密データの露出 | 高 | 暗号化・アクセス制御 | セキュリティ監査 |
| 設定の不備 | 中 | セキュリティヘッダー・最小権限の原則 | 構成スキャン |
| ログの不足 | 中 | 構造化ログ・監査証跡 | ログ分析 |

### セキュアコーディングのベストプラクティス

```python
# セキュアコーディング例
import hashlib
import secrets
import hmac
from typing import Optional

class SecurityUtils:
    """セキュリティユーティリティ"""

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """暗号学的に安全なトークン生成"""
        return secrets.token_urlsafe(length)

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple:
        """パスワードのハッシュ化"""
        if salt is None:
            salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations=100000
        )
        return hashed.hex(), salt

    @staticmethod
    def verify_password(password: str, hashed: str, salt: str) -> bool:
        """パスワードの検証"""
        new_hash, _ = SecurityUtils.hash_password(password, salt)
        return hmac.compare_digest(new_hash, hashed)

    @staticmethod
    def sanitize_input(value: str) -> str:
        """入力値のサニタイズ"""
        dangerous_chars = ['<', '>', '"', "'", '&', '\\']
        result = value
        for char in dangerous_chars:
            result = result.replace(char, '')
        return result.strip()

# 使用例
token = SecurityUtils.generate_token()
hashed, salt = SecurityUtils.hash_password("my_password")
is_valid = SecurityUtils.verify_password("my_password", hashed, salt)
```

### セキュリティチェックリスト

- [ ] 全ての入力値がバリデーションされている
- [ ] 機密情報がログに出力されていない
- [ ] HTTPS が強制されている
- [ ] CORS ポリシーが適切に設定されている
- [ ] 依存パッケージの脆弱性スキャンが実施されている
- [ ] エラーメッセージに内部情報が含まれていない
---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## まとめ

### 録音の基本

```
□ Buffer Size: 128 (録音時)
□ レベル: ピーク -12 〜 -6 dB
□ Monitor: Auto (推奨)
□ Count In: 1-2 Bars
□ Metronome: On
□ Bit Depth: 24bit
□ Sample Rate: 44.1kHz
```

### Audio録音

```
□ IF接続確認
□ +48V (コンデンサーマイク)
□ ゲイン調整
□ クリップさせない
□ 24bit録音
□ 環境ノイズ対策
□ マイキング確認
□ ポップガード使用
```

### MIDI録音

```
□ MIDIキーボード認識
□ 音源挿入
□ Monitor: In
□ Recording Quantize: 1/16
□ テイク保存
□ Velocity表現
□ Capture MIDI活用
□ CC録音確認
```

### 重要ポイント

```
□ レイテンシー対策
□ オーバーダビング活用
□ コンピングでベストテイク
□ Direct Monitoring検討
□ 複数テイク録音
□ リサンプリング技術
□ プロジェクト管理
□ バックアップ習慣
```

### 録音チェックリスト（本番前）

```
機材:
□ IF電源On
□ +48V On（コンデンサー時）
□ ケーブル全接続確認
□ ヘッドフォン接続

Ableton:
□ Buffer Size: 128
□ Sample Rate: 44.1kHz
□ Bit Depth: 24bit
□ Input Config確認

トラック:
□ Input設定正しい
□ Monitor設定正しい
□ Arm有効
□ レベル適正

環境:
□ エアコンOff
□ 不要機器Off
□ スマホ機内モード
□ 窓閉め

録音:
□ Count In設定
□ Metronome設定
□ テスト録音
□ レベル最終確認
□ 本番スタート
```

---

**次は:** [Automation](./automation.md) - パラメーターを自動制御

---

## 次に読むべきガイド

- [Warp機能](./warp-modes.md) - 次のトピックへ進む

---

## 参考文献

- [MDN Web Docs](https://developer.mozilla.org/) - Web技術のリファレンス
- [Wikipedia](https://ja.wikipedia.org/) - 技術概念の概要
