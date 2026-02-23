# Gain Staging

ヘッドルーム確保の基礎。Master -6 dB目標、適切な音量バランス、Utility活用、Pink Noise法、Metering技術を完全マスターします。

## この章で学ぶこと

- Gain Stagingとは何か（物理的・デジタル的な理解）
- Master -6 dB目標設定の根拠
- Utility活用法とシグナルチェイン
- トラック別音量バランスの設計
- Fader 0 dB vs Utility の使い分け
- Pink Noise法による客観的バランシング
- Metering（Peak / RMS / LUFS）の使い分け
- ダイナミクスレンジの管理
- セクション別の音量管理
- Gain Staging自動化テクニック
- ジャンル別の目標レベル
- よくある失敗と対処法の詳細解説

---

## なぜGain Stagingが重要なのか

**ミキシングの土台:**

```
Gain Staging なし:

Master: 0 dB (クリッピング)
ダイナミクス: ゼロ
マスタリング: 不可能

結果:
音が潰れる
歪む
プラグインが正常に動作しない

Gain Staging あり:

Master: -6 dB (ヘッドルーム)
ダイナミクス: 維持
マスタリング: 可能

結果:
クリア、太い
プラグインが最適に動作

プロの真実:

Gain Staging:
ミキシング開始前
必須

理由:
全ての基礎
これなしは始まらない

使用頻度:
100%
全プロジェクト
```

### Gain Stagingの歴史的背景

```
アナログ時代:

コンソール・テープレコーダーの時代
→ 各機器に最適な動作レベルが存在
→ レベルが低すぎ → ノイズフロアに埋もれる
→ レベルが高すぎ → テープ飽和、歪み

0 VU = +4 dBu:
プロオーディオの標準動作レベル
ヘッドルーム: 約20 dB
ノイズフロア: 約-60 dB以下
有効ダイナミクスレンジ: 約80 dB

アナログの特徴:
テープ飽和は「心地よい歪み」
→ サチュレーション効果
→ 倍音付加
→ 意図的に使われることも

デジタル時代:

0 dBFS = 絶対最大値
→ 超えると即クリッピング
→ アナログのような心地よい歪みはない
→ デジタルクリッピング = 不快な歪み

32-bit float:
DAW内部処理は32-bit/64-bit float
→ 内部でのクリッピングは事実上発生しない
→ しかしD/A変換時に0 dBFSを超えると問題

重要な理解:
デジタル環境でも適切なGain Stagingは必要
→ プラグインの最適動作
→ マスタリングの余裕
→ ダイナミクスの維持
→ 良い習慣の確立
```

### Gain Stagingが全てに影響する理由

```
影響1: コンプレッサーの動作

レベルが高すぎ:
→ コンプレッサーが常に深く圧縮
→ 意図しない音質変化
→ ダイナミクスの喪失

レベルが適切:
→ コンプレッサーが設計通りに動作
→ 必要な時だけ圧縮
→ 自然なダイナミクス制御

影響2: EQの効果

レベルが高すぎ:
→ ブースト → クリッピング
→ EQ Eight内部での歪み

レベルが適切:
→ ブーストしてもヘッドルームあり
→ クリーンなEQ処理

影響3: サチュレーター/ディストーション

レベル依存のプラグイン:
→ 入力レベルで歪みの量が変わる
→ 高すぎると過剰な歪み
→ 低すぎると効果不足

最適入力:
→ プラグインごとに異なる
→ Gain In/Outで調整

影響4: リバーブ/ディレイ

Send量に影響:
→ トラックレベルが異なると
→ 同じSend量でもリバーブ量が変わる
→ 統一されたレベルが前提

影響5: マスタリング

ヘッドルーム確保:
→ Master -6 dBなら
→ マスタリングエンジニアに余裕を提供
→ Limiterが適切に動作
→ -14 LUFS目標が達成可能
```

---

## Gain Stagingとは

**適切な音量設定:**

### 基本概念

```
定義:

各段階で:
適切な音量レベル維持

「Gain」= 増幅/減衰
「Staging」= 段階的

つまり:
シグナルチェインの各段階で
適切なレベルを維持すること

目的:

1. ヘッドルーム確保:
   Master -6 dB以上
   → マスタリングの余裕

2. ノイズフロア回避:
   小さすぎない
   → アナログプラグインのノイズに埋もれない

3. ダイナミクス維持:
   圧縮避ける
   → トランジェント（アタック）が生きる

4. プラグインの最適動作:
   各プラグインの設計動作レベルで
   → コンプ、サチュレーター等が正常動作

デジタルオーディオ:

0 dBFS:
絶対最大
Full Scale

0 dBFS超え:
クリッピング
歪み
デジタルでは即座に不快

目標:
-6 dBFS (Master Peak)

理由:

マスタリング:
+6 dB Gain可能

Limiter:
GR -6 dB余裕

ダイナミクス:
圧縮なし
トランジェント維持
```

### シグナルフローの理解

```
信号の流れ:

音源（サンプル/シンセ）
  ↓ [レベル1]
Utility (Gain In)
  ↓ [レベル2]
EQ Eight
  ↓ [レベル3]
Compressor
  ↓ [レベル4]
Saturator
  ↓ [レベル5]
Send (Reverb/Delay)
  ↓ [レベル6]
Utility (Gain Out)
  ↓ [レベル7]
Track Fader
  ↓ [レベル8]
Group/Bus Fader
  ↓ [レベル9]
Master Fader
  ↓ [レベル10]
Output (D/A変換)

各段階で適切なレベルを維持:
レベル1-9: -18 〜 -6 dBFS
レベル10: -6 dBFS（最終目標）

重要:
DAW内部は32-bit floatだが
各プラグインの入力レベルは重要
→ アナログモデリングプラグインは特に
→ -18 dBFS前後が最適なことが多い
```

---

## Master -6 dB目標

**なぜ-6 dBなのか:**

### 理由（詳細版）

```
理由1: マスタリング余裕

現状:
Master -6 dB

Mastering:
Limiter Gain +6 dB
EQ調整 ±2 dB
Stereo処理

結果:
-14 LUFS達成
クリーンなリミッティング

計算:
Mix: -6 dBFS Peak → 約 -18 LUFS
Mastering: +6 dB Gain → 約 -12 LUFS
Limiter: -2 dB GR → -14 LUFS
→ ストリーミング最適化達成

理由2: ダイナミクス保持

-6 dB余裕:
Transient保持
ピーク成分が潰れない
Kickのアタック、Snareのスナップが生きる

0 dB:
圧縮される
Master limitingでTransient消失
パンチ感喪失

理由3: エフェクト余裕

EQ Boost:
+3 dB可能（余裕あり）

Compressor Make-Up:
+4 dB可能

合計 +7 dB使用:
-6 dB + 7 = +1 dB
→ クリッピング危険だが余裕はある

だから:
-6 dBは最低限の余裕
-8〜-10 dBならさらに安心

理由4: プラグインの最適動作

多くのプラグイン:
-18 dBFS前後で最適動作
（0 VU = -18 dBFS の慣例）

Master -6 dB:
個別トラック: -12〜-18 dBFS
→ プラグインの理想的な入力レベル

業界標準:

ポップ/ロック: -6 dB
クラシック/ジャズ: -10 dB
EDM: -3〜-6 dB (aggressive)
Film/TV: -24 dBFS True Peak (EBU R128)

推奨:
-6 dB (安全かつ標準的)
-3 dBでも可（EDM等）
-10 dB以上は安全だが小さすぎ注意
```

### -6 dBの具体的な意味

```
数値の意味:

-6 dBFS Peak:
最も大きいピーク（瞬間最大値）が
0 dBFSから6 dB下

RMS値:
通常 -18〜-12 dBFS程度

LUFS値:
通常 -20〜-16 LUFS程度

確認方法:
Master Track のメーター
最も大きいセクション（Drop等）で
ピークが -6 dB付近

注意:
常に -6 dBではなく
最大ピーク時に -6 dB
静かなセクションはもっと低い

測定のタイミング:
楽曲の最も大きいセクション（Drop / Chorus）
最初から最後まで通して確認
ピークを見落とさない
```

---

## Gain Staging手順

**10ステップ:**

### Step 1: 全トラックFader確認

```
開始前:

1. 全トラック選択
   Cmd+A (Mac) / Ctrl+A (Win)

2. Fader位置確認:
   バラバラ?
   作曲中に適当に動かしている可能性

3. 全て 0 dB にリセット:
   Option + クリック (Mac)
   Alt + クリック (Win)

理由:
統一した状態から開始
先入観を排除
すべてのトラックを同じスタートラインに

重要:
この時点でMasterが大きくても気にしない
これから調整していく
```

### Step 2: Master確認

```
再生:

全トラック:
Fader 0 dB

Master:
何dB?

例:
Master: +3 dB (赤)
→ クリッピング → 要修正

Master: -2 dB
→ ヘッドルーム少ない → やや下げ

Master: -15 dB
→ 小さすぎ → やや上げ

Master: -6 dB
→ 理想的 → このまま

目標:
Master: -6 dB

確認方法:
楽曲の最も大きいセクションで確認
通常はDrop / Chorus部分
```

### Step 3: Utility挿入（全トラック）

```
方法:

1. 全トラック選択

2. Browser > Audio Effects > Utility

3. 全トラックにドラッグ&ドロップ
   → 最初のデバイスとして挿入

4. Gain: 計算して設定

例:

現在 Master: +3 dB
目標: -6 dB
差: -9 dB

全トラック Utility:
Gain: -9 dB

結果:
Master: -6 dB

効率化:

テンプレート作成:
Utility Gain -6 dB
全トラックにドロップ
→ 大体のプロジェクトはこれで範囲内

または Audio Effect Rack:
Utility (Gain In) + Utility (Gain Out)
→ プリセットとして保存

ショートカット:
Cmd+G: グループ化
→ グループにUtility
→ 全体のレベル調整
```

### Step 4: Kick基準設定

```
Kick = 基準:

なぜKickが基準か:
→ EDM/Dance Musicの最重要要素
→ 最もピークが大きい
→ 全体のレベルを決定する

手順:
1. 全トラック Mute

2. Kick のみ Solo

3. 再生（Drop部分）

4. Master確認:
   目標 -6 dB（Peak）

5. Kick Fader（またはUtility）調整:
   Master -6 dBになるまで

例:

Kick Fader: 0 dB
→ Master: -4 dB (大きすぎ)

Kick Fader: -2 dB
→ Master: -6 dB (完璧)

固定:
Kick Fader -2 dB
→ これが基準

注意:
Kickのピーク値で -6 dB
RMSではなくPeak
クリッピング防止のため
```

### Step 5: Bass追加

```
1. Bass Un-Mute

2. Kick + Bass再生

3. Master確認:
   目標 -6 dB維持

4. Bass Fader調整:
   Kickとのバランスを聴きながら
   Masterが-6 dBを超えないように

例:

Bass Fader: 0 dB
→ Master: -3 dB (大きすぎ)

Bass Fader: -3 dB
→ Master: -6 dB (完璧)

バランス:
Kick -2 dB
Bass -3 dB

聴感チェック:
Kickが最前列
Bassがすぐ後ろ
両方明確に聴こえる
低域が太い
```

### Step 6: Drumsグループ追加

```
1. Snare・Hi-Hat・Percussion Un-Mute

2. 再生

3. Master確認:
   目標 -6 dB

4. Snare Fader:
   Kickより小さく
   でも明確に聴こえる
   スネアのスナップが感じられる

5. Hi-Hat Fader:
   さらに小さく
   リズムが明確に聴こえる程度

6. Percussion:
   装飾的
   控えめ

例:

Kick: -2 dB（基準）
Bass: -3 dB
Snare: -6 dB
Hi-Hat: -12 dB
Percussion: -12 dB
Master: -6 dB維持

ポイント:
Masterが-6 dBを超える場合
→ 追加トラックを下げる
→ 全体を下げない
```

### Step 7: Melody追加

```
1. Lead・Pad Un-Mute

2. 再生

3. バランス:

Lead: 前に出る
→ Fader -6 dB
→ Snareと同程度

Pad: 後ろに
→ Fader -12 dB
→ 空間を埋める程度

Master: -6 dB維持

調整:
各Fader微調整
Lead > Pad の関係維持

ポイント:
Padは「聴こえない」くらいが適切
→ Muteすると寂しくなる
→ Un-Muteすると気づかないくらい
→ それが「ちょうどいい」レベル
```

### Step 8: Vocal追加

```
Vocal あり:

1. Vocal Un-Mute

2. 最前列:
   Kickと同じくらいの存在感
   ただしピークレベルはKickより低くてよい

3. Fader調整:
   -4〜-6 dB程度
   楽曲のジャンルによる

Vocal なし:
このステップをスキップ

Master: -6 dB維持

ポイント:
Vocalが入るとMasterレベルが大きく変わることがある
→ 他のトラックのバランスを微調整
→ 特にLead SynthとのMid帯域の関係
```

### Step 9: 全体バランス

```
1. 全トラック再生

2. Master確認:
   -6 dB?

3. 微調整:
   各Fader ±1〜2 dB
   大きな変更は不要

4. 確認:
   全セクション (Intro・Verse・Buildup・Drop・Outro)
   特にDropが最大ピーク

5. 最も大きい部分:
   Master -6 dB

6. 最も小さい部分:
   Master -12〜-15 dB程度
   極端に小さくなりすぎない

セクション間の音量差:
Verse → Drop: +3〜+6 dB の差
→ 適度なダイナミクス
→ Drop のインパクトを確保
```

### Step 10: 保存

```
完成:

全トラック Fader設定完了
Master -6 dB

保存:
Cmd+S

バックアップ:
File > Save As
"Track Name - Gain Staged"

次:
EQ・Compression開始
Gain Stagingが完了してから
ミキシングの本番を開始

テンプレート化:
このバランスをテンプレートとして保存
→ 次回のプロジェクトで再利用
→ 時間短縮
```

---

## Utility活用法

**Gain調整の最良ツール:**

### Utilityとは

```
機能:

Gain:
±35 dB
音量の増減

Width:
Stereo幅
0% = Mono, 100% = Stereo, 200% = 過剰Wide

Bass Mono:
低域Mono化
特定周波数以下をMono

Phase:
位相反転
L/R個別に

Pan:
左右の配置
精密なパンニング

Channel Mode:
Left, Right, Mono, Stereo
チャンネル選択

Mute:
無音化
デバッグ用

推奨:
各トラックに必ず挿入
シグナルチェインの最初と最後
```

### Utility配置

```
Chain順序:

1. Utility (Gain In)
   ↓ 入力レベルを最適化
2. EQ Eight
   ↓ 適切なレベルでEQ処理
3. Compressor
   ↓ 適切なレベルでコンプレッション
4. その他エフェクト
   ↓ サチュレーター等
5. Utility (Gain Out)
   ↓ 出力レベルを最適化

理由:

Gain In:
入力レベル調整
プラグインへの適切な入力確保
アナログモデリングプラグインに最適なレベルを送る

Gain Out:
出力レベル調整
プラグインの増幅分を補正
トラックフェーダーに送る前のレベル管理

メリット:
Fader触らない
→ Faderは最終バランス調整専用
Automation簡単
→ Utility GainのAutomationが直感的
プラグインの動作最適化
→ 各段階で適切なレベル
```

### Utilityの高度な活用

```
テクニック1: A/B比較用

Before:
Utility Gain In をバイパス
→ 元の音量で確認

After:
Utility Gain In をオン
→ 調整後の音量で確認

→ プラグインの効果を音量差なしで比較

テクニック2: Gain Matching

プラグイン挿入前後で音量を一致させる:

1. プラグインなし → Masterレベル測定
2. プラグイン挿入 → Masterレベル測定
3. 差分をUtility Gain Outで補正

例:
Compressor: -3 dB GR（平均）
Utility Gain Out: +3 dB
→ 音量一致

テクニック3: モノ確認

Master Track:
Utility Width: 0%
→ 全体がMono化
→ ステレオ問題の確認

テクニック4: 位相確認

Utility Phase: Invert
→ 片方のチャンネルを反転
→ Monoで消える成分がStereo成分
→ 残る成分がMono成分

テクニック5: 低域Mono化

Bass Mono: On
Freq: 120 Hz
→ 120 Hz以下を自動的にMono
→ 位相問題の防止
→ 業界標準の処理
```

---

## Pink Noise法

**プロの技:**

### 原理

```
Pink Noise:

定義:
全周波数帯域で均等なエネルギー密度
（各オクターブで同じパワー）

特徴:
「シーーー」という音
White Noiseより低域が多い
人間の聴覚特性に近い

なぜPink Noise?
White Noise: 高域が過剰に感じる
Pink Noise: 人間の耳に自然なバランス
→ -3 dB/octave の傾斜
→ 理想的なミックスの傾斜に近い

用途:
音量バランスの客観的参照
主観的な判断を排除
短時間で正確なバランスを実現

科学的根拠:
人間の聴覚は対数的
→ Pink Noiseはオクターブ単位で均一
→ 聴感上「均等」に感じる
→ 良いミックスバランスのモデル
```

### 実践手順（詳細版）

```
Step 1: Pink Noise設定

1. 新規Audio Track作成
   名前: "Pink Noise Reference"

2. Operator挿入（または他のシンセ）
   Wave: Noise
   Noise Type: Pink

3. または Sample:
   Pink Noise サンプルを配置
   ループ: On

4. Fader: -12 dB
   （十分聴こえるが大きすぎない）

5. 再生
   ループ確認

Step 2: トラック個別調整

準備:
ヘッドフォン使用推奨
モニタースピーカーでもOK

手順:

1. Kick Solo + Pink Noise
   → Pink Noiseに「埋もれて聴こえない」レベルに
   → Kick FaderをPink Noiseに馴染むまで下げる
   → わずかに聴こえる程度

2. Bass Solo + Pink Noise
   → 同様にPink Noiseに馴染むレベルに
   → Bass Fader調整

3. Snare Solo + Pink Noise
   → 同様に調整

4. Hi-Hat Solo + Pink Noise
   → 同様に調整

5. Lead Solo + Pink Noise
   → 同様に調整

6. Pad Solo + Pink Noise
   → 同様に調整

7. Vocal Solo + Pink Noise
   → 同様に調整

8. 全トラック繰り返し

ポイント:
「Pink Noiseにちょうど埋もれるレベル」が目標
→ 各トラックが均等なパワーで配置される
→ 主観を排除した客観的バランス

Step 3: 微調整

Pink Noiseを停止
全トラック再生

確認:
1. 全体のバランスを確認
2. Kickが基準として適切か
3. Vocalが前に出ているか
4. 各楽器が聴こえるか

微調整:
Vocal: +1〜+2 dB（前に出す）
Kick: +1 dB（基準強化）
→ Pink Noiseバランスからの微修正

Step 4: 削除

Pink Noise Track削除
または Mute して保持（後で再利用）

確認:
Master -6 dB

メリット:

客観的:
感覚に頼らない
疲れた耳でも正確

正確:
周波数バランス良い
Pink Noiseの均等エネルギーを利用

高速:
15-20分で完了
通常のバランシングより速い

再現性:
毎回同じ方法
一貫した結果

プロ使用:
業界標準テクニック
多くのプロが使用
```

### Pink Noise法の注意点

```
注意1: 完璧ではない
Pink Noiseバランスはスタートポイント
→ 最終的には耳で微調整
→ 楽曲の意図を反映させる

注意2: ジャンルによる違い
EDM: Kickを+2〜+3 dB上げる
Pop: Vocalを+2〜+3 dB上げる
Jazz: ドラムを-2〜-3 dB下げる
→ Pink Noiseからの意図的な逸脱

注意3: モニター環境の影響
Pink Noiseの聴こえ方はモニターに依存
→ 一貫したモニター環境で行う
→ ヘッドフォン使用推奨

注意4: Pink Noiseのレベル
-12 dBが推奨
→ 小さすぎ: 各トラックも小さくなりすぎ
→ 大きすぎ: Master -6 dBを超える可能性
```

---

## トラック別目標レベル

**参考値:**

### 標準設定

```
Kick:

Fader: -2 dB
Peak: -6 dB (Master基準)
最も大きい
楽曲の基盤

Bass:

Fader: -3 dB
Peak: -9 dB
Kickの次
低域のパワー

Snare/Clap:

Fader: -6 dB
Peak: -12 dB
明確だが控えめ
リズムのアクセント

Hi-Hat:

Fader: -12 dB
Peak: -18 dB
聴こえる程度
リズムの細分化

Lead:

Fader: -6 dB
Peak: -12 dB
前に出る
メロディの主役

Vocal:

Fader: -6 dB
Peak: -12 dB
Kick・Bassの次に重要
ジャンルにより最重要

Pad:

Fader: -12 dB
Peak: -18 dB
後ろに
空間を埋める

FX:

Fader: -15 dB
Peak: -21 dB
装飾
気づかない程度

Reverb Return:

Fader: -12〜-18 dB
空間を追加
過剰注意

Delay Return:

Fader: -15〜-18 dB
リズム的効果
控えめ

ルール:

これは参考値
楽曲により変わる
大切なのは相対関係
Kick > Bass > Vocal/Lead > Snare > Pad/FX > Hi-Hat
```

### ジャンル別の音量バランス

```
Techno:

Kick: 基準 (最大)
Bass: Kickの-2 dB
Hi-Hat: 控えめ
Lead: 中程度
Pad: 最小限
Vocal: なし or 控えめ

特徴:
Kick・Bassが全体の70%
リズム中心

House:

Kick: 基準
Bass: Kickと同程度
Vocal: 前に出す (+2 dB)
Pad: 適度
Lead: 中程度

特徴:
Vocalが重要
グルーヴ重視

Pop/EDM:

Vocal: 基準 (最大)
Kick: Vocalの-2 dB
Bass: Kickの-2 dB
Lead: 中程度
Pad: 広い

特徴:
Vocal中心
広いステレオイメージ

Ambient:

Pad: 基準 (最大)
Lead: 中程度
Bass: 控えめ
Kick: なし or 最小限
FX: 重要 (+2 dB)

特徴:
空間が主役
ダイナミクスは広い
```

---

## Fader 0 dB vs Utility

**どちらを使う？:**

### 比較

```
Fader使用:

メリット:
視覚的わかりやすい
直感的操作
リアルタイム調整が容易
ミキサービューで一覧

デメリット:
Automation複雑
→ FaderをAutomationすると後で変更困難
後で変更困難
→ 基本レベルが変わると全Automationがずれる
プラグインの前に調整できない
→ Faderはチェインの最後

Utility使用:

メリット:
Fader 0 dB維持
→ 視覚的にクリーン
Automation簡単
→ Utility GainとFaderを独立にAutomation可能
後で変更簡単
→ 基本レベルをUtilityで、動的変化をFaderで
プラグインの前に配置可能
→ 入力レベル制御

デメリット:
1ステップ多い
→ Utility挿入が必要
画面上見えにくい
→ デバイスチェイン内

推奨:

Utility:
Gain Staging（基本レベル設定）
プラグインの入出力レベル

Fader:
最終バランス調整
Volume Automation
リアルタイム調整
```

### プロのワークフロー

```
プロの使い方:

Step 1: Utility Gain In
→ 入力レベルを-18 dBFS前後に

Step 2: プラグインチェイン
→ 各プラグインが最適レベルで動作

Step 3: Utility Gain Out
→ プラグインによるレベル変化を補正

Step 4: Fader
→ 0 dB付近を維持
→ 最終的なバランス微調整
→ Volume Automation用

利点:
→ Faderの位置が「バランスの調整量」を示す
→ 0 dBから大きく離れていれば
→ Gain Stagingの問題を示唆
→ 管理しやすい
```

---

## セクション別の音量管理

**楽曲構造と音量:**

```
各セクションの目標:

Intro:
Master: -12〜-15 dB
静かなスタート
要素が少ない

Verse:
Master: -8〜-10 dB
メイン要素が入る
まだ控えめ

Buildup:
Master: -8 → -4 dB（徐々に増加）
テンション上昇
Automation活用

Drop:
Master: -6 dB（最大ピーク）
全要素が入る
楽曲のクライマックス

Breakdown:
Master: -10〜-12 dB
静かなセクション
Dropとのコントラスト

Outro:
Master: -10 → -18 dB（徐々に減少）
フェードアウト
またはカットアウト

管理方法:

1. Drop基準:
   Dropで -6 dB
   他はそれより小さい

2. Volume Automation:
   Utility GainまたはFader

3. 要素の増減:
   トラックのMute/Un-Muteで
   自然な音量変化

注意:
セクション間の差が大きすぎると
→ リスナーが音量調整してしまう
→ 適度なダイナミクスレンジを維持

目安:
最大（Drop）と最小（Intro/Breakdown）の差:
6-12 dB
```

---

## ダイナミクスレンジ

**適切な変化:**

### 概念（詳細版）

```
ダイナミクスレンジ:

定義:
最大音量と最小音量の差

最大音量:
Kickのピーク
楽曲の最大瞬間値

最小音量:
最も静かなセクション
Padの平均等

差:

適切: 12-18 dB
狭すぎ: 6 dB (圧縮過剰)
広すぎ: 24 dB (バランス悪い)

ジャンル別:

Techno/House:
ダイナミクス: やや狭い (8-12 dB)
理由: クラブ再生、大音量
一定のエネルギーが求められる

Pop/Rock:
ダイナミクス: 中程度 (12-18 dB)
理由: 多様なリスニング環境

Classical:
ダイナミクス: 広い (20-30 dB)
理由: 表現の幅

Film:
ダイナミクス: 非常に広い (30+ dB)
理由: ダイアログと爆発音の差

マイクロダイナミクス vs マクロダイナミクス:

マイクロ:
各トラック内のダイナミクス
→ Compressorで制御

マクロ:
セクション間のダイナミクス
→ Arrangement と Automationで制御

両方が適切であることが重要
```

### Crest Factor

```
定義:
Peak値とRMS値の差

計算:
Crest Factor = Peak (dB) - RMS (dB)

意味:
大きい → トランジェント（アタック）が明確
小さい → 圧縮されている

目安:
ミックス: 12-18 dB
マスタリング後: 8-12 dB
過剰圧縮: 6 dB以下

なぜ重要:
Crest Factorが大きい → ダイナミクスがある → パンチ感
Crest Factorが小さい → 圧縮されている → 疲れる音

確認方法:
Peak MeterとRMS Meterの差を見る
または LUFS Meter の True Peak と Short-term の差
```

---

## Metering

**測定ツール:**

### 必須メーター（詳細版）

```
Peak Meter:

定義:
瞬間最大値
サンプル単位での最大

単位:
dBFS (decibels Full Scale)

目標:
-6 dBFS (Mix)
-1.0 dBFS True Peak (Master後)

用途:
クリッピング防止
ヘッドルーム確認
Gain Stagingの基本指標

特徴:
一瞬の最大値
ダイナミクスの情報はない
クリッピングの有無のみ

True Peak:
サンプル間ピークを推定
実際のアナログ波形のピーク
通常のPeakより+1〜+2 dB高い場合あり
マスタリングでは必ずTrue Peakを確認

RMS Meter:

定義:
Root Mean Square
二乗平均平方根
平均的なエネルギーレベル

目標:
-18 〜 -12 dBFS (Mix)

用途:
全体的な音量感の確認
トラック間のバランス確認

特徴:
平均値のため安定した表示
ダイナミクスの概要がわかる
ピーク情報は見えない

LUFS Meter:

定義:
Loudness Units Full Scale
知覚ラウドネスの測定
人間の聴覚特性を反映

種類:
Integrated: 全体平均
Short-term: 3秒平均
Momentary: 0.4秒

目標:
Mix: -20 〜 -16 LUFS (Integrated)
Master後: -14 LUFS (Spotify/Apple Music)

用途:
ラウドネス正規化
ストリーミングサービスの基準確認
Reference Trackとの比較

特徴:
人間の聴覚特性を反映
K-weighted（周波数重み付け）
ITU-R BS.1770規格

ストリーミング基準:
Spotify: -14 LUFS
Apple Music: -16 LUFS
YouTube: -14 LUFS
Tidal: -14 LUFS
Amazon Music: -14 LUFS

推奨ツール:

Ableton:
Master Track Meter (Peak)
→ 基本的な確認は可能

無料:
Youlean Loudness Meter
→ LUFS, True Peak, RMS
→ 十分な機能
→ 推奨！

有料:
iZotope Insight 2
→ 総合メータリング
→ Spectrum, Stereo, LUFS, True Peak

Waves WLM Plus
→ ラウドネスメーター
→ 放送規格対応

TC Electronic LM2n
→ ラウドネスメーター
→ 放送規格対応

MeterPlugs Lcast / PERCEPTION
→ ラウドネスメーター
```

### K-System Metering

```
Bob Katz K-System:

K-12:
Pop, Rock, EDM
RMS 0 = -12 dBFS
ヘッドルーム: 12 dB

K-14:
Acoustic, Folk, Jazz
RMS 0 = -14 dBFS
ヘッドルーム: 14 dB

K-20:
Classical, Film
RMS 0 = -20 dBFS
ヘッドルーム: 20 dB

意味:
0 VU = 指定されたdBFS値
→ アナログ感覚でのメータリング
→ ヘッドルームが確保される

推奨:
EDM/Dance: K-12
→ -12 dBFS RMS = 0 VU
→ ヘッドルーム12 dB

一般的なミキシング:
K-14推奨
→ 十分なヘッドルーム
→ マスタリングの余裕
```

---

## Gain Stagingの自動化

**効率化テクニック:**

```
テクニック1: テンプレート

作成:
全トラックにUtility挿入済みのテンプレート
→ Gain In / Gain Out 両方
→ LUFS Meter (Master)
→ Spectrum (Master)

保存:
File > Save As Template

使用:
新規プロジェクト → テンプレートから開始
→ 毎回の挿入作業が不要

テクニック2: Default Audio Effect Rack

作成:
Utility (Gain In) + Utility (Gain Out)
→ Audio Effect Rackとして保存
→ デフォルトのインストゥルメントに追加

テクニック3: Max for Live デバイス

Gain Tool:
自動的にレベルを測定
目標レベルに調整

Auto Gain:
プラグイン挿入後の自動補正

テクニック4: Grouping

Drums Group:
Kick + Snare + Hi-Hat
→ グループにUtility
→ 一括レベル管理

Music Group:
Lead + Pad
→ グループにUtility

テクニック5: Pre/Post Fader Metering

確認:
各トラックのPre-Faderレベル
→ プラグイン後のレベル確認
→ Post-Faderレベルと比較
→ Faderの変更量を把握
```

---

## よくある失敗

**Gain Stagingの罠:**

### 1. Master 0 dB

```
問題:
クリッピング
歪み
マスタリング不可
デジタルクリッピングの不快な歪み

原因:
ヘッドルームなし
Gain Staging未実施
作曲中に音量を上げすぎた

症状:
Master meterが赤点灯
高域が歪んで聴こえる
Limiterが常に大きくGR
マスタリングエンジニアから指摘

解決:

全トラック Utility:
Gain -6 dB
一括でレベルダウン

Master:
-6 dB確保

確認:
最も大きいセクションで確認

理由:
必須
全てのミキシングの前提条件
```

### 2. トラック小さすぎ

```
問題:
ノイズフロア近い
音質劣化
S/N比の悪化

原因:
過剰にGain下げ
必要以上にヘッドルームを取った

症状:
音が痩せて聴こえる
プラグインのノイズが相対的に大きい
Faderを大幅に上げる必要がある

解決:

目標:

Individual Track:
-12 〜 -18 dB (Peak)
この範囲が最適

Master:
-6 dB (Peak)

ルール:
小さすぎもダメ
大きすぎもダメ
適切なバランスを
```

### 3. Faderバラバラ

```
問題:
管理困難
視覚的わかりにくい
Automationが複雑になる

原因:
計画なし
作曲中に適当にFaderを動かした

症状:
Faderが-30 dBのトラックがある
Faderが+6 dBのトラックがある
ミキサービューが見にくい

解決:

Fader:
なるべく 0 dB近く
±6 dB以内が理想

Gain調整:
Utility使用
Faderは最終微調整のみ

手順:
1. 全Fader 0 dBにリセット
2. Utilityで基本レベル設定
3. Faderは±3 dB以内で微調整

理由:
視覚的整理
Automationの基準点が明確
```

### 4. セクションごとに差が大きすぎ

```
問題:

Verse: -8 dB
Drop: -2 dB (大きすぎ)
→ Dropでクリッピング

原因:
Drop要素多い
Buildupの加算
リバーブの蓄積

症状:
Dropの瞬間にMasterが赤
Verse → Dropの音量差が極端

解決:

方法1: Drop Gain下げ
Drop要素の各トラックUtility:
Gain -3 dB程度
Master -6 dB維持

方法2: Verse Gain上げ
Verse要素を少し上げる
全体のバランスを統一

方法3: マスタートラックAutomation
Master Utility Gain:
Drop: -3 dB
Verse: 0 dB
→ 全体を制御

目標:
全セクション最大ピーク -6 dB
セクション間の差: 6-12 dB

確認:
全セクションを通して再生
Masterメーターを常時監視
```

### 5. プラグイン挿入後のレベル変化を無視

```
問題:
プラグインを挿入するたびにレベルが変わる
最終的にMasterが0 dBを超える

原因:
EQブースト
Compressor Make-up Gain
Saturatorの音量増加
各プラグインのゲイン変化の累積

症状:
プラグインを追加するたびにMasterが上がる
ミキシングの後半でクリッピング

解決:

原則: Unity Gain
各プラグインの前後で音量を一致させる

手順:
1. プラグイン挿入前: レベル記録
2. プラグイン挿入後: レベル記録
3. 差分をUtility Gain Outで補正

例:
Compressor: GR -4 dB, Make-up +4 dB
→ Unity Gain維持

EQ: ブースト +3 dB 合計
→ Utility Gain Out: -3 dB
```

### 6. モニターボリュームで補正する

```
問題:
ミックスが大きい/小さい
→ モニターの音量を上げ/下げで対処

原因:
Gain Stagingの概念を理解していない

症状:
モニター音量が極端に大きい/小さい
異なる環境で再生するとバランスが違う

解決:

原則:
モニター音量は固定
ミックス内でレベルを調整

推奨モニターレベル:
85 dB SPL (ITU推奨)
→ 実際にはこれは大きめ
→ 70-80 dB SPL程度が現実的
→ 一定の音量で作業する
```

---

## Gain Stagingチェックリスト

**完全チェック:**

```
開始前:
□ 全Fader 0 dBにリセット
□ Master現在のレベルを確認
□ 全トラックにUtility挿入
□ LUFS Meter挿入 (Master)
□ Spectrum挿入 (Master)

基本設定:
□ Kick Solo → Master -6 dB
□ Bass追加 → バランス確認
□ Drums追加 → Master -6 dB維持
□ Melody追加 → Master -6 dB維持
□ Vocal追加 → Master -6 dB維持

セクション確認:
□ Intro: Master確認
□ Verse: Master確認
□ Buildup: Master確認
□ Drop: Master -6 dB（最大ピーク）
□ Outro: Master確認

品質確認:
□ クリッピングなし（全セクション）
□ ダイナミクスレンジ: 8-18 dB
□ 全トラックバランス: 適切
□ Fader位置: 0 dB付近

保存:
□ プロジェクト保存
□ バックアップ作成
□ テンプレート更新（必要に応じて）
```

---

## 実践ワークフロー

**30分で完成:**

### Step-by-Step

```
0-5分: 準備

1. 全トラック Fader 0 dB
2. 再生・Master確認（現在のレベル記録）
3. 全トラック Utility挿入
4. LUFS Meter挿入 (Master Track)
5. Spectrum挿入 (Master Track)

5-10分: Kick・Bass

1. 全トラックMute
2. Kick Solo
3. Master -6 dB調整（Kick Fader/Utility）
4. Bass追加
5. Bass Fader調整
6. サイドチェイン確認（低域の住み分け）

10-15分: Drums

1. Snare追加 → Fader調整
2. Hi-Hat追加 → Fader調整
3. Percussion追加 → Fader調整
4. バランス確認（Drums全体）
5. Master -6 dB確認

15-20分: Melody

1. Lead追加 → Fader調整
2. Pad追加 → Fader調整
3. 前後関係確認（Lead前、Pad後）
4. Master -6 dB確認

20-25分: Vocal・FX

1. Vocal追加 (あれば) → Fader調整
2. FX追加 → Fader調整
3. 全体バランス確認
4. Master -6 dB確認

25-30分: 確認

1. 全セクション再生（Intro → Outro）
2. Master -6 dB確認（全セクション）
3. クリッピングチェック
4. ダイナミクスレンジ確認
5. 保存
```

---

## まとめ

### Gain Staging

```
□ Master -6 dB目標（Peak）
□ Kick基準で調整
□ Utility活用（Gain In / Gain Out）
□ Fader 0 dB維持 (Utility使用時)
□ 全セクション確認
□ Unity Gain原則（プラグイン前後）
□ Pink Noise法で客観的バランス
```

### 目標値

```
Master Peak: -6 dBFS
Individual Track Peak: -12 〜 -18 dBFS
ダイナミクスレンジ: 8-18 dB（ジャンル依存）
LUFS (Mix): -20 〜 -16 LUFS
LUFS (Master後): -14 LUFS
```

### 重要原則

```
□ ミキシング最初のステップ
□ これなしは始まらない
□ ヘッドルーム必須
□ クリッピング厳禁
□ 全トラックバランス
□ プラグインの最適動作レベル
□ Unity Gain維持
□ モニターボリュームは固定
□ テンプレートで効率化
□ Pink Noise法で客観性確保
```

### トラブルシューティング

```
クリッピング → 全トラックUtility -3〜-6 dB
小さすぎ → 全トラックUtility +3〜+6 dB
バランス悪い → Pink Noise法で再調整
セクション差大 → Drop要素を下げる/Verse要素を上げる
プラグイン後レベル変化 → Unity Gain補正
Faderバラバラ → 全リセット + Utility調整
```

---

**次は:** [Frequency Balance](./frequency-balance.md) - 周波数分離でクリアなミックスを実現
