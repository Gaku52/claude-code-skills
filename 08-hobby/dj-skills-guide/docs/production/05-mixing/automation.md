# Automation

動的ミックスで時間軸の変化を作ります。Volume・Filter・Send Automationを完全マスターします。

## この章で学ぶこと

- Automationとは
- Volume Automation
- Filter Cutoff Automation
- Send Automation
- Pan Automation
- ビルドアップ作成
- ドロップ演出
- Automation描画方法

---

## なぜAutomationが重要なのか

**時間軸の変化:**

```
Automation なし:

特徴:
静的
単調
変化なし

Automation あり:

特徴:
動的
劇的
展開

プロとアマの差:

アマ:
Automation: 10%
最小限

プロ:
Automation: 60%+
多用

真実:

「プロの楽曲」=
Automationが緻密

例:

Buildup:
Filter Cutoff上昇

Drop:
Volume突然大きく

Result:
劇的な展開
```

---

## Automationとは

**時間軸パラメーター変化:**

### 基本概念

```
定義:

パラメーター:
時間とともに変化

例:

Volume:
0分: -12 dB
1分: -6 dB
徐々に大きく

自動:
再生で自動変化

用途:

Volume: セクション別
Filter: ビルドアップ
Send: ドロップ演出
Pan: 動き

頻度:

EDM: 最も多用
Techno/House: 多用
Rock: 少ない
```

---

## Automation描画方法

**Ableton Live:**

### 基本操作

```
表示:

Arrangement View:
A (Show/Hide Automation)

表示:
赤い線 = Automation

描画モード:

Draw Mode:
B (On/Off)

ペン:
クリック&ドラッグ

ブレークポイント:

クリック:
追加

Delete:
削除

カーブ:

Shift + ドラッグ:
カーブ調整

選択パラメーター:

Device Chooser:
デバイス選択

Parameter Chooser:
パラメーター選択

例:
Volume → Track Volume
Filter → Cutoff

推奨:
マウス描画
または
MIDI Controller録音
```

---

## Volume Automation

**最も基本:**

### 用途

```
セクション別音量:

Intro:
Pad -18 dB (小さく)

Verse:
Pad -15 dB

Buildup:
Pad -12 dB → -6 dB
徐々に大きく

Drop:
Pad -6 dB (最大)

Outro:
Pad -6 dB → -∞ dB
フェードアウト

推奨:

Breakdown:
要素減らす
Volume -∞ dB (Mute)

Drop復帰:
突然 -6 dB

効果:
劇的
```

### 実践例: Techno Drop

```
Timing:

Bar 64-65: Buildup最後
Bar 65-66: Drop

Automation:

Pad Volume:

Bar 1-64: -15 dB (通常)
Bar 64-65: -15 → -6 dB (上昇)
Bar 65: -6 dB (維持、Drop)

Percussion Volume:

Bar 1-64: -12 dB
Bar 60-64: -12 → -∞ dB (消える)
Bar 65: -∞ → -12 dB (突然復帰)

FX Volume:

Bar 60-65: 徐々に+6 dB
Bar 65: 突然-∞ dB (消える)

Kick Volume:

Bar 64: -∞ dB (消える)
Bar 65: -6 dB (突然復帰)

効果:
劇的なDrop
```

---

## Filter Automation

**ビルドアップ必須:**

### Auto Filter Cutoff

```
ビルドアップ定番:

Pad・Lead:
Auto Filter挿入

Cutoff Automation:

Bar 60: 200 Hz (暗い)
Bar 64: 5000 Hz (明るい)

徐々に:
4小節かけて上昇

Resonance:

Bar 60: 10%
Bar 64: 60%

同時に上昇

効果:
緊張感
期待感

Drop:

Bar 65: Cutoff 5000 Hz維持
または
完全にBypass

結果:
開放感
```

### 実践設定

```
Techno Buildup (8小節):

Bar 57-65:

1. Auto Filter挿入 (Pad)

2. Cutoff Automation:
   Bar 57: 300 Hz
   Bar 65: 8000 Hz
   直線的上昇

3. Resonance Automation:
   Bar 57: 20%
   Bar 65: 70%

4. LFO Amount:
   Bar 57: 0%
   Bar 61: 50%
   うねり追加

Drop (Bar 65):

Auto Filter: Bypass
または
Cutoff 8000 Hz固定

効果:
最大の開放感
```

---

## Send Automation

**空間変化:**

### Reverb Send

```
用途:

Buildup:
Send増加
広がり
浮遊感

Drop:
Send減少
タイト

設定例:

Snare Reverb Send:

Verse: 25%
Buildup Bar 60-64:
25% → 60% (増加)

Drop Bar 65:
60% → 25% (戻す)
または
60% → 80% (さらに広げる)

Vocal Delay Send:

Verse: 20%
Buildup: 20% → 50%
Drop: 20% (戻す)

効果:
劇的な空間変化
```

---

## Pan Automation

**動き:**

### Auto Pan的

```
用途:

Hi-Hat:
L/R往復

Percussion:
動き

設定:

Hi-Hat Pan:

Bar 1-32: Center (0%)
Bar 32-64:
L -50% ⇔ R +50%
4拍ごと

または:

Automation:
手動でL/R描画

推奨:
Auto Pan使用
Automation不要

例外:

FX:
極端な動き
L -100% → R +100%
1小節

効果:
劇的
```

---

## ビルドアップ完全ガイド

**8小節テンプレート:**

### 標準構成

```
Bar 57-65 (8小節):

Phase 1 (Bar 57-60, 4小節):

Pad Filter:
300 Hz → 1000 Hz

Percussion Volume:
-12 dB → -15 dB (減少)

FX Volume:
-∞ → -12 dB (出現)

Phase 2 (Bar 61-64, 4小節):

Pad Filter:
1000 Hz → 8000 Hz (加速)

Pad Resonance:
20% → 70%

Snare Send:
25% → 60%

Kick Volume:
Bar 64: -∞ dB (消える)

White Noise:
-∞ → -6 dB (Riser)

Phase 3 (Drop, Bar 65):

全て:
突然変化

Kick: 復帰 (-6 dB)
Pad Filter: Bypass
FX: -∞ (消える)
White Noise: -∞

効果:
完璧なBuildup → Drop
```

---

## ドロップ演出

**劇的変化:**

### Drop Automation

```
直前 (Bar 64.4):

全楽器:
-∞ dB (瞬間停止)
0.25拍

または:

Lowpass Filter:
全体に0 Hz
一瞬

Drop (Bar 65.1):

Kick:
復帰 -6 dB

Bass:
復帰 -9 dB

全楽器:
復帰

Reverb Send:
瞬間増加 50% → 戻す 25%
残響のみ一瞬

効果:
最大のインパクト
```

---

## Macro Knob Automation

**効率的:**

### 複数パラメーター同時

```
Audio Effect Rack:

Macro 1 (Buildup):

Map:
- Pad Filter Cutoff: 300 → 8000 Hz
- Resonance: 20 → 70%
- Reverb Send: 25 → 60%

Automation:

Macro 1のみ:
0% → 100%

結果:
3つ同時変化

メリット:

効率:
1つのAutomation

管理:
簡単

推奨:
上級者
```

---

## Clip Automation vs Track Automation

**違い:**

### 比較

```
Clip Automation:

場所:
MIDI/Audio Clip内

特徴:
Clip移動で一緒に移動

用途:
Clip固有
ループ

Track Automation:

場所:
Track (Arrangement)

特徴:
固定位置

用途:
楽曲全体
Buildup・Drop

推奨:

Buildup・Drop:
Track Automation

ループ:
Clip Automation
```

---

## よくある失敗

### 1. Automation急すぎ

```
問題:
不自然
機械的

原因:
直線、急激

解決:

カーブ使用:
Shift + ドラッグ

時間:
2-8小節かけて

推奨:
緩やかに
```

### 2. Automation多すぎ

```
問題:
落ち着かない
うるさい

原因:
全パラメーターAutomation

解決:

厳選:
2-3パラメーターのみ

重要:
Filter・Volume・Send

推奨:
Less is More
```

### 3. Drop後処理なし

```
問題:
Drop後平坦

原因:
Drop瞬間のみ

解決:

Drop後:
徐々に変化
次のセクションへ

例:
Drop後8小節
Send徐々に減少

効果:
自然な流れ
```

### 4. Automation録音失敗

```
問題:
意図しない変化

原因:
Automation Mode: Touch/Latch

解決:

Mode: Off (描画)
または
Read (再生のみ)

録音:
Write/Touch

完成後:
Read

推奨:
描画 (マウス)
確実
```

---

## Automation Mode

**Ableton:**

### モード説明

```
Off:

Automation: 無視
手動操作可

Read (デフォルト):

Automation: 再生
手動操作不可

Touch:

再生: Automation
Touch時: 手動
離す: Automation戻る

Latch:

Touch時: 手動
離す: 維持

Write:

全て上書き録音
危険

推奨:

通常: Read
描画: Off → Draw Mode
録音: Touch
```

---

## 実践ワークフロー

**30分で完成:**

### Step-by-Step

```
0-10分: Volume Automation

1. Intro Fadeセクション音量調整
2. Breakdown要素削除
3. Drop復帰設定

10-20分: Filter Automation

1. Pad Auto Filter挿入
2. Bar 57-65 Cutoff
   300 Hz → 8000 Hz
3. Resonance 20% → 70%

20-25分: Send Automation

1. Buildup Reverb Send増加
2. Drop戻す
3. Delay Send調整

25-30分: 確認

1. 全体再生
2. 各Automation確認
3. Drop効果確認
4. 微調整
```

---

## まとめ

### Automation

```
□ Volume: セクション別調整
□ Filter: Buildup必須 (300 → 8000 Hz)
□ Send: 空間変化 (Reverb・Delay)
□ Pan: 動き (Hi-Hat・FX)
□ Drop: 劇的変化
```

### ビルドアップ

```
8小節テンプレート:
Bar 57-60: Phase 1 (緩やか)
Bar 61-64: Phase 2 (加速)
Bar 65: Drop (劇的変化)
```

### 重要原則

```
□ 緩やかな変化 (2-8小節)
□ 厳選 (2-3パラメーターのみ)
□ Macro Knob活用
□ カーブ使用
□ Drop後も処理
```

---

**次は:** [Reference Mixing](./reference-mixing.md) - リファレンストラックでプロレベル到達
