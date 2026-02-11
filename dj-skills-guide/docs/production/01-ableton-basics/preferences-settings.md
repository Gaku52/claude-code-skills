# 環境設定（Preferences）

Ableton Liveを最適化する。Preferences設定で制作効率とパフォーマンスを最大化します。

## この章で学ぶこと

- Preferences完全ガイド
- Look/Feel設定（見た目と操作感）
- File Folder設定（ファイル管理）
- Record Warp Launch設定（録音とWarp）
- CPU/RAM最適化
- キーボードショートカット
- ライセンスとLibrary

---

## なぜPreferences設定が重要なのか

**快適な制作環境:**

```
デフォルト設定:

万人向け:
平均的な設定

最適ではない:
あなたのPCに合っていない
あなたのワークフローに合っていない

最適化後:

高速:
CPU負荷軽減
サクサク動く

効率的:
ショートカット設定
作業時間短縮

安定:
クラッシュ防止

差:

設定前: 重い、不安定
設定後: 軽い、快適

時間:
5分の設定で
数百時間の効率化
```

---

## Preferencesを開く

**アクセス方法:**

```
Mac:
Live > Preferences
または
Cmd+, (カンマ)

Windows:
Options > Preferences
または
Ctrl+, (カンマ)

画面:

┌──────────────────────────┐
│ Preferences              │
├────────┬─────────────────┤
│ Look/  │ [設定内容]       │
│ Feel   │                 │
│ Audio  │                 │
│ Link/  │                 │
│ Tempo  │                 │
│ MIDI   │                 │
│ File/  │                 │
│ Folder │                 │
│ Librry │                 │
│ Record │                 │
│ ...    │                 │
└────────┴─────────────────┘

左: カテゴリ
右: 設定項目
```

---

## Look/Feel（見た目と操作感）

**最初に設定:**

### テーマ

```
Preferences > Look/Feel > Theme:

┌──────────────────┐
│ Dark (デフォルト) │ ← 推奨
│ Light            │
│ Mid-Dark         │
│ Mid-Light        │
└──────────────────┘

Dark:
目に優しい
長時間作業向き
スタジオの標準

Light:
明るい環境向き
昼間の作業

推奨:
Dark
→ プロの標準
→ 夜間作業に最適
```

### Brightness（明るさ）

```
スライダー:

Dark ←―●―――――――→ Bright

推奨:
中央やや暗め
→ 目が疲れない
```

### Language（言語）

```
言語選択:

English (推奨)
日本語

なぜEnglish:

チュートリアル:
99%が英語

ヘルプ:
英語版が充実

慣れ:
1ヶ月で慣れる

日本語:
初心者向き
でも後で英語に慣れる必要

妥協案:
最初: 日本語
慣れたら: English
```

### HiDPI Display Support (Mac)

```
Mac Retina ディスプレイ:

☑ HiDPI

効果:
画面が鮮明
文字が読みやすい

必須:
MacBook Pro/Air
iMac 4K/5K

Windowsの場合:
自動対応
設定不要
```

---

## File/Folder（ファイル管理）

**重要な設定:**

### Temporary Folder

```
Preferences > File/Folder > Temporary Folder:

デフォルト:
~/Library/Application Support/Ableton/Live/Temp

変更推奨:
外付けSSD
(システムドライブ以外)

理由:

録音ファイル:
一時的にここに保存
→ 大容量必要

SSD推奨:
高速アクセス

設定:

Browse...
→ 外付けSSD選択
例: /Volumes/Music SSD/Ableton Temp/
```

### Library

```
Preferences > File/Folder > Library:

User Library:
~/Music/Ableton/User Library/

ここに保存:
ダウンロードしたパック
自作プリセット
サンプル

変更:
外付けSSDに移動可能
(容量節約)

Install Packs:
パックのインストール先
```

### Plug-In Sources

```
VST Plugins:

Use VST Plug-In System Folders:
☑ On (推奨)

Custom Folder:
追加のVSTフォルダ指定可能

Mac:
~/Library/Audio/Plug-Ins/VST/
/Library/Audio/Plug-Ins/VST/

Windows:
C:\Program Files\VSTPlugins\

Re-scan:
新しいVST追加後
→ Re-scan ボタン
```

---

## Record Warp Launch

**録音とWarp設定:**

### Count-In

```
Preferences > Record Warp Launch > Count-In:

Count-In:
録音開始前のカウント

設定:

None: カウントなし
1 Bar: 1小節
2 Bars: 2小節 (推奨)
4 Bars: 4小節

推奨: 2 Bars
理由:
録音準備の時間
「1, 2, 3, 4, 1, 2...」

Metronome During Count-In:
☑ On
→ カウント中にクリック音
```

### Auto-Warp Long Samples

```
Auto-Warp Long Samples:
☑ On (推奨)

効果:

長いオーディオ:
自動でWarp
→ BPMに追従

短いサンプル:
Warpなし
(キック、スネア等)

境界:
デフォルト: 30秒以上

利点:
ボーカル、ループ
自動でテンポ合う
```

### Default Launch Mode

```
Launch Mode:
クリップの再生方法

Gate:
押している間だけ再生

Trigger:
クリックで再生開始 (推奨)

Toggle:
on/off切り替え

Repeat:
繰り返し再生

推奨: Trigger
理由:
一般的な使い方
```

---

## CPU/メモリ最適化

**パフォーマンス向上:**

### Multicore/Multiprocessor Support

```
Preferences > CPU > Multicore Support:

┌─────┐
│ On  │ ← 必須
└─────┘

効果:
複数CPUコア使用
→ 重いプロジェクトも快適

必須:
現代のPCはマルチコア
必ずOn

Offにする理由:
なし
```

### Buffer Size

```
Preferences > Audio > Buffer Size:

128 samples ← デフォルト
256 samples ← 推奨(制作時)
512 samples
1024 samples

設定:
次のセクション (audio-midi-setup.md) で詳細

簡単に:

小さい (64-128):
レイテンシー低い
→ リアルタイム演奏向き
→ CPU負荷高い

大きい (512-1024):
レイテンシー高い
→ ミックス向き
→ CPU負荷低い

切り替え:
録音時: 128
ミックス時: 512
```

### Freeze Tracks

```
トラックのFreeze:

右クリック > Freeze Track

効果:
重い音源・エフェクトを
一時的にオーディオ化
→ CPU負荷激減

いつ使う:
プロジェクトが重くなったら

解除:
Unfreeze
→ 再編集可能

自動:
Preferences で設定不要
手動でトラックごとに
```

### Reduce Latency When Monitoring

```
Preferences > Audio:

Reduce Latency When Monitoring:
☑ On

効果:
録音中のレイテンシー低減

必須:
ボーカル録音
ギター録音

不要:
MIDIキーボードのみ
```

---

## キーボードショートカット

**カスタマイズ:**

### ショートカット表示

```
Help > Show Keyboard Shortcuts

または:
Opt+Cmd+K (Mac)
Alt+Ctrl+K (Win)

画面:
キーボード全体の割り当て表示

便利:
よく使う機能を確認
```

### カスタマイズ方法

```
Preferences > MIDI > Key Map Mode:

Key ボタンクリック
→ オレンジ色に

割り当て:
1. 機能をクリック (例: Play)
2. キーを押す (例: F5)
→ 割り当て完了

削除:
機能クリック → Delete

保存:
Key ボタン再クリック
→ オレンジ解除

おすすめカスタマイズ:

F5: Play/Stop (デフォルトはSpace)
F9: Record (そのまま)
F12: Export (自分で設定)
```

---

## ライセンスとアカウント

**認証管理:**

### ライセンス認証

```
Preferences > Licenses:

Authorize with ableton.com:
Abletonアカウントでログイン

効果:
他のPCでも使用可能
(最大2台同時)

Deauthorize:
このPCの認証解除
→ 別PCで使う

オフライン認証:
インターネットなしで認証
(レアケース)
```

### User Account

```
ログイン:
Preferences > Account

Ableton Account:
メールアドレス
パスワード

できること:

パック追加:
購入したパックをDL

クラウド同期:
プリセット、設定

サポート:
フォーラム、ヘルプ
```

---

## その他の重要設定

### Auto Save

```
残念ながら:
Ableton Live に自動保存機能なし

対策:

手動保存:
Cmd+S 連打

Time Machine (Mac):
1時間ごとに自動バックアップ

File History (Win):
同様の機能

習慣:
5分ごとにCmd+S
```

### Undo Steps

```
Preferences > Edit:

Maximum Number of Undo Steps:

デフォルト: 64 (十分)
最大: 1000

推奨: 64
理由:
多すぎるとメモリ消費

Cmd+Z:
1つ戻る

Cmd+Shift+Z:
1つ進む
```

### Snap to Grid

```
Preferences > Record Warp Launch:

Global Quantization:

None
8 Bars
4 Bars
1 Bar (推奨)

効果:
クリップが小節頭にスナップ

便利:
きれいに整列

無効化:
Cmd (Mac) / Ctrl (Win)
押しながらドラッグ
→ 自由配置
```

---

## 実践: Preferences最適化

**30分の設定:**

### Step 1: Look/Feel (5分)

```
1. Preferences開く: Cmd+,

2. Look/Feel:
   Theme: Dark
   Brightness: 中央やや暗め
   Language: English (または日本語)

3. Mac: HiDPI ☑

4. 閉じる
```

### Step 2: File/Folder (10分)

```
1. File/Folder:

2. Temporary Folder:
   Browse...
   → 外付けSSD選択(あれば)
   なければデフォルトのまま

3. Library確認:
   User Library の場所メモ

4. Plug-In Sources:
   Use VST: ☑
   Re-scan (VSTインストール済みの場合)
```

### Step 3: Record Warp Launch (5分)

```
1. Record Warp Launch:

2. Count-In: 2 Bars
   Metronome During Count-In: ☑

3. Auto-Warp Long Samples: ☑

4. Default Launch Mode: Trigger
```

### Step 4: CPU (5分)

```
1. CPU:

2. Multicore Support: On

3. Audio:
   Buffer Size: 256 samples
   Reduce Latency: ☑
```

### Step 5: ショートカット確認 (5分)

```
1. Help > Show Keyboard Shortcuts
   または Opt+Cmd+K

2. よく使う機能確認:
   Space: Play/Stop
   F9: Record
   Cmd+S: Save
   Cmd+Z: Undo

3. 覚える
```

---

## よくある質問

### Q1: Preferencesが反映されない

**A:** Ableton再起動

```
設定変更後:

一部設定:
すぐ反映

Audio/CPU設定:
再起動必要

手順:
1. Preferences設定
2. File > Quit (Cmd+Q)
3. 再起動
4. 確認
```

### Q2: デフォルトに戻したい

**A:** Preferences削除

```
Mac:
~/Library/Preferences/Ableton/Live x.x/
→ フォルダ削除
→ 再起動

Windows:
C:\Users\[ユーザー名]\AppData\Roaming\Ableton\Live x.x\Preferences\
→ フォルダ削除
→ 再起動

注意:
全設定がリセット
ライセンス再認証必要

やる前に:
フォルダをバックアップ
```

### Q3: 他のPCに設定を移行したい

**A:** Preferencesフォルダコピー

```
Export:

元PC:
~/Library/Preferences/Ableton/Live x.x/
→ フォルダをUSBにコピー

Import:

新PC:
同じ場所にフォルダ貼り付け
→ 再起動

注意:
Liveバージョン一致必要
(11.x → 11.x)

ライセンス:
別途認証必要
```

---

## まとめ

### 必須設定

```
Look/Feel:
□ Theme: Dark
□ Language: English (または日本語)

File/Folder:
□ Temporary Folder設定
□ Plug-In Sources確認

Record Warp Launch:
□ Count-In: 2 Bars
□ Auto-Warp: On
□ Launch Mode: Trigger

CPU:
□ Multicore: On
□ Buffer Size: 256
□ Reduce Latency: On
```

### 設定完了後

```
1. Ableton再起動
2. 動作確認
3. プロジェクト作成して試す
4. 快適になったか確認
```

### ショートカット

```
Preferences: Cmd+,
Keyboard Map: Opt+Cmd+K
```

---

**次は:** [Audio/MIDI設定](./audio-midi-setup.md) - オーディオインターフェイス設定
