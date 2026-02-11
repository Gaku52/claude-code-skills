# オーディオインターフェイス

DJ用オーディオインターフェイスの選び方とセットアップを学びます。

## この章で学ぶこと

- オーディオインターフェイスとは何か
- なぜDJに必要か
- 接続方式の違い（USB-A、USB-C、Thunderbolt）
- チャンネル数とDVS対応
- おすすめモデル
- Rekordboxでの設定
- レイテンシー調整
- トラブルシューティング

## なぜオーディオインターフェイスが必要か

**PC内蔵サウンドカードの限界:**
```
✗ レイテンシー（遅延）が大きい
✗ 音質が悪い
✗ 入出力チャンネル不足
✗ DVS不可
```

**専用オーディオIFのメリット:**
```
✓ 低レイテンシー（5-10ms）
✓ 高音質（24bit/96kHz）
✓ 複数チャンネル
✓ DVS対応（モデルによる）
✓ 安定動作
```

---

## 1. オーディオインターフェイスとは

### 定義

```
オーディオインターフェイス（Audio Interface）
= PC/Mac ⇔ 音響機器の橋渡し

役割:
- デジタル → アナログ変換（DAC）
- アナログ → デジタル変換（ADC）
- 低レイテンシー処理
```

### DJ用途

**コントローラーなし構成:**
```
PC/Mac
  ↓ USB
Audio Interface
  ↓ RCA/TRS
Mixer → Speakers
```

**DVS構成:**
```
Turntables
  ↓ Phono
Mixer
  ↓ RCA
Audio Interface（DVS対応）
  ↓ USB
PC/Mac（Rekordbox DVS）
```

---

## 2. 接続方式の違い

### USB-A（USB 2.0/3.0）

**特徴:**
```
✓ 最も一般的
✓ 対応PC多い
✓ 安定性高い
✓ ケーブル長めでもOK

レイテンシー: 5-10ms
帯域: 十分（DJ用途）
```

### USB-C（USB 3.1/3.2）

**特徴:**
```
✓ 最新標準
✓ 高速転送
✓ 将来性高い
✓ MacBook Air/Pro標準

レイテンシー: 3-8ms
帯域: 余裕
注意: アダプタ必要な場合あり
```

### Thunderbolt 3/4

**特徴:**
```
✓ 最速
✓ 最低レイテンシー
✓ プロスタジオ用
✗ 高価
✗ Mac専用（ほぼ）

レイテンシー: 2-5ms
帯域: 40Gbps（オーバースペック）
```

---

## 3. チャンネル数とDVS対応

### 2 in / 2 out（基本）

**用途:**
```
- コントローラーなしDJ
- Rekordbox Performance Mode
- 1台のターンテーブル（DVS）

構成:
Input: Mixer Return
Output: Master Out
```

### 4 in / 4 out（DVS推奨）

**用途:**
```
- DVS（2デッキ）
- ターンテーブル × 2

構成:
Input 1-2: Turntable 1
Input 3-4: Turntable 2
Output 1-2: Deck A → Mixer
Output 3-4: Deck B → Mixer
```

### 8 in / 8 out（プロ）

**用途:**
```
- 4デッキDVS
- 複雑なルーティング
- ライブセット統合
```

---

## 4. おすすめモデル

### Pioneer DJ INTERFACE 2（DVS入門）

**価格:** 約2.5万円

**スペック:**
```
- 2 in / 2 out（DVS: 4 in / 4 out）
- USB-C + USB-A
- Rekordbox DVS対応
- Phono/Line入力
- MIDI対応
```

**おすすめポイント:**
```
✓ Rekordbox完璧対応
✓ DVS対応
✓ コンパクト
✓ コスパ良い
```

### Rane SL3（DVS定番）

**価格:** 約4万円

**スペック:**
```
- 4 in / 4 out
- USB 2.0
- Serato DVS対応
- Phono/Line入力
- 業界標準
```

**おすすめポイント:**
```
✓ Serato DVS最適化
✓ 超安定動作
✓ プロ使用率高い
✓ 中古市場豊富
```

### Focusrite Scarlett 2i2（制作兼用）

**価格:** 約1.5万円

**スペック:**
```
- 2 in / 2 out
- USB-C
- 24bit/192kHz
- コンボジャック × 2
- ヘッドフォン出力
```

**おすすめポイント:**
```
✓ コスパ最高
✓ 音質良い
✓ 制作にも使える
✗ DVS非対応
```

### Native Instruments Traktor Audio 6（Traktor専用）

**価格:** 約3万円

**スペック:**
```
- 4 in / 4 out
- USB 2.0
- Traktor専用設計
- DVS対応
```

### RME Babyface Pro FS（プロ）

**価格:** 約12万円

**スペック:**
```
- 4 in / 4 out（ADAT拡張可能）
- USB 3.0 / USB-C
- 24bit/192kHz
- 超低レイテンシー（<2ms）
- TotalMix FX
```

**おすすめポイント:**
```
✓ 最高音質
✓ 最低レイテンシー
✓ 制作/DJどちらもプロ級
✗ 高価
```

---

## 5. セットアップ手順

### Windows

**Step 1: ドライバインストール**
```
1. メーカーサイトから最新ドライバDL
2. インストーラー実行
3. PC再起動
4. デバイスマネージャーで確認
```

**Step 2: 接続**
```
1. Audio IFをUSB接続
2. 電源確認（LEDランプ）
3. Windows認識確認
```

**Step 3: ASIO設定**
```
1. ASIO4ALL インストール（推奨）
2. Rekordbox起動
3. [Preferences] → [Audio]
4. Audio Device: ASIO4ALL選択
5. Buffer Size: 512 samples
```

### macOS

**Step 1: 接続（ドライバ不要）**
```
1. Audio IFをUSB接続
2. 自動認識（Core Audio）
3. システム環境設定で確認
```

**Step 2: Rekordbox設定**
```
1. Rekordbox起動
2. [Preferences] → [Audio]
3. Audio Device: Audio IF名選択
4. Buffer Size: 512 samples
5. Sample Rate: 44100Hz
```

---

## 6. Rekordbox 7.2.7での設定

### 基本設定

**Preferences → Audio**
```
Audio Device: [Audio IF名]
  - Pioneer DJ INTERFACE 2
  - Focusrite Scarlett 2i2
  - 等

Sample Rate: 44100Hz推奨
  - 48000Hzも可
  - 96000Hzは不要（負荷大）

Buffer Size: 512 samples推奨
  - 256: 低レイテンシーだが負荷大
  - 512: バランス良い
  - 1024: 安定だがレイテンシー大
```

### Output設定

**Master Out:**
```
Output Channel: 1-2（通常）
→ スピーカー/アンプへ

Booth: 使用しない（通常）
Headphones: 3-4（4ch以上のIF）
```

### DVS設定（INTERFACE 2使用時）

**Preferences → DVS**
```
Input:
- CH1: Input 1-2（Turntable 1）
- CH2: Input 3-4（Turntable 2）

Input Type: Phono/Line切り替え
Position: RELATIVE推奨
Tracking: Medium推奨
```

---

## 7. レイテンシー調整

### レイテンシーとは

```
レイテンシー = 処理遅延時間

許容範囲（DJ用途）:
- 10ms以下: 理想
- 10-20ms: 許容範囲
- 20ms以上: 違和感あり
```

### 最適化手順

**Step 1: Buffer Size調整**
```
Rekordbox → Audio設定
Buffer Size: 512 samples

レイテンシー確認:
256 samples: ~6ms（負荷大）
512 samples: ~12ms（バランス）
1024 samples: ~23ms（安定）
```

**Step 2: Sample Rate**
```
44100Hz推奨
→ CDと同じ
→ 負荷小さい

48000Hz: プロ用
96000Hz: 不要（DJ用途）
```

**Step 3: PC/Mac最適化**
```
Windows:
- 電源プラン: 高パフォーマンス
- バックグラウンドアプリ終了
- Wifi OFF（有線推奨）

macOS:
- 省エネルギー: 無効
- Spotlight: 一時停止
- 不要アプリ終了
```

---

## 8. 接続パターン

### パターン1: コントローラーなし構成

```
PC/Mac
  ↓ USB
Audio Interface
  ↓ RCA/TRS
Active Speakers

用途:
- Rekordbox Performance Mode
- ライブセット
- 制作モニタリング
```

### パターン2: DVS構成（2デッキ）

```
Turntables × 2
  ↓ Phono
DJ Mixer
  ↓ Send (RCA)
Audio Interface（Input 1-4）
  ↓ USB
PC/Mac（Rekordbox DVS）
  ↓ USB
Audio Interface（Output 1-4）
  ↓ Return (RCA)
DJ Mixer
  ↓ Master Out
PA System
```

### パターン3: ハイブリッド構成

```
CDJ × 1 + Turntable × 1 (DVS)
  ↓
DJ Mixer (4ch)
  ↓
Audio Interface
  ↓
PC/Mac

用途:
- CDJとDVS併用
- 柔軟なセットアップ
```

---

## 9. トラブルシューティング

### 音が出ない

**チェック項目:**
```
□ Audio IF認識確認
□ ドライバ最新版
□ Rekordbox Audio設定確認
□ Output Channel正しいか
□ ケーブル接続確認
□ スピーカー電源ON
□ マスター音量確認
```

### ノイズが入る

**対処法:**
```
□ USBケーブル交換（品質良いもの）
□ USB Hub使わない（直接接続）
□ アース接続確認
□ 他のUSB機器を外す
□ 電源ノイズ分離（別コンセント）
```

### レイテンシーが大きい

**対処法:**
```
□ Buffer Size下げる（512→256）
□ Sample Rate下げる（96k→44.1k）
□ バックグラウンドアプリ終了
□ ドライバ再インストール
□ ASIO4ALL使用（Windows）
```

### DVSが追従しない

**対処法:**
```
□ 針圧確認（3g推奨）
□ タイムコードバイナル状態確認
□ Calibration実行
□ Tracking設定変更
□ Input Type（Phono/Line）確認
```

---

## 10. 予算別おすすめ

### 1.5万円: Focusrite Scarlett 2i2

```
用途: DJ + 制作
✓ 高音質
✓ 制作でも使える
✗ DVS非対応
```

### 2.5万円: Pioneer DJ INTERFACE 2

```
用途: Rekordbox DVS
✓ DVS対応
✓ コンパクト
✓ Rekordbox最適化
```

### 4万円: Rane SL3

```
用途: Serato DVS
✓ 業界標準
✓ 超安定
✓ プロ仕様
```

### 12万円: RME Babyface Pro FS

```
用途: プロDJ + 制作
✓ 最高音質
✓ 最低レイテンシー
✓ 将来性◎
```

---

## まとめ

- **コントローラー**: 内蔵Audio IFで十分
- **DVS**: 4in/4out以上必須
- **接続**: USB-C推奨（将来性）
- **レイテンシー**: 512 samples（~12ms）が最適
- **Rekordbox**: 44100Hz、512 samplesで設定
- **入門**: Pioneer DJ INTERFACE 2（¥2.5万）
- **制作兼用**: Focusrite Scarlett 2i2（¥1.5万）
- **プロ**: RME Babyface Pro FS（¥12万）

**次のステップ:** [基本テクニック](../03-basic-techniques/README.md) でDJスキルを磨く

---

## 参考リンク

- [ターンテーブル](./turntables.md)
- [DJミキサー](./dj-mixers.md)
- [ソフトウェア比較](./software-comparison.md)
- [ビートマッチング](../03-basic-techniques/beatmatching.md)
