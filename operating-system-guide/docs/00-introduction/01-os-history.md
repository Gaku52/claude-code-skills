# OSの歴史と進化

> OSの歴史は「抽象化の積み重ね」の歴史である — ハードウェアの複雑さから人間を解放する戦いの記録。

## この章で学ぶこと

- [ ] OSの進化の主要なマイルストーンを知る
- [ ] バッチ処理→タイムシェアリング→GUIの流れを理解する
- [ ] 現代OSのルーツを辿れる

---

## 1. OSの進化年表

```
1950年代: バッチ処理の時代
  ┌──────────────────────────────────────────┐
  │ パンチカード → コンピュータ → 結果出力    │
  │ 人間がプログラムを1つずつ投入             │
  │ OSは存在しない（オペレータが手動管理）    │
  │                                          │
  │ 1956: GM-NAA I/O（最初のOS）             │
  │ → IBM 704用のバッチ処理システム           │
  │ → ジョブを自動的に次々実行               │
  └──────────────────────────────────────────┘

1960年代: マルチプログラミングとタイムシェアリング
  ┌──────────────────────────────────────────┐
  │ 1961: CTSS（Compatible Time-Sharing System）│
  │ → MIT。最初のタイムシェアリングOS         │
  │ → 複数ユーザーが同時にコンピュータを使用  │
  │                                          │
  │ 1964: Multics（MULTiplexed Information   │
  │        and Computing Service）            │
  │ → MIT + Bell Labs + GE                   │
  │ → 野心的すぎて商業的には失敗             │
  │ → しかしUnixに多大な影響を与えた         │
  │                                          │
  │ 1964: IBM OS/360                         │
  │ → 最初の汎用OS（複数機種で共通）         │
  │ → Fred Brooks "Mythical Man-Month"の題材 │
  └──────────────────────────────────────────┘

1970年代: Unixの誕生
  ┌──────────────────────────────────────────┐
  │ 1969: Unix（Ken Thompson & Dennis Ritchie）│
  │ → Multicsの反省から「シンプル」を追求     │
  │ → 最初はPDP-7上でアセンブラで記述        │
  │                                          │
  │ 1973: C言語でUnixを書き直し              │
  │ → 「OSを高級言語で書く」革命的な発想     │
  │ → 移植性が飛躍的に向上                   │
  │                                          │
  │ Unix哲学:                                │
  │ 1. 1つのことをうまくやる                 │
  │ 2. テキストストリームで連携              │
  │ 3. 小さなツールを組み合わせる            │
  └──────────────────────────────────────────┘

1980年代: PCとGUI
  ┌──────────────────────────────────────────┐
  │ 1981: MS-DOS（Microsoft）                │
  │ → IBM PC用。コマンドライン               │
  │                                          │
  │ 1984: Macintosh（Apple）                 │
  │ → 初の商業的成功を収めたGUI OS           │
  │ → Xerox PARCのAltoに着想                 │
  │                                          │
  │ 1987: MINIX（Andrew Tanenbaum）          │
  │ → 教育用マイクロカーネルOS               │
  │ → Linusに影響を与えた                    │
  └──────────────────────────────────────────┘

1990年代: Linux, Windows, Web
  ┌──────────────────────────────────────────┐
  │ 1991: Linux（Linus Torvalds）            │
  │ → 「趣味のOS」として開始                 │
  │ → GPLライセンスでオープンソース           │
  │ → 世界中の開発者が貢献                   │
  │                                          │
  │ 1993: Windows NT                         │
  │ → 本格的な32bit OS（Dave Cutler設計）    │
  │ → 現在のWindowsの基盤                    │
  │                                          │
  │ 1995: Windows 95                         │
  │ → GUIの普及、Start Menu                  │
  │ → PCの一般家庭への普及                   │
  └──────────────────────────────────────────┘

2000年代〜: モバイルとクラウド
  ┌──────────────────────────────────────────┐
  │ 2001: Mac OS X（macOS）                  │
  │ → NeXTSTEP + FreeBSD = Darwin            │
  │ → Unix基盤の商用デスクトップOS           │
  │                                          │
  │ 2007: iPhone OS（iOS）                   │
  │ → モバイルOS時代の幕開け                 │
  │                                          │
  │ 2008: Android                            │
  │ → Linux カーネル上に構築                  │
  │ → 世界最大のモバイルOS                   │
  │                                          │
  │ 2013: Docker                             │
  │ → コンテナ技術でOS仮想化を革新           │
  │                                          │
  │ 2020: Apple Silicon（M1）                │
  │ → ARM + macOS でPC性能の常識を覆す       │
  └──────────────────────────────────────────┘
```

---

## 2. 重要な概念の進化

```
マルチタスクの進化:

  バッチ処理（1950s）:
  Job1 ──────→ Job2 ──────→ Job3
  → 1つずつ順番に実行

  マルチプログラミング（1960s）:
  Job1 ██░░██░░██
  Job2 ░░██░░██░░
  → I/O待ちの間に他のジョブを実行

  タイムシェアリング（1960s）:
  User1 █░░█░░█░░
  User2 ░█░░█░░█░
  User3 ░░█░░█░░█
  → 各ユーザーに短い時間を交互に割り当て

  プリエンプティブマルチタスク（1990s〜）:
  → OSがプロセスを強制的に切り替え
  → 1つのプロセスが暴走しても他に影響しない
  → 現代の標準

メモリ管理の進化:

  固定パーティション → 可変パーティション → ページング → 仮想メモリ
  → 各プロセスが独立した広大なアドレス空間を持つ
  → 物理メモリ以上のメモリを使用可能
```

---

## 3. OSの系譜図

```
  Multics (1964)
    │
    ├──→ Unix (1969) ──────────────────────────────────┐
    │      ├── BSD (1977) ──→ FreeBSD ──→ macOS/iOS   │
    │      ├── System V ──→ Solaris, AIX              │
    │      └── Philosophy ──→ GNU (1983)               │
    │                          └── + Linux (1991)      │
    │                               ├── Ubuntu         │
    │                               ├── RHEL           │
    │                               ├── Android        │
    │                               └── Chrome OS      │
    │
  CP/M (1974) ──→ MS-DOS (1981) ──→ Windows 95/98/Me │
                                                       │
  VMS (1977) ──→ Windows NT (1993) ──→ Win 2000/XP    │
                                    ──→ Win 7/10/11    │
                                                       │
  Xerox Alto (1973) ──→ Macintosh (1984)               │
                    ──→ Windows GUI                     │
                                                       │
  NeXTSTEP (1989) ──→ Mac OS X (2001) ──→ macOS       │
```

---

## 実践演習

### 演習1: [基礎] — OS情報の確認

```bash
# 自分のOSの情報を確認
# Linux:
uname -a
cat /etc/os-release
cat /proc/version

# macOS:
sw_vers
uname -a
system_profiler SPSoftwareDataType

# カーネルバージョン、ビルド日、アーキテクチャを記録せよ
```

### 演習2: [応用] — Unix哲学の実践

```bash
# パイプで小さなツールを組み合わせて以下を実現せよ:

# 1. /etc/passwdからシェルの使用統計を取る
cat /etc/passwd | cut -d: -f7 | sort | uniq -c | sort -rn

# 2. 自分で同様のパイプラインを3つ考えて実行せよ
# ヒント: ps, netstat, df, du, wc, grep, awk を活用
```

---

## FAQ

### Q1: なぜLinuxには多くのディストリビューションがあるのか？

Linuxカーネルはオープンソースであるため、誰でもカーネル+独自のパッケージ管理+独自の設定を組み合わせてディストリビューションを作れる。目的に応じた最適化が異なるため多様性が生まれた（サーバー向けRHEL、デスクトップ向けUbuntu、セキュリティ向けKali等）。

### Q2: macOSはなぜUnixベースなのか？

Apple がNeXT社を買収（1997年）した際に、Steve Jobsが開発したNeXTSTEP（Mach+BSD）が基盤となった。Darwin（macOSのカーネル）はMachマイクロカーネル+FreeBSDコンポーネントで構成され、正式にUNIX 03認証を取得している。

### Q3: Windows NTとWindows 95は何が違ったのか？

Windows 95はMS-DOS上に構築された16/32bitハイブリッドで不安定だった。一方Windows NTはVMS設計者Dave Cutlerがゼロから設計した完全な32bit OSで、メモリ保護、プリエンプティブマルチタスク、NTFSを備えた。現在のWindows 11はNT系の子孫。

---

## まとめ

| 時代 | 革新 | 代表OS |
|------|------|--------|
| 1950s | バッチ処理 | GM-NAA I/O |
| 1960s | タイムシェアリング | Multics, CTSS |
| 1970s | Unix誕生、C言語 | Unix |
| 1980s | PC、GUI | MS-DOS, Macintosh |
| 1990s | オープンソース | Linux, Windows NT |
| 2000s | モバイル、クラウド | iOS, Android, Docker |

---

## 次に読むべきガイド
→ [[../01-process-management/00-processes.md]] — プロセスの概念

---

## 参考文献
1. Ritchie, D. & Thompson, K. "The UNIX Time-Sharing System." CACM, 1974.
2. Raymond, E. "The Art of Unix Programming." Addison-Wesley, 2003.
3. Campbell-Kelly, M. "From Airline Reservations to Sonic the Hedgehog: A History of the Software Industry." MIT Press, 2003.
