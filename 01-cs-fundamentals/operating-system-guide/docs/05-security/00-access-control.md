# アクセス制御

> OSのセキュリティの基盤は「誰が何にアクセスできるか」を厳密に管理するアクセス制御である。

## この章で学ぶこと

- [ ] DAC、MAC、RBACの違いを説明できる
- [ ] Unixパーミッションモデルを理解する
- [ ] 最小権限の原則を実践できる
- [ ] ACL（アクセス制御リスト）の設計と運用ができる
- [ ] SELinux / AppArmor のポリシーを設定できる
- [ ] Linux Capabilities を活用して権限を最小化できる
- [ ] ABAC（属性ベースアクセス制御）の概念を理解する
- [ ] 実務でのアクセス制御監査を行える

---

## 1. アクセス制御モデル

### 1.1 アクセス制御の基本概念

```
アクセス制御の3要素（AAA）:

  1. 認証（Authentication）:
     「あなたは誰か？」を確認
     → パスワード、生体認証、証明書、MFA
     → OS ログイン時の /etc/passwd + /etc/shadow
     → PAM（Pluggable Authentication Modules）による統一認証

  2. 認可（Authorization）:
     「あなたは何ができるか？」を判定
     → パーミッション、ACL、セキュリティポリシー
     → カーネル内のセキュリティチェック

  3. 監査（Accounting / Auditing）:
     「あなたは何をしたか？」を記録
     → auditd, syslog, journald
     → コンプライアンス要件への対応

アクセス制御の判定フロー:
  ユーザーリクエスト
       ↓
  [認証] → 失敗 → アクセス拒否
       ↓ 成功
  [認可チェック] → 拒否 → アクセス拒否 + ログ記録
       ↓ 許可
  リソースアクセス + 監査ログ記録

セキュリティ参照モニタ（Reference Monitor）:
  カーネル内でアクセス制御を強制する仕組み
  要件:
  1. 完全仲介（Complete Mediation）: すべてのアクセスをチェック
  2. タンパー耐性（Tamper Proof）: 改ざん不可能
  3. 検証可能性（Verifiable）: 正しさを検証可能
  → Linuxのセキュリティフック（LSM: Linux Security Modules）が実現
```

### 1.2 主要なアクセス制御モデル

```
3つのアクセス制御モデル:

  1. DAC（Discretionary Access Control）— 任意アクセス制御:
     ファイルの所有者がアクセス権を自由に設定
     → Unixの標準パーミッション（rwx）
     → Windows NTFS のACL
     → 柔軟だが、所有者が誤った設定をすると脆弱

     DAC の特徴:
     - リソースの所有者がアクセス権を「任意に」設定できる
     - 所有者は他ユーザーに権限を委譲可能
     - 権限の伝播が制御困難（コピー問題）
     - Trojan Horse 攻撃に脆弱
       → 悪意あるプログラムがユーザーの権限で動作
       → ユーザーが読めるファイルをすべて盗める

     DAC の実装例:
     ┌──────────────────────────────────────────────┐
     │ Unix パーミッション                          │
     │ → owner / group / other の3段階             │
     │ → read(4), write(2), execute(1)             │
     │ → シンプルだが表現力に限界                   │
     │                                              │
     │ POSIX ACL                                    │
     │ → 特定ユーザー/グループに個別権限設定        │
     │ → Unix パーミッションを拡張                  │
     │                                              │
     │ Windows DACL                                 │
     │ → セキュリティ記述子による詳細な制御          │
     │ → ACE（Access Control Entry）の集合          │
     │ → 継承と拒否ルールのサポート                 │
     └──────────────────────────────────────────────┘

  2. MAC（Mandatory Access Control）— 強制アクセス制御:
     システム管理者が定めたポリシーに従う（所有者でも変更不可）
     → SELinux, AppArmor
     → 軍事・政府システムで必須
     → Bell-LaPadula モデル: 上位の情報は下位に読めない

     MAC の理論的基盤:
     ┌──────────────────────────────────────────────────┐
     │ Bell-LaPadula モデル（機密性重視）:              │
     │                                                  │
     │ セキュリティレベル:                              │
     │   Top Secret > Secret > Confidential > Unclass   │
     │                                                  │
     │ Simple Security Property（ss-property）:         │
     │   「No Read Up」: 上位レベルのデータを読めない   │
     │   → Confidential ユーザーは Secret を読めない   │
     │                                                  │
     │ *-property（Star Property）:                     │
     │   「No Write Down」: 下位レベルに書き込めない    │
     │   → Secret ユーザーは Confidential に書けない   │
     │   → 情報の漏洩を防止                            │
     │                                                  │
     │ Strong Star Property:                            │
     │   同一レベルのみ読み書き可能                     │
     └──────────────────────────────────────────────────┘

     ┌──────────────────────────────────────────────────┐
     │ Biba モデル（完全性重視）:                       │
     │                                                  │
     │ Simple Integrity Axiom:                          │
     │   「No Read Down」: 下位レベルのデータを読めない │
     │   → 信頼性の低いデータによる汚染を防止          │
     │                                                  │
     │ *-Integrity Axiom:                               │
     │   「No Write Up」: 上位レベルに書き込めない      │
     │   → 信頼性の低い主体による改ざんを防止          │
     │                                                  │
     │ Bell-LaPadula と Biba は正反対:                  │
     │   機密性 vs 完全性のトレードオフ                 │
     └──────────────────────────────────────────────────┘

     ┌──────────────────────────────────────────────────┐
     │ Clark-Wilson モデル（商用向け完全性）:           │
     │                                                  │
     │ Well-Formed Transaction:                         │
     │   データは検証済みプログラムのみが変更可能       │
     │   → 銀行取引、在庫管理など商用システム向け      │
     │                                                  │
     │ Separation of Duty:                              │
     │   1人のユーザーがすべての処理を行えない          │
     │   → 不正防止のための職務分離                    │
     └──────────────────────────────────────────────────┘

  3. RBAC（Role-Based Access Control）— ロールベース:
     ユーザーにロール（役割）を割り当て、ロールに権限を付与
     → 管理者、開発者、閲覧者 等の役割
     → AWS IAM, Kubernetes RBAC
     → 大規模組織の標準

     RBAC のコンポーネント:
     ┌──────────────────────────────────────────────┐
     │ User（ユーザー）                             │
     │   ↓ 割り当て                                │
     │ Role（ロール）                               │
     │   ↓ 付与                                    │
     │ Permission（パーミッション）                  │
     │   ↓ 適用                                    │
     │ Object（リソース）                           │
     └──────────────────────────────────────────────┘

     RBAC の階層（RBAC0〜RBAC3）:
     - RBAC0（Flat RBAC）: 基本的なロール割り当て
     - RBAC1（Hierarchical RBAC）: ロール継承
       → 「管理者」は「開発者」の権限を継承
     - RBAC2（Constrained RBAC）: 制約付き
       → 相互排他ロール（SoD: Separation of Duties）
       → 1人が「承認者」と「申請者」を兼ねない
     - RBAC3: RBAC1 + RBAC2 の統合

  比較:
  ┌──────┬──────────────┬──────────┬──────────────┐
  │ モデル│ 決定者       │ 柔軟性   │ セキュリティ  │
  ├──────┼──────────────┼──────────┼──────────────┤
  │ DAC  │ ファイル所有者│ 高       │ 中            │
  │ MAC  │ システム管理者│ 低       │ 高            │
  │ RBAC │ 管理者(ロール)│ 中       │ 高            │
  └──────┴──────────────┴──────────┴──────────────┘
```

### 1.3 ABAC（Attribute-Based Access Control）

```
ABAC（属性ベースアクセス制御）:
  ユーザー、リソース、環境の「属性」に基づいてアクセスを判定
  → RBAC の拡張として登場
  → NIST SP 800-162 で標準化
  → より柔軟できめ細かい制御が可能

  属性の種類:
  ┌────────────────────────────────────────────────────┐
  │ Subject Attributes（主体属性）:                    │
  │   → 部署、役職、クリアランスレベル、勤続年数      │
  │                                                    │
  │ Object Attributes（客体属性）:                     │
  │   → ファイル種別、機密レベル、作成日時             │
  │                                                    │
  │ Environment Attributes（環境属性）:                │
  │   → アクセス時刻、場所（IP）、デバイス種別        │
  │                                                    │
  │ Action Attributes（操作属性）:                     │
  │   → 読み取り、書き込み、削除、実行                │
  └────────────────────────────────────────────────────┘

  ポリシー例:
  「部署=経理部 AND 役職=マネージャー以上 AND
   時間=営業時間内 AND 場所=社内ネットワーク
   → 財務レポートの読み取り/書き込みを許可」

  XACML（eXtensible Access Control Markup Language）:
  ABACポリシーの標準記述言語
  ┌──────────────────────────────────────────────────┐
  │ PEP (Policy Enforcement Point):                  │
  │   アクセス要求を受け取り、判定結果を強制する      │
  │                                                    │
  │ PDP (Policy Decision Point):                      │
  │   ポリシーに基づいてアクセスの可否を判定する      │
  │                                                    │
  │ PAP (Policy Administration Point):                │
  │   ポリシーを管理・配布する                        │
  │                                                    │
  │ PIP (Policy Information Point):                   │
  │   属性情報を提供する                              │
  └──────────────────────────────────────────────────┘

  ABAC vs RBAC:
  ┌─────────┬──────────────────┬──────────────────┐
  │ 項目    │ RBAC             │ ABAC             │
  ├─────────┼──────────────────┼──────────────────┤
  │ 粒度    │ ロール単位       │ 属性の組み合わせ │
  │ 柔軟性  │ 中               │ 高               │
  │ 管理    │ 比較的容易       │ 複雑             │
  │ 動的制御│ 限定的           │ リアルタイム     │
  │ 適用    │ 社内システム     │ クラウド、IoT    │
  │ 標準    │ NIST SP 800-xx   │ XACML, ALFA     │
  └─────────┴──────────────────┴──────────────────┘
```

### 1.4 ReBAC（Relationship-Based Access Control）

```
ReBAC（関係ベースアクセス制御）:
  エンティティ間の「関係性」に基づいてアクセスを判定
  → Google Zanzibar（Google Drive, YouTube の認可基盤）
  → OpenFGA, SpiceDB が OSS 実装

  基本概念:
  ┌────────────────────────────────────────────────────┐
  │ Tuple（タプル）:                                   │
  │   user:alice → relation:viewer → object:doc:123   │
  │   「aliceはdoc:123のviewerである」                 │
  │                                                    │
  │ 関係の推移:                                        │
  │   group:engineering#member → relation:viewer       │
  │   → object:folder:shared                           │
  │   user:alice → relation:member → group:engineering│
  │   ∴ alice は folder:shared の viewer              │
  └────────────────────────────────────────────────────┘

  Zanzibar のアーキテクチャ:
  - 超大規模対応（Google 全サービスの認可）
  - 低レイテンシ（99th percentile < 10ms）
  - 強整合性（New Enemy Problem の解決）
  - グラフベースの関係探索

  実務での適用:
  → SNS のフォロー/フォロワー権限
  → ファイル共有の継承（フォルダ → ファイル）
  → 組織階層に基づく権限（部門長 → メンバーのリソース）
```

---

## 2. Unixパーミッション

### 2.1 基本パーミッション

```
パーミッションの表示:
  $ ls -la
  -rwxr-xr-- 1 user group 4096 Jan 1 file.txt
  │├─┤├─┤├─┤
  │ │  │  └── other（その他）: r-- (読み取りのみ)
  │ │  └── group（グループ）: r-x (読取+実行)
  │ └── owner（所有者）: rwx (読取+書込+実行)
  └── タイプ: - (ファイル), d (ディレクトリ), l (リンク)

  ファイルタイプの一覧:
  ┌──────┬──────────────────────────────────┐
  │ 記号 │ 種類                             │
  ├──────┼──────────────────────────────────┤
  │ -    │ 通常ファイル                     │
  │ d    │ ディレクトリ                     │
  │ l    │ シンボリックリンク               │
  │ c    │ キャラクターデバイス             │
  │ b    │ ブロックデバイス                 │
  │ p    │ 名前付きパイプ（FIFO）           │
  │ s    │ ソケット                         │
  └──────┴──────────────────────────────────┘

  数値表現:
  rwx = 4+2+1 = 7
  r-x = 4+0+1 = 5
  r-- = 4+0+0 = 4
  → chmod 754 file.txt

  パーミッションの意味（ファイル vs ディレクトリ）:
  ┌──────┬──────────────────────┬──────────────────────────┐
  │ 権限 │ ファイル             │ ディレクトリ             │
  ├──────┼──────────────────────┼──────────────────────────┤
  │ r(4) │ ファイル内容の読取   │ ディレクトリ一覧の取得   │
  │ w(2) │ ファイル内容の変更   │ ファイルの作成/削除/名変 │
  │ x(1) │ ファイルの実行       │ ディレクトリへのアクセス  │
  └──────┴──────────────────────┴──────────────────────────┘

  重要: ディレクトリの x 権限がないと中のファイルにアクセスできない
  → chmod 644 dir/ は ls はできるが cd はできない
  → chmod 711 dir/ は中のファイル名を知っていればアクセス可能
```

### 2.2 chmod の詳細

```bash
# シンボリックモード
chmod u+x file.txt      # 所有者に実行権限を追加
chmod g-w file.txt      # グループから書込権限を削除
chmod o=r file.txt      # その他の権限を読取のみに設定
chmod a+r file.txt      # 全員に読取権限を追加
chmod u+s file.txt      # SUID を設定
chmod g+s dir/          # SGID を設定
chmod +t dir/           # Sticky bit を設定

# 数値モード
chmod 755 file.txt      # rwxr-xr-x
chmod 644 file.txt      # rw-r--r--
chmod 600 file.txt      # rw-------（機密ファイル）
chmod 700 dir/          # rwx------（プライベートディレクトリ）
chmod 4755 file.txt     # SUID + rwxr-xr-x
chmod 2755 dir/         # SGID + rwxr-xr-x
chmod 1777 dir/         # Sticky + rwxrwxrwx

# 再帰的変更
chmod -R 755 dir/       # ディレクトリ以下すべて（注意が必要）
# より安全な方法:
find dir/ -type d -exec chmod 755 {} \;  # ディレクトリのみ
find dir/ -type f -exec chmod 644 {} \;  # ファイルのみ
```

### 2.3 特殊パーミッション

```
特殊パーミッション:

  SUID (4xxx): 実行時に所有者の権限で動作
  → /usr/bin/passwd は SUID root（一般ユーザーが/etc/shadowを更新）
  → セキュリティリスクが高いため最小限に

  SUID の仕組み:
  ┌──────────────────────────────────────────────────┐
  │ 通常の実行:                                      │
  │   user(uid=1000) → exec(program) → uid=1000     │
  │                                                    │
  │ SUID 付きの実行:                                  │
  │   user(uid=1000) → exec(passwd) → euid=0(root)  │
  │   → プログラムは root 権限で動作                  │
  │   → /etc/shadow への書き込みが可能                │
  │   → プログラム終了後、権限は元に戻る              │
  └──────────────────────────────────────────────────┘

  SGID (2xxx): 実行時にグループの権限で動作
  → ファイルの場合: グループ権限での実行
  → ディレクトリの場合: 作成されるファイルがディレクトリのグループを継承

  SGID ディレクトリの実務活用:
  ┌──────────────────────────────────────────────────┐
  │ # 共有ディレクトリの作成                          │
  │ $ sudo mkdir /shared/project                      │
  │ $ sudo chgrp developers /shared/project           │
  │ $ sudo chmod 2775 /shared/project                 │
  │                                                    │
  │ # alice がファイルを作成すると:                    │
  │ $ touch /shared/project/code.py                   │
  │ $ ls -la /shared/project/code.py                  │
  │ -rw-rw-r-- 1 alice developers ...                 │
  │                    ↑ ディレクトリのグループを継承  │
  │ → チームメンバー全員がファイルを編集可能          │
  └──────────────────────────────────────────────────┘

  Sticky (1xxx): ディレクトリ内の削除を所有者のみに制限
  → /tmp は sticky bit が設定されている
  → 他ユーザーのファイルを削除できない

  Sticky bit の仕組み:
  ┌──────────────────────────────────────────────────┐
  │ /tmp (chmod 1777):                                │
  │   rwxrwxrwt ← 最後の 't' が Sticky bit          │
  │                                                    │
  │ Sticky bit なし:                                   │
  │   ディレクトリへの w 権限 → 任意のファイル削除可  │
  │                                                    │
  │ Sticky bit あり:                                   │
  │   ファイルの削除/名前変更は以下のみ可能:          │
  │   1. ファイルの所有者                              │
  │   2. ディレクトリの所有者                          │
  │   3. root                                          │
  └──────────────────────────────────────────────────┘
```

### 2.4 所有権の管理

```bash
# 所有者の変更
chown alice file.txt            # 所有者を alice に変更
chown alice:developers file.txt # 所有者とグループを変更
chown :developers file.txt      # グループのみ変更
chown -R alice:developers dir/  # 再帰的に変更

# グループの変更
chgrp developers file.txt       # グループを変更

# 新規ファイルのデフォルト権限（umask）
umask                  # 現在の umask を表示（例: 0022）
umask 0077             # 自分以外のアクセスを禁止

# umask の計算:
# ファイルのデフォルト: 666 - umask
# ディレクトリのデフォルト: 777 - umask
#
# umask = 0022 の場合:
# ファイル: 666 - 022 = 644 (rw-r--r--)
# ディレクトリ: 777 - 022 = 755 (rwxr-xr-x)
#
# umask = 0077 の場合:
# ファイル: 666 - 077 = 600 (rw-------)
# ディレクトリ: 777 - 077 = 700 (rwx------)
#
# セキュアなサーバーでは umask 0077 を推奨
```

### 2.5 ACL（Access Control List）

```
ACL（Access Control List）:
  標準パーミッションを超える細かい制御
  → 特定のユーザーやグループに個別の権限を設定
  → 標準の owner/group/other では不十分な場合に使用

  ACL の種類:
  1. Access ACL: ファイル/ディレクトリへのアクセス制御
  2. Default ACL: ディレクトリ内に新規作成されるファイルのデフォルトACL

  ACL が設定されたファイルの見分け方:
  $ ls -la
  -rw-rwxr--+ 1 user group 4096 Jan 1 file.txt
             ↑ '+' マークが ACL の存在を示す
```

```bash
# ACL の確認
getfacl file.txt
# file: file.txt
# owner: user
# group: group
# user::rw-
# user:alice:rw-          # alice に rw 権限
# user:bob:r--            # bob に r 権限
# group::r-x
# group:devteam:rwx       # devteam グループに rwx 権限
# mask::rwx               # 有効な最大権限
# other::r--

# ACL の設定
setfacl -m u:alice:rw file.txt          # alice に読み書き許可
setfacl -m u:bob:r file.txt             # bob に読み取りのみ
setfacl -m g:devteam:rwx file.txt       # devteam に全権限
setfacl -m o::--- file.txt              # other のアクセスを禁止

# ACL の削除
setfacl -x u:alice file.txt             # alice の ACL を削除
setfacl -b file.txt                     # すべての ACL を削除

# デフォルト ACL（ディレクトリに対して）
setfacl -d -m u:alice:rw /shared/       # 新規ファイルに自動適用
setfacl -d -m g:devteam:rwx /shared/

# ACL のバックアップとリストア
getfacl -R /shared/ > acl_backup.txt    # バックアップ
setfacl --restore=acl_backup.txt        # リストア

# mask の理解:
# mask は ACL のユーザー/グループエントリと
# グループ所有者エントリの有効な最大権限を制限
# 例: mask::r-- の場合、ACL で rwx を設定しても
#      実効権限は r-- になる
```

### 2.6 実務でのパーミッション設計パターン

```
Webサーバーのパーミッション設計:
  ┌──────────────────────────────────────────────────┐
  │ /var/www/html/                                    │
  │ 所有者: www-data:www-data                        │
  │ パーミッション:                                  │
  │                                                    │
  │ 静的ファイル:                                     │
  │   chmod 644 *.html *.css *.js                    │
  │   → Web サーバーが読み取り可能                   │
  │                                                    │
  │ 実行スクリプト:                                   │
  │   chmod 755 *.cgi *.sh                           │
  │   → 実行可能だが変更不可                         │
  │                                                    │
  │ アップロードディレクトリ:                         │
  │   chmod 770 uploads/                              │
  │   → グループメンバーのみ書き込み可能             │
  │                                                    │
  │ 設定ファイル:                                     │
  │   chmod 600 .env *.conf                          │
  │   → 所有者のみ読み書き可能                       │
  │                                                    │
  │ ログディレクトリ:                                 │
  │   chmod 750 logs/                                 │
  │   → グループメンバーは読み取り可能               │
  └──────────────────────────────────────────────────┘

SSH の権限設定:
  ┌──────────────────────────────────────────────────┐
  │ ~/.ssh/                 → chmod 700             │
  │ ~/.ssh/authorized_keys  → chmod 600             │
  │ ~/.ssh/id_rsa           → chmod 600（秘密鍵）  │
  │ ~/.ssh/id_rsa.pub       → chmod 644（公開鍵）  │
  │ ~/.ssh/config           → chmod 600             │
  │ ~/.ssh/known_hosts      → chmod 644             │
  │                                                    │
  │ 権限が緩いと SSH が接続を拒否する:               │
  │ "Permissions 0644 for 'id_rsa' are too open."    │
  │ → 秘密鍵は所有者のみ読み取り可能にすること      │
  └──────────────────────────────────────────────────┘

データベースのパーミッション設計:
  ┌──────────────────────────────────────────────────┐
  │ PostgreSQL:                                       │
  │ /var/lib/postgresql/data/  → 所有者: postgres    │
  │ パーミッション: 700                               │
  │ → データベースファイルはpostgresユーザーのみ      │
  │                                                    │
  │ MySQL:                                            │
  │ /var/lib/mysql/            → 所有者: mysql       │
  │ パーミッション: 750                               │
  │ /etc/mysql/my.cnf          → chmod 644           │
  │ → 設定ファイルは読み取り可能だが変更は root のみ │
  └──────────────────────────────────────────────────┘
```

---

## 3. Linux セキュリティモジュール（LSM）

### 3.1 LSM アーキテクチャ

```
Linux Security Modules (LSM) フレームワーク:
  Linuxカーネルにセキュリティフックを提供する仕組み
  → カーネルの各操作ポイントにフックを配置
  → セキュリティモジュールがフックで判定を行う

  LSM の動作フロー:
  ユーザープロセス
       ↓ システムコール
  カーネル（VFS等）
       ↓
  DAC チェック → 失敗 → EACCES
       ↓ 成功
  LSM フック → SELinux/AppArmor/SMACK がチェック
       ↓ 許可
  実際のリソースアクセス

  主要な LSM:
  ┌──────────────┬──────────────────────────────────┐
  │ モジュール   │ 特徴                             │
  ├──────────────┼──────────────────────────────────┤
  │ SELinux      │ ラベルベースMAC。最も強力        │
  │ AppArmor     │ パスベースMAC。設定が容易        │
  │ SMACK        │ シンプルなMAC。組み込み向け      │
  │ TOMOYO       │ パスベース。学習モード有り       │
  │ Yama         │ ptrace制限など補助的セキュリティ │
  │ LoadPin      │ カーネルモジュール署名検証       │
  │ Lockdown     │ カーネル機能の制限               │
  │ BPF          │ BPFプログラムのセキュリティ      │
  │ Landlock     │ ユーザー空間からのサンドボックス │
  └──────────────┴──────────────────────────────────┘

  LSM スタッキング（Linux 5.1+）:
  複数のLSMを同時に使用可能
  → SELinux + Yama + Lockdown の組み合わせ
  → 「マイナーLSM」は常にスタック可能
  → 「メジャーLSM」は1つのみ（SELinux or AppArmor）
    ※ Linux 6.x でメジャーLSMのスタッキングも進展中
```

### 3.2 SELinux

```
SELinux（Security-Enhanced Linux, NSA開発）:
  MAC の実装。全プロセスとファイルにラベルを付与

  SELinux コンテキスト:
  ┌──────────────────────────────────────────────────┐
  │ user:role:type:level                              │
  │                                                    │
  │ 例: system_u:system_r:httpd_t:s0                  │
  │     │        │        │      │                    │
  │     │        │        │      └── MLS レベル       │
  │     │        │        └── タイプ（最も重要）      │
  │     │        └── ロール                           │
  │     └── SELinux ユーザー                          │
  └──────────────────────────────────────────────────┘

  タイプエンフォースメント（TE）:
  httpd_t プロセスは httpd_sys_content_t ファイルのみアクセス可能
  → Webサーバーが乗っ取られても他のファイルにアクセスできない

  TE ルールの書式:
  allow source_type target_type : object_class { permissions };
  allow httpd_t httpd_sys_content_t : file { read open getattr };

  モード:
  - Enforcing: ポリシー違反を拒否+ログ
  - Permissive: ログのみ（デバッグ用）
  - Disabled: 無効

  基本的な操作:
  $ getenforce                # 現在のモード確認
  $ sudo setenforce 0         # Permissive に一時変更
  $ sudo setenforce 1         # Enforcing に変更

  コンテキストの確認と変更:
  $ ls -Z                     # ファイルのコンテキスト
  $ ps -eZ                    # プロセスのコンテキスト
  $ id -Z                     # 現在のユーザーコンテキスト

  $ sudo chcon -t httpd_sys_content_t /var/www/html/index.html
  $ sudo restorecon -Rv /var/www/html/   # デフォルトに復元
```

```bash
# SELinux のトラブルシューティング

# 拒否ログの確認
sudo ausearch -m AVC -ts recent
# type=AVC msg=audit(1234567890.123:456):
#   avc: denied { read } for pid=1234 comm="httpd"
#   name="config.php" dev="sda1" ino=789
#   scontext=system_u:system_r:httpd_t:s0
#   tcontext=unconfined_u:object_r:user_home_t:s0
#   tclass=file permissive=0

# audit2allow でポリシーモジュールを生成
sudo ausearch -m AVC -ts recent | audit2allow -M mypolicy
sudo semodule -i mypolicy.pp

# sealert によるわかりやすい分析
sudo sealert -a /var/log/audit/audit.log

# Boolean の管理（ポリシーの微調整）
getsebool -a                                  # 全Boolean一覧
getsebool httpd_can_network_connect            # 個別確認
sudo setsebool -P httpd_can_network_connect on # 永続的に有効化

# よく使う Boolean:
# httpd_can_network_connect      → HTTPDの外部接続
# httpd_can_sendmail             → HTTPDのメール送信
# httpd_enable_homedirs          → HTTPDのホームディレクトリアクセス
# allow_ftpd_full_access         → FTPの全アクセス
# samba_enable_home_dirs         → Sambaのホームディレクトリ

# ファイルコンテキストの永続設定
sudo semanage fcontext -a -t httpd_sys_content_t "/web(/.*)?"
sudo restorecon -Rv /web/

# ポートの管理
sudo semanage port -l | grep http              # 許可済みポート一覧
sudo semanage port -a -t http_port_t -p tcp 8080  # ポート追加

# SELinux ユーザーマッピング
sudo semanage login -l                         # マッピング一覧
sudo semanage login -a -s staff_u alice        # ユーザーマッピング
```

### 3.3 AppArmor

```
AppArmor（Ubuntu デフォルト）:
  パスベースのMAC。SELinuxより設定が簡単
  → プロファイルでプロセスのアクセスを制限
  → /etc/apparmor.d/ にプロファイル

  AppArmor vs SELinux:
  ┌─────────────┬──────────────────┬──────────────────┐
  │ 項目        │ SELinux          │ AppArmor         │
  ├─────────────┼──────────────────┼──────────────────┤
  │ アプローチ  │ ラベルベース     │ パスベース        │
  │ 学習コスト  │ 高い             │ 低い              │
  │ 設定の柔軟性│ 非常に高い       │ 中程度            │
  │ デフォルト  │ RHEL/CentOS      │ Ubuntu/SUSE      │
  │ ファイル移動│ ラベルが追従     │ パスが変わると    │
  │             │                  │ ポリシーも変わる  │
  │ inode依存   │ はい             │ いいえ            │
  │ ネットワーク│ 詳細制御可能     │ 基本的な制御      │
  └─────────────┴──────────────────┴──────────────────┘

  AppArmor のモード:
  - Enforce: 違反をブロック+ログ
  - Complain: ログのみ（学習用）
  - Unconfined: 制限なし
```

```bash
# AppArmor の基本操作
sudo aa-status                   # 現在の状態確認
sudo aa-enforce /etc/apparmor.d/usr.sbin.nginx   # Enforceモード
sudo aa-complain /etc/apparmor.d/usr.sbin.nginx  # Complainモード
sudo aa-disable /etc/apparmor.d/usr.sbin.nginx   # 無効化

# プロファイルの新規作成
sudo aa-genprof /usr/bin/myapp   # 対話的にプロファイル生成
# → アプリを操作してログを収集 → ルールを自動生成

# ログの分析（Complainモードで収集した違反）
sudo aa-logprof                  # ログからルールを提案
```

```
AppArmor プロファイルの例（/etc/apparmor.d/usr.sbin.nginx）:

  #include <tunables/global>

  /usr/sbin/nginx {
    #include <abstractions/base>
    #include <abstractions/nameservice>

    # 実行権限
    /usr/sbin/nginx mr,

    # 設定ファイル
    /etc/nginx/** r,
    /etc/ssl/certs/** r,
    /etc/ssl/private/** r,

    # Webコンテンツ
    /var/www/** r,

    # ログ
    /var/log/nginx/** w,

    # PIDファイル
    /run/nginx.pid rw,

    # ネットワーク
    network inet stream,
    network inet6 stream,

    # 子プロセス
    /usr/sbin/nginx ix,

    # 一時ファイル
    /var/lib/nginx/tmp/** rw,

    # deny ルール（明示的な拒否）
    deny /etc/shadow r,
    deny /root/** rwx,
  }

  プロファイルの権限記号:
  ┌──────┬──────────────────────────┐
  │ 記号 │ 意味                     │
  ├──────┼──────────────────────────┤
  │ r    │ 読み取り                 │
  │ w    │ 書き込み                 │
  │ a    │ 追記                     │
  │ x    │ 実行                     │
  │ m    │ メモリマップ             │
  │ k    │ ファイルロック           │
  │ l    │ ハードリンク作成         │
  │ ix   │ Inherit execute          │
  │ px   │ Profile execute          │
  │ cx   │ Child profile execute    │
  │ ux   │ Unconfined execute       │
  └──────┴──────────────────────────┘
```

### 3.4 seccomp と Landlock

```
seccomp（Secure Computing Mode）:
  プロセスが使用できるシステムコールを制限
  → Dockerコンテナのデフォルトセキュリティ
  → Chromeのサンドボックスでも使用

  seccomp のモード:
  1. Strict Mode: read, write, _exit, sigreturn のみ許可
  2. Filter Mode (seccomp-bpf): BPFフィルタで細かく制御

  Docker のデフォルト seccomp プロファイル:
  ┌──────────────────────────────────────────────────┐
  │ 約300以上のシステムコールのうち約40を制限:        │
  │                                                    │
  │ ブロックされるシステムコール例:                   │
  │ - clone (CLONE_NEWUSER): ユーザーNamespace作成    │
  │ - mount: ファイルシステムマウント                 │
  │ - reboot: システム再起動                          │
  │ - kexec_load: カーネル入れ替え                    │
  │ - bpf: BPFプログラムのロード                      │
  │ - unshare: 新しいNamespace作成                    │
  │ - ptrace: 他プロセスのデバッグ                    │
  │ - swapon/swapoff: スワップ管理                    │
  │ - init_module: カーネルモジュールロード            │
  │                                                    │
  │ カスタムプロファイルの適用:                        │
  │ docker run --security-opt seccomp=profile.json    │
  └──────────────────────────────────────────────────┘

Landlock（Linux 5.13+）:
  非特権プロセスからのサンドボックス化
  → root権限不要でアクセス制限を設定可能
  → アプリケーション自身が自分を制限

  Landlock の特徴:
  - ユーザー空間から LSM を利用可能
  - プロセスの子孫にも制限が継承される
  - ファイルシステムアクセスの制限に特化（v1）
  - ネットワーク制限もサポート（v4, Linux 6.7+）
```

```c
/* Landlock の使用例（C言語） */
#include <linux/landlock.h>
#include <sys/prctl.h>
#include <sys/syscall.h>

/* Landlock ルールセットの作成 */
struct landlock_ruleset_attr ruleset_attr = {
    .handled_access_fs =
        LANDLOCK_ACCESS_FS_READ_FILE |
        LANDLOCK_ACCESS_FS_WRITE_FILE |
        LANDLOCK_ACCESS_FS_EXECUTE,
};

int ruleset_fd = syscall(SYS_landlock_create_ruleset,
    &ruleset_attr, sizeof(ruleset_attr), 0);

/* /tmp への読み書きを許可するルール */
struct landlock_path_beneath_attr path_beneath = {
    .allowed_access =
        LANDLOCK_ACCESS_FS_READ_FILE |
        LANDLOCK_ACCESS_FS_WRITE_FILE,
    .parent_fd = open("/tmp", O_PATH | O_CLOEXEC),
};

syscall(SYS_landlock_add_rule, ruleset_fd,
    LANDLOCK_RULE_PATH_BENEATH, &path_beneath, 0);

/* ルールセットを適用（以降、制限が有効） */
prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
syscall(SYS_landlock_restrict_self, ruleset_fd, 0);

/* これ以降、/tmp 以外のファイルアクセスは拒否される */
```

---

## 4. 最小権限の原則

### 4.1 基本概念

```
最小権限の原則（Principle of Least Privilege）:
  プロセスやユーザーに必要最小限の権限のみを付与

  理論的背景:
  Saltzer & Schroeder (1975) の "The Protection of Information
  in Computer Systems" で提唱された8つの設計原則の1つ:

  1. Economy of Mechanism: メカニズムは単純に
  2. Fail-safe Defaults: デフォルトはアクセス拒否
  3. Complete Mediation: すべてのアクセスをチェック
  4. Open Design: 設計の秘匿に依存しない
  5. Separation of Privilege: 権限の分離
  6. Least Privilege: 最小権限           ← これ
  7. Least Common Mechanism: 共有メカニズムの最小化
  8. Psychological Acceptability: 使いやすさ

  実践例:
  ┌──────────────────────────────────────────┐
  │ BAD: root で Web サーバーを実行         │
  │ GOOD: 専用ユーザー（www-data）で実行    │
  │                                          │
  │ BAD: chmod 777 で全開放                 │
  │ GOOD: 必要な権限のみ設定                │
  │                                          │
  │ BAD: アプリにroot権限を付与             │
  │ GOOD: capabilities で必要な権限のみ付与 │
  │                                          │
  │ BAD: 全員に管理者権限を付与             │
  │ GOOD: ロールベースで必要な権限のみ      │
  │                                          │
  │ BAD: sudo ALL=(ALL) NOPASSWD: ALL       │
  │ GOOD: sudo で特定コマンドのみ許可       │
  └──────────────────────────────────────────┘
```

### 4.2 Linux Capabilities

```
Linux Capabilities:
  root の権限を細分化して必要なものだけ付与
  → 「全か無か」の root 権限を約40の細かい権限に分割

  主要な Capabilities:
  ┌──────────────────────────┬────────────────────────────────┐
  │ Capability               │ 説明                           │
  ├──────────────────────────┼────────────────────────────────┤
  │ CAP_NET_BIND_SERVICE     │ 1024未満のポートにバインド     │
  │ CAP_NET_RAW              │ RAWソケットの使用              │
  │ CAP_NET_ADMIN            │ ネットワーク設定の変更         │
  │ CAP_SYS_PTRACE           │ 他プロセスのデバッグ          │
  │ CAP_SYS_ADMIN            │ 多数のシステム管理操作        │
  │ CAP_DAC_OVERRIDE         │ ファイルパーミッション無視    │
  │ CAP_DAC_READ_SEARCH      │ 読み取りとディレクトリ検索    │
  │ CAP_CHOWN                │ ファイル所有者の変更          │
  │ CAP_FOWNER               │ 所有者チェックのバイパス      │
  │ CAP_KILL                 │ 他ユーザーのプロセスにシグナル│
  │ CAP_SETUID               │ UID の変更                    │
  │ CAP_SETGID               │ GID の変更                    │
  │ CAP_SYS_CHROOT           │ chroot の使用                 │
  │ CAP_SYS_TIME             │ システム時刻の変更            │
  │ CAP_AUDIT_WRITE          │ 監査ログへの書き込み          │
  │ CAP_SYS_RESOURCE         │ リソース制限の変更            │
  │ CAP_IPC_LOCK             │ メモリのロック（mlock）       │
  │ CAP_SYS_RAWIO            │ rawI/Oポートの操作            │
  └──────────────────────────┴────────────────────────────────┘

  Capability セット:
  ┌──────────────────────────────────────────────────────┐
  │ Permitted:   プロセスが持てる最大の Capabilities    │
  │ Effective:   現在有効な Capabilities                │
  │ Inheritable: execve 時に継承される Capabilities     │
  │ Bounding:    上限セット（これ以上は取得不可）       │
  │ Ambient:     非特権プログラムに渡される Capabilities│
  └──────────────────────────────────────────────────────┘
```

```bash
# Capabilities の確認と設定

# ファイルの Capabilities 確認
getcap /usr/bin/ping
# /usr/bin/ping cap_net_raw=ep

# Capabilities の設定
sudo setcap cap_net_bind_service=+ep ./server
# → rootなしで80番ポートにバインド可能

# 複数の Capabilities を設定
sudo setcap 'cap_net_bind_service,cap_net_raw=+ep' ./server

# Capabilities の削除
sudo setcap -r ./server

# プロセスの Capabilities 確認
cat /proc/self/status | grep Cap
# CapInh: 0000000000000000  (Inheritable)
# CapPrm: 0000000000000000  (Permitted)
# CapEff: 0000000000000000  (Effective)
# CapBnd: 000001ffffffffff  (Bounding)
# CapAmb: 0000000000000000  (Ambient)

# 16進数をデコード
capsh --decode=000001ffffffffff

# 特定の Capabilities でプロセスを起動
sudo capsh --caps="cap_net_bind_service+eip cap_setpcap,cap_setuid,cap_setgid+ep" \
  --keep=1 --user=www-data --addamb=cap_net_bind_service -- -c ./server

# Docker での Capabilities 管理
docker run --cap-drop ALL --cap-add NET_BIND_SERVICE nginx
# → 全 Capabilities を落としてから必要なものだけ追加
```

### 4.3 sudo の詳細設定

```
sudo の設定（/etc/sudoers）:
  visudo コマンドで編集（構文チェック付き）

  基本書式:
  ユーザー ホスト=(実行ユーザー) コマンド

  例:
  # alice は全ホストで全コマンドを実行可能
  alice ALL=(ALL) ALL

  # alice は パスワードなしで nginx の再起動のみ可能
  alice ALL=(root) NOPASSWD: /usr/bin/systemctl restart nginx

  # developers グループは特定コマンドのみ
  %developers ALL=(root) /usr/bin/docker, /usr/bin/docker-compose

  # bob は特定コマンドを特定ユーザーとして実行
  bob ALL=(www-data) /usr/bin/php, /usr/bin/composer

  セキュアな sudoers 設計:
  ┌──────────────────────────────────────────────────┐
  │ 1. NOPASSWD は最小限に:                          │
  │    → 自動化スクリプト用のみ                      │
  │    → 対話的な使用では常にパスワード要求          │
  │                                                    │
  │ 2. コマンドは絶対パスで指定:                      │
  │    → /usr/bin/systemctl（○）                     │
  │    → systemctl（×: PATH操作で偽物を実行される） │
  │                                                    │
  │ 3. ワイルドカードは避ける:                        │
  │    → ALL=(ALL) /usr/bin/* は危険                  │
  │    → 必要なコマンドを個別に列挙                  │
  │                                                    │
  │ 4. エイリアスで管理:                              │
  │    Cmnd_Alias WEBADMIN = /usr/bin/systemctl       │
  │      restart nginx, /usr/bin/certbot              │
  │    %webteam ALL=(root) WEBADMIN                   │
  │                                                    │
  │ 5. sudo ログの監査:                               │
  │    Defaults logfile="/var/log/sudo.log"            │
  │    Defaults log_input, log_output                  │
  │    → 全セッションを記録                           │
  └──────────────────────────────────────────────────┘

  sudo の危険なパターン:
  ┌──────────────────────────────────────────────────┐
  │ 危険: alice ALL=(ALL) /usr/bin/vi                │
  │ → vi から :!/bin/bash で root シェル取得可能    │
  │                                                    │
  │ 危険: alice ALL=(ALL) /usr/bin/less              │
  │ → less から !bash で root シェル取得可能        │
  │                                                    │
  │ 危険: alice ALL=(ALL) /usr/bin/find              │
  │ → find -exec /bin/bash で root シェル取得可能   │
  │                                                    │
  │ 対策: sudoedit を使う、または制限付きコマンドを  │
  │ 使用する（rnano等のrestricted editor）           │
  └──────────────────────────────────────────────────┘
```

### 4.4 PAM（Pluggable Authentication Modules）

```
PAM（Pluggable Authentication Modules）:
  認証の仕組みをモジュール化して柔軟に組み合わせる

  PAM の設定ファイル:
  /etc/pam.d/ ディレクトリ内にサービスごとの設定
  → /etc/pam.d/sshd, /etc/pam.d/login, /etc/pam.d/sudo

  設定の書式:
  タイプ  制御  モジュール  [オプション]

  タイプ:
  ┌──────────┬──────────────────────────────────────┐
  │ auth     │ ユーザー認証（パスワード、MFA等）   │
  │ account  │ アカウント検証（有効期限、時間制限）│
  │ password │ パスワード変更の処理                │
  │ session  │ セッション管理（ログ、環境設定）    │
  └──────────┴──────────────────────────────────────┘

  制御:
  ┌──────────┬──────────────────────────────────────┐
  │ required │ 失敗しても次のモジュールを実行      │
  │ requisite│ 失敗したら即座に拒否                │
  │ sufficient│ 成功したら以降のモジュールをスキップ│
  │ optional │ 他の結果に影響しない                │
  │ include  │ 他の設定ファイルをインクルード      │
  └──────────┴──────────────────────────────────────┘

  実務的な PAM 設定例:

  /etc/pam.d/sshd（SSH 認証の強化）:
  ┌──────────────────────────────────────────────────┐
  │ # TOTP（Google Authenticator）の追加              │
  │ auth required pam_google_authenticator.so         │
  │                                                    │
  │ # パスワードの品質要件                            │
  │ password requisite pam_pwquality.so retry=3       │
  │   minlen=12 dcredit=-1 ucredit=-1 lcredit=-1     │
  │   ocredit=-1                                      │
  │                                                    │
  │ # ログイン失敗のロックアウト                      │
  │ auth required pam_faillock.so                     │
  │   preauth deny=5 unlock_time=900                  │
  │                                                    │
  │ # アクセス時間の制限                              │
  │ account required pam_time.so                      │
  │   → /etc/security/time.conf で時間帯を設定      │
  │                                                    │
  │ # ログインセッションのリソース制限                │
  │ session required pam_limits.so                    │
  │   → /etc/security/limits.conf でリソース制限    │
  └──────────────────────────────────────────────────┘
```

---

## 5. Windows のアクセス制御

### 5.1 Windows セキュリティモデル

```
Windows のアクセス制御体系:

  セキュリティプリンシパル:
  ┌──────────────────────────────────────────────────┐
  │ - ユーザーアカウント                              │
  │ - グループ                                        │
  │ - コンピューターアカウント                        │
  │ - サービスアカウント                              │
  │                                                    │
  │ 各プリンシパルは SID（Security Identifier）で    │
  │ 一意に識別される                                  │
  │ 例: S-1-5-21-3623811015-3361044348-30300820-1013 │
  │     ↑ ↑ ↑   ↑ ドメイン固有識別子      ↑ RID   │
  └──────────────────────────────────────────────────┘

  アクセストークン:
  ログイン時に作成され、プロセスに付与される
  ┌──────────────────────────────────────────────────┐
  │ アクセストークンの内容:                           │
  │ - ユーザー SID                                    │
  │ - グループ SID のリスト                           │
  │ - 特権（Privileges）のリスト                     │
  │ - 整合性レベル（Integrity Level）                │
  │ - セッション ID                                   │
  └──────────────────────────────────────────────────┘

  セキュリティ記述子（Security Descriptor）:
  各オブジェクト（ファイル、レジストリ等）に付与
  ┌──────────────────────────────────────────────────┐
  │ - Owner SID: 所有者                               │
  │ - Group SID: プライマリグループ                   │
  │ - DACL: 任意アクセス制御リスト                    │
  │   → ACE のリスト（Allow/Deny + 権限 + SID）     │
  │ - SACL: システムアクセス制御リスト                │
  │   → 監査設定（成功/失敗の記録）                  │
  └──────────────────────────────────────────────────┘

  Windows の整合性レベル（MIC: Mandatory Integrity Control）:
  ┌──────────────────────────────────────────────────┐
  │ System: サービス、カーネルオブジェクト            │
  │ High: 管理者プロセス（UAC昇格後）                │
  │ Medium: 通常のユーザープロセス                    │
  │ Low: ブラウザのサンドボックス（Internet Explorer）│
  │ Untrusted: 最低レベル                             │
  │                                                    │
  │ → 低い整合性レベルのプロセスは高いレベルの       │
  │   オブジェクトに書き込めない（No Write Up）      │
  └──────────────────────────────────────────────────┘
```

```powershell
# Windows のアクセス制御操作（PowerShell）

# ファイルの ACL 確認
Get-Acl C:\Data\report.xlsx | Format-List

# ACL の詳細表示
(Get-Acl C:\Data\report.xlsx).Access | Format-Table -AutoSize

# ACL の設定
$acl = Get-Acl C:\Data\report.xlsx
$rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
    "DOMAIN\alice", "Read", "Allow")
$acl.SetAccessRule($rule)
Set-Acl C:\Data\report.xlsx $acl

# 継承の無効化
$acl.SetAccessRuleProtection($true, $false)  # 継承を切断し、既存ACEを削除
Set-Acl C:\Data\report.xlsx $acl

# icacls コマンド（コマンドライン）
icacls C:\Data\report.xlsx
icacls C:\Data\report.xlsx /grant "alice:(R)"
icacls C:\Data\report.xlsx /deny "guest:(W)"
icacls C:\Data\report.xlsx /remove "bob"
icacls C:\Data /grant "developers:(OI)(CI)F" /T  # 再帰的に適用
```

---

## 6. クラウドのアクセス制御

### 6.1 AWS IAM

```
AWS IAM（Identity and Access Management）:
  AWS リソースへのアクセスを制御する中核サービス

  IAM の主要コンポーネント:
  ┌──────────────────────────────────────────────────┐
  │ User: 人間のユーザー（長期認証情報）             │
  │ Group: ユーザーの集合                             │
  │ Role: 一時的な認証情報（推奨）                   │
  │ Policy: アクセス許可/拒否のルール                │
  │ Identity Provider: 外部認証との連携              │
  └──────────────────────────────────────────────────┘

  IAM ポリシーの構造:
```

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowS3ReadOnly",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::my-bucket",
                "arn:aws:s3:::my-bucket/*"
            ],
            "Condition": {
                "IpAddress": {
                    "aws:SourceIp": "203.0.113.0/24"
                },
                "StringEquals": {
                    "aws:RequestedRegion": "ap-northeast-1"
                }
            }
        },
        {
            "Sid": "DenyDeleteBucket",
            "Effect": "Deny",
            "Action": "s3:DeleteBucket",
            "Resource": "*"
        }
    ]
}
```

```
  IAM ポリシー評価ロジック:
  ┌──────────────────────────────────────────────────┐
  │ 1. 明示的 Deny がある → 拒否                    │
  │ 2. SCP（組織ポリシー）で許可されていない → 拒否 │
  │ 3. リソースポリシーで許可 → 許可                │
  │ 4. IAMポリシーで許可 → 許可                     │
  │ 5. Permission Boundary で許可されていない → 拒否│
  │ 6. セッションポリシーで許可されていない → 拒否  │
  │ 7. いずれの許可もない → 暗黙的拒否              │
  │                                                    │
  │ 原則: デフォルト拒否、明示的許可が必要           │
  │ Deny は Always Win（Denyが最優先）                │
  └──────────────────────────────────────────────────┘

  IAM のベストプラクティス:
  ┌──────────────────────────────────────────────────┐
  │ 1. root アカウントを日常使用しない               │
  │ 2. MFA を必須にする                               │
  │ 3. IAM Role を使用（長期認証情報を避ける）       │
  │ 4. 最小権限ポリシーを設計                         │
  │ 5. IAM Access Analyzer で未使用権限を検出        │
  │ 6. SCP で組織全体のガードレールを設定            │
  │ 7. タグベースのアクセス制御（ABAC）を活用       │
  │ 8. 定期的な認証情報のローテーション               │
  │ 9. CloudTrail で全 API コールを監査              │
  │ 10. 条件キーで細かくアクセスを制限               │
  └──────────────────────────────────────────────────┘
```

### 6.2 Kubernetes RBAC

```yaml
# Kubernetes RBAC の設定例

# Role: Namespace内の権限を定義
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: production
  name: pod-reader
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "watch", "list"]
- apiGroups: [""]
  resources: ["pods/log"]
  verbs: ["get"]

---
# RoleBinding: ユーザーにRoleを紐付け
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: read-pods
  namespace: production
subjects:
- kind: User
  name: alice
  apiGroup: rbac.authorization.k8s.io
- kind: Group
  name: developers
  apiGroup: rbac.authorization.k8s.io
- kind: ServiceAccount
  name: monitoring-sa
  namespace: monitoring
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io

---
# ClusterRole: クラスタ全体の権限
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: secret-reader
rules:
- apiGroups: [""]
  resources: ["secrets"]
  verbs: ["get", "list"]
  resourceNames: ["app-config", "tls-cert"]  # 特定リソースのみ
```

```
Kubernetes RBAC のベストプラクティス:
┌──────────────────────────────────────────────────┐
│ 1. ClusterRole より Role を優先                  │
│    → Namespace スコープで権限を最小化            │
│                                                    │
│ 2. ワイルドカード（*）を避ける                   │
│    → verbs: ["*"] は危険                          │
│                                                    │
│ 3. ServiceAccount は Pod ごとに分離              │
│    → デフォルトの ServiceAccount を使わない      │
│                                                    │
│ 4. RBAC の監査                                    │
│    kubectl auth can-i --list --as=alice            │
│    kubectl auth can-i create pods --as=alice       │
│                                                    │
│ 5. Aggregated ClusterRole で管理を簡素化         │
│    → ラベルベースの自動集約                       │
│                                                    │
│ 6. audit ログで権限使用を監視                     │
│    → 未使用権限の特定と削除                       │
└──────────────────────────────────────────────────┘
```

---

## 7. アクセス制御の監査と運用

### 7.1 監査の仕組み

```
Linux auditd（監査デーモン）:
  カーネルレベルでシステムコールを監視・記録

  auditd の設定:
  /etc/audit/auditd.conf    → デーモンの設定
  /etc/audit/rules.d/*.rules → 監査ルール

  監査ルールの例:
  ┌──────────────────────────────────────────────────┐
  │ # ファイルの監視（変更を検知）                   │
  │ -w /etc/passwd -p wa -k user-modify              │
  │ -w /etc/shadow -p wa -k shadow-modify            │
  │ -w /etc/sudoers -p wa -k sudoers-modify          │
  │ -w /etc/ssh/sshd_config -p wa -k sshd-config     │
  │                                                    │
  │ # ファイルの権限変更を監視                        │
  │ -a always,exit -F arch=b64 -S chmod,fchmod        │
  │   -F auid>=1000 -F auid!=4294967295 -k perm-change│
  │                                                    │
  │ # 特権コマンドの実行を監視                        │
  │ -a always,exit -F path=/usr/bin/sudo -F perm=x    │
  │   -k privileged-cmd                                │
  │                                                    │
  │ # ユーザー認証イベント                            │
  │ -w /var/log/faillog -p wa -k login-failures       │
  │ -w /var/log/lastlog -p wa -k last-login           │
  └──────────────────────────────────────────────────┘
```

```bash
# 監査ログの検索
ausearch -k user-modify            # キーで検索
ausearch -m USER_AUTH -ts today    # 今日の認証イベント
ausearch -ui 1000 -ts recent       # 特定ユーザーの最近のイベント

# 監査レポートの生成
aureport --summary                 # サマリー
aureport --auth                    # 認証レポート
aureport --file --failed           # ファイルアクセス失敗
aureport --login --failed          # ログイン失敗

# リアルタイム監視
tail -f /var/log/audit/audit.log | ausearch --interpret
```

### 7.2 セキュリティ監査チェックリスト

```
定期的なアクセス制御監査チェックリスト:

  [ ] SUID/SGID ファイルの棚卸し
      find / -perm -4000 -o -perm -2000 -type f 2>/dev/null
      → 不正な SUID ファイルがないか確認

  [ ] ワールドライタブルファイルの確認
      find / -perm -002 -type f ! -path "/proc/*" 2>/dev/null
      → 機密データが誰でも書き込み可能になっていないか

  [ ] 所有者なしファイルの確認
      find / -nouser -o -nogroup 2>/dev/null
      → 削除されたユーザーのファイルが残っていないか

  [ ] sudo 設定の確認
      sudo -l                      # 自分の sudo 権限
      visudo -c                    # sudoers の構文チェック

  [ ] 不要なユーザーアカウントの確認
      awk -F: '$7 !~ /(nologin|false)$/ {print $1}' /etc/passwd
      → ログイン可能なアカウントの一覧

  [ ] パスワードポリシーの確認
      chage -l username            # パスワード有効期限
      grep -E '^PASS' /etc/login.defs  # パスワードポリシー

  [ ] SSH 設定の確認
      /etc/ssh/sshd_config:
      - PermitRootLogin no
      - PasswordAuthentication no
      - PubkeyAuthentication yes
      - MaxAuthTries 3
      - AllowUsers / AllowGroups の設定

  [ ] ファイアウォールルールの確認
      iptables -L -n -v            # iptables
      nft list ruleset             # nftables
      ufw status verbose           # UFW

  [ ] SELinux/AppArmor の状態確認
      getenforce                   # SELinux
      aa-status                    # AppArmor
      → Enforcing/Enforce モードになっているか

  [ ] ログの確認
      /var/log/auth.log            # 認証ログ
      /var/log/secure              # セキュアログ（RHEL系）
      journalctl -u sshd --since today  # SSH ログ
```

### 7.3 コンプライアンス対応

```
主要なコンプライアンス規格とアクセス制御要件:

  PCI DSS（クレジットカード業界）:
  ┌──────────────────────────────────────────────────┐
  │ Req 7: ビジネスで知る必要のある情報のみアクセス  │
  │ Req 8: ユーザーの一意識別                        │
  │ Req 10: ネットワークリソースとカード会員データへ │
  │         の全アクセスの追跡・監視                  │
  │                                                    │
  │ 具体的対応:                                       │
  │ - ロールベースアクセス制御の実装                  │
  │ - 共有アカウントの禁止                            │
  │ - 定期的なアクセス権レビュー（四半期ごと）       │
  │ - 全アクセスログの1年間保存                      │
  └──────────────────────────────────────────────────┘

  SOX（企業改革法）:
  ┌──────────────────────────────────────────────────┐
  │ 職務分離（Separation of Duties）の実装            │
  │ → 承認者と実行者を分離                           │
  │ → 開発者が本番環境に直接アクセスしない           │
  │                                                    │
  │ 変更管理プロセスの記録                            │
  │ → 全変更にチケット番号を紐付け                   │
  │ → 承認フローの証跡を保存                         │
  └──────────────────────────────────────────────────┘

  GDPR（EU一般データ保護規則）:
  ┌──────────────────────────────────────────────────┐
  │ データ最小化の原則                                │
  │ → 必要最小限のデータのみ収集・保持               │
  │ → アクセス権も最小限に設定                       │
  │                                                    │
  │ アクセスログの保持                                │
  │ → 個人データへのアクセス履歴を記録               │
  │ → データ主体の要求に応じて開示可能に             │
  └──────────────────────────────────────────────────┘

  CIS Benchmarks:
  → OS のセキュリティ設定ガイドライン
  → 自動スキャンツール（OpenSCAP, Lynis）で準拠確認
  → レベル1（基本）とレベル2（強化）の2段階
```

---

## 実践演習

### 演習1: [基礎] -- パーミッション操作

```bash
# ファイルの権限を操作
touch test.txt
chmod 644 test.txt && ls -la test.txt
chmod u+x test.txt && ls -la test.txt
chmod o-r test.txt && ls -la test.txt

# 所有者変更
sudo chown root:root test.txt

# umask の確認と設定
umask
umask 0077
touch secret.txt && ls -la secret.txt   # rw------- になる
umask 0022  # 元に戻す
```

### 演習2: [基礎] -- ACL の設定

```bash
# ACL の設定と確認
touch shared.txt
setfacl -m u:alice:rw shared.txt
setfacl -m u:bob:r shared.txt
setfacl -m g:developers:rw shared.txt
getfacl shared.txt

# デフォルト ACL の設定
mkdir /tmp/shared_dir
setfacl -d -m g:developers:rw /tmp/shared_dir
touch /tmp/shared_dir/newfile.txt
getfacl /tmp/shared_dir/newfile.txt  # デフォルトACLが適用される
```

### 演習3: [応用] -- セキュリティ監査

```bash
# SUID ファイルの検索（セキュリティ監査で重要）
find / -perm -4000 -type f 2>/dev/null

# 書き込み可能なファイルの検索
find /etc -writable -type f 2>/dev/null

# 所有者なしファイルの検索
find / -nouser -o -nogroup 2>/dev/null | head -20

# ログイン可能なアカウント一覧
awk -F: '$7 !~ /(nologin|false)$/ {print $1, $7}' /etc/passwd

# パスワードのない（空の）アカウント検索
sudo awk -F: '($2 == "" || $2 == "!") {print $1}' /etc/shadow

# 最近変更されたファイルの確認（侵害調査）
find /etc -mtime -1 -type f 2>/dev/null
find /usr/bin -mtime -1 -type f 2>/dev/null
```

### 演習4: [応用] -- Capabilities の活用

```bash
# 特権ポートバインドの設定
gcc -o webserver webserver.c
sudo setcap cap_net_bind_service=+ep ./webserver
getcap ./webserver
# → root でなくても 80 番ポートでリッスン可能

# 現在の Capabilities 確認
cat /proc/self/status | grep Cap
capsh --print

# Docker コンテナでの Capabilities 制限
docker run --cap-drop ALL \
  --cap-add NET_BIND_SERVICE \
  --cap-add CHOWN \
  -p 80:80 nginx
```

### 演習5: [実務] -- SELinux トラブルシューティング

```bash
# SELinux が有効な環境で Web サーバーのアクセス問題を解決

# 1. 問題の確認
sudo ausearch -m AVC -ts recent

# 2. 詳細な分析
sudo sealert -a /var/log/audit/audit.log

# 3. ファイルコンテキストの確認
ls -Z /var/www/html/
ls -Z /home/user/public_html/

# 4. 正しいコンテキストの設定
sudo semanage fcontext -a -t httpd_sys_content_t "/web(/.*)?"
sudo restorecon -Rv /web/

# 5. Boolean の確認と設定
getsebool httpd_enable_homedirs
sudo setsebool -P httpd_enable_homedirs on
```

### 演習6: [実務] -- 包括的なアクセス制御設計

```bash
# Webアプリケーションサーバーのアクセス制御設計

# 1. 専用ユーザーとグループの作成
sudo groupadd webapp
sudo useradd -r -g webapp -s /sbin/nologin webapp-user

# 2. ディレクトリ構造の作成と権限設定
sudo mkdir -p /opt/webapp/{app,config,data,logs,tmp}
sudo chown -R webapp-user:webapp /opt/webapp
sudo chmod 750 /opt/webapp
sudo chmod 750 /opt/webapp/app
sudo chmod 700 /opt/webapp/config    # 設定は所有者のみ
sudo chmod 770 /opt/webapp/data      # グループも書き込み可
sudo chmod 750 /opt/webapp/logs      # グループは読み取り可
sudo chmod 700 /opt/webapp/tmp       # 一時ファイルは所有者のみ

# 3. デプロイユーザーの権限設定
sudo usermod -aG webapp deploy-user
setfacl -R -m u:deploy-user:rwx /opt/webapp/app
setfacl -R -d -m u:deploy-user:rwx /opt/webapp/app

# 4. Capabilities の設定（root不要の特権ポートバインド）
sudo setcap cap_net_bind_service=+ep /opt/webapp/app/server

# 5. sudo の設定（デプロイ操作のみ許可）
# visudo で以下を追加:
# deploy-user ALL=(webapp-user) NOPASSWD: /usr/bin/systemctl restart webapp

# 6. 監査ルールの設定
sudo auditctl -w /opt/webapp/config -p wa -k webapp-config
sudo auditctl -w /opt/webapp/app -p wa -k webapp-deploy
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| DAC | 所有者が権限設定。Unix標準。柔軟だがTrojan Horse攻撃に脆弱 |
| MAC | 管理者が強制。SELinux, AppArmor。Bell-LaPadula, Bibaモデル |
| RBAC | ロールベース。AWS IAM, K8s。大規模組織の標準 |
| ABAC | 属性ベース。XACML。最も柔軟なモデル |
| ReBAC | 関係ベース。Google Zanzibar。SNS、ファイル共有 |
| ACL | 標準パーミッション以上の細かい制御。POSIX ACL |
| Capabilities | root権限の細分化。最小権限の実現 |
| LSM | SELinux/AppArmorのフレームワーク。カーネルレベルのMAC |
| PAM | 認証のモジュール化。MFA、ロックアウト、パスワードポリシー |
| 最小権限 | 必要最小限の権限のみ付与。全セキュリティの基本原則 |
| 監査 | auditd, CloudTrail。コンプライアンス対応の要 |

---

## 次に読むべきガイド
→ [[01-sandboxing.md]] -- サンドボックスと隔離

---

## 参考文献
1. Bishop, M. "Computer Security: Art and Science." 2nd Ed, Addison-Wesley, 2018.
2. Saltzer, J. H. & Schroeder, M. D. "The Protection of Information in Computer Systems." Proceedings of the IEEE, 1975.
3. Smalley, S. et al. "Implementing SELinux as a Linux Security Module." NAI Labs Report, 2001.
4. Ferraiolo, D. F. et al. "Proposed NIST Standard for Role-Based Access Control." ACM TISSEC, 2001.
5. Hu, V. C. et al. "Guide to Attribute Based Access Control (ABAC) Definition and Considerations." NIST SP 800-162, 2014.
6. Zanzibar: "Google's Consistent, Global Authorization System." USENIX ATC, 2019.
7. Red Hat. "SELinux User's and Administrator's Guide." Red Hat Enterprise Linux Documentation, 2024.
8. Canonical. "AppArmor Documentation." Ubuntu Security Documentation, 2024.
9. AWS. "IAM Best Practices." AWS Documentation, 2024.
10. NIST. "SP 800-53: Security and Privacy Controls for Information Systems and Organizations." Rev. 5, 2020.
