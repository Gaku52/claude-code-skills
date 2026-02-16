# パーミッションと所有者

> Unixの「全てはファイル」哲学において、パーミッション管理はセキュリティの要。
> 適切なパーミッション設定は、不正アクセス防止・データ保護・システム安定性の基盤となる。

## この章で学ぶこと

- [ ] ファイルパーミッションの読み方・変更方法をマスターする
- [ ] 所有者とグループの管理ができる
- [ ] 特殊パーミッション（SUID/SGID/Sticky Bit）を理解する
- [ ] ACL（Access Control List）で細かなアクセス制御ができる
- [ ] umaskの仕組みと適切な設定方法を理解する
- [ ] セキュリティベストプラクティスを実践できる
- [ ] パーミッション関連のトラブルシューティングができる

---

## 1. パーミッションの基本

### 1.1 パーミッションの読み方

```bash
# パーミッションの表示
$ ls -la
-rwxr-xr-- 1 user group 4096 Jan 1 file.txt
│├─┤├─┤├─┤   │     │
│ │  │  │     │     └── グループ
│ │  │  │     └── 所有者
│ │  │  └── other: 読み取りのみ (r--)
│ │  └── group: 読取+実行 (r-x)
│ └── owner: 全権限 (rwx)
└── タイプ: - file, d dir, l link

# ファイルタイプ一覧
# -  : 通常ファイル
# d  : ディレクトリ
# l  : シンボリックリンク
# c  : キャラクタデバイス（/dev/tty等）
# b  : ブロックデバイス（/dev/sda等）
# p  : 名前付きパイプ（FIFO）
# s  : ソケット

# 各パーミッションの意味
# ファイルの場合:
#   r (read)    : ファイルの内容を読み取れる（cat, less等）
#   w (write)   : ファイルの内容を変更できる（vi, echo >>等）
#   x (execute) : ファイルを実行できる（./script.sh）
#
# ディレクトリの場合:
#   r (read)    : ディレクトリ内のファイル一覧を見れる（ls）
#   w (write)   : ディレクトリ内にファイルを作成・削除できる
#   x (execute) : ディレクトリに入れる（cd）、中のファイルにアクセスできる
```

### 1.2 パーミッションとディレクトリの関係

```bash
# ディレクトリのパーミッション実験

# ケース1: x がないディレクトリ
$ mkdir /tmp/test_dir
$ chmod 644 /tmp/test_dir     # rw-r--r-- (xなし)
$ ls /tmp/test_dir             # エラー: Permission denied
$ cd /tmp/test_dir             # エラー: Permission denied

# ケース2: r がないディレクトリ
$ chmod 311 /tmp/test_dir     # --x--x--x (rなし)
$ cd /tmp/test_dir             # OK: 入れる
$ ls /tmp/test_dir             # エラー: 一覧表示できない
$ cat /tmp/test_dir/known.txt  # OK: ファイル名を知っていればアクセス可能

# ケース3: w + x の組み合わせ
$ chmod 733 /tmp/test_dir     # rwx-wx-wx
$ touch /tmp/test_dir/new.txt  # OK: ファイル作成可能
$ rm /tmp/test_dir/old.txt     # OK: ファイル削除可能
$ ls /tmp/test_dir             # エラー: 一覧表示はできない

# ケース4: w があっても x がなければ書き込み不可
$ chmod 622 /tmp/test_dir     # rw--w--w- (xなし)
$ touch /tmp/test_dir/new.txt  # エラー: x がないのでアクセス不可
```

### 1.3 数値（8進数）表現

```bash
# 各ビットの値
# r = 4 (100 in binary)
# w = 2 (010 in binary)
# x = 1 (001 in binary)

# 計算方法: 各カテゴリのビットを足す
# rwx = 4+2+1 = 7
# rw- = 4+2+0 = 6
# r-x = 4+0+1 = 5
# r-- = 4+0+0 = 4
# -wx = 0+2+1 = 3
# -w- = 0+2+0 = 2
# --x = 0+0+1 = 1
# --- = 0+0+0 = 0

# よく使うパーミッション
chmod 777 file    # rwxrwxrwx — 全員にフルアクセス（非推奨）
chmod 755 file    # rwxr-xr-x — 実行ファイル/ディレクトリの標準
chmod 750 file    # rwxr-x--- — グループまで実行可能
chmod 700 file    # rwx------ — 所有者のみフルアクセス
chmod 644 file    # rw-r--r-- — 通常ファイルの標準
chmod 640 file    # rw-r----- — グループまで読み取り可能
chmod 600 file    # rw------- — 秘密ファイル（SSH鍵等）
chmod 555 file    # r-xr-xr-x — 読取+実行のみ（編集不可）
chmod 444 file    # r--r--r-- — 読み取り専用
chmod 400 file    # r-------- — 所有者のみ読み取り

# ディレクトリによく使うパーミッション
chmod 755 dir/    # rwxr-xr-x — 公開ディレクトリ
chmod 750 dir/    # rwxr-x--- — グループ限定ディレクトリ
chmod 700 dir/    # rwx------ — 個人ディレクトリ
chmod 1777 dir/   # rwxrwxrwt — 共有ディレクトリ（/tmp等）
chmod 2755 dir/   # rwxr-sr-x — SGIDディレクトリ

# 4桁表記（特殊ビット含む）
# 1つ目の数字: 特殊ビット
#   4 = SUID
#   2 = SGID
#   1 = Sticky Bit
chmod 4755 file   # SUID + rwxr-xr-x
chmod 2755 dir/   # SGID + rwxr-xr-x
chmod 1777 dir/   # Sticky + rwxrwxrwx
chmod 6755 file   # SUID + SGID + rwxr-xr-x
```

### 1.4 シンボリック表現

```bash
# 対象の指定
# u = user (所有者)
# g = group (グループ)
# o = others (その他)
# a = all (全員、ugo と同じ)

# 操作の指定
# + = 権限を追加
# - = 権限を削除
# = = 権限を設定（指定した権限のみにする）

# 基本的な使い方
chmod u+x file.txt         # 所有者に実行権追加
chmod g-w file.txt         # グループから書込権削除
chmod o=r file.txt         # その他に読取のみ設定
chmod a+r file.txt         # 全員に読取権追加
chmod ug+rw file.txt       # 所有者とグループに読み書き追加
chmod u=rwx,g=rx,o=r file  # 明示的に全て設定（= 754）

# 複数の操作を同時に
chmod u+x,g-w,o-rwx file   # 所有者に+x、グループから-w、その他から全削除
chmod a=r,u+w file          # 全員にr、所有者に+w（= 644）

# 参照コピー
chmod --reference=ref.txt target.txt  # ref.txt と同じパーミッションに設定

# 再帰的に変更
chmod -R 755 dir/           # ディレクトリ内を全て再帰的に変更

# ディレクトリとファイルを区別して再帰的に設定
# ディレクトリ: 755, ファイル: 644 にしたい場合
find /path -type d -exec chmod 755 {} \;
find /path -type f -exec chmod 644 {} \;

# 大文字 X: ディレクトリまたは既に実行権があるファイルにのみ x を設定
chmod -R u=rwX,g=rX,o=rX dir/
# → ディレクトリには x が付く
# → ファイルには x が付かない（元々 x があるファイルを除く）

# verbose モード: 変更内容を表示
chmod -v 644 *.txt
# mode of 'file1.txt' changed from 0755 (rwxr-xr-x) to 0644 (rw-r--r--)
# mode of 'file2.txt' retained as 0644 (rw-r--r--)

# changes モード: 実際に変更があった場合のみ表示
chmod -c 644 *.txt
# mode of 'file1.txt' changed from 0755 (rwxr-xr-x) to 0644 (rw-r--r--)
```

### 1.5 stat コマンドによる詳細表示

```bash
# stat でパーミッションの詳細を確認
$ stat file.txt
  File: file.txt
  Size: 4096        Blocks: 8          IO Block: 4096   regular file
Device: fd00h/64768d Inode: 1234567     Links: 1
Access: (0644/-rw-r--r--)  Uid: ( 1000/   user)  Gid: ( 1000/  group)
Access: 2026-01-15 10:30:00.000000000 +0900
Modify: 2026-01-14 09:20:00.000000000 +0900
Change: 2026-01-14 09:20:00.000000000 +0900
 Birth: 2026-01-10 08:00:00.000000000 +0900

# macOS の stat（BSD版）
$ stat -f "%Sp %Su %Sg %z %N" file.txt
# -rw-r--r-- user group 4096 file.txt

# Linux の stat（GNU版）でフォーマット指定
$ stat -c "%A %U %G %s %n" file.txt
# -rw-r--r-- user group 4096 file.txt

# 数値表現で表示（Linux）
$ stat -c "%a %n" file.txt
# 644 file.txt

# macOS での数値表現
$ stat -f "%OLp %N" file.txt
# 644 file.txt

# アクセス時刻の確認
$ stat -c "Access: %x\nModify: %y\nChange: %z" file.txt
# Access: 2026-01-15 10:30:00
# Modify: 2026-01-14 09:20:00
# Change: 2026-01-14 09:20:00

# 複数ファイルのパーミッション一括確認
$ stat -c "%a %A %U:%G %n" /etc/passwd /etc/shadow /etc/group
# 644 -rw-r--r-- root:root /etc/passwd
# 640 -rw-r----- root:shadow /etc/shadow
# 644 -rw-r--r-- root:root /etc/group
```

---

## 2. 所有者とグループ

### 2.1 chown — 所有者の変更

```bash
# 基本構文: chown [OPTION] [OWNER][:[GROUP]] FILE

# 所有者変更（要root）
sudo chown user file.txt             # 所有者をuserに変更
sudo chown user:group file.txt       # 所有者+グループを同時に変更
sudo chown :group file.txt           # グループのみ変更（chgrpと同等）
sudo chown user: file.txt            # 所有者変更 + グループを所有者のデフォルトグループに

# 再帰的に変更
sudo chown -R user:group dir/        # ディレクトリ内を全て変更
sudo chown -R --preserve-root user:group /  # / への再帰変更を防止

# verbose/changes モード
sudo chown -v user:group file.txt    # 変更内容を全て表示
sudo chown -c user:group file.txt    # 実際に変更があった場合のみ表示

# 参照コピー
sudo chown --reference=ref.txt target.txt  # ref.txt と同じ所有者に設定

# シンボリックリンクの扱い
sudo chown user symlink              # リンク先のファイルを変更
sudo chown -h user symlink           # シンボリックリンク自体を変更

# from オプション: 現在の所有者が一致する場合のみ変更
sudo chown --from=olduser newuser file.txt
sudo chown --from=:oldgroup user:newgroup file.txt

# 実践例: Webサーバーのドキュメントルート
sudo chown -R www-data:www-data /var/www/html/
sudo chown -R nginx:nginx /usr/share/nginx/html/

# 実践例: ホームディレクトリの修復
sudo chown -R $USER:$(id -gn $USER) ~/

# 実践例: 特定ユーザーのファイルだけ変更
sudo find /shared -user olduser -exec chown newuser {} \;
```

### 2.2 chgrp — グループの変更

```bash
# 基本構文: chgrp [OPTION] GROUP FILE

# グループ変更
sudo chgrp developers project/       # グループをdevelopersに変更
sudo chgrp -R developers project/    # 再帰的にグループ変更

# verbose モード
sudo chgrp -vc developers *.py       # 変更があったファイルのみ表示

# 参照コピー
sudo chgrp --reference=ref.txt target.txt

# 自分が所属するグループへの変更はsudo不要
# （自分がそのグループのメンバーの場合）
chgrp mygroup file.txt

# 実践例: 共有プロジェクトディレクトリ
sudo chgrp -R devteam /opt/project/
sudo chmod -R g+rw /opt/project/
sudo chmod g+s /opt/project/         # 新規ファイルにもグループを継承
```

### 2.3 ユーザーとグループの管理

```bash
# 現在のユーザー情報
whoami                               # 現在のユーザー名
id                                   # UID, GID, 全グループ一覧
id -u                                # UID のみ
id -g                                # プライマリGID のみ
id -G                                # 全GID一覧
id -Gn                               # 全グループ名一覧
id username                          # 特定ユーザーの情報

# グループ一覧
groups                               # 自分のグループ一覧
groups username                      # 特定ユーザーのグループ一覧
getent group                         # システム全体のグループ一覧
getent group groupname               # 特定グループのメンバー確認

# ユーザー追加
sudo useradd -m -s /bin/bash newuser          # ホーム作成 + シェル指定
sudo useradd -m -G sudo,docker newuser        # 追加グループ指定
sudo adduser newuser                          # 対話形式（Debian系）

# グループの作成・管理
sudo groupadd developers                      # グループ作成
sudo groupadd -g 1500 custom                  # GID指定で作成
sudo groupdel oldgroup                         # グループ削除

# ユーザーをグループに追加
sudo usermod -aG docker $USER                  # dockerグループに追加
sudo usermod -aG sudo,adm,www-data user       # 複数グループに追加
# 注意: -a を忘れると既存グループから外れる！
# sudo usermod -G docker user  ← 危険！docker以外から全て外れる

# グループの変更を反映
newgrp docker                                  # 新しいグループでシェル開始
# または一度ログアウトして再ログイン

# ユーザーのプライマリグループ変更
sudo usermod -g newgroup user

# グループからユーザーを削除
sudo gpasswd -d user groupname

# /etc/passwd の確認
getent passwd username
# username:x:1000:1000:Full Name:/home/username:/bin/bash
# ユーザー名:パスワード(x=shadow):UID:GID:コメント:ホーム:シェル

# /etc/group の確認
getent group groupname
# groupname:x:1000:user1,user2
# グループ名:パスワード:GID:メンバー一覧

# /etc/shadow（パスワード情報、要root）
sudo getent shadow username
```

### 2.4 UID と GID の理解

```bash
# UID/GIDの範囲（一般的なLinuxディストリビューション）
# 0       : root
# 1-999   : システムユーザー/グループ（デーモン等）
# 1000+   : 一般ユーザー/グループ
# 65534   : nobody/nogroup（権限なしユーザー）

# 重要なシステムユーザー
id root          # uid=0(root) gid=0(root)
id nobody        # uid=65534(nobody) gid=65534(nogroup)
id www-data      # uid=33(www-data) gid=33(www-data) (Debian系)

# Docker でのUID/GIDマッピング
# コンテナ内のUID = ホストのUID
# コンテナ内でroot(0)で実行 → ホストでもroot権限
# セキュリティのため非rootユーザーでの実行を推奨
docker run --user 1000:1000 myimage

# ファイルのUID/GIDを数値で確認
ls -ln file.txt
# -rw-r--r-- 1 1000 1000 4096 Jan 1 file.txt

# 存在しないUID/GIDのファイル
# ユーザー削除後やNFSマウント時に発生
$ ls -la orphaned.txt
-rw-r--r-- 1 5001 5001 100 Jan 1 orphaned.txt
# 数値表示 = そのUID/GIDに対応するユーザー/グループが存在しない

# 孤児ファイルの検索
find / -nouser -o -nogroup 2>/dev/null
```

---

## 3. umask — デフォルトパーミッション

### 3.1 umaskの仕組み

```bash
# umask は新規ファイル/ディレクトリのパーミッションを制御する「マスク」
# 「この権限は付与しない」というビットを指定

# 基本計算:
# ファイルの最大パーミッション    : 666 (実行権なし)
# ディレクトリの最大パーミッション: 777

# umask = 022 の場合:
# ファイル    : 666 - 022 = 644 (rw-r--r--)
# ディレクトリ: 777 - 022 = 755 (rwxr-xr-x)

# umask = 002 の場合:
# ファイル    : 666 - 002 = 664 (rw-rw-r--)
# ディレクトリ: 777 - 002 = 775 (rwxrwxr-x)

# umask = 077 の場合:
# ファイル    : 666 - 077 = 600 (rw-------)
# ディレクトリ: 777 - 077 = 700 (rwx------)

# 現在のumask確認
umask            # 数値表示（例: 0022）
umask -S         # シンボリック表示（例: u=rwx,g=rx,o=rx）

# umask の一時変更（現在のシェルのみ）
umask 077        # セキュアな設定
touch secret.txt # → 600 (rw-------)
mkdir private/   # → 700 (rwx------)

# umask の恒久設定
# ~/.bashrc または ~/.zshrc に追加
echo "umask 022" >> ~/.bashrc
```

### 3.2 umask の正確な計算方法

```bash
# 実は「引き算」ではなく「ビット演算」
# result = max_perm AND (NOT umask)
#
# 例: umask=033 の場合
# ファイル: 666 AND (NOT 033)
# 666 = 110 110 110
# 033 = 000 011 011
# NOT = 111 100 100
# AND = 110 100 100 = 644
#
# 引き算だと 666 - 033 = 633 になるが、実際は 644
# → ビット演算なので「引きすぎ」は起きない

# 検証
$ umask 033
$ touch test_umask.txt
$ stat -c "%a" test_umask.txt
644    # 633 ではなく 644

# よく使われるumask値
# 022 : デフォルト（一般的なLinux）
# 002 : グループ共有向き（Red Hat系のデフォルト）
# 027 : セキュア（otherに何も許可しない）
# 077 : 最もセキュア（所有者のみ）
# 000 : 最も緩い（テスト用）

# プロセスごとのumask
# umask はプロセスの属性であり、子プロセスに継承される
bash -c 'umask; umask 077; umask'
# 0022  ← 親のumask
# 0077  ← 変更後のumask（このサブシェル内のみ）
umask  # 親シェルでは変更されていない
# 0022
```

### 3.3 umask のユースケース別設定

```bash
# 個人作業用（デフォルト）
umask 022
# ファイル: 644, ディレクトリ: 755

# チーム開発用
umask 002
# ファイル: 664, ディレクトリ: 775
# グループメンバーがファイルを編集できる

# セキュアサーバー用
umask 077
# ファイル: 600, ディレクトリ: 700
# 所有者以外は一切アクセス不可

# 条件付きumask設定（~/.bashrc）
# SSH接続時はセキュアに
if [ -n "$SSH_CLIENT" ]; then
    umask 077
else
    umask 022
fi

# ディレクトリ別umask（direnvとの組み合わせ）
# /shared/project/.envrc
# umask 002

# systemd サービスでのumask設定
# /etc/systemd/system/myapp.service
# [Service]
# UMask=0027
```

---

## 4. 特殊パーミッション

### 4.1 SUID（Set User ID）

```bash
# SUID: 実行時にファイル所有者の権限で実行される
# 数値: 4000
# シンボリック: u+s
# 表示: -rwsr-xr-x（所有者のxがsに変わる）

# SUID の設定
chmod u+s executable
chmod 4755 executable

# SUID が大文字Sの場合: 実行権がない + SUID設定
# -rwSr-xr-x → 所有者にxがなくSUIDが設定されている（意味がない設定）

# 代表的なSUID付きコマンド
$ ls -la /usr/bin/passwd
-rwsr-xr-x 1 root root 68208 May 28 2020 /usr/bin/passwd
# → 一般ユーザーが実行してもroot権限で/etc/shadowを更新

$ ls -la /usr/bin/sudo
-rwsr-xr-x 1 root root 166056 Jan 19 2021 /usr/bin/sudo
# → root権限でコマンドを実行

$ ls -la /usr/bin/ping
-rwsr-xr-x 1 root root 64424 Jun 28 2019 /usr/bin/ping
# → rawソケットを使用するためroot権限が必要

# SUID付きファイルの検索（セキュリティ監査）
find / -perm -4000 -type f 2>/dev/null
find / -perm -4000 -type f -exec ls -la {} \; 2>/dev/null

# SUID のセキュリティリスク
# - 不適切なSUID設定は権限昇格の脆弱性につながる
# - カスタムスクリプトにはSUIDを設定しない
# - 定期的にSUID付きファイルを監査する
# - シェルスクリプトのSUIDは多くのOSで無視される

# SUID付きファイルの一覧を保存（ベースライン作成）
find / -perm -4000 -type f 2>/dev/null | sort > /tmp/suid_baseline.txt
# 定期的に比較
find / -perm -4000 -type f 2>/dev/null | sort | diff /tmp/suid_baseline.txt -
```

### 4.2 SGID（Set Group ID）

```bash
# SGID on ファイル: 実行時にファイルグループの権限で実行される
# SGID on ディレクトリ: 新規作成ファイルに親ディレクトリのグループが継承される
# 数値: 2000
# シンボリック: g+s
# 表示: -rwxr-sr-x（グループのxがsに変わる）

# ファイルへのSGID設定
chmod g+s executable
chmod 2755 executable

# ディレクトリへのSGID設定（最も実用的な使い方）
chmod g+s /shared/project/
chmod 2775 /shared/project/

# SGIDディレクトリの動作確認
$ mkdir /tmp/sgid_test
$ sudo chown :developers /tmp/sgid_test
$ chmod 2775 /tmp/sgid_test
$ ls -la /tmp/ | grep sgid_test
drwxrwsr-x 2 user developers 4096 Jan 1 sgid_test
#                ^ s が付いている

$ touch /tmp/sgid_test/newfile.txt
$ ls -la /tmp/sgid_test/newfile.txt
-rw-r--r-- 1 user developers 0 Jan 1 newfile.txt
#                  ^ 親ディレクトリのグループ(developers)が継承された

# SGID なしの場合
$ mkdir /tmp/nosgid_test
$ sudo chown :developers /tmp/nosgid_test
$ chmod 775 /tmp/nosgid_test
$ touch /tmp/nosgid_test/newfile.txt
$ ls -la /tmp/nosgid_test/newfile.txt
-rw-r--r-- 1 user user 0 Jan 1 newfile.txt
#                  ^ ユーザーのプライマリグループになる

# チーム共有ディレクトリの正しい設定
sudo mkdir -p /opt/shared/project
sudo groupadd devteam
sudo chown root:devteam /opt/shared/project
sudo chmod 2775 /opt/shared/project
# → メンバーが作成したファイルは全てdevteamグループになる

# SGID付きファイルの検索
find / -perm -2000 -type f 2>/dev/null
find / -perm -2000 -type d 2>/dev/null  # ディレクトリも
```

### 4.3 Sticky Bit

```bash
# Sticky Bit: ディレクトリ内のファイル削除を所有者とrootのみに制限
# 数値: 1000
# シンボリック: +t
# 表示: drwxrwxrwt（otherのxがtに変わる）

# Sticky Bitの設定
chmod +t /shared/
chmod 1777 /shared/

# /tmp が代表的な Sticky Bit ディレクトリ
$ ls -ld /tmp
drwxrwxrwt 20 root root 4096 Jan 1 /tmp
#                             ^ t が付いている

# Sticky Bitの動作確認
# ユーザーAがファイルを作成
$ touch /tmp/userA_file.txt

# ユーザーBが削除を試みる
$ sudo -u userB rm /tmp/userA_file.txt
rm: cannot remove '/tmp/userA_file.txt': Operation not permitted
# → Sticky Bitにより、所有者以外は削除できない

# ただし所有者とrootは削除可能
$ rm /tmp/userA_file.txt        # OK (所有者)
$ sudo rm /tmp/userA_file.txt   # OK (root)

# Sticky Bit + SGID の組み合わせ（チーム共有）
sudo mkdir /shared/team
sudo chown root:devteam /shared/team
sudo chmod 3775 /shared/team
# → 3 = SGID(2) + Sticky(1)
# → グループは継承されるが、他人のファイルは削除できない

# 大文字T: 実行権がない + Sticky Bit設定
# drwxrwxrwT → otherにxがなくSticky設定（意味がない設定）

# Sticky Bit付きディレクトリの検索
find / -perm -1000 -type d 2>/dev/null
```

### 4.4 特殊パーミッションの表示と確認

```bash
# ls -la での特殊パーミッション表示
# SUID: 所有者のx位置
#   s = SUID + 実行権あり
#   S = SUID + 実行権なし（通常は設定ミス）

# SGID: グループのx位置
#   s = SGID + 実行権あり
#   S = SGID + 実行権なし

# Sticky: otherのx位置
#   t = Sticky + 実行権あり
#   T = Sticky + 実行権なし

# 具体的な表示例
# -rwsr-xr-x : SUID設定、全員実行可能
# -rwxr-sr-x : SGID設定
# drwxrwxrwt : Sticky Bit設定
# -rwSr--r-- : SUID設定だが所有者に実行権なし（問題あり）
# -rwxr-Sr-- : SGID設定だがグループに実行権なし（問題あり）
# drwxrwxrwT : Sticky設定だがotherに実行権なし（問題あり）

# stat での数値確認
stat -c "%a %A %n" /usr/bin/passwd
# 4755 -rwsr-xr-x /usr/bin/passwd

stat -c "%a %A %n" /tmp
# 1777 drwxrwxrwt /tmp

# 全ての特殊パーミッション付きファイルを検索
find / -perm /7000 -type f 2>/dev/null | head -20
# /7000 = SUID(4000) OR SGID(2000) OR Sticky(1000) のいずれか
```

---

## 5. ACL（Access Control List）

### 5.1 ACLの基本概念

```bash
# ACL = 標準のuser/group/other以上の細かなアクセス制御
# 特定のユーザーやグループに個別の権限を付与できる

# ACLが必要なケース:
# - 所有者グループ以外の特定グループにもアクセスを許可したい
# - 特定のユーザーだけに書き込み権限を与えたい
# - 複数のグループに異なるレベルのアクセスを設定したい

# ACLのインストール確認
which getfacl setfacl
# Ubuntu/Debian: sudo apt install acl
# CentOS/RHEL: sudo yum install acl

# ファイルシステムのACLサポート確認
mount | grep acl
# ext4, xfs, btrfs は通常ACLをサポート
# mount -o acl でマウントされていること

# tune2fs でACLサポート確認（ext4）
sudo tune2fs -l /dev/sda1 | grep -i acl
# Default mount options: ... acl ...
```

### 5.2 getfacl — ACLの表示

```bash
# 基本的なACL表示
$ getfacl file.txt
# file: file.txt
# owner: user
# group: group
user::rw-           # 所有者の権限
group::r--          # グループの権限
other::r--          # その他の権限

# ACLが設定されている場合
$ getfacl file.txt
# file: file.txt
# owner: user
# group: group
user::rw-           # 所有者
user:alice:rw-      # aliceに個別のrw権限
user:bob:r--        # bobに個別のr権限
group::r--          # 所有者グループ
group:devteam:rw-   # devteamグループにrw権限
mask::rw-           # 有効な最大権限
other::r--          # その他

# ACLが設定されているファイルの見分け方
$ ls -la
-rw-rw-r--+ 1 user group 4096 Jan 1 file.txt
#          ^ + マークがACL設定済みの印

# ディレクトリのACL（デフォルトACL含む）
$ getfacl dir/
# file: dir/
# owner: user
# group: group
user::rwx
group::r-x
other::r-x
default:user::rwx          # 新規ファイルのデフォルト: 所有者
default:user:alice:rw-     # 新規ファイルのデフォルト: alice
default:group::r-x         # 新規ファイルのデフォルト: グループ
default:mask::rwx          # 新規ファイルのデフォルト: mask
default:other::r-x         # 新規ファイルのデフォルト: その他

# 再帰的にACL表示
getfacl -R dir/

# 数値で表示
getfacl --omit-header -e n file.txt

# ACL付きファイルだけを検索
getfacl -R -s -p dir/ 2>/dev/null
```

### 5.3 setfacl — ACLの設定

```bash
# 基本構文: setfacl [OPTIONS] [操作] FILE

# ユーザーACLの追加
setfacl -m u:alice:rw file.txt       # aliceにrw権限を追加
setfacl -m u:bob:r file.txt          # bobにr権限を追加
setfacl -m u:charlie:rwx script.sh   # charlieにrwx権限を追加

# グループACLの追加
setfacl -m g:devteam:rw file.txt     # devteamグループにrw権限
setfacl -m g:qa:r file.txt           # qaグループにr権限

# その他のACL設定
setfacl -m o::r file.txt             # other の権限を設定

# 複数のACLを同時に設定
setfacl -m u:alice:rw,u:bob:r,g:devteam:rw file.txt

# ACLの削除
setfacl -x u:alice file.txt          # aliceのACLエントリを削除
setfacl -x g:devteam file.txt        # devteamのACLエントリを削除

# 全てのACLを削除
setfacl -b file.txt                  # 全ACL削除（基本パーミッションは残る）

# デフォルトACL（ディレクトリ用）
# 新規作成されるファイル/ディレクトリに自動適用されるACL
setfacl -d -m u:alice:rw dir/        # デフォルトACLを設定
setfacl -d -m g:devteam:rwx dir/     # デフォルトACLを設定

# デフォルトACLの削除
setfacl -k dir/                      # デフォルトACLのみ削除

# 再帰的にACLを設定
setfacl -R -m u:alice:rw dir/        # 既存ファイルにも適用
setfacl -R -d -m u:alice:rw dir/     # 既存ディレクトリにデフォルトACL

# ACLのバックアップと復元
getfacl -R dir/ > acl_backup.txt     # バックアップ
setfacl --restore=acl_backup.txt     # 復元

# mask の設定
setfacl -m m::r file.txt             # maskをrに設定
# mask はACLユーザー/グループの有効な最大権限を制限する
# 例: u:alice:rw でも mask::r なら、実効権限は r のみ

# mask の自動再計算を防ぐ
setfacl -n -m u:alice:rw file.txt    # maskを自動更新しない
```

### 5.4 ACLの実践パターン

```bash
# パターン1: プロジェクトディレクトリの共有設定
sudo mkdir -p /projects/webapp
sudo chown root:devteam /projects/webapp
sudo chmod 2770 /projects/webapp

# 開発チーム: フルアクセス
setfacl -R -m g:devteam:rwx /projects/webapp
setfacl -R -d -m g:devteam:rwx /projects/webapp

# QAチーム: 読み取りのみ
setfacl -R -m g:qa:rx /projects/webapp
setfacl -R -d -m g:qa:rx /projects/webapp

# 外部コンサルタント: 特定ディレクトリのみ
setfacl -m u:consultant:rx /projects/webapp/docs
setfacl -R -m u:consultant:rx /projects/webapp/docs/

# パターン2: ログファイルのアクセス制御
# 特定ユーザーにログ閲覧権限を付与
setfacl -m u:logviewer:r /var/log/app/app.log
setfacl -d -m u:logviewer:r /var/log/app/

# パターン3: Webサーバーとデプロイユーザーの共存
setfacl -R -m u:www-data:rx /var/www/html/
setfacl -R -m u:deploy:rwx /var/www/html/
setfacl -R -d -m u:www-data:rx /var/www/html/
setfacl -R -d -m u:deploy:rwx /var/www/html/

# パターン4: バックアップ用の読み取り専用アクセス
setfacl -R -m u:backup:rx /important/data/
setfacl -R -d -m u:backup:rx /important/data/

# ACLの確認（結果の見方）
$ getfacl /projects/webapp/newfile.txt
# file: projects/webapp/newfile.txt
# owner: developer1
# group: devteam
user::rw-
user:consultant:r-x     #effective:r--   ← mask による制限
group::rwx               #effective:rw-   ← mask による制限
group:devteam:rwx        #effective:rw-   ← mask による制限
group:qa:r-x             #effective:r--   ← mask による制限
mask::rw-                                  ← 有効な最大権限
other::---
```

### 5.5 ACL と標準パーミッションの関係

```bash
# ACL と chmod の相互作用
# chmod はACLのmaskに影響する

# 例: ACLでaliceにrwx権限を設定
setfacl -m u:alice:rwx file.txt
getfacl file.txt
# user:alice:rwx
# mask::rwx

# chmod でグループ権限を変更すると mask も変わる
chmod 644 file.txt
getfacl file.txt
# user:alice:rwx    #effective:r--  ← mask により制限！
# mask::r--         ← chmod がmaskを変更した

# 対処法: chmod 後に mask を再設定
chmod 644 file.txt
setfacl -m m::rwx file.txt  # mask を明示的にrwxに設定

# ACL設定時の優先順位
# 1. 所有者（user::）→ 常に適用
# 2. 名前付きユーザーACL（user:name:） → mask で制限
# 3. 所有者グループ（group::） → mask で制限（ACLが存在する場合）
# 4. 名前付きグループACL（group:name:） → mask で制限
# 5. その他（other::） → 常に適用

# cp と mv でのACL保持
cp --preserve=all src.txt dst.txt    # ACLを保持してコピー
cp -a src/ dst/                       # ACLを保持して再帰コピー
mv src.txt dst.txt                    # 同一FS内ならACL保持

# tar でのACL保持
tar --acls -czf backup.tar.gz dir/   # ACL付きでアーカイブ
tar --acls -xzf backup.tar.gz       # ACL付きで展開

# rsync でのACL保持
rsync -avA src/ dst/                 # -A でACLを同期
```

---

## 6. セキュリティ用途のパーミッション設定

### 6.1 SSH 関連の必須パーミッション

```bash
# SSH は厳格なパーミッションチェックを行う
# 不適切なパーミッションだとSSH接続が拒否される

# ホームディレクトリ
chmod 755 ~                   # または 700
# ホームディレクトリが他人に書き込み可能だとSSH拒否

# .ssh ディレクトリ
chmod 700 ~/.ssh              # 所有者のみ
chown $USER:$(id -gn) ~/.ssh

# 秘密鍵
chmod 600 ~/.ssh/id_rsa       # 所有者のみ読み書き
chmod 600 ~/.ssh/id_ed25519   # Ed25519鍵
# 400 でもOK（読み取り専用）
chmod 400 ~/.ssh/id_rsa       # より安全

# 公開鍵
chmod 644 ~/.ssh/id_rsa.pub   # 誰でも読み取り可能

# authorized_keys
chmod 600 ~/.ssh/authorized_keys
# または 644

# known_hosts
chmod 644 ~/.ssh/known_hosts

# config ファイル
chmod 600 ~/.ssh/config

# SSH サーバー側の設定
chmod 600 /etc/ssh/sshd_config
chmod 600 /etc/ssh/ssh_host_*_key      # ホスト秘密鍵
chmod 644 /etc/ssh/ssh_host_*_key.pub  # ホスト公開鍵

# 一括修正スクリプト
fix_ssh_permissions() {
    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/id_* 2>/dev/null
    chmod 644 ~/.ssh/*.pub 2>/dev/null
    chmod 600 ~/.ssh/authorized_keys 2>/dev/null
    chmod 600 ~/.ssh/config 2>/dev/null
    chmod 644 ~/.ssh/known_hosts 2>/dev/null
    echo "SSH permissions fixed."
}

# SSH パーミッションエラーの確認
ssh -vvv user@host 2>&1 | grep -i permission
# debug1: identity file /home/user/.ssh/id_rsa type 0
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @ WARNING: UNPROTECTED PRIVATE KEY FILE!      @
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Permissions 0644 for '/home/user/.ssh/id_rsa' are too open.
```

### 6.2 Webサーバーのパーミッション設定

```bash
# ===== Apache / Nginx 共通 =====

# ドキュメントルート
sudo chown -R root:www-data /var/www/html/
sudo chmod -R 750 /var/www/html/
# ファイル: 640 (rw-r-----)
# ディレクトリ: 750 (rwxr-x---)

find /var/www/html -type f -exec chmod 640 {} \;
find /var/www/html -type d -exec chmod 750 {} \;

# アップロードディレクトリ
sudo mkdir -p /var/www/html/uploads
sudo chown www-data:www-data /var/www/html/uploads
sudo chmod 770 /var/www/html/uploads

# CGI/スクリプト
sudo chmod 750 /var/www/cgi-bin/*.cgi

# 設定ファイル
sudo chmod 600 /etc/apache2/apache2.conf
sudo chmod 600 /etc/nginx/nginx.conf
sudo chmod 600 /etc/nginx/conf.d/*.conf

# SSL証明書
sudo chmod 600 /etc/ssl/private/server.key
sudo chmod 644 /etc/ssl/certs/server.crt

# ログファイル
sudo chmod 640 /var/log/apache2/*.log
sudo chmod 640 /var/log/nginx/*.log

# ===== PHP-FPM =====
# phpのセッション/テンプファイル
sudo chown www-data:www-data /var/lib/php/sessions
sudo chmod 730 /var/lib/php/sessions

# .htaccess のセキュリティ（Apache）
# .htaccess 自体のパーミッション
chmod 644 /var/www/html/.htaccess

# .env ファイルへのWebアクセスをブロック
# .htaccess に追加:
# <Files ".env">
#     Require all denied
# </Files>

# ===== WordPress の推奨パーミッション =====
# ファイル: 644
# ディレクトリ: 755
# wp-config.php: 440 または 400
# .htaccess: 644
find /var/www/wordpress -type f -exec chmod 644 {} \;
find /var/www/wordpress -type d -exec chmod 755 {} \;
chmod 440 /var/www/wordpress/wp-config.php
```

### 6.3 データベースのパーミッション設定

```bash
# ===== MySQL / MariaDB =====
# データディレクトリ
sudo chown -R mysql:mysql /var/lib/mysql/
sudo chmod 750 /var/lib/mysql/
sudo chmod 660 /var/lib/mysql/ib_logfile*

# 設定ファイル
sudo chmod 644 /etc/mysql/my.cnf
sudo chmod 640 /etc/mysql/conf.d/*.cnf

# ログファイル
sudo chmod 640 /var/log/mysql/error.log
sudo chown mysql:adm /var/log/mysql/error.log

# ===== PostgreSQL =====
# データディレクトリ
sudo chown -R postgres:postgres /var/lib/postgresql/
sudo chmod 700 /var/lib/postgresql/*/main/

# 認証設定
sudo chmod 640 /etc/postgresql/*/main/pg_hba.conf
sudo chown postgres:postgres /etc/postgresql/*/main/pg_hba.conf

# ===== SQLite =====
# データベースファイル
chmod 660 /path/to/database.db
# ディレクトリにも書き込み権限が必要（WALモード用）
chmod 770 /path/to/

# ===== Redis =====
sudo chown redis:redis /var/lib/redis/
sudo chmod 750 /var/lib/redis/
sudo chmod 640 /etc/redis/redis.conf
```

### 6.4 アプリケーションのパーミッション設定

```bash
# ===== 秘密情報ファイル =====
# .env ファイル
chmod 600 .env
chmod 600 .env.production

# API キーファイル
chmod 600 credentials.json
chmod 600 service-account.json
chmod 600 api_key.txt

# 暗号化キー
chmod 600 encryption.key
chmod 600 master.key

# ===== Docker =====
# Docker ソケット
sudo chmod 660 /var/run/docker.sock
sudo chown root:docker /var/run/docker.sock

# Docker Compose ファイル（秘密情報を含む場合）
chmod 600 docker-compose.override.yml

# ===== systemd サービス =====
# ユニットファイル
sudo chmod 644 /etc/systemd/system/myapp.service

# 環境変数ファイル（秘密情報含む）
sudo chmod 600 /etc/myapp/env
sudo chown root:root /etc/myapp/env

# ===== cron =====
# crontab ファイル
chmod 600 /var/spool/cron/crontabs/*
# cronスクリプト
chmod 700 /etc/cron.d/myscript
chmod 755 /etc/cron.daily/backup.sh

# ===== Git フック =====
chmod 755 .git/hooks/pre-commit
chmod 755 .git/hooks/post-merge

# ===== Python 仮想環境 =====
chmod 755 venv/bin/activate
chmod 755 venv/bin/python
```

### 6.5 重要なシステムファイルの標準パーミッション

```bash
# 認証関連
ls -la /etc/passwd         # 644 -rw-r--r-- (全員が読める)
ls -la /etc/shadow         # 640 -rw-r----- (rootとshadowグループのみ)
ls -la /etc/group          # 644 -rw-r--r-- (全員が読める)
ls -la /etc/gshadow        # 640 -rw-r----- (rootとshadowグループのみ)
ls -la /etc/sudoers        # 440 -r--r----- (rootのみ読み取り)

# ネットワーク設定
ls -la /etc/hosts          # 644 -rw-r--r--
ls -la /etc/hostname       # 644 -rw-r--r--
ls -la /etc/resolv.conf    # 644 -rw-r--r--

# サービス設定
ls -la /etc/ssh/sshd_config  # 600 -rw------- (rootのみ)
ls -la /etc/crontab          # 644 -rw-r--r--

# ブート関連
ls -la /boot/vmlinuz-*       # 600 -rw------- (Debian系)
ls -la /boot/grub/grub.cfg   # 400 -r-------- (rootのみ読み取り)

# デバイスファイル
ls -la /dev/null             # 666 crw-rw-rw-
ls -la /dev/zero             # 666 crw-rw-rw-
ls -la /dev/random           # 666 crw-rw-rw-
ls -la /dev/sda              # 660 brw-rw---- (rootとdiskグループ)
ls -la /dev/tty              # 666 crw-rw-rw-

# パーミッション監査: 重要ファイルのチェック
check_critical_permissions() {
    echo "=== Critical File Permissions ==="
    for f in /etc/passwd /etc/shadow /etc/group /etc/sudoers \
             /etc/ssh/sshd_config; do
        if [ -f "$f" ]; then
            stat -c "%a %A %U:%G %n" "$f"
        fi
    done
}
```

---

## 7. パーミッション管理の実践テクニック

### 7.1 find を使ったパーミッション操作

```bash
# 特定のパーミッションを持つファイルを検索
find /path -perm 777                  # 正確に 777 のファイル
find /path -perm -777                 # 少なくとも 777 を含む
find /path -perm /777                 # いずれかのビットが一致

# 世界書き込み可能なファイルを検索（セキュリティ監査）
find / -perm -002 -type f 2>/dev/null
# -002 = other に w がある全てのファイル

# 世界書き込み可能なディレクトリ（Sticky Bitなし）を検索
find / -perm -002 -not -perm -1000 -type d 2>/dev/null

# SUID/SGID 付きファイルの検索
find / -perm -4000 -type f 2>/dev/null  # SUID
find / -perm -2000 -type f 2>/dev/null  # SGID
find / -perm /6000 -type f 2>/dev/null  # SUID または SGID

# 実行権限のあるファイルを検索
find /path -perm /111 -type f          # 誰かが実行可能

# パーミッションが不適切なファイルを修正
# ファイル: 644, ディレクトリ: 755 に統一
find /path -type f -not -perm 644 -exec chmod 644 {} \;
find /path -type d -not -perm 755 -exec chmod 755 {} \;

# 所有者がいないファイルを検索
find / -nouser 2>/dev/null
find / -nogroup 2>/dev/null
find / -nouser -o -nogroup 2>/dev/null

# 特定ユーザーのファイルを検索
find / -user username 2>/dev/null
find / -uid 1000 2>/dev/null
find / -group groupname 2>/dev/null
find / -gid 1000 2>/dev/null

# 最近パーミッションが変更されたファイル
find / -cmin -60 2>/dev/null           # 過去60分以内にctime変更
# ctime = inode変更時刻（パーミッション変更を含む）

# 書き込み可能なSUIDファイル（深刻な脆弱性）
find / -perm -4002 -type f 2>/dev/null
```

### 7.2 パーミッション設定スクリプト

```bash
#!/bin/bash
# fix-web-permissions.sh
# Webプロジェクトのパーミッションを修正するスクリプト

WEB_ROOT="${1:?Usage: $0 <web-root-path>}"
WEB_USER="www-data"
WEB_GROUP="www-data"

# 存在確認
if [ ! -d "$WEB_ROOT" ]; then
    echo "Error: $WEB_ROOT does not exist"
    exit 1
fi

echo "Fixing permissions for: $WEB_ROOT"

# 所有者設定
echo "Setting ownership to ${WEB_USER}:${WEB_GROUP}..."
sudo chown -R "${WEB_USER}:${WEB_GROUP}" "$WEB_ROOT"

# ディレクトリ: 755
echo "Setting directory permissions to 755..."
find "$WEB_ROOT" -type d -exec chmod 755 {} \;

# ファイル: 644
echo "Setting file permissions to 644..."
find "$WEB_ROOT" -type f -exec chmod 644 {} \;

# 実行可能ファイル: 755
echo "Setting executable permissions..."
find "$WEB_ROOT" -name "*.sh" -exec chmod 755 {} \;
find "$WEB_ROOT" -name "*.py" -exec chmod 755 {} \;
find "$WEB_ROOT" -name "*.cgi" -exec chmod 755 {} \;

# 秘密ファイル: 600
echo "Securing sensitive files..."
find "$WEB_ROOT" -name ".env" -exec chmod 600 {} \;
find "$WEB_ROOT" -name "*.key" -exec chmod 600 {} \;
find "$WEB_ROOT" -name "*.pem" -exec chmod 600 {} \;
find "$WEB_ROOT" -name "wp-config.php" -exec chmod 440 {} \;

# アップロードディレクトリ
if [ -d "$WEB_ROOT/uploads" ]; then
    echo "Setting upload directory permissions..."
    chmod 770 "$WEB_ROOT/uploads"
fi

# ログディレクトリ
if [ -d "$WEB_ROOT/logs" ]; then
    echo "Setting log directory permissions..."
    chmod 750 "$WEB_ROOT/logs"
    find "$WEB_ROOT/logs" -type f -exec chmod 640 {} \;
fi

echo "Done! Permissions fixed."
```

```bash
#!/bin/bash
# permission-audit.sh
# セキュリティ監査用パーミッションチェックスクリプト

echo "============================================"
echo " Permission Security Audit Report"
echo " Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Host: $(hostname)"
echo "============================================"
echo ""

# 1. 世界書き込み可能ファイル
echo "=== World-Writable Files ==="
world_writable=$(find / -perm -002 -type f -not -path "/proc/*" -not -path "/sys/*" 2>/dev/null)
if [ -n "$world_writable" ]; then
    echo "$world_writable" | head -20
    echo "Total: $(echo "$world_writable" | wc -l) files"
else
    echo "None found. [OK]"
fi
echo ""

# 2. 世界書き込み可能ディレクトリ（Sticky Bitなし）
echo "=== World-Writable Directories (without Sticky Bit) ==="
ww_dirs=$(find / -perm -002 -not -perm -1000 -type d \
    -not -path "/proc/*" -not -path "/sys/*" 2>/dev/null)
if [ -n "$ww_dirs" ]; then
    echo "$ww_dirs" | head -20
    echo "Total: $(echo "$ww_dirs" | wc -l) directories"
    echo "[WARNING] These directories should have Sticky Bit set"
else
    echo "None found. [OK]"
fi
echo ""

# 3. SUID/SGID ファイル
echo "=== SUID Files ==="
find / -perm -4000 -type f -not -path "/proc/*" -not -path "/sys/*" \
    -exec ls -la {} \; 2>/dev/null | head -20
echo ""

echo "=== SGID Files ==="
find / -perm -2000 -type f -not -path "/proc/*" -not -path "/sys/*" \
    -exec ls -la {} \; 2>/dev/null | head -20
echo ""

# 4. 孤児ファイル
echo "=== Orphaned Files (no user/group) ==="
orphans=$(find / -nouser -o -nogroup 2>/dev/null | head -20)
if [ -n "$orphans" ]; then
    echo "$orphans"
    echo "[WARNING] These files have no valid owner/group"
else
    echo "None found. [OK]"
fi
echo ""

# 5. 重要ファイルのパーミッションチェック
echo "=== Critical File Permissions ==="
check_perm() {
    local file="$1"
    local expected="$2"
    if [ -f "$file" ]; then
        actual=$(stat -c "%a" "$file" 2>/dev/null || stat -f "%OLp" "$file" 2>/dev/null)
        if [ "$actual" = "$expected" ]; then
            echo "[OK]   $file ($actual)"
        else
            echo "[WARN] $file ($actual, expected $expected)"
        fi
    fi
}

check_perm "/etc/passwd" "644"
check_perm "/etc/shadow" "640"
check_perm "/etc/group" "644"
check_perm "/etc/sudoers" "440"
check_perm "/etc/ssh/sshd_config" "600"
echo ""

# 6. SSH 鍵のパーミッション
echo "=== SSH Key Permissions ==="
if [ -d ~/.ssh ]; then
    ls -la ~/.ssh/ 2>/dev/null
    ssh_dir_perm=$(stat -c "%a" ~/.ssh 2>/dev/null || stat -f "%OLp" ~/.ssh 2>/dev/null)
    if [ "$ssh_dir_perm" != "700" ]; then
        echo "[WARN] ~/.ssh should be 700, is $ssh_dir_perm"
    fi
fi
echo ""

echo "============================================"
echo " Audit Complete"
echo "============================================"
```

### 7.3 よくあるパーミッション設定パターン

```bash
# ===== パターン1: 共有プロジェクトディレクトリ =====
sudo mkdir -p /opt/project
sudo groupadd project-team
sudo usermod -aG project-team user1
sudo usermod -aG project-team user2

sudo chown root:project-team /opt/project
sudo chmod 2775 /opt/project
# SGID(2) + 所有者rwx(7) + グループrwx(7) + その他rx(5)
# 新規ファイルはproject-teamグループに自動設定

# ===== パターン2: 機密ドキュメント管理 =====
sudo mkdir -p /secure/documents
sudo chmod 700 /secure
sudo chmod 700 /secure/documents
sudo chown manager:management /secure/documents

# ACLで特定メンバーにアクセス許可
setfacl -m u:assistant:rx /secure/documents
setfacl -d -m u:assistant:rx /secure/documents

# ===== パターン3: ログ収集ディレクトリ =====
sudo mkdir -p /var/log/app
sudo chown root:adm /var/log/app
sudo chmod 2750 /var/log/app
# アプリケーションが書き込み、admグループが読み取り

# logrotate 用の設定
sudo chmod 640 /var/log/app/*.log

# ===== パターン4: CI/CDデプロイ用 =====
sudo mkdir -p /var/www/app
sudo groupadd deployers
sudo chown root:deployers /var/www/app
sudo chmod 2775 /var/www/app

# デプロイユーザーに権限
sudo usermod -aG deployers deploy-bot
setfacl -R -m g:deployers:rwx /var/www/app
setfacl -R -d -m g:deployers:rwx /var/www/app

# Webサーバーには読み取りのみ
setfacl -R -m u:www-data:rx /var/www/app
setfacl -R -d -m u:www-data:rx /var/www/app

# ===== パターン5: バックアップディレクトリ =====
sudo mkdir -p /backup
sudo chown root:backup /backup
sudo chmod 770 /backup
# バックアップスクリプトはbackupグループで実行

# 個別バックアップの保護
sudo chmod 600 /backup/*.tar.gz

# ===== パターン6: 一時的な作業ディレクトリ =====
sudo mkdir -p /tmp/shared-work
sudo chmod 1777 /tmp/shared-work
# Sticky Bit: 他人のファイルは削除不可、全員が書き込み可能
```

---

## 8. Linux のセキュリティモジュール

### 8.1 SELinux の基本

```bash
# SELinux (Security-Enhanced Linux)
# Red Hat系（RHEL, CentOS, Fedora）でデフォルト有効
# 標準のUNIXパーミッションに加えて、強制アクセス制御(MAC)を提供

# SELinuxの状態確認
getenforce
# Enforcing  : ポリシーを強制（違反はブロック）
# Permissive : ポリシー違反をログのみ（ブロックしない）
# Disabled   : SELinux無効

sestatus
# SELinux status:                 enabled
# SELinuxfs mount:                /sys/fs/selinux
# SELinux root directory:         /etc/selinux
# Loaded policy name:             targeted
# Current mode:                   enforcing

# 一時的にモード変更（再起動で戻る）
sudo setenforce 0       # Permissive に変更
sudo setenforce 1       # Enforcing に変更

# 恒久的な設定変更
sudo vi /etc/selinux/config
# SELINUX=enforcing   ← enforcing, permissive, disabled

# SELinuxコンテキストの表示
ls -Z file.txt
# -rw-r--r--. user group unconfined_u:object_r:user_home_t:s0 file.txt
#                         ↑ユーザー    ↑ロール   ↑タイプ    ↑レベル

ps -eZ | grep httpd
# system_u:system_r:httpd_t:s0  1234 ?  00:00:01 httpd

# SELinuxコンテキストの変更
sudo chcon -t httpd_sys_content_t /var/www/html/index.html
sudo chcon -R -t httpd_sys_content_t /var/www/html/

# デフォルトコンテキストに復元
sudo restorecon -v file.txt
sudo restorecon -Rv /var/www/html/

# SELinuxブール値の管理
getsebool -a                             # 全ブール値を表示
getsebool httpd_can_network_connect      # 特定のブール値

# ブール値の設定
sudo setsebool httpd_can_network_connect on        # 一時的
sudo setsebool -P httpd_can_network_connect on     # 恒久的

# よく使うSELinuxブール値
sudo setsebool -P httpd_can_network_connect on     # Apache外部接続許可
sudo setsebool -P httpd_enable_homedirs on         # ホームディレクトリ許可
sudo setsebool -P httpd_can_sendmail on            # メール送信許可

# SELinux違反ログの確認
sudo ausearch -m AVC -ts recent
sudo sealert -a /var/log/audit/audit.log
sudo journalctl -t setroubleshoot

# ポート管理
sudo semanage port -l | grep http       # HTTP関連ポート一覧
sudo semanage port -a -t http_port_t -p tcp 8080  # カスタムポート追加
```

### 8.2 AppArmor の基本

```bash
# AppArmor
# Ubuntu, Debian, SUSE でデフォルト有効
# プロファイルベースの強制アクセス制御

# AppArmorの状態確認
sudo aa-status
# apparmor module is loaded.
# 42 profiles are loaded.
# 25 profiles are in enforce mode.
# 17 profiles are in complain mode.

# プロファイルの場所
ls /etc/apparmor.d/

# プロファイルのモード
# enforce  : ポリシーを強制
# complain : ポリシー違反をログのみ（学習モード）
# disabled : プロファイル無効

# モード変更
sudo aa-enforce /etc/apparmor.d/usr.sbin.apache2    # 強制モード
sudo aa-complain /etc/apparmor.d/usr.sbin.apache2   # 学習モード
sudo aa-disable /etc/apparmor.d/usr.sbin.apache2    # 無効化

# プロファイルの読み込み
sudo apparmor_parser -r /etc/apparmor.d/usr.sbin.apache2  # 再読み込み
sudo apparmor_parser -R /etc/apparmor.d/usr.sbin.apache2  # 削除

# ログの確認（違反の検出）
sudo journalctl -k | grep apparmor
sudo dmesg | grep apparmor
# DENIED を検索
sudo journalctl -k | grep "apparmor.*DENIED"

# プロファイルの生成
sudo aa-genprof /usr/bin/myapp
# → アプリケーションを操作して学習させる
# → プロファイルが自動生成される

# 簡易プロファイル例（/etc/apparmor.d/usr.local.bin.myapp）
# /usr/local/bin/myapp {
#   /usr/local/bin/myapp mr,         # 自身の読み取り+実行
#   /etc/myapp/** r,                 # 設定ファイル読み取り
#   /var/log/myapp/** w,             # ログ書き込み
#   /tmp/myapp-* rw,                 # 一時ファイル読み書き
#   /usr/lib/** rm,                  # ライブラリ読み取り+実行
#   network tcp,                      # TCP通信許可
# }
```

### 8.3 Linux Capabilities

```bash
# Linux Capabilities: root権限を細分化した仕組み
# 特定の権限だけをプロセス/ファイルに付与できる

# 主要な Capabilities
# CAP_NET_BIND_SERVICE : 1024未満のポートにバインド
# CAP_NET_RAW          : RAWソケットの使用（ping等）
# CAP_SYS_ADMIN        : 多くの管理操作
# CAP_DAC_OVERRIDE     : ファイルアクセス制御のバイパス
# CAP_CHOWN            : ファイル所有者の変更
# CAP_KILL             : 他ユーザーのプロセスにシグナル送信
# CAP_SETUID/SETGID    : UID/GIDの変更

# ファイルの Capability 確認
getcap /usr/bin/ping
# /usr/bin/ping cap_net_raw=ep

# 全ファイルの Capability を検索
getcap -r / 2>/dev/null

# Capability の設定（要root）
sudo setcap cap_net_bind_service=+ep /usr/bin/myapp
# → myapp は非rootでも80番ポートにバインド可能

# Capability の削除
sudo setcap -r /usr/bin/myapp

# プロセスの Capability 確認
cat /proc/$$/status | grep -i cap
# CapInh: 0000000000000000  (継承可能)
# CapPrm: 0000000000000000  (許可)
# CapEff: 0000000000000000  (有効)
# CapBnd: 000001ffffffffff  (バウンディングセット)
# CapAmb: 0000000000000000  (アンビエント)

# 数値を読みやすく変換
capsh --decode=000001ffffffffff

# 実践例: SUIDの代わりにCapabilityを使用
# pingコマンド（従来はSUID）
sudo chmod u-s /usr/bin/ping
sudo setcap cap_net_raw=ep /usr/bin/ping
# → SUIDなしでもpingが動作

# 実践例: Node.jsアプリを非rootで80番ポートで動作
sudo setcap cap_net_bind_service=+ep /usr/bin/node
# または
sudo setcap 'cap_net_bind_service=+ep' $(which node)
```

---

## 9. トラブルシューティング

### 9.1 よくあるパーミッションエラー

```bash
# ===== エラー1: Permission denied =====
$ cat /etc/shadow
cat: /etc/shadow: Permission denied

# 原因: 読み取り権限がない
# 解決:
ls -la /etc/shadow     # パーミッション確認
sudo cat /etc/shadow   # root権限で実行

# ===== エラー2: Operation not permitted =====
$ chown user file.txt
chown: changing ownership of 'file.txt': Operation not permitted

# 原因: chown はroot権限が必要
# 解決:
sudo chown user file.txt

# ===== エラー3: SSH Permission denied =====
# "Permissions 0644 for '~/.ssh/id_rsa' are too open."
# 原因: SSH鍵のパーミッションが緩すぎる
# 解決:
chmod 600 ~/.ssh/id_rsa
chmod 700 ~/.ssh

# ===== エラー4: bash: ./script.sh: Permission denied =====
# 原因: 実行権限がない
# 解決:
chmod +x script.sh
# または
bash script.sh         # bash を明示的に呼ぶ

# ===== エラー5: mkdir: cannot create directory: Permission denied =====
# 原因: 親ディレクトリへの書き込み権限がない
# 解決:
ls -la /parent/dir/    # 親ディレクトリのパーミッション確認
sudo mkdir /parent/dir/newdir

# ===== エラー6: rm: cannot remove: Operation not permitted =====
# 原因1: ファイルがimmutableフラグ付き
lsattr file.txt        # フラグ確認
# ----i--------e-- file.txt  ← i=immutable
sudo chattr -i file.txt  # immutableフラグ解除
rm file.txt

# 原因2: Sticky Bit付きディレクトリで他人のファイル
ls -ld /tmp            # Sticky Bit確認
# drwxrwxrwt ...       ← t がある

# ===== エラー7: cp: cannot create regular file: Permission denied =====
# 原因: コピー先のディレクトリへの書き込み権限がない
ls -la /destination/   # コピー先のパーミッション確認
# 解決:
sudo cp file.txt /destination/
# または
chmod u+w /destination/
```

### 9.2 ファイル属性（chattr / lsattr）

```bash
# chattr: 拡張ファイル属性の設定（パーミッションとは別の制御レイヤー）
# lsattr: 拡張ファイル属性の表示

# 主要な属性
# i (immutable) : 変更・削除・名前変更・リンク作成不可（rootでも！）
# a (append)    : 追記のみ可能（ログファイル向き）
# e (extent)    : ext4のエクステント使用（通常デフォルト）
# A (noatime)   : アクセス時刻を更新しない
# S (sync)      : 即座にディスクに書き込み
# d (nodump)    : dump によるバックアップ対象外
# c (compress)  : 自動圧縮（対応FS必要）

# 属性の表示
lsattr file.txt
# ----i--------e-- file.txt

lsattr -d dir/        # ディレクトリの属性
lsattr -R dir/        # 再帰的に表示

# immutable 属性の設定（最強の保護）
sudo chattr +i important.conf
# → root でも変更・削除できなくなる
# → 解除するには chattr -i が必要

rm important.conf
# rm: cannot remove 'important.conf': Operation not permitted

sudo rm important.conf
# rm: cannot remove 'important.conf': Operation not permitted
# → root でも削除できない！

# immutable の解除
sudo chattr -i important.conf
rm important.conf     # OK

# append-only 属性（ログファイル向き）
sudo chattr +a /var/log/secure.log
echo "new entry" >> /var/log/secure.log    # OK（追記）
echo "overwrite" > /var/log/secure.log     # エラー（上書き不可）
rm /var/log/secure.log                      # エラー（削除不可）

# 実践例: 重要な設定ファイルの保護
sudo chattr +i /etc/resolv.conf    # DNS設定の保護
sudo chattr +i /etc/passwd         # パスワードファイルの保護
sudo chattr +i /etc/shadow         # シャドウファイルの保護
# 注意: ユーザー管理ツールが動作しなくなるので一時的な保護のみ推奨

# 実践例: ブートファイルの保護
sudo chattr +i /boot/vmlinuz-*
sudo chattr +i /boot/initrd.img-*

# immutable フラグ付きファイルの検索
lsattr -R / 2>/dev/null | grep -- "----i"
```

### 9.3 デバッグテクニック

```bash
# ===== namei: パス全体のパーミッションチェック =====
# ファイルにアクセスできない時、パスのどこで権限がないか確認
namei -l /var/www/html/index.html
# f: /var/www/html/index.html
# dr-xr-xr-x root root /
# drwxr-xr-x root root var
# drwxr-xr-x root root www
# drwxr-xr-x root root html
# -rw-r--r-- root root index.html

# 各ディレクトリの x ビットを確認
# x がないディレクトリがあるとそこでブロックされる

# ===== strace: システムコールレベルでの権限エラー追跡 =====
strace -f -e trace=open,openat,access cat /etc/shadow 2>&1
# openat(AT_FDCWD, "/etc/shadow", O_RDONLY) = -1 EACCES (Permission denied)

# ===== sudo -l: 利用可能なsudo権限の確認 =====
sudo -l
# User user may run the following commands on host:
#     (ALL : ALL) ALL
#     (root) /usr/bin/systemctl restart apache2

# ===== getfacl: ACLを含む完全なパーミッション確認 =====
getfacl /path/to/file

# ===== /proc/self/status: 現在のプロセスの権限情報 =====
cat /proc/self/status | grep -E "Uid|Gid|Groups|Cap"

# ===== loginctl: ログインセッションの確認 =====
loginctl show-user $USER

# ===== パーミッション変更履歴の確認 =====
# auditd が有効な場合
sudo ausearch -f /path/to/file -ts recent
# ファイルへのアクセス試行と権限変更を記録

# auditルールの追加
sudo auditctl -w /etc/passwd -p wa -k passwd_changes
# -w: 監視対象ファイル
# -p: 監視する操作 (w=write, a=attribute change)
# -k: ログ検索用のキー

# ===== inotifywait: リアルタイムファイル監視 =====
# パーミッション変更をリアルタイムで監視
inotifywait -m -e attrib /path/to/file
# /path/to/file ATTRIB  ← パーミッションや属性が変更された

# ===== テスト用ユーザーでの確認 =====
# 特定ユーザーの視点でアクセスをテスト
sudo -u www-data cat /var/www/html/config.php
sudo -u postgres ls /var/lib/postgresql/

# ===== access コマンド（test）での確認 =====
# 現在のユーザーがアクセスできるかテスト
test -r file.txt && echo "Readable" || echo "Not readable"
test -w file.txt && echo "Writable" || echo "Not writable"
test -x file.txt && echo "Executable" || echo "Not executable"

# スクリプトでの活用
if [ ! -r "$config_file" ]; then
    echo "Error: Cannot read $config_file" >&2
    echo "Current permissions: $(ls -la "$config_file")" >&2
    echo "Run: sudo chmod +r $config_file" >&2
    exit 1
fi
```

---

## 10. macOS 固有のパーミッション

### 10.1 macOS のパーミッション特性

```bash
# macOS は BSD ベースのパーミッションシステム + 独自拡張
# POSIX パーミッション + ACL + SIP + TCC + Sandbox

# 基本的なパーミッション操作はLinuxと同じ
chmod 755 file.txt
chown user:group file.txt
# ただし stat コマンドの書式がBSD形式

# macOS の stat（BSD版）
stat -f "%Sp %Su %Sg %z %N" file.txt
# -rw-r--r-- user staff 4096 file.txt

# 数値表示
stat -f "%OLp %N" file.txt
# 644 file.txt

# macOS のデフォルトグループは "staff"
id
# uid=501(user) gid=20(staff) groups=20(staff),80(admin),...

# macOS のACL
# macOS は独自のACLフォーマットを使用
ls -le file.txt              # ACL表示（-e オプション）
# -rw-r--r--+ 1 user staff 4096 Jan 1 file.txt
#  0: user:alice allow read,write
#  1: group:devteam allow read

# macOS ACLの設定
chmod +a "user:alice allow read,write" file.txt
chmod +a "group:devteam allow read" file.txt
chmod +a "user:bob deny write" file.txt       # 拒否ルール

# macOS ACLの削除
chmod -a "user:alice allow read,write" file.txt
chmod -a# 0 file.txt          # インデックス0のACLを削除
chmod -N file.txt              # 全ACL削除

# ACLの順序指定
chmod +a# 0 "user:alice allow read,write" file.txt  # 先頭に挿入
```

### 10.2 SIP（System Integrity Protection）

```bash
# SIP: macOS のシステムファイル保護機能
# rootでも保護されたファイル/ディレクトリの変更不可

# SIPの状態確認
csrutil status
# System Integrity Protection status: enabled.

# SIPで保護されるディレクトリ
# /System
# /usr (ただし /usr/local は除く)
# /bin
# /sbin
# /Applications（プリインストールアプリ）

# SIP保護下でのエラー例
sudo rm /usr/bin/python3
# rm: /usr/bin/python3: Operation not permitted
# → SIPにより rootでも削除不可

# SIPの無効化（推奨されない、開発時のみ）
# リカバリモードで起動 → ターミナル → csrutil disable
# 再起動後に有効になる

# /usr/local は SIP の対象外
# Homebrew はここにインストールされる
ls -la /usr/local/bin/
```

### 10.3 xattr（拡張属性）

```bash
# macOS の拡張属性（Extended Attributes）
# ファイルにメタデータを付加する仕組み

# 拡張属性の表示
xattr file.txt
xattr -l file.txt                    # 値も表示

# よく見る拡張属性
# com.apple.quarantine        : ダウンロードしたファイルの隔離フラグ
# com.apple.metadata:kMDItemWhereFroms : ダウンロード元URL
# com.apple.FinderInfo        : Finderの表示情報
# com.apple.ResourceFork      : リソースフォーク

# 隔離フラグの確認
xattr -p com.apple.quarantine downloaded_file
# 0083;5f8b1234;Chrome;...

# 隔離フラグの削除（Gatekeeperの警告解除）
xattr -d com.apple.quarantine downloaded_file
# ディレクトリ全体から削除
xattr -dr com.apple.quarantine ~/Downloads/app.app

# 全拡張属性の削除
xattr -c file.txt

# 拡張属性の設定
xattr -w com.example.note "important file" file.txt

# 再帰的に拡張属性を削除
xattr -cr dir/

# @ マーク: ls で拡張属性の存在を示す
ls -la
# -rw-r--r--@ 1 user staff 4096 Jan 1 downloaded.zip
#            ^ @ は拡張属性あり
```

---

## 実践演習

### 演習1: [基礎] — パーミッション操作

```bash
# スクリプトに実行権限を付与
echo '#!/bin/bash' > /tmp/test.sh
echo 'echo "Hello from test.sh!"' >> /tmp/test.sh
echo 'echo "Running as: $(whoami)"' >> /tmp/test.sh
echo 'echo "PID: $$"' >> /tmp/test.sh

# パーミッション確認
ls -la /tmp/test.sh
# -rw-r--r-- 1 user group ... test.sh  (実行権なし)

# 実行を試みる（失敗するはず）
/tmp/test.sh
# bash: /tmp/test.sh: Permission denied

# 実行権限付与
chmod +x /tmp/test.sh
ls -la /tmp/test.sh
# -rwxr-xr-x 1 user group ... test.sh  (実行権あり)

# 実行
/tmp/test.sh
# Hello from test.sh!
# Running as: user
# PID: 12345

# パーミッションを数値で確認
stat -c "%a %n" /tmp/test.sh 2>/dev/null || stat -f "%OLp %N" /tmp/test.sh
# 755 /tmp/test.sh

# クリーンアップ
rm /tmp/test.sh
```

### 演習2: [基礎] — umask の理解

```bash
# 現在のumaskを確認
echo "Current umask: $(umask)"
echo "Symbolic: $(umask -S)"

# umask 022 でファイル作成
umask 022
touch /tmp/umask_022.txt
mkdir /tmp/umask_022_dir
echo "umask 022:"
stat -c "  File: %a %n" /tmp/umask_022.txt 2>/dev/null
stat -c "  Dir:  %a %n" /tmp/umask_022_dir 2>/dev/null
# File: 644
# Dir:  755

# umask 077 でファイル作成
umask 077
touch /tmp/umask_077.txt
mkdir /tmp/umask_077_dir
echo "umask 077:"
stat -c "  File: %a %n" /tmp/umask_077.txt 2>/dev/null
stat -c "  Dir:  %a %n" /tmp/umask_077_dir 2>/dev/null
# File: 600
# Dir:  700

# umask 002 でファイル作成
umask 002
touch /tmp/umask_002.txt
mkdir /tmp/umask_002_dir
echo "umask 002:"
stat -c "  File: %a %n" /tmp/umask_002.txt 2>/dev/null
stat -c "  Dir:  %a %n" /tmp/umask_002_dir 2>/dev/null
# File: 664
# Dir:  775

# umask を元に戻す
umask 022

# クリーンアップ
rm -f /tmp/umask_*.txt
rmdir /tmp/umask_*_dir
```

### 演習3: [中級] — チーム共有ディレクトリの構築

```bash
# チーム共有ディレクトリを作成して適切なパーミッションを設定
# この演習はroot権限が必要

# 1. グループの作成
sudo groupadd exercise-team 2>/dev/null

# 2. テストユーザーの作成（演習用）
sudo useradd -m -G exercise-team member1 2>/dev/null
sudo useradd -m -G exercise-team member2 2>/dev/null

# 3. 共有ディレクトリの作成
sudo mkdir -p /tmp/team-share

# 4. 所有者とグループの設定
sudo chown root:exercise-team /tmp/team-share

# 5. SGID + Sticky Bit の設定
sudo chmod 3770 /tmp/team-share
# 3 = SGID(2) + Sticky(1)
# 770 = rwx rwx ---

# 確認
ls -ld /tmp/team-share
# drwxrws--T 2 root exercise-team 4096 ... team-share
# s = SGID（新規ファイルにグループが継承される）
# T = Sticky（他メンバーのファイルは削除不可）
# 大文字Tはotherにxがないため

# 6. テスト: member1がファイル作成
sudo -u member1 touch /tmp/team-share/member1_file.txt
ls -la /tmp/team-share/member1_file.txt
# -rw-r--r-- 1 member1 exercise-team ... member1_file.txt
# → グループが exercise-team に自動設定（SGID効果）

# 7. テスト: member2がmember1のファイルを削除（失敗するはず）
sudo -u member2 rm /tmp/team-share/member1_file.txt 2>&1
# rm: cannot remove ...: Operation not permitted（Sticky Bit効果）

# 8. テスト: member2が自分のファイルを作成・削除（成功するはず）
sudo -u member2 touch /tmp/team-share/member2_file.txt
sudo -u member2 rm /tmp/team-share/member2_file.txt

# クリーンアップ
sudo rm -rf /tmp/team-share
sudo userdel -r member1 2>/dev/null
sudo userdel -r member2 2>/dev/null
sudo groupdel exercise-team 2>/dev/null
```

### 演習4: [中級] — ACL の設定と検証

```bash
# ACL を使って複雑なアクセス制御を実装

# 1. テスト環境の作成
mkdir -p /tmp/acl-exercise/docs
mkdir -p /tmp/acl-exercise/code

echo "Secret document" > /tmp/acl-exercise/docs/secret.txt
echo "Public code" > /tmp/acl-exercise/code/main.py

# 2. 基本パーミッションの設定
chmod 700 /tmp/acl-exercise/docs/
chmod 755 /tmp/acl-exercise/code/

# 3. ACLの設定（自分のユーザーで実験）
# 特定ユーザーにdocsへの読み取りアクセスを許可
# （存在するユーザー名で実行してください）
setfacl -m u:$(whoami):rx /tmp/acl-exercise/docs/
setfacl -m u:$(whoami):r /tmp/acl-exercise/docs/secret.txt

# 4. ACLの確認
echo "=== Directory ACL ==="
getfacl /tmp/acl-exercise/docs/
echo ""
echo "=== File ACL ==="
getfacl /tmp/acl-exercise/docs/secret.txt

# 5. デフォルトACLの設定（新規ファイルに自動適用）
setfacl -d -m u:$(whoami):r /tmp/acl-exercise/docs/

# 6. デフォルトACLの検証
touch /tmp/acl-exercise/docs/new_file.txt
echo "=== New file ACL (should inherit default) ==="
getfacl /tmp/acl-exercise/docs/new_file.txt

# 7. ls で + マークを確認
ls -la /tmp/acl-exercise/docs/
# + マークが表示されるはず

# 8. ACLのバックアップと復元
getfacl -R /tmp/acl-exercise/ > /tmp/acl_backup.txt
echo "=== ACL Backup ==="
cat /tmp/acl_backup.txt

# 9. ACLの削除
setfacl -b /tmp/acl-exercise/docs/secret.txt
echo "=== After ACL removal ==="
getfacl /tmp/acl-exercise/docs/secret.txt

# 10. バックアップからの復元
setfacl --restore=/tmp/acl_backup.txt
echo "=== After ACL restore ==="
getfacl /tmp/acl-exercise/docs/secret.txt

# クリーンアップ
rm -rf /tmp/acl-exercise /tmp/acl_backup.txt
```

### 演習5: [上級] — セキュリティ監査スクリプト

```bash
#!/bin/bash
# security-check.sh
# 自分のホームディレクトリのパーミッションを監査

HOME_DIR="${HOME}"
ISSUES=0

echo "============================================"
echo " Home Directory Security Check"
echo " User: $(whoami)"
echo " Home: ${HOME_DIR}"
echo " Date: $(date)"
echo "============================================"
echo ""

# 1. ホームディレクトリのパーミッション
echo "=== 1. Home Directory ==="
home_perm=$(stat -c "%a" "${HOME_DIR}" 2>/dev/null || stat -f "%OLp" "${HOME_DIR}" 2>/dev/null)
if [ "$home_perm" -gt 755 ]; then
    echo "[WARN] Home directory is too permissive: $home_perm (should be 755 or less)"
    ISSUES=$((ISSUES + 1))
else
    echo "[OK]   Home directory: $home_perm"
fi
echo ""

# 2. SSH ディレクトリ
echo "=== 2. SSH Configuration ==="
if [ -d "${HOME_DIR}/.ssh" ]; then
    ssh_perm=$(stat -c "%a" "${HOME_DIR}/.ssh" 2>/dev/null || stat -f "%OLp" "${HOME_DIR}/.ssh" 2>/dev/null)
    if [ "$ssh_perm" != "700" ]; then
        echo "[WARN] .ssh directory: $ssh_perm (should be 700)"
        ISSUES=$((ISSUES + 1))
    else
        echo "[OK]   .ssh directory: $ssh_perm"
    fi

    # 秘密鍵のチェック
    for key in "${HOME_DIR}"/.ssh/id_*; do
        if [ -f "$key" ] && [[ ! "$key" == *.pub ]]; then
            key_perm=$(stat -c "%a" "$key" 2>/dev/null || stat -f "%OLp" "$key" 2>/dev/null)
            if [ "$key_perm" != "600" ] && [ "$key_perm" != "400" ]; then
                echo "[WARN] Private key $key: $key_perm (should be 600 or 400)"
                ISSUES=$((ISSUES + 1))
            else
                echo "[OK]   Private key $key: $key_perm"
            fi
        fi
    done
else
    echo "[INFO] No .ssh directory found"
fi
echo ""

# 3. 秘密情報ファイル
echo "=== 3. Sensitive Files ==="
for pattern in ".env" ".env.*" "*.pem" "*.key" "credentials*" "secret*"; do
    while IFS= read -r -d '' sensitive; do
        s_perm=$(stat -c "%a" "$sensitive" 2>/dev/null || stat -f "%OLp" "$sensitive" 2>/dev/null)
        if [ "$s_perm" -gt 600 ]; then
            echo "[WARN] $sensitive: $s_perm (should be 600 or less)"
            ISSUES=$((ISSUES + 1))
        else
            echo "[OK]   $sensitive: $s_perm"
        fi
    done < <(find "${HOME_DIR}" -maxdepth 3 -name "$pattern" -type f -print0 2>/dev/null)
done
echo ""

# 4. 世界書き込み可能ファイル
echo "=== 4. World-Writable Files ==="
ww_files=$(find "${HOME_DIR}" -perm -002 -type f 2>/dev/null | head -10)
if [ -n "$ww_files" ]; then
    echo "$ww_files"
    ww_count=$(find "${HOME_DIR}" -perm -002 -type f 2>/dev/null | wc -l)
    echo "[WARN] Found $ww_count world-writable files"
    ISSUES=$((ISSUES + $ww_count))
else
    echo "[OK]   No world-writable files found"
fi
echo ""

# 5. 実行可能な隠しファイル
echo "=== 5. Hidden Executable Files ==="
hidden_exec=$(find "${HOME_DIR}" -maxdepth 2 -name ".*" -perm /111 -type f 2>/dev/null | head -10)
if [ -n "$hidden_exec" ]; then
    echo "$hidden_exec"
    echo "[INFO] Review these hidden executable files"
else
    echo "[OK]   No suspicious hidden executables"
fi
echo ""

# サマリー
echo "============================================"
if [ $ISSUES -eq 0 ]; then
    echo " Result: All checks passed!"
else
    echo " Result: $ISSUES issue(s) found"
    echo " Run the suggested fixes to resolve them."
fi
echo "============================================"
```

### 演習6: [上級] — パーミッション変更の監視

```bash
# inotifywait を使ったリアルタイムパーミッション監視
# インストール: sudo apt install inotify-tools

#!/bin/bash
# watch-permissions.sh
# 指定ディレクトリのパーミッション変更を監視

WATCH_DIR="${1:-.}"
LOG_FILE="/tmp/permission_changes.log"

echo "Watching permission changes in: $WATCH_DIR"
echo "Log file: $LOG_FILE"
echo "Press Ctrl+C to stop"

# inotifywait で attrib（属性変更）イベントを監視
inotifywait -m -r -e attrib --format '%T %w%f %e' \
    --timefmt '%Y-%m-%d %H:%M:%S' \
    "$WATCH_DIR" 2>/dev/null | while read -r line; do

    timestamp=$(echo "$line" | awk '{print $1, $2}')
    filepath=$(echo "$line" | awk '{print $3}')
    event=$(echo "$line" | awk '{print $4}')

    if [ -e "$filepath" ]; then
        perms=$(stat -c "%a %A %U:%G" "$filepath" 2>/dev/null)
        echo "[$timestamp] $filepath -> $perms ($event)" | tee -a "$LOG_FILE"
    fi
done

# 別ターミナルでテスト:
# chmod 777 /path/to/watched/file
# → 監視画面にリアルタイムで表示される

# auditd を使った監視（より本格的）
# sudo auditctl -w /etc/passwd -p wa -k passwd_watch
# sudo ausearch -k passwd_watch
```

---

## まとめ

| カテゴリ | 操作 | コマンド |
|---------|------|---------|
| パーミッション表示 | 一覧表示 | `ls -la`, `stat` |
| パーミッション変更 | 数値指定 | `chmod 755 file` |
| パーミッション変更 | シンボリック | `chmod u+x file` |
| パーミッション変更 | 再帰的 | `chmod -R 755 dir/` |
| パーミッション変更 | ファイル/ディレクトリ分離 | `find -type f/d -exec chmod` |
| 所有者変更 | 所有者+グループ | `chown user:group file` |
| グループ変更 | グループのみ | `chgrp group file` |
| デフォルトパーミッション | umask設定 | `umask 022` |
| 特殊ビット | SUID | `chmod u+s file` / `chmod 4755` |
| 特殊ビット | SGID | `chmod g+s dir/` / `chmod 2755` |
| 特殊ビット | Sticky | `chmod +t dir/` / `chmod 1777` |
| ACL表示 | ACL確認 | `getfacl file` |
| ACL設定 | ユーザーACL | `setfacl -m u:user:rw file` |
| ACL設定 | グループACL | `setfacl -m g:group:rx file` |
| ACL設定 | デフォルトACL | `setfacl -d -m u:user:rw dir/` |
| ACL削除 | 全削除 | `setfacl -b file` |
| ファイル属性 | 不変設定 | `chattr +i file` |
| ファイル属性 | 追記のみ | `chattr +a file` |
| ファイル属性 | 属性確認 | `lsattr file` |
| Capability | 確認 | `getcap file` |
| Capability | 設定 | `setcap cap_xxx=+ep file` |
| セキュリティ監査 | SUID検索 | `find / -perm -4000` |
| セキュリティ監査 | 世界書込検索 | `find / -perm -002` |
| セキュリティ監査 | 孤児ファイル | `find / -nouser -o -nogroup` |
| SELinux | 状態確認 | `getenforce`, `sestatus` |
| AppArmor | 状態確認 | `aa-status` |

### パーミッション設定クイックリファレンス

```
用途                    パーミッション  コマンド
─────────────────────────────────────────────────
公開Webファイル          644            chmod 644 index.html
公開ディレクトリ         755            chmod 755 public/
実行スクリプト           755            chmod 755 script.sh
SSH秘密鍵              600 or 400     chmod 600 ~/.ssh/id_rsa
.sshディレクトリ        700            chmod 700 ~/.ssh
秘密設定ファイル         600            chmod 600 .env
共有ディレクトリ         2775           chmod 2775 /shared/
一時ディレクトリ         1777           chmod 1777 /tmp/
ログファイル            640            chmod 640 app.log
SSL秘密鍵              600            chmod 600 server.key
SSL証明書              644            chmod 644 server.crt
crontab                600            chmod 600 crontab
sudoers                440            chmod 440 /etc/sudoers
```

---

## 次に読むべきガイド
-> [[03-find-and-locate.md]] -- ファイル検索

---

## 参考文献
1. Kerrisk, M. "The Linux Programming Interface." Ch.15: File Attributes, 2010.
2. Shotts, W. "The Linux Command Line." Ch.9: Permissions, 5th ed., 2019.
3. Nemeth, E. et al. "UNIX and Linux System Administration Handbook." Ch.5: Access Control, 5th ed., 2017.
4. Red Hat. "SELinux User's and Administrator's Guide." Red Hat Enterprise Linux 8 Documentation.
5. Ubuntu. "AppArmor." Ubuntu Community Help Wiki.
6. POSIX.1-2017. "IEEE Std 1003.1: File Access Permissions."
7. Grüenbacher, A. "POSIX Access Control Lists on Linux." USENIX, 2003.
