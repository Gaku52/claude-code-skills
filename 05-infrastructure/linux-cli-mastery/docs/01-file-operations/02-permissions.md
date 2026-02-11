# パーミッションと所有者

> Unixの「全てはファイル」哲学において、パーミッション管理はセキュリティの要。

## この章で学ぶこと

- [ ] ファイルパーミッションの読み方・変更方法をマスターする
- [ ] 所有者とグループの管理ができる

---

## 1. パーミッションの基本

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

# 数値表現
# r=4, w=2, x=1
# rwxr-xr-- = 754
chmod 755 file.txt         # rwxr-xr-x
chmod 644 file.txt         # rw-r--r--
chmod 600 file.txt         # rw-------（秘密ファイル）

# シンボリック表現
chmod u+x file.txt         # 所有者に実行権追加
chmod g-w file.txt         # グループから書込権削除
chmod o=r file.txt         # その他に読取のみ
chmod a+r file.txt         # 全員に読取権追加
chmod -R 755 dir/          # 再帰的に変更

# umask（デフォルトパーミッション）
umask                      # 現在のumask表示（例: 022）
# ファイル: 666 - 022 = 644 (rw-r--r--)
# ディレクトリ: 777 - 022 = 755 (rwxr-xr-x)
```

---

## 2. 所有者とグループ

```bash
# 所有者変更（要root）
sudo chown user file.txt           # 所有者変更
sudo chown user:group file.txt     # 所有者+グループ変更
sudo chown -R user:group dir/      # 再帰的に変更
sudo chgrp group file.txt          # グループのみ変更

# グループ管理
groups                             # 自分のグループ一覧
id                                 # UID, GID, グループ一覧
sudo usermod -aG docker $USER      # dockerグループに追加

# 特殊パーミッション
chmod u+s file       # SUID: 実行時に所有者権限
chmod g+s dir        # SGID: ディレクトリ内新規ファイルに親のグループ
chmod +t dir         # Sticky: ディレクトリ内の削除を所有者のみ許可
```

---

## 実践演習

### 演習1: [基礎] — パーミッション操作

```bash
# スクリプトに実行権限を付与
echo '#!/bin/bash' > /tmp/test.sh
echo 'echo "Hello!"' >> /tmp/test.sh
ls -la /tmp/test.sh             # パーミッション確認
chmod +x /tmp/test.sh           # 実行権限付与
/tmp/test.sh                    # 実行
```

---

## まとめ

| 操作 | コマンド |
|------|---------|
| パーミッション変更 | chmod 755, chmod u+x |
| 所有者変更 | chown user:group |
| デフォルト | umask 022 |
| SUID/SGID/Sticky | chmod u+s, g+s, +t |

---

## 次に読むべきガイド
→ [[03-find-and-locate.md]] — ファイル検索

---

## 参考文献
1. Kerrisk, M. "The Linux Programming Interface." Ch.15, 2010.
