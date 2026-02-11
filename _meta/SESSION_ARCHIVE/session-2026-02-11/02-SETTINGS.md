# 動作確認済み権限設定

## 3箇所全て同一内容にすること

パス:
1. `/Users/gaku/.claude/settings.json`
2. `/Users/gaku/.claude/settings.local.json`
3. `/Users/gaku/.claude/projects/-Users-gaku/settings.json`

```json
{
  "permissions": {
    "allow": [
      "Bash(*)",
      "Write",
      "Edit",
      "Read",
      "WebSearch",
      "WebFetch"
    ],
    "deny": [],
    "ask": []
  }
}
```

## 注意
- `Bash(mkdir *)` のような個別パターンはSubagentで正しくマッチしない
- `Write(/path/**)` のようなパス付き指定もSubagentで失敗した
- `Write`, `Bash(*)` のように広い許可が最も確実
- settings.json にはhooksとalwaysThinkingEnabledも含まれている（上書き注意）

## settings.json の完全な内容

```json
{
  "permissions": {
    "allow": [
      "Bash(*)",
      "Write",
      "Edit",
      "Read",
      "WebSearch",
      "WebFetch"
    ]
  },
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "bash ~/.claude/memory-guard.sh",
            "timeout": 10000
          }
        ]
      }
    ]
  },
  "alwaysThinkingEnabled": true
}
```
