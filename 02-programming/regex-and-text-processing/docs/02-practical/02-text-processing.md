# Text Processing -- sed/awk/grep, Log Parsing, CSV

> Unix text processing tools (grep, sed, awk) represent the most practical applications of regular expressions. Through log parsing, CSV processing, and building data transformation pipelines, this chapter systematically explains how to leverage regular expressions on the command line.

## What You Will Learn

1. **Regex syntax of grep/sed/awk and how to choose between them** -- Each tool's strengths and how to select the right one
2. **Building log parsing pipelines** -- Practical workflows for extraction, aggregation, and formatting
3. **Regex approaches to CSV/TSV processing and their limits** -- The scope of regex applicability for structured data
4. **Integration with modern tools** -- Combined usage with ripgrep, miller, jq, and more
5. **Best practices in production** -- Techniques that balance performance, safety, and maintainability


## Prerequisites

For deeper understanding of this guide, the following knowledge is helpful:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content of [Common Patterns -- Email, URL, Date, Phone Number](./01-common-patterns.md)

---

## 1. grep -- Pattern Search

### 1.1 Basic Usage

```bash
# Basic search: display lines matching the pattern
grep 'ERROR' /var/log/syslog

# -E: use extended regular expressions (ERE)
grep -E 'ERROR|WARN' /var/log/syslog

# -i: ignore case
grep -i 'error' /var/log/syslog

# -n: show line numbers
grep -n 'ERROR' /var/log/syslog

# -c: count matching lines
grep -c 'ERROR' /var/log/syslog

# -v: show non-matching lines (invert)
grep -v 'DEBUG' /var/log/syslog

# -o: show only the matched part
grep -oE '\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}' access.log

# -A/-B/-C: show surrounding context lines
grep -A 3 'ERROR' /var/log/syslog    # 3 lines after
grep -B 2 'ERROR' /var/log/syslog    # 2 lines before
grep -C 2 'ERROR' /var/log/syslog    # 2 lines before and after
```

### 1.2 grep's Regex Options

```bash
# BRE (Basic Regular Expression) -- default
# Metacharacters + ? | ( ) { } require escaping
grep 'hello\(world\)' file.txt
grep 'a\{3\}' file.txt

# ERE (Extended Regular Expression) -- -E option
# Metacharacters can be used directly (recommended)
grep -E 'hello(world)' file.txt
grep -E 'a{3}' file.txt

# PCRE (Perl Compatible) -- -P option (GNU grep)
# Lookahead, lookbehind, \d etc. are available
grep -P '(?<=\$)\d+' file.txt
grep -P '\d+(?=円)' file.txt

# Fixed-string search -- -F option (fast)
# Does not use regex (metacharacters are literal)
grep -F '*.txt' file.txt    # search "*" as a literal
```

### 1.3 Practical grep Patterns

```bash
# Extract IP addresses
grep -oE '\b([0-9]{1,3}\.){3}[0-9]{1,3}\b' access.log

# Extract email addresses
grep -oE '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}' contacts.txt

# Specific HTTP status codes
grep -E 'HTTP/[0-9.]+" (4[0-9]{2}|5[0-9]{2})' access.log

# Filter by date range
grep -E '2026-02-(1[0-9]|2[0-9])' logfile.txt

# AND of multiple conditions (chained via pipes)
grep 'ERROR' logfile.txt | grep 'database' | grep -v 'timeout'
```

### 1.4 Recursive Search and File Selection

```bash
# -r: recursively search a directory
grep -r 'TODO' /path/to/project/

# -l: show only matching file names
grep -rl 'deprecated' src/

# --include: specify target files by pattern
grep -rn 'import' --include='*.py' src/

# --exclude: exclude specific files
grep -rn 'password' --exclude='*.log' --exclude='*.bak' .

# --exclude-dir: exclude specific directories
grep -rn 'API_KEY' --exclude-dir=node_modules --exclude-dir=.git .

# -L: show file names that did NOT match
grep -rL 'Copyright' --include='*.py' src/
```

### 1.5 Controlling grep Output and Formatting

```bash
# --color=auto: highlight matched portions
grep --color=auto 'ERROR' logfile.txt

# -H/-h: show/hide file names
grep -H 'ERROR' *.log    # with file name (default for multiple files)
grep -h 'ERROR' *.log    # without file name

# -w: word-boundary match (prevents partial matches)
grep -w 'error' logfile.txt    # matches "error" but not "errors"

# -x: only match lines that match the pattern entirely
grep -x 'OK' status.txt

# -m: limit the number of matches
grep -m 5 'ERROR' huge.log    # stop after the first 5 hits

# -q: no output (for use in conditional scripts)
if grep -q 'ERROR' logfile.txt; then
    echo "An error has been detected"
fi

# Combination of --count and --files-with-matches
grep -rl 'TODO' src/ | xargs grep -c 'TODO' | sort -t: -k2 -rn | head -10
# -> Ranking of files with the most TODOs
```

### 1.6 Performance Tuning grep and Regex

```bash
# -F (fixed strings) is faster than regex
# Always use -F when regex is unnecessary
grep -F 'NullPointerException' huge.log

# LC_ALL=C speeds up processing (avoids locale overhead)
LC_ALL=C grep 'ERROR' huge.log

# Control line buffering
grep --line-buffered 'ERROR' /var/log/syslog    # for real-time monitoring

# Handling binary files
grep -a 'pattern' binary_file       # treat binary as text
grep -I 'pattern' mixed_files       # skip binary files

# Parallel search of many files (xargs + grep)
find /var/log -name '*.log' -print0 | xargs -0 -P 4 grep -l 'ERROR'
# -P 4: run with 4 parallel processes
```

---

## 2. sed -- Stream Editing

### 2.1 Basic Usage

```bash
# Substitution: s/pattern/replacement/
sed 's/old/new/' file.txt          # replace first match per line
sed 's/old/new/g' file.txt         # replace all matches (global)
sed 's/old/new/gi' file.txt        # ignore case

# Deletion: d
sed '/^#/d' file.txt               # delete comment lines
sed '/^$/d' file.txt               # delete empty lines
sed '1,5d' file.txt                # delete lines 1-5

# Line targeting
sed '3s/old/new/' file.txt         # replace only on line 3
sed '1,10s/old/new/g' file.txt     # replace within lines 1-10
sed '/ERROR/s/old/new/g' file.txt  # replace only on lines containing ERROR

# In-place editing (-i)
sed -i 's/old/new/g' file.txt      # modify file directly (GNU)
sed -i '' 's/old/new/g' file.txt   # macOS (backup extension required)
sed -i.bak 's/old/new/g' file.txt  # with backup
```

### 2.2 Advanced sed Patterns

```bash
# Capture groups and backreferences
# Date format conversion: YYYY-MM-DD -> DD/MM/YYYY
sed -E 's/([0-9]{4})-([0-9]{2})-([0-9]{2})/\3\/\2\/\1/g' dates.txt

# Strip HTML tags
sed 's/<[^>]*>//g' page.html

# Add to start/end of line
sed 's/^/PREFIX: /' file.txt       # prepend
sed 's/$/ SUFFIX/' file.txt        # append

# Run multiple substitutions in sequence
sed -e 's/foo/bar/g' -e 's/baz/qux/g' file.txt

# Extract lines between patterns
sed -n '/START/,/END/p' file.txt   # output from START to END

# Odd/even lines
sed -n '1~2p' file.txt             # only odd lines (GNU sed)
sed -n '2~2p' file.txt             # only even lines (GNU sed)

# Whitespace normalization
```

### 2.3 sed Script Examples

```bash
# Cleansing a log file
sed -E '
    /^$/d                          # delete empty lines
    s/\t/  /g                      # tabs to 2 spaces
    s/([0-9]{4})-([0-9]{2})-([0-9]{2})/\1年\2月\3日/g  # date conversion
' logfile.txt
```

### 2.4 Advanced Operations Using sed's Hold Space

```bash
# Hold space: sed's secondary buffer
# Pattern space: the line currently being processed (used most often)
# Hold space: storage area for cross-line processing

# h: pattern space -> hold space (copy)
# H: pattern space -> hold space (append)
# g: hold space -> pattern space (copy)
# G: hold space -> pattern space (append)
# x: swap pattern space and hold space

# Reverse the order of lines
sed -n '1!G;h;$p' file.txt

# Collapse consecutive empty lines into one
sed '/^$/N;/^\n$/d' file.txt

# Join every two lines
sed 'N;s/\n/ /' file.txt

# Delete lines between patterns (keep START and END themselves)
sed '/START/,/END/{/START/!{/END/!d}}' file.txt

# Insert a line before a pattern
sed '/TARGET/i\--- inserted here ---' file.txt

# Append a line after a pattern
sed '/TARGET/a\--- appended here ---' file.txt

# Complex processing with labels and branching
# Join consecutive lines into one (handling backslash-continued lines)
sed -E ':loop; /\\$/{ N; s/\\\n/ /; b loop }' file.txt
```

### 2.5 Practical Examples of File Transformation with sed

```bash
# Convert INI file to JSON-like
# Input: [section]
#         key=value
sed -E '
    /^\[.*\]$/ {
        s/\[(.*)\]/"\1": {/
    }
    /^[^[#].*=/ {
        s/^([^=]+)=(.*)$/  "\1": "\2",/
    }
    /^$/d
' config.ini

# Convert Markdown headings to HTML
sed -E '
    s/^### (.*)$/<h3>\1<\/h3>/
    s/^## (.*)$/<h2>\1<\/h2>/
    s/^# (.*)$/<h1>\1<\/h1>/
    s/\*\*([^*]+)\*\*/<strong>\1<\/strong>/g
    s/\*([^*]+)\*/<em>\1<\/em>/g
' document.md

# Change values in a specific section of a config file
sed -E '/^\[database\]$/,/^\[/ {
    s/^(host\s*=\s*).*/\1db.production.example.com/
    s/^(port\s*=\s*).*/\15432/
}' config.ini

# Mask a CSV column value (mask the third column)
sed -E 's/^([^,]+,[^,]+,)[^,]+(.*)/\1****\2/' data.csv

# Bulk replacement across multiple files (with safe backup)
find src/ -name '*.py' -exec sed -i.bak -E \
    's/from old_module import/from new_module import/g' {} +
# Delete backups after verification
find src/ -name '*.py.bak' -delete

# Mask personal information in log files
sed -E '
    s/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/[EMAIL MASKED]/g
    s/\b[0-9]{3}-[0-9]{4}-[0-9]{4}\b/[PHONE MASKED]/g
    s/\b[0-9]{1,3}(\.[0-9]{1,3}){3}\b/[IP MASKED]/g
' sensitive.log > sanitized.log
```

---

## 3. awk -- Pattern Scanning and Processing

### 3.1 Basic Usage

```bash
# Field extraction (default separator: whitespace)
awk '{print $1}' file.txt          # 1st field
awk '{print $1, $3}' file.txt      # 1st and 3rd fields
awk '{print $NF}' file.txt         # last field
awk '{print NR, $0}' file.txt      # with line number

# Specifying the separator
awk -F',' '{print $1, $2}' data.csv      # CSV
awk -F'\t' '{print $1, $2}' data.tsv     # TSV
awk -F':' '{print $1, $3}' /etc/passwd   # colon-separated

# Pattern matching
awk '/ERROR/ {print}' logfile.txt           # lines containing ERROR
awk '/^2026-02-11/ {print}' logfile.txt     # filter by date
awk '$3 > 100 {print $1, $3}' data.txt     # 3rd field > 100

# Regex matching
awk '$2 ~ /^ERR/ {print}' logfile.txt       # 2nd field starts with ERR
awk '$2 !~ /DEBUG/ {print}' logfile.txt     # 2nd field is not DEBUG
```

### 3.2 awk's Aggregation Features

```bash
# Line count
awk 'END {print NR}' file.txt

# Sum
awk '{sum += $3} END {print "Sum:", sum}' data.txt

# Average
awk '{sum += $3; n++} END {print "Average:", sum/n}' data.txt

# Max/min
awk 'NR==1 || $3 > max {max=$3} END {print "Max:", max}' data.txt

# Group-wise aggregation
awk '{count[$1]++} END {for (k in count) print k, count[k]}' access.log

# Unique count
awk '!seen[$0]++' file.txt         # remove duplicate lines (preserves order)
```

### 3.3 Using Regex in awk

```bash
# Split fields by regex
awk -F'[,;|]' '{print $1, $2}' data.txt

# Extract substring with the match() function
awk '{
    if (match($0, /[0-9]{4}-[0-9]{2}-[0-9]{2}/)) {
        print substr($0, RSTART, RLENGTH)
    }
}' logfile.txt

# Substitution with gsub()
awk '{gsub(/ERROR/, "***ERROR***"); print}' logfile.txt

# Processing multiple patterns
awk '
    /ERROR/  {errors++}
    /WARN/   {warns++}
    /INFO/   {infos++}
    END {
        print "ERROR:", errors+0
        print "WARN:",  warns+0
        print "INFO:",  infos+0
    }
' logfile.txt
```

### 3.4 awk Built-in Variables and Advanced Features

```bash
# Key built-in variables
# NR:   current line number (cumulative across all input)
# NF:   number of fields in the current line
# FNR:  current line number within the current file
# FS:   input field separator
# OFS:  output field separator
# RS:   record separator
# ORS:  output record separator
# FILENAME: name of the file currently being processed

# Control output format using OFS
awk -F',' 'BEGIN {OFS="\t"} {print $1, $2, $3}' data.csv
# Convert CSV -> TSV

# Change record separator (paragraph-level processing)
awk 'BEGIN {RS=""; FS="\n"} {print NR": "$1}' paragraphs.txt
# Process each paragraph separated by blank lines

# Processing multiple files and using FNR
awk 'FNR==1 {print "=== " FILENAME " ==="} {print}' file1.txt file2.txt

# Formatted output with printf
awk '{printf "%-20s %10d %8.2f\n", $1, $2, $3}' data.txt

# Complex aggregation using associative arrays
awk -F',' '{
    category = $1
    amount = $3
    total[category] += amount
    count[category]++
}
END {
    for (cat in total) {
        avg = total[cat] / count[cat]
        printf "%-15s Total: %10.0f  Avg: %8.0f  Count: %d\n", cat, total[cat], avg, count[cat]
    }
}' sales.csv
```

### 3.5 Practical awk Programming Techniques

```bash
# Generate a histogram with awk
awk '{
    len = length($0)
    bucket = int(len / 10) * 10
    hist[bucket]++
}
END {
    for (b in hist) {
        printf "%3d-%3d: ", b, b+9
        for (i = 0; i < hist[b]; i++) printf "#"
        printf " (%d)\n", hist[b]
    }
}' file.txt

# Top-N aggregation in awk (without sorting)
awk '{
    count[$1]++
}
END {
    # Get top 5
    for (i = 1; i <= 5; i++) {
        max_val = 0; max_key = ""
        for (k in count) {
            if (count[k] > max_val) {
                max_val = count[k]; max_key = k
            }
        }
        if (max_key != "") {
            printf "%5d %s\n", max_val, max_key
            delete count[max_key]
        }
    }
}' access.log

# Sliding window (moving average) in awk
awk '{
    window[NR % 5] = $1
    if (NR >= 5) {
        sum = 0
        for (i in window) sum += window[i]
        printf "%d: %.2f\n", NR, sum / 5
    }
}' numbers.txt

# JOIN two files with awk
awk -F',' '
    NR==FNR { lookup[$1] = $2; next }
    { print $0, ($1 in lookup) ? lookup[$1] : "N/A" }
' master.csv detail.csv

# Session analysis of a transaction log with awk
awk '
    /session_start/ {
        match($0, /session_id=([^ ]+)/, arr)
        sid = arr[1]
        start_time[sid] = $1
    }
    /session_end/ {
        match($0, /session_id=([^ ]+)/, arr)
        sid = arr[1]
        if (sid in start_time) {
            duration = $1 - start_time[sid]
            total_duration += duration
            session_count++
            if (duration > max_duration) max_duration = duration
        }
    }
    END {
        printf "Sessions: %d\n", session_count
        printf "Average duration: %.2fs\n", total_duration / session_count
        printf "Max duration: %.2fs\n", max_duration
    }
' transaction.log
```

---

## 4. Log Parsing Pipelines

### 4.1 Parsing Apache Access Logs

```bash
# Apache Combined Log Format:
# 192.168.1.1 - - [11/Feb/2026:10:30:45 +0900] "GET /path HTTP/1.1" 200 1234

# Top IP addresses (by request count)
awk '{print $1}' access.log | sort | uniq -c | sort -rn | head -10

# Counts by HTTP status code
awk '{print $9}' access.log | sort | uniq -c | sort -rn

# Extract URLs of 404 errors
awk '$9 == 404 {print $7}' access.log | sort | uniq -c | sort -rn

# Access counts by hour
awk -F'[\\[:]' '{print $3}' access.log | sort | uniq -c

# Total response size
awk '{sum += $10} END {printf "Total: %.2f MB\n", sum/1024/1024}' access.log

# Slow requests (response time at or above threshold)
awk '$NF > 1000 {print $7, $NF "ms"}' access.log | sort -t' ' -k2 -rn | head -10
```

### 4.2 Parsing Application Logs

```bash
# JSON logs (combined with jq)
# {"timestamp":"2026-02-11T10:30:45","level":"ERROR","message":"..."}

# Extract ERROR-level logs
grep '"level":"ERROR"' app.log | head -20

# Extract fields with jq
grep '"level":"ERROR"' app.log | jq -r '.timestamp + " " + .message'

# Pattern parsing of plain text logs
# [2026-02-11 10:30:45] [ERROR] [module] message

# Error counts by hour
grep -E '\[ERROR\]' app.log | \
    grep -oE '\d{2}:\d{2}' | \
    awk -F: '{print $1":00"}' | \
    sort | uniq -c | sort -rn

# Top 10 error messages
grep -E '\[ERROR\]' app.log | \
    sort | uniq -c | sort -rn | head -10
```

### 4.3 Real-Time Log Monitoring

```bash
# Real-time monitoring with tail -f + grep
tail -f /var/log/syslog | grep --color=auto -E 'ERROR|WARN'

# Watch multiple files at once
tail -f /var/log/*.log | grep --color=auto -E 'ERROR|CRITICAL'

# Real-time aggregation with awk
tail -f access.log | awk '
    {
        status[$9]++
        if (NR % 100 == 0) {
            for (s in status) printf "%s: %d  ", s, status[s]
            print ""
        }
    }
'
```

### 4.4 Advanced Nginx Log Analysis

```bash
# Example Nginx log format:
# $remote_addr - $remote_user [$time_local] "$request" $status $body_bytes_sent
# "$http_referer" "$http_user_agent" $request_time

# Show distribution of request times as a histogram
awk '{
    time = $NF
    if (time < 0.1) bucket = "0-0.1s"
    else if (time < 0.5) bucket = "0.1-0.5s"
    else if (time < 1.0) bucket = "0.5-1.0s"
    else if (time < 5.0) bucket = "1.0-5.0s"
    else bucket = "5.0s+"
    count[bucket]++
}
END {
    order[1] = "0-0.1s"; order[2] = "0.1-0.5s"; order[3] = "0.5-1.0s"
    order[4] = "1.0-5.0s"; order[5] = "5.0s+"
    for (i = 1; i <= 5; i++) {
        b = order[i]
        printf "%-12s %6d ", b, count[b]+0
        for (j = 0; j < count[b] / 100; j++) printf "#"
        print ""
    }
}' access.log

# Access breakdown by user agent
awk -F'"' '{print $6}' access.log | \
    sed -E 's/([^ ]+).*/\1/' | \
    sort | uniq -c | sort -rn | head -10

# Inbound traffic analysis by referer
awk -F'"' '$4 !~ /^-$/ && $4 !~ /^$/ {print $4}' access.log | \
    awk -F'/' '{print $1"//"$3}' | \
    sort | uniq -c | sort -rn | head -10

# Time-series of 5xx errors (per minute)
awk '$9 >= 500 {
    match($0, /\[([0-9]+\/[A-Za-z]+\/[0-9]+:[0-9]+:[0-9]+)/, arr)
    print arr[1]
}' access.log | \
    sort | uniq -c | \
    awk '{printf "%s %5d ", $2, $1; for(i=0;i<$1;i++) printf "*"; print ""}'

# P50/P90/P99 response times for a specific endpoint
awk '$7 == "/api/users" {print $NF}' access.log | \
    sort -n | awk '
    {vals[NR] = $1}
    END {
        n = NR
        printf "Count: %d\n", n
        printf "P50: %.3fs\n", vals[int(n*0.50)]
        printf "P90: %.3fs\n", vals[int(n*0.90)]
        printf "P99: %.3fs\n", vals[int(n*0.99)]
        printf "Max: %.3fs\n", vals[n]
    }'
```

### 4.5 Compound Log Parsing Pipelines

```bash
# Auto-generate a report from access logs
cat > /tmp/log_report.sh << 'SCRIPT'
#!/bin/bash
LOG_FILE=${1:-/var/log/nginx/access.log}
echo "=== Log Analysis Report ==="
echo "Target: $LOG_FILE"
echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

echo "--- Total requests ---"
wc -l < "$LOG_FILE"
echo ""

echo "--- By status code ---"
awk '{print $9}' "$LOG_FILE" | sort | uniq -c | sort -rn
echo ""

echo "--- Top 10 IP addresses ---"
awk '{print $1}' "$LOG_FILE" | sort | uniq -c | sort -rn | head -10
echo ""

echo "--- Top 10 URLs ---"
awk '{print $7}' "$LOG_FILE" | sort | uniq -c | sort -rn | head -10
echo ""

echo "--- Access counts by hour ---"
awk -F'[\\[:]' '{print $3":00"}' "$LOG_FILE" | sort | uniq -c | \
    awk '{printf "%s %5d ", $2, $1; for(i=0;i<$1/50;i++) printf "#"; print ""}'
echo ""

echo "--- Error rate ---"
awk '
    {total++; if ($9 >= 400) errors++}
    END {printf "Total: %d, Errors: %d, Error rate: %.2f%%\n", total, errors+0, (errors+0)*100/total}
' "$LOG_FILE"
SCRIPT
chmod +x /tmp/log_report.sh

# Analyzing systemd journal logs
journalctl -u nginx --since "1 hour ago" --no-pager | \
    grep -E 'error|warn' -i | \
    awk '{print $1, $2, $3}' | \
    sort | uniq -c | sort -rn

# Combine multiple days of logs and analyze
zcat /var/log/nginx/access.log.*.gz | \
    cat - /var/log/nginx/access.log | \
    awk '$9 == 500 {print $7}' | \
    sort | uniq -c | sort -rn | head -20
```

### 4.6 Security Log Analysis

```bash
# Detect SSH brute-force attempts
grep 'Failed password' /var/log/auth.log | \
    awk '{print $(NF-3)}' | \
    sort | uniq -c | sort -rn | head -20

# Signs of malicious access (SQL injection attempts)
grep -iE "(union\+select|or\+1=1|drop\+table|;--)" access.log | \
    awk '{print $1, $7}' | sort | uniq -c | sort -rn

# Analyzing blocked requests from WAF logs
grep 'BLOCKED' waf.log | \
    awk -F'|' '{print $3}' | \    # attack category
    sort | uniq -c | sort -rn

# Anomaly detection of login attempts (many attempts from a single IP in a short time)
awk '/login_attempt/ {
    # Extract timestamp and IP
    match($0, /ip=([0-9.]+)/, ip_arr)
    match($0, /\[([0-9:]+)\]/, time_arr)
    ip = ip_arr[1]
    attempts[ip]++
}
END {
    for (ip in attempts) {
        if (attempts[ip] > 10) {
            printf "Suspicious IP: %-15s  Attempts: %d\n", ip, attempts[ip]
        }
    }
}' auth.log

# File access auditing (auditd logs)
grep 'type=SYSCALL' /var/log/audit/audit.log | \
    grep -E 'syscall=(2|257)' | \
    awk -F' ' '{
        for (i=1; i<=NF; i++) {
            if ($i ~ /^comm=/) comm = $i
            if ($i ~ /^uid=/) uid = $i
        }
        print uid, comm
    }' | sort | uniq -c | sort -rn | head -20
```

---

## 5. CSV Processing

### 5.1 Basic CSV Processing

```bash
# Simple CSV (without quotes)
# name,age,city
# Alice,30,Tokyo
# Bob,25,Osaka

# Extract specific columns
awk -F',' '{print $1, $3}' data.csv

# Conditional filtering
awk -F',' '$2 > 25 {print}' data.csv

# Reorder columns
awk -F',' '{print $3","$1","$2}' data.csv

# Process with header
awk -F',' 'NR==1 {print; next} $2 > 25 {print}' data.csv
```

### 5.2 Issues with Quoted CSV

```
Problems with CSV:
+------------------------------------------------+
| When a field contains a comma:                 |
|   "Tokyo, Japan",30,engineer                   |
|                                                |
| When a field contains quotes:                  |
|   "He said ""hello""",30,engineer              |
|                                                |
| When a field contains newlines:                |
|   "Line 1                                      |
|   Line 2",30,engineer                          |
|                                                |
| -> Cannot be processed correctly with regex    |
|    alone!                                      |
| -> Use a dedicated parser                      |
+------------------------------------------------+
```

```python
# Correct CSV processing in Python
import csv
import re

# BAD: split CSV with regex
def parse_csv_bad(line):
    return line.split(',')  # breaks on commas inside quotes

# GOOD: use the csv module
with open('data.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

# When regex is useful: processing each field of a CSV
with open('data.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        # Apply regex per field
        for field in row:
            if re.match(r'\d{4}-\d{2}-\d{2}', field):
                print(f"  Date field: {field}")
```

### 5.3 Advanced CSV Processing with csvkit and miller

```bash
# === Using csvkit ===

# csvlook: display CSV in a readable table format
csvlook data.csv

# csvcut: column extraction (can specify by column name)
csvcut -c name,age data.csv
csvcut -c 1,3 data.csv              # indexes also work

# csvgrep: grep over CSV (handles quotes correctly)
csvgrep -c status -m 'active' data.csv
csvgrep -c age -r '^[3-4][0-9]$' data.csv  # regex is supported too

# csvsort: sort a CSV
csvsort -c age -r data.csv           # sort by age column descending

# csvstat: display statistics
csvstat data.csv

# csvjoin: JOIN two CSVs
csvjoin -c user_id users.csv orders.csv

# csvsql: run SQL queries over CSV
csvsql --query "SELECT name, AVG(score) as avg_score \
    FROM data GROUP BY name HAVING avg_score > 80" data.csv

# csvformat: format conversion
csvformat -T data.csv                # CSV -> TSV
csvformat -D '|' data.csv           # CSV -> pipe-separated

# === Using miller (mlr) ===

# Basic filtering
mlr --csv filter '$age > 25' data.csv

# Column selection
mlr --csv cut -f name,city data.csv

# Renaming columns
mlr --csv rename name,full_name data.csv

# Group-wise aggregation
mlr --csv stats1 -a mean,count -f age -g city data.csv

# Sorting
mlr --csv sort-by -nr age data.csv

# Format conversion
mlr --icsv --ojson cat data.csv          # CSV -> JSON
mlr --icsv --opprint cat data.csv        # CSV -> pretty table
mlr --ijson --ocsv cat data.json         # JSON -> CSV

# Complex transformation pipeline
mlr --csv \
    filter '$status == "active"' \
    then sort-by -nr revenue \
    then head -n 10 \
    then put '$revenue_formatted = format_values($revenue, "%,.0f")' \
    data.csv

# Time-series data processing
mlr --csv \
    put '$date = strftime(strptime($timestamp, "%Y-%m-%d %H:%M:%S"), "%Y-%m-%d")' \
    then group-by date \
    then stats1 -a sum,mean -f amount \
    transactions.csv
```

### 5.4 Processing TSV/Fixed-Width/Other Formats

```bash
# TSV processing
awk -F'\t' '{print $1, $3}' data.tsv
awk -F'\t' 'BEGIN{OFS=","} {print $1,$2,$3}' data.tsv > data.csv

# Fixed-width record processing
# Example: name (20 chars), age (3 chars), city (15 chars)
awk '{
    name = substr($0, 1, 20)
    age  = substr($0, 21, 3)
    city = substr($0, 24, 15)
    printf "%s,%s,%s\n", name, age+0, city
}' fixed_width.dat

# Processing LTSV (Labeled Tab-Separated Values)
# host:127.0.0.1\tident:-\tuser:-\ttime:[10/Oct/2000:13:55:36 -0700]
awk -F'\t' '{
    for (i=1; i<=NF; i++) {
        split($i, kv, ":")
        fields[kv[1]] = substr($i, length(kv[1])+2)
    }
    print fields["host"], fields["status"], fields["size"]
}' access.ltsv

# Convert Apache logs to CSV
sed -E 's/^([^ ]+) [^ ]+ [^ ]+ \[([^\]]+)\] "([^"]+)" ([0-9]+) ([0-9]+|-)/\1,\2,\3,\4,\5/' access.log

# Convert JSON Lines to CSV
jq -r '[.timestamp, .level, .message] | @csv' app.jsonl > app.csv
```

---

## 6. ASCII Diagrams

### 6.1 Choosing Between grep/sed/awk

```
Tool selection by use case:

Text search (line filtering)
  -> grep
  grep 'ERROR' log.txt

Text replacement (line-level transformation)
  -> sed
  sed 's/old/new/g' file.txt

Field processing (column extraction/aggregation)
  -> awk
  awk -F',' '{print $1, $3}' data.csv

+----------+------------+----------+----------+
| Operation| grep       | sed      | awk      |
+----------+------------+----------+----------+
| Search   | ***        | *        | **       |
| Replace  | -          | ***      | **       |
| Extract  | ** (-o)    | *        | ***      |
| Aggregate| -          | -        | ***      |
| Filter   | ***        | **       | ***      |
| Transform| -          | ***      | ***      |
+----------+------------+----------+----------+
```

### 6.2 Pipeline Structure

```
A typical log analysis pipeline:

access.log
    |
    v
+--------+   lines containing  +--------+
|  grep  | ------ ERROR -----> |  awk   |
| 'ERROR'|                     | '{$1}' |
+--------+                     +---+----+
                                   | IP address column
                                   v
                              +--------+
                              |  sort  |
                              +---+----+
                                  | sorted
                                  v
                              +--------+
                              | uniq -c|
                              +---+----+
                                  | with counts
                                  v
                              +----------+
                              |sort -rn  |
                              | head -10 |
                              +----------+
                                  |
                                  v
                              Top 10 IPs
```

### 6.3 Syntax Differences Between BRE / ERE / PCRE

```
Different ways to write the same pattern:

"3 or more digits"
  BRE:  [0-9]\{3,\}        (braces require escaping)
  ERE:  [0-9]{3,}          (as is)
  PCRE: \d{3,}             (\d is available)

"cat or dog"
  BRE:  cat\|dog           (pipe requires escaping)
  ERE:  cat|dog            (as is)
  PCRE: cat|dog            (as is)

"group + backreference"
  BRE:  \(hello\) \1       (parentheses require escaping)
  ERE:  (hello) \1         (parens as is, reference is \1)
  PCRE: (hello) \1         (parens as is, reference is \1)

Tool support:
  grep:       BRE (default), ERE (-E), PCRE (-P)
  sed:        BRE (default), ERE (-E)
  awk:        ERE (default)
```

### 6.4 Big-Picture Data Flow of Text Processing

```
The overall architecture of a text processing pipeline:

[Input source]                    [Processing pipeline]                [Output]

 Log file -----+              +--> grep (filter) --+              +--> file
               |              |                    |              |
 stdin -------+  -> select ---+--> sed  (transform)+--> format ---+--> stdout
               |              |                    |              |
 Network -----+              +--> awk  (aggregate)-+              +--> pipe
               |
 Compressed --+
   (zcat/zgrep)

Example data transformation flows:

 Raw JSON log                          Structured data
 ------------                          ----------------
 {"ts":"...",     extract              CSV/TSV
  "level":"ERR",  fields with    -->   aggregate/   --> report
  "msg":"..."}    jq                   analyze with mlr

 Apache log                            Statistics
 -----------                           -----------
 192.168.1.1 -    extract              sort |
 [date] "GET      fields with    -->   uniq -c   --> ranking
 /path" 200       awk                  sort -rn
```

---

## 7. Comparison Tables

### 7.1 Tool Characteristics

| Trait | grep | sed | awk |
|-------|------|-----|-----|
| Primary use | Pattern search | Stream editing | Field processing |
| Regex flavor | BRE/ERE/PCRE | BRE/ERE | ERE |
| Line operations | Filter | Transform | Transform + aggregate |
| Fields | None | Limited | Powerful |
| Computation | None | None | Yes |
| Variables | None | Hold space | Arrays/variables |
| Speed | Fastest | Fast | Somewhat slower |
| Learning curve | Low | Medium | High |

### 7.2 Modern Alternatives

| Traditional tool | Modern alternative | Features |
|------------------|--------------------|----------|
| grep | ripgrep (rg) | Fast, .gitignore aware, Unicode support |
| sed | sd | Simple syntax, PCRE support |
| awk | miller (mlr) | Native CSV/JSON/TSV support |
| find + grep | fd + rg | Fast, intuitive UI |
| cat + grep | bat | Syntax highlighting |
| -- | jq | Specialized JSON processing |
| -- | xsv/qsv | Specialized CSV processing |

### 7.3 Recommended Tool Choice by Use Case

| Use case | Best tool | Reason |
|----------|-----------|--------|
| Source code search | ripgrep (rg) | Honors .gitignore, recursive by default |
| Real-time log monitoring | tail -f + grep | Simple, lightweight |
| Log aggregation/statistics | awk | Built-in arithmetic |
| Bulk config file changes | sed -i | In-place editing |
| CSV aggregation | miller (mlr) | Native CSV support |
| JSON log analysis | jq | Native JSON support |
| Massive parallel data processing | GNU parallel + grep | Faster via parallelism |
| Structured log conversion | awk + jq | Field extraction + structured transformation |
| Searching binary files | grep -a / strings | Binary-aware modes |
| Multibyte string processing | grep -P / rg | Unicode support |

---

## 8. Anti-Patterns

### 8.1 Anti-Pattern: Processing CSV with Regex Alone

```bash
# BAD: split on commas (breaks on commas inside quotes)
awk -F',' '{print $2}' data.csv
# Input: "Tokyo, Japan",30 -> $2 = " Japan" (broken)

# GOOD: use a dedicated tool
# csvkit
csvcut -c 2 data.csv

# Miller
mlr --csv cut -f name data.csv

# Python
python3 -c "
import csv, sys
for row in csv.reader(sys.stdin):
    print(row[1])
" < data.csv
```

### 8.2 Anti-Pattern: Inefficient Pipelines on Huge Files

```bash
# BAD: inefficient (scans the file multiple times)
ERROR_COUNT=$(grep -c 'ERROR' huge.log)
WARN_COUNT=$(grep -c 'WARN' huge.log)
INFO_COUNT=$(grep -c 'INFO' huge.log)
# -> reads the file 3 times

# GOOD: count everything in a single pass
awk '
    /ERROR/ {e++}
    /WARN/  {w++}
    /INFO/  {i++}
    END {
        print "ERROR:", e+0
        print "WARN:",  w+0
        print "INFO:",  i+0
    }
' huge.log
# -> reads the file only once
```

### 8.3 Anti-Pattern: Useless Use of Cat

```bash
# BAD: useless cat (UUOC)
cat file.txt | grep 'ERROR'
cat file.txt | awk '{print $1}'
cat file.txt | sed 's/old/new/g'

# GOOD: pass the file directly as an argument
grep 'ERROR' file.txt
awk '{print $1}' file.txt
sed 's/old/new/g' file.txt

# GOOD: use redirection
grep 'ERROR' < file.txt
```

### 8.4 Anti-Pattern: Writing Complex Logic in sed

```bash
# BAD: heavy use of branches/loops in sed (hard to maintain)
sed -E '
    :start
    /\\$/ {
        N
        s/\\\n/ /
        b start
    }
    /^#/d
    /^$/d
    s/([^=]+)=([^;]+);?/\1 = "\2"\n/g
' complex_config.txt

# GOOD: write clearly with Python or awk
awk '
    /\\$/ {
        # join backslash-continued lines
        line = line substr($0, 1, length($0)-1)
        next
    }
    {
        line = line $0
        # exclude comments and empty lines
        if (line !~ /^#/ && line != "") {
            print line
        }
        line = ""
    }
' complex_config.txt
```

### 8.5 Anti-Pattern: Overconfidence in Regex-Based Validation

```bash
# BAD: try to do "complete" email validation with regex
# An RFC 5322 compliant email regex is thousands of characters long
grep -E '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$' emails.txt
# -> this is only a rough check (does not handle IDN domains, etc.)

# GOOD: rough check + validation with a dedicated library
# narrow with grep -> strictly validate with a dedicated library
grep -E '@.*\.' emails.txt | python3 -c "
import sys
from email_validator import validate_email, EmailNotValidError
for line in sys.stdin:
    email = line.strip()
    try:
        validate_email(email)
        print(f'VALID: {email}')
    except EmailNotValidError as e:
        print(f'INVALID: {email} ({e})')
"
```

### 8.6 Anti-Pattern: Not Inspecting Intermediate Pipeline Results

```bash
# BAD: write a long pipeline at once without debugging
awk -F',' '{print $3}' data.csv | sed 's/"//g' | sort | uniq -c | sort -rn | head -5

# GOOD: inspect intermediate results with the tee command
awk -F',' '{print $3}' data.csv | \
    tee /dev/stderr | \           # send intermediate result to stderr
    sed 's/"//g' | \
    tee /tmp/debug_step2.txt | \  # also save to a file
    sort | uniq -c | sort -rn | head -5

# GOOD: build the pipeline incrementally
# Step 1: verify awk output
awk -F',' '{print $3}' data.csv | head -5
# Step 2: add sed
awk -F',' '{print $3}' data.csv | sed 's/"//g' | head -5
# Step 3: add sort and aggregation
awk -F',' '{print $3}' data.csv | sed 's/"//g' | sort | uniq -c | sort -rn | head -5
```


---

## Practical Exercises

### Exercise 1: Basic Implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Validate input data
- Implement appropriate error handling
- Also write test code

```python
# Exercise 1: template for the basic implementation
class Exercise1:
    """Exercise on basic implementation patterns"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Validate the input value"""
        if value is None:
            raise ValueError("Input value is None")
        return True

    def process(self, value):
        """Main logic for data processing"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """Get processing results"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# Tests
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "An exception should be raised"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Advanced Patterns

Extend the basic implementation by adding the following features.

```python
# Exercise 2: advanced patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """Exercise on advanced patterns"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """Add an item (with size limit)"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """Search by key"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """Remove by key"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """Statistics"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# Tests
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # size limit
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("All advanced tests passed!")

test_advanced()
```

### Exercise 3: Performance Optimization

Improve the performance of the following code.

```python
# Exercise 3: performance optimization
import time
from functools import lru_cache

# Before optimization (O(n^2))
def slow_search(data: list, target: int) -> int:
    """Inefficient search"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# After optimization (O(n))
def fast_search(data: list, target: int) -> tuple:
    """Efficient search using a hash map"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# Benchmark
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

    print(f"Inefficient: {slow_time:.4f}s")
    print(f"Efficient:   {fast_time:.6f}s")
    print(f"Speedup:     {slow_time/fast_time:.0f}x")

benchmark()
```

**Key points:**
- Be aware of algorithmic complexity
- Choose the right data structures
- Measure the effect with benchmarks
---

## 9. FAQ

### Q1: What is the difference between grep -E and egrep?

**A**: They are functionally identical. `egrep` is an alias for `grep -E`, but it has been deprecated in POSIX.1-2008. Use `grep -E`. Likewise, prefer `grep -F` over `fgrep`.

### Q2: Does sed's `-i` option behave differently between GNU and macOS (BSD)?

**A**: **Yes, it differs.** GNU sed works with just `-i`, but BSD sed (macOS) requires an empty extension argument such as `-i ''`:

```bash
# GNU (Linux):
sed -i 's/old/new/g' file.txt

# BSD (macOS):
sed -i '' 's/old/new/g' file.txt

# Works on both:
sed -i.bak 's/old/new/g' file.txt && rm file.txt.bak
```

### Q3: Can awk use regex capture group backreferences?

**A**: POSIX awk does not support backreferences to capture groups. Use the `match()` function and `substr()` instead:

```bash
# POSIX awk: match + substr
awk '{
    if (match($0, /([0-9]+)-([0-9]+)/, arr)) {
        print arr[1], arr[2]
    }
}' file.txt
# Note: the third (array) argument is GNU awk (gawk) only

# gawk: array captures
gawk 'match($0, /([0-9]+)-([0-9]+)/, a) {print a[1], a[2]}' file.txt
```

### Q4: Should I use ripgrep (rg) or grep?

**A**: For new projects, **ripgrep is recommended**. Reasons:

- Honors `.gitignore` by default
- Recursive search by default
- Full Unicode support
- Several times faster on large repositories
- Supports PCRE2 (lookahead, etc.)

```bash
# Basic usage of ripgrep
rg 'ERROR' .                      # recursive search (default)
rg -i 'error' --type py           # only Python files
rg 'pattern' -g '*.log'           # specify files via glob
rg -P '(?<=\$)\d+' .              # PCRE2 (lookahead/lookbehind)
```

### Q5: Tips for processing very large log files (tens of GB)?

**A**: Combine the following approaches:

```bash
# 1. Use LC_ALL=C to skip locale processing (2-3x speedup)
LC_ALL=C grep 'ERROR' huge.log

# 2. Early termination with grep -m
LC_ALL=C grep -m 1000 'ERROR' huge.log    # stop after the first 1000 hits

# 3. Parallel processing with GNU parallel
parallel --pipepart -a huge.log --block 100M grep 'ERROR'

# 4. split + parallel for chunked processing
split -l 1000000 huge.log /tmp/chunk_
ls /tmp/chunk_* | parallel "grep -c 'ERROR' {}" | awk '{sum+=$1} END{print sum}'

# 5. Direct processing of compressed files
zgrep 'ERROR' huge.log.gz           # search inside gzip
bzgrep 'ERROR' huge.log.bz2        # search inside bzip2
xzgrep 'ERROR' huge.log.xz         # search inside xz

# 6. Cases where awk is more efficient than grep
# When handling multiple conditions in a single pass, awk wins
awk '/ERROR/{e++} /WARN/{w++} END{print e+0, w+0}' huge.log
```

### Q6: How do I correctly process multibyte characters (e.g., Japanese) with grep/sed/awk?

**A**: Pay attention to locale settings and character encoding:

```bash
# Check locale
locale

# Search Japanese in a UTF-8 environment
grep '東京' data.txt

# Be careful with character classes (locale-dependent)
grep '[a-zA-Z]' data.txt        # ASCII only

# Replace Japanese in sed
sed 's/東京都/東京/g' addresses.txt

# Process Japanese in awk
awk '/東京/ {count++} END {print count+0}' data.txt

# Combine with character encoding conversion
# Convert Shift_JIS -> UTF-8 before processing
iconv -f SHIFT_JIS -t UTF-8 sjis_file.txt | grep 'パターン'

# Encoding conversion with nkf
nkf -w sjis_file.txt | grep 'パターン'
```

### Q7: When awk scripts get long, should I move them to a file?

**A**: For awk scripts longer than 10 lines, **moving them to a file is recommended**:

```bash
# awk script file: analyze.awk
cat > analyze.awk << 'AWK'
BEGIN {
    FS = ","
    OFS = "\t"
    print "Category", "Count", "Total", "Average"
}
NR > 1 {
    category = $1
    amount = $3
    count[category]++
    total[category] += amount
}
END {
    for (cat in count) {
        avg = total[cat] / count[cat]
        printf "%-15s\t%d\t%.0f\t%.0f\n", cat, count[cat], total[cat], avg
    }
}
AWK

# Run
awk -f analyze.awk data.csv

# Benefits:
# - Easier to put under version control
# - Editor syntax highlighting works
# - Testable
# - Easy to reuse
```

### Q8: Should I use sed or perl -pe?

**A**: For basic substitutions, prefer sed; for complex patterns, prefer perl -pe:

```bash
# Examples where sed is sufficient
sed 's/old/new/g' file.txt
sed '/pattern/d' file.txt

# Examples where perl -pe is more appropriate
# Zero-width assertions (lookahead/lookbehind)
perl -pe 's/(?<=\$)\d+/XXX/g' file.txt

# Non-greedy match
perl -pe 's/<.*?>/[TAG]/g' file.html

# Multiline match
perl -0pe 's/start.*?end/REPLACED/gs' file.txt

# Substitution with computed values
perl -pe 's/(\d+)/sprintf("%05d", $1)/ge' file.txt
```

---

## 10. Real-World Scenarios

### 10.1 Log Analysis for Incident Investigation

```bash
# Scenario: 500 errors are spiking on a web app; investigate the cause

# Step 1: check error frequency over time
awk '$9 == 500 {
    match($0, /\[([0-9]+\/[A-Za-z]+\/[0-9]+:[0-9]+:[0-9]+)/, arr)
    print arr[1]
}' access.log | sort | uniq -c | tail -20

# Step 2: identify which endpoints are erroring
awk '$9 == 500 {print $7}' access.log | sort | uniq -c | sort -rn | head -10

# Step 3: check IP addresses of error requests (is it an attack from a specific IP?)
awk '$9 == 500 {print $1}' access.log | sort | uniq -c | sort -rn | head -10

# Step 4: cross-reference with the application log
# Use timestamps from access.log to find corresponding errors in app.log
awk '$9 == 500 {
    match($4, /([0-9]+:[0-9]+:[0-9]+)/, arr)
    print arr[1]
}' access.log | head -5 | while read ts; do
    grep "$ts" app.log | grep -i 'error\|exception\|traceback'
done

# Step 5: classify root causes by pattern
grep -A 5 'ERROR' app.log | \
    grep -E 'Exception|Error' | \
    sed -E 's/^.*: //' | \
    sort | uniq -c | sort -rn | head -10
```

### 10.2 Pre-Deployment Code Quality Check

```bash
# List TODO/FIXME/HACK
rg -n 'TODO|FIXME|HACK|XXX' --type py src/ | \
    awk -F: '{printf "%-40s %s: %s\n", $1, $2, $3}'

# Detect debug print statements
rg -n '^\s*(print\(|console\.log|System\.out\.print)' src/

# Detect hardcoded credentials
rg -in '(password|secret|api_key|token)\s*=\s*["\x27][^"\x27]+["\x27]' \
    --type py --type js --type ts src/

# Detect unused imports (Python)
for f in $(find src/ -name '*.py'); do
    awk '
        /^import / { modules[$2] = NR }
        /^from .* import / {
            split($0, a, "import ")
            split(a[2], b, ",")
            for (i in b) {
                modules[b[i]] = NR
            }
        }
        !/^import |^from / {
            for (m in modules) {
                if (index($0, m) > 0) delete modules[m]
            }
        }
        END {
            for (m in modules) print FILENAME":"modules[m]": unused import: "m
        }
    ' "$f"
done

# Detect overly long lines
awk 'length > 120 {printf "%s:%d: line length %d chars\n", FILENAME, NR, length}' src/*.py
```

### 10.3 Pre-Processing for Data Migration

```bash
# Scenario: migrate CSV data from a legacy system to a new system

# Step 1: get an overview of the data
head -1 old_system.csv                    # check the header
wc -l old_system.csv                      # number of rows
awk -F',' '{print NF}' old_system.csv | sort -u  # check field counts

# Step 2: data quality checks
# Detect empty fields
awk -F',' '{
    for (i=1; i<=NF; i++) {
        if ($i == "" || $i == "NULL" || $i == "null") {
            empty[i]++
        }
    }
    total++
}
END {
    for (i in empty) {
        printf "Column %d: %d empty (%.1f%%)\n", i, empty[i], empty[i]*100/total
    }
}' old_system.csv

# Step 3: unify date formats
sed -E '
    # MM/DD/YYYY -> YYYY-MM-DD
    s,([0-9]{2})/([0-9]{2})/([0-9]{4}),\3-\1-\2,g
    # DD-Mon-YYYY -> YYYY-MM-DD (simplified)
    s/Jan/01/g; s/Feb/02/g; s/Mar/03/g; s/Apr/04/g
    s/May/05/g; s/Jun/06/g; s/Jul/07/g; s/Aug/08/g
    s/Sep/09/g; s/Oct/10/g; s/Nov/11/g; s/Dec/12/g
' old_system.csv > normalized_dates.csv

# Step 4: normalize phone numbers
sed -E '
    s/\+81-?/0/g           # international -> domestic
    s/[()-]//g              # remove punctuation
    s/([0-9]{3})([0-9]{4})([0-9]{4})/\1-\2-\3/g  # insert hyphens
' normalized_dates.csv > normalized_phones.csv

# Step 5: detect duplicate records
awk -F',' 'NR>1 {
    key = $1","$2","$3    # treat name+email+phone as the dedup key
    if (key in seen) {
        print "Duplicate: line "seen[key]" and line "NR": "$0
    } else {
        seen[key] = NR
    }
}' old_system.csv
```

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining hands-on experience is most important. Beyond theory, your understanding deepens when you actually write code and verify how it behaves.

### Q2: What mistakes do beginners commonly make?

Skipping the basics and jumping into advanced material. We recommend solidly grasping the foundational concepts described in this guide before moving on to the next step.

### Q3: How is this used in real-world work?

Knowledge of this topic is frequently used in everyday development. It is especially important during code reviews and architectural design.

---

## Summary

| Item | Description |
|------|-------------|
| grep | The fundamental tool for pattern search; leverage -E (ERE) and -P (PCRE) |
| sed | Stream-based substitution/deletion/transformation; use -E for ERE |
| awk | Field processing and aggregation; pattern match with regex |
| Pipelines | Combining `grep \| awk \| sort \| uniq -c` is powerful |
| CSV | Regex alone is insufficient; combine with dedicated tools (csvkit, miller) |
| Log analysis | Aggregate multiple metrics in a single pass for efficiency |
| Modern alternatives | ripgrep, sd, miller speed up the traditional tools |
| Performance | Handle large data with LC_ALL=C, the -F option, and parallel |
| Safety | Always edit with backups and build pipelines incrementally |

## What to Read Next

- [03-regex-alternatives.md](./03-regex-alternatives.md) -- Alternatives to regular expressions
- [../01-advanced/03-performance.md](../01-advanced/03-performance.md) -- Performance optimization

## References

1. **Dale Dougherty & Arnold Robbins** "sed & awk, 2nd Edition" O'Reilly, 1997 -- the classic on sed/awk
2. **GNU Grep Manual** https://www.gnu.org/software/grep/manual/ -- official reference for GNU grep
3. **The AWK Programming Language** Aho, Kernighan, Weinberger, 2024 (2nd Edition) -- revised edition by the original authors of awk
4. **ripgrep** https://github.com/BurntSushi/ripgrep -- a modern grep alternative
5. **Miller (mlr)** https://miller.readthedocs.io/ -- Swiss Army knife for CSV/JSON/TSV processing
6. **csvkit** https://csvkit.readthedocs.io/ -- command-line toolkit for CSV processing
7. **jq Manual** https://stedolan.github.io/jq/manual/ -- the standard tool for JSON processing
8. **GNU Parallel** https://www.gnu.org/software/parallel/ -- a tool for running commands in parallel
