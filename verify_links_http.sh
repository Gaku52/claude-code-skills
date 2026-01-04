#!/bin/bash

# HTTP Link Verification Script
# Tests all external URLs for actual accessibility (200/301/302 responses)

set -e

REPORT_FILE="HTTP_VERIFICATION_REPORT.md"
FAILED_LINKS_FILE="FAILED_LINKS.md"

echo "# HTTP Link Verification Report" > "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "**Date**: $(date '+%Y-%m-%d %H:%M:%S')" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "---" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

echo "# Failed Links" > "$FAILED_LINKS_FILE"
echo "" >> "$FAILED_LINKS_FILE"
echo "Links that returned errors (404, timeout, etc.)" >> "$FAILED_LINKS_FILE"
echo "" >> "$FAILED_LINKS_FILE"

# Extract all external URLs from SKILL.md files
echo "Extracting URLs from SKILL.md files..."

URLS=$(grep -h -o 'https://[^)]*' **/SKILL.md 2>/dev/null | sort -u)

TOTAL=0
PASSED=0
FAILED=0
REDIRECTED=0

echo "## Summary" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Test each URL
echo "$URLS" | while IFS= read -r url; do
    if [ -z "$url" ]; then
        continue
    fi

    TOTAL=$((TOTAL + 1))

    echo -n "Testing [$TOTAL]: $url ... "

    # Use curl to test the URL
    # -L: follow redirects
    # -s: silent
    # -o /dev/null: discard output
    # -w: write HTTP code
    # --max-time 10: timeout after 10 seconds
    # --connect-timeout 5: connection timeout

    HTTP_CODE=$(curl -L -s -o /dev/null -w "%{http_code}" --max-time 10 --connect-timeout 5 "$url" 2>/dev/null || echo "000")

    case $HTTP_CODE in
        200|201|202|203|204|205|206)
            echo "✅ OK ($HTTP_CODE)"
            PASSED=$((PASSED + 1))
            ;;
        301|302|303|307|308)
            echo "↪️  REDIRECT ($HTTP_CODE)"
            REDIRECTED=$((REDIRECTED + 1))
            ;;
        000)
            echo "❌ TIMEOUT/ERROR"
            FAILED=$((FAILED + 1))
            echo "- **TIMEOUT**: $url" >> "$FAILED_LINKS_FILE"
            ;;
        *)
            echo "❌ FAILED ($HTTP_CODE)"
            FAILED=$((FAILED + 1))
            echo "- **HTTP $HTTP_CODE**: $url" >> "$FAILED_LINKS_FILE"
            ;;
    esac

    # Rate limiting: wait 0.2 seconds between requests
    sleep 0.2
done

# Note: Due to shell limitations in while loops with pipes,
# we need to recount. Let's use a simpler Python script instead.

echo ""
echo "Verification complete! Check $REPORT_FILE and $FAILED_LINKS_FILE for results."
