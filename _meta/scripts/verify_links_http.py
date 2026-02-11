#!/usr/bin/env python3
"""
HTTP Link Verification Script
Tests all external URLs from SKILL.md files for actual accessibility
"""

import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def extract_urls_from_file(filepath):
    """Extract all HTTPS URLs from a markdown file"""
    urls = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # Find all https:// URLs (clean extraction without trailing punctuation)
            pattern = r'https://[^\s\)>\]"\'`,]+'
            raw_urls = re.findall(pattern, content)
            # Further clean: remove trailing punctuation that might have been captured
            urls = [url.rstrip('\'",`') for url in raw_urls]
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return urls

def test_url(url, timeout=10):
    """Test a URL and return status code"""
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.geturl()
    except urllib.error.HTTPError as e:
        return e.code, url
    except urllib.error.URLError as e:
        return 'TIMEOUT', url
    except Exception as e:
        return 'ERROR', url

def main():
    print("=" * 60)
    print("HTTP Link Verification - Testing All External URLs")
    print("=" * 60)
    print()

    # Find all SKILL.md files
    skill_files = list(Path('.').rglob('**/SKILL.md'))
    print(f"Found {len(skill_files)} SKILL.md files")
    print()

    # Extract all URLs
    all_urls = set()
    url_sources = defaultdict(list)

    for filepath in skill_files:
        urls = extract_urls_from_file(filepath)
        for url in urls:
            all_urls.add(url)
            url_sources[url].append(str(filepath))

    print(f"Found {len(all_urls)} unique URLs to test")
    print()

    # Test each URL
    results = {
        'success': [],
        'redirect': [],
        'client_error': [],
        'server_error': [],
        'timeout': [],
        'error': []
    }

    total = len(all_urls)
    for i, url in enumerate(sorted(all_urls), 1):
        print(f"[{i}/{total}] Testing: {url[:70]}...", end=' ', flush=True)

        status, final_url = test_url(url)

        if isinstance(status, int):
            if 200 <= status < 300:
                print(f"✅ {status}")
                results['success'].append((url, status, final_url))
            elif 300 <= status < 400:
                print(f"↪️  {status} → {final_url[:50]}...")
                results['redirect'].append((url, status, final_url))
            elif 400 <= status < 500:
                print(f"❌ {status}")
                results['client_error'].append((url, status, final_url))
            elif 500 <= status < 600:
                print(f"❌ {status}")
                results['server_error'].append((url, status, final_url))
        elif status == 'TIMEOUT':
            print(f"⏱️  TIMEOUT")
            results['timeout'].append((url, 'TIMEOUT', ''))
        else:
            print(f"❌ ERROR")
            results['error'].append((url, 'ERROR', ''))

        # Rate limiting
        time.sleep(0.3)

    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print()

    print(f"Total URLs tested: {total}")
    print(f"✅ Success (2xx):        {len(results['success'])}")
    print(f"↪️  Redirects (3xx):      {len(results['redirect'])}")
    print(f"❌ Client Errors (4xx):  {len(results['client_error'])}")
    print(f"❌ Server Errors (5xx):  {len(results['server_error'])}")
    print(f"⏱️  Timeouts:            {len(results['timeout'])}")
    print(f"❌ Other Errors:         {len(results['error'])}")
    print()

    success_rate = (len(results['success']) + len(results['redirect'])) / total * 100
    print(f"Success Rate: {success_rate:.1f}%")
    print()

    # Write detailed report
    report_file = 'HTTP_VERIFICATION_REPORT.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# HTTP Link Verification Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("## Summary\n\n")
        f.write(f"- **Total URLs tested**: {total}\n")
        f.write(f"- **✅ Success (2xx)**: {len(results['success'])}\n")
        f.write(f"- **↪️ Redirects (3xx)**: {len(results['redirect'])}\n")
        f.write(f"- **❌ Client Errors (4xx)**: {len(results['client_error'])}\n")
        f.write(f"- **❌ Server Errors (5xx)**: {len(results['server_error'])}\n")
        f.write(f"- **⏱️ Timeouts**: {len(results['timeout'])}\n")
        f.write(f"- **❌ Other Errors**: {len(results['error'])}\n")
        f.write(f"- **Success Rate**: {success_rate:.1f}%\n\n")

        if results['client_error'] or results['server_error'] or results['timeout'] or results['error']:
            f.write("## ❌ FAILED LINKS\n\n")

            if results['client_error']:
                f.write("### Client Errors (4xx)\n\n")
                for url, status, _ in results['client_error']:
                    f.write(f"- **HTTP {status}**: {url}\n")
                    f.write(f"  - Used in: {', '.join(url_sources[url])}\n")
                f.write("\n")

            if results['server_error']:
                f.write("### Server Errors (5xx)\n\n")
                for url, status, _ in results['server_error']:
                    f.write(f"- **HTTP {status}**: {url}\n")
                    f.write(f"  - Used in: {', '.join(url_sources[url])}\n")
                f.write("\n")

            if results['timeout']:
                f.write("### Timeouts\n\n")
                for url, _, _ in results['timeout']:
                    f.write(f"- {url}\n")
                    f.write(f"  - Used in: {', '.join(url_sources[url])}\n")
                f.write("\n")

            if results['error']:
                f.write("### Other Errors\n\n")
                for url, _, _ in results['error']:
                    f.write(f"- {url}\n")
                    f.write(f"  - Used in: {', '.join(url_sources[url])}\n")
                f.write("\n")

        if results['redirect']:
            f.write("## ↪️ Redirects (Working but redirected)\n\n")
            for url, status, final_url in results['redirect']:
                f.write(f"- **HTTP {status}**: {url}\n")
                f.write(f"  - Redirects to: {final_url}\n")
            f.write("\n")

    print(f"Detailed report written to: {report_file}")

    # Write failed links for easy fixing
    if results['client_error'] or results['server_error'] or results['timeout'] or results['error']:
        failed_file = 'FAILED_LINKS.md'
        with open(failed_file, 'w', encoding='utf-8') as f:
            f.write("# Failed Links - Need Fixing\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            all_failed = (
                results['client_error'] +
                results['server_error'] +
                results['timeout'] +
                results['error']
            )

            for url, status, _ in all_failed:
                f.write(f"## {url}\n\n")
                f.write(f"- **Status**: {status}\n")
                f.write(f"- **Used in**:\n")
                for source in url_sources[url]:
                    f.write(f"  - {source}\n")
                f.write("\n")

        print(f"Failed links written to: {failed_file}")
        print()
        print("⚠️  ATTENTION: Some links failed! Please review and fix.")
    else:
        print()
        print("🎉 All links are working!")

if __name__ == '__main__':
    main()
