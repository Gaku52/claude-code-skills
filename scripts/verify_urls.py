#!/usr/bin/env python3
"""Verify and analyze URLs from SKILL.md documentation sections."""

import re
import os
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

def extract_urls_from_file(file_path: str) -> List[Dict[str, str]]:
    """Extract URLs from the documentation section of a SKILL.md file."""
    urls = []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the documentation section
    doc_section_pattern = r'## 📚 公式ドキュメント・参考リソース(.*?)(?=\n##\s|\Z)'
    match = re.search(doc_section_pattern, content, re.DOTALL)

    if not match:
        return urls

    doc_section = match.group(1)

    # Extract markdown links: [text](url)
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    matches = re.finditer(link_pattern, doc_section)

    for match in matches:
        link_text = match.group(1)
        url = match.group(2)

        # Skip internal links (anchors and relative paths)
        if url.startswith('#'):
            continue

        urls.append({
            'text': link_text,
            'url': url,
            'file': os.path.basename(os.path.dirname(file_path))
        })

    return urls

def analyze_url(url: str) -> Dict[str, any]:
    """Analyze a URL for potential issues."""
    issues = []

    # Check if it's a relative path (internal link)
    if url.startswith('./'):
        return {
            'type': 'internal',
            'valid': True,
            'issues': []
        }

    # Check if it's an HTTP/HTTPS URL
    if not url.startswith(('http://', 'https://')):
        issues.append('Not a valid HTTP/HTTPS URL')
        return {
            'type': 'invalid',
            'valid': False,
            'issues': issues
        }

    # Check for common issues
    if url.startswith('http://') and 'localhost' not in url:
        issues.append('Uses HTTP instead of HTTPS (security concern)')

    # Check for malformed URLs
    if ' ' in url:
        issues.append('Contains spaces')

    # Check for missing trailing slash on domain-only URLs
    # (Some sites require it, some don't - this is just a warning)
    url_parts = re.match(r'https?://([^/]+)(/.*)?$', url)
    if url_parts and not url_parts.group(2):
        # Domain-only URL without trailing slash
        pass  # This is actually fine for most sites

    # Check for suspicious patterns
    if '..' in url:
        issues.append('Contains ".." which may indicate a malformed path')

    # Known potentially problematic patterns
    if url.endswith('/en/'):
        issues.append('Language-specific URL - may change or redirect')

    # Check for very long URLs (potential copy-paste errors)
    if len(url) > 200:
        issues.append('Very long URL (>200 chars) - potential copy-paste error')

    return {
        'type': 'external',
        'valid': len(issues) == 0 or all('HTTP instead of HTTPS' not in i for i in issues),
        'issues': issues,
        'domain': re.match(r'https?://([^/]+)', url).group(1) if re.match(r'https?://([^/]+)', url) else None
    }

def main():
    """Main function to process and analyze all URLs."""
    base_dir = Path('/Users/gaku/claude-code-skills')
    skill_files = list(base_dir.glob('*/SKILL.md'))

    all_urls = []
    internal_links = []

    for skill_file in sorted(skill_files):
        urls = extract_urls_from_file(str(skill_file))
        all_urls.extend(urls)

    # Analyze each URL
    url_analysis = defaultdict(list)
    issues_found = []

    for url_info in all_urls:
        analysis = analyze_url(url_info['url'])
        url_info['analysis'] = analysis

        if analysis['type'] == 'internal':
            internal_links.append(url_info)
        elif not analysis['valid'] or analysis['issues']:
            issues_found.append(url_info)

        url_analysis[analysis['type']].append(url_info)

    # Print comprehensive report
    print("=" * 100)
    print("URL VERIFICATION REPORT")
    print("=" * 100)
    print(f"\nTotal SKILL files checked: {len(skill_files)}")
    print(f"Total URLs extracted: {len(all_urls)}")
    print(f"  - External URLs: {len(url_analysis['external'])}")
    print(f"  - Internal links: {len(url_analysis['internal'])}")
    print(f"  - Invalid URLs: {len(url_analysis['invalid'])}")

    # Count valid vs issues
    valid_count = len([u for u in all_urls if u['analysis']['valid']])
    issues_count = len(issues_found)

    print(f"\nValidation Summary:")
    print(f"  - Valid URLs: {valid_count}")
    print(f"  - URLs with issues: {issues_count}")

    # Report internal links (for verification)
    if internal_links:
        print("\n" + "=" * 100)
        print("INTERNAL LINKS (Relative Paths)")
        print("=" * 100)
        print("\nThese are relative file paths that need manual verification:\n")

        for url_info in internal_links:
            print(f"File: {url_info['file']}/")
            print(f"  Link text: {url_info['text']}")
            print(f"  Path: {url_info['url']}")
            print()

    # Report issues
    if issues_found:
        print("\n" + "=" * 100)
        print("URLS WITH POTENTIAL ISSUES")
        print("=" * 100)

        by_file = defaultdict(list)
        for url_info in issues_found:
            by_file[url_info['file']].append(url_info)

        for file_name in sorted(by_file.keys()):
            url_list = by_file[file_name]
            print(f"\n{file_name}/ ({len(url_list)} issue(s))")
            print("-" * 100)

            for i, url_info in enumerate(url_list, 1):
                print(f"\n{i}. Link text: {url_info['text']}")
                print(f"   URL: {url_info['url']}")
                print(f"   Issues:")
                for issue in url_info['analysis']['issues']:
                    print(f"     - {issue}")
    else:
        print("\n" + "=" * 100)
        print("NO ISSUES FOUND!")
        print("=" * 100)
        print("\nAll external URLs appear to be properly formatted.")

    # Domain statistics
    print("\n" + "=" * 100)
    print("DOMAIN STATISTICS")
    print("=" * 100)

    domain_counts = defaultdict(int)
    for url_info in url_analysis['external']:
        if url_info['analysis']['domain']:
            domain_counts[url_info['analysis']['domain']] += 1

    print(f"\nTotal unique domains: {len(domain_counts)}")
    print("\nTop 15 most referenced domains:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {count:3d}  {domain}")

    # Recommendations
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)
    print("""
1. INTERNAL LINKS:
   - All internal links (starting with ./) should be verified manually
   - Check that the referenced files actually exist in the repository
   - Ensure the paths are correct relative to each SKILL.md file

2. EXTERNAL URLS:
   - URLs using HTTP should be checked if HTTPS versions exist
   - Very long URLs should be reviewed for potential copy-paste errors
   - Language-specific URLs (/en/, /ja/) may redirect or change over time

3. URL ACCESSIBILITY TESTING:
   - The URLs extracted above should be tested with automated tools
   - Recommended tools: curl, wget, or HTTP client libraries
   - Check for: 200 OK, 301/302 redirects, 404 Not Found, SSL errors

4. MANUAL VERIFICATION:
   - Official documentation URLs should be checked periodically
   - Some sites may reorganize content causing link rot
   - Consider implementing automated link checking in CI/CD

5. LINK MAINTENANCE:
   - Set up quarterly reviews of all documentation links
   - Use tools like linkchecker or broken-link-checker
   - Monitor for deprecated or moved documentation pages
    """)

    print("\n" + "=" * 100)
    print("NEXT STEPS")
    print("=" * 100)
    print("""
To perform actual HTTP/HTTPS verification, you can:

1. Use curl to test each URL:
   curl -I -L --max-time 10 "URL"

2. Use a dedicated link checker:
   npm install -g broken-link-checker
   blc https://your-site.com --recursive

3. Use Python requests:
   python -c "import requests; r=requests.head('URL', timeout=10); print(r.status_code)"

4. Use GitHub Actions with link checker:
   - lychee-action (Rust-based, fast)
   - linkinator (Node.js-based)
   - markdown-link-check (Focused on Markdown files)
    """)

if __name__ == '__main__':
    main()
