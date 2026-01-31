#!/usr/bin/env python3
"""Extract URLs from SKILL.md documentation sections."""

import re
import os
from pathlib import Path
from typing import List, Dict, Tuple

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

        # Skip internal links (anchors)
        if url.startswith('#'):
            continue

        urls.append({
            'text': link_text,
            'url': url,
            'file': os.path.basename(os.path.dirname(file_path))
        })

    return urls

def main():
    """Main function to process all SKILL.md files."""
    base_dir = Path('/Users/gaku/claude-code-skills')
    skill_files = list(base_dir.glob('*/SKILL.md'))

    all_urls = []

    for skill_file in sorted(skill_files):
        urls = extract_urls_from_file(str(skill_file))
        all_urls.extend(urls)

    # Group by file
    by_file = {}
    for url_info in all_urls:
        file_name = url_info['file']
        if file_name not in by_file:
            by_file[file_name] = []
        by_file[file_name].append(url_info)

    # Print report
    print("=" * 80)
    print("URL EXTRACTION REPORT")
    print("=" * 80)
    print(f"\nTotal files processed: {len(skill_files)}")
    print(f"Total URLs extracted: {len(all_urls)}")
    print("\n" + "=" * 80)

    for file_name in sorted(by_file.keys()):
        urls = by_file[file_name]
        print(f"\n{file_name}/ ({len(urls)} links)")
        print("-" * 80)
        for i, url_info in enumerate(urls, 1):
            print(f"{i:2d}. [{url_info['text']}]")
            print(f"    {url_info['url']}")

    print("\n" + "=" * 80)
    print("UNIQUE DOMAINS")
    print("=" * 80)

    domains = set()
    for url_info in all_urls:
        url = url_info['url']
        if url.startswith('http'):
            domain = re.match(r'https?://([^/]+)', url)
            if domain:
                domains.add(domain.group(1))

    for domain in sorted(domains):
        print(f"  - {domain}")

    print(f"\nTotal unique domains: {len(domains)}")

if __name__ == '__main__':
    main()
