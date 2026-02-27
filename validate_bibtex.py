#!/usr/bin/env python3
"""
Comprehensive BibTeX validation and cleanup analysis for Master's thesis.
Detects duplicates, validates entries, and identifies format issues.
"""

import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from difflib import SequenceMatcher
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[0]

class BibTexValidator:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.entries = []
        self.duplicates = []
        self.validation_issues = []
        self.format_issues = []
        self.citation_key_mapping = {}
        self.authenticity_concerns = []
        
    def parse_bibtex(self):
        """Parse BibTeX file into structured entries."""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by entry start (@ followed by entry type)
        entry_pattern = r'@(\w+)\{([^,]+),\s*\n(.*?)\n\}'
        matches = re.finditer(entry_pattern, content, re.DOTALL)
        
        for match in matches:
            entry_type = match.group(1)
            citation_key = match.group(2).strip()
            fields_text = match.group(3)
            
            # Parse fields
            fields = {}
            field_pattern = r'(\w+)\s*=\s*\{([^}]*)\}|(\w+)\s*=\s*"([^"]*)"'
            for field_match in re.finditer(field_pattern, fields_text):
                if field_match.group(1):
                    field_name = field_match.group(1).lower()
                    field_value = field_match.group(2)
                else:
                    field_name = field_match.group(3).lower()
                    field_value = field_match.group(4)
                fields[field_name] = field_value.strip()
            
            self.entries.append({
                'type': entry_type.lower(),
                'key': citation_key,
                'fields': fields,
                'raw': match.group(0)
            })
    
    def normalize_doi(self, doi: str) -> str:
        """Normalize DOI by removing URL prefixes."""
        if not doi:
            return ""
        # Remove common URL prefixes
        doi = doi.replace('https://doi.org/', '')
        doi = doi.replace('http://doi.org/', '')
        doi = doi.replace('doi:', '')
        return doi.strip()
    
    def similar_titles(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles (0-1 range)."""
        if not title1 or not title2:
            return 0.0
        # Normalize titles
        t1 = title1.lower().strip()
        t2 = title2.lower().strip()
        return SequenceMatcher(None, t1, t2).ratio()
    
    def detect_duplicates(self):
        """Detect all types of duplicates."""
        # Track by citation key
        key_counts = defaultdict(list)
        for i, entry in enumerate(self.entries):
            key_counts[entry['key']].append(i)
        
        # Exact duplicate keys
        for key, indices in key_counts.items():
            if len(indices) > 1:
                self.duplicates.append({
                    'type': 'exact_key',
                    'key': key,
                    'indices': indices,
                    'entries': [self.entries[i] for i in indices]
                })
        
        # DOI duplicates
        doi_map = defaultdict(list)
        for i, entry in enumerate(self.entries):
            doi = entry['fields'].get('doi', '')
            if doi:
                normalized_doi = self.normalize_doi(doi)
                if normalized_doi:
                    doi_map[normalized_doi].append(i)
        
        for doi, indices in doi_map.items():
            if len(indices) > 1:
                keys = [self.entries[i]['key'] for i in indices]
                if len(set(keys)) > 1:  # Different citation keys
                    self.duplicates.append({
                        'type': 'same_doi',
                        'doi': doi,
                        'indices': indices,
                        'keys': keys,
                        'entries': [self.entries[i] for i in indices]
                    })
        
        # Title similarity duplicates
        for i in range(len(self.entries)):
            for j in range(i + 1, len(self.entries)):
                title1 = self.entries[i]['fields'].get('title', '')
                title2 = self.entries[j]['fields'].get('title', '')
                
                if title1 and title2:
                    similarity = self.similar_titles(title1, title2)
                    if similarity > 0.90:
                        # Check if not already caught by other duplicate detection
                        already_found = False
                        for dup in self.duplicates:
                            if i in dup.get('indices', []) and j in dup.get('indices', []):
                                already_found = True
                                break
                        
                        if not already_found:
                            self.duplicates.append({
                                'type': 'similar_title',
                                'similarity': similarity,
                                'indices': [i, j],
                                'keys': [self.entries[i]['key'], self.entries[j]['key']],
                                'entries': [self.entries[i], self.entries[j]]
                            })
        
        # Author+Year duplicates
        author_year_map = defaultdict(list)
        for i, entry in enumerate(self.entries):
            author = entry['fields'].get('author', '')
            year = entry['fields'].get('year', '')
            
            if author and year:
                # Extract first author's last name
                first_author = author.split(' and ')[0].split(',')[0].strip()
                key_signature = f"{first_author.lower()}_{year}"
                author_year_map[key_signature].append(i)
        
        for signature, indices in author_year_map.items():
            if len(indices) > 1:
                keys = [self.entries[i]['key'] for i in indices]
                # Only flag if different keys and similar titles
                if len(set(keys)) > 1:
                    titles = [self.entries[i]['fields'].get('title', '') for i in indices]
                    max_similarity = 0
                    for t1 in range(len(titles)):
                        for t2 in range(t1 + 1, len(titles)):
                            sim = self.similar_titles(titles[t1], titles[t2])
                            max_similarity = max(max_similarity, sim)
                    
                    if max_similarity > 0.5:  # Lower threshold for author+year duplicates
                        self.duplicates.append({
                            'type': 'author_year',
                            'signature': signature,
                            'similarity': max_similarity,
                            'indices': indices,
                            'keys': keys,
                            'entries': [self.entries[i] for i in indices]
                        })
    
    def validate_entries(self):
        """Validate all entries for common issues."""
        for i, entry in enumerate(self.entries):
            fields = entry['fields']
            key = entry['key']
            
            # Check DOI format
            if 'doi' in fields:
                doi = fields['doi']
                normalized_doi = self.normalize_doi(doi)
                
                # Check if DOI is actually a URL
                if 'http' in doi or 'doi.org' in doi:
                    self.format_issues.append({
                        'key': key,
                        'issue': 'doi_has_url',
                        'field': 'doi',
                        'current': doi,
                        'suggested': normalized_doi
                    })
                
                # Check if DOI starts with 10.
                if normalized_doi and not normalized_doi.startswith('10.'):
                    self.validation_issues.append({
                        'key': key,
                        'issue': 'invalid_doi_format',
                        'field': 'doi',
                        'value': doi,
                        'message': 'DOI should start with "10."'
                    })
            
            # Check year plausibility
            if 'year' in fields:
                year = fields['year']
                try:
                    year_int = int(year)
                    if year_int < 1900 or year_int > 2026:
                        self.validation_issues.append({
                            'key': key,
                            'issue': 'implausible_year',
                            'field': 'year',
                            'value': year,
                            'message': f'Year {year} is outside reasonable range (1900-2026)'
                        })
                except ValueError:
                    if year != 'n.d.':
                        self.validation_issues.append({
                            'key': key,
                            'issue': 'invalid_year',
                            'field': 'year',
                            'value': year,
                            'message': 'Year is not a valid number'
                        })
            
            # Check for missing required fields
            required_fields = {
                'article': ['author', 'title', 'journal', 'year'],
                'book': ['author', 'title', 'publisher', 'year'],
                'inproceedings': ['author', 'title', 'booktitle', 'year'],
                'incollection': ['author', 'title', 'booktitle', 'publisher', 'year']
            }
            
            entry_type = entry['type']
            if entry_type in required_fields:
                for req_field in required_fields[entry_type]:
                    if req_field not in fields or not fields[req_field].strip():
                        self.validation_issues.append({
                            'key': key,
                            'issue': 'missing_required_field',
                            'field': req_field,
                            'message': f'Missing required field for {entry_type}'
                        })
            
            # Check for suspicious author fields
            if 'author' in fields:
                author = fields['author']
                if 'others' in author.lower() or 'et al' in author.lower():
                    self.authenticity_concerns.append({
                        'key': key,
                        'issue': 'generic_author',
                        'field': 'author',
                        'value': author,
                        'message': 'Author field contains generic placeholders'
                    })
                
                # Check for "Author, A. and others" pattern
                if re.search(r'Author,?\s+[A-Z]\.?\s+and\s+others', author):
                    self.authenticity_concerns.append({
                        'key': key,
                        'issue': 'placeholder_author',
                        'field': 'author',
                        'value': author,
                        'message': 'Author field appears to be a placeholder'
                    })
            
            # Check page ranges for proper formatting
            if 'pages' in fields:
                pages = fields['pages']
                # Check if using single dash instead of double dash
                if re.search(r'\d+-\d+', pages) and '--' not in pages:
                    self.format_issues.append({
                        'key': key,
                        'issue': 'page_dash_format',
                        'field': 'pages',
                        'current': pages,
                        'suggested': pages.replace('-', '--')
                    })
    
    def generate_citation_key_mapping(self):
        """Generate mapping for citation keys that will be changed."""
        for dup in self.duplicates:
            if dup['type'] == 'exact_key':
                # All duplicates should map to the first occurrence
                primary_key = dup['key']
                # No mapping needed - they're already the same key
                # But we need to remove duplicates
                pass
            
            elif dup['type'] in ['same_doi', 'similar_title', 'author_year']:
                # Choose which entry to keep
                entries = dup['entries']
                keys = dup['keys']
                
                # Prefer entry with more complete information
                scores = []
                for entry in entries:
                    score = 0
                    fields = entry['fields']
                    # More fields = better
                    score += len(fields)
                    # Has DOI = better
                    if 'doi' in fields and fields['doi']:
                        score += 5
                    # Has proper author (not placeholder) = better
                    author = fields.get('author', '')
                    if author and 'others' not in author.lower():
                        score += 3
                    scores.append(score)
                
                # Keep the entry with highest score
                best_idx = scores.index(max(scores))
                primary_key = keys[best_idx]
                
                # Map all other keys to the primary
                for i, key in enumerate(keys):
                    if i != best_idx:
                        self.citation_key_mapping[key] = {
                            'new_key': primary_key,
                            'reason': f'duplicate ({dup["type"]})',
                            'duplicate_type': dup['type']
                        }
    
    def generate_report(self) -> str:
        """Generate comprehensive validation report."""
        report = []
        report.append("# BibTeX Validation and Cleanup Report")
        report.append("=" * 80)
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        report.append(f"- **Total Entries**: {len(self.entries)}")
        report.append(f"- **Duplicates Found**: {len(self.duplicates)}")
        report.append(f"- **Validation Issues**: {len(self.validation_issues)}")
        report.append(f"- **Format Issues**: {len(self.format_issues)}")
        report.append(f"- **Authenticity Concerns**: {len(self.authenticity_concerns)}")
        report.append(f"- **Citation Keys to Update**: {len(self.citation_key_mapping)}")
        report.append("")
        
        # Entry type statistics
        type_counts = defaultdict(int)
        for entry in self.entries:
            type_counts[entry['type']] += 1
        
        report.append("### Entries by Type")
        for entry_type, count in sorted(type_counts.items()):
            report.append(f"- {entry_type}: {count}")
        report.append("")
        
        # Duplicates Section
        report.append("## 1. DUPLICATES FOUND")
        report.append("=" * 80)
        report.append("")
        
        if not self.duplicates:
            report.append("✓ No duplicates detected.")
            report.append("")
        else:
            for i, dup in enumerate(self.duplicates, 1):
                report.append(f"### Duplicate #{i}: {dup['type'].replace('_', ' ').title()}")
                report.append("")
                
                if dup['type'] == 'exact_key':
                    report.append(f"**Citation Key**: `{dup['key']}`")
                    report.append(f"**Occurrences**: {len(dup['indices'])}")
                    report.append("")
                    for idx in dup['indices']:
                        entry = self.entries[idx]
                        report.append(f"- Entry #{idx + 1}:")
                        report.append(f"  - Title: {entry['fields'].get('title', 'N/A')}")
                        report.append(f"  - Year: {entry['fields'].get('year', 'N/A')}")
                        report.append(f"  - DOI: {entry['fields'].get('doi', 'N/A')}")
                    
                    report.append("")
                    report.append("**RECOMMENDATION**: Keep first occurrence, remove duplicates.")
                    report.append("")
                
                elif dup['type'] == 'same_doi':
                    report.append(f"**DOI**: `{dup['doi']}`")
                    report.append(f"**Different Citation Keys**: {', '.join([f'`{k}`' for k in dup['keys']])}")
                    report.append("")
                    for idx, key in zip(dup['indices'], dup['keys']):
                        entry = self.entries[idx]
                        report.append(f"- `{key}`:")
                        report.append(f"  - Title: {entry['fields'].get('title', 'N/A')}")
                        report.append(f"  - Year: {entry['fields'].get('year', 'N/A')}")
                        report.append(f"  - Journal: {entry['fields'].get('journal', 'N/A')}")
                    
                    report.append("")
                    # Determine which to keep
                    best_key = max(dup['keys'], key=lambda k: len(self.entries[dup['indices'][dup['keys'].index(k)]]['fields']))
                    report.append(f"**RECOMMENDATION**: Keep `{best_key}`, merge others.")
                    report.append("")
                
                elif dup['type'] == 'similar_title':
                    report.append(f"**Similarity Score**: {dup['similarity']:.2%}")
                    report.append(f"**Citation Keys**: {', '.join([f'`{k}`' for k in dup['keys']])}")
                    report.append("")
                    for idx, key in zip(dup['indices'], dup['keys']):
                        entry = self.entries[idx]
                        report.append(f"- `{key}`:")
                        report.append(f"  - Title: {entry['fields'].get('title', 'N/A')}")
                        report.append(f"  - Author: {entry['fields'].get('author', 'N/A')[:80]}...")
                        report.append(f"  - Year: {entry['fields'].get('year', 'N/A')}")
                        report.append(f"  - DOI: {entry['fields'].get('doi', 'N/A')}")
                    
                    report.append("")
                    report.append("**RECOMMENDATION**: Review manually - likely duplicates but verify.")
                    report.append("")
                
                elif dup['type'] == 'author_year':
                    report.append(f"**Author+Year Signature**: `{dup['signature']}`")
                    report.append(f"**Title Similarity**: {dup.get('similarity', 0):.2%}")
                    report.append(f"**Citation Keys**: {', '.join([f'`{k}`' for k in dup['keys']])}")
                    report.append("")
                    for idx, key in zip(dup['indices'], dup['keys']):
                        entry = self.entries[idx]
                        report.append(f"- `{key}`:")
                        report.append(f"  - Title: {entry['fields'].get('title', 'N/A')}")
                        report.append(f"  - Journal: {entry['fields'].get('journal', 'N/A')}")
                    
                    report.append("")
                    report.append("**RECOMMENDATION**: Review - may be different works by same author.")
                    report.append("")
                
                report.append("-" * 80)
                report.append("")
        
        # Citation Key Mapping
        report.append("## 2. CITATION KEY CHANGE MAPPING")
        report.append("=" * 80)
        report.append("")
        report.append("**CRITICAL**: Update these citation keys in all LaTeX files before cleaning the .bib file.")
        report.append("")
        
        if not self.citation_key_mapping:
            report.append("✓ No citation key changes required.")
            report.append("")
        else:
            report.append("```")
            for old_key, mapping_info in sorted(self.citation_key_mapping.items()):
                report.append(f"{old_key} -> {mapping_info['new_key']} (reason: {mapping_info['reason']})")
            report.append("```")
            report.append("")
        
        # Validation Issues
        report.append("## 3. VALIDATION ISSUES")
        report.append("=" * 80)
        report.append("")
        
        if not self.validation_issues:
            report.append("✓ No validation issues detected.")
            report.append("")
        else:
            # Group by issue type
            issues_by_type = defaultdict(list)
            for issue in self.validation_issues:
                issues_by_type[issue['issue']].append(issue)
            
            for issue_type, issues in sorted(issues_by_type.items()):
                report.append(f"### {issue_type.replace('_', ' ').title()} ({len(issues)} entries)")
                report.append("")
                for issue in issues[:10]:  # Limit to first 10 per type
                    report.append(f"- **`{issue['key']}`**: {issue.get('message', '')}")
                    if 'value' in issue:
                        report.append(f"  - Current value: `{issue['value']}`")
                
                if len(issues) > 10:
                    report.append(f"  ... and {len(issues) - 10} more entries with this issue.")
                report.append("")
        
        # Format Issues
        report.append("## 4. FORMAT STANDARDIZATION ISSUES")
        report.append("=" * 80)
        report.append("")
        
        if not self.format_issues:
            report.append("✓ No format issues detected.")
            report.append("")
        else:
            # Group by issue type
            issues_by_type = defaultdict(list)
            for issue in self.format_issues:
                issues_by_type[issue['issue']].append(issue)
            
            for issue_type, issues in sorted(issues_by_type.items()):
                report.append(f"### {issue_type.replace('_', ' ').title()} ({len(issues)} entries)")
                report.append("")
                for issue in issues[:10]:
                    report.append(f"- **`{issue['key']}`**:")
                    report.append(f"  - Current: `{issue['current']}`")
                    report.append(f"  - Suggested: `{issue['suggested']}`")
                
                if len(issues) > 10:
                    report.append(f"  ... and {len(issues) - 10} more entries.")
                report.append("")
        
        # Authenticity Concerns
        report.append("## 5. AUTHENTICITY CONCERNS")
        report.append("=" * 80)
        report.append("")
        
        if not self.authenticity_concerns:
            report.append("✓ No authenticity concerns detected.")
            report.append("")
        else:
            for concern in self.authenticity_concerns:
                report.append(f"### `{concern['key']}`")
                report.append("")
                report.append(f"- **Issue**: {concern['issue'].replace('_', ' ').title()}")
                report.append(f"- **Field**: {concern['field']}")
                report.append(f"- **Value**: {concern['value']}")
                report.append(f"- **Message**: {concern['message']}")
                report.append("")
                report.append("**ACTION REQUIRED**: Manual review and correction needed.")
                report.append("")
        
        # Recommended Actions
        report.append("## 6. RECOMMENDED MANUAL REVIEWS")
        report.append("=" * 80)
        report.append("")
        
        manual_review_needed = []
        
        # Entries with placeholder authors
        for entry in self.entries:
            author = entry['fields'].get('author', '')
            if 'Author, A.' in author or 'Author, B.' in author:
                manual_review_needed.append({
                    'key': entry['key'],
                    'reason': 'Placeholder author detected',
                    'details': f"Author: {author}"
                })
        
        # Entries without DOI in recent years
        for entry in self.entries:
            year = entry['fields'].get('year', '')
            try:
                year_int = int(year)
                if year_int >= 2020 and 'doi' not in entry['fields']:
                    manual_review_needed.append({
                        'key': entry['key'],
                        'reason': 'Recent entry without DOI',
                        'details': f"Year: {year}, Type: {entry['type']}"
                    })
            except:
                pass
        
        if manual_review_needed:
            for item in manual_review_needed[:20]:  # Limit output
                report.append(f"- **`{item['key']}`**: {item['reason']}")
                report.append(f"  - {item['details']}")
                report.append("")
            
            if len(manual_review_needed) > 20:
                report.append(f"... and {len(manual_review_needed) - 20} more entries requiring review.")
                report.append("")
        else:
            report.append("✓ No additional manual reviews recommended.")
            report.append("")
        
        # Summary statistics
        report.append("## FINAL STATISTICS")
        report.append("=" * 80)
        report.append("")
        
        # Count unique entries requiring attention
        entries_needing_attention = set()
        for d in self.duplicates:
            for idx in d.get('indices', []):
                entries_needing_attention.add(self.entries[idx]['key'])
        for i in self.validation_issues:
            entries_needing_attention.add(i['key'])
        for i in self.format_issues:
            entries_needing_attention.add(i['key'])
        
        report.append(f"- Total entries requiring attention: {len(entries_needing_attention)}")
        report.append(f"- Entries that will be removed (duplicates): {len(self.citation_key_mapping)}")
        report.append(f"- Entries requiring format fixes: {len(self.format_issues)}")
        report.append(f"- Entries requiring manual review: {len(manual_review_needed)}")
        report.append("")
        
        return "\n".join(report)
    
    def run_analysis(self):
        """Run complete analysis pipeline."""
        print("Parsing BibTeX file...")
        self.parse_bibtex()
        
        print(f"Found {len(self.entries)} entries.")
        
        print("Detecting duplicates...")
        self.detect_duplicates()
        
        print("Validating entries...")
        self.validate_entries()
        
        print("Generating citation key mapping...")
        self.generate_citation_key_mapping()
        
        print("Generating report...")
        return self.generate_report()


if __name__ == "__main__":
    filepath = str(_PROJECT_ROOT / r"thesis/bibliography/references.bib")
    
    validator = BibTexValidator(filepath)
    report = validator.run_analysis()
    
    # Save report
    output_file = str(_PROJECT_ROOT / r"bibtex_validation_report.md")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nReport saved to: {output_file}")
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total entries: {len(validator.entries)}")
    print(f"Duplicates found: {len(validator.duplicates)}")
    print(f"Validation issues: {len(validator.validation_issues)}")
    print(f"Format issues: {len(validator.format_issues)}")
    print(f"Citation keys to update: {len(validator.citation_key_mapping)}")
