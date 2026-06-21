# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| latest  | Yes       |

Only the latest release receives security patches.

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

To report a vulnerability, use [GitHub's private vulnerability reporting](https://github.com/laplacef/digest-generator/security/advisories/new). You can expect an initial response within 72 hours.

Please include:
- A description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment

## Scope

The following are considered security issues:
- Credential exposure (e.g., `HF_TOKEN` leakage through logs or output files)
- Dependency vulnerabilities in direct dependencies
- Injection risks via malicious RSS feed content (e.g., crafted HTML parsed by BeautifulSoup)
- Arbitrary code execution through model loading or deserialization

## Out of Scope

The following are **not** security issues:
- RSS feed unavailability or content quality
- Model output accuracy or bias
- Rate limiting or denial of service against upstream RSS feeds
- Vulnerabilities in transitive dependencies not exploitable through this project
