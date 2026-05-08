# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in KDP Scout, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please email: **randypellegrini@gmail.com**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You can expect an initial response within 48 hours. We will work with you to understand the issue and coordinate a fix before any public disclosure.

## Scope

KDP Scout runs locally and does not operate a hosted service. Security concerns primarily relate to:
- Credential handling (API keys stored in `.env`)
- Dependencies with known vulnerabilities
- Data scraping that could expose user activity
