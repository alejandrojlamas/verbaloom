# Security policy

## Supported code

Security fixes target the current `main` branch. Historical snapshots and local
modifications should be updated before reporting a vulnerability.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting for this repository. Do not place
credentials, personal documents, unpublished books, provider responses, or a
working exploit in a public issue.

Please include the affected commit, impact and prerequisites, minimal steps
using synthetic data, and a proposed mitigation when available. Revoke and
rotate any credential that may have been disclosed.

## Operating boundary

VerbaLoom is a local-first, single-user research application. It does not implement
accounts or application-level authentication.

- Keep the Flask service bound to loopback.
- Put any remote route behind an identity-aware authenticated proxy and a
  restrictive network ACL.
- Treat Tailnet or LAN membership as reachability, not identity.
- Configure exact browser origins; never use wildcard CORS.
- Keep `.env`, uploads, outputs, checkpoints, generated profiles, reports, and
  runtime databases outside Git.
- Supply provider keys at runtime. A key loaded from the environment must not be
  forwarded to a custom endpoint unless that exact origin is explicitly
  trusted.
- Process only documents you are authorized to transform and review generated
  output before publication.

The test suite exercises origin checks, path confinement, archive budgets,
credential routing, safe output extensions, and runtime response redaction.
Real providers, reverse proxies, containers, and complete book workflows still
require environment-specific validation.
