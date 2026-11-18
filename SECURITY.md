# Security Policy

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue. Use GitHub private vulnerability
reporting: <https://github.com/zachshallbetter/perception-is-how-we-experience/security/advisories/new>.

Include:

- a description of the issue and its impact,
- steps to reproduce (a minimal example is ideal),
- the affected commit or version.

This is a solo-maintained project. Responses and fixes are best-effort, with no SLA and no guaranteed response time.
Please give a reasonable window to ship a fix before any public disclosure.

## Scope

perception-is-how-we-experience is a research codebase (Sensory Perceiver) with training and evaluation scripts, a paper, and generated results. It is run locally or on the maintainer's own GPU host; it exposes no hosted service. Reports about unsafe deserialization of checkpoints or datasets, shell-script handling of untrusted arguments, or vulnerable pinned dependencies in `requirements.txt` are in scope.

## Supported versions

Only the latest commit on `main` receives fixes. The repository has one tag, `v4.0.0`; older tags are not patched.
