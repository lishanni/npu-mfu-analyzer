# Security Policy / 安全策略

## Reporting Security Issues

Please report security issues privately to the maintainer instead of opening a public issue. If you do not have a private contact path, open a GitHub issue with only a minimal description and ask for a private channel.

请不要在公开 issue 中粘贴密钥、内部 trace、集群路径或其他敏感数据。如需报告安全问题，请先联系维护者建立私密沟通渠道。

## Sensitive Data

Do not commit:

- API keys, tokens, cookies, SSH keys, or cloud credentials.
- Raw internal profiling traces with user names, host names, job names, or cluster topology.
- Proprietary model names, dataset paths, or command-line arguments that reveal private infrastructure.

Use `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and other keys only as runtime environment variables.

## Supported Versions

Security fixes target the latest `master` branch until the project starts maintaining release branches.

