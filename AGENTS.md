# Instructions for AI assistants

Follow https://github.com/romkey/python-ai-rules/blob/main/RULES.md.
Follows python-ai-rules v0.0.0 (main at 790642a, 2026-11-11; the rules
have no release yet). When re-scanning, apply the entries in its
UPGRADING.md newer than that, then update this line.
With Claude Code, the `python-rules` plugin loads them for you.

Read docs/overview.md first. `main` is protected: work on a branch and
open a pull request; CI and Lint must pass before it can merge.

<!-- Project-specific notes go below. -->

## Deliberate differences from the template

- The app doesn't listen on a port, so there is no `/healthz` and
  `docker-compose.prod.yaml` has no health check for `app`.
- PostgreSQL is optional (`DATABASE_URL` empty disables it), so the
  production `db` and `migrate` services are behind the `postgres`
  compose profile.
- Database tests read `TEST_DATABASE_URL`, not `DATABASE_URL`, so a real
  `DATABASE_URL` in the environment is never touched by tests.
