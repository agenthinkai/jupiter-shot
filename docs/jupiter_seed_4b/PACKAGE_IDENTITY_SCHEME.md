# Jupiter Seed 4B — Review-Package Identity Scheme

> **Purpose:** prevent a stale or substituted reviewer package from passing Gate 16, without attempting an impossible self-referential Git commit hash.

## Problem

A reviewer artifact cannot safely embed the hash of the commit that first contains that same artifact. Git computes a commit hash from its tree, and the tree would contain the artifact that itself embeds the desired hash. Requiring an artifact to embed the hash of the commit that contains it therefore creates a circular dependency.

## Defined Commit Roles

| Term | Definition | Where it is recorded |
| :--- | :--- | :--- |
| **Artifact source commit** | The exact source-tree commit from which the reviewer artifacts were generated. | `package_commit` in the sidecar, every reviewer JSONL and CSV record, and the Markdown metadata line. |
| **Artifact release commit** | The child commit that records the generated artifacts. | Git `HEAD` at release-gate verification time. |
| **Dataset content commit** | The immutable commit that established the frozen 101-record corpus. | Corpus manifest and authorization controls. |

The `package_commit` field always means **artifact source commit**, not artifact release commit.

## Release Procedure

1. Complete code and test changes in an **artifact source commit** `S`.
2. Run the production package generator while `HEAD=S`. Every representation embeds `package_commit=S`.
3. Commit only the generated reviewer artifacts as the **artifact release commit** `R`, whose direct parent is `S`.
4. Verify Gate 16 at `R`. Gate 16 resolves both `HEAD` (`R`) and `HEAD^` (`S`) through the deterministic repository root. It requires all representations to contain the valid 40-character hash `S`.

This relationship is deterministic and fail-closed:

```text
S  (artifact source commit)  <-- package_commit stored in artifacts
|
R  (artifact release commit; checked-out HEAD during release audit)
```

A subsequent code, artifact, or history change invalidates the relationship unless the package is regenerated from that new source commit and recorded in a new child artifact-release commit.

## Gate 16 Requirements

Gate 16 fails if any of the following is true:

| Check | Expected result |
| :--- | :--- |
| Repository root cannot be resolved from `readiness_gate.py` | FAIL |
| `git rev-parse HEAD` or `git rev-parse HEAD^` cannot resolve a valid full Git hash | FAIL |
| Sidecar `package_commit` is absent, malformed, or differs from `HEAD^` | FAIL |
| Any reviewer JSONL or CSV record has an absent, malformed, or mismatched `package_commit` | FAIL |
| Markdown package metadata is absent, malformed, or mismatched | FAIL |
| A queued identifier is missing, duplicated, or unexpected in a reviewer representation | FAIL |
| A flagged record is represented as `OK`, `CLEAR`, or `NONE` | FAIL |

No exception path may replace an identity-resolution failure with an empty value or a passing result.

## Human and Training Controls

This identity scheme does not alter the frozen corpus, content-risk findings, human-review fields, legal-review fields, or training authorization policy. A package with Gate 14 findings remains `HUMAN_REVIEW_REQUIRED`; training remains unauthorized until a valid human-completed authorization artifact and readiness code `0` are both present.
