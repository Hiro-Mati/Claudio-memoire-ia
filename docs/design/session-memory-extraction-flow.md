# Session Memory Extraction Flow

This document records the current implementation. It is meant to be used as a
code-modification reference, so it avoids proposed or removed flows.

## Policy

`memory_policy` carries target switches plus an optional global memory type
whitelist, and can disable per-archive Working Memory summaries:

```json
{
  "self": { "enabled": true },
  "peer": { "enabled": false },
  "working_memory": { "enabled": false },
  "memory_types": ["profile", "preferences"]
}
```

When `memory_types` is omitted or `null`, all enabled schemas from
`MemoryTypeRegistry` are allowed, including custom prompt/schema types. When it
is set, extraction is limited to those names for both self and peer writes.
When `working_memory.enabled` is `false`, commit still archives messages and
runs configured memory extraction, but skips the archive summary.

`memory_extraction_config.extraction_context_policy` controls the context that
is recalled before V3 extraction. It is session configuration and can be set at
session creation or patched later:

```json
{
  "memory_extraction_config": {
    "extraction_context_policy": {
      "memory_recall": {
        "enabled": true,
        "mode": "selective",
        "max_queries": 2,
        "max_entries": 4,
        "max_tokens": 3000
      },
      "event_recall": {
        "enabled": true,
        "mode": "selective",
        "max_queries": 2,
        "max_entries": 4,
        "max_tokens": 3000
      },
      "resource_recall": {
        "enabled": true,
        "mode": "selective",
        "max_queries": 2,
        "max_entries": 4,
        "max_tokens": 4000,
        "scopes": ["viking://resources/"],
        "external_providers": [
          {
            "name": "team-knowledge",
            "type": "openviking_http",
            "target_uri": "viking://resources/team/",
            "max_entries": 2,
            "max_tokens": 2000
          }
        ]
      }
    }
  }
}
```

All recall modes currently accept `off` and `selective`. `memory_recall` and
`event_recall` use long-term memory/event indexes from the current OpenViking
service. `resource_recall` can read local OpenViking resources through `scopes`
and can also read configured external resource providers.

External resource providers are read-only. They enrich extraction context so the
model can align new writes with existing team knowledge, but they are not a
source of new memories by themselves; memory operations must still be grounded
in the committed conversation ranges.

Supported external provider types:

| Type | Transport | Required environment |
| --- | --- | --- |
| `openviking_http` | OpenViking HTTP `search/search` plus `content/read` | `OPENVIKING_RESOURCE_RECALL_URL` or `OPENVIKING_RESOURCE_RECALL_BASE_URL` |
| `third_ov` | Same HTTP shape, with `THIRD_OV_*` environment prefix | `THIRD_OV_OPENVIKING_URL` or `THIRD_OV_BASE_URL` |

Optional environment keys for both prefixes:

| Suffix | Purpose |
| --- | --- |
| `_API_PREFIX` | Prefix before `/api/v1`, for example `/openviking` |
| `_API_KEY` or `_TOKEN` | Auth token sent as `X-API-Key`; `_TOKEN` is also sent as `token` |
| `_OPENVIKING` | Provider-specific OpenViking identifier header |
| `_REGION` | Provider-specific region header |
| `_ACCOUNT` | Sent as `X-OpenViking-Account` |
| `_USER` | Sent as `X-OpenViking-User` |
| `_TIMEOUT` or `_TIMEOUT_SECONDS` | HTTP timeout in seconds, default `30` |

Example local setup:

```bash
export OPENVIKING_RESOURCE_RECALL_URL=http://127.0.0.1:30303
export OPENVIKING_RESOURCE_RECALL_API_KEY="$USER_OPENVIKING_API_KEY"
```

Then create or patch a session with the policy above and commit the session.
During prefetch, the extractor adds synthetic `recall_resource` tool results
for matched external resources. The final commit response includes resource
references in recall metadata when the recalled context was used.

## Memory Type Groups

| Group | Types | Target |
| --- | --- | --- |
| User-memory extraction | Enabled registry schemas with `stage: user`, including `cases` | Self and peer, subject to schema policy |
| Case-driven training | `trajectories`, `experiences` | Self only |
| Executable session skills | Optional output of case-driven training | Self only |

Memory schemas default to `stage: user` and `peer_enabled: true`. Set
`peer_enabled: false` for user-stage schemas that should ignore `peer_id` and
`ranges` peer targets and remain under the current user space (for example
`cases`). Execution-derived types are not exposed to the ordinary user-memory
extractor.

`SessionCompressorV3.extract_long_term_memories` is the only public extraction
entry. It trains trajectories, experiences, and optional executable session
skills only when ordinary extraction produces at least one case. An explicit
execution-only `memory_types` policy does not invoke ordinary extraction, so it
cannot create a case and does not trigger training.

## Commit Flow

Implemented in `openviking/session/session.py`:

1. Load the session-level policy from session metadata.
2. Archive the current message batch.
3. Hydrate tool outputs for extraction.
4. If peer memory is enabled, collect safe `message.peer_id` values from the
   archived batch into `allowed_peer_ids`.
5. Start archive summary generation.
6. Remove execution-derived types from the schema whitelist passed to ordinary
   extraction. If enabled user-memory types remain, call
   `SessionCompressorV3.extract_long_term_memories` once with the full archived
   batch, `allow_self_memory`, and `allowed_peer_ids`.
7. V3 applies ordinary memory operations and collects extracted `cases`. When
   at least one case exists, V3 runs streaming training for trajectories and
   experiences and, when enabled, an executable session skill. With no case,
   all three training outputs are skipped.

The current flow does not build separate buckets such as
`self_identity_messages`, `self_experience_messages`,
`peer_user_message_groups`, or `peer_assistant_message_groups`.

## Long-Term Routing

Implemented in `openviking/session/memory/memory_isolation_handler.py`.

`MemoryIsolationHandler.calculate_memory_uris` resolves each extracted operation
independently:

| Operation fields | Result |
| --- | --- |
| No `peer_id`, no `ranges` | Write self if self memory is enabled |
| Safe `peer_id` in `allowed_peer_ids` | Write that peer |
| Unsafe `peer_id` | Skip |
| Safe but unallowed `peer_id` | Skip |
| `ranges` present | Read the message range; no-peer messages route to self, allowed peer messages route to peer |
| Schema has `peer_enabled: false` | Ignore `peer_id` and `ranges` peer targets; write self if self memory is enabled |
| Only disabled targets found | Skip |

The router does not rewrite message roles. A `role=user` message remains user
content, a `role=assistant` message remains assistant content, and tool parts
stay on the message where they were recorded.

## Storage Targets

For current user space `viking://user/<user_id>`:

| Target | Storage space |
| --- | --- |
| Self | `viking://user/<user_id>/...` |
| Peer | `viking://user/<user_id>/peers/<peer_id>/...` |

Peer-only extraction does not initialize self default files. Default self files
are initialized only when `allow_self_memory` is true.

## Practical Invariants

- V3 user-memory extraction sees the full archived batch once.
- Selective recall context is appended before extraction and is visible as
  synthetic recall tool results, not as user conversation content.
- External resource recall is skipped when the provider has no base URL, the
  provider errors, or `max_entries`/`max_tokens` is zero.
- The extractor may emit self and peer operations in the same response.
- Final write targets are decided per operation by the isolation handler.
- Peer writes require safe peer IDs observed in the archived batch.
- `trajectories`, `experiences`, and executable session skills are trained only
  from an extracted case and never write peer memory.

## Deploy And Verify

For local development, install the editable package with test dependencies and
run the focused tests:

```bash
UV_CACHE_DIR=/tmp/openviking-uv-cache uv sync --all-extras
UV_CACHE_DIR=/tmp/openviking-uv-cache uv pip install -e . --force-reinstall
UV_CACHE_DIR=/tmp/openviking-uv-cache uv run pytest \
  tests/server/test_api_sessions_event_tags.py \
  tests/session/memory/test_memory_react_system_prompt.py \
  data-reader/tests/test_server.py
```

Run the server with a local config file that is not committed to git:

```bash
OPENVIKING_CONFIG_FILE=ov_conf/ov.conf \
OPENVIKING_RESOURCE_RECALL_URL=http://127.0.0.1:30303 \
OPENVIKING_RESOURCE_RECALL_API_KEY="$USER_OPENVIKING_API_KEY" \
uv run openviking-server --config ov_conf/ov.conf
```

To inspect results, use Studio for the integrated product view:

```text
http://127.0.0.1:30303/studio
```

Look at the session commit result and the Resources or Memories panels. For
API-level checks, read the session metadata after patching config and confirm
that `memory_extraction_config.extraction_context_policy.resource_recall` keeps
the expected `scopes` and `external_providers`. After a commit, confirm recalled
resources appear in the recall references and that long-term memory writes are
still tied to conversation message ranges.
