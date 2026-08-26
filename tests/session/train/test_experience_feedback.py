# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

from openviking.session.memory.dataclass import MemoryFile
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.session.train.components.experience_feedback import (
    parse_experience_effects,
    record_experience_feedback_stats,
)
from openviking.session.train.domain import Trajectory


class FakeVikingFS:
    def __init__(self, files: dict[str, str]):
        self.files = dict(files)
        self.writes = []

    async def read_file(self, uri: str, ctx=None):
        return self.files[uri]

    async def write_file(self, uri: str, content: str, ctx=None):
        self.files[uri] = content
        self.writes.append((uri, content, ctx))


def _experience_file(uri: str, *, stats=None, status: str = "promoted") -> str:
    extra_fields = {
        "memory_type": "experiences",
        "experience_name": "payment_guard",
        "status": status,
        "trigger_code": 'def should_trigger(ctx):\n    return ctx.get("candidate_tool") == "book"\n',
    }
    if stats is not None:
        extra_fields["feedback_stats"] = stats
    return MemoryFileUtils.write(
        MemoryFile(
            uri=uri,
            content="## Failure Pattern\n- x",
            memory_type="experiences",
            extra_fields=extra_fields,
        )
    )


def test_parse_experience_effects_accepts_compact_json_string():
    assert parse_experience_effects(
        '{"positive_ids":["E1"],"negative_ids":[],"weak_ids":["E2"]}'
    ) == {
        "positive_ids": {"E1"},
        "negative_ids": set(),
        "weak_ids": {"E2"},
    }


async def test_record_experience_feedback_stats_updates_hidden_metadata():
    exp_uri = "viking://user/u/memories/experiences/payment_guard.md"
    traj_uri = "viking://user/u/memories/trajectories/t1.md"
    fs = FakeVikingFS({exp_uri: _experience_file(exp_uri)})

    result = await record_experience_feedback_stats(
        trajectories=[
            Trajectory(
                name="t1",
                uri=traj_uri,
                content="# t1",
                outcome="failure",
                retrieval_anchor="Stage: before_write",
                metadata={
                    "case_name": "case_1",
                    "experience_effects": '{"positive_ids":[],"negative_ids":["E1"],"weak_ids":[]}',
                },
            )
        ],
        injected_reminders=[
            {
                "id": "E1",
                "experience_name": "payment_guard",
                "experience_uri": exp_uri,
                "triggered_before_tool": "book_reservation",
            }
        ],
        viking_fs=fs,
        ctx=None,
        observed_at="2026-07-06T00:00:00+00:00",
    )

    assert result.updated_uris == [exp_uri]
    written = MemoryFileUtils.read(fs.files[exp_uri], uri=exp_uri)
    stats = written.extra_fields["feedback_stats"]
    assert stats["schema_version"] == 1
    assert stats["injected_count"] == 1
    assert stats["negative_count"] == 1
    assert stats["positive_count"] == 0
    assert stats["weak_count"] == 0
    assert stats["neutral_count"] == 0
    assert set(stats) == {
        "schema_version",
        "injected_count",
        "positive_count",
        "negative_count",
        "weak_count",
        "neutral_count",
    }


async def test_record_experience_feedback_stats_stores_aggregate_counts_only():
    exp_uri = "viking://user/u/memories/experiences/payment_guard.md"
    traj_uri = "viking://user/u/memories/trajectories/t1.md"
    fs = FakeVikingFS({exp_uri: _experience_file(exp_uri)})
    trajectory = Trajectory(
        name="t1",
        uri=traj_uri,
        content="# t1",
        outcome="success",
        retrieval_anchor="Stage: final",
        metadata={"experience_effects": '{"positive_ids":["E1"],"negative_ids":[],"weak_ids":[]}'},
    )
    reminder = {"id": "E1", "experience_uri": exp_uri}

    first = await record_experience_feedback_stats(
        trajectories=[trajectory],
        injected_reminders=[reminder],
        viking_fs=fs,
        ctx=None,
        observed_at="2026-07-06T00:00:00+00:00",
    )
    assert first.updated_uris == [exp_uri]
    stats = MemoryFileUtils.read(fs.files[exp_uri], uri=exp_uri).extra_fields["feedback_stats"]
    assert stats == {
        "schema_version": 1,
        "injected_count": 1,
        "positive_count": 1,
        "negative_count": 0,
        "weak_count": 0,
        "neutral_count": 0,
    }


async def test_reproducible_negative_transfer_degrades_promoted_experience():
    exp_uri = "viking://user/u/memories/experiences/payment_guard.md"
    fs = FakeVikingFS({exp_uri: _experience_file(exp_uri)})
    reminder = {"id": "E1", "experience_uri": exp_uri}

    def negative_trajectory(name: str) -> Trajectory:
        return Trajectory(
            name=name,
            uri=f"viking://user/u/memories/trajectories/{name}.md",
            content=f"# {name}",
            outcome="failure",
            retrieval_anchor="Stage: final",
            metadata={
                "experience_effects": (
                    '{"positive_ids":[],"negative_ids":["E1"],"weak_ids":[]}'
                )
            },
        )

    await record_experience_feedback_stats(
        trajectories=[negative_trajectory("t1")],
        injected_reminders=[reminder],
        viking_fs=fs,
        ctx=None,
    )
    after_first = MemoryFileUtils.read(fs.files[exp_uri], uri=exp_uri).extra_fields
    assert after_first["status"] == "promoted"
    assert after_first["negative_source_count"] == 1

    # Replaying one trajectory must not count as independent evidence.
    await record_experience_feedback_stats(
        trajectories=[negative_trajectory("t1")],
        injected_reminders=[reminder],
        viking_fs=fs,
        ctx=None,
    )
    after_replay = MemoryFileUtils.read(fs.files[exp_uri], uri=exp_uri).extra_fields
    assert after_replay["status"] == "promoted"
    assert after_replay["negative_source_count"] == 1

    await record_experience_feedback_stats(
        trajectories=[negative_trajectory("t2")],
        injected_reminders=[reminder],
        viking_fs=fs,
        ctx=None,
    )
    after_second = MemoryFileUtils.read(fs.files[exp_uri], uri=exp_uri).extra_fields
    assert after_second["status"] == "degraded"
    assert after_second["negative_source_count"] == 2
    assert after_second["promotion_reason"] == "reproducible_negative_transfer"
