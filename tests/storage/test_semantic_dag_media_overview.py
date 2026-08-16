# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from openviking.storage.queuefs.directory_semantic import DirectorySemanticTask


def _direct_overview(filename, summary):
    uri = "viking://resources/media"
    return DirectorySemanticTask._select_direct_media_overview(
        (f"{uri}/{filename}",),
        (),
        [{"name": filename, "summary": summary}],
    )


def test_valid_single_media_summary_is_direct_overview():
    for filename in ("demo.mp4", "recording.mp3"):
        summary = f"# Demo\n\nDense brief.\n\n### {filename}\n\nDetailed event."
        assert _direct_overview(filename, summary) == summary


def test_malformed_media_summary_is_not_direct():
    cases = [
        ("demo.mp4", ""),
        ("demo.mp4", "# Demo\n\nBrief without filename heading"),
    ]

    for filename, summary in cases:
        assert _direct_overview(filename, summary) is None
