# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""PDF conversion contract shared by AnyDoc and OpenViking."""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openviking.parse.base import NodeType, ResourceNode, create_parse_result
from openviking.parse.parsers import anydoc
from openviking.parse.parsers.pdf import PDFParser


@pytest.mark.asyncio
async def test_pdf_parser_preserves_anydoc_image_binding_and_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    conversion = SimpleNamespace(
        markdown="# Before\n\n![Image: Im0](p1_i1.png)\n\n# After\n",
        images=[
            SimpleNamespace(
                filename="p1_i1.png",
                page=1,
                bbox=(72.0, 400.0, 240.0, 144.0),
                width=557,
                height=334,
                data=b"png-data",
                warnings=["unsupported_color_space"],
            )
        ],
        page_count=2,
        pages_needing_ocr=[2],
        ocr_reasons_by_page=[SimpleNamespace(page=2, reasons=["no_text", "image_coverage"])],
        pages_with_tables=[1],
        pages_with_columns=[],
        is_complex_layout=True,
        has_encoding_issues=True,
    )
    monkeypatch.setitem(
        sys.modules,
        "anydoc",
        SimpleNamespace(
            format_from_bytes=lambda _data: "pdf",
            format_from_extension=lambda _extension: "pdf",
            pdf_to_markdown_with_images=lambda _data: conversion,
        ),
    )
    monkeypatch.setattr(anydoc, "is_valid_image", lambda *_args: True)
    monkeypatch.setattr(anydoc, "version", lambda _package: "0.1.8")

    media_dir = tmp_path / "media"
    saved: list[tuple[str, bytes, str, str]] = []

    def save_image(
        resource_name: str,
        data: bytes,
        *,
        filename: str,
        extension: str,
    ) -> Path:
        saved.append((resource_name, data, filename, extension))
        return media_dir / resource_name / "images" / f"{filename}{extension}"

    storage = SimpleNamespace(media_dir=media_dir, save_image=save_image)
    monkeypatch.setattr("openviking_cli.utils.storage.get_storage", lambda: storage)

    parser = PDFParser()
    markdown_call: dict[str, Any] = {}

    async def parse_markdown(content: str, **kwargs: Any):
        markdown_call.update(content=content, **kwargs)
        return create_parse_result(
            root=ResourceNode(type=NodeType.ROOT),
            source_path=kwargs["source_path"],
            source_format="markdown",
            parser_name="MarkdownParser",
        )

    parser._markdown_parser.parse_content = parse_markdown
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF fixture")
    caller_media_dir = tmp_path / "caller-media"

    result = await parser.parse(
        source,
        resource_name="report",
        allowed_media_dirs=[caller_media_dir],
        split_content=False,
    )

    assert markdown_call["content"] == (
        "# Before\n\n![Image: Im0](report/images/p1_i1.png)\n\n# After\n"
    )
    assert markdown_call["allowed_media_dirs"] == [caller_media_dir, media_dir]
    assert markdown_call["split_content"] is False
    assert saved == [("report", b"png-data", "p1_i1", ".png")]
    assert result.parser_name == "PDFParser"
    assert result.source_format == "pdf"
    assert result.warnings == [
        "PDF page 1 image rendering warning: unsupported_color_space",
        "PDF page 2 requires OCR: no_text, image_coverage",
        "PDF contains broken font encodings; extracted text may be garbled",
    ]
    assert result.meta == {
        "library": "firecrawl-anydoc",
        "library_version": "0.1.8",
        "detected_format": "pdf",
        "assets_referenced": 1,
        "images_extracted": 1,
        "intermediate_markdown_length": 58,
        "total_pages": 2,
        "pages_needing_ocr": [2],
        "ocr_reasons_by_page": [{"page": 2, "reasons": ["no_text", "image_coverage"]}],
        "pages_with_tables": [1],
        "pages_with_columns": [],
        "is_complex_layout": True,
        "has_encoding_issues": True,
    }
