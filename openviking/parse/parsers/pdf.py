# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""PDF parser backed by AnyDoc's positioned-image conversion."""

import time
from pathlib import Path
from typing import Optional, Union

from openviking.parse.base import NodeType, ParseResult, ResourceNode, create_parse_result
from openviking.parse.parsers.anydoc import AnyDocParser
from openviking_cli.utils import get_logger
from openviking_cli.utils.config.parser_config import PDFConfig

logger = get_logger(__name__)


class PDFParser(AnyDocParser):
    """Parse PDF text and positioned images through AnyDoc."""

    _SUPPORTED_EXTENSIONS = [".pdf"]
    _PARSER_NAME = "PDFParser"
    _PARSER_VERSION = "4.0"

    def __init__(self, config: Optional[PDFConfig] = None):
        super().__init__(config=config or PDFConfig())

    async def parse(
        self,
        source: Union[str, Path],
        instruction: str = "",
        **kwargs,
    ) -> ParseResult:
        started = time.time()
        path = Path(source)
        try:
            return await super().parse(path, instruction=instruction, **kwargs)
        except Exception as exc:
            logger.error(f"Failed to parse PDF {path}: {exc}", exc_info=True)
            return create_parse_result(
                root=ResourceNode(type=NodeType.ROOT),
                source_path=str(path),
                source_format="pdf",
                parser_name=self._PARSER_NAME,
                parse_time=time.time() - started,
                warnings=[f"Failed to parse PDF: {exc}"],
            )
