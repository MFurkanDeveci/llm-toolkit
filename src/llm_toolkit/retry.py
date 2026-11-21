"""Geçici hatalarda üstel geri çekilme (exponential backoff) ile tekrar deneme."""

import logging
import random
import time
from collections.abc import Callable

from llm_toolkit.errors import TransientLLMError

logger = logging.getLogger(__name__)


def with_retries[T](
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """fn'i çalıştırır; TransientLLMError gelirse 1, 2, 4... saniye bekleyerek tekrar dener.

    sleep parametresi testlerde gerçekten beklememek için dışarıdan verilebilir.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except TransientLLMError as exc:
            if attempt == max_attempts:
                raise
            # Jitter: binlerce istemci aynı anda tekrar denemesin diye süreyi biraz rastgele yap
            delay = base_delay * 2 ** (attempt - 1) * random.uniform(0.5, 1.5)
            logger.warning(
                "Geçici hata (%s). %d/%d. deneme, %.1f sn sonra tekrar.",
                exc,
                attempt,
                max_attempts,
                delay,
            )
            sleep(delay)
    raise AssertionError("buraya ulaşılmamalı")
