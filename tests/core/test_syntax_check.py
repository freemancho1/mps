from __future__ import annotations

import pytest
from typing import Optional

def mock_func(a, b: int, c: Optional[int], d: Optional[int] = None) -> None:
    print(a, b, c, d)


def test_non_params():
    # 에러 발생 케이스(값이 할당되지 않는 모든 인자는 호출 시 할당해야 함)
    with pytest.raises(TypeError):
        mock_func()                         # a, b, c 모두 누락
    with pytest.raises(TypeError):
        mock_func(None, 1)                  # c 누락

    # 정상 케이스 
    mock_func(None, None, None)             # 정의 시 타입이 정의되지 않는 a는 아무거나 와도 됨
    mock_func(None, 1, None)
    mock_func(None, 1, None, 2)
    mock_func(None, 1, None, 2)
    mock_func(c=1, b=1, d=None, a=1)
