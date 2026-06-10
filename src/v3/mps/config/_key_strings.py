from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _KeyValues:
    # 알파벳 순
    
    capital                 : str = "--capital"
    end                     : str = "--end"
    start                   : str = "--start"
    ticker                  : str = "--ticker"
    
    
@dataclass 
class _StringValue:
    pass