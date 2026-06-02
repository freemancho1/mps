import os 
import platform 
import matplotlib.pyplot as plt 
import matplotlib.font_manager as fm
import mplfinance as mpf 

from dataclasses import dataclass, field 
from typing import Any, Optional


@dataclass 
class _FontConfig:
    font_paths: dict[str, list[str]] = field(default_factory=lambda: {
        "Windows": [
            "C:\\Windows\\Fonts\\malgun.ttf",
            "C:\\Windows\\Fonts\\gulim.ttc",
        ],
        "Darwin": [
            "Library/Fonts/AppleGothic.ttf",
            "System/Library/Fonts/AppleGothic.ttf",
        ],
        "Linux": [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",            
        ],
    })
    

def _configure_warnings():
    """ 
    애플리케이션 실행 중 특정 경고 비활성화 
    
    - 필요시 경고 필터링 로직을 이곳에 추가
      → 예: warnings.filterwarnings('ignore', category=FutureWarning)
    """
    pass


def _configure_matplotlib() -> Optional[dict[str, Any]]:
    """ 
    실행 환경에 맞춰 Matplotlob의 한글 폰트 및 기본 스타일 설정
    
    - mplfinance에서 사용하는 기본 스타일 지정후 리턴(필요 시 사용하면 됨.)
    """
    system_os = platform.system()
    font_config = _FontConfig()
    font_paths = font_config.font_paths.get(system_os, [])
    
    font_path = next((fpath for fpath in font_paths if os.path.exists(fpath)), None)
    if not font_path:
        print("한글 폰트가 존재하지 않습니다.")
        return None
    
    font_name = fm.FontProperties(fname=font_path).get_name()
    rc = {
        'font.family': font_name,       # 한글 폰트 지정
        'axes.unicode_minus': False,    # 차트에서 '-' 표현 방법 설정('-'로 표시)
        'axes.grid': False,             # 차트에서 그리드 표시 여부(표시 안함)
        'figure.figsize': (12, 6)       # 기본 차트 크기
    }
    plt.rcParams.update(rc)
    print(f"정상적으로 한글 및 차트 설정이 완료되었습니다. (한글폰트: {font_name})")
    
    return mpf.make_mpf_style(base_mpf_style="charles", rc=rc)
    

def bootstrap() -> Optional[dict[str, Any]]:
    _configure_warnings()
    mpl_style = _configure_matplotlib()
    return mpl_style