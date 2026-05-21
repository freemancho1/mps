import inspect 
from mps.sys.free import DictDot


def _MF(msg: str) -> str:
    frame = inspect.stack()[2]
    location = f"{frame.filename}:[{frame.lineno}]"
    
    WIDTH = 25
    if len(location) > WIDTH:
        location = location[-WIDTH:]
    else:
        location = location.rjust(WIDTH)
        
    return f"{location} = {msg}"

run = DictDot(
    info = DictDot(
        title           = "MPS Phase-1 백테스트",
        ticker          = "종목코드 (기본값: 삼성전자(005930))",
        start           = "시작일 YYYYMMDD",
        end             = "종료일 YYYYMMDD",
        capital         = "초기 자본",
        test_days       = "테스트 거래일 (기본값: 10)",
        summary         = lambda a, c, m: _MF(
                            f"\n{'='*60}\n"
                            f"  {m.run.info.title}\n"
                            f"    - 종목: {a.ticker}\n"
                            f"    - 기간: {a.start} ~ {a.end}\n"
                            f"    - 초기자본: {a.capital:,.0f}원\n"
                            f"    - 테스트 일자: {a.test_days}일\n"
                            f"{'='*60}\n"            
                            f"  시스템 정보\n"
                            f"    - Phase: {c.sys.phase}\n"
                            f"    - Seed: {c.sys.seed}\n"
                            f"    - 거래 시간: {c.sys.market_open_time} ~ {c.sys.market_close_time} ({c.sys.timezone})\n"
                            f"    - 일일 거래 분(봉) 수: {c.sys.minutes_per_day}분/일\n"
                            f"    - 룩백 윈도우 수: {c.sys.lookback_minutes}분(개)\n"
                            f"    - 강제 종료 시간: {c.sys.force_close_minutes_before}분\n"
                            f"{'='*60}\n"            
                        ),
    ),
    data_load = DictDot(
        title           = lambda d: _MF(f"[{d}] 데이터 로드 중..."),  
        result          = lambda load, bars: _MF(f"[{load}]에서 로드된 데이터 수: {len(bars):,}봉"),
        result_error    = lambda load: _MF(f"[{load}]에서 생성 또는 읽어온 데이터가 없어 프로그램을 종료합니다."),
    ),
    wf = DictDot(       # Walk-Forward 검증
        title           = lambda d: _MF(f"[{d}] Walk-Forward 검증 실행 중..."),
    ),
)

store = DictDot(
    init_info           = lambda in_bdir, bdir: _MF(f"base_dir: [{in_bdir}], self._base_dir: [{bdir}]"),
    fpath               = lambda fpath: _MF(f"store filepath: [{fpath}]"),
    fpath_not_found     = lambda fpath: _MF(f"{fpath} 파일이 존재하지 않습니다."),
    load_bars = DictDot(
        dates           = lambda s, e, m: _MF(f"불러올 대상 날짜: [{s} ~ {e}], Mask: [{m}]"),
        size            = lambda df: _MF(f"불러온 데이터프레임 크기: [{df.size:,}]"),
        return_size     = lambda l: _MF(f"리턴할 데이터 크기: [{len(l):,}]"),
    ),
)

loader = DictDot(
    curr_store          = lambda store, self_store: _MF(f"store: [{store}], self._store: [{self_store}]"),
    process_dt          = lambda s, e: _MF(f"처리 일시: {s} ~ {e}"),
    fetch = DictDot(
        data_size       = lambda l: _MF(f"불러온 데이터 크기: [{len(l):,}]"),  
        from_kis        = lambda: "KIS REST API를 이용해 데이터 수집",
        from_synthetic  = lambda: "pykrx 라이브러리를 이용해 데이터 합성",
        kis_not_implemented = "[ERROR] KIS REST API분봉 수집 함수는 KIS_APP_KEY 설정 후 구현 예정",
        pykrx_info      = lambda s, e, t: _MF(f"pykrx 데이터 생성 기간: {s} ~ {e}, 종목 코드: [{t}]"),
        pykrx_error     = lambda df: _MF(f"pykrx 라이브러리가 데이터를 합성하지 못했습니다. 합성결과 데이터 사이즈 = {df.size:,}"),
        pykrx_result    = lambda df: _MF(f"pykrx 라이브러리 분봉 생성 결과: 데이터프레임 사이즈 = [{df.size:,}]"),
    ),
    
)

wfv = DictDot(          # Walk-Forward Validator

)