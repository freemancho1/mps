import inspect
from time import process_time 
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
        process_time    = lambda st, et: _MF(f"\n전체 테스트 시간: 시작[{st}]~종료[{et}], 총 처리 시간[{et-st}]"),
    ),
    data_load = DictDot(
        title           = lambda d: _MF(f"[{d}] 데이터 로드 중..."),  
        result          = lambda load, bars: _MF(f"[{load}]에서 로드된 데이터 수: {len(bars):,}봉"),
        result_error    = lambda load: _MF(f"[{load}]에서 생성 또는 읽어온 데이터가 없어 프로그램을 종료합니다."),
    ),
    wf = DictDot(       # Walk-Forward 검증
        title           = lambda d: _MF(f"[{d}] Walk-Forward 검증 실행 중..."),
        results         = lambda r: _MF(f"Walk-Forward 결과 = ({len(r)}개 구간):"),
    ),
)

store = DictDot(
    init_info           = lambda in_bdir, bdir: _MF(f"base_dir: [{in_bdir}], self._base_dir: [{bdir}]"),
    fpath               = lambda fpath: _MF(f"store filepath: [{fpath}]"),
    fpath_not_found     = lambda fpath: _MF(f"{fpath} 파일이 존재하지 않습니다."),
    load_bars = DictDot(
        dates           = lambda s, e, m: _MF(f"불러올 대상 날짜: [{s} ~ {e}], Mask: [{m}]"),
        size            = lambda df: _MF(f"불러온 데이터프레임 크기: [{df.shape}]"),
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
        pykrx_error     = lambda df: _MF(f"pykrx 라이브러리가 데이터를 합성하지 못했습니다. 합성결과 데이터 사이즈 = {df.shape}"),
        pykrx_result    = lambda df: _MF(f"pykrx 라이브러리 분봉 생성 결과: 데이터프레임 사이즈 = [{df.shape}]"),
    ),
    
)

wfv = DictDot(          # Walk-Forward Validator
    init                = lambda s: _MF(f"WFV 초기 설정값: buffer_days={s._buffer_days}일, test_days={s._test_days}일, capital={s._capital:,}원"),
    run_info            = lambda tdays, wsize: _MF(
                            f"학습 데이타 일자: [{tdays[0]} ~ {tdays[-1]} ({len(tdays)})], "
                            f"윈도우 크기: {wsize}"
                        ),
    win_bars_info       = lambda s, w: _MF(f"[{s:>3}] 윈도우 기간: {w[0].timestamp.date()} ~ {w[-1].timestamp.date()}, 크기: {len(w)}"),
    err_win_bars        = lambda e: _MF(f"[ERROR] {e}"),
    
)

hs = DictDot(           # HistoricalSimulator
    init                = lambda s: _MF(f"HistoricalSimulator 초기 설정값: capital={s._capital}, lookback_minutes={s._lookback_minutes}"),
    run_info            = lambda b: _MF(f"윈도우 정보: {b[0].timestamp.date()} ~ {b[-1].timestamp.date()}, size={len(b):,}"),
    lookback_under_err  = lambda bars, lb: f"입력된 데이터가 백테스트를 위한 최소 데이터({lb})보다 적습니다.(입력 데이터: {len(bars)}봉)",
    size_check          = lambda bars, buff: _MF(f"사이즈 비교: len(bars) = {len(bars)}, len(buffer) = {len(buff)}"),
    extract_result      = lambda b, d, f: _MF(f"원본 봉 갯 수: {len(b)}, 변경된 데이터프레임 사이즈: {d.shape}, ndarray shape: {f.shape}"),
    normal = DictDot(   # Normalizer
        size_err        = lambda ws, ib: f"롤백 윈도우({ws})보다 데이터가 부족합니다: 입력 봉수={len(ib)}",
    ),
)