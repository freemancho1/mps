from mps.sys.free import DictDot


run = DictDot(
    sys = DictDot(
        summary         = lambda v: (
                            f"{'='*60}\n"
                            f"  시스템 정보\n"
                            f"    - Phase: {v.phase}\n"
                            f"    - seed: {v.seed}\n"
                            f"    - 거래 시간: {v.market_open_time} ~ {v.market_close_time} ({v.timezone})\n"
                            f"    - 일일 거래 분(봉) 수: {v.minutes_per_day}분/일\n"
                            f"    - 룩백 윈도우 수: {v.lookback_minutes}분(개)\n"
                            f"    - 강제 종료 시간: {v.force_close_minutes_before}분\n"
                            f"{'='*60}\n"            
                        ),
    ),
    info = DictDot(
        title           = "MPS Phase-1 백테스트",
        ticker          = "종목코드 (기본값: 삼성전자(005930))",
        start           = "시작일 YYYY-MM-DD",
        end             = "종료일 YYYY-MM-DD",
        capital         = "초기 자본",
        summary         = lambda args: (
                            f"\n{'='*60}\n"
                            f"  {args['title']}\n"
                            f"    - 종목: {args['ticker']}\n"
                            f"    - 기간: {args['start']} ~ {args['end']}\n"
                            f"    - 초기자본: {args['capital']:,.0f}원\n"
                            f"    - 보수적 왕복 비용: {args['roundtrip_cost']:.2%}\n"
                            f"{'='*60}\n"            
                        ),
    ),
    data_load           = "데이터 로드 중...",
)


store = DictDot(
    init_info           = lambda in_bdir, bdir: (
                            f"  = base_dir: [{in_bdir}], self._base_dir: [{bdir}]"
                        ),
    fpath               = lambda fpath: f"  = store filepath: [{fpath}]",
    fpath_not_found     = lambda fpath: f"  = {fpath} 파일이 존재하지 않습니다.",
    load_bars = DictDot(
        dates           = lambda s, e, m: f"  = 불러올 대상 날짜: [{s} ~ {e}], Mask: [{m}]",
        size            = lambda df: f"  = 불러온 데이터프레임 크기: [{df.size()}]",
        return_size     = lambda l: f"  = 리턴할 데이터 크기: [{len(l)}]",
    ),
)

loader = DictDot(
    curr_store          = lambda store, self_store: f"  = store: [{store}], self._store: [{self_store}]",
    process_dt          = lambda s, e: f"  = 처리 일시: {s} ~ {e}",
    fetch = DictDot(
        data_size       = lambda l: f"  = 패치 데이터 크기: [{len(l)}]",  
        from_kis        = "  = KIS REST API를 이용해 데이터 수집",
        from_synthetic  = "  = pykrx 라이브러리를 이용해 데이터 합성",
        kis_not_implemented = "[ERROR] KIS REST API분봉 수집 함수는 KIS_APP_KEY 설정 후 구현 예정",
        pykrx_info      = lambda s, e, t: f"  = pykrx 데이터 생성 기간: {s} ~ {e}, 종목 코드: [{t}]",
        pykrx_error     = "  = pykrx 라이브러리가 데이터를 합성하지 못했습니다.",
        pykrx_result    = lambda df: f"  = pykrx 라이브러리 분봉 생성 결과: 데이터프레임 사이즈 = [{df.size()}]",
    ),
    
)