from sqlalchemy import Column, Integer, String, Float, Date, Boolean, ForeignKey, DateTime
from db.database import Base

class Shipping(Base):
    __tablename__ = 'shipping'

    # 기본 식별자 및 상태 정보
    id = Column(Integer, primary_key=True)
    lss_id = Column(String(50))
    lss_name = Column(String(100))
    current_status = Column(String(50))
    current_location = Column(String(100))
    ts_yn = Column(String(5))

    # 문서 번호
    sr_no = Column(String(50))
    hbl_no = Column(String(50))
    mbl_no = Column(String(50))

    # ULD 정보
    uld_no = Column(String(50))
    uld_cd = Column(String(50))

    # 서비스 정보
    service_nm = Column(String(100))
    d_s_t = Column(String(50))
    service_term = Column(String(50))
    incoterms = Column(String(20))

    # 발송지 정보
    from_site_id = Column(String(50))
    from_site_name = Column(String(100))
    to_site_id = Column(String(50))
    to_site_name = Column(String(100))

    # 화주 및 수하인 정보
    shipper_id = Column(String(50))
    shipper_name = Column(String(100))
    consignee_id = Column(String(50))
    consignee_nm = Column(String(100))

    # 플랜트 정보
    from_plant_cd = Column(String(50))
    from_plant_nm = Column(String(100))
    to_plant_cd = Column(String(50))
    to_plant_nm = Column(String(100))
    delivery_lane = Column(String(100))

    # POL (Port of Loading) 정보
    pol_nation = Column(String(50))
    pol_loc_id = Column(String(50))
    pol_nm = Column(String(100))

    # POD (Port of Discharge) 정보
    pod_nation = Column(String(50))
    pod_loc_id = Column(String(50))
    pod_nm = Column(String(100))

    # Final Destination 정보
    fd_nation_nm = Column(String(50))
    fd_loc_id = Column(String(50))
    fd_nm = Column(String(100))

    # Transshipment 정보 (1st ~ 5th)
    # 1st Transshipment
    ts1_nation = Column(String(50))
    ts1_loc_id = Column(String(50))
    ts1_loc_nm = Column(String(100))

    # 2nd Transshipment
    ts2_nation = Column(String(50))
    ts2_loc_id = Column(String(50))
    ts2_loc_nm = Column(String(100))

    # 3rd Transshipment
    ts3_nation = Column(String(50))
    ts3_loc_id = Column(String(50))
    ts3_loc_nm = Column(String(100))

    # 4th Transshipment
    ts4_nation = Column(String(50))
    ts4_loc_id = Column(String(50))
    ts4_loc_nm = Column(String(100))

    # 5th Transshipment
    ts5_nation = Column(String(50))
    ts5_loc_id = Column(String(50))
    ts5_loc_nm = Column(String(100))

    # LSP 및 운송사 정보
    lsp_id = Column(String(50))
    lsp_nm = Column(String(100))
    carrier_code = Column(String(50))
    carrier_nm = Column(String(100))

    # 선박 정보
    pre_vessel_name = Column(String(100))
    pre_vessel_voy_no = Column(String(50))

    # POL 상세 정보
    pol_detail_loc_id = Column(String(50))
    pol_detail_loc_nm = Column(String(100))
    pol_initial_etd = Column(DateTime)
    pol_as_is_etd = Column(DateTime)
    pol_atd = Column(DateTime)
    pol_etd_week = Column(String(10))
    pol_atd_week = Column(String(10))
    pol_etd_month = Column(String(10))
    pol_vessel_flight_nm = Column(String(100))
    pol_vessel_flight_no = Column(String(50))

    # POL 시간 계산 필드
    pol_plan_vs_plan = Column(Integer)  # POL ETD - Initial ETD
    pol_plan_vs_actual = Column(Integer)  # POL ATD - Initial ETD
    pol_plan_vs_actual_diff = Column(Integer)  # POL ATD - ETD
    pol_plan_vs_actual_wk = Column(Integer)  # POL ATD - Initial ETD (주 단위)
    pol_plan_vs_actual_month = Column(Integer)  # POL ATD - Initial ETD (월 단위)
    pol_ontime_delay_early = Column(String(20))  # POL 적기/지연/조기 출발
    pol_aging_period = Column(Integer)  # POL Aging 기간
    pol_aging = Column(String(20))  # POL Aging

    # 1st Transshipment 상세 정보
    ts1_detail_loc_id = Column(String(50))
    ts1_detail_loc_nm = Column(String(100))
    ts1_initial_eta = Column(DateTime)
    ts1_as_is_eta = Column(DateTime)
    ts1_ata = Column(DateTime)
    ts1_initial_etd = Column(DateTime)
    ts1_as_is_etd = Column(DateTime)
    ts1_atd = Column(DateTime)
    ts1_etd_week = Column(String(10))
    ts1_atd_week = Column(String(10))
    ts1_vessel_flight_nm = Column(String(100))
    ts1_vessel_flight_no = Column(String(50))

    # 1st Transshipment 시간 계산 필드
    ts1_plan_vs_plan_eta = Column(Integer)  # 1st T/S ETA - Initial ETA
    ts1_plan_vs_actual_eta = Column(Integer)  # 1st T/S ATA - Initial ETA
    ts1_plan_vs_actual_eta_diff = Column(Integer)  # 1st T/S ATA - ETA
    ts1_plan_vs_plan_etd = Column(Integer)  # 1st T/S ETD - Initial ETD
    ts1_plan_vs_actual_etd = Column(Integer)  # 1st T/S ATD - Initial ETD
    ts1_plan_vs_actual_etd_diff = Column(Integer)  # 1st T/S ATD - ETD
    ts1_aging_period = Column(Integer)  # 1st T/S Aging 기간
    ts1_aging = Column(String(20))  # 1st T/S Aging

    # 2nd ~ 5th Transshipment의 유사한 필드는 생략됩니다 - 실제로는 위와 유사한 패턴으로 추가

    # POD 도착 정보
    pod_detail_loc_id = Column(String(50))
    pod_detail_loc_nm = Column(String(100))
    pod_initial_eta = Column(DateTime)
    pod_as_is_eta = Column(DateTime)
    pod_ata = Column(DateTime)
    pod_vessel_flight_nm = Column(String(100))
    pod_vessel_flight_no = Column(String(50))
    pod_eta_week = Column(String(10))
    pod_ata_week = Column(String(10))
    pod_initial_etd = Column(DateTime)
    pod_as_is_etd = Column(DateTime)
    pod_atd = Column(DateTime)
    region_cd = Column(String(20))

    # POD 시간 계산 필드
    pod_plan_vs_plan = Column(Integer)  # POD ETA - ETA 기준일
    pod_plan_vs_actual = Column(Integer)  # POD ATA - ETA 기준일
    pod_plan_vs_actual_diff = Column(Integer)  # POD ATA - ETA
    pod_plan_vs_plan_wk = Column(Integer)  # POD ETA - ETA 기준일 (주 단위)
    pod_plan_vs_plan_month = Column(Integer)  # POD ETA - ETA 기준일 (월 단위)
    pod_ontime_delay_early = Column(String(20))  # POD 적기/지연/조기 도착

    # 리드타임 정보
    lt_day = Column(Integer)  # LT day (POL ATD ~ POD ATA)
    lt_wk = Column(Float)  # LT WK (POL ATD ~ POD ATA)
    pod_plan_vs_actual_etd_diff = Column(Integer)  # POD ATD - ETD
    pod_aging_period = Column(Integer)  # POD Aging 기간
    pod_aging = Column(String(20))  # POD Aging

    # 컨테이너 및 기타 정보
    total = Column(Integer)
    rep_carrier = Column(String(100))
    cntr_duplicate = Column(String(5))  # CNTR 중복 O/X
    hbl_mbl_cntr = Column(String(100))  # HBL+MBL+CNTR
    mbl_cntr = Column(String(100))  # MBL+CNTR
    mbl_cntr_duplicate = Column(String(5))  # MBL+CNTR 중복 O/X
    arrival_cy_or_dr = Column(String(10))  # 도착지 CY or DR

    # 타켓 및 기준일 정보
    target_eta = Column(DateTime)  # Target ETA (final)
    eta_base_date = Column(DateTime)  # ETA 기준일

    # 추가 계산 필드
    pod_plan_vs_plan_initial = Column(Integer)  # POD ETA - Initial ETA
    pod_plan_vs_actual_initial = Column(Integer)  # POD ATA - Initial ETA
    pod_plan_vs_actual_diff_initial = Column(Integer)  # POD ATA - ETA
    pod_plan_vs_plan_wk_initial = Column(Integer)  # POD ETA - Initial ETA (주 단위)
    pod_plan_vs_plan_month_initial = Column(Integer)  # POD ETA - Initial ETA (월 단위)
    pod_ontime_delay_early_initial = Column(String(20))  # Initial ETA POD 적기/지연/조기 도착

    # 추가 리드타임 정보
    lt_day_initial = Column(Integer)  # LT day (POL ATD ~ POD Initial ETA)
    lt_day_as_is = Column(Integer)  # LT day (POL ATD ~ POD AS-IS ETA)
    lt_day_target = Column(Integer)  # LT day (POL ATD ~ POD Target ETA)
    lt_day_base = Column(Integer)  # LT day (POL ATD ~ POD ETA 기준일)

    # ETA 기준일 변경 정보
    previous_eta_base_date = Column(DateTime)  # 이전 'ETA 기준일'
    eta_base_date_diff = Column(Integer)  # 현 'ETA 기준일 - 이전 'ETA 기준일

    # Transshipment Aging 정보
    ts1_aging_days_plan = Column(Integer)  # 1st TS Aging days (ETA ~ ETD)
    ts2_aging_days_plan = Column(Integer)  # 2nd TS Aging days (ETA ~ ETD)
    ts3_aging_days_plan = Column(Integer)  # 3rd TS Aging days (ETA ~ ETD)
    ts4_aging_days_plan = Column(Integer)  # 4th TS Aging days (ETA ~ ETD)
    ts5_aging_days_plan = Column(Integer)  # 5th TS Aging days (ETA ~ ETD)

    ts1_aging_days_actual = Column(Integer)  # 1st TS Aging days (ATA ~ ETD)
    ts2_aging_days_actual = Column(Integer)  # 2nd TS Aging days (ATA ~ ETD)
    ts3_aging_days_actual = Column(Integer)  # 3rd TS Aging days (ATA ~ ETD)
    ts4_aging_days_actual = Column(Integer)  # 4th TS Aging days (ATA ~ ETD)
    ts5_aging_days_actual = Column(Integer)  # 5th TS Aging days (ATA ~ ETD)

    # TS 확인 필요 여부
    ts1_check_required = Column(String(5))  # 1st TS check required Y/N
    ts2_check_required = Column(String(5))  # 2nd TS check required Y/N
    ts3_check_required = Column(String(5))  # 3rd TS check required Y/N
    ts4_check_required = Column(String(5))  # 4th TS check required Y/N
    ts5_check_required = Column(String(5))  # 5th TS check required Y/N

    def __repr__(self):
        return f"<ShippingData(lss_id='{self.lss_id}', hbl_no='{self.hbl_no}', mbl_no='{self.mbl_no}')>"