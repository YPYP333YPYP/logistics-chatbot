from sqlalchemy import Column, String, DateTime, Integer, Boolean
from db.database import Base


class LogisticsShipment(Base):
    __tablename__ = 'logistics_shipments'

    id = Column(Integer, primary_key=True, autoincrement=True)

    lss_id = Column(String(50))
    lss_name = Column(String(100))

    # 상태 정보
    current_status = Column(String(100))
    current_location = Column(String(100))
    ts_yn = Column(Boolean)

    # 참조 번호
    sr_no = Column(String(50))
    hbl_no = Column(String(50))
    mbl_no = Column(String(50))
    uld_no = Column(String(50))
    uld_cd = Column(String(10))

    # 서비스 정보
    service_nm = Column(String(50))
    d_s_t = Column(String(50))
    service_term = Column(String(50))
    incoterms = Column(String(20))

    # 출발지/도착지 정보
    from_site_id = Column(String(20))
    from_site_name = Column(String(100))
    to_site_id = Column(String(20))
    to_site_name = Column(String(100))

    # 화주/수하인 정보
    shipper_id = Column(String(20))
    shipper_name = Column(String(100))
    consignee_id = Column(String(20))
    consignee_nm = Column(String(100))

    # 공장 정보
    from_plant_cd = Column(String(20))
    from_plant_nm = Column(String(100))
    to_plant_cd = Column(String(20))
    to_plant_nm = Column(String(100))

    # 경로 정보
    delivery_lane = Column(String(100))
    pol_nation = Column(String(100))
    pol_loc_id = Column(String(20))
    pol_nm = Column(String(100))
    pod_nation = Column(String(100))
    pod_loc_id = Column(String(20))
    pod_nm = Column(String(100))
    fd_nation_nm = Column(String(100))
    fd_loc_id = Column(String(20))
    fd_nm = Column(String(100))

    # 운송사 정보
    lsp_id = Column(String(20))
    lsp_nm = Column(String(100))
    carrier_code = Column(String(20))
    carrier_nm = Column(String(100))

    # POL(선적항) 정보
    pol_vessel_flight_nm = Column(String(100))
    pol_vessel_flight_no = Column(String(50))
    pol_initial_etd = Column(DateTime)
    pol_as_is_etd = Column(DateTime)
    pol_atd = Column(DateTime)
    pol_etd_week = Column(String(10))
    pol_atd_week = Column(String(10))
    pol_etd_month = Column(String(10))

    # POL 상태
    pol_ontime_status = Column(String(20))
    pol_aging = Column(String(20))
    pol_aging_period = Column(Integer)
    # POD(도착항) 정보
    pod_vessel_flight_nm = Column(String(100))
    pod_vessel_flight_no = Column(String(50))
    pod_initial_eta = Column(DateTime)
    pod_as_is_eta = Column(DateTime)
    pod_region_cd = Column(String(50))

    # 리드타임 계산 정보
    lt_day_pol_atd_pod_initial_eta = Column(Integer)
    lt_day_pol_atd_pod_as_is_eta = Column(Integer)

    # 컨테이너 정보
    cntr_duplicate = Column(String(1))
