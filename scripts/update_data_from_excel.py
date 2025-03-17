from typing import List
from datetime import datetime
from pyxlsb import open_workbook
from db.database import async_get_db
from model.logistics_shipment import LogisticsShipment


async def get_session():
    session_generator = async_get_db()
    session = await anext(session_generator)
    try:
        yield session
    finally:
        try:
            await session_generator.aclose()
        except:
            pass


async def read_excel_and_save_to_db(excel_file_path: str, sheet_name=0) -> List[LogisticsShipment]:
    """
    엑셀 파일을 읽어서 LogisticsShipment 모델을 통해 데이터베이스에 저장합니다.

    Args:
        excel_file_path (str): 엑셀 파일 경로
        sheet_name (str or int, optional): 읽을 시트 이름 또는 인덱스. 기본값은 0(첫 번째 시트).

    Returns:
        List[LogisticsShipment]: 저장된 레코드 리스트
    """

    shipment_list = []
    with open_workbook(excel_file_path) as wb:
        with wb.get_sheet("Raw") as sheet:
            data = [row for row in sheet.rows()]

    # 헤더 행(3행)과 데이터 행 분리
    header_row = data[2]  # 3번째 행(인덱스 2)이 컬럼명

    # Cell 객체에서 값(v) 추출
    headers = [cell.v if hasattr(cell, 'v') else None for cell in header_row]
    # 4번째 행(인덱스 3)부터 데이터
    rows = data[3:]  # 모든 데이터 행 처리

    # 각 행을 딕셔너리로 변환
    for row in rows:
        row_dict = {}
        for i, cell in enumerate(row):
            if i < len(headers) and headers[i] is not None:
                # Cell 객체에서 값 추출
                value = cell.v if hasattr(cell, 'v') else cell
                if value == "":
                    value = None
                row_dict[headers[i]] = value

        # 빈 행은 건너뛰기
        if not any(value for value in row_dict.values()):
            continue

        # LogisticsShipment 객체로 변환
        shipment_data = convert_to_logistics_shipment(row_dict)
        shipment_list.append(shipment_data)

    print(f"총 {len(shipment_list)}개의 레코드가 변환되었습니다.")

    try:
        # 비동기 세션으로 데이터 저장
        async for session in get_session():
            async with session.begin():
                for shipment in shipment_list:
                    session.add(shipment)

        print(f"총 {len(shipment_list)}개의 레코드가 저장되었습니다.")
    except Exception as e:
        print(f"데이터베이스 저장 중 오류 발생: {e}")
        raise

    return shipment_list


def convert_to_logistics_shipment(row_dict):
    """
    엑셀에서 읽은 데이터를 LogisticsShipment 모델 객체로 변환합니다.

    Args:
        row_dict (dict): 컬럼명과 값이 매핑된 딕셔너리

    Returns:
        LogisticsShipment: 변환된 LogisticsShipment 객체
    """
    # 필드 매핑 - 엑셀 컬럼명을 SQLAlchemy 모델 필드명으로 변환
    field_mapping = {
        'LSS ID': 'lss_id',
        'LSS Name': 'lss_name',
        'Current Status': 'current_status',
        'Current Location': 'current_location',
        'TS Y/N': 'ts_yn',
        'SR No.': 'sr_no',
        'HBL No.': 'hbl_no',
        'MBL No.': 'mbl_no',
        'ULD No.': 'uld_no',
        'ULD Cd': 'uld_cd',
        'Service Nm': 'service_nm',
        'D. S. T.': 'd_s_t',
        'Service Term': 'service_term',
        'Incoterms': 'incoterms',
        'From Site ID': 'from_site_id',
        'From Site Name': 'from_site_name',
        'To Site ID': 'to_site_id',
        'To Site Name': 'to_site_name',
        'Shipper ID': 'shipper_id',
        'Shipper Name': 'shipper_name',
        'Consignee ID': 'consignee_id',
        'Consignee Nm': 'consignee_nm',
        'From Plant Cd': 'from_plant_cd',
        'From Plant Nm': 'from_plant_nm',
        'To Plant Cd': 'to_plant_cd',
        'To Plant Nm': 'to_plant_nm',
        'Delivery Lane': 'delivery_lane',
        'POL Nation': 'pol_nation',
        'POL Loc. ID': 'pol_loc_id',
        'POL Nm': 'pol_nm',
        'POD Nation': 'pod_nation',
        'POD Loc. ID': 'pod_loc_id',
        'POD Nm': 'pod_nm',
        'FD Nation Nm': 'fd_nation_nm',
        'FD Loc. ID': 'fd_loc_id',
        'FD Nm': 'fd_nm',
        'LSP ID': 'lsp_id',
        'LSP Nm': 'lsp_nm',
        'Carrier Code': 'carrier_code',
        'Carrier Nm': 'carrier_nm',
        'POL Vessel Flight Nm': 'pol_vessel_flight_nm',
        'POL Vessel Flight No': 'pol_vessel_flight_no',
        'POL Initial ETD': 'pol_initial_etd',
        'POL AS-IS ETD': 'pol_as_is_etd',
        'POL ATD': 'pol_atd',
        'POL ETD WEEK': 'pol_etd_week',
        'POL ATD WEEK': 'pol_atd_week',
        'POL ETD Month': 'pol_etd_month',
        'POL 적기/지연/조기 출발': 'pol_ontime_status',
        'POL Aging': 'pol_aging',
        'POL Aging 기간': 'pol_aging_period',
        'POD Vessel Flight Nm': 'pod_vessel_flight_nm',
        'POD Vessel Flight No': 'pod_vessel_flight_no',
        'POD Initial ETA': 'pod_initial_eta',
        'POD AS-IS ETA': 'pod_as_is_eta',
        'Region Cd': 'pod_region_cd',
        'LT day (POL ATD ~ POD Initial ETA)': 'lt_day_pol_atd_pod_initial_eta',
        'LT day (POL ATD ~ POD AS-IS ETA)': 'lt_day_pol_atd_pod_as_is_eta',
        'CNTR 중복 O/X': 'cntr_duplicate',
        'Target ETA (final)': 'target_eta'
    }

    # LogisticsShipment 객체 생성
    shipment = LogisticsShipment()

    # 각 필드에 데이터 설정
    for excel_col, model_field in field_mapping.items():
        try:
            if excel_col in row_dict:
                value = row_dict[excel_col]

                # Boolean 필드 변환 ('Y'/'N' -> True/False)
                if model_field == 'ts_yn':
                    if isinstance(value, str):
                        value = value.upper() == 'Y'

                # DateTime 필드 처리
                if model_field.endswith('_eta') or model_field.endswith('_etd') or model_field.endswith(
                        '_ata') or model_field.endswith('_atd'):
                    if value:
                        try:
                            # 날짜 문자열 파싱
                            if isinstance(value, str):
                                value = parse_date_string(value)
                        except Exception as e:
                            print(f"날짜 변환 오류 ({excel_col}): {e}")
                            value = None

                # Integer 필드 처리
                if model_field.endswith('_period') or model_field.startswith('lt_day'):
                    if value:
                        try:
                            value = int(float(value)) if not isinstance(value, int) else value
                        except:
                            value = None

                # 모델에 값 설정
                if hasattr(shipment, model_field):
                    setattr(shipment, model_field, value)
        except Exception as e:
            print(f"필드 설정 오류 ({excel_col} -> {model_field}): {e}")

    return shipment


def parse_date_string(date_str):
    """
    다양한 형식의 날짜 문자열을 datetime 객체로 변환합니다.

    Args:
        date_str (str): 날짜 문자열

    Returns:
        datetime: 변환된 datetime 객체
    """
    from datetime import datetime

    # 엑셀의 날짜 형식에 따라 여러 포맷 시도
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y/%m/%d %H:%M:%S',
        '%d-%m-%Y %H:%M:%S',
        '%d/%m/%Y %H:%M:%S',
        '%Y-%m-%d',
        '%Y/%m/%d',
        '%d-%m-%Y',
        '%d/%m/%Y'
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    # 모든 포맷 시도 실패 시 원본 값 반환
    return date_str


async def main():
    excel_file = "shipping_data.xlsb"
    await read_excel_and_save_to_db(excel_file)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())