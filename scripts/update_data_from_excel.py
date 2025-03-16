from typing import List

import pandas as pd

from datetime import datetime

from fastapi import Depends
from pyxlsb import open_workbook
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import Base, async_get_db
from model.shipping import Shipping

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

async def read_excel_and_save_to_db(excel_file_path: str, sheet_name=0) -> List[Shipping]:
    """
    엑셀 파일을 읽어서 Shipping 모델을 통해 데이터베이스에 저장합니다.

    Args:
        excel_file_path (str): 엑셀 파일 경로
        sheet_name (str or int, optional): 읽을 시트 이름 또는 인덱스. 기본값은 0(첫 번째 시트).

    Returns:
        int: 저장된 레코드 수
    """

    shipping_list = []
    with open_workbook(excel_file_path) as wb:
        with wb.get_sheet("Raw") as sheet:
            data = [row for row in sheet.rows()]

    # 헤더 행(3행)과 데이터 행 분리
    header_row = data[2]  # 3번째 행(인덱스 2)이 컬럼명

    # Cell 객체에서 값(v) 추출
    headers = [cell.v if hasattr(cell, 'v') else None for cell in header_row]
    # 4번째 행(인덱스 3)부터 데이터
    rows = data[3:6]

    # 각 행을 딕셔너리로 변환
    for row in rows:
        row_dict = {}
        for i, cell in enumerate(row):
            if i < len(headers) and headers[i] is not None:
                # Cell 객체에서 값 추출
                value = cell.v if hasattr(cell, 'v') else cell
                row_dict[headers[i]] = value

        # 빈 행은 건너뛰기
        if not any(value for value in row_dict.values()):
            continue

        # Shipping 객체로 변환
        shipping_data = convert_to_shipping_data(row_dict)
        shipping_list.append(shipping_data)


    try:
        # 비동기 세션으로 데이터 저장
        async for session in get_session():
            async with session.begin():
                for shipping in shipping_list:
                    session.add(shipping)

            # 세션 커밋은 context manager가 자동으로 처리

        print(f"총 {len(shipping_list)}개의 레코드가 저장되었습니다.")
    except Exception as e:
        print(f"데이터베이스 저장 중 오류 발생: {e}")
        raise

    return shipping_list


def convert_to_shipping_data(row_dict):
    """
    엑셀에서 읽은 데이터를 ShippingData 모델 객체로 변환합니다.

    Args:
        row_dict (dict): 컬럼명과 값이 매핑된 딕셔너리

    Returns:
        ShippingData: 변환된 ShippingData 객체
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
        '1st TS Nation': 'ts1_nation',
        '1st TS Loc. ID': 'ts1_loc_id',
        '1st TS Loc. Nm': 'ts1_loc_nm',
        '2nd TS Nation': 'ts2_nation',
        '2nd TS Loc. ID': 'ts2_loc_id',
        '2nd TS Loc. Nm': 'ts2_loc_nm',
        '3rd TS Nation': 'ts3_nation',
        '3rd TS Loc. ID': 'ts3_loc_id',
        '3rd TS Loc. Nm': 'ts3_loc_nm',
        '4th TS Nation': 'ts4_nation',
        '4th TS Loc. ID': 'ts4_loc_id',
        '4th TS Loc. Nm': 'ts4_loc_nm',
        '5th TS Nation': 'ts5_nation',
        '5th TS Loc. ID': 'ts5_loc_id',
        '5th TS Loc. Nm': 'ts5_loc_nm',
        'LSP ID': 'lsp_id',
        'LSP Nm': 'lsp_nm',
        'Carrier Code': 'carrier_code',
        'Carrier Nm': 'carrier_nm',
        'Pre-Vessel Name': 'pre_vessel_name',
        'Pre-Vessel Voy.no': 'pre_vessel_voy_no',
        'POL Loc. ID': 'pol_detail_loc_id',
        'POL Loc. Nm': 'pol_detail_loc_nm',
        'POL Initial ETD': 'pol_initial_etd',
        'POL AS-IS ETD': 'pol_as_is_etd',
        'POL ATD': 'pol_atd',
        'POL ETD WEEK': 'pol_etd_week',
        'POL ATD WEEK': 'pol_atd_week',
        'POL ETD Month': 'pol_etd_month',
        'POL Vessel Flight Nm': 'pol_vessel_flight_nm',
        'POL Vessel Flight No': 'pol_vessel_flight_no',
        '계획 vs 계획 (POL ETD - Initial ETD)': 'pol_plan_vs_plan',
        '계획 vs 실적 (POL ATD - Initial ETD)': 'pol_plan_vs_actual',
        '계획 vs 실적 (POL ATD - ETD)': 'pol_plan_vs_actual_diff',
        '계획 vs 실적 WK (POL ATD - Initial ETD)': 'pol_plan_vs_actual_wk',
        '계획 vs 실적 월 (POL ATD - Initial ETD)': 'pol_plan_vs_actual_month',
        'POL 적기/지연/조기 출발': 'pol_ontime_delay_early',
        'POL Aging 기간': 'pol_aging_period',
        'POL Aging': 'pol_aging',
        '1st T/S Loc. ID': 'ts1_detail_loc_id',
        '1st T/S Loc. Nm': 'ts1_detail_loc_nm',
        '1st T/S Initial ETA': 'ts1_initial_eta',
        '1st T/S AS-IS ETA': 'ts1_as_is_eta',
        '1st T/S ATA': 'ts1_ata',
        '1st T/S Initial ETD': 'ts1_initial_etd',
        '1st T/S AS-IS ETD': 'ts1_as_is_etd',
        '1st T/S ATD': 'ts1_atd',
        '1st T/S ETD WEEK': 'ts1_etd_week',
        '1st T/S ATD WEEK': 'ts1_atd_week',
        '1st T/S Vessel Flight Nm': 'ts1_vessel_flight_nm',
        '1st T/S Vessel Flight No': 'ts1_vessel_flight_no',
        '계획 vs 계획 (1st T/S ETA - Initial ETA)': 'ts1_plan_vs_plan_eta',
        '계획 vs 실적 (1st T/S ATA - Initial ETA)': 'ts1_plan_vs_actual_eta',
        '계획 vs 실적 (1st T/S ATA - ETA)': 'ts1_plan_vs_actual_eta_diff',
        '계획 vs 계획 (1st T/S ETD - Initial ETD)': 'ts1_plan_vs_plan_etd',
        '계획 vs 실적 (1st T/S ATD - Initial ETD)': 'ts1_plan_vs_actual_etd',
        '계획 vs 실적 (1st T/S ATD - ETD)': 'ts1_plan_vs_actual_etd_diff',
        '1st T/S Aging 기간': 'ts1_aging_period',
        '1st T/S Aging': 'ts1_aging',
        '2nd T/S Loc. ID': 'ts2_detail_loc_id',
        '2nd T/S Loc. Nm': 'ts2_detail_loc_nm',
        '2nd T/S Initial ETA': 'ts2_initial_eta',
        '2nd T/S AS-IS ETA': 'ts2_as_is_eta',
        '2nd T/S ATA': 'ts2_ata',
        '2nd T/S Initial ETD': 'ts2_initial_etd',
        '2nd T/S AS-IS ETD': 'ts2_as_is_etd',
        '2nd T/S ATD': 'ts2_atd',
        '2nd T/S ETD WEEK': 'ts2_etd_week',
        '2nd T/S ATD WEEK': 'ts2_atd_week',
        '2nd T/S Vessel Flight Nm': 'ts2_vessel_flight_nm',
        '2nd T/S Vessel Flight No': 'ts2_vessel_flight_no',
        '계획 vs 계획 (2nd T/S ETA - Initial ETA)': 'ts2_plan_vs_plan_eta',
        '계획 vs 실적 (2nd T/S ATA - Initial ETA)': 'ts2_plan_vs_actual_eta',
        '계획 vs 실적 (2nd T/S ATA - ETA)': 'ts2_plan_vs_actual_eta_diff',
        '계획 vs 계획 (2nd T/S ETD - Initial ETD)': 'ts2_plan_vs_plan_etd',
        '계획 vs 실적 (2nd T/S ATD - Initial ETD)': 'ts2_plan_vs_actual_etd',
        '계획 vs 실적 (2nd T/S ATD - ETD)': 'ts2_plan_vs_actual_etd_diff',
        '2nd T/S Aging 기간': 'ts2_aging_period',
        '2nd T/S Aging': 'ts2_aging',
        '3rd T/S Loc. ID': 'ts3_detail_loc_id',
        '3rd T/S Loc. Nm': 'ts3_detail_loc_nm',
        '3rd T/S Initial ETA': 'ts3_initial_eta',
        '3rd T/S AS-IS ETA': 'ts3_as_is_eta',
        '3rd T/S ATA': 'ts3_ata',
        '3rd T/S Initial ETD': 'ts3_initial_etd',
        '3rd T/S AS-IS ETD': 'ts3_as_is_etd',
        '3rd T/S ATD': 'ts3_atd',
        '3rd T/S ETD WEEK': 'ts3_etd_week',
        '3rd T/S ATD WEEK': 'ts3_atd_week',
        '3rd T/S Vessel Flight Nm': 'ts3_vessel_flight_nm',
        '3rd T/S Vessel Flight No': 'ts3_vessel_flight_no',
        '계획 vs 계획 (3rd T/S ETA - Initial ETA)': 'ts3_plan_vs_plan_eta',
        '계획 vs 실적 (3rd T/S ATA - Initial ETA)': 'ts3_plan_vs_actual_eta',
        '계획 vs 실적 (3rd T/S ATA - ETA)': 'ts3_plan_vs_actual_eta_diff',
        '계획 vs 계획 (3rd T/S ETD - Initial ETD)': 'ts3_plan_vs_plan_etd',
        '계획 vs 실적 (3rd T/S ATD - Initial ETD)': 'ts3_plan_vs_actual_etd',
        '계획 vs 실적 (3rd T/S ATD - ETD)': 'ts3_plan_vs_actual_etd_diff',
        '3rd T/S Aging 기간': 'ts3_aging_period',
        '3rd T/S Aging': 'ts3_aging',
        '4th T/S Loc. ID': 'ts4_detail_loc_id',
        '4th T/S Loc. Nm': 'ts4_detail_loc_nm',
        '4th T/S Initial ETA': 'ts4_initial_eta',
        '4th T/S AS-IS ETA': 'ts4_as_is_eta',
        '4th T/S ATA': 'ts4_ata',
        '4th T/S Initial ETD': 'ts4_initial_etd',
        '4th T/S AS-IS ETD': 'ts4_as_is_etd',
        '4th T/S ATD': 'ts4_atd',
        '4th T/S ETD WEEK': 'ts4_etd_week',
        '4th T/S ATD WEEK': 'ts4_atd_week',
        '4th T/S Vessel Flight Nm': 'ts4_vessel_flight_nm',
        '4th T/S Vessel Flight No': 'ts4_vessel_flight_no',
        '계획 vs 계획 (4th T/S ETA - Initial ETA)': 'ts4_plan_vs_plan_eta',
        '계획 vs 실적 (4th T/S ATA - Initial ETA)': 'ts4_plan_vs_actual_eta',
        '계획 vs 실적 (4th T/S ATA - ETA)': 'ts4_plan_vs_actual_eta_diff',
        '계획 vs 계획 (4th T/S ETD - Initial ETD)': 'ts4_plan_vs_plan_etd',
        '계획 vs 실적 (4th T/S ATD - Initial ETD)': 'ts4_plan_vs_actual_etd',
        '계획 vs 실적 (4th T/S ATD - ETD)': 'ts4_plan_vs_actual_etd_diff',
        '4th T/S Aging 기간': 'ts4_aging_period',
        '4th T/S Aging': 'ts4_aging',
        '5th T/S Loc. ID': 'ts5_detail_loc_id',
        '5th T/S Loc. Nm': 'ts5_detail_loc_nm',
        '5th T/S Initial ETA': 'ts5_initial_eta',
        '5th T/S AS-IS ETA': 'ts5_as_is_eta',
        '5th T/S ATA': 'ts5_ata',
        '5th T/S Initial ETD': 'ts5_initial_etd',
        '5th T/S AS-IS ETD': 'ts5_as_is_etd',
        '5th T/S ATD': 'ts5_atd',
        '5th T/S ETD WEEK': 'ts5_etd_week',
        '5th T/S ATD WEEK': 'ts5_atd_week',
        '5th T/S Vessel Flight Nm': 'ts5_vessel_flight_nm',
        '5th T/S Vessel Flight No': 'ts5_vessel_flight_no',
        '계획 vs 계획 (5th T/S ETA - Initial ETA)': 'ts5_plan_vs_plan_eta',
        '계획 vs 실적 (5th T/S ATA - Initial ETA)': 'ts5_plan_vs_actual_eta',
        '계획 vs 실적 (5th T/S ATA - ETA)': 'ts5_plan_vs_actual_eta_diff',
        '계획 vs 계획 (5th T/S ETD - Initial ETD)': 'ts5_plan_vs_plan_etd',
        '계획 vs 실적 (5th T/S ATD - Initial ETD)': 'ts5_plan_vs_actual_etd',
        '계획 vs 실적 (5th T/S ATD - ETD)': 'ts5_plan_vs_actual_etd_diff',
        '5th T/S Aging 기간': 'ts5_aging_period',
        '5th T/S Aging': 'ts5_aging',
        'POD Loc. ID': 'pod_detail_loc_id',
        'POD Loc. Nm': 'pod_detail_loc_nm',
        'POD Initial ETA': 'pod_initial_eta',
        'POD AS-IS ETA': 'pod_as_is_eta',
        'POD ATA': 'pod_ata',
        'POD Vessel Flight Nm': 'pod_vessel_flight_nm',
        'POD Vessel Flight No': 'pod_vessel_flight_no',
        'POD ETA Week': 'pod_eta_week',
        'POD ATA Week': 'pod_ata_week',
        'POD Initial ETD': 'pod_initial_etd',
        'POD AS-IS ETD': 'pod_as_is_etd',
        'POD ATD': 'pod_atd',
        'Region Cd': 'region_cd',
        '계획 vs 계획 (POD ETA - ETA 기준일)': 'pod_plan_vs_plan',
        '계획 vs 실적 (POD ATA - ETA 기준일)': 'pod_plan_vs_actual',
        '계획 vs 실적 (POD ATA - ETA)': 'pod_plan_vs_actual_diff',
        '계획 vs 계획 WK (POD ETA - ETA 기준일)': 'pod_plan_vs_plan_wk',
        '계획 vs 계획 월 (POD ETA - ETA 기준일)': 'pod_plan_vs_plan_month',
        '계획 \'ETA 기준일\' POD 적기/지연/조기 도착': 'pod_ontime_delay_early',
        'LT day (POL ATD ~ POD ATA)': 'lt_day',
        'LT WK (POL ATD ~ POD ATA)': 'lt_wk',
        '계획 vs 실적 (POD ATD - ETD)': 'pod_plan_vs_actual_etd_diff',
        'POD Aging 기간': 'pod_aging_period',
        'POD Aging': 'pod_aging',
        'Total': 'total',
        'Rep. Carrier': 'rep_carrier',
        'CNTR 중복 O/X': 'cntr_duplicate',
        'HBL+MBL+CNTR': 'hbl_mbl_cntr',
        'MBL+CNTR': 'mbl_cntr',
        'MBL+CNTR 중복 O/X': 'mbl_cntr_duplicate',
        '도착지 CY or DR': 'arrival_cy_or_dr',
        'Target ETA (final)': 'target_eta',
        'ETA 기준일 (적용기준 : Target ETA > Initial ETA)': 'eta_base_date',
        '계획 vs 계획 (POD ETA - Initial ETA)': 'pod_plan_vs_plan_initial',
        '계획 vs 실적 (POD ATA - Initial ETA)': 'pod_plan_vs_actual_initial',
        '계획 vs 실적 (POD ATA - ETA)': 'pod_plan_vs_actual_diff_initial',
        '계획 vs 계획 WK (POD ETA - Initial ETA)': 'pod_plan_vs_plan_wk_initial',
        '계획 vs 계획 월 (POD ETA - Initial ETA)': 'pod_plan_vs_plan_month_initial',
        '계획 Initial ETA POD 적기/지연/조기 도착': 'pod_ontime_delay_early_initial',
        'LT day (POL ATD ~ POD Initial ETA)': 'lt_day_initial',
        'LT day (POL ATD ~ POD AS-IS ETA)': 'lt_day_as_is',
        'LT day (POL ATD ~ POD Target ETA)': 'lt_day_target',
        'LT day (POL ATD ~ POD ETA 기준일)': 'lt_day_base',
        '이전 \'ETA 기준일\'': 'previous_eta_base_date',
        '현 \'ETA 기준일 - 이전 \'ETA 기준일': 'eta_base_date_diff',
        '1st TS Aging days (ETA ~ ETD)': 'ts1_aging_days_plan',
        '2nd TS Aging days (ETA ~ ETD)': 'ts2_aging_days_plan',
        '3rd TS Aging days (ETA ~ ETD)': 'ts3_aging_days_plan',
        '4th TS Aging days (ETA ~ ETD)': 'ts4_aging_days_plan',
        '5th TS Aging days (ETA ~ ETD)': 'ts5_aging_days_plan',
        '1st TS Aging days (ATA ~ ETD)': 'ts1_aging_days_actual',
        '2nd TS Aging days (ATA ~ ETD)': 'ts2_aging_days_actual',
        '3rd TS Aging days (ATA ~ ETD)': 'ts3_aging_days_actual',
        '4th TS Aging days (ATA ~ ETD)': 'ts4_aging_days_actual',
        '5th TS Aging days (ATA ~ ETD)': 'ts5_aging_days_actual',
        '1st TS check required Y/N': 'ts1_check_required',
        '2nd TS check required Y/N': 'ts2_check_required',
        '3rd TS check required Y/N': 'ts3_check_required',
        '4th TS check required Y/N': 'ts4_check_required',
        '5th TS check required Y/N': 'ts5_check_required'
    }

    # ShippingData 객체 생성
    shipping_data = Shipping()

    # 각 필드에 데이터 설정
    for excel_col, model_field in field_mapping.items():
        try:
            if excel_col in row_dict:
                value = row_dict[excel_col]

                # DateTime 필드 처리
                if model_field.endswith('_eta') or model_field.endswith('_etd') or model_field.endswith(
                        '_ata') or model_field.endswith('_atd'):
                    if isinstance(value, str) and value:
                        try:
                            # 날짜 문자열 파싱 (형식에 따라 조정 필요)
                            value = parse_date_string(value)
                        except:
                            value = None

                # 모델에 값 설정
                if hasattr(shipping_data, model_field):
                    setattr(shipping_data, model_field, value)
        except Exception as e:
            print(excel_col, model_field, e)
    return shipping_data


def parse_date_string(date_str):
    """
    다양한 형식의 날짜 문자열을 datetime 객체로 변환합니다.

    Args:
        date_str (str): 날짜 문자열

    Returns:
        datetime: 변환된 datetime 객체
    """
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