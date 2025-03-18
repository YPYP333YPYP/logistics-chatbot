import json
import os

from datetime import datetime
from typing import List, Dict
from fastapi import Depends
from pyxlsb import open_workbook
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import async_get_db
from model.logistics_shipment import LogisticsShipment


class DataService:

    def __init__(self, session: AsyncSession = Depends(async_get_db)):
        self.session = session

    async def read_excel_and_save_to_db(self, excel_file_path: str, sheet_name: str = "Raw") -> List[LogisticsShipment]:
        """
        엑셀 파일을 읽어서 LogisticsShipment 모델을 통해 데이터베이스에 저장합니다.

        Args:
             excel_file_path (str): 업로드된 파일 경로
            sheet_name (str, optional): 읽을 시트 이름. 기본값은 "Raw".

        Returns:
            List[LogisticsShipment]: 저장된 레코드 리스트
        """

        shipment_list = []

        try:
            # 엑셀 파일 열고 데이터 읽기
            with open_workbook(excel_file_path) as wb:
                with wb.get_sheet(sheet_name) as sheet:
                    data = [row for row in sheet.rows()]

            # 헤더 행(3행)과 데이터 행 분리
            header_row = data[2]  # 3번째 행(인덱스 2)이 컬럼명

            # Cell 객체에서 값(v) 추출 - Cell(r(row), c(column), v(value))
            headers = [cell.v if hasattr(cell, 'v') else None for cell in header_row]

            # 4번째 행(인덱스 3)부터 데이터
            rows = data[3:103]  # 데이터 수 제한

            # 각 행을 딕셔너리로 변환
            for row in rows:
                row_dict = {}
                for i, cell in enumerate(row):
                    if i < len(headers) and headers[i] is not None:
                        # Cell 객체에서 값 추출
                        value = cell.v if hasattr(cell, 'v') else cell

                        # 공백 값 None 처리
                        if value == "":
                            value = None
                        row_dict[headers[i]] = value

                # 빈 행은 건너뛰기
                if not any(value for value in row_dict.values()):
                    continue

                # LogisticsShipment 객체로 변환
                shipment_data = self.convert_to_logistics_shipment(row_dict)
                shipment_list.append(shipment_data)

            print(f"총 {len(shipment_list)}개의 레코드가 변환되었습니다.")

            try:
                # 데이터베이스에 저장
                async with self.session.begin():
                    for shipment in shipment_list:
                        self.session.add(shipment)
                await self.session.commit()
                print(f"총 {len(shipment_list)}개의 레코드가 저장되었습니다.")
            except Exception as e:
                await self.session.rollback()
                print(f"데이터베이스 저장 중 오류 발생: {e}")
                raise

            return shipment_list

        except Exception as e:
            print(f"엑셀 파일 처리 중 오류: {e}")
            raise

    def convert_to_logistics_shipment(self, row_dict: Dict):
        """
        엑셀에서 읽은 데이터를 LogisticsShipment 모델 객체로 변환합니다.

        Args:
            row_dict (dict): 컬럼명과 값이 매핑된 딕셔너리

        Returns:
            LogisticsShipment: 변환된 LogisticsShipment 객체
        """
        # 필드 매핑
        field_mapping = self._load_field_mapping()

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
                                    value = self.parse_date_string(value)
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

    def parse_date_string(self, date_str):
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

    def _load_field_mapping(self):
        """
        외부 JSON 파일에서 필드 매핑 정보를 로드합니다.
        """
        # 매핑 파일 경로 (환경 변수 또는 기본값)
        mapping_file = os.environ.get('FIELD_MAPPING_PATH', 'resource/field_mapping.json')

        try:
            # JSON 파일 로드 시도
            with open(mapping_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"매핑 파일 로드 실패: {e}.")
