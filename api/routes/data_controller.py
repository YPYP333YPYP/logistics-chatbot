# 라우터 생성
import os
import shutil
import traceback
from typing import Dict, Any

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from starlette import status

from service.data_service import DataService

router = APIRouter(
    prefix="/api/data",
    tags=["data"],
    responses={404: {"description": "Not found"}}
)


@router.post("/upload-excel", response_model=Dict[str, Any])
async def upload_excel_file(
        file: UploadFile = File(...),
        sheet_name: str = "Raw",
        service: DataService = Depends(DataService)
):
    """
    엑셀 파일을 업로드하여 물류 데이터를 데이터베이스에 저장합니다.

    Args:
        file (UploadFile): 업로드할 엑셀 파일 (.xlsb 형식)
        sheet_name (str, optional): 처리할 시트 이름. 기본값은 "Raw"

    Returns:
        Dict[str, Any]: 처리 결과 정보
    """
    # 파일 형식 확인 (확장자 체크)
    if not file.filename.lower().endswith('.xlsb'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="지원하지 않는 파일 형식입니다. .xlsb 파일만 업로드 가능합니다."
        )

    # 환경 변수에서 업로드 디렉토리 경로 가져오기
    EXCEL_FILE_PATH = os.getenv('EXCEL_FILE_PATH', 'uploads')

    # 디렉토리가 없으면 생성
    os.makedirs(EXCEL_FILE_PATH, exist_ok=True)

    # 파일명 충돌 방지를 위한 타임스탬프 추가
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    safe_filename = f"{timestamp}_{file.filename}"

    # 파일 저장 경로
    file_path = os.path.join(EXCEL_FILE_PATH, safe_filename)

    try:
        # 1. 업로드된 파일을 디스크에 저장
        with open(file_path, "wb") as buffer:
            # file.file에서 읽은 내용을 buffer에 복사
            shutil.copyfileobj(file.file, buffer)

        # 2. 저장된 파일 경로를 서비스 메서드에 전달
        await service.read_excel_and_save_to_db(file_path, sheet_name)

        # 3. 결과 반환
        return {
            "status": "success",
            "message": "레코드가 성공적으로 저장되었습니다.",
            "filename": file.filename
        }

    except Exception as e:
        error_detail = str(e)
        print(f"파일 처리 중 오류: {error_detail}")
        print(traceback.format_exc())

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"파일 처리 중 오류가 발생했습니다: {error_detail}"
        )

    finally:
        # 자동 삭제 옵션이 활성화되어 있다면 파일 삭제
        if os.getenv('AUTO_DELETE_UPLOADS', 'false').lower() == 'true':
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"임시 파일 삭제: {file_path}")
            except Exception as e:
                print(f"파일 삭제 중 오류: {e}")