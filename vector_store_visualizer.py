import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

st.set_page_config(page_title="벡터 스토어 시각화 도구", layout="wide")


class VectorStoreVisualizer:
    """벡터 스토어 시각화 및 관리 도구"""

    def __init__(self, vector_store_path: str = "../data/vector_store"):
        """
        시각화 도구 초기화

        Args:
            vector_store_path (str): 벡터 스토어 경로
        """
        self.vector_store_path = vector_store_path
        self.embeddings = None
        self.vector_store = None

        # 벡터 스토어가 존재하는지 확인
        if os.path.exists(os.path.join(vector_store_path, "chroma.sqlite3")):
            self.load_vector_store()
        else:
            print(f"벡터 스토어를 찾을 수 없습니다: {vector_store_path}")

    def load_vector_store(self):
        """벡터 스토어 로드"""
        try:
            # 임베딩 모델 초기화
            self.embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

            # 벡터 스토어 로드
            self.vector_store = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings
            )

            # 문서 수 체크
            count = self.vector_store._collection.count()
            if count == 0:
                st.warning("벡터 스토어에 문서가 없습니다.")
            else:
                print(f"벡터 스토어 로드 완료. 문서 수: {count}")

            # 데이터 샘플링 테스트
            try:
                sample = self.vector_store._collection.get(limit=1)
                print(f"샘플 데이터 확인: {len(sample.get('documents', []))}개 문서, "
                      f"{len(sample.get('embeddings', []))}개 임베딩, "
                      f"{len(sample.get('metadatas', []))}개 메타데이터")
            except Exception as e:
                print(f"샘플 데이터 확인 중 오류: {str(e)}")

        except Exception as e:
            print(f"벡터 스토어 로드 중 오류: {str(e)}")
            self.vector_store = None
            raise

    def get_document_count(self):
        """총 문서 수 반환"""
        if self.vector_store:
            return self.vector_store._collection.count()
        return 0

    def get_document_types(self):
        """문서 유형별 개수 반환"""
        if not self.vector_store:
            return {}

        data = self.vector_store._collection.get()
        metadatas = data["metadatas"]

        doc_types = {}
        for metadata in metadatas:
            doc_type = metadata.get("doc_type", "unknown")
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

        return doc_types

    def search_documents(self, query: str, k: int = 5):
        """문서 검색"""
        if not self.vector_store:
            return []

        docs = self.vector_store.similarity_search(query, k=k)
        return docs

    def search_with_filter(self, query: str, filter_dict: dict, k: int = 5):
        """필터를 적용하여 문서 검색"""
        if not self.vector_store:
            return []

        # Chroma 필터 형식으로 변환
        filter_dict_chroma = {}
        for key, value in filter_dict.items():
            filter_dict_chroma[key] = {"$eq": value}

        docs = self.vector_store.similarity_search(
            query,
            k=k,
            filter=filter_dict_chroma
        )
        return docs

    def get_metadata_fields(self):
        """사용 가능한 메타데이터 필드 목록 반환"""
        if not self.vector_store:
            return []

        data = self.vector_store._collection.get()
        metadatas = data["metadatas"]

        all_fields = set()
        for metadata in metadatas:
            all_fields.update(metadata.keys())

        return sorted(list(all_fields))

    def get_metadata_values(self, field: str):
        """특정 메타데이터 필드의 고유 값 목록 반환"""
        if not self.vector_store:
            return []

        data = self.vector_store._collection.get()
        metadatas = data["metadatas"]

        values = set()
        for metadata in metadatas:
            if field in metadata:
                values.add(metadata[field])

        return sorted(list(values))

    def delete_documents(self, filter_dict: dict):
        """필터 조건에 맞는 문서 삭제"""
        if not self.vector_store:
            return 0

        # Chroma 필터 형식으로 변환
        filter_dict_chroma = {}
        for key, value in filter_dict.items():
            filter_dict_chroma[key] = {"$eq": value}

        # 삭제 전 문서 수
        before_count = self.vector_store._collection.count()

        # 문서 삭제
        self.vector_store._collection.delete(filter=filter_dict_chroma)

        # 삭제 후 문서 수
        after_count = self.vector_store._collection.count()

        # 삭제된 문서 수 반환
        return before_count - after_count

    def visualize_embeddings(self, method: str = "tsne", n_docs: int = 1000):
        """임베딩 시각화 (t-SNE 또는 PCA)"""
        if not self.vector_store:
            return None, None

        # 문서 수 제한
        n_docs = min(n_docs, self.vector_store._collection.count())

        # 임베딩 및 메타데이터 가져오기
        data = self.vector_store._collection.get(limit=n_docs)
        embeddings = data.get("embeddings", [])
        metadatas = data.get("metadatas", [])
        documents = data.get("documents", [])

        # 데이터가 비어있는지 확인
        if not embeddings or not metadatas or not documents:
            st.warning("데이터를 가져올 수 없습니다. 벡터 스토어가 비어있을 수 있습니다.")
            return None, None

        # NaN 값을 포함하는 임베딩 제거
        valid_indices = []
        valid_embeddings = []
        for i, embedding in enumerate(embeddings):
            # NaN 값이 없는 임베딩만 사용
            if embedding is not None and not np.isnan(np.array(embedding)).any():
                valid_indices.append(i)
                valid_embeddings.append(embedding)

        # 유효한 임베딩이 없으면 오류 반환
        if len(valid_embeddings) == 0:
            st.warning("유효한 임베딩이 없습니다. 모든 임베딩에 NaN 값이 포함되어 있습니다.")
            return None, None

        # 유효한 임베딩만 차원 축소에 사용
        valid_embeddings = np.array(valid_embeddings)

        # 차원 축소
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
        else:  # PCA
            reducer = PCA(n_components=2, random_state=42)

        reduced_embeddings = reducer.fit_transform(valid_embeddings)

        # 데이터프레임 생성 (유효한 인덱스만 사용)
        df = pd.DataFrame({
            "x": reduced_embeddings[:, 0],
            "y": reduced_embeddings[:, 1],
            "document": [documents[i][:100] + "..." if i < len(documents) else "Unknown" for i in valid_indices],
        })

        # 메타데이터 추가 - 메타데이터가 비어있지 않은지 확인
        if metadatas and len(metadatas) > 0 and metadatas[0] is not None:
            meta_keys = metadatas[0].keys()
            for key in meta_keys:
                df[key] = [metadatas[i].get(key, "") if i < len(metadatas) and metadatas[i] is not None else "" for i in
                           valid_indices]

        return df, method

    def export_to_csv(self, output_path: str):
        """벡터 스토어 데이터를 CSV로 내보내기"""
        if not self.vector_store:
            return False

        data = self.vector_store._collection.get()
        documents = data["documents"]
        metadatas = data["metadatas"]

        # 데이터프레임 생성
        df_data = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            row = {"id": i, "content": doc}
            row.update(meta)
            df_data.append(row)

        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False, encoding="utf-8")

        return True

    def get_all_data(self, page: int = 1, page_size: int = 100, filter_dict: dict = None):
        """모든 데이터 페이지 단위로 가져오기"""
        if not self.vector_store:
            return None, 0

        # 전체 문서 수
        total_count = self.vector_store._collection.count()

        # 페이지 계산
        total_pages = (total_count + page_size - 1) // page_size
        offset = (page - 1) * page_size

        # 필터 적용 시 Chroma 필터 형식으로 변환
        filter_dict_chroma = None
        if filter_dict:
            filter_dict_chroma = {}
            for key, value in filter_dict.items():
                if value:  # 값이 비어있지 않은 경우에만 필터 적용
                    filter_dict_chroma[key] = {"$eq": value}

        try:
            # 데이터 가져오기
            if filter_dict_chroma:
                # 필터링된 데이터의 총 개수를 먼저 확인 (제한된 API로 인해 근사치)
                # 실제 ChromaDB에서는 필터와 함께 count를 지원하지 않으므로 전체를 가져와서 카운트
                all_ids = self.vector_store._collection.get(
                    filter=filter_dict_chroma,
                    include=["documents"]
                )
                filtered_count = len(all_ids.get("ids", []))
                total_pages = (filtered_count + page_size - 1) // page_size

                # 필터링된 데이터 가져오기 (offset과 limit 적용)
                data = self.vector_store._collection.get(
                    filter=filter_dict_chroma,
                    limit=page_size,
                    offset=offset
                )
            else:
                # 전체 데이터 가져오기 (offset과 limit 적용)
                data = self.vector_store._collection.get(
                    limit=page_size,
                    offset=offset
                )
                filtered_count = total_count

            # 데이터 추출
            ids = data.get("ids", [])
            documents = data.get("documents", [])
            metadatas = data.get("metadatas", [])

            # 데이터프레임 생성
            df_data = []
            for i, (doc_id, doc, meta) in enumerate(zip(ids, documents, metadatas)):
                row = {"id": doc_id, "content": doc[:200] + "..." if len(doc) > 200 else doc}
                if meta:
                    row.update(meta)
                df_data.append(row)

            df = pd.DataFrame(df_data)

            return df, filtered_count, total_pages

        except Exception as e:
            st.error(f"데이터 조회 중 오류: {str(e)}")
            return None, 0, 0

    def get_document_by_id(self, doc_id: str):
        """ID로 문서 상세 조회"""
        if not self.vector_store:
            return None

        try:
            data = self.vector_store._collection.get(
                ids=[doc_id],
                include=["documents", "metadatas", "embeddings"]
            )

            if data and len(data.get("ids", [])) > 0:
                return {
                    "id": data["ids"][0],
                    "document": data["documents"][0],
                    "metadata": data["metadatas"][0],
                    "embedding": data["embeddings"][0]
                }
            else:
                return None

        except Exception as e:
            st.error(f"문서 조회 중 오류: {str(e)}")
            return None


# Streamlit 앱
def run_streamlit_app():
    st.title("물류 데이터 벡터 스토어 시각화 및 관리 도구")

    # 벡터 스토어 경로 입력
    vector_store_path = st.sidebar.text_input(
        "벡터 스토어 경로",
        value="./data/vector_store"
    )

    # 시각화 도구 초기화
    visualizer = VectorStoreVisualizer(vector_store_path)

    # 문서 통계
    st.sidebar.header("문서 통계")
    st.sidebar.write(f"총 문서 수: {visualizer.get_document_count()}")

    doc_types = visualizer.get_document_types()
    st.sidebar.write("문서 유형별 개수:")
    for doc_type, count in doc_types.items():
        st.sidebar.write(f"- {doc_type}: {count}")

    # 탭 구성
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["문서 검색", "데이터 조회", "시각화", "관리", "내보내기"])

    # 탭 1: 문서 검색
    with tab1:
        st.header("문서 검색")

        # 검색 쿼리
        query = st.text_input("검색어 입력")

        # 필터 적용 옵션
        use_filter = st.checkbox("필터 적용")

        filter_dict = {}
        if use_filter:
            # 메타데이터 필드 선택
            fields = visualizer.get_metadata_fields()

            # 최대 3개 필터 적용
            for i in range(3):
                cols = st.columns(3)

                with cols[0]:
                    field = st.selectbox(f"필드 {i + 1}", ["선택..."] + fields, key=f"field_{i}")

                if field != "선택...":
                    with cols[1]:
                        values = ["선택..."] + visualizer.get_metadata_values(field)
                        value = st.selectbox(f"값 {i + 1}", values, key=f"value_{i}")

                    with cols[2]:
                        apply = st.checkbox("적용", key=f"apply_{i}", value=True)

                    if apply and value != "선택...":
                        filter_dict[field] = value

        # 검색 결과 개수
        k = st.slider("검색 결과 수", min_value=1, max_value=20, value=5)

        # 검색 버튼
        if st.button("검색", key="search_button"):
            if query:
                with st.spinner("검색 중..."):
                    if use_filter and filter_dict:
                        docs = visualizer.search_with_filter(query, filter_dict, k)
                    else:
                        docs = visualizer.search_documents(query, k)

                    if docs:
                        for i, doc in enumerate(docs, 1):
                            with st.expander(f"결과 {i}"):
                                st.markdown("**문서 내용:**")
                                st.text(doc.page_content)
                                st.markdown("**메타데이터:**")
                                st.json(doc.metadata)
                    else:
                        st.warning("검색 결과가 없습니다.")
            else:
                st.warning("검색어를 입력하세요.")

    # 탭 2: 데이터 조회
    with tab2:
        st.header("벡터 스토어 데이터 조회")

        # 필터 옵션
        st.subheader("데이터 필터링")

        filter_dict = {}

        # 메타데이터 필드 선택
        fields = visualizer.get_metadata_fields()

        # 필터 UI
        col1, col2 = st.columns(2)

        with col1:
            # 첫 번째 필터 옵션
            if fields:
                field1 = st.selectbox("필터 필드", ["선택..."] + fields, key="browse_field_1")
                if field1 != "선택...":
                    values = ["선택..."] + visualizer.get_metadata_values(field1)
                    value1 = st.selectbox("필터 값", values, key="browse_value_1")
                    if value1 != "선택...":
                        filter_dict[field1] = value1

        with col2:
            # 두 번째 필터 옵션
            if fields:
                field2 = st.selectbox("필터 필드", ["선택..."] + fields, key="browse_field_2")
                if field2 != "선택..." and field2 != field1:
                    values = ["선택..."] + visualizer.get_metadata_values(field2)
                    value2 = st.selectbox("필터 값", values, key="browse_value_2")
                    if value2 != "선택...":
                        filter_dict[field2] = value2

        # 페이지네이션 설정
        page_size = st.slider("페이지당 문서 수", min_value=10, max_value=200, value=50, step=10)

        # 세션 상태 초기화 (페이지 번호용)
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1

        # 필터 적용 버튼
        if st.button("필터 적용", key="apply_browse_filter"):
            st.session_state.current_page = 1

        # 데이터 가져오기
        df, total_count, total_pages = visualizer.get_all_data(
            page=st.session_state.current_page,
            page_size=page_size,
            filter_dict=filter_dict
        )

        # 페이지네이션 UI
        col1, col2, col3 = st.columns([1, 3, 1])

        with col1:
            if st.button("이전 페이지", disabled=st.session_state.current_page <= 1):
                st.session_state.current_page -= 1
                st.rerun()

        with col2:
            st.write(f"페이지: {st.session_state.current_page}/{max(1, total_pages)} (총 {total_count}개 문서)")

        with col3:
            if st.button("다음 페이지", disabled=st.session_state.current_page >= total_pages):
                st.session_state.current_page += 1
                st.rerun()

        # 데이터 테이블 표시
        if df is not None and not df.empty:
            # 데이터 테이블 표시 (스타일링 포함)
            st.dataframe(
                df,
                use_container_width=True,
                height=500,
                column_config={
                    "id": st.column_config.TextColumn("문서 ID", width="small"),
                    "content": st.column_config.TextColumn("문서 내용", width="large")
                }
            )

            # 문서 ID로 상세 조회
            st.subheader("문서 상세 조회")
            selected_id = st.text_input("문서 ID 입력", value="", placeholder="조회할 문서 ID를 입력하세요")

            if st.button("상세 조회", key="view_document") and selected_id:
                with st.spinner("문서 조회 중..."):
                    doc_detail = visualizer.get_document_by_id(selected_id)

                    if doc_detail:
                        # 문서 상세 정보 표시
                        st.subheader("문서 정보")
                        st.text_input("문서 ID", value=doc_detail["id"], disabled=True)

                        # 문서 내용
                        st.text_area("문서 내용", value=doc_detail["document"], height=200)

                        # 메타데이터
                        st.subheader("메타데이터")
                        st.json(doc_detail["metadata"])

                        # 임베딩 (처음 10개 요소만 표시)
                        st.subheader("임베딩 벡터 (처음 10개 요소)")
                        embedding = doc_detail["embedding"][:10]
                        st.write(embedding)

                        # 임베딩 차원 표시
                        st.write(f"임베딩 차원: {len(doc_detail['embedding'])}")
                    else:
                        st.warning(f"ID가 '{selected_id}'인 문서를 찾을 수 없습니다.")
        else:
            st.warning("표시할 데이터가 없습니다.")

    # 탭 3: 시각화
    with tab3:
        st.header("임베딩 시각화")

        # 시각화 방법 선택
        method = st.radio("시각화 방법", ["t-SNE", "PCA"], horizontal=True)
        method_key = "tsne" if method == "t-SNE" else "pca"

        # 문서 수 제한
        n_docs = st.slider("시각화할 문서 수", min_value=100, max_value=5000, value=1000, step=100)

        # 시각화 버튼
        if st.button("시각화", key="viz_button"):
            with st.spinner("시각화 계산 중..."):
                df, method_used = visualizer.visualize_embeddings(method_key, n_docs)

                if df is not None:
                    # 색상 매핑할 열 선택
                    color_by = st.selectbox("색상 구분", ["doc_type"] + df.columns.tolist())

                    # Plotly 그래프 생성
                    fig = px.scatter(
                        df, x="x", y="y",
                        color=color_by if color_by in df.columns else None,
                        hover_data=["document"] + [col for col in df.columns if col not in ["x", "y", "document"]],
                        title=f"{method} 시각화 (문서 {len(df)}개)"
                    )

                    # 그래프 표시
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("시각화 데이터를 생성할 수 없습니다.")

    # 탭 4: 관리
    with tab4:
        st.header("벡터 스토어 관리")

        st.markdown("### 문서 삭제")
        st.warning("주의: 삭제된 문서는 복구할 수 없습니다.")

        # 삭제 필터 선택
        st.subheader("삭제할 문서 필터")
        delete_filter = {}

        # 필드 선택
        fields = visualizer.get_metadata_fields()
        field = st.selectbox("필드", ["선택..."] + fields, key="delete_field")

        if field != "선택...":
            # 값 선택
            values = ["선택..."] + visualizer.get_metadata_values(field)
            value = st.selectbox("값", values, key="delete_value")

            if value != "선택...":
                delete_filter[field] = value

        # 삭제 버튼
        if st.button("선택 문서 삭제", key="delete_button"):
            if delete_filter:
                with st.spinner("문서 삭제 중..."):
                    num_deleted = visualizer.delete_documents(delete_filter)
                    st.success(f"{num_deleted}개 문서가 삭제되었습니다.")
            else:
                st.warning("삭제할 문서 필터를 선택하세요.")

    # 탭 5: 내보내기
    with tab5:
        st.header("벡터 스토어 내보내기")

        # 내보내기 경로
        export_path = st.text_input("내보내기 파일 경로", value="./data/vector_store_export.csv")

        # 내보내기 버튼
        if st.button("CSV로 내보내기", key="export_button"):
            with st.spinner("내보내기 중..."):
                success = visualizer.export_to_csv(export_path)
                if success:
                    st.success(f"벡터 스토어가 {export_path}로 내보내기되었습니다.")
                else:
                    st.error("내보내기 실패. 벡터 스토어가 로드되었는지 확인하세요.")


if __name__ == "__main__":
    run_streamlit_app()

