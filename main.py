"""
测试用例智构系统 - 轻量级 RAG（NumPy TF‑IDF）  无用户登录、项目知识库
阿里云OSS持久化存储 - 性能优化版
"""
import streamlit as st

st.set_page_config(page_title="测试用例智构系统", layout="wide")

import pandas as pd
from datetime import datetime
from config import AppConfig
from llm_service import LLMService
from core import (
    RDMService,
    ProjectManager,
    EnhancedKnowledgeBase,
    TestCaseService,
    ExportService
)

# ====================== 主视图 ======================
class MainView:
    def __init__(self):
        if "current_project" not in st.session_state:
            projects = ProjectManager.get_all_projects()
            if projects:
                st.session_state.current_project = projects[0]
            else:
                ProjectManager.create_project("默认项目")
                st.session_state.current_project = "默认项目"
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0
        if "show_new_project_input" not in st.session_state:
            st.session_state.show_new_project_input = False

        self.current_project = st.session_state.current_project
        self.kb = EnhancedKnowledgeBase(self.current_project)

    def render(self):
        st.title("测试用例智构系统")

        with st.sidebar:
            st.subheader("模型选择")
            model_key = st.selectbox("", AppConfig.get_model_list(), label_visibility="collapsed")
            st.divider()

            col_title, col_new = st.columns([3, 3])
            with col_title:
                st.subheader("项目管理")
            with col_new:
                if st.button("+ 新建项目", key="new_project_btn", use_container_width=True):
                    st.session_state.show_new_project_input = True

            if st.session_state.show_new_project_input:
                new_project_name = st.text_input("项目名称", placeholder="请输入项目名称", key="dialog_project_name")
                col_ok, col_cancel = st.columns(2)
                with col_ok:
                    if st.button("确定", key="confirm_new_project"):
                        if new_project_name and new_project_name.strip():
                            if ProjectManager.create_project(new_project_name):
                                st.success(f"项目 '{new_project_name}' 创建成功")
                                st.session_state.current_project = new_project_name
                                st.session_state.show_new_project_input = False
                                st.rerun()
                            else:
                                st.error("项目创建失败，可能名称已存在")
                        else:
                            st.warning("请输入项目名称")
                with col_cancel:
                    if st.button("取消", key="cancel_new_project", use_container_width=True):
                        st.session_state.show_new_project_input = False
                        st.rerun()

            projects = ProjectManager.get_all_projects()
            if projects:
                selected_project = st.selectbox(
                    "当前项目",
                    options=projects,
                    index=projects.index(self.current_project) if self.current_project in projects else 0,
                    key="project_selector"
                )
                if selected_project != self.current_project:
                    st.session_state.current_project = selected_project
                    st.rerun()
            else:
                st.warning("暂无项目，请先创建")

            st.divider()

            st.subheader("知识库上传")
            uploaded_files = st.file_uploader(
                "选择文件（可多选）",
                type=list(AppConfig.ALLOWED_FILE_TYPES),
                accept_multiple_files=True,
                key=f"file_uploader_{st.session_state.uploader_key}",
                # help="支持Word/PDF/Excel/PPT/Xmind等格式",
            )
            if st.button("上传", key="upload_btn", use_container_width=True):
                if uploaded_files:
                    success_count = 0
                    fail_count = 0
                    for uploaded_file in uploaded_files:
                        success = self.kb.upload_file(uploaded_file.name, uploaded_file.getvalue())
                        if success:
                            success_count += 1
                        else:
                            fail_count += 1
                    if success_count > 0:
                        st.success(f"成功上传 {success_count} 个文件")
                    if fail_count > 0:
                        st.warning(f"{fail_count} 个文件已存在，未重复上传")
                    st.session_state.uploader_key += 1
                    st.rerun()
                else:
                    st.warning("请先选择文件")

        self._kb_panel()
        st.divider()
        self._gen_panel(model_key)

    def _kb_panel(self):
        st.subheader(f"知识库 - {self.current_project}")
        files = self.kb.get_file_list()
        built_files = self.kb.get_built_files_safe()
        if files:
            for f in files:
                col1, col2, col3, _ = st.columns([0.5, 3, 0.5, 2])
                is_built = f['name'] in built_files
                if is_built:
                    col1.write("✅已构建")
                else:
                    col1.write("⏳未构建")
                col2.write(f"📄 {f['name']} ({f['size']})")
                delete_key = f"del_{self.current_project}_{f['name']}"
                if is_built:
                    with col3.popover("删除", use_container_width=False):
                        st.write(f"确定要删除文档 **{f['name']}** 吗？")
                        st.write("删除后，该文档构建的索引也会被清除。")
                        col_confirm, col_cancel = st.columns(2)
                        with col_confirm:
                            if st.button("确认删除", key=f"confirm_{delete_key}", type="primary"):
                                self._delete_file_and_rebuild(f['name'])
                                st.rerun()
                else:
                    if col3.button("删除", key=delete_key):
                        self._delete_file_and_rebuild(f['name'])
                        st.rerun()
        else:
            st.info("暂无文档，请先上传文件")

        col_btn1, col_btn2, _ = st.columns([1, 1, 6])
        with col_btn1:
            if st.button("构建知识库", type="primary", use_container_width=True):
                if not files:
                    st.warning("请先上传文件")
                else:
                    with st.status("正在构建知识库...", expanded=True) as status:
                        prog = st.progress(0)
                        stat = st.empty()

                        def cb(p, msg):
                            prog.progress(p)
                            stat.text(msg)

                        res = self.kb.build_knowledge_base(cb, force=False)
                        if res["status"] == "success":
                            status.update(state="complete")
                            st.toast(res['message'], icon="✅")
                        else:
                            status.update(label="", state="error")
                            st.toast(res['message'], icon="❌")
                    st.rerun()
        with col_btn2:
            if st.button("刷新索引", use_container_width=True):
                res = self.kb.refresh_index()
                if res["status"] == "success":
                    st.toast(res["message"], icon="🔄")
                else:
                    st.toast(res["message"], icon="⚠️")
                st.rerun()

    def _delete_file_and_rebuild(self, filename: str):
        self.kb.delete_file(filename)
        st.rerun()

    def _gen_panel(self, model_key):
        st.subheader("生成测试用例")

        # 初始化 session_state
        if "uploaded_file_content" not in st.session_state:
            st.session_state.uploaded_file_content = ""
        if "uploaded_file_name" not in st.session_state:
            st.session_state.uploaded_file_name = ""

        # 输入方式选择（需求描述 或 RDM单号）
        input_type = st.radio("输入方式", ["需求描述", "RDM单号"], horizontal=True, key="input_type_radio")

        prompt = ""

        if input_type == "需求描述":
            # 需求描述输入框
            text_prompt = st.text_area(
                "需求描述",
                height=200,
                max_chars=5000,
                key="prompt_text",
                placeholder="请在此输入详细的需求描述...\n\n示例： 配件搜索功能：\n   - 支持关键词搜索\n   - 支持价格区间筛选\n   - 搜索结果按相关性排序"
            )

            # 上传文档区域
            st.text("🔗 需求上传")
            uploaded_file = st.file_uploader(
                "选择需求文档",
                type=list(AppConfig.ALLOWED_FILE_TYPES),
                key="req_file_uploader",
                label_visibility="collapsed",
                help="上传需求并解析传给大模型生成测试用例"
            )

            if uploaded_file:
                with st.spinner(f"正在解析文档 {uploaded_file.name}..."):
                    try:
                        content_bytes = uploaded_file.getvalue()
                        file_content = self.kb._extract_text_from_bytes(uploaded_file.name, content_bytes)

                        if file_content.startswith("请安装") or file_content.startswith(
                                "暂无法识别") or file_content.startswith("提取失败"):
                            st.error(f"文档解析失败：{file_content}")
                            st.session_state.uploaded_file_content = ""
                            st.session_state.uploaded_file_name = ""
                        elif file_content:
                            st.session_state.uploaded_file_content = file_content
                            st.session_state.uploaded_file_name = uploaded_file.name

                            with st.expander(f"📄 文档内容预览（{uploaded_file.name}）"):
                                preview_length = min(len(file_content), 2000)
                                st.text(file_content[:preview_length] + (
                                    "..." if len(file_content) > preview_length else ""))
                                if len(file_content) > preview_length:
                                    st.caption(f"文档总长度：{len(file_content)} 字符，已截取前{preview_length}字符预览")
                    except Exception as e:
                        st.error(f"解析文档失败：{str(e)}")
                        st.session_state.uploaded_file_content = ""
                        st.session_state.uploaded_file_name = ""
            else:
                # 没有上传文件时，清空缓存
                if st.session_state.uploaded_file_content:
                    st.session_state.uploaded_file_content = ""
                    st.session_state.uploaded_file_name = ""

            # 组合内容：需求描述 + 文档内容（如果有）
            if text_prompt or st.session_state.uploaded_file_content:
                prompt = text_prompt
                if st.session_state.uploaded_file_content:
                    if prompt:
                        prompt += "\n\n---\n\n【补充文档内容】\n"
                    else:
                        prompt = "【补充文档内容】\n"
                    prompt += st.session_state.uploaded_file_content
                st.session_state.final_prompt = prompt
        else:
            # RDM单号输入   - 不显示上传文档按钮
            rdm_input = st.text_area(
                "RDM单号",
                height=200,
                key="rdm_input_area",
                placeholder="请输入RDM单号，多个单号请用逗号或换行分隔\n例如：DEMO-001, DEMO-002\n或：\nDEMO-001\nDEMO-002"
            )

            if rdm_input:
                # 提取RDM单号
                rdm_codes = RDMService.extract_rdm_codes(rdm_input)

                if rdm_codes:
                    st.info(f"✅ 检测到 {len(rdm_codes)} 个RDM单号：{', '.join(rdm_codes)}")

                    # 显示获取RDM内容的按钮
                    col_btn1, col_btn2 = st.columns([1, 5])
                    with col_btn1:
                        fetch_clicked = st.button("📥 获取RDM需求内容", key="fetch_rdm_content",
                                                  use_container_width=True)

                    if fetch_clicked:
                        rdm_contents = []
                        with st.spinner("正在获取RDM需求内容..."):
                            for rdm_code in rdm_codes:
                                result = RDMService.fetch_rdm_content(rdm_code)
                                if result["success"]:
                                    rdm_contents.append({
                                        "code": rdm_code,
                                        "content": result["content"],
                                        "url": result["url"]
                                    })
                                    st.success(f"✅ 成功获取 {rdm_code} 的内容")
                                else:
                                    st.error(f"❌ 获取 {rdm_code} 失败：{result['error']}")

                            if rdm_contents:
                                # 组合所有RDM内容
                                combined_content = "\n\n".join([
                                    f"【RDM单号：{item['code']}】\n{item['content']}"
                                    for item in rdm_contents
                                ])
                                st.session_state.rdm_prompt = combined_content
                                st.success(f"✅ 已获取 {len(rdm_contents)} 个RDM单号的需求内容")

                    # 如果已经获取过内容，显示预览
                    if "rdm_prompt" in st.session_state and st.session_state.rdm_prompt:
                        prompt = st.session_state.rdm_prompt
                        st.session_state.final_prompt = prompt
                        with st.expander("📋 RDM需求内容预览", expanded=False):
                            st.text(prompt[:1000] + ("..." if len(prompt) > 1000 else ""))
                else:
                    st.warning("未检测到有效的RDM单号格式（格式如：NJ_SPM-15530）")

        # RAG选项
        col_check, _ = st.columns([1, 6])
        with col_check:
            use_rag = st.checkbox(
                "启用RAG知识库",
                value=True,
                help="从当前项目的知识库中检索相关内容，提升生成质量"
            )

        # 生成和清空按钮
        col_gen, col_clear, col_status = st.columns([1, 1, 4])
        with col_gen:
            generate_clicked = st.button("生成测试用例", type="primary", use_container_width=True)
        with col_clear:
            clear_clicked = st.button("清空结果", use_container_width=True)

        # 显示提示信息
        with col_status:
            if prompt:
                st.success(f"✅ 内容已就绪 ({len(prompt)} 字符)")

        if clear_clicked:
            if "cases" in st.session_state:
                del st.session_state.cases
            if "rdm_prompt" in st.session_state:
                del st.session_state.rdm_prompt
            if "final_prompt" in st.session_state:
                del st.session_state.final_prompt
            st.rerun()

        if generate_clicked:
            if not prompt:
                if input_type == "需求描述":
                    st.error("请先输入需求描述")
                else:
                    st.error("请先输入RDM单号并获取内容")
                return

            with st.spinner("正在生成测试用例，请稍候..."):
                context = ""
                if use_rag:
                    files = self.kb.get_file_list()
                    if files:
                        if not self.kb.index_loaded:
                            with st.spinner("正在加载知识库索引..."):
                                self.kb.refresh_index()
                        if self.kb.index_loaded:
                            with st.expander("🔍 RAG检索详情", expanded=False):
                                results = self.kb.search_knowledge(prompt, top_k=5)
                                if results:
                                    st.write(f"找到 {len(results)} 个相关知识片段：")
                                    for i, r in enumerate(results, 1):
                                        with st.container():
                                            st.write(
                                                f"**片段 {i}** - 相似度: {r['similarity']:.3f} | 来源: {r['metadata'].get('source', '未知')}")
                                            st.caption(r['content'][:300] + ("..." if len(r['content']) > 300 else ""))
                                            st.divider()
                                else:
                                    st.info("未检索到相关知识，将基于模型自身知识生成")
                            context = self.kb.get_knowledge_context(prompt, max_chunks=5)
                        else:
                            st.info("知识库尚未构建，将直接使用模型生成。如需检索知识，请先构建知识库。")
                    else:
                        st.info("当前项目无文档，将直接使用模型生成。")
                # ========== 流式生成 ==========
                llm = LLMService(model_key)
                # 创建显示区域
                st.markdown("#### 📝 AI 生成过程")
                response_placeholder = st.empty()
                full_response = ""
                # 调用流式方法
                for chunk in llm.generate_cases_streaming(prompt, context):
                    full_response += chunk
                    # 实时显示当前已生成的内容（带光标效果）
                    response_placeholder.code(full_response + "▌", language="text")

                # 生成完成，移除光标并显示完整内容
                response_placeholder.code(full_response, language="text")
                resp = llm.generate_cases(prompt, context)
                # 生成完成，清除流式占位符
                response_placeholder.empty()
                # 将完整生成内容放入默认折叠的展开器中（作为思考过程记录）
                with st.expander("📜 查看生成过程（点击展开）", expanded=False):
                    st.code(full_response, language="text")

                # ========== 解析测试用例 ==========
                if resp["status"] == "error":
                    st.error(f"❌ {resp['message']}")
                else:
                    # 提取RDM单号
                    rdm_codes = RDMService.extract_rdm_codes(prompt)

                    cases = TestCaseService.parse(resp["content"], rdm_codes)
                    if cases:
                        st.session_state.cases = cases
                        st.success(f"成功解析出 {len(cases)} 个测试用例")
                    else:
                        st.warning("解析失败，显示原始内容")
                        st.code(resp["content"], language="text")

        # 显示结果
        if "cases" in st.session_state:
            self._show_results()

    @staticmethod
    def _show_results():
        cases = st.session_state.cases
        if not cases:
            return
        st.subheader("测试用例")
        # 创建DataFrame
        df = pd.DataFrame(cases)
        # 定义列顺序（确保优先级列存在）
        cols = ["RDM单号", "用例ID", "优先级","用例名称", "前置条件", "测试步骤", "预期结果"]
        df = df[[c for c in cols if c in df.columns]]
        st.markdown("""
        <style>
        .auto-wrap-table { width: 100%; border-collapse: collapse; }
        .auto-wrap-table th, .auto-wrap-table td { border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; white-space: normal; word-wrap: break-word; }
        .auto-wrap-table th:nth-child(1), .auto-wrap-table td:nth-child(1) { width: 100px; }
        .auto-wrap-table th:nth-child(2), .auto-wrap-table td:nth-child(2) { width: 60px; }
        .auto-wrap-table th:nth-child(3), .auto-wrap-table td:nth-child(3) { width: 60px; }
        .auto-wrap-table th:nth-child(4), .auto-wrap-table td:nth-child(4) { width: 220px; }
        .auto-wrap-table th:nth-child(5), .auto-wrap-table td:nth-child(5) { width: 160px; }
        .auto-wrap-table th:nth-child(6), .auto-wrap-table td:nth-child(6) { width: 320px; }
        .auto-wrap-table th:nth-child(7), .auto-wrap-table td:nth-child(7) { width: 320px; }
        </style>
        """, unsafe_allow_html=True)
        # 显示表格
        html_table = df.to_html(index=False, classes='auto-wrap-table', escape=False)
        st.markdown(html_table, unsafe_allow_html=True)
        # 导出按钮
        csv = ExportService.to_csv(cases)
        st.download_button(
            "导出 CSV",
            data=csv,
            file_name=f"cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )


# ====================== 入口 ======================
def main():
    MainView().render()


if __name__ == "__main__":
    main()