import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from config import AppConfig


# ====================== LLM 服务 ======================
class LLMService:
    # 缓存系统 Prompt（避免重复构建字符串）
    _SYSTEM_PROMPT_CACHE = None

    def __init__(self, model_key: str):
        ident = model_key.split("(")[-1].rstrip(")")
        # 赋值给 self.config
        self.config = AppConfig.MODELS.get(ident, AppConfig.MODELS["local"])

    @classmethod
    def _get_system_prompt(cls) -> str:
        """获取系统 Prompt（带缓存）"""
        if cls._SYSTEM_PROMPT_CACHE is None:
            cls._SYSTEM_PROMPT_CACHE = cls._build_system_prompt()
        return cls._SYSTEM_PROMPT_CACHE

    @staticmethod
    def _build_system_prompt() -> str:
        return """你是资深测试工程师。请根据【需求内容】（包含需求描述和需求文档内容）、【测试要点】（如提供，请重点覆盖）和【知识库】中的业务规则、历史测试用例等信息，生成**精简、高价值**的测试用例。
    
    ## 核心原则（严格遵循）
    1. **质量优先于数量**：每个用例必须有独特测试价值，**禁止**生成重复、类似或明显无必要的用例（例如：仅当需求明确要求性能测试时才生成大数据用例）。
    2. **输出稳定**：基于需求中的关键路径和典型边界/异常生成用例，不要随意发散。相同需求多次生成结果应高度一致。
    3. **格式固定**：严格按下文模板输出，不添加任何额外解释或标记。禁止换行、空行。

    ## 输出格式
    【思考过程】
    - 功能点拆解：列出本需求涉及的独立功能点
    - 关键路径：主流程
    - 风险点：需要验证的边界/异常

    【测试用例】
    格式：用例ID|优先级|用例名称|前置条件|步骤1；步骤2；步骤3|预期1；预期2；预期3
    （步骤与预期数量必须相等，无预期时写“无”；步骤内用“->”表示页面跳转）

    ## 用例数量与分布（硬性约束）
    - 总用例数 = 功能点数量 × 2 至 4。
    - 优先级分布：P0（核心流程）占30%，P1（重要功能）占40%，P2（边界/异常）占20%，P3（性能/兼容）占10%。
    - 大数据/批量操作类用例：**最多1个**，且仅当需求明确提到“批量”、“大数据”、“性能”时才生成，否则跳过。

    ## 优先级定义
    - **Highest**：主流程，不通过则系统不可用（如：入库创建失败、订单无法提交）
    - **High**：重要功能，影响用户体验或数据正确性（如：库存扣减错误、权限校验）
    - **Medium**：边界值、异常输入、网络中断等（如：数量为0、负数、未登录操作）
    - **Low**：大数据量、并发、界面兼容（仅在需求明确要求时生成）

    ## 用例名称规范
    - 格式：[场景/模块] + [操作] + 应/可 + [预期结果]，15字以内。例如：“创建销售申请单选择入库单提交应成功”

    ## 前置条件规范
    - 仅写绝对必要的前提，无则填“无”。禁止出现“已经”、“需要”等词，不超过10字。

    ## 步骤与预期规范
    - 每个操作步骤明确具体数据或动作（如：输入“100”、点击“确认”按钮）。
    - 预期结果必须可客观判断（如：“页面提示‘库存不足’；数量变为0”），禁止模糊描述（如“系统正常”）。

    ## 示例（仓储场景）
    【思考过程】
    - 功能点拆解：创建入库单、提交审核
    - 关键路径：填写信息->提交成功
    - 风险点：必填项为空、数量为0

    【测试用例】
    TC001|Highest|提交入库单应成功|已登录且SKU存在|1.打开入库单创建页；2.输入SKU编码“A001”->输入数量“100”->点击提交|1.无；2.页面提示“提交成功”；生成入库单号
    TC002|High|数量为0提交应失败|已登录|1.打开入库单创建页；2.输入SKU编码“A001”->输入数量“0”->点击提交|1.无；2.页面提示“数量必须大于0”；提交按钮不可点
    TC003|High|SKU不存在提交应失败|已登录|1.打开入库单创建页；2.输入SKU编码“XYZ999”->输入数量“10”->点击提交|1.无；2.页面提示“SKU不存在”；未生成入库单

    现在，请根据以上规则生成测试用例。"""

    @staticmethod
    def _local_generate():
        return """
用例ID： TC001
RDM单号： DEMO-001
优先级： Highest
用例名称： 功能验证
前置条件： 环境正常
测试步骤： 1.操作;2.验证
预期结果： 1.成功;2.正常"""

    def generate_cases(self, prompt: str, context: str) -> dict:
        if self.config.model != "local" and not self.config.api_key:
            return {"status": "error", "message": f"未配置 {self.config.name} API Key"}
        if self.config.model == "local":
            return {"status": "success", "content": self._local_generate(), "message": "本地生成"}

        for attempt in range(AppConfig.API_MAX_RETRIES):
            try:
                client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url,
                                timeout=AppConfig.API_TIMEOUT)
                messages = [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": f"需求：{prompt}\n\n知识库：\n{context}"}
                ]
                with ThreadPoolExecutor(max_workers=5) as executor:  #多线程
                    future = executor.submit(
                        lambda: client.chat.completions.create(
                            model=self.config.model,
                            messages=messages,  # type: ignore  #忽略该行类型检查
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_tokens
                        )
                    )
                    resp = future.result(timeout=AppConfig.API_TIMEOUT)
                return {"status": "success", "content": resp.choices[0].message.content.strip(),
                        "message": "生成成功"}
            except FuturesTimeoutError:
                if attempt < AppConfig.API_MAX_RETRIES - 1:
                    time.sleep(AppConfig.API_RETRY_DELAY)
                    continue
                return {"status": "error", "message": "请求超时，请稍后重试"}
            except Exception as e:
                if attempt == AppConfig.API_MAX_RETRIES - 1:
                    return {"status": "error", "message": f"生成失败: {str(e)}"}
                time.sleep(AppConfig.API_RETRY_DELAY)
        return {"status": "error", "message": "未知错误"}

    # 调用流式生成接口，prompt 为用户需求，context 为 RAG 检索到的知识片段
    def generate_cases_streaming(self, prompt: str, context: str):
        """流式生成测试用例，yield 文本块"""
        if self.config.model == "local":
            yield self._local_generate()
            return

        for attempt in range(AppConfig.API_MAX_RETRIES):
            try:
                client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
                messages = [
                    {"role": "system", "content": self._get_system_prompt()},  # 修复：使用正确的系统提示获取方法
                    {"role": "user", "content": f"需求：{prompt}\n\n知识库：\n{context}"}  #用户输入需求prompt和检索到的知识库内容context拼接到用户消息中content，最终输入
                ]
                stream = client.chat.completions.create(  # type: ignore
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    stream=True
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return  # 成功结束
            except Exception as e:
                if attempt == AppConfig.API_MAX_RETRIES - 1:
                    yield f"\n\n❌ 生成失败: {str(e)}"
                else:
                    time.sleep(AppConfig.API_RETRY_DELAY)
                    continue