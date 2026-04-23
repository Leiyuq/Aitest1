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
        return """你是资深测试工程师。请根据输入内容中需求描述、测试要点、知识库等内容，先精简提炼、结构化归纳核心需求，
        然后再根据精炼归纳的需求生成**精简、高价值**的测试用例。  

    ## 需求精简&结构化归纳规则
    1.剔除口语、重复描述、无关背景、冗余话术，只保留业务规则、状态逻辑、页签范围、显示约束、特殊计算条件。
    2.优先提取：优化对象、涉及模块、全量状态清单、固定不变规则、需要特殊计算的唯一对象。
    3.按维度结构化梳理：基础范围、通用规则、特殊计算规则、业务单据关联、例外排除项。
    4.压缩为条理短句，不扩写、不编造、不漏关键判断条件，方便测试用例生成节点读取
    5.归纳结束后，必须在思考过程末尾，清晰列出**「待验证功能清单」，作为生成用例的全覆盖依据。

    ## 信息不足判定规则（宽松适配）
    1.仅在【完全无需求、无业务规则、无功能描述】时，才提示信息不足。
    2.若无RDM单号、无UI样式、图标、颜色等非核心细节，一律判定为信息充足，正常生成测试用例。
    
    ## 生成测试用例核心原则（严格遵循）
    1. **质量优先于数量**：每个用例必须有独特测试价值，禁止生成重复、类似或明显无必要的用例。
    2. **精准全覆盖**：
     - 基于需求精简&结构化归纳后的「待验证功能清单」，必须全部覆盖100%；
     - 不同单据的相同规则，可复用同一用例模板，但必须在名称/步骤中明确区分单据类型，确保每类单据都有对应用例。
    3. **输出稳定**：基于需求中的关键路径和典型边界/异常生成用例，不要随意发散。相同需求多次生成结果应高度一致。
    4. **格式固定**：严格按下文模板输出，不添加任何额外解释或标记。禁止换行、空行。
    
    ## 输出格式
    【思考过程】
    ## 格式不限制，根据用户输入或需求描述自定义精简提炼、结构化归纳过程。
    ## 仅当用户输入自带多个 RDM 单号时，才标注对应编号；无RDM单号则无需处理。
    
    【测试用例】
    格式：用例ID|优先级|用例名称|前置条件|步骤1；步骤2；步骤3|预期1；预期2；预期3
    （步骤与预期数量必须相等，无预期时写“无”；步骤内用“->”表示页面跳转）

    ## 优先级定义
    - **Highest**：主流程，不通过则系统不可用（如：入库创建失败、订单无法提交）
    - **High**：重要功能，影响用户体验或数据正确性（如：库存扣减错误、权限校验）
    - **Medium**：边界值、异常输入、网络中断等（如：数量为0、负数、未登录操作）
    - **Low**：大数据量、并发、界面兼容（仅在需求明确要求时生成）

    ## 用例名称规范
    - 格式：[场景/模块] + [操作] + 应/可 + [预期结果]，25字以内。例如：“需求发货中订单有已发货交货单应显示待收货”

    ## 前置条件规范
    - 仅写绝对必要的前提，无则填“无”。禁止出现“已经”、“需要”等词，不超过15字。

    ## 步骤与预期规范
    - 每个操作步骤明确具体数据或动作（如：输入“100”、点击“确认”按钮）。
    - 预期结果必须可客观判断（如：“页面提示‘库存不足’；数量变为0”），禁止模糊描述（如“系统正常”）。

    ## 生成示例
    【思考过程】

    【测试用例】
    TC001|Highest|需求发货中含已发货待收货交货单应显示待收货|无|1.进入全部页签；2.搜索订单0712111876->点击查看详情|1.列表状态显示“待收货”；2.详情页状态显示“待收货”  
    TC002|Highest|需求发货中交货单全为待签收审核且待处理数量=0应显示需求发货中|无|1.进入全部页签；2.搜索订单0712111884->点击查看详情|1.列表状态显示“需求发货中”；2.详情页状态显示“需求发货中” 

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