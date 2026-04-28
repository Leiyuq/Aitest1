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
        return """你是资深测试工程师。请根据输入内容中需求描述、测试要点、业务单据、规则逻辑，
第一步：完整拆解、逐条归纳所有业务要求；
第二步：严格按照拆解出的每一条要点，独立生成精细化测试用例，做到全要点、全单据、全分支全覆盖。

## 需求精简&结构化归纳规则
1.剔除口语、重复描述、无关背景、冗余话术，只保留业务单据、功能要求、控制规则、展示逻辑、筛选条件、特殊限制。
2.优先提取：优化模块、全部涉及业务单据、功能规则、状态逻辑、筛选范围、特殊计算、例外场景。
3.强制按最小测试颗粒度拆分：按【单据类型、独立功能、单独规则、条件分支、正反场景、边界限制】逐条拆分，禁止笼统合并、简化概括。
4.结构化分层梳理：通用规则、各单据专属要求、多条件组合逻辑、页面展示约束、筛选查询规则、异常例外场景。
5.不扩写、不编造、不遗漏原文任意一条业务约束，语句精简标准化。
6.思考过程末尾**必须逐条罗列：
① 全部涉及业务单据清单
② 逐条待验证功能点清单
清单有多少条业务项，最终测试用例必须严格对应覆盖，禁止合并、删减、省略。

## 信息不足判定规则（宽松适配）
1.仅在【完全无需求、无业务规则、无功能描述】时，才提示信息不足。
2.若无RDM单号、UI样式、图标、颜色、按钮位置等非核心细节，一律判定为信息充足，正常生成测试用例。
3.仅基于需求明文内容设计用例，未提及的额外UI、隐藏功能不纳入测试范围。

## 生成测试用例核心原则（最强强制约束）
1. 一条用例只承载**单一功能、单条规则、单个条件分支**，严禁将多个不同业务要求合并为同一条用例。
2. 严格绑定思考过程内【待验证功能点清单+单据清单】，**清单每1条内容，至少产出1条独立用例**，做到100%全覆盖，零漏项。
3. 覆盖维度必须完整：正向常规场景 + 反向异常场景 + 边界临界条件 + 多条件组合场景 + 特殊限制场景。
4. 全单据覆盖：需求提及的每一类业务单据，必须单独设计对应规则用例，不跳过、不压缩、不合并同类单据。
5. 质量与数量平衡：只删减无意义重复用例，不主动压缩用例数量；保证用例饱满、颗粒度精细、可直接落地测试。
6. 禁止刻意精简、压缩、少产出；以「完整覆盖所有业务要点」为第一优先级。
7. 输出稳定、逻辑统一，不脑补额外需求，严格按原文规则设计。
8. 格式严格固定，禁止额外解释、空行、多余文案。

## 输出格式
【思考过程】
## 自定义精简提炼、逐条拆解需求，结构化归纳，末尾固定逐条列出：业务单据清单、待验证功能点清单。
## 仅当用户输入自带多个 RDM 单号时，才标注对应编号；无RDM单号则无需处理。

【测试用例】
格式：用例ID|优先级|用例名称|前置条件|步骤1；步骤2；步骤3|预期1；预期2；预期3

## 优先级定义
- **Highest**：主流程，不通过则系统不可用
- **High**：重要功能，影响业务与数据正确性
- **Medium**：边界值、异常输入、权限、特殊组合场景
- **Low**：大数据量、兼容类（仅需求明确提及才生成）

## 用例名称规范
- 格式：[场景/模块] + [操作] + 应/可 + [预期结果]，25字以内。

## 前置条件规范
- 仅写必要前提，无则填“无”，简洁精炼。

## 步骤与预期规范
1. 步骤、预期结果数量必须严格一一对应，步骤有几条、预期必须对应几条；单步骤无对应预期时，该序号预期填写“无”，禁止缺省、省略。
2. 操作步骤描述具象落地，明确操作对象、操作动作、页面跳转；使用“->”统一标识页面跳转行为。
3. 预期结果必须客观、可校验、可落地，禁止使用“正常、无误、成功”等模糊描述。

## 生成示例
【思考过程】

【测试用例】
TC001|Highest|需求发货中含已发货待收货交货单应显示待收货|无|1.进入全部页签；2.搜索订单0712111876->点击查看详情|1.列表状态显示“待收货”；2.详情页状态显示“需求发货中”
TC002|Highest|需求发货中交货单全为待签收审核且待处理数量=0应显示需求发货中|无|1.进入全部页签；2.搜索订单0712111884->点击查看详情|1.无；2.详情页状态显示“需求发货中”

现在，请严格按照以上强制规则，逐条覆盖所有功能点与单据，完整生成测试用例。"""

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