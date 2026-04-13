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
        self.config = AppConfig.MODELS.get(ident, AppConfig.MODELS["local"])

    @classmethod
    def _get_system_prompt(cls) -> str:
        """获取系统 Prompt（带缓存）"""
        if cls._SYSTEM_PROMPT_CACHE is None:
            cls._SYSTEM_PROMPT_CACHE = cls._build_system_prompt()
        return cls._SYSTEM_PROMPT_CACHE


    @staticmethod
    def _build_system_prompt() -> str:
        return """你是资深测试工程师，擅长仓储、SAP、CRM、制造业工业互联网。根据需求和知识库生成测试用例。
## 输出格式（严格遵循）
【思考过程】
- 需求分析：简述核心功能点
- 测试场景：列出需要覆盖的场景（正常、边界、异常、大数据等）
- 优先级判断：说明P0/P1/P2/P3的分配依据

【测试用例】
用例ID： TC001
RDM单号： DEMO-001
优先级： Highest
用例名称： 登录成功跳转首页（≤15字）
前置条件： 已注册账号（≤10字，无则填"无"）
测试步骤： 1. 打开"登录"页；2. 输入正确用户名密码；3. 点击"登录"
预期结果： 1. 页面正常；2. 输入正常；3. 跳转首页显示用户名

多个用例之间空行分隔。

## 重要原则
1. **完整性优先**：不要因为篇幅限制而减少测试用例数量，除非需求本身非常简单（有缩减需要注明）
2. **覆盖全面**：必须考虑正常场景、边界条件、异常情况、大数据量等测试场景

## 测试用例设计覆盖要求
- **正常场景**：主要业务流程，预期正常执行的用例
- **边界场景**：临界值、极限值、边界条件测试
- **异常场景**：错误输入、网络异常、权限不足、数据不存在等
- **大数据量场景**：大量数据查询、批量操作、性能相关测试

## 字段详细规则

### 用例名称
- 格式：简短一句话[模块/场景]下执行[操作]应[预期/校验点]。

### 优先级说明
- **Highest（核心）**：主流程必须通过，不通过则系统不可用
- **High（重要）**：重要功能，影响用户体验
- **Medium（一般）**：一般功能，边界和异常场景
- **Low（次要）**：非关键功能，大数据和极端场景

### 前置条件
- 只写**必要**的环境、状态或数据前提。
- 如果没有任何必要性前提，必须填"无"。
- 禁止写入测试步骤中才会使用的具体测试数据（如"用户名=test，密码=123"应写在步骤里）。
- 去掉所有"已经"、"需要"、"请确保"等虚词
- 不超过10字

### 测试步骤 & 预期结果（一一对应）
- 步骤与结果的数量必须相同，按相同数字编号一一对应。
- 若某步骤无预期结果（如中间过渡操作），预期结果对应编号写"无"。
- **步骤编写要求**：
  - 用"->"表示页面/弹窗/菜单的切换路径（例如：登录页->首页->设置中心）。
  - 描述清晰，包含具体操作数据（如输入值、点击坐标/元素）。
- **预期结果编写要求**：
  1. 优先依据需求说明中描述的操作结果。
  2. 若需求未说明，参考同类成熟产品的典型行为。
  3. 结果必须**肯定无疑义、可客观判定**（如"页面弹出提示'密码错误'；停留在当前页面"），禁止模糊描述（如"系统正常"）。
  
## 用例数量指引
### 简单需求（1-2个功能点）
- 用例数量：3-6个
- 覆盖：正常场景2-3个，边界1-2个，异常1-2个，大数据1-2个

### 中等需求（3-5个功能点）
- 用例数量：6-12个
- 覆盖：正常场景3-5个，边界2-3个，异常3-5个，大数据2-3个

### 复杂需求（5个以上功能点）
- 用例数量：10-30个
- 如需精简，按优先级输出，标注【核心用例】和【扩展用例】

## 重要禁止事项
- 不要在前置条件中写入测试数据（数据必须放在步骤里）。
- 不要使步骤和结果的数量不一致。

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

        max_tokens = 3000  #输出toknes
        temperature = 0.4  #创造性0-3

        for attempt in range(AppConfig.API_MAX_RETRIES):
            try:
                client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url,
                                timeout=AppConfig.API_TIMEOUT)
                messages = [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": f"需求：{prompt}\n\n知识库：\n{context}"}
                ]
                with ThreadPoolExecutor(max_workers=5) as executor:
                    future = executor.submit(
                        lambda: client.chat.completions.create(
                            model=self.config.model,
                            messages=messages, # type: ignore  #忽略该行类型检查
                            temperature=temperature,
                            max_tokens=max_tokens
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

    def generate_cases_streaming(self, prompt: str, context: str):
        """流式生成测试用例，yield 文本块"""
        if self.config.model == "local":
            yield self._local_generate()
            return

        max_tokens = 2000
        temperature = 0.4

        for attempt in range(AppConfig.API_MAX_RETRIES):
            try:
                client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
                messages = [
                    {"role": "system", "content": self._get_system_prompt()},  # 修复：使用正确的系统提示获取方法
                    {"role": "user", "content": f"需求：{prompt}\n\n知识库：\n{context}"}
                ]
                stream = client.chat.completions.create(# type: ignore
                    model=self.config.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
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