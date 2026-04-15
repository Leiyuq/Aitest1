from dataclasses import dataclass

# ====================== 配置 ======================
@dataclass
class ModelConfig:
    name: str
    api_key: str
    base_url: str
    model: str
    max_tokens = 32000  # 输出tokens
    temperature = 0.1  # 创造性0-3

class AppConfig:
    PAGE_TITLE = "测试用例智构系统"
    LAYOUT = "wide"
    BASE_ROOT = "knowledge_projects"  # OSS 中的根目录前缀
    ALLOWED_FILE_TYPES = {
        "txt", "csv", "docx", "doc", "pdf", "xlsx", "xls", "pptx", "xmind", "md"
    }
    RAG_TOP_K = 5   # 检索最大数量
    RAG_SIMILARITY_THRESHOLD = 0.2   # 检索匹配最低相识度
    API_TIMEOUT = 60
    API_MAX_RETRIES = 2
    API_RETRY_DELAY = 2
    RDM_BASE_URL = "http://rdm.zvos.zoomlion.com/browse/"  # RDM跳转地址

    MODELS = {

        "plus": ModelConfig(
            name="通义千问",
            api_key="sk-02c7ef707efa48e6a5555eabbc54aaa8",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-plus"  #相比turbo，推理更强，tokens更大
        ),
        "local": ModelConfig(
            name="本地离线",
            api_key="none",
            base_url="none",
            model="local"
        )
    }

    @classmethod
    def get_model_list(cls):
        return [f"{m.name}({k})" for k, m in cls.MODELS.items() if m.api_key or k == "local"]