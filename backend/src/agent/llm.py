from openai import OpenAI, RateLimitError, APIError, NotFoundError
import os
import time

# 加载 .env 文件（如果存在）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class OpenRouterLLM:
    """通过 OpenRouter 统一路由所有 LLM 调用，支持多模型自动轮询"""

    def __init__(self, model_id: str = None):
        self.primary_model = model_id or os.getenv("DEFAULT_MODEL", "openai/gpt-3.5-turbo")
        # 解析备选模型列表
        fallback_str = os.getenv("FALLBACK_MODELS", "")
        self.fallback_models = [m.strip() for m in fallback_str.split(",") if m.strip()]
        self.all_models = [self.primary_model] + self.fallback_models
        
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
        
        # 轮询配置
        self.max_retries = len(self.all_models) * 2  # 每模型最多重试2次
        self.retry_delay = 2  # 基础重试间隔（秒）
        self.current_model_index = 0  # 当前使用的模型索引

    def _get_extra_headers(self):
        """OpenRouter 需要额外 headers"""
        return {
            "HTTP-Referer": "https://rag-teaching.local",
            "X-Title": "RAG Teaching Project",
        }

    def _try_generate(self, model: str, prompt: str, system: str, temperature: float = 0.7) -> str:
        """尝试用指定模型生成，失败则抛出异常"""
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            extra_headers=self._get_extra_headers(),
        )
        return response.choices[0].message.content

    def generate(self, prompt: str, system: str = "You are a helpful assistant.", temperature: float = 0.7) -> str:
        """
        生成回答，支持多模型自动轮询
        
        轮询策略:
        1. 先尝试主模型
        2. 主模型失败（429限流/404不可用/超时）→ 切换到备选模型1
        3. 备选1失败 → 切换到备选2 → ...
        4. 所有模型都失败 → 返回错误提示
        """
        last_error = None
        tried_models = []
        
        for attempt in range(self.max_retries):
            # 选择当前模型（轮询）
            model_index = attempt % len(self.all_models)
            model = self.all_models[model_index]
            
            if model in tried_models:
                # 已试过，增加等待时间
                wait_time = self.retry_delay * (attempt + 1)
                time.sleep(wait_time)
            
            tried_models.append(model)
            
            try:
                result = self._try_generate(model, prompt, system, temperature)
                # 成功！如果用了备选模型，打印日志
                if model != self.primary_model:
                    print(f"[LLM] 主模型不可用，已切换到备选模型: {model}")
                return result
                
            except RateLimitError as e:
                # 429 限流 → 立即切换模型
                retry_after = getattr(e, 'retry_after', None) or self.retry_delay
                print(f"[LLM] 模型 {model} 限流 (429)，等待 {retry_after}s 后切换...")
                last_error = e
                time.sleep(min(retry_after, 30))  # 最多等30秒
                continue
                
            except NotFoundError as e:
                # 404 模型不可用 → 立即切换
                print(f"[LLM] 模型 {model} 不可用 (404)，切换下一模型...")
                last_error = e
                continue
                
            except APIError as e:
                # 其他 API 错误 → 短暂等待后切换
                print(f"[LLM] 模型 {model} API错误: {e}, 切换...")
                last_error = e
                time.sleep(self.retry_delay)
                continue
                
            except Exception as e:
                # 未知错误 → 记录并切换
                print(f"[LLM] 模型 {model} 异常: {type(e).__name__}: {e}")
                last_error = e
                time.sleep(self.retry_delay)
                continue
        
        # 所有模型都失败
        error_msg = f"所有模型均不可用。已尝试: {', '.join(set(tried_models))}. 最后错误: {type(last_error).__name__}"
        print(f"[LLM] {error_msg}")
        raise RuntimeError(error_msg) from last_error

    def generate_with_fallback(self, prompt: str, system: str = "You are a helpful assistant.", temperature: float = 0.7) -> str:
        """生成回答，所有模型失败时返回友好错误信息（不抛异常）"""
        try:
            return self.generate(prompt, system, temperature)
        except RuntimeError as e:
            return f"抱歉，所有大模型当前均不可用。请稍后重试。错误: {str(e)[:200]}"
