import os
import sys

# 加载 .env 文件
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

from openai import OpenAI


def test_openrouter_connection():
    """测试 OpenRouter 连接是否可用"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("DEFAULT_MODEL", "openai/gpt-3.5-turbo")
    
    print(f"API Key: {api_key[:10]}...{api_key[-4:]}")
    print(f"Model: {model}")
    print(f"Base URL: https://openrouter.ai/api/v1")
    print("-" * 40)
    
    if not api_key or api_key == "your_openrouter_api_key_here":
        print("❌ API Key 未配置！")
        return False
    
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        
        print("🔄 正在测试连接...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'OpenRouter connection is working!' in Chinese."},
            ],
            temperature=0.7,
            max_tokens=50,
        )
        
        result = response.choices[0].message.content
        print(f"✅ 连接成功！")
        print(f"📝 响应内容: {result}")
        print(f"🔢 使用模型: {model}")
        return True
        
    except Exception as e:
        print(f"❌ 连接失败: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_openrouter_connection()
    sys.exit(0 if success else 1)
