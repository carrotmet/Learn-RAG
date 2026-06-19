"""
RAG 教学项目 — 虚拟数据生成与测试

参考: /home/ubuntu/.openclaw/workspace/RAG教学/docs/监控指标体系搭建规划.md 2.1-2.5 节

功能:
- 生成 10-30 条不同类型的虚拟数据
- 测试数据采集模块的完整功能
- 验证建表、写入、查询、导出等操作

运行方式:
    python backend/src/data_collection/demo_data.py

测试内容:
1. 建表验证 — 检查所有表是否创建成功
2. 在线采集 — 模拟 API 调用写入对话记录
3. 离线上传 — 模拟 JSON/JSONL/CSV 上传
4. 用户反馈 — 模拟点赞/点踩/评论
5. 检索日志 — 模拟检索结果记录
6. LLM 调用 — 模拟 LLM 调用记录
7. 数据查询 — 查询统计和详情
8. 数据导出 — 导出 JSONL 测试集
"""

import json
import os
import sys
from datetime import datetime, timedelta
import random

# 将项目根目录加入路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 直接导入当前目录模块（避免 backend 包路径问题）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlite_store import SQLiteCollector
from uploader import DataUploader
from config import CollectionConfig


# ==================== 虚拟数据生成器 ====================

class DemoDataGenerator:
    """生成 10-30 条不同类型的虚拟 RAG 数据"""
    
    # 预定义的领域和问题类型
    DOMAINS = ['自指学', '数学', 'AI', '其他']
    QUESTION_TYPES = ['factual', 'comparative', 'procedural', 'open']
    DIFFICULTIES = ['easy', 'medium', 'hard']
    
    # 自指学相关问题模板
    ZIZHI_QUESTIONS = [
        {
            'question': '什么是自指学？',
            'question_type': 'factual',
            'domain': '自指学',
            'difficulty': 'easy',
            'contexts': [
                '自指学（Self-reference）是研究系统如何引用自身的学科。它涉及逻辑、语言学和计算机科学等多个领域。'
            ],
            'answer': '自指学是研究系统如何引用自身的学科，涉及逻辑、语言学和计算机科学等领域。',
            'ground_truth': '自指学（Self-reference）是研究系统如何引用自身的学科，涉及逻辑学、语言学、计算机科学和数学等多个领域。'
        },
        {
            'question': '哥德尔不完备定理和自指有什么关系？',
            'question_type': 'comparative',
            'domain': '自指学',
            'difficulty': 'hard',
            'contexts': [
                '哥德尔不完备定理证明了在任何足够强大且一致的形式系统中，存在既不能被证明也不能被否定的命题。',
                '哥德尔通过构造一个自指的命题"这个命题不能被证明"来证明了不完备定理。'
            ],
            'answer': '哥德尔不完备定理通过构造自指命题"这个命题不能被证明"来证明，展示了自指在数学逻辑中的重要作用。',
            'ground_truth': '哥德尔不完备定理通过构造一个自指命题来证明，即"这个命题在此系统中不可证明"。这种自指构造是证明的核心技巧。'
        },
        {
            'question': '如何构造一个自指的程序？',
            'question_type': 'procedural',
            'domain': '自指学',
            'difficulty': 'medium',
            'contexts': [
                '在计算机科学中，自指程序可以通过Quine（自产生程序）来实现。Quine是一个输出自身源代码的程序。',
                '构造Quine的基本方法是将程序分为数据和代码两部分，数据部分包含代码的字符串表示，代码部分输出数据。'
            ],
            'answer': '构造自指程序可以使用Quine技术：将程序分为数据和代码两部分，数据部分存储代码的字符串表示，代码部分输出数据。',
            'ground_truth': '构造自指程序（Quine）的基本方法：1. 将程序分为数据和代码两部分；2. 数据部分包含代码的字符串表示；3. 代码部分输出数据；4. 通过适当的编码和解码实现自引用。'
        },
        {
            'question': '自指在人工智能中有什么应用？',
            'question_type': 'open',
            'domain': '自指学',
            'difficulty': 'medium',
            'contexts': [
                '自指在AI中可用于元学习（learning to learn），即让AI系统学习如何改进自己的学习算法。',
                '自指也与AI安全相关，例如AI系统需要理解自身的能力和局限性以避免危险行为。'
            ],
            'answer': '自指在AI中应用于元学习、AI安全、自我改进系统等方面，帮助AI理解自身能力和局限性。',
            'ground_truth': '自指在AI中的应用包括：1. 元学习（学习如何学习）；2. AI安全（理解自身能力边界）；3. 自我改进系统；4. 反思性AI（能够推理自身推理过程）。'
        },
    ]
    
    # 数学相关问题
    MATH_QUESTIONS = [
        {
            'question': '什么是贝叶斯定理？',
            'question_type': 'factual',
            'domain': '数学',
            'difficulty': 'easy',
            'contexts': [
                '贝叶斯定理是概率论中的一个基本定理，描述了在已知某些条件下事件发生的概率。',
                'P(A|B) = P(B|A) * P(A) / P(B)，其中P(A|B)是在B发生的条件下A发生的后验概率。'
            ],
            'answer': '贝叶斯定理描述了在已知条件下事件发生的概率，公式为 P(A|B) = P(B|A) * P(A) / P(B)。',
            'ground_truth': '贝叶斯定理：P(A|B) = P(B|A) * P(A) / P(B)。它描述了在观察到证据B后，更新假设A概率的方法。'
        },
        {
            'question': '比较梯度下降和牛顿法的优缺点',
            'question_type': 'comparative',
            'domain': '数学',
            'difficulty': 'hard',
            'contexts': [
                '梯度下降法只使用一阶导数信息，每次迭代沿负梯度方向更新参数，计算简单但收敛慢。',
                '牛顿法使用二阶导数（Hessian矩阵）信息，收敛速度快但需要计算和存储Hessian矩阵，计算复杂度高。'
            ],
            'answer': '梯度下降计算简单但收敛慢；牛顿法收敛快但计算复杂度高，需要Hessian矩阵。',
            'ground_truth': '梯度下降：优点-计算简单、内存需求低；缺点-收敛慢、需要调学习率。牛顿法：优点-收敛快、二阶精度；缺点-需要Hessian矩阵、计算复杂度高、可能不收敛。'
        },
    ]
    
    # AI 相关问题
    AI_QUESTIONS = [
        {
            'question': '什么是Transformer架构？',
            'question_type': 'factual',
            'domain': 'AI',
            'difficulty': 'easy',
            'contexts': [
                'Transformer是一种基于自注意力机制的神经网络架构，由Vaswani等人在2017年提出。',
                'Transformer完全基于注意力机制，不需要循环或卷积，可以并行处理序列数据。'
            ],
            'answer': 'Transformer是基于自注意力机制的神经网络架构，可以并行处理序列数据，不需要循环或卷积。',
            'ground_truth': 'Transformer是2017年提出的基于自注意力机制的神经网络架构，核心创新包括：多头注意力、位置编码、编码器-解码器结构，完全摒弃了RNN和CNN。'
        },
        {
            'question': '如何微调大语言模型以适应特定任务？',
            'question_type': 'procedural',
            'domain': 'AI',
            'difficulty': 'medium',
            'contexts': [
                '微调大语言模型的常用方法包括：全参数微调、LoRA（低秩适配）、Prompt Tuning、Prefix Tuning等。',
                'LoRA通过在原始权重矩阵旁添加低秩矩阵来进行微调，只训练少量参数，效率更高。'
            ],
            'answer': '微调大语言模型可使用全参数微调、LoRA、Prompt Tuning等方法，其中LoRA只训练少量参数，效率更高。',
            'ground_truth': '微调大语言模型的方法：1. 全参数微调-更新所有参数；2. LoRA-添加低秩适配矩阵；3. Prompt Tuning-优化输入提示；4. Prefix Tuning-优化前缀嵌入；5. Adapter-添加小型适配层。'
        },
        {
            'question': 'RAG和微调有什么区别，什么时候用哪个？',
            'question_type': 'comparative',
            'domain': 'AI',
            'difficulty': 'medium',
            'contexts': [
                'RAG（检索增强生成）通过检索外部知识来增强生成，不需要修改模型参数，适合知识更新频繁的场景。',
                '微调通过训练数据调整模型参数，使模型学习特定任务或风格，适合需要改变模型行为的情况。'
            ],
            'answer': 'RAG通过检索外部知识增强生成，适合知识更新频繁的场景；微调通过训练调整模型参数，适合改变模型行为。',
            'ground_truth': 'RAG vs 微调：RAG优点-无需训练、知识可更新、减少幻觉；缺点-依赖检索质量。微调优点-改变模型行为、特定任务优化；缺点-需要训练数据、知识固定。选择：知识更新频繁→RAG；需要改变行为→微调。'
        },
    ]
    
    # 其他领域问题
    OTHER_QUESTIONS = [
        {
            'question': '什么是Python中的装饰器？',
            'question_type': 'factual',
            'domain': '其他',
            'difficulty': 'easy',
            'contexts': [
                '装饰器是Python中的一种语法糖，允许在不修改函数定义的情况下扩展函数功能。',
                '@decorator 语法等价于 function = decorator(function)。'
            ],
            'answer': 'Python装饰器是一种语法糖，允许在不修改函数定义的情况下扩展函数功能，使用@decorator语法。',
            'ground_truth': 'Python装饰器是用于修改或增强函数/类行为的高阶函数。语法：@decorator 等价于 func = decorator(func)。常见用途：日志记录、权限检查、缓存、性能计时。'
        },
        {
            'question': '如何优化SQLite数据库的查询性能？',
            'question_type': 'procedural',
            'domain': '其他',
            'difficulty': 'medium',
            'contexts': [
                'SQLite性能优化方法包括：创建合适的索引、使用EXPLAIN分析查询、避免全表扫描、使用事务批量操作。',
                '索引可以显著提高查询速度，但会增加写入开销和存储空间，需要权衡。'
            ],
            'answer': '优化SQLite查询性能的方法：创建索引、使用EXPLAIN分析查询、避免全表扫描、使用事务批量操作。',
            'ground_truth': 'SQLite优化方法：1. 创建索引（WHERE/JOIN/ORDER BY列）；2. 使用EXPLAIN QUERY PLAN分析；3. 避免SELECT *；4. 使用事务批量插入；5. 使用VACUUM整理碎片；6. 适当使用PRAGMA优化。'
        },
    ]
    
    def __init__(self):
        self.all_questions = (
            self.ZIZHI_QUESTIONS + 
            self.MATH_QUESTIONS + 
            self.AI_QUESTIONS + 
            self.OTHER_QUESTIONS
        )
    
    def generate_conversations(self, count: int = 15) -> list:
        """生成虚拟对话记录
        
        生成 count 条记录，覆盖不同领域、题型和难度。
        如果 count > 预定义数量，会循环使用并添加随机变化。
        """
        conversations = []
        
        for i in range(count):
            base = self.all_questions[i % len(self.all_questions)]
            
            # 添加随机变化，使数据更真实
            conv = {
                'question': base['question'],
                'answer': base['answer'],
                'contexts': base['contexts'],
                'ground_truth': base['ground_truth'],
                'model_version': random.choice(['v1.0.0', 'v1.1.0', 'v1.2.0']),
                'source': random.choice(['online_api', 'offline_upload', 'batch_export']),
                'metadata': {
                    'question_type': base['question_type'],
                    'domain': base['domain'],
                    'difficulty': base['difficulty'],
                    'session_id': f'session_{random.randint(1000, 9999)}',
                    'user_id': f'user_{random.randint(1, 100)}',
                    'timestamp': (datetime.now() - timedelta(hours=random.randint(0, 168))).isoformat()
                }
            }
            conversations.append(conv)
        
        return conversations
    
    def generate_feedbacks(self, conv_ids: list) -> list:
        """为对话生成用户反馈"""
        feedbacks = []
        
        feedback_types = ['thumbs_up', 'thumbs_down', 'comment', 'correction']
        
        for conv_id in conv_ids:
            # 70% 概率有反馈
            if random.random() < 0.7:
                fb_type = random.choice(feedback_types)
                
                if fb_type == 'thumbs_up':
                    content = random.choice(['很好', '有帮助', '准确', '感谢'])
                    rating = random.choice([4, 5])
                elif fb_type == 'thumbs_down':
                    content = random.choice(['不准确', '没帮助', '太简略', '错误'])
                    rating = random.choice([1, 2])
                elif fb_type == 'comment':
                    content = random.choice([
                        '希望能更详细一些',
                        '例子不够具体',
                        '术语解释不清楚',
                        '整体不错，但缺少实践案例'
                    ])
                    rating = random.choice([3, 4])
                else:  # correction
                    content = random.choice([
                        '公式有误，应该是 P(A|B) = P(B|A)P(A)/P(B)',
                        '年份应该是2017年不是2018年',
                        '缺少对边缘情况的讨论'
                    ])
                    rating = 3
                
                feedbacks.append({
                    'conversation_id': conv_id,
                    'feedback_type': fb_type,
                    'content': content,
                    'rating': rating
                })
        
        return feedbacks
    
    def generate_retrieval_logs(self, conv_ids: list) -> list:
        """为对话生成检索日志"""
        logs = []
        
        for conv_id in conv_ids:
            # 80% 概率有检索日志
            if random.random() < 0.8:
                num_chunks = random.randint(1, 5)
                chunks = []
                scores = []
                
                for j in range(num_chunks):
                    chunks.append({
                        'content': f'检索到的文档片段 {j+1}，包含相关知识点...',
                        'source': f'doc_{random.randint(1, 100)}.pdf',
                        'page': random.randint(1, 50),
                        'score': round(random.uniform(0.6, 0.98), 3)
                    })
                    scores.append(round(random.uniform(0.6, 0.98), 3))
                
                logs.append({
                    'conversation_id': conv_id,
                    'query': f'query_for_{conv_id[:8]}',
                    'retrieved_chunks': chunks,
                    'scores': scores,
                    'latency_ms': random.randint(10, 500)
                })
        
        return logs
    
    def generate_llm_calls(self, conv_ids: list) -> list:
        """为对话生成LLM调用记录"""
        calls = []
        
        models = ['gpt-4o-mini', 'kimi-k2.6', 'claude-3-haiku', 'deepseek-chat']
        
        for conv_id in conv_ids:
            # 90% 概率有LLM调用
            if random.random() < 0.9:
                prompt_tokens = random.randint(500, 3000)
                completion_tokens = random.randint(100, 1500)
                
                calls.append({
                    'conversation_id': conv_id,
                    'prompt': f'基于以下文档回答问题：\n[文档内容]\n\n问题：[用户问题]',
                    'response': '这是LLM生成的回答内容...',
                    'model_name': random.choice(models),
                    'token_usage': {
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'total': prompt_tokens + completion_tokens
                    },
                    'latency_ms': random.randint(500, 5000)
                })
        
        return calls
    
    def generate_raw_uploads(self, count: int = 5) -> list:
        """生成模拟离线上传的原始数据"""
        uploads = []
        
        for i in range(count):
            # 模拟不同格式的原始数据
            raw = {
                'q': f'上传的问题 {i+1}',
                'a': f'上传的答案 {i+1}',
                'ctx': [f'上下文片段 {j+1}' for j in range(random.randint(1, 3))],
                'gt': f'标准答案 {i+1}',
                'extra_field': '额外字段'  # 测试字段映射
            }
            
            uploads.append({
                'raw_content': raw,
                'source_type': 'offline_upload',
                'original_format': 'json',
                'metadata': {
                    'uploader': f'user_{random.randint(1, 10)}',
                    'upload_time': datetime.now().isoformat()
                }
            })
        
        return uploads


# ==================== 测试执行器 ====================

class DataCollectionTester:
    """数据采集模块测试器"""
    
    def __init__(self, db_path: str = "data/rag_data.db"):
        self.db_path = db_path
        self.config = CollectionConfig(db_path=db_path)
        self.collector = SQLiteCollector(db_path, self.config)
        self.uploader = DataUploader(db_path, self.config)
        self.generator = DemoDataGenerator()
        
        self.test_results = []
    
    def _log(self, step: str, status: str, details: str = ""):
        """记录测试步骤"""
        result = {
            'step': step,
            'status': status,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        icon = '✅' if status == 'PASS' else '❌' if status == 'FAIL' else '⏳'
        print(f"{icon} [{status}] {step}")
        if details:
            print(f"   {details}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("RAG 教学项目 — 数据采集模块测试")
        print("=" * 60)
        print()
        
        try:
            # 1. 建表验证
            self._test_table_creation()
            
            # 2. 在线采集测试
            self._test_online_collection()
            
            # 3. 离线上传测试
            self._test_offline_upload()
            
            # 4. 用户反馈测试
            self._test_user_feedback()
            
            # 5. 检索日志测试
            self._test_retrieval_logs()
            
            # 6. LLM调用测试
            self._test_llm_calls()
            
            # 7. 数据查询测试
            self._test_data_query()
            
            # 8. 数据导出测试
            self._test_data_export()
            
            # 9. 统计报告
            self._print_statistics()
            
        except Exception as e:
            self._log("测试执行", "FAIL", str(e))
            raise
        
        # 打印测试总结
        self._print_summary()
    
    def _test_table_creation(self):
        """测试1: 建表验证"""
        print("\n" + "-" * 40)
        print("测试1: 数据库建表验证")
        print("-" * 40)
        
        try:
            # 检查数据库文件是否存在
            assert os.path.exists(self.db_path), f"数据库文件未创建: {self.db_path}"
            
            # 检查所有表是否存在
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            expected_tables = [
                'conversations', 'retrieval_logs', 'llm_calls',
                'user_feedback', 'raw_data', 'processed_data'
            ]
            
            for table in expected_tables:
                if table in tables:
                    self._log(f"表 {table}", "PASS", "已创建")
                else:
                    self._log(f"表 {table}", "FAIL", "未创建")
            
            # 检查索引
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            self._log("索引创建", "PASS", f"共 {len(indexes)} 个索引")
            
        except Exception as e:
            self._log("建表验证", "FAIL", str(e))
    
    def _test_online_collection(self):
        """测试2: 在线采集 — 模拟API调用写入"""
        print("\n" + "-" * 40)
        print("测试2: 在线采集测试")
        print("-" * 40)
        
        try:
            # 生成 15 条虚拟对话
            conversations = self.generator.generate_conversations(15)
            
            saved_ids = []
            for i, conv in enumerate(conversations, 1):
                conv_id = self.collector.save_conversation(
                    question=conv['question'],
                    answer=conv['answer'],
                    contexts=conv['contexts'],
                    ground_truth=conv['ground_truth'],
                    model_version=conv['model_version'],
                    source=conv['source'],
                    metadata=conv['metadata']
                )
                saved_ids.append(conv_id)
                
                if i <= 3:  # 只打印前3条
                    print(f"   写入 [{i}/15]: {conv['question'][:30]}... -> {conv_id[:8]}")
            
            self._log("在线采集写入", "PASS", f"成功写入 {len(saved_ids)} 条对话")
            
            # 验证查询
            recent = self.collector.get_recent_conversations(limit=5)
            assert len(recent) == 5, f"查询返回 {len(recent)} 条，期望 5 条"
            
            self._log("在线采集查询", "PASS", f"成功查询最近 {len(recent)} 条对话")
            
            # 保存ID供后续测试使用
            self.conv_ids = saved_ids
            
        except Exception as e:
            self._log("在线采集", "FAIL", str(e))
            self.conv_ids = []
    
    def _test_offline_upload(self):
        """测试3: 离线上传 — 模拟JSON/JSONL/CSV上传"""
        print("\n" + "-" * 40)
        print("测试3: 离线上传测试")
        print("-" * 40)
        
        try:
            # 测试 JSONL 字符串上传
            jsonl_content = ""
            for i in range(5):
                record = {
                    'question': f'离线上传问题 {i+1}',
                    'answer': f'离线上传答案 {i+1}',
                    'contexts': [f'上下文 {j+1}' for j in range(2)],
                    'ground_truth': f'标准答案 {i+1}',
                    'domain': random.choice(['自指学', '数学', 'AI']),
                    'difficulty': random.choice(['easy', 'medium', 'hard'])
                }
                jsonl_content += json.dumps(record, ensure_ascii=False) + '\n'
            
            result = self.uploader.upload_jsonl(
                content=jsonl_content,
                source_type='manual_entry',
                metadata={'uploader': 'test_script'}
            )
            
            self._log("JSONL上传", "PASS", 
                      f"批次: {result['upload_batch']}, 成功: {result['saved']}/{result['total_records']}")
            
            # 测试字段映射（使用别名）
            alias_content = ""
            for i in range(3):
                record = {
                    'q': f'别名测试问题 {i+1}',  # 使用别名 'q' 而不是 'question'
                    'a': f'别名测试答案 {i+1}',
                    'ctx': ['上下文片段'],
                    'gt': f'标准答案 {i+1}'
                }
                alias_content += json.dumps(record, ensure_ascii=False) + '\n'
            
            result2 = self.uploader.upload_jsonl(
                content=alias_content,
                source_type='manual_entry'
            )
            
            self._log("字段映射测试", "PASS", 
                      f"成功映射并保存 {result2['saved']} 条记录")
            
            # 测试 CSV 文件上传
            csv_path = 'data/test_upload.csv'
            os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
            
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                import csv
                writer = csv.DictWriter(f, fieldnames=['question', 'answer', 'contexts', 'ground_truth'])
                writer.writeheader()
                for i in range(3):
                    writer.writerow({
                        'question': f'CSV问题 {i+1}',
                        'answer': f'CSV答案 {i+1}',
                        'contexts': json.dumps([f'CSV上下文 {j+1}' for j in range(2)]),
                        'ground_truth': f'CSV标准答案 {i+1}'
                    })
            
            result3 = self.uploader.upload_file(csv_path, source_type='offline_upload')
            
            self._log("CSV上传", "PASS", 
                      f"成功解析并保存 {result3['saved']}/{result3['total_records']} 条")
            
            # 清理测试文件
            if os.path.exists(csv_path):
                os.remove(csv_path)
            
        except Exception as e:
            self._log("离线上传", "FAIL", str(e))
    
    def _test_user_feedback(self):
        """测试4: 用户反馈"""
        print("\n" + "-" * 40)
        print("测试4: 用户反馈测试")
        print("-" * 40)
        
        try:
            if not hasattr(self, 'conv_ids') or not self.conv_ids:
                print("   跳过：没有可用的对话ID")
                return
            
            # 生成反馈
            feedbacks = self.generator.generate_feedbacks(self.conv_ids[:10])
            
            for fb in feedbacks:
                self.collector.save_feedback(
                    conversation_id=fb['conversation_id'],
                    feedback_type=fb['feedback_type'],
                    content=fb['content'],
                    rating=fb['rating']
                )
            
            self._log("用户反馈", "PASS", f"成功写入 {len(feedbacks)} 条反馈")
            
            # 验证反馈分布
            stats = self.collector.get_statistics()
            fb_dist = stats.get('feedback_distribution', {})
            
            self._log("反馈分布", "PASS", 
                      f"thumbs_up: {fb_dist.get('thumbs_up', 0)}, "
                      f"thumbs_down: {fb_dist.get('thumbs_down', 0)}, "
                      f"comment: {fb_dist.get('comment', 0)}")
            
        except Exception as e:
            self._log("用户反馈", "FAIL", str(e))
    
    def _test_retrieval_logs(self):
        """测试5: 检索日志"""
        print("n" + "-" * 40)
        print("测试5: 检索日志测试")
        print("-" * 40)
        
        try:
            if not hasattr(self, 'conv_ids') or not self.conv_ids:
                print("   跳过：没有可用的对话ID")
                return
            
            # 生成检索日志
            logs = self.generator.generate_retrieval_logs(self.conv_ids[:10])
            
            for log in logs:
                self.collector.save_retrieval_log(
                    conversation_id=log['conversation_id'],
                    query=log['query'],
                    retrieved_chunks=log['retrieved_chunks'],
                    scores=log['scores'],
                    latency_ms=log['latency_ms']
                )
            
            self._log("检索日志", "PASS", f"成功写入 {len(logs)} 条检索日志")
            
        except Exception as e:
            self._log("检索日志", "FAIL", str(e))
    
    def _test_llm_calls(self):
        """测试6: LLM调用记录"""
        print("\n" + "-" * 40)
        print("测试6: LLM调用记录测试")
        print("-" * 40)
        
        try:
            if not hasattr(self, 'conv_ids') or not self.conv_ids:
                print("   跳过：没有可用的对话ID")
                return
            
            # 生成LLM调用
            calls = self.generator.generate_llm_calls(self.conv_ids[:10])
            
            for call in calls:
                self.collector.save_llm_call(
                    conversation_id=call['conversation_id'],
                    prompt=call['prompt'],
                    response=call['response'],
                    model_name=call['model_name'],
                    token_usage=call['token_usage'],
                    latency_ms=call['latency_ms']
                )
            
            self._log("LLM调用", "PASS", f"成功写入 {len(calls)} 条LLM调用记录")
            
        except Exception as e:
            self._log("LLM调用", "FAIL", str(e))
    
    def _test_data_query(self):
        """测试7: 数据查询"""
        print("\n" + "-" * 40)
        print("测试7: 数据查询测试")
        print("-" * 40)
        
        try:
            # 查询最近对话
            recent = self.collector.get_recent_conversations(limit=5)
            self._log("最近对话查询", "PASS", f"返回 {len(recent)} 条记录")
            
            # 按来源过滤
            online = self.collector.get_recent_conversations(
                limit=10, source='online_api'
            )
            self._log("按来源过滤", "PASS", f"online_api: {len(online)} 条")
            
            # 查询单条详情
            if hasattr(self, 'conv_ids') and self.conv_ids:
                details = self.collector.get_conversation_with_details(self.conv_ids[0])
                has_retrieval = len(details.get('retrieval_logs', [])) > 0
                has_llm = len(details.get('llm_calls', [])) > 0
                has_feedback = len(details.get('feedbacks', [])) > 0
                
                self._log("对话详情查询", "PASS", 
                          f"检索日志: {len(details.get('retrieval_logs', []))}, "
                          f"LLM调用: {len(details.get('llm_calls', []))}, "
                          f"反馈: {len(details.get('feedbacks', []))}")
            
            # 查询统计
            stats = self.collector.get_statistics()
            self._log("统计查询", "PASS", 
                      f"conversations: {stats.get('conversations', 0)}, "
                      f"raw_data: {stats.get('raw_data', 0)}, "
                      f"user_feedback: {stats.get('user_feedback', 0)}")
            
        except Exception as e:
            self._log("数据查询", "FAIL", str(e))
    
    def _test_data_export(self):
        """测试8: 数据导出"""
        print("\n" + "-" * 40)
        print("测试8: 数据导出测试")
        print("-" * 40)
        
        try:
            # 导出 conversations 到 JSONL
            export_path = 'data/export_conversations.jsonl'
            count = self.collector.export_to_jsonl(
                output_path=export_path,
                table='conversations',
                limit=100
            )
            
            assert os.path.exists(export_path), "导出文件未创建"
            assert count > 0, "导出记录数为0"
            
            self._log("JSONL导出", "PASS", f"导出 {count} 条记录到 {export_path}")
            
            # 验证导出文件内容
            with open(export_path, 'r', encoding='utf-8') as f:
                first_line = json.loads(f.readline())
                assert 'question' in first_line, "导出记录缺少 question 字段"
                assert 'metadata' in first_line, "导出记录缺少 metadata 字段"
            
            self._log("导出文件验证", "PASS", "JSON格式正确，字段完整")
            
        except Exception as e:
            self._log("数据导出", "FAIL", str(e))
    
    def _print_statistics(self):
        """打印数据库统计信息"""
        print("\n" + "-" * 40)
        print("数据库统计信息")
        print("-" * 40)
        
        stats = self.collector.get_statistics()
        
        print(f"\n📊 表记录数:")
        for table, count in stats.items():
            if isinstance(count, int):
                print(f"   {table:20s}: {count:4d} 条")
        
        print(f"\n📈 数据来源分布:")
        for source, count in stats.get('source_distribution', {}).items():
            print(f"   {source:20s}: {count:4d} 条")
        
        print(f"\n👍 用户反馈分布:")
        for fb_type, count in stats.get('feedback_distribution', {}).items():
            print(f"   {fb_type:20s}: {count:4d} 条")
        
        print(f"\n📦 原始数据状态:")
        for status, count in stats.get('raw_status', {}).items():
            print(f"   {status:20s}: {count:4d} 条")
    
    def _print_summary(self):
        """打印测试总结"""
        print("\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)
        
        passed = sum(1 for r in self.test_results if r['status'] == 'PASS')
        failed = sum(1 for r in self.test_results if r['status'] == 'FAIL')
        total = len(self.test_results)
        
        print(f"\n总计: {total} 项测试")
        print(f"通过: {passed} ✅")
        print(f"失败: {failed} ❌")
        
        if failed > 0:
            print(f"\n失败项:")
            for r in self.test_results:
                if r['status'] == 'FAIL':
                    print(f"   ❌ {r['step']}: {r['details']}")
        
        print(f"\n数据库文件: {os.path.abspath(self.db_path)}")
        print(f"数据库大小: {os.path.getsize(self.db_path) / 1024:.1f} KB")


# ==================== 主入口 ====================

if __name__ == '__main__':
    # 使用测试数据库路径，避免污染生产数据
    test_db_path = "data/rag_data_demo.db"
    
    # 如果存在旧的测试数据库，先删除
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
        print(f"已清理旧测试数据库: {test_db_path}")
    
    # 运行测试
    tester = DataCollectionTester(test_db_path)
    tester.run_all_tests()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
