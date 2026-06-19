"""
RAG 教学项目 — 测试集搭建测试脚本

生成虚拟数据覆盖不同种类，测试 3.1-3.4 完整流程:
1. 数据格式统一 (3.1)
2. 离线数据导入 (3.2)
3. 数据解析 (3.3)
4. 数据集搭建 (3.4)

使用已有的 test_rag_data.db（测试数据库）避免污染生产数据。
"""

import sqlite3
import json
import uuid
import random
import os
import sys
from datetime import datetime

# 将 src 加入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from testset.testset_builder import (
    DataFormatValidator, DataImporter, DataParser, TestSetBuilder
)

# ==================== 虚拟数据定义 ====================

DOMAINS = ['自指学', '数学', 'AI', '哲学', '逻辑学']
QUESTION_TYPES = ['factual', 'comparative', 'procedural', 'open']
DIFFICULTIES = ['easy', 'medium', 'hard']
SOURCES = ['online_api', 'offline_upload', 'batch_export']

# 不同领域的问题模板
QUESTION_TEMPLATES = {
    '自指学': [
        {
            'question': '什么是自指学？',
            'question_type': 'factual',
            'difficulty': 'easy',
            'contexts': ['自指学（Self-reference）是研究系统如何引用自身的学科。'],
            'answer': '自指学是研究系统如何引用自身的学科。',
            'ground_truth': '自指学（Self-reference）是研究系统如何引用自身的学科，涉及逻辑学、语言学、计算机科学和数学等多个领域。'
        },
        {
            'question': '比较自指学与递归的区别',
            'question_type': 'comparative',
            'difficulty': 'medium',
            'contexts': [
                '自指学关注系统对自身的引用，递归是一种编程技术，函数调用自身。',
                '递归是自指概念在编程中的具体实现，但自指学涵盖更广泛的内容。'
            ],
            'answer': '自指学是研究系统引用自身的学科，递归是函数调用自身的技术。递归是自指概念的一种实现。',
            'ground_truth': '自指学关注系统引用自身的能力，递归是函数调用自身的技术。递归是自指在编程领域的具体实现，但自指学范围更广，还包括语言、逻辑等领域。'
        },
        {
            'question': '如何构建一个自指系统？',
            'question_type': 'procedural',
            'difficulty': 'hard',
            'contexts': ['构建自指系统需要满足以下条件：1. 系统能够引用自身 2. 引用不会导致无限循环 3. 系统能够处理引用的语义。'],
            'answer': '构建自指系统需要满足三个条件：能够引用自身、避免无限循环、处理语义。',
            'ground_truth': '构建自指系统需要：1. 系统能够引用自身结构或输出；2. 建立终止条件避免无限循环；3. 设计语义处理机制处理自指内容。'
        },
        {
            'question': '自指学在现代人工智能中的应用前景如何？',
            'question_type': 'open',
            'difficulty': 'hard',
            'contexts': ['自指学在AI中可用于元学习、自我改进系统、元认知模型等。'],
            'answer': '自指学在AI中有广泛应用，包括元学习和自我改进。',
            'ground_truth': '自指学在AI中的应用前景包括：元学习（系统学习如何学习）、自我改进系统（系统优化自身算法）、元认知模型（系统对自身推理过程的认知）。这些领域是当前AI研究的前沿。'
        },
    ],
    '数学': [
        {
            'question': '什么是对角线论证？',
            'question_type': 'factual',
            'difficulty': 'medium',
            'contexts': ['对角线论证由康托尔提出，用于证明实数集不可数。'],
            'answer': '对角线论证是康托尔提出的证明方法。',
            'ground_truth': '对角线论证（Diagonal Argument）由乔治·康托尔于1891年提出，用于证明实数集是不可数集。其核心思想是构造一个与列表中所有元素都不同的新元素。'
        },
        {
            'question': 'Cantor定理和Russell悖论有什么关系？',
            'question_type': 'comparative',
            'difficulty': 'hard',
            'contexts': [
                'Cantor定理通过证明不存在从集合到其幂集的满射来建立集合的层次结构。',
                'Russell悖论则通过构造不属于自己的集合的集合来展示朴素集合论中的矛盾。'
            ],
            'answer': '两者都涉及集合论中的自指问题。',
            'ground_truth': 'Cantor定理和Russell悖论都涉及自指概念。Cantor定理通过限制自指建立了集合论的层次结构，而Russell悖论展示了无限制自指会导致矛盾。两者共同推动了公理化集合论的发展。'
        },
        {
            'question': '如何证明一个集合是不可数的？',
            'question_type': 'procedural',
            'difficulty': 'medium',
            'contexts': ['证明集合不可数通常使用对角线论证或反证法。'],
            'answer': '使用对角线论证或反证法证明。',
            'ground_truth': '证明集合不可数的方法：1. 假设集合可数；2. 枚举所有元素；3. 构造一个不在枚举中的新元素；4. 得出矛盾。这是康托尔对角线论证的核心步骤。'
        },
        {
            'question': '笛卡尔闭范畴在计算机科学中的意义是什么？',
            'question_type': 'open',
            'difficulty': 'hard',
            'contexts': ['笛卡尔闭范畴（CCC）是带有指数对象的范畴，对应于函数类型。'],
            'answer': '笛卡尔闭范畴在CS中对应函数类型。',
            'ground_truth': '笛卡尔闭范畴（CCC）在计算机科学中是λ演算的语义模型。范畴论提供了一种统一的数学框架来理解类型系统、函数编程和语义。'
        },
    ],
    'AI': [
        {
            'question': '什么是RAG？',
            'question_type': 'factual',
            'difficulty': 'easy',
            'contexts': ['RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术。'],
            'answer': 'RAG是检索增强生成技术。',
            'ground_truth': 'RAG（Retrieval-Augmented Generation）是一种结合信息检索（Retrieval）和文本生成（Generation）的技术。它通过检索相关文档来增强大语言模型的回答质量。'
        },
        {
            'question': 'RAG和微调（Fine-tuning）有什么区别？',
            'question_type': 'comparative',
            'difficulty': 'medium',
            'contexts': [
                'RAG通过检索外部知识来增强回答，不需要修改模型参数。',
                '微调通过调整模型参数来适应特定任务，需要训练数据。'
            ],
            'answer': 'RAG通过检索增强回答，微调通过修改模型参数适应任务。',
            'ground_truth': 'RAG和微调的主要区别：RAG不改变模型参数，通过检索外部知识来增强回答；微调修改模型参数使其适应特定任务。RAG更适合知识更新频繁的场景，微调更适合风格/格式调整。'
        },
        {
            'question': '如何构建一个RAG系统？',
            'question_type': 'procedural',
            'difficulty': 'medium',
            'contexts': ['构建RAG系统需要：1. 文档索引 2. 检索模块 3. 生成模块。'],
            'answer': '构建RAG系统需要文档索引、检索和生成模块。',
            'ground_truth': '构建RAG系统的步骤：1. 文档预处理与分块；2. 向量嵌入与索引；3. 检索模块（语义检索+重排序）；4. 提示词工程（上下文整合）；5. 生成模块（大语言模型）。'
        },
        {
            'question': '大语言模型在处理自指问题时会遇到什么困难？',
            'question_type': 'open',
            'difficulty': 'hard',
            'contexts': ['大语言模型在处理自指时可能陷入循环或产生矛盾输出。'],
            'answer': 'LLM处理自指时可能陷入循环或产生矛盾。',
            'ground_truth': '大语言模型在处理自指问题时的困难：1. 可能陷入无限循环；2. 难以区分元层次和对象层次；3. 缺乏真正的自我模型；4. 容易产生矛盾输出。这些限制源于Transformer架构的因果性质。'
        },
    ],
    '哲学': [
        {
            'question': '什么是哥德尔不完备定理？',
            'question_type': 'factual',
            'difficulty': 'medium',
            'contexts': ['哥德尔不完备定理证明任何一致的形式系统都包含不可证明的真命题。'],
            'answer': '哥德尔定理证明一致形式系统有不可证明的真命题。',
            'ground_truth': '哥德尔不完备定理（1931）：在任何一致的形式系统F中，存在一个命题G，G在F中既不可证明也不可否证。这个定理揭示了形式系统的根本局限性。'
        },
        {
            'question': '哥德尔定理和图灵机停机问题有什么关系？',
            'question_type': 'comparative',
            'difficulty': 'hard',
            'contexts': [
                '哥德尔定理通过自指构造证明了形式系统的局限性。',
                '图灵机停机问题通过归约证明了计算的不可判定性。'
            ],
            'answer': '两者都揭示了形式系统的局限性。',
            'ground_truth': '哥德尔不完备定理和图灵机停机问题都是不可判定性结果。哥德尔定理通过算术化自指构造展示了形式系统的局限性，停机问题通过计算归约展示了计算模型的局限性。两者本质上是同一数学事实的不同表现。'
        },
    ],
    '逻辑学': [
        {
            'question': '什么是排中律？',
            'question_type': 'factual',
            'difficulty': 'easy',
            'contexts': ['排中律（Law of Excluded Middle）指命题要么真要么假。'],
            'answer': '排中律指命题要么真要么假。',
            'ground_truth': '排中律（Law of Excluded Middle）是经典逻辑的基本定律之一，指对于任何命题P，P要么为真要么为假，不存在中间状态。公式表示为：P ∨ ¬P。'
        },
        {
            'question': '如何构造一个有效但非直觉主义的证明？',
            'question_type': 'procedural',
            'difficulty': 'hard',
            'contexts': ['直觉主义逻辑拒绝排中律，要求构造性证明。'],
            'answer': '使用反证法或排中律构造证明。',
            'ground_truth': '构造有效但非直觉主义的证明：使用反证法（假设¬P推出矛盾，从而得出P）。直觉主义不接受这种证明，因为它没有直接构造出P的 witness。'
        },
    ],
}


def generate_variations(base_template: dict, domain: str, count: int = 3) -> list:
    """基于模板生成变体数据"""
    variations = []
    
    for i in range(count):
        var = base_template.copy()
        var['id'] = str(uuid.uuid4())
        var['domain'] = domain
        
        # 随机变化难度
        if i == 0:
            var['difficulty'] = base_template['difficulty']
        else:
            var['difficulty'] = random.choice(DIFFICULTIES)
        
        # 添加微小变化到问题
        if i > 0:
            suffixes = ['请详细说明。', '简要回答。', '给出具体例子。']
            var['question'] = base_template['question'] + random.choice(suffixes)
        
        var['source'] = random.choice(SOURCES)
        var['timestamp'] = datetime.now().isoformat()
        var['model_version'] = 'v1.0.0'
        var['metadata'] = {
            'upload_time': datetime.now().isoformat(),
            'uploader': 'test_generator',
            'original_format': 'json',
            'data_quality': 'annotated' if random.random() > 0.5 else 'raw',
            'domain': domain,
            'difficulty': var['difficulty'],
            'question_type': var['question_type']
        }
        
        variations.append(var)
    
    return variations


def generate_test_data(db_path: str = "data/test_rag_data.db", total_count: int = 50):
    """生成测试数据到测试数据库"""
    
    print(f"=== 生成虚拟测试数据到 {db_path} ===")
    
    # 确保数据库存在
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    with sqlite3.connect(db_path) as conn:
        # 确保表存在
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT,
                contexts TEXT,
                ground_truth TEXT,
                model_version TEXT DEFAULT 'v1.0.0',
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                source TEXT DEFAULT 'online_api',
                metadata TEXT,
                processing_stage TEXT DEFAULT 'raw'
            );
            
            CREATE TABLE IF NOT EXISTS raw_data (
                id TEXT PRIMARY KEY,
                source_type TEXT,
                original_format TEXT,
                raw_content TEXT,
                upload_batch TEXT,
                metadata TEXT,
                status TEXT DEFAULT 'pending',
                error_message TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS processed_data (
                id TEXT PRIMARY KEY,
                raw_id TEXT,
                question TEXT NOT NULL,
                question_type TEXT DEFAULT 'factual',
                domain TEXT DEFAULT '其他',
                difficulty TEXT DEFAULT 'medium',
                contexts TEXT,
                answer TEXT,
                ground_truth TEXT,
                metadata TEXT,
                evaluation TEXT,
                processing_stage TEXT DEFAULT 'parsed',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS testset_versions (
                version_id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                created_at TEXT,
                size INTEGER,
                domain_distribution TEXT,
                difficulty_distribution TEXT,
                type_distribution TEXT,
                avg_quality_score REAL,
                has_ground_truth_ratio REAL,
                status TEXT,
                file_paths TEXT
            );
        """)
        
        # 清空旧数据（测试环境）
        conn.execute("DELETE FROM conversations")
        conn.execute("DELETE FROM raw_data")
        conn.execute("DELETE FROM processed_data")
        conn.execute("DELETE FROM testset_versions")
        conn.commit()
        
        print("已清空旧测试数据")
        
        # 生成 conversations 数据
        generated = 0
        for domain, templates in QUESTION_TEMPLATES.items():
            for template in templates:
                # 每个模板生成 2-4 个变体
                count = random.randint(2, 4)
                variations = generate_variations(template, domain, count)
                
                for var in variations:
                    conn.execute("""
                        INSERT INTO conversations 
                        (id, question, answer, contexts, ground_truth, 
                         model_version, timestamp, source, metadata, processing_stage)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        var['id'],
                        var['question'],
                        var['answer'],
                        json.dumps(var['contexts'], ensure_ascii=False),
                        var.get('ground_truth'),
                        var['model_version'],
                        var['timestamp'],
                        var['source'],
                        json.dumps(var['metadata'], ensure_ascii=False),
                        'raw'
                    ))
                    generated += 1
        
        # 生成一些 raw_data 数据（用于测试从 raw_data 导入）
        raw_count = 0
        for domain, templates in QUESTION_TEMPLATES.items():
            for template in templates[:2]:  # 每个领域取前2个模板
                raw_id = str(uuid.uuid4())
                content = {
                    "question": template['question'],
                    "answer": template['answer'],
                    "contexts": template['contexts'],
                    "ground_truth": template.get('ground_truth', ''),
                    "domain": domain,
                    "difficulty": template['difficulty']
                }
                conn.execute("""
                    INSERT INTO raw_data (id, source_type, original_format, raw_content, 
                                          upload_batch, metadata, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    raw_id,
                    'batch_export',
                    'json',
                    json.dumps(content, ensure_ascii=False),
                    f"batch_{datetime.now().strftime('%Y%m%d')}_001",
                    json.dumps({"domain": domain, "imported": False}),
                    'pending'
                ))
                raw_count += 1
        
        conn.commit()
        
        print(f"✅ 生成 {generated} 条 conversations 记录")
        print(f"✅ 生成 {raw_count} 条 raw_data 记录")
        
        # 验证数据分布
        print("\n=== 数据分布 ===")
        
        # 按领域统计
        cursor = conn.execute("""
            SELECT SUBSTR(question, 1, 20) as prefix, COUNT(*) 
            FROM conversations 
            GROUP BY prefix
            ORDER BY COUNT(*) DESC
        """)
        
        # 按来源统计
        cursor = conn.execute("SELECT source, COUNT(*) FROM conversations GROUP BY source")
        print("来源分布:")
        for row in cursor.fetchall():
            print(f"  {row[0]}: {row[1]}")
        
        # 按 domain 统计（通过 metadata）
        cursor = conn.execute("SELECT metadata FROM conversations")
        domain_counts = {}
        for row in cursor.fetchall():
            meta = json.loads(row[0] or '{}')
            domain = meta.get('domain', '其他')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        print("领域分布:")
        for domain, count in domain_counts.items():
            print(f"  {domain}: {count}")
        
        return generated, raw_count


def test_pipeline(db_path: str = "data/test_rag_data.db"):
    """测试完整流程 3.1 -> 3.2 -> 3.3 -> 3.4"""
    
    print("\n" + "=" * 60)
    print("🚀 测试集搭建流程测试 (3.1-3.4)")
    print("=" * 60)
    
    # 1. 生成虚拟数据
    print("\n【步骤 0】生成虚拟数据...")
    generate_test_data(db_path)
    
    # 2. 3.2 数据导入
    print("\n【步骤 1】3.2 离线数据导入 (conversations -> processed_data)")
    importer = DataImporter(db_path)
    
    # 从 conversations 导入
    result1 = importer.import_from_conversations(source=None, batch_size=100)
    print(f"  conversations 导入: {result1}")
    
    # 从 raw_data 导入
    result2 = importer.import_from_raw_data(status="pending", batch_size=100)
    print(f"  raw_data 导入: {result2}")
    
    # 3. 3.3 数据解析
    print("\n【步骤 2】3.3 数据解析 (parsed -> validated)")
    parser = DataParser()
    result3 = parser.parse_all(db_path, stage="parsed")
    print(f"  解析结果: {result3}")
    
    # 4. 3.4 数据集搭建
    print("\n【步骤 3】3.4 测试集搭建 (validated -> testset)")
    builder = TestSetBuilder(db_path)
    result4 = builder.build_testset(
        output_prefix="data/testset",
        golden_size=10,
        validation_size=20,
        stress_size=5
    )
    print(f"  搭建结果: {json.dumps(result4, ensure_ascii=False, indent=2)}")
    
    # 5. 验证输出
    print("\n【步骤 4】验证输出文件...")
    for name in ['golden', 'validation', 'stress']:
        path = result4.get(name)
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                lines = f.readlines()
            print(f"  {name}: {len(lines)} 条记录 -> {path}")
            if lines:
                first = json.loads(lines[0])
                print(f"    示例: {first['question'][:50]}...")
    
    # 6. 验证数据库状态
    print("\n【步骤 5】验证数据库状态...")
    with sqlite3.connect(db_path) as conn:
        for table in ['conversations', 'raw_data', 'processed_data', 'testset_versions']:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table}: {count} 条记录")
    
    print("\n" + "=" * 60)
    print("✅ 测试集搭建流程测试完成！")
    print("=" * 60)
    
    return {
        "import": result1,
        "raw_import": result2,
        "parse": result3,
        "build": result4
    }


def test_format_validation():
    """测试 3.1 数据格式统一"""
    print("\n=== 3.1 数据格式统一测试 ===")
    
    validator = DataFormatValidator()
    
    # 测试有效数据
    valid_records = [
        {
            "question": "什么是对角线论证？",
            "answer": "对角线论证是...",
            "contexts": ["对角线论证由康托尔提出..."],
            "domain": "数学",
            "difficulty": "medium"
        },
        {
            "question": "RAG和微调有什么区别？",
            "answer": "RAG通过检索...",
            "contexts": [
                {"content": "RAG检索外部知识", "source": "doc1", "page": 1},
                {"content": "微调修改模型参数", "source": "doc2", "page": 2}
            ],
            "domain": "AI",
            "difficulty": "hard"
        }
    ]
    
    for i, record in enumerate(valid_records):
        normalized = validator.normalize(record)
        is_valid, errors = validator.validate(normalized)
        print(f"  记录 {i+1}: {'✅ 通过' if is_valid else '❌ 失败'}")
        if errors:
            print(f"    错误: {errors}")
    
    # 测试无效数据
    invalid_records = [
        {"question": "短", "answer": "", "domain": "数学"},  # 问题过短
        {"question": "问题", "answer": "", "difficulty": "invalid"},  # 无效难度
    ]
    
    for i, record in enumerate(invalid_records):
        normalized = validator.normalize(record)
        is_valid, errors = validator.validate(normalized)
        print(f"  无效记录 {i+1}: {'✅ 通过' if is_valid else '❌ 失败'} (预期失败)")
        if errors:
            print(f"    错误: {errors}")


if __name__ == "__main__":
    # 使用测试数据库避免污染生产数据
    test_db = "data/test_rag_data.db"
    
    # 测试数据格式统一
    test_format_validation()
    
    # 运行完整流程测试
    results = test_pipeline(test_db)
    
    print("\n📊 最终测试统计:")
    print(json.dumps({
        "conversations_imported": results["import"]["imported"],
        "raw_data_imported": results["raw_import"]["imported"],
        "parsed_valid": results["parse"]["parsed"],
        "parsed_invalid": results["parse"]["invalid"],
        "testset_version": results["build"].get("version_id", "N/A"),
        "testset_stats": results["build"].get("stats", {})
    }, ensure_ascii=False, indent=2))
