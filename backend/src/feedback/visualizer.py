"""
RAG 教学项目 — 可视化报告生成器

4.2 可视化
- 生成 HTML 评估报告
- 指标看板、趋势分析、错误分析

参考: /home/ubuntu/.openclaw/workspace/RAG教学/docs/监控指标体系搭建规划.md
"""

import sqlite3
import json
import os
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import uuid


class EvaluationVisualizer:
    """评估可视化 — 生成 HTML 报告"""
    
    def __init__(self, db_path: str = "data/rag_data.db"):
        self.db_path = db_path
    
    def generate_report(self,
                        testset_version: str,
                        output_path: str = "reports/evaluation_report.html") -> str:
        """
        生成完整的 HTML 评估报告
        
        Returns:
            报告文件路径
        """
        # 获取数据
        summary = self._get_summary(testset_version)
        distributions = self._get_distributions(testset_version)
        failures = self._get_failures(testset_version, limit=20)
        records = self._get_all_records(testset_version)
        
        # 生成 HTML
        html = self._build_html(
            testset_version=testset_version,
            summary=summary,
            distributions=distributions,
            failures=failures,
            records=records
        )
        
        # 写入文件
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return output_path
    
    def generate_mini_report(self,
                            testset_version: str,
                            output_path: str = "reports/evaluation_mini.html") -> str:
        """生成简化版报告（仅核心指标）"""
        summary = self._get_summary(testset_version)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>RAGAS 评估摘要 - {testset_version}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                margin: 0; padding: 40px; background: #f5f5f5; }}
        .card {{ background: white; border-radius: 12px; padding: 24px; 
                 margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }}
        .metric {{ text-align: center; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
        .metric-value {{ font-size: 36px; font-weight: bold; margin: 8px 0; }}
        .metric-label {{ font-size: 14px; color: #666; }}
        .pass {{ color: #28a745; }}
        .warn {{ color: #ffc107; }}
        .fail {{ color: #dc3545; }}
        .header {{ font-size: 24px; font-weight: bold; margin-bottom: 8px; }}
        .subtitle {{ color: #666; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="card">
        <div class="header">RAGAS 评估摘要</div>
        <div class="subtitle">版本: {testset_version} | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
    </div>
    
    <div class="card">
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">RAGAS Score</div>
                <div class="metric-value {'pass' if summary.get('avg_ragas_score', 0) > 0.75 else 'warn' if summary.get('avg_ragas_score', 0) > 0.6 else 'fail'}">
                    {summary.get('avg_ragas_score', 0):.3f}
                </div>
                <div class="metric-label">{'✅ 通过' if summary.get('avg_ragas_score', 0) > 0.75 else '⚠️ 警告' if summary.get('avg_ragas_score', 0) > 0.6 else '❌ 未通过'}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Faithfulness</div>
                <div class="metric-value {'pass' if summary.get('avg_faithfulness', 0) > 0.8 else 'warn' if summary.get('avg_faithfulness', 0) > 0.6 else 'fail'}">
                    {summary.get('avg_faithfulness', 0):.3f}
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">Relevance</div>
                <div class="metric-value {'pass' if summary.get('avg_relevance', 0) > 0.8 else 'warn' if summary.get('avg_relevance', 0) > 0.6 else 'fail'}">
                    {summary.get('avg_relevance', 0):.3f}
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">Precision</div>
                <div class="metric-value {'pass' if summary.get('avg_precision', 0) > 0.7 else 'warn' if summary.get('avg_precision', 0) > 0.5 else 'fail'}">
                    {summary.get('avg_precision', 0):.3f}
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">通过率</div>
                <div class="metric-value">{summary.get('pass_rate', 0):.1%}</div>
                <div class="metric-label">{summary.get('passed', 0)}/{summary.get('total', 0)}</div>
            </div>
        </div>
    </div>
</body>
</html>"""
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return output_path
    
    def _build_html(self, testset_version: str, summary: dict,
                   distributions: dict, failures: List[dict],
                   records: List[dict]) -> str:
        """构建完整 HTML 报告"""
        
        # 生成图表数据
        chart_data = self._prepare_chart_data(records)
        
        # 生成失败样本表格
        failures_table = self._build_failures_table(failures)
        
        # 生成分布图表
        distribution_charts = self._build_distribution_charts(distributions)
        
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAGAS 评估报告 - {testset_version}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        :root {{
            --primary: #2563eb;
            --success: #16a34a;
            --warning: #ca8a04;
            --danger: #dc2626;
            --bg: #f8fafc;
            --card: #ffffff;
            --text: #1e293b;
            --text-secondary: #64748b;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: var(--bg); color: var(--text); line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
        
        .header {{ background: var(--card); border-radius: 12px; padding: 32px; margin-bottom: 24px;
                   box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .header h1 {{ font-size: 28px; margin-bottom: 8px; }}
        .header .meta {{ color: var(--text-secondary); font-size: 14px; }}
        
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .metric-card {{ background: var(--card); border-radius: 12px; padding: 24px;
                       box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }}
        .metric-value {{ font-size: 42px; font-weight: 700; margin: 8px 0; }}
        .metric-label {{ font-size: 14px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; }}
        .metric-status {{ font-size: 12px; margin-top: 8px; padding: 4px 12px; border-radius: 20px; display: inline-block; }}
        .status-pass {{ background: #dcfce7; color: #166534; }}
        .status-warn {{ background: #fef9c3; color: #854d0e; }}
        .status-fail {{ background: #fee2e2; color: #991b1b; }}
        
        .section {{ background: var(--card); border-radius: 12px; padding: 24px; margin-bottom: 24px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .section h2 {{ font-size: 20px; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 2px solid var(--bg); }}
        
        .chart-container {{ height: 300px; margin: 16px 0; }}
        .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 24px; }}
        
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #f1f5f9; font-weight: 600; color: var(--text-secondary); font-size: 12px; text-transform: uppercase; }}
        tr:hover {{ background: #f8fafc; }}
        .score {{ font-weight: 600; }}
        .score-high {{ color: var(--success); }}
        .score-mid {{ color: var(--warning); }}
        .score-low {{ color: var(--danger); }}
        
        .question-preview {{ max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        
        .footer {{ text-align: center; color: var(--text-secondary); font-size: 12px; margin-top: 40px; padding: 24px; }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>📊 RAGAS 评估报告</h1>
            <div class="meta">
                测试集版本: <strong>{testset_version}</strong> | 
                生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
                总样本: {summary.get('total', 0)} 条
            </div>
        </div>
        
        <!-- Core Metrics -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">RAGAS Score</div>
                <div class="metric-value" style="color: {'var(--success)' if summary.get('avg_ragas_score', 0) > 0.75 else 'var(--warning)' if summary.get('avg_ragas_score', 0) > 0.6 else 'var(--danger)'}">
                    {summary.get('avg_ragas_score', 0):.3f}
                </div>
                <span class="metric-status {'status-pass' if summary.get('avg_ragas_score', 0) > 0.75 else 'status-warn' if summary.get('avg_ragas_score', 0) > 0.6 else 'status-fail'}">
                    {'✅ 通过' if summary.get('avg_ragas_score', 0) > 0.75 else '⚠️ 警告' if summary.get('avg_ragas_score', 0) > 0.6 else '❌ 未通过'}
                </span>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Faithfulness</div>
                <div class="metric-value" style="color: {'var(--success)' if summary.get('avg_faithfulness', 0) > 0.8 else 'var(--warning)' if summary.get('avg_faithfulness', 0) > 0.6 else 'var(--danger)'}">
                    {summary.get('avg_faithfulness', 0):.3f}
                </div>
                <span class="metric-status {'status-pass' if summary.get('avg_faithfulness', 0) > 0.8 else 'status-warn' if summary.get('avg_faithfulness', 0) > 0.6 else 'status-fail'}">
                    {'✅' if summary.get('avg_faithfulness', 0) > 0.8 else '⚠️' if summary.get('avg_faithfulness', 0) > 0.6 else '❌'} > 0.8
                </span>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Answer Relevance</div>
                <div class="metric-value" style="color: {'var(--success)' if summary.get('avg_relevance', 0) > 0.8 else 'var(--warning)' if summary.get('avg_relevance', 0) > 0.6 else 'var(--danger)'}">
                    {summary.get('avg_relevance', 0):.3f}
                </div>
                <span class="metric-status {'status-pass' if summary.get('avg_relevance', 0) > 0.8 else 'status-warn' if summary.get('avg_relevance', 0) > 0.6 else 'status-fail'}">
                    {'✅' if summary.get('avg_relevance', 0) > 0.8 else '⚠️' if summary.get('avg_relevance', 0) > 0.6 else '❌'} > 0.8
                </span>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Context Precision</div>
                <div class="metric-value" style="color: {'var(--success)' if summary.get('avg_precision', 0) > 0.7 else 'var(--warning)' if summary.get('avg_precision', 0) > 0.5 else 'var(--danger)'}">
                    {summary.get('avg_precision', 0):.3f}
                </div>
                <span class="metric-status {'status-pass' if summary.get('avg_precision', 0) > 0.7 else 'status-warn' if summary.get('avg_precision', 0) > 0.5 else 'status-fail'}">
                    {'✅' if summary.get('avg_precision', 0) > 0.7 else '⚠️' if summary.get('avg_precision', 0) > 0.5 else '❌'} > 0.7
                </span>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">通过率</div>
                <div class="metric-value" style="color: var(--primary)">
                    {summary.get('pass_rate', 0):.1%}
                </div>
                <div class="metric-label">
                    {summary.get('passed', 0)} 通过 / {summary.get('failed', 0)} 失败
                </div>
            </div>
        </div>
        
        <!-- Distribution Charts -->
        <div class="section">
            <h2>📈 指标分布</h2>
            <div class="chart-grid">
                <div>
                    <div class="chart-container">
                        <canvas id="scoreChart"></canvas>
                    </div>
                </div>
                <div>
                    <div class="chart-container">
                        <canvas id="metricChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Failure Analysis -->
        <div class="section">
            <h2>🔍 低分样本分析（Top {len(failures)}）</h2>
            {failures_table}
        </div>
        
        <!-- Distribution Analysis -->
        <div class="section">
            <h2>📊 分布分析</h2>
            {distribution_charts}
        </div>
        
        <!-- Footer -->
        <div class="footer">
            RAG 教学项目 | RAGAS 评估报告 | 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
    
    <script>
        // Score Distribution Chart
        const scoreCtx = document.getElementById('scoreChart').getContext('2d');
        new Chart(scoreCtx, {{
            type: 'bar',
            data: {{
                labels: {chart_data['score_labels']},
                datasets: [{{
                    label: 'RAGAS Score 分布',
                    data: {chart_data['score_data']},
                    backgroundColor: [
                        'rgba(220, 38, 38, 0.6)',
                        'rgba(234, 179, 8, 0.6)',
                        'rgba(22, 163, 74, 0.6)'
                    ],
                    borderColor: [
                        'rgb(220, 38, 38)',
                        'rgb(234, 179, 8)',
                        'rgb(22, 163, 74)'
                    ],
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    title: {{ display: true, text: 'RAGAS Score 分布' }}
                }},
                scales: {{
                    y: {{ beginAtZero: true, title: {{ display: true, text: '样本数' }} }}
                }}
            }}
        }});
        
        // Metric Comparison Chart
        const metricCtx = document.getElementById('metricChart').getContext('2d');
        new Chart(metricCtx, {{
            type: 'radar',
            data: {{
                labels: ['Faithfulness', 'Relevance', 'Precision', 'RAGAS Score'],
                datasets: [{{
                    label: '平均得分',
                    data: [
                        {summary.get('avg_faithfulness', 0)},
                        {summary.get('avg_relevance', 0)},
                        {summary.get('avg_precision', 0)},
                        {summary.get('avg_ragas_score', 0)}
                    ],
                    backgroundColor: 'rgba(37, 99, 235, 0.2)',
                    borderColor: 'rgb(37, 99, 235)',
                    pointBackgroundColor: 'rgb(37, 99, 235)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgb(37, 99, 235)'
                }}, {{
                    label: '目标阈值',
                    data: [0.8, 0.8, 0.7, 0.75],
                    backgroundColor: 'rgba(22, 163, 74, 0.1)',
                    borderColor: 'rgba(22, 163, 74, 0.5)',
                    borderDash: [5, 5],
                    pointRadius: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{ display: true, text: '指标雷达图' }}
                }},
                scales: {{
                    r: {{
                        beginAtZero: true,
                        max: 1,
                        ticks: {{ stepSize: 0.2 }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
        
        return html
    
    def _prepare_chart_data(self, records: List[dict]) -> dict:
        """准备图表数据"""
        # RAGAS Score 分布
        low = sum(1 for r in records if r.get('ragas_score', 0) < 0.6)
        mid = sum(1 for r in records if 0.6 <= r.get('ragas_score', 0) < 0.75)
        high = sum(1 for r in records if r.get('ragas_score', 0) >= 0.75)
        
        return {
            "score_labels": json.dumps(["低分 (<0.6)", "中等 (0.6-0.75)", "高分 (>0.75)"]),
            "score_data": json.dumps([low, mid, high])
        }
    
    def _build_failures_table(self, failures: List[dict]) -> str:
        """构建失败样本表格"""
        if not failures:
            return "<p style='color: #666;'>🎉 所有样本均通过评估！</p>"
        
        rows = []
        for f in failures:
            score_class = "score-low" if f.get('ragas_score', 0) < 0.6 else "score-mid"
            rows.append(f"""
                <tr>
                    <td><code>{f.get('record_id', '')[:8]}...</code></td>
                    <td class="question-preview">{f.get('question', 'N/A')[:80]}</td>
                    <td class="score {score_class}">{f.get('ragas_score', 0):.3f}</td>
                    <td>{f.get('faithfulness', 0):.3f}</td>
                    <td>{f.get('answer_relevance', 0):.3f}</td>
                    <td>{f.get('context_precision', 0):.3f}</td>
                </tr>
            """)
        
        return f"""
            <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>问题</th>
                        <th>RAGAS Score</th>
                        <th>Faithfulness</th>
                        <th>Relevance</th>
                        <th>Precision</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
            </div>
        """
    
    def _build_distribution_charts(self, distributions: dict) -> str:
        """构建分布图表 HTML"""
        if not distributions:
            return "<p>暂无分布数据</p>"
        
        charts = []
        for key, data in distributions.items():
            if not data:
                continue
            labels = list(data.keys())
            values = list(data.values())
            
            chart_id = f"dist_{key}_{uuid.uuid4().hex[:8]}"
            
            charts.append(f"""
                <div style="margin-bottom: 24px;">
                    <h3 style="font-size: 16px; margin-bottom: 12px; color: var(--text-secondary);">{key}</h3>
                    <div style="display: flex; gap: 16px; flex-wrap: wrap;">
                        {''.join(f'<div style="background: #f1f5f9; padding: 12px 20px; border-radius: 8px;"><strong>{k}</strong>: {v}</div>' for k, v in data.items())}
                    </div>
                </div>
            """)
        
        return '\n'.join(charts)
    
    def _get_summary(self, testset_version: str) -> dict:
        """获取评估汇总"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    AVG(faithfulness) as avg_faithfulness,
                    AVG(answer_relevance) as avg_relevance,
                    AVG(context_precision) as avg_precision,
                    AVG(ragas_score) as avg_ragas_score,
                    SUM(CASE WHEN passed THEN 1 ELSE 0 END) as passed,
                    SUM(CASE WHEN NOT passed THEN 1 ELSE 0 END) as failed
                FROM evaluation_results
                WHERE testset_version = ?
            """, (testset_version,)).fetchone()
            
            total = row[0] or 0
            passed = row[5] or 0
            
            return {
                "total": total,
                "avg_faithfulness": round(row[1] or 0, 3),
                "avg_relevance": round(row[2] or 0, 3),
                "avg_precision": round(row[3] or 0, 3),
                "avg_ragas_score": round(row[4] or 0, 3),
                "passed": passed,
                "failed": row[6] or 0,
                "pass_rate": round(passed / total, 3) if total > 0 else 0.0,
            }
    
    def _get_distributions(self, testset_version: str) -> dict:
        """获取评估分布"""
        with sqlite3.connect(self.db_path) as conn:
            # Score 分布
            cursor = conn.execute("""
                SELECT 
                    CASE 
                        WHEN ragas_score >= 0.75 THEN 'high'
                        WHEN ragas_score >= 0.6 THEN 'mid'
                        ELSE 'low'
                    END as score_range,
                    COUNT(*) as count
                FROM evaluation_results
                WHERE testset_version = ?
                GROUP BY score_range
            """, (testset_version,))
            score_dist = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                "Score Distribution": score_dist
            }
    
    def _get_failures(self, testset_version: str, limit: int = 20) -> List[dict]:
        """获取低分样本"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM evaluation_results
                WHERE testset_version = ? AND passed = FALSE
                ORDER BY ragas_score ASC
                LIMIT ?
            """, (testset_version, limit)).fetchall()
            
            return [dict(row) for row in rows]
    
    def _get_all_records(self, testset_version: str) -> List[dict]:
        """获取所有评估记录"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM evaluation_results
                WHERE testset_version = ?
                ORDER BY ragas_score DESC
            """, (testset_version,)).fetchall()
            
            return [dict(row) for row in rows]


# ==================== 主入口 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAGAS 可视化报告生成")
    parser.add_argument("--db", default="data/rag_data.db", help="数据库路径")
    parser.add_argument("--version", required=True, help="测试集版本")
    parser.add_argument("--output", default="reports/evaluation_report.html", help="输出路径")
    parser.add_argument("--mini", action="store_true", help="生成简化版报告")
    
    args = parser.parse_args()
    
    visualizer = EvaluationVisualizer(args.db)
    
    if args.mini:
        path = visualizer.generate_mini_report(args.version, args.output)
    else:
        path = visualizer.generate_report(args.version, args.output)
    
    print(f"✅ 报告已生成: {path}")
    print(f"   大小: {os.path.getsize(path):,} bytes")
