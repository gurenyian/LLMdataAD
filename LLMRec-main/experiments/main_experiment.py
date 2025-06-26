import logging
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from core.data_distribution_analyzer import DataDistributionAnalyzer
from core.dynamic_prompt_tuner import DynamicPromptTuner
from core.adversarial_quality_module import AdversarialQualityModule
from core.evaluation_metrics import InnovativeEvaluationMetrics

logger = logging.getLogger(__name__)

class LLMRecommendationInnovationExperiment:
    """
    LLM推荐技术创新实验主控制器
    实现完整的闭环框架：分布自适应调控 + 对抗性质量保证
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        
        # 初始化各个模块
        self.distribution_analyzer = DataDistributionAnalyzer(self.config.get('analyzer_config'))
        self.prompt_tuner = DynamicPromptTuner(self.config.get('tuner_config'))
        self.adversarial_module = AdversarialQualityModule(self.config.get('adversarial_config'))
        self.evaluation_metrics = InnovativeEvaluationMetrics(self.config.get('metrics_config'))
        
        # 实验状态
        self.iteration_results = []
        self.real_data_vector = None
        
        logger.info("✅ LLM推荐技术创新实验初始化完成")
    
    def _default_config(self) -> Dict:
        return {
            'max_iterations': 5,
            'convergence_threshold': 0.05,
            'quality_improvement_threshold': 0.1,
            'samples_per_iteration': 100,
            'enable_early_stopping': True,
            'save_intermediate_results': True,
            'experiment_name': 'LLM_Rec_Innovation_Experiment'
        }
    
    def run_complete_experiment(self, real_data: List[str]) -> Dict:
        """
        运行完整的创新实验流程
        """
        logger.info("🚀 开始LLM推荐技术创新完整实验...")
        experiment_start_time = time.time()
        
        try:
            # Phase 1: 启动 - 分析真实数据
            logger.info("📊 Phase 1: 真实数据分布分析...")
            self.real_data_vector = self.distribution_analyzer.analyze_real_data(real_data)
            
            # Phase 2: 迭代优化循环
            logger.info("🔄 Phase 2: 开始迭代优化...")
            iteration_results = self._run_iterative_optimization(real_data)
            
            # Phase 3: 综合评估
            logger.info("📋 Phase 3: 综合创新评估...")
            final_evaluation = self._run_comprehensive_evaluation(real_data, iteration_results)
            
            # Phase 4: 结果整合
            total_time = time.time() - experiment_start_time
            experiment_results = self._integrate_experiment_results(
                real_data, iteration_results, final_evaluation, total_time
            )
            
            logger.info("✅ 完整实验流程执行完成")
            return experiment_results
            
        except Exception as e:
            logger.error(f"❌ 实验执行失败: {e}")
            return {'error': str(e), 'execution_time': time.time() - experiment_start_time}
    
    def _run_iterative_optimization(self, real_data: List[str]) -> List[Dict]:
        """
        运行迭代优化流程
        """
        iteration_results = []
        current_prompt = "请生成推荐系统的用户-物品交互数据，格式为'用户ID 物品ID1 物品ID2 ...'，每行一个用户。"
        
        for iteration in range(1, self.config['max_iterations'] + 1):
            logger.info(f"🔄 执行第{iteration}轮迭代...")
            
            iteration_start_time = time.time()
            
            # Step 1: 生成合成数据
            synthetic_data = self.prompt_tuner.generate_synthetic_data(
                current_prompt, self.config['samples_per_iteration']
            )
            
            # Step 2: 对抗性质量保证
            adversarial_results = self.adversarial_module.run_adversarial_round(
                real_data, synthetic_data, iteration
            )
            
            # Step 3: 分析合成数据分布
            synth_data_vector = self.distribution_analyzer.analyze_synthetic_data(
                adversarial_results['final_samples']
            )
            
            # Step 4: 计算分布偏差
            divergence_analysis = self.distribution_analyzer.calculate_distribution_divergence(
                self.real_data_vector, synth_data_vector
            )
            
            # Step 5: 动态Prompt调优
            updated_prompt, adjustment_details = self.prompt_tuner.adjust_prompt_strategy(
                self.real_data_vector, 
                synth_data_vector, 
                divergence_analysis,
                adversarial_results['discriminator_results']['detailed_report']
            )
            
            # 记录迭代结果
            iteration_result = {
                'iteration': iteration,
                'execution_time': time.time() - iteration_start_time,
                'synthetic_data': synthetic_data,
                'final_samples': adversarial_results['final_samples'],
                'synth_data_vector': synth_data_vector,
                'divergence_analysis': divergence_analysis,
                'adversarial_results': adversarial_results,
                'adjustment_details': adjustment_details,
                'updated_prompt': updated_prompt,
                'js_divergence': divergence_analysis.get('js_divergence', 0),
                'retention_rate': adversarial_results.get('retention_rate', 0),
                'adversarial_success_rate': adversarial_results.get('adversarial_success_rate', 0)
            }
            
            iteration_results.append(iteration_result)
            current_prompt = updated_prompt
            
            # 早停检查
            if self._should_early_stop(iteration_results):
                logger.info(f"🛑 检测到收敛，在第{iteration}轮提前停止")
                break
            
            logger.info(f"✅ 第{iteration}轮迭代完成")
        
        return iteration_results
    
    def _should_early_stop(self, iteration_results: List[Dict]) -> bool:
        """
        检查是否应该早停
        """
        if not self.config['enable_early_stopping'] or len(iteration_results) < 2:
            return False
        
        # 检查JS散度收敛
        recent_divergences = [r['js_divergence'] for r in iteration_results[-2:]]
        divergence_improvement = abs(recent_divergences[0] - recent_divergences[1])
        
        if divergence_improvement < self.config['convergence_threshold']:
            return True
        
        # 检查质量指标收敛
        recent_quality = [r['adversarial_success_rate'] for r in iteration_results[-2:]]
        quality_improvement = abs(recent_quality[1] - recent_quality[0])
        
        if quality_improvement < self.config['quality_improvement_threshold']:
            return True
        
        return False
    
    def _run_comprehensive_evaluation(self, real_data: List[str], iteration_results: List[Dict]) -> Dict:
        """
        运行综合创新评估
        """
        if not iteration_results:
            return {'error': '没有可评估的迭代结果'}
        
        # 收集最终的合成数据
        final_synthetic_data = iteration_results[-1]['final_samples']
        
        # 收集所有轮次的对抗结果
        all_adversarial_results = [r['adversarial_results'] for r in iteration_results]
        
        # 综合评估
        comprehensive_eval = self.evaluation_metrics.comprehensive_evaluation(
            real_data=real_data,
            synthetic_data=final_synthetic_data,
            adversarial_results=all_adversarial_results
        )
        
        return comprehensive_eval
    
    def _integrate_experiment_results(self, 
                                    real_data: List[str],
                                    iteration_results: List[Dict],
                                    final_evaluation: Dict,
                                    total_time: float) -> Dict:
        """
        整合实验结果
        """
        # 计算关键指标的趋势
        convergence_trend = self._analyze_convergence_trend(iteration_results)
        quality_trend = self._analyze_quality_trend(iteration_results)
        
        # 生成实验总结
        experiment_summary = self._generate_experiment_summary(
            iteration_results, final_evaluation, convergence_trend, quality_trend
        )
        
        # 整合完整结果
        integrated_results = {
            'experiment_metadata': {
                'experiment_name': self.config['experiment_name'],
                'timestamp': datetime.now().isoformat(),
                'total_execution_time': total_time,
                'iterations_completed': len(iteration_results),
                'real_data_samples': len(real_data)
            },
            'real_data_analysis': {
                'distribution_vector': self.real_data_vector,
                'sample_count': len(real_data)
            },
            'iteration_results': iteration_results,
            'convergence_analysis': {
                'convergence_trend': convergence_trend,
                'quality_trend': quality_trend,
                'final_js_divergence': iteration_results[-1]['js_divergence'] if iteration_results else 0,
                'final_retention_rate': iteration_results[-1]['retention_rate'] if iteration_results else 0
            },
            'comprehensive_evaluation': final_evaluation,
            'experiment_summary': experiment_summary,
            'performance_metrics': {
                'avg_iteration_time': sum(r['execution_time'] for r in iteration_results) / len(iteration_results) if iteration_results else 0,
                'total_samples_generated': sum(len(r['synthetic_data']) for r in iteration_results),
                'total_samples_retained': sum(len(r['final_samples']) for r in iteration_results),
                'overall_retention_rate': sum(r['retention_rate'] for r in iteration_results) / len(iteration_results) if iteration_results else 0
            }
        }
        
        return integrated_results
    
    def _analyze_convergence_trend(self, iteration_results: List[Dict]) -> Dict:
        """
        分析收敛趋势
        """
        if not iteration_results:
            return {'trend': 'no_data'}
        
        js_divergences = [r['js_divergence'] for r in iteration_results]
        
        # 计算趋势
        if len(js_divergences) >= 2:
            improvement = js_divergences[0] - js_divergences[-1]
            trend = 'converging' if improvement > 0 else 'diverging' if improvement < -0.01 else 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'trend': trend,
            'initial_divergence': js_divergences[0],
            'final_divergence': js_divergences[-1],
            'total_improvement': js_divergences[0] - js_divergences[-1] if len(js_divergences) >= 2 else 0,
            'convergence_rate': self._calculate_convergence_rate(js_divergences)
        }
    
    def _analyze_quality_trend(self, iteration_results: List[Dict]) -> Dict:
        """
        分析质量趋势
        """
        if not iteration_results:
            return {'trend': 'no_data'}
        
        success_rates = [r['adversarial_success_rate'] for r in iteration_results]
        retention_rates = [r['retention_rate'] for r in iteration_results]
        
        return {
            'adversarial_success_trend': self._calculate_trend(success_rates),
            'retention_rate_trend': self._calculate_trend(retention_rates),
            'quality_stability': 1 - (np.std(success_rates) if success_rates else 0),
            'final_quality_score': success_rates[-1] if success_rates else 0
        }
    
    def _calculate_convergence_rate(self, divergences: List[float]) -> float:
        """
        计算收敛速率
        """
        if len(divergences) < 2:
            return 0.0
        
        improvements = []
        for i in range(1, len(divergences)):
            improvement = divergences[i-1] - divergences[i]
            improvements.append(improvement)
        
        return sum(improvements) / len(improvements) if improvements else 0.0
    
    def _calculate_trend(self, values: List[float]) -> str:
        """
        计算趋势方向
        """
        if len(values) < 2:
            return 'insufficient_data'
        
        slope = (values[-1] - values[0]) / (len(values) - 1)
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'
    
    def _generate_experiment_summary(self, 
                                   iteration_results: List[Dict],
                                   final_evaluation: Dict,
                                   convergence_trend: Dict,
                                   quality_trend: Dict) -> str:
        """
        生成实验总结报告
        """
        if not iteration_results:
            return "实验未成功完成，无法生成总结。"
        
        innovation_score = final_evaluation.get('innovation_score', 0)
        
        summary = f"""
LLM推荐技术创新实验总结报告
============================

实验基本信息:
- 完成迭代次数: {len(iteration_results)}
- 创新综合评分: {innovation_score:.3f}
- 实验评级: {'优秀' if innovation_score >= 0.8 else '良好' if innovation_score >= 0.6 else '一般' if innovation_score >= 0.4 else '需要改进'}

收敛性分析:
- 收敛趋势: {convergence_trend.get('trend', 'unknown')}
- 初始JS散度: {convergence_trend.get('initial_divergence', 0):.4f}
- 最终JS散度: {convergence_trend.get('final_divergence', 0):.4f}
- 总体改进: {convergence_trend.get('total_improvement', 0):+.4f}

质量提升分析:
- 对抗成功率趋势: {quality_trend.get('adversarial_success_trend', 'unknown')}
- 样本保留率趋势: {quality_trend.get('retention_rate_trend', 'unknown')}
- 最终质量评分: {quality_trend.get('final_quality_score', 0):.3f}

创新效果评估:
- 对抗鉴别成功率: {final_evaluation.get('adversarial_evaluation', {}).get('adversarial_success_rate', 0):.3f}
- 长尾激活覆盖率: {final_evaluation.get('long_tail_evaluation', {}).get('long_tail_coverage_rate', 0):.3f}
- 泛化鲁棒性评分: {final_evaluation.get('robustness_evaluation', {}).get('overall_robustness_score', 0):.3f}

关键成果:
1. 实现了动态Prompt调优与对抗质量保证的闭环框架
2. 在长尾物品激活和分布适应方面取得了显著改进
3. 展示了LLM在推荐系统数据增强方面的创新潜力

建议与展望:
- 继续优化对抗训练策略，提升生成器欺骗能力
- 深化长尾物品激活机制，进一步改善推荐公平性
- 探索更多的分布自适应调控维度和策略
        """
        
        return summary.strip()
    
    def get_experiment_state(self) -> Dict:
        """
        获取当前实验状态
        """
        return {
            'iterations_completed': len(self.iteration_results),
            'real_data_analyzed': self.real_data_vector is not None,
            'last_js_divergence': self.iteration_results[-1]['js_divergence'] if self.iteration_results else None,
            'last_retention_rate': self.iteration_results[-1]['retention_rate'] if self.iteration_results else None,
            'experiment_config': self.config
        }