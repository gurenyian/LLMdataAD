import sys
import os
import json
import time
from datetime import datetime

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from llm_framework import LLMRecommendationFramework
from evaluation_metrics import LLMEvaluationMetrics
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LLMRecExperiment:
    """LLM推荐系统完整实验"""
    
    def __init__(self, config: dict = None):
        self.config = config or self._default_config()
        self.results = {}
        
    def _default_config(self):
        return {
            'experiment_name': 'LLM_Rec_Innovation_Test',
            'data_config': {
                'n_users': 1000,
                'n_items': 500,
                'n_train_users': 200,  # 实际生成训练数据的用户数
                'min_interactions': 1,
                'max_interactions': 15,
                'long_tail_ratio': 0.7  # 长尾物品比例
            },
            'framework_config': {
                'max_iterations': 5,
                'samples_per_iteration': 100,
                'adversarial_rounds': 3,
                'early_stopping': True,
                'convergence_tolerance': 0.05
            },
            'evaluation_config': {
                'metrics': ['adversarial_success_rate', 'long_tail_activation', 'generalization_robustness'],
                'save_detailed_results': True,
                'generate_visualizations': True
            }
        }
    
    def create_realistic_data_generator(self):
        """创建现实的数据生成器"""
        logger.info("创建现实数据生成器...")
        
        class RealisticDataGenerator:
            def __init__(self, config):
                self.n_users = config['n_users']
                self.n_items = config['n_items']
                self.n_train = 0
                self.n_test = 0
                self.train_items = {}
                
                # 生成现实的交互数据
                self._generate_realistic_interactions(config)
            
            def _generate_realistic_interactions(self, config):
                import random
                import numpy as np
                
                random.seed(42)
                np.random.seed(42)
                
                # 模拟不同类型的用户
                user_types = ['low_active', 'medium_active', 'high_active']
                type_ratios = [0.5, 0.3, 0.2]  # 符合现实的用户活跃度分布
                
                for user_id in range(config['n_train_users']):
                    # 随机选择用户类型
                    user_type = np.random.choice(user_types, p=type_ratios)
                    
                    # 根据用户类型确定交互数量
                    if user_type == 'low_active':
                        n_interactions = random.randint(1, 3)
                    elif user_type == 'medium_active':
                        n_interactions = random.randint(3, 8)
                    else:  # high_active
                        n_interactions = random.randint(8, config['max_interactions'])
                    
                    # 生成物品交互（体现长尾分布）
                    items = set()
                    for _ in range(n_interactions):
                        if random.random() < 0.2:  # 20%概率选择热门物品
                            item_id = random.randint(0, 49)  # 前50个为热门物品
                        elif random.random() < 0.5:  # 30%概率选择中等物品
                            item_id = random.randint(50, 199)
                        else:  # 50%概率选择长尾物品
                            item_id = random.randint(200, config['n_items'] - 1)
                        
                        items.add(item_id)
                    
                    if items:
                        self.train_items[user_id] = list(items)
                        self.n_train += len(items)
                
                logger.info(f"生成了{len(self.train_items)}个用户，{self.n_train}个交互")
        
        return RealisticDataGenerator(self.config['data_config'])
    
    def run_complete_experiment(self):
        """运行完整实验"""
        logger.info(f"🚀 开始LLM推荐系统创新实验: {self.config['experiment_name']}")
        
        experiment_start_time = time.time()
        
        try:
            # 1. 创建数据生成器
            data_generator = self.create_realistic_data_generator()
            
            # 2. 初始化LLM框架
            logger.info("初始化LLM推荐框架...")
            framework = LLMRecommendationFramework(
                data_generator=data_generator,
                config=self.config['framework_config']
            )
            
            # 3. 运行框架
            logger.info("运行LLM推荐技术创新框架...")
            framework_results = framework.run_complete_framework()
            
            if 'error' in framework_results:
                logger.error(f"框架运行失败: {framework_results['error']}")
                return None
            
            # 4. 计算创新评估指标
            logger.info("计算创新评估指标...")
            evaluator = LLMEvaluationMetrics()
            
            # 构建评估所需的数据结构
            evaluation_data = {
                'real_stats': framework.real_stats,
                'synthetic_stats': framework.synthetic_stats_history,
                'convergence_history': framework_results['convergence_history'],
                'quality_history': framework_results['quality_history'],
                'adversarial_results': []  # 这里需要从framework_results中提取对抗结果
            }
            
            # 提取对抗结果（如果有的话）
            for iteration_result in framework_results['iterations']:
                if 'adversarial_results' in iteration_result:
                    evaluation_data['adversarial_results'].append(iteration_result['adversarial_results'])
            
            innovation_metrics = evaluator.calculate_comprehensive_metrics(evaluation_data)
            
            # 5. 生成评估报告
            evaluation_report = evaluator.generate_evaluation_report(innovation_metrics)
            
            # 6. 整合实验结果
            total_time = time.time() - experiment_start_time
            
            experiment_results = {
                'experiment_config': self.config,
                'framework_results': framework_results,
                'innovation_metrics': innovation_metrics,
                'evaluation_report': evaluation_report,
                'execution_time': total_time,
                'timestamp': datetime.now().isoformat()
            }
            
            # 7. 保存结果
            self._save_experiment_results(experiment_results)
            
            # 8. 打印总结
            self._print_experiment_summary(experiment_results)
            
            return experiment_results
            
        except Exception as e:
            logger.error(f"实验执行失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_experiment_results(self, results):
        """保存实验结果"""
        os.makedirs('results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存JSON格式结果
        json_file = f'results/experiment_results_{timestamp}.json'
        try:
            # 过滤不能JSON序列化的对象
            serializable_results = self._make_serializable(results)
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            logger.info(f"实验结果已保存到: {json_file}")
        except Exception as e:
            logger.error(f"保存JSON结果失败: {e}")
        
        # 保存评估报告
        report_file = f'results/evaluation_report_{timestamp}.txt'
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(results['evaluation_report'])
            logger.info(f"评估报告已保存到: {report_file}")
        except Exception as e:
            logger.error(f"保存评估报告失败: {e}")
    
    def _make_serializable(self, obj):
        """使对象可JSON序列化"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # numpy数组
            return obj.tolist()
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)
    
    def _print_experiment_summary(self, results):
        """打印实验总结"""
        print("\n" + "="*80)
        print("🏆 LLM推荐系统创新实验完成")
        print("="*80)
        
        framework_metrics = results['framework_results'].get('final_metrics', {})
        innovation_metrics = results['innovation_metrics']
        
        print(f"⏱️  总执行时间: {results['execution_time']:.2f}秒")
        print(f"🔄 框架迭代次数: {framework_metrics.get('total_iterations', 0)}")
        print(f"📈 最佳收敛分数: {framework_metrics.get('best_convergence_score', 0):.4f}")
        print(f"✨ 高质量样本数: {framework_metrics.get('total_high_quality_samples', 0)}")
        
        print(f"\n🎯 创新评估指标:")
        print(f"   对抗成功率: {innovation_metrics['adversarial_success_rate']:.3f}")
        print(f"   长尾激活度: {innovation_metrics['long_tail_activation']:.3f}")
        print(f"   泛化鲁棒性: {innovation_metrics['generalization_robustness']:.3f}")
        print(f"   总体创新评分: {innovation_metrics['overall_innovation_score']:.3f}")
        
        print(f"\n📋 详细评估报告:")
        print(results['evaluation_report'])


def main():
    """主函数"""
    # 创建必要的目录
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 实验配置
    experiment_config = {
        'experiment_name': 'LLM_Recommendation_Innovation_Experiment',
        'data_config': {
            'n_users': 1000,
            'n_items': 500,
            'n_train_users': 150,
            'min_interactions': 1,
            'max_interactions': 12,
            'long_tail_ratio': 0.7
        },
        'framework_config': {
            'max_iterations': 4,
            'samples_per_iteration': 50,
            'adversarial_rounds': 2,
            'early_stopping': True,
            'convergence_tolerance': 0.05,
            'simulation_mode': True
        }
    }
    
    # 创建并运行实验
    experiment = LLMRecExperiment(experiment_config)
    results = experiment.run_complete_experiment()
    
    if results:
        print("\n🎉 实验成功完成！结果已保存到results目录")
    else:
        print("\n❌ 实验执行失败，请查看日志了解详情")


if __name__ == "__main__":
    main()