import logging
from typing import Dict, List, Tuple, Any
import json
import numpy as np

logger = logging.getLogger(__name__)

class DynamicPromptTuner:
    """
    动态Prompt调优器
    根据分布偏差和判别器反馈智能调整生成策略
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.prompt_library = self._initialize_prompt_library()
        self.adjustment_history = []
        
    def _default_config(self) -> Dict:
        return {
            'divergence_threshold': 0.1,
            'max_prompt_length': 512,
            'adjustment_strength': 0.3,
            'feedback_weight': 0.7,
            'simulation_mode': True,
            'max_iterations': 5,
            'convergence_tolerance': 0.05
        }
    
    def _initialize_prompt_library(self) -> Dict:
        """
        初始化结构化Prompt库
        """
        return {
            'base_prompt': "请生成推荐系统的用户-物品交互数据",
            'session_length_control': {
                'shorten': "限制每个用户的交互物品数量在{max_items}个以内",
                'lengthen': "确保每个用户至少与{min_items}个物品交互",
                'balance': "保持交互长度在{min_items}到{max_items}个物品之间"
            },
            'user_activity_control': {
                'increase_casual': "增加低活跃度用户的比例，生成更多仅有少量交互的用户",
                'increase_power': "增加高活跃度用户的比例，生成更多有大量交互的用户",
                'balance_activity': "平衡不同活跃度用户的分布"
            },
            'item_popularity_control': {
                'reduce_head_bias': "减少热门物品的集中度，增加长尾物品的出现频率",
                'increase_diversity': "提高物品多样性，避免过度集中在少数物品上",
                'natural_distribution': "遵循自然的物品流行度分布"
            },
            'quality_enhancement': {
                'realism': "确保生成的数据符合真实用户行为模式",
                'diversity': "保持用户兴趣和物品类型的多样性",
                'consistency': "维持数据的内在逻辑一致性"
            }
        }
    
    def optimize_prompt(self, real_stats, synthetic_stats_history, iteration):
        """
        适配器方法，为实验框架提供接口
        兼容 adjust_prompt_strategy 的功能
        """
        try:
            # 如果有历史数据，计算散度分析
            if synthetic_stats_history and len(synthetic_stats_history) > 0:
                latest_synth_stats = synthetic_stats_history[-1]
                
                # 计算简化的散度分析
                divergence_analysis = self._calculate_simple_divergence(real_stats, latest_synth_stats)
                
                # 使用现有的 adjust_prompt_strategy 方法
                optimized_prompt, adjustment_details = self.adjust_prompt_strategy(
                    real_vector=real_stats,
                    synth_vector=latest_synth_stats,
                    divergence_analysis=divergence_analysis,
                    discriminator_feedback=""
                )
            else:
                # 首次迭代，生成基础prompt
                optimized_prompt = self._generate_initial_prompt(real_stats)
                adjustment_details = {
                    'critical_dimensions': [],
                    'adjustment_strategies': {},
                    'discriminator_feedback_used': False,
                    'js_divergence': 0.0
                }
            
            # 生成反馈信息
            feedback = self._generate_feedback(real_stats, synthetic_stats_history, iteration)
            
            return {
                'optimized_prompt': optimized_prompt,
                'feedback': feedback,
                'iteration': iteration,
                'prompt_version': f'v{iteration + 1}',
                'adjustment_details': adjustment_details
            }
            
        except Exception as e:
            logger.error(f"   ⚠️ Prompt优化失败: {e}")
            return {
                'optimized_prompt': self._get_fallback_prompt(real_stats),
                'feedback': "",
                'iteration': iteration,
                'prompt_version': 'fallback'
            }

    def _calculate_simple_divergence(self, real_stats, synth_stats):
        """
        计算简化的散度分析
        """
        divergence_analysis = {}
        
        # 比较用户统计
        real_user = real_stats.get('user_stats', {})
        synth_user = synth_stats.get('user_stats', {})
        
        for key in ['mean', 'std']:
            real_val = real_user.get(key, 0)
            synth_val = synth_user.get(key, 0)
            divergence_analysis[f'user_activity_{key}_diff'] = abs(real_val - synth_val)
        
        # 比较物品统计
        real_item = real_stats.get('item_stats', {})
        synth_item = synth_stats.get('item_stats', {})
        
        for key in ['gini', 'long_tail_ratio']:
            real_val = real_item.get(key, 0)
            synth_val = synth_item.get(key, 0)
            divergence_analysis[f'item_{key}_diff'] = abs(real_val - synth_val)
        
        # 计算整体JS散度（简化版本）
        real_vector = real_stats.get('feature_vector', [])
        synth_vector = synth_stats.get('feature_vector', [])
        
        if real_vector and synth_vector:
            min_len = min(len(real_vector), len(synth_vector))
            real_arr = np.array(real_vector[:min_len])
            synth_arr = np.array(synth_vector[:min_len])
            
            # 简化的JS散度计算
            js_div = np.mean(np.abs(real_arr - synth_arr))
            divergence_analysis['js_divergence'] = js_div
        else:
            divergence_analysis['js_divergence'] = 0.0
        
        return divergence_analysis

    def _generate_initial_prompt(self, real_stats):
        """
        生成初始prompt
        """
        user_stats = real_stats.get('user_stats', {})
        item_stats = real_stats.get('item_stats', {})
        
        base_prompt = self.prompt_library['base_prompt']
        
        enhanced_prompt = f"""
{base_prompt}，要求如下：

数据特征要求：
1. 用户平均交互数约{user_stats.get('mean', 3):.1f}个物品
2. 物品流行度基尼系数约{item_stats.get('gini', 0.5):.2f}
3. 长尾物品比例约{item_stats.get('long_tail_ratio', 0.3):.2f}

格式要求：
- 每行一个用户的交互记录
- 格式：用户ID 物品ID1 物品ID2 物品ID3...
- 用户ID范围：0-999，物品ID范围：0-499
- 每个用户2-10个不重复物品交互

请生成真实、多样化的推荐系统交互数据。
"""
        
        return enhanced_prompt

    def _generate_feedback(self, real_stats, synthetic_stats_history, iteration):
        """
        生成反馈信息
        """
        if iteration == 0:
            return "首次生成，请确保数据真实性和多样性"
        
        if not synthetic_stats_history:
            return "缺少历史数据，请保持数据质量"
        
        latest_stats = synthetic_stats_history[-1]
        feedback_items = []
        
        # 检查用户活跃度
        real_mean = real_stats.get('user_stats', {}).get('mean', 3)
        synth_mean = latest_stats.get('user_stats', {}).get('mean', 3)
        
        if abs(real_mean - synth_mean) > 0.5:
            if synth_mean > real_mean:
                feedback_items.append("减少用户交互数量，降低平均活跃度")
            else:
                feedback_items.append("增加用户交互数量，提高平均活跃度")
        
        # 检查物品流行度
        real_gini = real_stats.get('item_stats', {}).get('gini', 0.5)
        synth_gini = latest_stats.get('item_stats', {}).get('gini', 0.5)
        
        if abs(real_gini - synth_gini) > 0.1:
            if synth_gini > real_gini:
                feedback_items.append("减少物品流行度集中度，增加分布均匀性")
            else:
                feedback_items.append("增加热门物品的集中度，符合真实分布")
        
        if feedback_items:
            return "第{}轮优化建议：{}".format(iteration, "；".join(feedback_items))
        else:
            return f"第{iteration}轮：数据质量良好，继续保持"

    def _get_fallback_prompt(self, real_stats):
        """
        获取后备prompt
        """
        return """
生成推荐系统用户-物品交互数据：

要求：
1. 每行格式：用户ID 物品ID1 物品ID2 ...
2. 用户ID：0-999，物品ID：0-499
3. 每用户2-8个不重复物品
4. 体现真实的用户行为和物品流行度分布

请生成高质量的交互数据。
"""
    
    def adjust_prompt_strategy(self, 
                             real_vector: Dict, 
                             synth_vector: Dict, 
                             divergence_analysis: Dict,
                             discriminator_feedback: str = "") -> Tuple[str, Dict]:
        """
        根据偏差分析和判别器反馈调整Prompt策略
        
        Returns:
            Tuple[str, Dict]: (调整后的prompt, 调整详情)
        """
        logger.info("🎯 开始动态Prompt调优...")
        
        # 1. 分析关键偏差维度
        critical_dimensions = self._identify_critical_divergences(divergence_analysis)
        
        # 2. 根据偏差生成调整策略
        adjustment_strategies = self._generate_adjustment_strategies(
            real_vector, synth_vector, critical_dimensions
        )
        
        # 3. 融合判别器反馈
        if discriminator_feedback:
            feedback_strategies = self._parse_discriminator_feedback(discriminator_feedback)
            adjustment_strategies.update(feedback_strategies)
        
        # 4. 构建优化后的Prompt
        optimized_prompt = self._construct_optimized_prompt(adjustment_strategies)
        
        # 5. 记录调整历史
        adjustment_details = {
            'critical_dimensions': critical_dimensions,
            'adjustment_strategies': adjustment_strategies,
            'discriminator_feedback_used': bool(discriminator_feedback),
            'js_divergence': divergence_analysis.get('js_divergence', 0)
        }
        
        self.adjustment_history.append(adjustment_details)
        
        logger.info(f"✅ Prompt调优完成，识别{len(critical_dimensions)}个关键偏差维度")
        
        return optimized_prompt, adjustment_details
    
    def _identify_critical_divergences(self, divergence_analysis: Dict) -> List[str]:
        """
        识别关键偏差维度
        """
        critical_dims = []
        threshold = self.config['divergence_threshold']
        
        for key, value in divergence_analysis.items():
            if key.endswith('_diff') and value > threshold:
                # 移除'_diff'后缀获取原始维度名
                original_dim = key[:-5]
                critical_dims.append(original_dim)
        
        # 根据重要性排序
        critical_dims.sort(key=lambda x: divergence_analysis.get(f"{x}_diff", 0), reverse=True)
        
        return critical_dims[:5]  # 只关注前5个最重要的偏差
    
    def _generate_adjustment_strategies(self, 
                                      real_vector: Dict, 
                                      synth_vector: Dict, 
                                      critical_dimensions: List[str]) -> Dict:
        """
        根据偏差维度生成调整策略
        """
        strategies = {}
        
        for dim in critical_dimensions:
            real_val = real_vector.get(dim, 0)
            synth_val = synth_vector.get(dim, 0)
            
            if 'session_length' in dim:
                strategies.update(self._adjust_session_length(real_val, synth_val, dim))
            elif 'user_activity' in dim:
                strategies.update(self._adjust_user_activity(real_val, synth_val, dim))
            elif 'item_popularity' in dim or 'item_diversity' in dim:
                strategies.update(self._adjust_item_distribution(real_val, synth_val, dim))
            elif 'user_type' in dim:
                strategies.update(self._adjust_user_types(real_val, synth_val, dim))
        
        return strategies
    
    def _adjust_session_length(self, real_val: float, synth_val: float, dim: str) -> Dict:
        """
        调整会话长度相关策略
        """
        strategies = {}
        
        if 'mean' in dim:
            if synth_val > real_val:
                # 合成数据会话过长
                max_items = max(3, int(real_val * 1.2))
                strategies['session_control'] = self.prompt_library['session_length_control']['shorten'].format(max_items=max_items)
            else:
                # 合成数据会话过短
                min_items = max(2, int(real_val * 0.8))
                strategies['session_control'] = self.prompt_library['session_length_control']['lengthen'].format(min_items=min_items)
        
        elif 'short_session_ratio' in dim:
            if synth_val > real_val:
                strategies['session_diversity'] = "减少过短会话的比例，增加中等长度的交互"
            else:
                strategies['session_diversity'] = "增加一些简短交互，保持用户行为的多样性"
        
        return strategies
    
    def _adjust_user_activity(self, real_val: float, synth_val: float, dim: str) -> Dict:
        """
        调整用户活跃度相关策略
        """
        strategies = {}
        
        if 'gini' in dim:
            if synth_val > real_val:
                strategies['activity_balance'] = self.prompt_library['user_activity_control']['balance_activity']
            else:
                strategies['activity_diversity'] = "增加用户活跃度的差异化，创造更明显的活跃度层次"
        
        elif 'casual_user_ratio' in dim:
            if synth_val < real_val:
                strategies['user_type_adjust'] = self.prompt_library['user_activity_control']['increase_casual']
        
        elif 'power_user_ratio' in dim:
            if synth_val < real_val:
                strategies['user_type_adjust'] = self.prompt_library['user_activity_control']['increase_power']
        
        return strategies
    
    def _adjust_item_distribution(self, real_val: float, synth_val: float, dim: str) -> Dict:
        """
        调整物品分布相关策略
        """
        strategies = {}
        
        if 'popularity_gini' in dim:
            if synth_val < real_val:
                strategies['item_concentration'] = "增加热门物品的集中度，符合真实的幂律分布"
            else:
                strategies['item_distribution'] = self.prompt_library['item_popularity_control']['reduce_head_bias']
        
        elif 'diversity_entropy' in dim:
            if synth_val < real_val:
                strategies['item_variety'] = self.prompt_library['item_popularity_control']['increase_diversity']
        
        elif 'head_items_dominance' in dim:
            if synth_val < real_val:
                strategies['popularity_realism'] = "增强头部物品的主导地位，反映真实的流行度分布"
        
        return strategies
    
    def _adjust_user_types(self, real_val: float, synth_val: float, dim: str) -> Dict:
        """
        调整用户类型分布策略
        """
        strategies = {}
        
        if 'diversity' in dim:
            if synth_val < real_val:
                strategies['user_type_variety'] = "增加用户类型的多样性，创造更丰富的用户画像"
        
        return strategies
    
    def _parse_discriminator_feedback(self, feedback: str) -> Dict:
        """
        解析判别器反馈并生成对应策略
        """
        strategies = {}
        
        # 简化的反馈解析逻辑
        feedback_lower = feedback.lower()
        
        if '过于集中' in feedback or 'concentrated' in feedback_lower:
            strategies['anti_concentration'] = self.prompt_library['item_popularity_control']['increase_diversity']
        
        if '不够真实' in feedback or 'unrealistic' in feedback_lower:
            strategies['realism_boost'] = self.prompt_library['quality_enhancement']['realism']
        
        if '模式单一' in feedback or 'pattern' in feedback_lower:
            strategies['pattern_diversity'] = self.prompt_library['quality_enhancement']['diversity']
        
        if '长度异常' in feedback or 'length' in feedback_lower:
            strategies['length_normalize'] = "确保交互长度符合自然分布"
        
        if '用户行为' in feedback or 'behavior' in feedback_lower:
            strategies['behavior_realism'] = "模拟更真实的用户行为模式和偏好"
        
        return strategies
    
    def _construct_optimized_prompt(self, strategies: Dict) -> str:
        """
        构建优化后的Prompt
        """
        base_prompt = self.prompt_library['base_prompt']
        
        # 添加策略指令
        strategy_instructions = []
        for strategy_name, instruction in strategies.items():
            strategy_instructions.append(f"- {instruction}")
        
        if strategy_instructions:
            optimized_prompt = f"{base_prompt}，请特别注意以下要求：\n" + "\n".join(strategy_instructions)
        else:
            optimized_prompt = base_prompt
        
        # 添加格式要求
        format_instruction = "\n\n请按以下格式生成数据：每行一个用户的交互记录，格式为'用户ID 物品ID1 物品ID2 ...'，用户ID和物品ID都是正整数。"
        optimized_prompt += format_instruction
        
        return optimized_prompt
    
    def generate_synthetic_data(self, prompt: str, num_samples: int = 50) -> List[str]:
        """
        使用优化后的Prompt生成合成数据
        """
        logger.info(f"🔄 使用调优Prompt生成{num_samples}个合成样本...")
        
        if self.config['simulation_mode']:
            return self._simulate_data_generation(prompt, num_samples)
        else:
            # 这里应该接入真实的LLM API
            return self._call_llm_api(prompt, num_samples)
    
    def _simulate_data_generation(self, prompt: str, num_samples: int) -> List[str]:
        """
        模拟数据生成（用于测试）
        """
        synthetic_data = []
        
        # 根据prompt中的指令调整生成参数
        avg_length = 5
        if '限制' in prompt and '个以内' in prompt:
            try:
                # 提取最大长度限制
                import re
                match = re.search(r'(\d+)个以内', prompt)
                if match:
                    avg_length = min(avg_length, int(match.group(1)))
            except:
                pass
        
        if '至少' in prompt and '个物品' in prompt:
            try:
                import re
                match = re.search(r'至少(\d+)个物品', prompt)
                if match:
                    avg_length = max(avg_length, int(match.group(1)))
            except:
                pass
        
        # 生成模拟数据
        for i in range(num_samples):
            user_id = i
            # 根据prompt调整会话长度
            if '增加低活跃度' in prompt:
                session_length = np.random.poisson(2) + 1
            elif '增加高活跃度' in prompt:
                session_length = np.random.poisson(8) + 3
            else:
                session_length = np.random.poisson(avg_length) + 1
            
            session_length = max(1, min(session_length, 15))
            
            # 生成物品ID
            if '长尾' in prompt or '多样性' in prompt:
                # 增加长尾物品
                items = list(np.random.choice(range(100, 1000), size=session_length, replace=False))
            else:
                # 常规物品分布
                items = list(np.random.choice(range(1, 100), size=session_length, replace=False))
            
            items_str = ' '.join(map(str, items))
            synthetic_data.append(f"{user_id} {items_str}")
        
        return synthetic_data
    
    def _call_llm_api(self, prompt: str, num_samples: int) -> List[str]:
        """
        调用真实LLM API生成数据
        """
        # 这里应该实现真实的API调用
        # 例如OpenAI GPT-4 API调用
        pass
    
    def get_adjustment_summary(self) -> Dict:
        """
        获取调整历史总结
        """
        if not self.adjustment_history:
            return {'total_adjustments': 0}
        
        return {
            'total_adjustments': len(self.adjustment_history),
            'avg_js_divergence': np.mean([h.get('js_divergence', 0) for h in self.adjustment_history]),
            'most_critical_dimensions': self._get_most_frequent_dimensions(),
            'feedback_usage_rate': sum(1 for h in self.adjustment_history if h['discriminator_feedback_used']) / len(self.adjustment_history)
        }
    
    def _get_most_frequent_dimensions(self) -> List[str]:
        """
        获取最频繁的偏差维度
        """
        dimension_counts = {}
        for history in self.adjustment_history:
            for dim in history.get('critical_dimensions', []):
                dimension_counts[dim] = dimension_counts.get(dim, 0) + 1
        
        return sorted(dimension_counts.keys(), key=lambda x: dimension_counts[x], reverse=True)[:3]