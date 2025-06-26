import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Any
from collections import Counter
from scipy.stats import entropy
import pandas as pd

logger = logging.getLogger(__name__)

class DataDistributionAnalyzer:
    """
    数据分布分析器
    分析真实数据和合成数据的分布特征，生成可量化的特征向量
    """
    
    def __init__(self, data_generator=None, config: Dict = None):
        """
        兼容实验框架的构造函数
        """
        self.data_generator = data_generator
        self.config = config or self._default_config()
        
        # 如果没有传入data_generator，使用原有逻辑
        if not data_generator:
            self.real_vector_cache = None
    
    def _default_config(self) -> Dict:
        """
        返回默认配置
        """
        return {
            'user_activity_bins': 10,
            'item_popularity_bins': 20,
            'session_length_bins': 15,
            'min_interactions_threshold': 2,
            'clustering_k': 5,
            'similarity_threshold': 0.8,
            'min_samples': 10,
            'max_iterations': 100,
            'convergence_tolerance': 1e-6,
            'feature_weights': {
                'user_activity': 0.3,
                'item_popularity': 0.3,
                'temporal_patterns': 0.2,
                'diversity_metrics': 0.2
            },
            'analysis_methods': [
                'user_activity_distribution',
                'item_popularity_distribution',
                'interaction_temporal_patterns',
                'diversity_analysis'
            ]
        }
    
    def generate_feature_vector(self):
        """
        适配器方法，为实验框架提供兼容接口
        """
        if hasattr(self, 'data_generator') and self.data_generator:
            # 从数据生成器提取样本
            real_samples = []
            for user_id, items in self.data_generator.train_items.items():
                if items:
                    sample = f"{user_id} " + " ".join(map(str, items))
                    real_samples.append(sample)
            
            # 使用现有的analyze_real_data方法
            try:
                if real_samples:
                    real_vector = self.analyze_real_data(real_samples)
                    
                    # 转换为实验框架期望的格式
                    return {
                        'feature_vector': list(real_vector.values()),
                        'user_stats': {
                            'mean': real_vector.get('user_activity_mean', 3.0),
                            'std': real_vector.get('user_activity_std', 1.5)
                        },
                        'item_stats': {
                            'gini': real_vector.get('item_popularity_gini', 0.5),
                            'long_tail_ratio': 1.0 - real_vector.get('head_items_dominance', 0.7)
                        }
                    }
                else:
                    return self._get_default_feature_vector()
                    
            except Exception as e:
                print(f"   ⚠️ 数据分析出错: {e}")
                # 返回基于数据生成器统计的默认值
                return self._generate_stats_from_data_generator()
        else:
            return self._get_default_feature_vector()
    
    def _generate_stats_from_data_generator(self):
        """
        从数据生成器直接计算统计信息
        """
        try:
            if not self.data_generator or not self.data_generator.train_items:
                return self._get_default_feature_vector()
            
            # 计算用户活跃度统计
            user_activities = [len(items) for items in self.data_generator.train_items.values()]
            user_mean = np.mean(user_activities) if user_activities else 3.0
            user_std = np.std(user_activities) if len(user_activities) > 1 else 1.5
            
            # 计算物品流行度统计
            item_counts = {}
            for items in self.data_generator.train_items.values():
                for item in items:
                    item_counts[item] = item_counts.get(item, 0) + 1
            
            if item_counts:
                count_values = list(item_counts.values())
                gini = self._calculate_gini_coefficient(count_values)
                
                # 计算长尾比例
                median_count = np.median(count_values)
                long_tail_count = sum(1 for c in count_values if c <= median_count)
                long_tail_ratio = long_tail_count / len(count_values)
            else:
                gini = 0.5
                long_tail_ratio = 0.3
            
            return {
                'feature_vector': [user_mean, user_std, gini, long_tail_ratio],
                'user_stats': {
                    'mean': float(user_mean),
                    'std': float(user_std)
                },
                'item_stats': {
                    'gini': float(gini),
                    'long_tail_ratio': float(long_tail_ratio)
                }
            }
            
        except Exception as e:
            print(f"   ⚠️ 从数据生成器计算统计失败: {e}")
            return self._get_default_feature_vector()
    
    def _get_default_feature_vector(self):
        """返回默认的特征向量"""
        return {
            'feature_vector': [3.0, 1.5, 0.5, 0.3],
            'user_stats': {'mean': 3.0, 'std': 1.5},
            'item_stats': {'gini': 0.5, 'long_tail_ratio': 0.3}
        }
    
    def analyze_real_data(self, real_data: List[str]) -> Dict[str, float]:
        """
        分析真实数据分布特征
        返回量化的特征向量
        """
        logger.info("🔍 开始分析真实数据分布...")
        
        # 解析数据
        parsed_data = self._parse_interaction_data(real_data)
        
        if not parsed_data:
            logger.warning("⚠️ 真实数据为空，返回默认向量")
            return self._get_default_vector()
        
        # 1. 用户活跃度分布分析
        user_activity_features = self._analyze_user_activity(parsed_data)
        
        # 2. 物品流行度分析
        item_popularity_features = self._analyze_item_popularity(parsed_data)
        
        # 3. 交互会话分析
        session_features = self._analyze_session_patterns(parsed_data)
        
        # 4. 用户聚类分析
        user_clustering_features = self._analyze_user_clustering(parsed_data)
        
        # 整合特征向量
        feature_vector = {
            **user_activity_features,
            **item_popularity_features,
            **session_features,
            **user_clustering_features
        }
        
        # 缓存结果
        self.real_vector_cache = feature_vector
        
        logger.info(f"✅ 真实数据分析完成，提取{len(feature_vector)}个特征")
        return feature_vector
    
    def analyze_synthetic_data(self, synthetic_data: List[str]) -> Dict[str, float]:
        """
        分析合成数据分布特征
        """
        logger.info("🔍 开始分析合成数据分布...")
        
        # 解析数据
        parsed_data = self._parse_interaction_data(synthetic_data)
        
        if not parsed_data:
            logger.warning("⚠️ 合成数据为空，返回默认向量")
            return self._get_default_vector()
        
        # 使用相同的分析流程
        user_activity_features = self._analyze_user_activity(parsed_data)
        item_popularity_features = self._analyze_item_popularity(parsed_data)
        session_features = self._analyze_session_patterns(parsed_data)
        user_clustering_features = self._analyze_user_clustering(parsed_data)
        
        feature_vector = {
            **user_activity_features,
            **item_popularity_features,
            **session_features,
            **user_clustering_features
        }
        
        logger.info(f"✅ 合成数据分析完成，提取{len(feature_vector)}个特征")
        return feature_vector
    
    def _parse_interaction_data(self, data: List[str]) -> List[Dict]:
        """
        解析交互数据
        格式: "user_id item1 item2 item3 ..."
        """
        parsed = []
        for line in data:
            if not line.strip():
                continue
            
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            
            try:
                user_id = int(parts[0])
                item_ids = [int(x) for x in parts[1:]]
                
                parsed.append({
                    'user_id': user_id,
                    'items': item_ids,
                    'session_length': len(item_ids)
                })
            except ValueError:
                continue
        
        return parsed
    
    def _analyze_user_activity(self, parsed_data: List[Dict]) -> Dict[str, float]:
        """
        用户活跃度分布分析
        """
        user_interactions = Counter()
        for record in parsed_data:
            user_interactions[record['user_id']] += record['session_length']
        
        if not user_interactions:
            return {'user_activity_mean': 0.0, 'user_activity_std': 0.0, 'user_activity_gini': 0.0}
        
        activity_values = list(user_interactions.values())
        
        return {
            'user_activity_mean': float(np.mean(activity_values)),
            'user_activity_std': float(np.std(activity_values)),
            'user_activity_gini': self._calculate_gini_coefficient(activity_values),
            'active_user_ratio': len([v for v in activity_values if v >= self.config['min_interactions_threshold']]) / len(activity_values)
        }
    
    def _analyze_item_popularity(self, parsed_data: List[Dict]) -> Dict[str, float]:
        """
        物品流行度分析
        """
        item_counts = Counter()
        for record in parsed_data:
            for item_id in record['items']:
                item_counts[item_id] += 1
        
        if not item_counts:
            return {'item_popularity_gini': 0.0, 'item_diversity_entropy': 0.0}
        
        popularity_values = list(item_counts.values())
        
        # 计算基尼系数
        gini = self._calculate_gini_coefficient(popularity_values)
        
        # 计算物品类别熵值
        popularity_probs = np.array(popularity_values) / sum(popularity_values)
        diversity_entropy = entropy(popularity_probs)
        
        return {
            'item_popularity_gini': gini,
            'item_diversity_entropy': float(diversity_entropy),
            'unique_items_ratio': len(item_counts) / sum(popularity_values),
            'head_items_dominance': sum(sorted(popularity_values, reverse=True)[:int(len(popularity_values)*0.2)]) / sum(popularity_values)
        }
    
    def _analyze_session_patterns(self, parsed_data: List[Dict]) -> Dict[str, float]:
        """
        交互会话模式分析
        """
        session_lengths = [record['session_length'] for record in parsed_data]
        
        if not session_lengths:
            return {'session_length_mean': 0.0, 'session_length_std': 0.0}
        
        return {
            'session_length_mean': float(np.mean(session_lengths)),
            'session_length_std': float(np.std(session_lengths)),
            'session_length_median': float(np.median(session_lengths)),
            'short_session_ratio': len([s for s in session_lengths if s <= 3]) / len(session_lengths),
            'long_session_ratio': len([s for s in session_lengths if s >= 10]) / len(session_lengths)
        }
    
    def _analyze_user_clustering(self, parsed_data: List[Dict]) -> Dict[str, float]:
        """
        用户聚类分析
        """
        # 简化的用户行为特征提取
        user_features = {}
        for record in parsed_data:
            user_id = record['user_id']
            if user_id not in user_features:
                user_features[user_id] = {
                    'total_interactions': 0,
                    'unique_items': set(),
                    'avg_session_length': 0,
                    'session_count': 0
                }
            
            user_features[user_id]['total_interactions'] += record['session_length']
            user_features[user_id]['unique_items'].update(record['items'])
            user_features[user_id]['session_count'] += 1
        
        if not user_features:
            return {'user_type_diversity': 0.0, 'casual_user_ratio': 0.0, 'power_user_ratio': 0.0}
        
        # 计算用户类型分布
        user_types = []
        for user_id, features in user_features.items():
            total_int = features['total_interactions']
            unique_items = len(features['unique_items'])
            
            if total_int <= 5:
                user_types.append('casual')
            elif total_int <= 20:
                user_types.append('regular')
            else:
                user_types.append('power')
        
        type_counts = Counter(user_types)
        total_users = len(user_types)
        
        return {
            'user_type_diversity': entropy(list(type_counts.values())),
            'casual_user_ratio': type_counts.get('casual', 0) / total_users,
            'regular_user_ratio': type_counts.get('regular', 0) / total_users,
            'power_user_ratio': type_counts.get('power', 0) / total_users
        }
    
    def _calculate_gini_coefficient(self, values: List) -> float:
        """
        计算基尼系数
        """
        if not values:
            return 0.0
        
        sorted_values = sorted([float(x) for x in values])
        n = len(sorted_values)
        
        if n == 0:
            return 0.0
        
        values_array = np.array(sorted_values, dtype=float)
        index_array = np.arange(1, n + 1, dtype=float)
        
        total_sum = float(np.sum(values_array))
        if total_sum == 0:
            return 0.0
        
        gini = (2 * float(np.sum(index_array * values_array))) / (n * total_sum) - (n + 1) / n
        return max(0.0, min(1.0, float(gini)))
    
    def _get_default_vector(self) -> Dict[str, float]:
        """
        返回默认特征向量
        """
        return {
            'user_activity_mean': 0.0,
            'user_activity_std': 0.0,
            'user_activity_gini': 0.0,
            'active_user_ratio': 0.0,
            'item_popularity_gini': 0.0,
            'item_diversity_entropy': 0.0,
            'unique_items_ratio': 0.0,
            'head_items_dominance': 0.0,
            'session_length_mean': 0.0,
            'session_length_std': 0.0,
            'session_length_median': 0.0,
            'short_session_ratio': 0.0,
            'long_session_ratio': 0.0,
            'user_type_diversity': 0.0,
            'casual_user_ratio': 0.0,
            'regular_user_ratio': 0.0,
            'power_user_ratio': 0.0
        }
    
    def calculate_distribution_divergence(self, real_vector: Dict, synth_vector: Dict) -> Dict:
        """
        计算真实数据与合成数据的分布偏差
        """
        divergence_results = {}
        
        # 计算各维度差值
        for key in real_vector.keys():
            if key in synth_vector:
                diff = abs(real_vector[key] - synth_vector[key])
                divergence_results[f"{key}_diff"] = diff
        
        # 计算JS散度（简化版本）
        real_values = list(real_vector.values())
        synth_values = [synth_vector.get(k, 0) for k in real_vector.keys()]
        
        # 归一化
        real_norm = np.array(real_values) / (np.sum(np.abs(real_values)) + 1e-8)
        synth_norm = np.array(synth_values) / (np.sum(np.abs(synth_values)) + 1e-8)
        
        # 计算JS散度
        js_divergence = self._jensen_shannon_divergence(real_norm, synth_norm)
        divergence_results['js_divergence'] = js_divergence
        
        return divergence_results
    
    def _jensen_shannon_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        计算JS散度
        """
        # 确保概率分布非负且归一化
        p = np.abs(p) + 1e-8
        q = np.abs(q) + 1e-8
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        m = 0.5 * (p + q)
        
        kl_pm = entropy(p, m)
        kl_qm = entropy(q, m)
        
        js_div = 0.5 * kl_pm + 0.5 * kl_qm
        return float(js_div)