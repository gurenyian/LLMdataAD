"""
实验配置文件
"""

# 默认实验配置
DEFAULT_EXPERIMENT_CONFIG = {
    'max_iterations': 5,
    'samples_per_iteration': 100,
    'convergence_threshold': 0.05,
    'quality_improvement_threshold': 0.1,
    'enable_early_stopping': True,
    'save_intermediate_results': True,
    'experiment_name': 'LLM_Rec_Innovation_Default'
}

# 数据分布分析器配置
DATA_ANALYZER_CONFIG = {
    'user_activity_bins': 10,
    'item_popularity_bins': 20,
    'session_length_bins': 15,
    'min_interactions_threshold': 2,
    'clustering_k': 5
}

# 动态Prompt调优器配置
PROMPT_TUNER_CONFIG = {
    'divergence_threshold': 0.1,
    'max_prompt_length': 512,
    'adjustment_strength': 0.3,
    'feedback_weight': 0.7,
    'simulation_mode': True
}

# 对抗性质量保证模块配置
ADVERSARIAL_CONFIG = {
    'quality_threshold': 0.7,
    'max_adversarial_rounds': 3,
    'discriminator_confidence_threshold': 0.8,
    'generator_improvement_threshold': 0.1,
    'simulation_mode': True
}

# 评估指标配置
EVALUATION_CONFIG = {
    'long_tail_threshold': 0.1,
    'robustness_test_ratio': 0.3,
    'temporal_window': 30,
    'domain_adaptation_threshold': 0.2
}

# 快速实验配置
QUICK_EXPERIMENT_CONFIG = {
    'max_iterations': 3,
    'samples_per_iteration': 50,
    'convergence_threshold': 0.1,
    'quality_improvement_threshold': 0.2,
    'enable_early_stopping': True,
    'experiment_name': 'LLM_Rec_Innovation_Quick'
}

# 完整实验配置
COMPLETE_EXPERIMENT_CONFIG = {
    'max_iterations': 8,
    'samples_per_iteration': 200,
    'convergence_threshold': 0.03,
    'quality_improvement_threshold': 0.05,
    'enable_early_stopping': True,
    'experiment_name': 'LLM_Rec_Innovation_Complete'
}

def get_experiment_config(experiment_type: str = 'default'):
    """
    获取实验配置
    """
    configs = {
        'default': DEFAULT_EXPERIMENT_CONFIG,
        'quick': QUICK_EXPERIMENT_CONFIG,
        'complete': COMPLETE_EXPERIMENT_CONFIG
    }
    
    base_config = configs.get(experiment_type, DEFAULT_EXPERIMENT_CONFIG).copy()
    
    # 添加子模块配置
    base_config.update({
        'analyzer_config': DATA_ANALYZER_CONFIG,
        'tuner_config': PROMPT_TUNER_CONFIG,
        'adversarial_config': ADVERSARIAL_CONFIG,
        'metrics_config': EVALUATION_CONFIG
    })
    
    return base_config