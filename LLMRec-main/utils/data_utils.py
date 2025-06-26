# filepath: d:\VScode\LLMRec-main\utils\data_utils.py
import json
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def load_data(file_path: str) -> List[str]:
    """
    加载数据文件
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"数据文件不存在: {file_path}")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    logger.warning("JSON文件格式不正确，应为列表格式")
                    return []
            else:
                # 按行读取
                return [line.strip() for line in f if line.strip()]
    
    except Exception as e:
        logger.error(f"加载数据文件失败: {e}")
        return []

def save_results(results: Dict[str, Any], file_path: str):
    """
    保存实验结果
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {file_path}")
    
    except Exception as e:
        logger.error(f"保存结果失败: {e}")

def save_data(data: List[str], file_path: str):
    """
    保存数据到文件
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if file_path.endswith('.json'):
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                for line in data:
                    f.write(line + '\n')
        
        logger.info(f"数据已保存到: {file_path}")
    
    except Exception as e:
        logger.error(f"保存数据失败: {e}")

def create_sample_data(num_users: int = 100, num_items: int = 200) -> List[str]:
    """
    创建示例数据
    """
    import numpy as np
    np.random.seed(42)
    
    sample_data = []
    
    for user_id in range(num_users):
        # 不同活跃度的用户
        if user_id < num_users * 0.3:  # 30%低活跃用户
            num_interactions = np.random.poisson(2) + 1
        elif user_id < num_users * 0.7:  # 40%中活跃用户
            num_interactions = np.random.poisson(5) + 2
        else:  # 30%高活跃用户
            num_interactions = np.random.poisson(8) + 3
        
        num_interactions = min(num_interactions, 15)
        
        # 物品选择考虑流行度分布
        if np.random.random() < 0.7:
            # 70%概率选择头部物品
            item_pool = range(1, min(50, num_items))
        else:
            # 30%概率选择长尾物品
            item_pool = range(50, num_items)
        
        items = np.random.choice(
            list(item_pool), 
            size=min(num_interactions, len(item_pool)), 
            replace=False
        )
        
        items_str = ' '.join(map(str, sorted(items)))
        sample_data.append(f"{user_id} {items_str}")
    
    return sample_data

def validate_data_format(data: List[str]) -> bool:
    """
    验证数据格式
    """
    if not data:
        return False
    
    for i, line in enumerate(data):
        parts = line.strip().split()
        if len(parts) < 2:
            logger.warning(f"第{i+1}行数据格式不正确: {line}")
            return False
        
        try:
            user_id = int(parts[0])
            items = [int(x) for x in parts[1:]]
        except ValueError:
            logger.warning(f"第{i+1}行包含非数字数据: {line}")
            return False
    
    return True

def split_data(data: List[str], train_ratio: float = 0.8) -> tuple:
    """
    分割数据为训练集和测试集
    """
    import random
    random.seed(42)
    
    data_copy = data.copy()
    random.shuffle(data_copy)
    
    split_point = int(len(data_copy) * train_ratio)
    train_data = data_copy[:split_point]
    test_data = data_copy[split_point:]
    
    return train_data, test_data