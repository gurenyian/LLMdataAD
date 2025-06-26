import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Any
import logging
from tqdm import tqdm  # 添加进度条库
import time

class NetflixDatasetLoader:
    """Netflix数据集加载器和预处理器"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.item_attributes = None
        self.user_interactions = {}
        self.item_id_mapping = {}
        self.n_users = 0
        self.n_items = 0
        self.n_train = 0
        self.n_test = 0
        self.train_items = {}
        self.test_items = {}
        
    def load_and_preprocess(self) -> Dict:
        """加载并预处理Netflix数据集"""
        try:
            # 加载item_attribute.csv
            self.logger.info(f"📂 加载数据集: {self.dataset_path}")
            print("🔄 正在读取CSV文件...")
            
            df = pd.read_csv(self.dataset_path)
            
            # 假设CSV格式: item_id, year, title
            if len(df.columns) >= 3:
                df.columns = ['item_id', 'year', 'title']
            else:
                raise ValueError("数据集格式不符合预期")
            
            print("📊 正在清洗数据...")
            # 数据清洗 - 添加进度条
            with tqdm(total=4, desc="数据清洗", ncols=80) as pbar:
                # 步骤1: 删除缺失值
                df = df.dropna(subset=['item_id', 'title'])
                pbar.update(1)
                pbar.set_postfix({"步骤": "删除缺失值"})
                
                # 步骤2: 转换年份格式
                df['year'] = pd.to_numeric(df['year'], errors='coerce')
                pbar.update(1)
                pbar.set_postfix({"步骤": "转换年份"})
                
                # 步骤3: 删除年份缺失的记录
                df = df.dropna(subset=['year'])
                pbar.update(1)
                pbar.set_postfix({"步骤": "过滤年份"})
                
                # 步骤4: 创建映射
                self.item_attributes = df
                self.n_items = len(df)
                
                # 创建物品ID映射
                unique_items = sorted(df['item_id'].unique())
                self.item_id_mapping = {item_id: idx for idx, item_id in enumerate(unique_items)}
                pbar.update(1)
                pbar.set_postfix({"步骤": "创建映射"})
            
            self.logger.info(f"✅ 成功加载 {self.n_items} 个物品")
            
            # 生成模拟用户交互数据
            self._generate_user_interactions()
            
            return {
                'n_items': self.n_items,
                'n_users': self.n_users,
                'items_loaded': len(df),
                'year_range': (df['year'].min(), df['year'].max())
            }
            
        except Exception as e:
            self.logger.error(f"❌ 数据集加载失败: {e}")
            raise
    
    def _generate_user_interactions(self):
        """基于物品属性生成真实的用户交互模式"""
        np.random.seed(42)
        
        print("🎯 正在分析物品特征...")
        # 分析物品特征
        years = self.item_attributes['year'].values
        year_popularity = self._calculate_year_popularity(years)
        
        # 生成用户偏好档案
        self.n_users = min(1000, self.n_items // 2)  # 动态调整用户数
        
        user_types = ['classic', 'modern', 'diverse', 'niche']
        type_ratios = [0.25, 0.35, 0.3, 0.1]
        
        self.logger.info(f"🔄 开始生成 {self.n_users} 个用户的交互数据...")
        print(f"👥 正在生成 {self.n_users} 个用户的交互数据...")
        
        # 使用tqdm添加进度条
        successful_users = 0
        train_users = 0
        test_users = 0
        
        with tqdm(total=self.n_users, desc="生成用户交互", ncols=100) as pbar:
            for user_id in range(self.n_users):
                user_type = np.random.choice(user_types, p=type_ratios)
                
                # 根据用户类型确定交互数量
                if user_type == 'classic':
                    n_interactions = np.random.poisson(4) + 1
                    year_preference = lambda y: 1.5 if y < 1990 else 0.8
                elif user_type == 'modern':
                    n_interactions = np.random.poisson(6) + 2
                    year_preference = lambda y: 1.5 if y > 2000 else 0.6
                elif user_type == 'diverse':
                    n_interactions = np.random.poisson(8) + 3
                    year_preference = lambda y: 1.0  # 无明显偏好
                else:  # niche
                    n_interactions = np.random.poisson(3) + 1
                    year_preference = lambda y: 2.0 if np.random.random() < 0.3 else 0.1
                
                n_interactions = min(n_interactions, min(15, self.n_items // 10))
                
                # 选择物品
                item_weights = []
                for _, row in self.item_attributes.iterrows():
                    year = row['year']
                    base_weight = year_popularity.get(year, 0.1)
                    preference_weight = year_preference(year)
                    item_weights.append(base_weight * preference_weight)
                
                # 归一化权重
                if sum(item_weights) > 0:
                    item_weights = np.array(item_weights) / sum(item_weights)
                else:
                    item_weights = np.ones(len(item_weights)) / len(item_weights)
                
                # 选择物品
                try:
                    selected_indices = np.random.choice(
                        len(self.item_attributes), 
                        size=min(n_interactions, len(self.item_attributes)), 
                        replace=False, 
                        p=item_weights
                    )
                    
                    selected_items = [self.item_attributes.iloc[idx]['item_id'] for idx in selected_indices]
                    
                    # 映射到连续ID
                    mapped_items = [self.item_id_mapping[item_id] for item_id in selected_items]
                    
                    # 80%作为训练数据，20%作为测试数据
                    if np.random.random() < 0.8:
                        self.train_items[user_id] = mapped_items
                        self.n_train += len(mapped_items)
                        train_users += 1
                    else:
                        self.test_items[user_id] = mapped_items
                        self.n_test += len(mapped_items)
                        test_users += 1
                    
                    successful_users += 1
                    
                except Exception as e:
                    # 如果选择失败，跳过这个用户
                    pass
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    "成功": successful_users,
                    "训练": train_users,
                    "测试": test_users,
                    "类型": user_type[:2]
                })
                
                # 每100个用户暂停一下，避免进度条更新太快
                if user_id % 100 == 0 and user_id > 0:
                    time.sleep(0.01)
        
        print(f"✅ 用户交互生成完成!")
        print(f"   📊 成功生成: {successful_users}/{self.n_users} 用户")
        print(f"   🏋️ 训练用户: {train_users} ({self.n_train} 交互)")
        print(f"   🧪 测试用户: {test_users} ({self.n_test} 交互)")
        
        self.logger.info(f"✅ 生成用户交互: {len(self.train_items)}训练用户, "
                        f"{len(self.test_items)}测试用户, "
                        f"{self.n_train}训练交互, {self.n_test}测试交互")
    
    def _calculate_year_popularity(self, years):
        """计算年份流行度权重"""
        print("📈 正在计算年份流行度...")
        year_counts = pd.Series(years).value_counts()
        year_popularity = {}
        
        # 添加年份分析的进度条
        with tqdm(year_counts.items(), desc="年份分析", ncols=80) as pbar:
            for year, count in pbar:
                # 新电影获得额外权重
                recency_bonus = max(0, (year - 1990) / 20) if year >= 1990 else 0
                popularity = np.log(count + 1) + recency_bonus
                year_popularity[year] = popularity
                
                pbar.set_postfix({
                    "年份": int(year),
                    "数量": count,
                    "权重": f"{popularity:.2f}"
                })
        
        # 归一化
        total_pop = sum(year_popularity.values())
        if total_pop > 0:
            year_popularity = {k: v/total_pop for k, v in year_popularity.items()}
        
        print(f"✅ 年份流行度分析完成，覆盖 {len(year_popularity)} 个年份")
        return year_popularity
    
    def get_item_metadata(self) -> Dict:
        """获取物品元数据"""
        return {
            'total_items': self.n_items,
            'year_distribution': self.item_attributes['year'].describe().to_dict(),
            'sample_titles': self.item_attributes['title'].head(10).tolist()
        }

class DatasetEnhancer:
    """数据集增强器"""
    
    def __init__(self, original_dataset, enhancement_config: Dict = None):
        self.original_dataset = original_dataset
        self.config = enhancement_config or self._default_enhancement_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 增强后的数据集
        self.enhanced_train_items = {}
        self.enhanced_test_items = {}
        self.enhancement_stats = {}
    
    def _default_enhancement_config(self):
        return {
            'augmentation_ratio': 0.3,  # 增强30%的数据
            'noise_level': 0.1,         # 10%的噪声
            'diversity_boost': True,    # 增加多样性
            'long_tail_emphasis': True, # 强调长尾物品
            'synthetic_users': 0.2      # 增加20%的合成用户
        }
    
    def enhance_dataset(self) -> Dict:
        """执行数据集增强"""
        self.logger.info("🔧 开始数据集增强...")
        print("🔧 开始数据集增强...")
        
        # 1. 复制原始数据
        print("📋 复制原始数据...")
        self.enhanced_train_items = self.original_dataset.train_items.copy()
        self.enhanced_test_items = self.original_dataset.test_items.copy()
        
        enhancement_results = {}
        
        # 计算总的增强步骤数
        total_steps = sum([
            1 if self.config['diversity_boost'] else 0,
            1 if self.config['long_tail_emphasis'] else 0,
            1 if self.config['synthetic_users'] > 0 else 0,
            1 if self.config['augmentation_ratio'] > 0 else 0
        ])
        
        print(f"🎯 将执行 {total_steps} 个增强步骤")
        
        # 2. 数据增强策略 - 添加总体进度条
        with tqdm(total=total_steps, desc="数据集增强", ncols=100) as main_pbar:
            
            if self.config['diversity_boost']:
                main_pbar.set_postfix({"当前步骤": "多样性增强"})
                diversity_result = self._boost_diversity()
                enhancement_results['diversity_boost'] = diversity_result
                main_pbar.update(1)
            
            if self.config['long_tail_emphasis']:
                main_pbar.set_postfix({"当前步骤": "长尾强调"})
                longtail_result = self._emphasize_long_tail()
                enhancement_results['long_tail_emphasis'] = longtail_result
                main_pbar.update(1)
            
            if self.config['synthetic_users'] > 0:
                main_pbar.set_postfix({"当前步骤": "合成用户"})
                synthetic_result = self._add_synthetic_users()
                enhancement_results['synthetic_users'] = synthetic_result
                main_pbar.update(1)
            
            if self.config['augmentation_ratio'] > 0:
                main_pbar.set_postfix({"当前步骤": "用户增强"})
                augment_result = self._augment_existing_users()
                enhancement_results['user_augmentation'] = augment_result
                main_pbar.update(1)
        
        # 3. 计算增强统计
        print("📊 计算增强统计...")
        original_interactions = sum(len(items) for items in self.original_dataset.train_items.values())
        enhanced_interactions = sum(len(items) for items in self.enhanced_train_items.values())
        
        self.enhancement_stats = {
            'original_users': len(self.original_dataset.train_items),
            'enhanced_users': len(self.enhanced_train_items),
            'original_interactions': original_interactions,
            'enhanced_interactions': enhanced_interactions,
            'enhancement_ratio': enhanced_interactions / original_interactions if original_interactions > 0 else 0,
            'strategies_applied': list(enhancement_results.keys())
        }
        
        print(f"✅ 数据集增强完成!")
        print(f"   👥 用户数: {self.enhancement_stats['original_users']} → {self.enhancement_stats['enhanced_users']}")
        print(f"   🔗 交互数: {original_interactions} → {enhanced_interactions}")
        print(f"   📈 增长率: +{self.enhancement_stats['enhancement_ratio']:.1%}")
        
        self.logger.info(f"✅ 数据集增强完成: {len(self.enhanced_train_items)}用户, "
                        f"{enhanced_interactions}交互 "
                        f"(增长{self.enhancement_stats['enhancement_ratio']:.1%})")
        
        return enhancement_results
    
    def _boost_diversity(self) -> Dict:
        """增强用户行为多样性"""
        print("🌈 正在增强用户行为多样性...")
        enhanced_count = 0
        total_users = len(self.enhanced_train_items)
        
        with tqdm(list(self.enhanced_train_items.items()), 
                 desc="多样性增强", ncols=80) as pbar:
            for user_id, items in pbar:
                if len(items) < 5:  # 对交互较少的用户增加多样性
                    # 计算当前用户的物品年份分布
                    current_years = []
                    for item_id in items:
                        if item_id < len(self.original_dataset.item_attributes):
                            year = self.original_dataset.item_attributes.iloc[item_id]['year']
                            current_years.append(year)
                    
                    # 如果年份过于集中，添加不同年份的物品
                    if len(set(current_years)) <= 2 and len(current_years) > 0:
                        # 选择不同年份的物品
                        available_items = list(range(self.original_dataset.n_items))
                        new_items = np.random.choice(available_items, size=2, replace=False)
                        
                        extended_items = list(set(items + new_items.tolist()))
                        self.enhanced_train_items[user_id] = extended_items
                        enhanced_count += 1
                
                pbar.set_postfix({"增强用户": enhanced_count})
        
        print(f"✅ 多样性增强完成，增强了 {enhanced_count} 个用户")
        return {'users_enhanced': enhanced_count}
    
    def _emphasize_long_tail(self) -> Dict:
        """强调长尾物品"""
        print("🦒 正在强调长尾物品...")
        
        # 计算物品流行度
        print("📊 计算物品流行度...")
        item_popularity = {}
        for items in tqdm(self.enhanced_train_items.values(), 
                         desc="统计物品流行度", ncols=80):
            for item_id in items:
                item_popularity[item_id] = item_popularity.get(item_id, 0) + 1
        
        # 识别长尾物品（出现次数少于平均值的50%）
        avg_popularity = np.mean(list(item_popularity.values())) if item_popularity else 1
        long_tail_items = [item_id for item_id, pop in item_popularity.items() 
                          if pop < avg_popularity * 0.5]
        
        print(f"🔍 识别出 {len(long_tail_items)} 个长尾物品")
        
        enhanced_count = 0
        user_list = list(self.enhanced_train_items.keys())
        
        # 为部分用户添加长尾物品
        with tqdm(user_list, desc="添加长尾物品", ncols=80) as pbar:
            for user_id in pbar:
                if np.random.random() < 0.3 and long_tail_items:  # 30%的用户
                    selected_longtail = np.random.choice(long_tail_items, size=1)[0]
                    if selected_longtail not in self.enhanced_train_items[user_id]:
                        self.enhanced_train_items[user_id].append(selected_longtail)
                        enhanced_count += 1
                
                pbar.set_postfix({"添加长尾": enhanced_count})
        
        print(f"✅ 长尾强调完成，为 {enhanced_count} 个用户添加了长尾物品")
        return {'long_tail_items': len(long_tail_items), 'users_enhanced': enhanced_count}
    
    def _add_synthetic_users(self) -> Dict:
        """添加合成用户"""
        n_synthetic = int(len(self.original_dataset.train_items) * self.config['synthetic_users'])
        print(f"🤖 正在添加 {n_synthetic} 个合成用户...")
        
        synthetic_added = 0
        
        if not self.enhanced_train_items:
            base_user_id = 0
        else:
            base_user_id = max(self.enhanced_train_items.keys()) + 1
        
        # 分析现有用户模式
        interaction_lengths = [len(items) for items in self.enhanced_train_items.values()]
        avg_length = np.mean(interaction_lengths) if interaction_lengths else 3
        
        with tqdm(range(n_synthetic), desc="生成合成用户", ncols=80) as pbar:
            for i in pbar:
                synthetic_user_id = base_user_id + i
                
                # 生成合成交互
                synthetic_length = max(1, int(np.random.poisson(avg_length)))
                synthetic_length = min(synthetic_length, 10)
                
                try:
                    synthetic_items = np.random.choice(
                        self.original_dataset.n_items, 
                        size=synthetic_length, 
                        replace=False
                    ).tolist()
                    
                    self.enhanced_train_items[synthetic_user_id] = synthetic_items
                    synthetic_added += 1
                except:
                    # 如果选择失败，跳过
                    pass
                
                pbar.set_postfix({"已添加": synthetic_added})
        
        print(f"✅ 合成用户添加完成，成功添加 {synthetic_added} 个用户")
        return {'synthetic_users_added': synthetic_added}
    
    def _augment_existing_users(self) -> Dict:
        """增强现有用户数据"""
        augment_count = int(len(self.enhanced_train_items) * self.config['augmentation_ratio'])
        print(f"🔄 正在增强 {augment_count} 个现有用户...")
        
        augmented_users = 0
        
        user_ids = list(self.enhanced_train_items.keys())
        if augment_count > len(user_ids):
            augment_count = len(user_ids)
        
        selected_users = np.random.choice(user_ids, size=augment_count, replace=False)
        
        with tqdm(selected_users, desc="增强现有用户", ncols=80) as pbar:
            for user_id in pbar:
                current_items = self.enhanced_train_items[user_id]
                
                # 添加1-2个新物品
                available_items = [i for i in range(self.original_dataset.n_items) 
                                 if i not in current_items]
                
                if available_items:
                    n_new_items = np.random.randint(1, 3)
                    try:
                        new_items = np.random.choice(
                            available_items, 
                            size=min(n_new_items, len(available_items)), 
                            replace=False
                        )
                        
                        self.enhanced_train_items[user_id].extend(new_items.tolist())
                        augmented_users += 1
                    except:
                        # 如果选择失败，跳过
                        pass
                
                pbar.set_postfix({"已增强": augmented_users})
        
        print(f"✅ 用户增强完成，增强了 {augmented_users} 个用户")
        return {'users_augmented': augmented_users}
    
    def get_enhanced_dataset(self):
        """获取增强后的数据集对象"""
        # 创建增强后的数据集对象
        enhanced_dataset = type('EnhancedDataset', (), {
            'train_items': self.enhanced_train_items,
            'test_items': self.enhanced_test_items,
            'n_users': len(self.enhanced_train_items),
            'n_items': self.original_dataset.n_items,
            'n_train': sum(len(items) for items in self.enhanced_train_items.values()),
            'n_test': sum(len(items) for items in self.enhanced_test_items.values()),
            'item_attributes': self.original_dataset.item_attributes,
            'enhancement_stats': self.enhancement_stats
        })()
        
        return enhanced_dataset