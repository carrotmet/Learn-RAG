"""策略注册中心

统一管理所有索引策略的注册与获取。
"""


class Registry:
    """策略注册中心
    
    使用类变量存储策略类和实例，支持延迟初始化。
    """

    _strategies = {}
    _instances = {}

    @classmethod
    def register(cls, name: str, klass):
        """注册策略类
        
        Args:
            name: 策略名称
            klass: 策略类（继承 IndexStrategy）
        """
        cls._strategies[name] = klass
        # 清除已有实例（如果重新注册）
        if name in cls._instances:
            del cls._instances[name]

    @classmethod
    def get(cls, name: str):
        """获取策略实例（单例模式）
        
        Args:
            name: 策略名称
        
        Returns:
            策略实例
        """
        if name not in cls._instances:
            if name not in cls._strategies:
                raise KeyError(f"未注册的策略: {name}。已注册: {list(cls._strategies.keys())}")
            cls._instances[name] = cls._strategies[name]()
        return cls._instances[name]

    @classmethod
    def list_strategies(cls) -> list:
        """列出所有已注册的策略名称"""
        return list(cls._strategies.keys())

    @classmethod
    def reset(cls):
        """重置所有实例（测试用）"""
        cls._instances.clear()


# ── 自动注册默认策略 ──────────────────────────────────────

def _auto_register():
    """自动注册所有内置策略"""
    from hybrid.strategies.standard import StandardStrategy
    from hybrid.strategies.summary import SummaryStrategy
    from hybrid.strategies.parent_child import ParentChildStrategy
    from hybrid.strategies.hypothetical import HypotheticalStrategy

    Registry.register("standard", StandardStrategy)
    Registry.register("summary", SummaryStrategy)
    Registry.register("parent_child", ParentChildStrategy)
    Registry.register("hypothetical", HypotheticalStrategy)


# 模块导入时自动注册
_auto_register()
