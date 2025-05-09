from typing import Dict, Type, Any, Callable

class Registry:
    def __init__(self):
        self._registry: Dict[str, Dict[str, Type]] = {}
    
    def register(self, category: str, name: str) -> Callable:
        def decorator(cls: Type):
            if category not in self._registry:
                self._registry[category] = {}
            self._registry[category][name] = cls
            return cls
        return decorator
    
    def get(self, category: str, name: str) -> Type:
        if category not in self._registry or name not in self._registry[category]:
            raise ValueError(f"No registered class under category '{category}' with name '{name}'")
        return self._registry[category][name]

    def get_instance(self, category: str, name: str, *args, **kwargs) -> Any:
        cls = self.get(category, name)
        return cls(*args, **kwargs)

GLOBAL_REGISTRY = Registry()