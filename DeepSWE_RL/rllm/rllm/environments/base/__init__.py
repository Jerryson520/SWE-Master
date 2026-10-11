from importlib import import_module

__all__ = ["BaseEnv", "SingleTurnEnvironment", "MultiTurnEnvironment"]

_LAZY_IMPORTS = {
    "BaseEnv": ("rllm.environments.base.base_env", "BaseEnv"),
    "SingleTurnEnvironment": ("rllm.environments.base.single_turn_env", "SingleTurnEnvironment"),
    "MultiTurnEnvironment": ("rllm.environments.base.multi_turn_env", "MultiTurnEnvironment"),
}


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    module_path, class_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_path), class_name)
    globals()[name] = value
    return value
