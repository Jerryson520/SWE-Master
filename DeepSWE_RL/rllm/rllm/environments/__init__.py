from importlib import import_module

__all__ = ["BaseEnv", "SingleTurnEnvironment", "ToolEnvironment"]

_LAZY_IMPORTS = {
    "BaseEnv": ("rllm.environments.base.base_env", "BaseEnv"),
    "SingleTurnEnvironment": ("rllm.environments.base.single_turn_env", "SingleTurnEnvironment"),
    "ToolEnvironment": ("rllm.environments.tools.tool_env", "ToolEnvironment"),
}


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    module_path, class_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_path), class_name)
    globals()[name] = value
    return value


def safe_import(module_path, class_name):
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError:
        return None


ENVIRONMENT_IMPORTS = [
    ("rllm.environments.browsergym.browsergym", "BrowserGymEnv"),
    ("rllm.environments.frozenlake.frozenlake", "FrozenLakeEnv"),
    # ("rllm.environments.swe.swe", "SWEEnv"),
    ("rllm.environments.code.competition_coding", "CompetitionCodingEnv"),
]

for module_path, class_name in ENVIRONMENT_IMPORTS:
    imported_class = safe_import(module_path, class_name)
    if imported_class is not None:
        globals()[class_name] = imported_class
        __all__.append(class_name)
