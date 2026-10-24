from models.cnn_femnist import FEMNIST_CNN, get_femnist_model
from models.cnn_cifar10 import CIFAR10_VGG, get_cifar10_model
from models.lstm_shakespeare import ShakespeareLSTM, get_shakespeare_model

MODEL_REGISTRY = {
    "cnn_femnist": get_femnist_model,
    "cnn_cifar10": get_cifar10_model,
    "lstm_shakespeare": get_shakespeare_model,
}


def get_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


def build_model(model_cfg: dict):

    import inspect

    name = model_cfg.get("name", "cnn_femnist")
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    factory = MODEL_REGISTRY[name]
    accepted = set(inspect.signature(factory).parameters.keys())
    kwargs = {
        k: v for k, v in model_cfg.items()
        if k != "name" and k in accepted
    }
    return factory(**kwargs)
