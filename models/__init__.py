"""Model factory."""

from models.dcnn_bilstm import SUPPORTED_NORMS, DCNNBiLSTM

SUPPORTED_MODELS = ("dcnn_bilstm",)


def build_model(model_config, *, input_dim: int, num_classes: int):
    """Instantiate the configured model."""
    if model_config.name == "dcnn_bilstm":
        return DCNNBiLSTM(
            input_dim=input_dim,
            num_classes=num_classes,
            conv_channels=model_config.conv_channels,
            kernel_size=model_config.kernel_size,
            lstm_hidden_size=model_config.lstm_hidden_size,
            lstm_layers=model_config.lstm_layers,
            dropout=model_config.dropout,
            norm=getattr(model_config, "norm", "batch"),
        )
    raise ValueError(f"model.name must be one of {SUPPORTED_MODELS}; got {model_config.name!r}")


__all__ = [
    "DCNNBiLSTM",
    "SUPPORTED_MODELS",
    "SUPPORTED_NORMS",
    "build_model",
]
