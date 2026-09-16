"""Flower ClientApp adapter.

This module intentionally delegates all data, model, and training work to
project modules so the Flower adapter stays thin.
"""

from flwr.app import Context, Message
from flwr.clientapp import ClientApp

from fl.client_runtime import evaluate_client, train_client


app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Handle a Flower train message."""
    return train_client(msg, context)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Handle a Flower evaluate message."""
    return evaluate_client(msg, context)

