"""Flower ServerApp adapter.

All experiment logic lives outside this adapter.
"""

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from fl.server_runtime import run_server


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run the configured federated experiment."""
    run_server(grid, context)

