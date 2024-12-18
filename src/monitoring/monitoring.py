from prometheus_fastapi_instrumentator import Instrumentator, metrics
from fastapi import FastAPI


def setup_monitoring(app: FastAPI) -> None:
    """
    Set up Prometheus monitoring for the FastAPI app.

    Parameters:
    - app (FastAPI): FastAPI application instance.

    Returns:
    - None
    """
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_instrument_requests_inprogress=True
    )

    # Add default and custom metrics
    instrumentator.add(metrics.default())
    instrumentator.add(metrics.request_size())
    instrumentator.add(metrics.response_size())
    instrumentator.add(metrics.latency())
    instrumentator.add(metrics.requests())

    # Expose metrics at /metrics
    instrumentator.instrument(app).expose(app, endpoint="/metrics")
    print("Monitoring setup complete. Metrics available at /metrics.")
