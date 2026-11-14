import logging

import uvicorn

from review_agents.api import create_app
from review_agents.config import Settings
from review_agents.credentials import TokenVault
from review_agents.pipeline.run import review_handler
from review_agents.redact import REDACT, install_log_filter, register_settings
from review_agents.worker import not_implemented


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    install_log_filter()  # every log line is masked (issue #15)
    settings = Settings()
    register_settings(REDACT, settings)
    vault = TokenVault()
    # Without a finder model the service still starts, answers /healthz and queues
    # reviews; each review then fails with a message saying what to configure.
    handler = review_handler(settings, vault) if settings.finder_model else not_implemented
    # log_config=None: uvicorn's loggers go through the root handler, and so the filter.
    uvicorn.run(
        create_app(settings, handler, vault),
        host=settings.host,
        port=settings.port,
        log_config=None,
    )


if __name__ == "__main__":
    main()
