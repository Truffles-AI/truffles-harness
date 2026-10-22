"""Loopback-only demo launcher; use your bank gateway in bank mode."""

import os


def main():
    # Disable inherited tracing to third parties for the reference PFM process.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    import uvicorn

    uvicorn.run(
        "truffles_pfm.api:create_app",
        factory=True,
        host="127.0.0.1",
        port=int(os.getenv("PFM_PORT", "8765")),
        access_log=False,
    )


if __name__ == "__main__":
    main()
