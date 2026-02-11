"""Main entry point for CLI usage."""

import argparse
import logging
import sys

from src.config import settings


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Rabbithole-Agent: Local Hybrid Researcher",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch Streamlit UI",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Run research query directly (non-interactive)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.streamlit_port,
        help=f"Streamlit port (default: {settings.streamlit_port})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else getattr(logging, settings.log_level)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.ui:
        _launch_ui(args.port)
    elif args.query:
        _run_query(args.query)
    else:
        parser.print_help()
        sys.exit(1)


def _launch_ui(port: int) -> None:
    """Launch Streamlit UI."""
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "src/ui/app.py",
        "--server.port",
        str(port),
    ]

    print(f"Launching Streamlit on port {port}...")
    subprocess.run(cmd)


def _run_query(query: str) -> None:
    """Run a research query in non-interactive mode."""
    from src.agents.graph import run_research

    print(f"Running research query: {query}")
    print("-" * 50)

    result = run_research(query)

    # Print results
    if result.get("final_report"):
        report = result["final_report"]
        print("\n" + "=" * 50)
        print("RESEARCH RESULTS")
        print("=" * 50)
        print(f"\nQuery: {report.get('query', '')}")
        print(f"\nAnswer:\n{report.get('answer', 'No answer generated')}")
        print(f"\nQuality Score: {report.get('quality_score', 0)}/500")
        print(f"Tasks Completed: {report.get('todo_items_completed', 0)}")

        if report.get("findings"):
            print("\nKey Findings:")
            for i, finding in enumerate(report["findings"], 1):
                print(f"  {i}. {finding.get('claim', '')[:100]}...")
    else:
        print("No report generated")
        if result.get("error"):
            print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()
