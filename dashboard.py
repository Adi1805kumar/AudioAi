"""
dashboard.py — Rich Terminal Dashboard
────────────────────────────────────────
Run alongside main.py to see live status in a pretty terminal UI.
Polls the /status and /health endpoints every second.

Usage:
    python dashboard.py
"""

import time
import httpx
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box
from rich.text import Text

BASE_URL = "http://127.0.0.1:8000"
console = Console()


def fetch(endpoint: str) -> dict:
    try:
        r = httpx.get(f"{BASE_URL}{endpoint}", timeout=2)
        return r.json()
    except Exception:
        return {}


def build_dashboard() -> Panel:
    health = fetch("/health")
    status = fetch("/status")

    if not health:
        return Panel(
            Text("⚠️  Cannot reach assistant (is main.py running?)", style="bold red"),
            title="[bold]Hands-Free AI Desktop Assistant[/bold]",
            border_style="red",
        )

    # ── Threads table ──────────────────────────────────────────────────────
    threads = health.get("threads", {})
    t_table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    t_table.add_column("Thread")
    t_table.add_column("Status")
    for name, alive in threads.items():
        icon = "🟢 alive" if alive else "🔴 dead"
        t_table.add_row(name.replace("_", " ").title(), icon)

    # ── Queues table ───────────────────────────────────────────────────────
    queues = health.get("queues", {})
    q_table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    q_table.add_column("Queue")
    q_table.add_column("Size")
    for name, size in queues.items():
        bar = "█" * min(size, 20)
        color = "green" if size < 20 else "yellow" if size < 50 else "red"
        q_table.add_row(name.replace("_", " ").title(), f"[{color}]{size:3d} {bar}[/{color}]")

    # ── State ──────────────────────────────────────────────────────────────
    listening = "🎤  Listening" if status.get("listening") else "⏸  Not listening"
    connected = "🌐  Connected to Gemini" if status.get("connected_to_gemini") else "🔴  Disconnected"
    speaking = "🔊  Speaking" if status.get("speaking") else "   Silent"
    interrupted = "⚡  Barge-in active!" if status.get("interrupted") else ""

    state_text = Text()
    state_text.append(f"{listening}\n", style="bold green" if status.get("listening") else "dim")
    state_text.append(f"{connected}\n", style="bold blue" if status.get("connected_to_gemini") else "dim")
    state_text.append(f"{speaking}\n", style="bold yellow" if status.get("speaking") else "dim")
    if interrupted:
        state_text.append(f"{interrupted}\n", style="bold red blink")

    content = Columns([t_table, q_table, state_text], equal=False, expand=True)
    return Panel(
        content,
        title="[bold white]🤖 Hands-Free AI Desktop Assistant[/bold white]",
        subtitle="[dim]Press Ctrl+C to exit dashboard[/dim]",
        border_style="bright_blue",
    )


def main():
    console.print("\n[bold cyan]Starting dashboard…[/bold cyan]\n")
    with Live(build_dashboard(), refresh_per_second=1, console=console) as live:
        while True:
            time.sleep(1)
            live.update(build_dashboard())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Dashboard closed.[/dim]")