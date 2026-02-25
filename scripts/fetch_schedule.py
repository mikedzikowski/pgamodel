#!/usr/bin/env python3
"""Print the PGA Tour schedule for the current or specified season."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pga_oad import DataGolfClient


def main():
    parser = argparse.ArgumentParser(description="Fetch PGA Tour schedule from DataGolf")
    parser.add_argument("--season", type=int, default=None, help="Season year (default: current)")
    parser.add_argument("--upcoming", action="store_true", help="Show upcoming events only")
    args = parser.parse_args()

    client = DataGolfClient()
    data = client.get_schedule(tour="pga", season=args.season, upcoming_only=args.upcoming)

    events = data.get("schedule", [])
    if not events:
        print("No events found.")
        return

    print(f"\n{'#':<4} {'Event':<45} {'Course':<35} {'Date':<12} {'Event ID'}")
    print("-" * 110)
    for i, event in enumerate(events, 1):
        name = event.get("event_name", "")[:44]
        course = event.get("course", "")[:34]
        date = event.get("date", "")
        event_id = event.get("event_id", "")
        print(f"{i:<4} {name:<45} {course:<35} {date:<12} {event_id}")

    print(f"\nTotal events: {len(events)}")


if __name__ == "__main__":
    main()
