"""Run the package inspector after installing with `python -m pip install -e .`."""

from ev_battery.inspection import main

if __name__ == "__main__":
    raise SystemExit(main())
