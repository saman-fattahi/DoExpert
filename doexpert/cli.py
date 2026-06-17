"""Console entry point: launch the DoExpert Streamlit application.

After ``pip install doexpert`` the application can be started with the
single command::

    doexpert
"""

import os
import sys


def main() -> None:
    app_path = os.path.join(os.path.dirname(__file__), "..", "streamlit_app_doexpert.py")
    app_path = os.path.abspath(app_path)
    if not os.path.exists(app_path):
        # Installed-package layout: app shipped as package data.
        app_path = os.path.join(os.path.dirname(__file__), "app", "streamlit_app_doexpert.py")
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
