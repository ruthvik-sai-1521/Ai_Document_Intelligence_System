# Entrypoint wrapper for Streamlit Community Cloud and direct python app.py execution
import sys
from pathlib import Path

root = Path(__file__).parent.resolve()
src_dir = root / "src"
ui_dir = root / "ui"

if str(root) not in sys.path:
    sys.path.insert(0, str(root))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

ui_app_path = ui_dir / "app.py"

# If invoked directly via `python app.py`, automatically boot the full Streamlit web server
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    ctx = get_script_run_ctx()
except Exception:
    ctx = None

if ctx is None and __name__ == "__main__":
    import streamlit.web.cli as stcli
    sys.argv = ["streamlit", "run", str(ui_app_path), *sys.argv[1:]]
    sys.exit(stcli.main())
else:
    import runpy
    runpy.run_path(str(ui_app_path), run_name="__main__")
