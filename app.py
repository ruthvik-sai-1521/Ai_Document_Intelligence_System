# Entrypoint wrapper for Streamlit Community Cloud
import runpy
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
runpy.run_path(str(ui_app_path), run_name="__main__")
