"""Run the optional MoviePy downloader in its own working directory/process."""

import json
import sys
from pathlib import Path

from process_control import configure_progress_lock


def main():
    configure_progress_lock()
    from mtslinker.webinar import fetch_webinar_data

    request = json.load(sys.stdin)
    result = fetch_webinar_data(event_sessions=request["event_session"],
                                record_id=request["record_id"] or "", session_id=request["session_id"])
    if not result:
        raise RuntimeError("mtslinker не создал запись")
    # mtslinker writes <recording-name>/<recording-name>.mp4. Downloaded
    # participant files can be larger than the final render and must not win.
    renders = [path.resolve() for path in Path.cwd().glob("*/*.mp4")
               if path.stem == path.parent.name and path.stat().st_size > 0]
    if len(renders) != 1:
        raise RuntimeError("Не удалось однозначно найти итоговый файл mtslinker")
    Path(request["manifest"]).write_text(json.dumps(str(renders[0])), encoding="utf-8")


if __name__ == "__main__":
    main()
