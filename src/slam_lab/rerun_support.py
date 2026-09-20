"""Start the viewer installed alongside the current Python environment."""

import shutil
import sysconfig


def init_recording(application_id, *, output):
    import rerun as rr

    rr.init(application_id, strict=True)
    if output is None:
        # Invoking .venv/bin/slam-lab directly does not add .venv/bin to PATH.
        # Prefer this environment's matching viewer version, then use Rerun's PATH lookup.
        executable = shutil.which("rerun", path=sysconfig.get_path("scripts"))
        rr.spawn(executable_path=executable)
    else:
        rr.save(str(output))
    recording = rr.get_global_data_recording()
    assert recording is not None
    return recording
