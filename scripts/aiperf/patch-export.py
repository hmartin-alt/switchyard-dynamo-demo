"""Version-guarded AIPerf 0.13.0 export fix used for the high-rate study.

This changes only artifact export: no request generation, cache busting, or
metric calculations. Run once in the dedicated load-generator environment.
"""

import importlib.metadata
from pathlib import Path


if importlib.metadata.version("aiperf") != "0.13.0":
    raise SystemExit("This export patch supports only AIPerf 0.13.0")

import aiperf.dataset.dataset_manager as dataset_module  # noqa: E402
import aiperf.post_processors.raw_record_writer_processor as raw_module  # noqa: E402


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    if new in source:
        return
    if source.count(old) != 1:
        raise RuntimeError(f"Unexpected AIPerf source in {path}; refusing to patch")
    path.write_text(source.replace(old, new))


replace_once(
    Path(dataset_module.__file__),
    "await self._generate_inputs_json_file()",
    'self.info("Skipping duplicate inputs.json export; raw request records remain enabled")',
)
replace_once(
    Path(raw_module.__file__),
    '''        record_count = 0
        async with aiofiles.open(self.output_file, "w") as export_file:
''',
    '''        # Move the completed single-writer JSONL file rather than copying
        # every record on the event loop during large exports.
        if len(raw_record_files) == 1:
            raw_record_files[0].replace(self.output_file)
            with contextlib.suppress(OSError):
                self.output_dir.rmdir()
            self.info(f"Moved single raw record file to {self.output_file}")
            return

        record_count = 0
        async with aiofiles.open(self.output_file, "w") as export_file:
''',
)
print("AIPerf 0.13.0 export patch applied")
