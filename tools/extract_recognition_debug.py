from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


def decode_data_url(value: str) -> bytes:
    _, encoded = value.split(',', 1)
    return base64.b64decode(encoded)


def decode_base64(value: str) -> bytes:
    return base64.b64decode(value)


def file_record(path: Path, data: bytes) -> dict[str, Any]:
    path.write_bytes(data)
    return {
        'file': path.name,
        'bytes': len(data),
        'sha256': hashlib.sha256(data).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('capture_json', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()

    source = args.capture_json.resolve()
    output_dir = (args.output_dir or source.with_suffix('')).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(source.read_text(encoding='utf-8'))
    summary = copy.deepcopy(payload)
    capture = payload['capture']
    summary_capture = summary['capture']

    image_files = {
        'sourcePngDataUrl': 'source.png',
        'compositePngDataUrl': 'composite.png',
    }
    for key, filename in image_files.items():
        data = decode_data_url(capture['images'][key])
        summary_capture['images'][key] = file_record(output_dir / filename, data)

    region_files = {
        'completed-hand': 'completed-hand.png',
        'dora-indicators': 'dora-indicators.png',
        'melds': 'melds.png',
    }
    for region, filename in region_files.items():
        data = decode_data_url(capture['images']['regionPngDataUrls'][region])
        summary_capture['images']['regionPngDataUrls'][region] = file_record(
            output_dir / filename,
            data,
        )

    detector_input = decode_base64(capture['detectorInput']['data'])
    input_record = file_record(output_dir / 'detector-input.f32', detector_input)
    input_record['float32Count'] = len(detector_input) // 4
    summary_capture['detectorInput']['data'] = input_record

    detector_output = decode_base64(capture['detectorOutput']['data'])
    output_record = file_record(output_dir / 'detector-output.f32', detector_output)
    output_record['float32Count'] = len(detector_output) // 4
    summary_capture['detectorOutput']['data'] = output_record

    summary_path = output_dir / 'summary.json'
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    print(output_dir)
    print(summary_path)


if __name__ == '__main__':
    main()
