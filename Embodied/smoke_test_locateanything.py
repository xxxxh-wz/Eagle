from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from locateanything_worker import LocateAnythingWorker


ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "test_assets" / "locateanything_kite.png"
RESULT_PATH = ROOT / "test_assets" / "locateanything_kite_result.json"
PREVIEW_PATH = ROOT / "test_assets" / "locateanything_kite_preview.png"


def main() -> None:
    image = Image.open(IMAGE_PATH).convert("RGB")
    worker = LocateAnythingWorker("nvidia/LocateAnything-3B")

    queries = [
        ("ground_single", "the colorful kite", worker.ground_single),
        ("ground_multi", "power poles", worker.ground_multi),
    ]

    outputs = []
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    colors = ["red", "yellow"]
    for index, (task, query, method) in enumerate(queries):
        result = method(
            image,
            query,
            max_new_tokens=256,
            temperature=0.1,
            verbose=True,
        )
        answer = result["answer"]
        boxes = LocateAnythingWorker.parse_boxes(answer, *image.size)
        for box in boxes:
            xy = [box["x1"], box["y1"], box["x2"], box["y2"]]
            draw.rectangle(xy, outline=colors[index], width=5)
        outputs.append({
            "task": task,
            "query": query,
            "answer": answer,
            "boxes": boxes,
            "stats": result.get("stats"),
        })

    payload = {
        "image": str(IMAGE_PATH),
        "queries": outputs,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    preview.save(PREVIEW_PATH)

    print(json.dumps(payload, indent=2))
    if not outputs[0]["boxes"]:
        raise SystemExit("No bounding boxes were parsed from the model output.")


if __name__ == "__main__":
    main()
