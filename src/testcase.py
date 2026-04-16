from multiprocessing import Pool
import time
import copy
import sys
from pathlib import Path
from PIL import Image

from get_binary_images import get_binary_dataset
from process_images import process_image_binary


def process_item(item, root, radius, threshold):
    #  копія щоб не було конфліктів
    local_item = copy.deepcopy(item)
    local_item.status = "preprocess_binary"

    start = time.perf_counter()

    process_image_binary(
        local_item,
        root,
        radius=radius,
        threshold=threshold,
        is_debug=False
    )

    elapsed = time.perf_counter() - start

    return (
        local_item.pixels,
        local_item.relative_path,
        radius,
        threshold,
        elapsed
    )


def main():
    root = Path(__file__).resolve().parent.parent
    input_dir = root / "input_images"
    output_dir = root / "output_images"

    dataset = get_binary_dataset(input_dir)

    n = int(input("Enter number of processes: "))

    #  тест-кейси
    testcases = [
        (1, 60),  (1, 100),
        (2, 110),
    ]

    #  формуємо задачі
    args_list = [
        (item, root, r, t)
        for item in dataset.items
        for (r, t) in testcases
    ]

    print(f"Total tasks: {len(args_list)}")

    with Pool(n) as pool:
        for i, (pixels, relative_path, r, t, elapsed) in enumerate(
            pool.starmap(process_item, args_list)
        ):
            #  унікальна назва
            output_name = f"{relative_path.stem}_R{r}_T{t}{relative_path.suffix}"
            output_path = output_dir / output_name

            output_path.parent.mkdir(parents=True, exist_ok=True)

            Image.fromarray(pixels).save(output_path)

            print(f"[{i+1}/{len(args_list)}] {relative_path} | R={r} T={t} | ⏱ {elapsed:.3f}s")


if __name__ == "__main__":
    main()