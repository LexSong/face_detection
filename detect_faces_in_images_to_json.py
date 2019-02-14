import click
import cv2
import json
from pathlib import Path
from tqdm import tqdm

from SFD_pytorch import FaceDetector

configs = {
    'detector_model': "models/S3FD/s3fd_convert.pth",
    'detector_resize': 480,
}


@click.command()
@click.argument(
    'input_dir',
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
)
@click.argument('output_file', type=click.Path(), nargs=-1)
def main(input_dir, output_file):
    input_dir = Path(input_dir)
    assert input_dir.is_dir()

    if output_file:
        output_file = Path(output_file[0])
    else:
        output_file = input_dir / "detections.json"

    if output_file.exists():
        raise FileExistsError(f"Output file {output_file} already exists.")

    detector = FaceDetector(configs['detector_model'])

    input_files = list(input_dir.glob('*.jpg'))

    detections = dict()

    for filename in tqdm(input_files, ascii=True):
        filename = str(filename)
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        boxes = detector.detect(image, resize_width=configs['detector_resize'])
        boxes = {x: boxes[x].tolist() for x in boxes.dtype.names}
        detections[filename] = boxes

    with output_file.open('w') as f:
        json.dump(detections, f)


if __name__ == "__main__":
    main()
