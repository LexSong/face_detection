import click
import json
from pathlib import Path
from tqdm import tqdm

from SFD_pytorch import FaceDetector
from utils import VideoFile
from utils import iter_by_step

configs = {
    'detector_model': "models/S3FD/s3fd_convert.pth",
    'detector_resize': 480,
}


@click.command()
@click.argument(
    'input_file',
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
)
@click.argument('output_file', type=click.Path(), nargs=-1)
@click.option('-n', '--every-n-frame', type=int, default=1)
def main(input_file, output_file, every_n_frame):
    input_file = Path(input_file)
    assert input_file.is_file()

    if output_file:
        output_file = Path(output_file[0])
    else:
        output_file = input_file.parent / "detections.json"

    if output_file.exists():
        raise FileExistsError(f"Output file {output_file} already exists.")

    detector = FaceDetector(configs['detector_model'])

    video = VideoFile(input_file)
    iter_frames = iter_by_step(enumerate(video), every_n_frame)
    progress_bar = tqdm(
        iter_frames, total=len(video) // every_n_frame, ascii=True)

    detections = dict()

    for index, frame in progress_bar:
        boxes = detector.detect(frame, resize_width=configs['detector_resize'])
        boxes = {x: boxes[x].tolist() for x in boxes.dtype.names}
        detections[index] = boxes

    with output_file.open('w') as f:
        json.dump(detections, f)


if __name__ == "__main__":
    main()
