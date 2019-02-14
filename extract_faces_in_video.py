import click
import cv2
import json
from pathlib import Path
from tqdm import tqdm

from utils import VideoFile
from utils import iter_by_step
from utils import crop_and_resize_face

configs = {
    'detector_model': "models/S3FD/s3fd_convert.pth",
    'detector_resize': 480,
}


@click.command()
@click.argument(
    'video_file',
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
)
@click.argument(
    'detections_file',
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
)
@click.argument('output_dir', type=click.Path(), nargs=-1)
@click.option('-n', '--every-n-frame', type=int, default=1)
def main(video_file, detections_file, output_dir, every_n_frame):
    video = VideoFile(video_file)

    detections_file = Path(detections_file)
    with detections_file.open() as f:
        detections = json.load(f)

    if output_dir:
        output_dir = Path(output_dir[0])
    else:
        output_dir = Path(video_file).parent / "faces"

    if output_dir.exists():
        raise FileExistsError(f"Output dir {output_dir} already exists.")

    output_dir.mkdir(parents=True)

    iter_frames = iter_by_step(enumerate(video), every_n_frame)
    progress_bar = tqdm(
        iter_frames, total=len(video) // every_n_frame, ascii=True)

    for index, frame in progress_bar:
        boxes = detections.get(str(index))
        if boxes is None:
            continue

        for i, box in enumerate(boxes['box']):
            face = crop_and_resize_face(frame, box)
            output_file = output_dir / f"{index}_{i}.jpg"
            cv2.imwrite(str(output_file), face)


if __name__ == "__main__":
    main()
