import click
import cv2
from pathlib import Path
from tqdm import tqdm

from face_project.face_detector import FaceDetector
from face_project.utils import crop_and_resize_face

configs = {
    'detector_model': "models/S3FD/s3fd_convert.pth",
    'detector_resize': 480,
}


@click.command()
@click.argument(
    'input_dir',
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
)
@click.argument('output_dir', type=click.Path(), nargs=-1)
def main(input_dir, output_dir):
    input_dir = Path(input_dir)
    assert input_dir.is_dir()

    if output_dir:
        output_dir = Path(output_dir[0])
    else:
        output_dir = input_dir / "faces"

    if output_dir.exists():
        raise FileExistsError(f"Output dir {output_dir} already exists.")

    detector = FaceDetector(configs['detector_model'])

    input_files = list(input_dir.glob('*.jpg'))
    output_dir.mkdir(parents=True)

    for filename in tqdm(input_files, ascii=True):
        image = cv2.imread(str(filename), cv2.IMREAD_COLOR)
        boxes = detector.detect(image, resize_width=configs['detector_resize'])

        for i, box in enumerate(boxes['box']):
            face = crop_and_resize_face(image, box)
            output_file = output_dir / f"{filename.stem}_{i}.jpg"
            cv2.imwrite(str(output_file), face)


if __name__ == "__main__":
    main()
