import numpy as np
from scipy.ndimage import rotate
import os
import imageio
import argparse


def augment(sample, do_flip=False, do_rotate=False, angle=0):
    if do_rotate:
        sample = rotate(sample, angle, axes=(0, 1), reshape=False)
    if do_flip:
        sample = np.ascontiguousarray(sample[::-1, ::-1])
    return sample


def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith('.bmp'):
            continue
        in_path = os.path.join(input_dir, fname)
        imgs = imageio.imread(in_path)
        imgs = np.array(imgs)

        for angle in [180]:  # Altere para os ângulos que desejar (ex: [90, 180, 270])
            sample = augment(imgs, do_rotate=True, do_flip=False, angle=angle)

            # 2D image
            if sample.ndim == 2:
                out_name = f"{angle}rot_{fname}"
                out_path = os.path.join(output_dir, out_name)
                imageio.imsave(out_path, sample.astype(np.uint8))

            # 3D image stack
            elif sample.ndim == 3:
                for slice_idx in range(sample.shape[0]):
                    slice_img = sample[slice_idx]
                    out_name = f"{angle}rot_{slice_idx}_{fname}"
                    out_path = os.path.join(output_dir, out_name)
                    imageio.imsave(out_path, slice_img.astype(np.uint8))


def main(args):
    data_path = args.data_path
    output_path = args.output_path

    subdirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    print(f"Subpastas encontradas em '{data_path}': {subdirs}")

    for class_name in subdirs:
        print(f"Processando pasta: {class_name}")
        class_input = os.path.join(data_path, class_name)
        class_output = os.path.join(output_path, class_name)
        process_directory(class_input, class_output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Gera imagens rotacionadas preservando estrutura de pastas.')
    parser.add_argument('--data-path', type=str,
                        default='/home/joyzinhw/Documentos/DMLN/BMP_classification/train',
                        help='Caminho da pasta de entrada com subpastas de classes')
    parser.add_argument('--output-path', type=str,
                        default='augment_img_rotate',
                        help='Caminho da pasta de saída onde serão criadas subpastas')
    opt = parser.parse_args()
    main(opt)
