import argparse
import multiprocessing
import os

import numpy as np
from tqdm import tqdm


def reflect_padding(image):
    for i in range(len(image)):
        if not np.all(image[i] == 9999):
            reflected_row = reflect(image[i])
            image[i] = reflected_row
    for i in range(len(image[0])):
        col = image[:, i]
        if not np.all(col == 9999):
            reflected_col = reflect(col)
            image[:, i] = reflected_col
    return image


def padding_idx(lst, pad=9999):
    no_pad_indices = np.where(lst != pad)[0]
    if no_pad_indices.size > 0:
        first_idx = no_pad_indices[0]
        last_idx = no_pad_indices[-1] + 1
        return first_idx, last_idx
    else:
        return None


def reflect(tableau):
    idx = padding_idx(tableau)
    true_image = tableau[idx[0] : idx[1]]
    n = len(true_image)
    inv_true_image = true_image[::-1]
    size = len(tableau)
    tail = np.tile(inv_true_image, ((((size - idx[1]) // n) + 1), 1))[: size - idx[1]]
    head = np.tile(inv_true_image, (0 if idx[0] == 0 else (idx[0] // n) + 1, 1))[
        -idx[0] :
    ]
    new_array = np.concatenate((head, true_image, tail))
    return new_array


def pad_image_with_mask(image, mask, pad_value=9999):
    assert (
        image.shape == mask.shape
    ), "Les dimensions de l'image et du masque ne correspondent pas."
    masked_pixels = np.all(mask == 1, axis=0)
    padded_image = image.copy()
    padded_image[:, masked_pixels] = pad_value
    return padded_image


def process_chunk(input_folder, output_folder, pad, chunk):
    processed_files = 0
    for file_name in tqdm(
        chunk, desc="Traitement des fichiers", position=0, leave=True
    ):
        if file_name.startswith("_sample") and file_name.endswith(".npy"):
            output_file_name = file_name
            output_path = os.path.join(output_folder, output_file_name)
            if not os.path.exists(output_path):
                image = np.load(os.path.join(input_folder, file_name))
                image = image.transpose((1, 2, 0))
                image_padded = reflect_padding(image)
                image_padded = image_padded.transpose((2, 0, 1))
                np.save(output_path, image_padded)
            processed_files += 1
    return processed_files


def main(input_folder, output_folder, pad):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_list = os.listdir(input_folder)
    num_processes = multiprocessing.cpu_count()
    chunk_size = len(file_list) // num_processes
    pool = multiprocessing.Pool(processes=num_processes)
    chunks = [
        file_list[i : i + chunk_size] for i in range(0, len(file_list), chunk_size)
    ]
    results = [
        pool.apply_async(process_chunk, (input_folder, output_folder, pad, chunk))
        for chunk in chunks
    ]
    pool.close()
    pool.join()
    processed_files = sum(result.get() for result in results)
    if processed_files > 0:
        image = np.load(os.path.join(input_folder, file_list[0]))
        mask = (image != pad).astype(np.float64)
        if mask is not None:
            np.save(os.path.join(output_folder, "mask.npy"), mask)
    print(f"Traitement terminé pour {processed_files} fichiers.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Applique reflect_padding aux fichiers _sampleXXXX.npy"
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Dossier d'entrée contenant les fichiers _sampleXXXX.npy",
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Dossier de sortie où enregistrer les images traitées",
    )
    parser.add_argument("--pad", type=int, default=9999, help="Valeur de rembourrage")
    args = parser.parse_args()

    main(args.input_folder, args.output_folder, args.pad)
