from tqdm import tqdm
import h5py
import json


# with h5py.File("output/dev/candidates_sample.h5", "r") as src_file:
#     # Open a new file in write mode
#     with h5py.File("output/dev/candidates_text_only.h5", "w") as dest_file:
#         # Copy only the "text" dataset
#         src_file.copy("text", dest_file)

with h5py.File("output/dev/candidates_sample.h5") as candidates_file:
    candidates_text_h5ds = candidates_file["text"]
    data_lines = open("vilem/scripts/data/jsonl/dev.jsonl").readlines()

    assert candidates_text_h5ds.shape[0] == len(data_lines)

    for i, data_line in enumerate(tqdm(data_lines)):

        data = json.loads(data_line)
        src = data["src"]
        tgts = [candidates_text_h5ds[i, j].decode() for j in range(candidates_text_h5ds.shape[1])]
        # Skip missing candidates (because some OOM on test set)
        if all(not tgt for tgt in tgts):
            continue

        breakpoint()

            