from tqdm import tqdm
import h5py
import json

def get_data():
    data_out = []
    with h5py.File("../data/candidates/beam_dev.h5") as candidates_file:
        candidates_text_h5ds = candidates_file["text"]
        data_lines = open("../data/jsonl/dev.jsonl").readlines()

        assert candidates_text_h5ds.shape[0] == len(data_lines)

        for i, data_line in enumerate(tqdm(data_lines)):

            data = json.loads(data_line)
            src = data["src"]
            tgts = [candidates_text_h5ds[i, j].decode() for j in range(candidates_text_h5ds.shape[1])]
            # Skip missing candidates (because some OOM on test set)
            if all(not tgt for tgt in tgts):
                continue
            
            data_out.append({
                "src": src,
                "tgt": tgts,
                "langs": data["langs"],
                "score": data["score"],
            })
    
    return data_out