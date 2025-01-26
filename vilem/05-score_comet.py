import comet
import csv
import scipy.stats
import numpy as np

model = comet.load_from_checkpoint("lightning_logs/version_19245790/checkpoints/epoch=3-step=46912-val_avg_pearson=0.306.ckpt")
data = list(csv.DictReader(open("data/csv/dev_da.csv", "r")))
pred_y = np.array(model.predict(data, batch_size=32)["scores"]).T

data_y = [float(x["score"]) for x in data]
for layer_i, layer_y in list(enumerate(pred_y))[1:]:
    corr_gold = scipy.stats.pearsonr(data_y, layer_y).correlation
    print(
        f"Layer {layer_i:0>2}:",
        f"h={corr_gold:<5.0%}",
        end=" "
    )
    for b_layer_i, b_layer_y in list(enumerate(pred_y))[1::3]:
        corr = scipy.stats.pearsonr(b_layer_y, layer_y).correlation
        print(f"l{b_layer_i:0>2}={corr:<5.0%}", end=" ")
    print()