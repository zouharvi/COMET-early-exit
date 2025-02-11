# %%

import json

def print_table(fname):
    data = json.load(open(fname, "r"))
    fout = open(fname.replace(".json", ".tex"), "w")

    def format_cell(y):
        y_color = (y+-0.10)/(1.0-0.10)*0.3+0.1
        y_color = max(0, y_color)
        return f"\\cellcolor{{green!{y_color*100:.0f}}} {y:.2f}"

    def format_cell_human(y):
        y_color = (y-0.10)/(0.45-0.10)*0.3+0.1
        y_color = max(0, y_color)
        return f"\\cellcolor{{purple!{y_color*100:.0f}}} {y:.3f}"


    layers1 = list(range(1, 23+1, 4))+[23]
    layers2 = list(range(1, 23+1, 4))+[23]
    # layers = list(range(1, 23+1))

    print(
        "", "", "",
        *[f"\\tiny {layer2+1:0>2}" for layer2 in layers2],
        sep=" & ",
        end="\\\\\n",
        file=fout,
    )
    for layer1 in layers1:
        print(
            f"& \\tiny {layer1+1:0>2}",
            file=fout,
        )
        for layer2 in layers2:
            print(f'& {format_cell(data[layer1]["corr"][layer2])}', end=" ", file=fout)

        print(
            f"& {format_cell_human(data[layer1]['corr_human'])}",
            file=fout,
        )
        print("\\\\", file=fout)

    fout.close()

    print(data[-1]["corr_human"])

print_table("../computed/10-eval_nitrogen.json")
print_table("../computed/10-eval_beryllium.json")
print_table("../computed/10-eval_helium2hydrogen.json")
# print_table("../computed/10-eval_hydrogen.json")
# print_table("../computed/eval_nitrogen.json")

"""
scp euler:/cluster/work/sachan/vilem/COMET-early-exit/logs/eval_nitrogen.out computed/10-eval_nitrogen.json
"""