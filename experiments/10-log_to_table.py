# %%

import json

def print_table(fname, big=False):
    data = json.load(open(fname, "r"))
    fout = open(fname.replace(".json", ("_big" if big else "") + ".tex"), "w")

    def format_cell(y):
        y_color = (y+-0.10)/(1.0-0.10)*0.3+0.1
        y_color = max(0, y_color)
        return f"\\cellcolor{{green!{y_color*100:.0f}}} {y:.2f}"

    def format_cell_human(y):
        y_color = (y-0.10)/(0.45-0.10)*0.3+0.1
        y_color = max(0, y_color)
        return f"\\cellcolor{{purple!{y_color*100:.0f}}} {y:.3f}"


    if big:
        layers = list(range(1, 23+1, 2))+[24]
    else:
        layers = list(range(1, 23+1, 4))+[24]


    print(
        "", "",
        *[f"\\tiny {layer2:0>2}\\,\\,\\,\\," for layer2 in layers],
        sep=" & ",
        end="\\\\\n",
        file=fout,
    )
    for layer1 in layers:
        print(
            f"& \\tiny {layer1:0>2}",
            file=fout,
        )
        for layer2 in layers:
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
print_table("../computed/10-eval_oxygen.json")

print_table("../computed/10-eval_oxygen.json", big=True)
print_table("../computed/10-eval_nitrogen.json", big=True)
print_table("../computed/10-eval_helium2hydrogen.json", big=True)

# print_table("../computed/10-eval_hydrogen.json")

"""
scp euler:/cluster/work/sachan/vilem/COMET-early-exit/logs/eval_oxygen.out computed/10-eval_oxygen.json
scp euler:/cluster/work/sachan/vilem/COMET-early-exit/logs/eval_nitrogen.out computed/10-eval_nitrogen.json
scp euler:/cluster/work/sachan/vilem/COMET-early-exit/logs/eval_beryllium.out computed/10-eval_beryllium.json
scp euler:/cluster/work/sachan/vilem/COMET-early-exit/logs/eval_helium2hydrogen.out computed/10-eval_helium2hydrogen.json
"""