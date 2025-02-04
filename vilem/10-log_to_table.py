# %%

import json

data = json.load(open("/home/vilda/Downloads/eval_beryllium.json", "r"))

# %%
def format_cell(y):
    y_color = (y-0.250)/(1.0-0.250)*0.3+0.1
    return f"\\cellcolor{{green!{y_color*100:.0f}}} {y:.2f}"

def format_cell_human(y):
    y_color = (y-0.10)/(0.45-0.10)*0.3+0.1
    return f"\\cellcolor{{purple!{y_color*100:.0f}}} {y:.2f}"


layers = list(range(1, 23+1, 4))+[23]

for layer1 in layers:
    print(f"& \\tiny {layer1:0>2} & {format_cell_human(data[layer1]['corr_human'])}")
    for layer2 in layers:
        print(f'& {format_cell(data[layer1]["corr"][layer2])}', end=" ")
    print("\\\\")