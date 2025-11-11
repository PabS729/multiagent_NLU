import datasets
import json


def load_data(args, split):
    if args.custom_data != '':
        loaded_data = []
        limit = 1600
        ct = 0
        with open(args.custom_data) as f:
            for i, line in enumerate(f):
                ct += 1
                if ct >= limit:
                    line_json = json.loads(line)
                    loaded_data.append(line_json)
                

    else:
        loaded_data = datasets.load_dataset(name=args.dataset_name, split=split)
    return loaded_data

def filter_train(input_map, output_map, args):
    loaded_data = load_data(args, "train")
    new_dat = []
    for r in loaded_data:
        dic = {}
        dic["inputs"] = [loaded_data[k] for k in input_map]
        dic["output"] = loaded_data[output_map]
        new_dat.append(dic)
    return new_dat