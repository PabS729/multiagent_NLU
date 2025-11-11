from evaluate import load
import argparse
import datasets
import json
import re
from collections import defaultdict


def evaluate_preds(dataset_name, pred, label):
    dataset = dataset_name.split("/")[1]
    subset = dataset_name.split("/")[-1]
    print("evaluating " + subset)
    try:
        metric = load(dataset, subset)
    except Exception as e:
        metric = load("super_glue", dataset.split("/")[-1])

    results = metric.compute(predictions=pred, references=label)

    return results

def load_preds_and_labels(dataset_meta: dict):
    """Load dataset, run inference, write preds."""
    name = dataset_meta["dataset_name"]
    label_tag = dataset_meta["output"]
    input_keys = dataset_meta["input"]
    pred_map = dataset_meta["output_maps"]
    if isinstance(input_keys, str):
        input_keys = [input_keys]

    # load HF dataset
    splits = dataset_meta["split"]
    custom_data = False
    if isinstance(splits, str):
        splits = [splits]
    try:
        raw_ds = datasets.load_dataset("/".join(name.split("/")[:-1]), name.split("/")[-1], split=splits[-1])   # use first split
        labels = raw_ds[label_tag]
    except Exception as e:
        custom_data = True
        labels = []
        with open("data/" + name.split("/")[-1] + "/" + name.split("/")[-1] + "_" + splits[-1] + ".jsonl", "r") as f:
            for line in f:
                l = json.loads(line)
                if name == "aps/super_glue/multirc":
                    ds = {}
                    for k in input_keys:
                        ds[k] = l[k]
                    ns = {}
                    ns["idx"] = ds
                    ns['prediction'] = r
                    labels.append(ns)
                else:
                    labels.append(l[label_tag])
    
    print(labels[0])

    #load predictions
    file = "outputs/" + name.split("/")[-1] + "_base" + ".jsonl"
    ct = 0
    with open(file) as pf:
        dic = defaultdict(list)
        preds = []
        for line in pf:
            l = json.loads(line)
            pred_crude = l["output"]
            res = re.search(r'{\"answer\": \"(.*?)\"}', pred_crude, re.I|re.S).group(1)
            if name == "aps/super_glue/record":
                r = [res]
                # ds = {k:v for k,v in zip(input_keys, l["input"])} 
                # ns = {}
                # ns["idx"] = ds
                # ns['prediction'] = r
                preds.append(r)
            elif name == "aps/super_glue/multirc":
                try: 
                    r = pred_map[res]
                except Exception as e:
                    r = 54
                ds = {k:v for k,v in zip(input_keys, l["input"])} 
                ns = {}
                ns["idx"] = ds
                ns['prediction'] = r
                preds.append(int(r))
                ct += 1
                
            else:
                try: 
                    r = pred_map[res]
                    preds.append(int(r))
                except Exception as e:
                    preds.append(54)
        
        # if name == "aps/super_glue/multirc":
        #     for k in dic.keys():
        #         preds.append(dic[k])

    result = evaluate_preds(name, preds, labels)
    print(result)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="HF hub model or local path")
    parser.add_argument("--task_type", default="nil", help="type of nlu task")
    parser.add_argument("--use_icl", default=False, help="using knn icl selection")
    parser.add_argument("--use_guide", default=False, help="using guidelines")
    parser.add_argument("--use_sota_icl", default=False, help="using sota icl--demonstration selection based on topic relavance")
    parser.add_argument("--use_multiagent", default=False, help="using proposed multiagent system")
    parser.add_argument("--temperature", default=0.8, help="temperature for predict")
    parser.add_argument("--top_p", default=0.95, help="temperature for predict")

    with open("dataset_meta.jsonl") as p:
        for l in p:
            meta = json.loads(l)
            load_preds_and_labels(meta)




    
    


        



