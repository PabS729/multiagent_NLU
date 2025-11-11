import json, argparse, logging, os, datasets, ast
from tqdm import tqdm
from collections import defaultdict
from LLM_all import *       # your HF_LLM class
from datetime import datetime





# ---------- generic helpers --------------------------------------------------

def format_prompt(template: str, inputs: list[str], input_map) -> str:
    """Map a list of strings to {text} or {text0},{text1},… inside template."""
    # if len(inputs) == 1:
    #     return template.format(text=inputs[0])
    if len(inputs) == 1:
        inputs = inputs[0]
    mapping = {n : inp for n, inp in zip(input_map, inputs)}
    print(input_map, inputs)
    return template.format(**mapping)


def predict_batch(model, template: str, examples, input_keys: list[str], args) -> list[str]:
    """Return raw string predictions for a list of HF examples."""
    prompts = []
    for ex in zip(*examples):
        ex = list(ex)
        prompts.append(format_prompt(template, ex, input_keys))
    # assume your HF_LLM class has a .generate() that accepts a list of strings
    out = model.predict(prompts, args.temperature, args.top_p)
    return out


# ---------- task-specific runners -------------------------------------------
def run_sentiment_analysis(dataset_meta: dict, args: argparse.Namespace, model: HF_LLM):
    return _run_classification(dataset_meta, args, model)


def run_sentence_completion(dataset_meta: dict, args: argparse.Namespace, model: HF_LLM):
    return _run_classification(dataset_meta, args, model)


def run_nli(dataset_meta: dict, args: argparse.Namespace, model: HF_LLM):
    return _run_classification(dataset_meta, args, model)


def run_coreference_resolution(dataset_meta: dict, args: argparse.Namespace, model: HF_LLM):
    return _run_classification(dataset_meta, args, model)


def run_question_answering(dataset_meta: dict, args: argparse.Namespace, model: HF_LLM):
    return _run_classification(dataset_meta, args, model)


def run_reading_comprehension(dataset_meta: dict, args: argparse.Namespace, model: HF_LLM):
    return _run_classification(dataset_meta, args, model)


def run_ner(dataset_meta: dict, args: argparse.Namespace, model: HF_LLM):
    return _run_classification(dataset_meta, args, model)


def run_re(dataset_meta: dict, args: argparse.Namespace, model: HF_LLM):
    return _run_classification(dataset_meta, args, model)


# ---------- generic classification (covers every task for now) ---------------
def _run_classification(dataset_meta: dict, args: argparse.Namespace, model: HF_LLM):
    """Load dataset, run inference, write preds."""
    name = dataset_meta["dataset_name"]
    logger.info("Working on %s (%s)", name, dataset_meta["task_category"])
    input_keys = dataset_meta["input"]
    if isinstance(input_keys, str):
        input_keys = [input_keys]

    # load HF dataset
    splits = dataset_meta["split"]
    custom_data = False
    if isinstance(splits, str):
        splits = [splits]
    try:
        raw_ds = datasets.load_dataset("/".join(name.split("/")[:-1]), name.split("/")[-1], split=splits[-1])   # use first split
    except Exception as e:
        custom_data = True
        raw_ds = []
        with open("data/" + name.split("/")[-1] + "/" + name.split("/")[-1] + "_" + splits[-1] + ".jsonl", "r") as f:
            for line in f:
                raw_ds.append(json.loads(line))
        


    # batch size tuned to your GPU
    dd = name.split("/")[-1]
    batch_size = 8
    out_file = f"outputs/{dd}_base.jsonl"
    prompt_file = "prompts/" + dd + ".txt"
    with open(prompt_file, "r") as txt:
        template = txt.read()
        

    with open(out_file, "w") as fw:
        
        for i in tqdm(range(0, len(raw_ds), batch_size), desc=name):
            
            batch = raw_ds[i : i + batch_size]
            # print(batch, input_keys)
            if custom_data:
                s = []
                for b in batch:
                    c = []
                    for ik in input_keys:
                        c.append(b[ik])
                    s.append(c)
                batch = [s]
            else:
                batch = [batch[j] for j in input_keys]
            preds = predict_batch(model, template, batch, input_keys, args)
            batch.append(preds)
            for g in zip(*batch):
                g = list(g)
                kg = {}
                kg["input"] = g[:-1]
                kg["output"] = g[-1]
                fw.write(json.dumps(kg, ensure_ascii=False) + "\n")
    logger.info("Finished %s -> %s", name, out_file)


# ---------- main driver ------------------------------------------------------
def run_predict(args: argparse.Namespace):
    # read meta file
    meta = defaultdict(list)
    with open("dataset_meta.jsonl") as f:
        for line in f:
            js = json.loads(line)
            meta[js["task_category"]].append(js)

    # instantiate model once
    model = HF_LLM(args.model_name)

    # --- run ONLY the single task requested -----------------------------
    task = args.task_type          # e.g. "sentiment_analysis"
    if task not in meta:
        raise ValueError(f"Task '{task}' not found in data_format.jsonl")
    runner = globals().get(f"run_{task}")
    if runner is None:
        raise ValueError(f"No runner function 'run_{task}' defined")
    for ds in meta[task]:
        runner(ds, args, model)


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


    timedate = datetime.now()
    date_time_string_1 = timedate.strftime("%Y-%m-%d %H:%M:%S")
    args = parser.parse_args()
    filename=f"{args.model_name}_{args.task_type}_{date_time_string_1}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    
    run_predict(args)

    