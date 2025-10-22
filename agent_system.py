from LLM import *
import datasets
from select_example import *
from collections import Counter
import argparse
import logging
import json
from tqdm import tqdm
import ast

def generate_using_metaprompt(model, prompt, data):
    result = model.run_inference(data["description"], data["sentence"], data["label"],prompt)

    return result["summarization"], result["verification"], result["execution"]

#evaluate labels for a single sentence
def error_type_ner(label, predict):
    # form of one label: [[entity_instance, type_of_entity]]
    tps = 0
    fns = 0
    fps = 0
    labs = []
    mods = []
    for j in label: 
        lab = [i.lower() for i in j]
        labs.append(lab)

    try:
        for j in predict: 
            mod = [i.lower() for i in j]
            mods.append(mod)
        for n in labs:
            if n not in mods:
                fns += 1
            else:
                tps += 1

        for n in mods:
            if n not in labs:
                fps += 1
        return tps, fns, fps
    except Exception as e:
        return tps, len(labs), 0
    

def evaluate_ner(labels, model_predicts):
    # labels: [[[a1,e1],[a2,e2]],[[a3,e3]]]
    tp = 0
    fp = 0
    fn = 0
    tot_pred = 0
    for (l, m) in zip(labels, model_predicts):
        tp_num, fn_num, fp_num = error_type_ner(l,m)
        tp += tp_num
        fp += fp_num
        fn += fn_num
    print(tp, fp, fn)
    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    f1_score = 2 * tp / float( 2 * tp + fp + fn)

    return precision, recall, f1_score




def is_equal(single_label, single_model_predict):
    if type(single_label) == list:
        lab = [i.lower() for i in single_label]
        try:
            mod = [i.lower() for i in single_model_predict]
            return Counter(lab) == Counter(mod)
        except Exception as e:
            return 0
    else:
        return int(single_label) == int(single_model_predict)


def run_multiagent_system_train(dataset_name, prompt, args):
    if args.model_name == "OSS":
        model = remote_LLM('http://10.244.50.59:1234/model', args.model_name)
    else:
        model = local_LLM()

    
    
    loaded_data = datasets.load_dataset(name=args.dataset_name, split="train")
    sentence_name = loaded_data[0].keys[0]
    label_name = loaded_data[0].keys[1]

    #select data sample for meta-prompt generation
    sample_data = loaded_data[0]

    #generate meta prompts
    
    
    #take some part of the data for training, using some data-filtering mechanism. 
    #goal: to create a subset of data which can represent the distribution of the entire training set. 
    filtered_data = filter_train(loaded_data)

    i = 0
    batch_size = 10
    retries = 0
    max_retries = 3
    # training + feedback stage
    while True:
        preds = []
        summaries = []

        if retries == 0:
            sum_prompt, ver_prompt, exe_prompt = generate_using_metaprompt(model, prompt, sample_data)
        else:
            sum_prompt = generate_using_metaprompt(model, prompt, sample_data)
        #go over labels first
        for sentence,label in filtered_data[i:i+batch_size]: 
            summary = model.summarize_patterns(sentence, label, sum_prompt)
            predict = model.pattern_work(sentence, exe_prompt, summary)

            summaries.append(summary)
            preds.append(predict)

        #evaluate batch results
        precision_batch, recall_batch, f1_score_batch = evaluate_ner(filtered_data[i:i+batch_size]["label"], preds)

        #if f1 score is too low (<80%), provide feedback to the first agent
        if f1_score_batch < 0.9 and retries < max_retries: 
            feedback_res = model.verify_patterns(filtered_data[i:i+batch_size]["sentences"], filtered_data[i:i+batch_size]["labels"], ver_prompt)
            max_retries += 1
        else:
            i += batch_size
            retries = 0


def run_multiagent_system_test(dataset_name, model_name, data, args):
    return

def run_baseline_test(dataset_name, prompt, args):
    if args.model_name == "OSS":
        model = remote_LLM('http://10.244.50.59:1234/model', args.model_name)
    else:
        model = local_LLM()

    if args.custom_data != '':
        loaded_data = []
        limit = 1600
        ct = 0
        with open(args.custom_data) as f:
            for i, line in enumerate(f):
                ct += 1
                line_json = json.loads(line)
                loaded_data.append(line_json)
                if ct == limit:
                    break

    else:
        loaded_data = datasets.load_dataset(name=args.dataset_name, split="test")

    preds = []
    labels = []
    for j in tqdm(loaded_data):
        sentence = j["text"]
        label = j["entity_labels"]
        labels.append(label)
        print("label: ", label)

        pred = model.predict_single(sentence, prompt)
        pred = ast.literal_eval(pred)
        print(pred)
        preds.append(pred)
    
    precision_batch, recall_batch, f1_score_batch = evaluate_ner(labels, preds)
    
    
    
    return precision_batch, recall_batch, f1_score_batch




        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process agent NLU')
    parser.add_argument('--dataset_name', type=str, default='', help='name of huggingface dataset')
    parser.add_argument('--custom_data', type=str, default='', help='custom dataset directory')
    parser.add_argument('--prompt', type=str, help='file containing prompt')
    # parser.add_argument('--r_path_j', type=str, help='judge输入文件地址')
    # parser.add_argument('save_log', type=str, help='save directory for log file')
    parser.add_argument('--model_name', type=str, help='name of LLM for the experiment')
    parser.add_argument('--task_type', type=str, help='specify what type of experiment to run e.g. train, test, baseline, etc.')
    #logger = logging.

    args = parser.parse_args()
    with open(args.prompt) as f:
        prompt = f.read()

    if args.task_type == "baseline":
        p, r, f1 = run_baseline_test("", prompt, args)
        print(p,r,f1)
    
    if args.task_type == "train":
        run_multiagent_system_train("", prompt, args)

    if args.task_type == "test":
        run_multiagent_system_test("", prompt, args)





    



        









    
    