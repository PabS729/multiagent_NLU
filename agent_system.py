from LLM_all import *
import datasets
from select_example import *
from collections import Counter
import argparse
import logging
import json
from tqdm import tqdm
import ast
from eval_all import *
from run_test_all import *
import copy




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

def load_pred(pred, output_map):
    try:
        b = json.loads(pred)
        r = output_map[b["answer"]]
        return r
    except Exception as e:
        print(e)
        return b


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


def run_multiagent_system_train(dataset_meta, task_meta, prompt, args):
    if args.gn_model_name == "OSS":
        gen_model = remote_LLM('http://10.244.50.59:1234/model', args.model_name)
    else:
        gen_model = HF_LLM(args.model_name)
    
    if args.exe_model_name == "OSS":
        exe_model = remote_LLM('http://10.244.50.59:1234/model', args.model_name)
    else:
        exe_model = HF_LLM(args.model_name)

    if args.sum_model_name == "OSS":
        sum_model = remote_LLM('http://10.244.50.59:1234/model', args.model_name)
    else:
        sum_model = exe_model
    
    if args.ver_model_name == "OSS":
        ver_model = remote_LLM('http://10.244.50.59:1234/model', args.model_name)
    else:
        ver_model = exe_model

    
    with open("prompts/meta_prompt.txt") as f:
        meta_prompt_tmpl = f.read()
    with open("prompts/meta_prompt_tmpl_w_feed.txt") as g:
        meta_prompt_tmpl_w_feed = g.read()

    input_example = task_meta["task_input"]
    label = task_meta["label"]
    output_format = task_meta["task_output"]
    task_description = task_meta["task_description"]
    inputs_meta = [task_description, input_example, label, output_format]

    input_map_meta = ["description", "inputs", "label", "outputs"]

    name = dataset_meta["name"]
    input_map = dataset_meta["input_format"]
    output_key = dataset_meta["output"]
    output_map = dataset_meta["output_map"]

    loaded_data = load_data(args, "train")


    #take some part of the data for training, using some data-filtering mechanism. 
    #goal: to create a subset of data which can represent the distribution of the entire training set. 
    filtered_data = filter_train(loaded_data, input_map, output_map=output_key) 

    i = 0
    batch_size = 100
    retries = 0
    max_retries = 3
    wrongs_batch = 4
    # training + feedback stage
    while True:
        preds = []
        summaries = []
        feedback = ""
        labels = []
        diff_idxs = []

        if retries == 0:
            meta_prompt = format_prompt(meta_prompt_tmpl, inputs_meta, input_map_meta)
            json_prompts = gen_model.run_inference(meta_prompt)
            sum_prompt, ver_prompt, exe_prompt = json_prompts["summarization_prompt"], json_prompts["evaluation_prompt"], json_prompts["execution_prompt"]

            for j in filtered_data: 
                #summarize
                inputs = j["inputs"]
                label = j["output"]
                inputs_s = copy.deepcopy(inputs) + [label]
                input_map_s = copy.deepcopy(input_map) + [output_key]
                form_sum = format_prompt(sum_prompt, inputs_s, input_map_s)
                summary = sum_model.run_inference(form_sum)

                #execute using given summary
                form_exec = format_prompt(exe_prompt, inputs, input_map)
                predict = exe_model.run_inference(form_exec)

                if args.task_type != "NER": 
                    predict = load_pred(predict, output_map)

                summaries.append(summary)
                preds.append(predict)
                labels.append(label)

                #need one model to verify and drop unecessary patterns for the correct predictions.
        else:
            preds = []
            labels = []
            meta_prompt = format_prompt(meta_prompt_tmpl_w_feed, inputs_meta, input_map_meta, feedbacks)
            json_prompts = gen_model.run_inference(meta_prompt)
            sum_prompt = json_prompts["summarization_prompt"]

            for j in wrong_idxs: 
                inputs = j["inputs"]
                label = j["output"]
                inputs_s = copy.deepcopy(inputs) + [label]
                input_map_s = copy.deepcopy(input_map) + [output_key]
                form_sum = format_prompt(sum_prompt, inputs_s, input_map_s)
                summary = sum_model.run_inference(form_sum)

                #execute using given summary
                form_exec = format_prompt(exe_prompt, inputs, input_map)
                predict = exe_model.run_inference(form_exec)

                if args.task_type != "NER": 
                    predict = load_pred(predict, output_map)

                summaries.append(summary)
                preds.append(predict)
                labels.append(label)


        for k in range(len(filtered_data)):
            if preds[k] != label[k]:
                diff_idxs.append(k)
            
        #evaluate batch results
        if args.task_category == "NER": 
            precision_batch, recall_batch, f1_score_batch = evaluate_ner(filtered_data["label"], preds)
            eval_res = f1_score_batch
        else:
            eval_res = evaluate_preds(name, labels, preds)

        #if f1 score is too low (<80%), provide feedback to the first agent
        if  eval_res < 0.8 and retries < max_retries: 
            #takes wrong samples （in batches）
            
            wrong_idxs = [filtered_data[i] for i in diff_idxs]
            wrong_preds = [preds[i] for i in diff_idxs]

            feedbacks = []
            for i in range(0, wrong_idxs, wrongs_batch):
                wrongs = ""
                
                for j in range(i,i+wrongs_batch):
                    wrong_ex = "{inputs: " + str({k:v for (k,v) in zip(input_map, wrong_idxs[j]["inputs"])}) + ", outputs: " + wrong_idxs[j]["output"] + ", model_predict: " + wrong_preds[i] + "}"
                    wrongs += wrong_ex + "\n"

                
                
                ver_format = ver_prompt.format(task_desc=task_description, wrong_examples=wrongs)

                feedback = ver_model.verify_patterns(ver_format)
                feedbacks.append(feedback)

            retries += 1
        else:
            i += batch_size
            retries = 0


def run_multiagent_system_test(prompt_exec, dataset_meta, model_name, patterns, data, args):
    loaded_data = load_data(args, "validation")

    if args.exe_model_name == "OSS":
        exe_model = remote_LLM('http://10.244.50.59:1234/model', args.model_name)
    else:
        exe_model = HF_LLM(args.model_name)

    input_map = dataset_meta["input_format"]
    output_key = dataset_meta["output"]

    processed_data = filter_test(input_map, output_key, args)



    #select patterns TODO

    #load patterns first, then choose from them using another LLM
    


    #predict using selected patterns

    for k in processed_data:
        inputs = k["inputs"]
        inputs.append(patterns)
        prompt_exec_form = format_prompt(prompt_exec, input_map, inputs)




    for j in loaded


    return

def run_baseline_test(dataset_name, prompt, args):
    if args.model_name == "OSS":
        model = remote_LLM('http://10.244.50.59:1234/model', args.model_name)
    else:
        model = HF_LLM(args.model_name)

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
        loaded_data = datasets.load_dataset(name=args.dataset_name, split="test")

    preds = []
    labels = []
    for j in tqdm(loaded_data):
        sentence = j["text"]
        label = j["entity_labels"]
        labels.append(label)
        print("label: ", label)

        pred = model.predict_single(sentence, prompt)
        # pred = ast.literal_eval(pred)
        # print(pred)
        # preds.append(pred)
        try:
            pred = ast.literal_eval(pred)
            print(pred)
            preds.append(pred)
        except Exception as e: 
            print(pred, " caught error")
            preds.append([])
    
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





    



        









    
    