# Multiagent-NLU

## About

This is an automatic evaluation framework for Natural language understanding (NLU) tasks. The repository now features default prompts for each NLU task, as well as customizable scripts for generating outputs and evaluation. It is built with the Huggingface library.  

The NLU tasks are as follows, presented in data_format.jsonl

| Name | Input key | Output key | split |
| --- | --- | --- | --- |
| **Sentiment Analysis** |  |  |  |
| atrost/financial_phrasebank | sentence | label (0,1,2), 0 negative, 1 neutral, 2 positive | train, test |
| mteb/poem_sentiment | text | label (0,1,2,3), 3 means mixed. 2 means no_impact, 1 means positive, 0 means negative | train, test |
| imdb (need preprocess) | text | label (0,1) neg, pos | test |
| **paraphrase detection** |  |  |  |
| curaihealth/medical_questions_pairs (need preprocess) | question_1, question_2 | label (0,1) not equivalent, equi | train (need split) |
| nyu-mll/glue/mrpc (mrpc subset) | sentence1, sentence2 | label (0,1) not equivalent, equi | train, validation |
| **Natural language inference** |  |  |  |
| climate_fever (need preprocess) | claim, evidence | claim_label (0,1,2,3) support refute not_enough_info, disputed | test |
| slhenty/climate-fever-nli-stsb | sentence1, sentence2 | label (SUPPORTS, REFUTES) | train, test |
| nyu-mll/glue/rte | sentence1, sentence2 | label (0,1) entailment, not entailment | train, validation |
| **Hate Speech Detection (Sentiment analysis)** |  |  |  |
| cardiffnlp/tweet_eval/hate | text | label (0,1) non-hate, hate | train, validation |
| ethos | tweet | sentiment (normal, offensive) | train, test |
| **Question Answering** |  |  |  |
| tau/commonsense_qa | question, choices | answerKey (A,B,C,D,E) | train, validation |
| allenai/ai2_arc | question, choices | answerKey (A,B,C,D) | train, validation |
| **Sentence Completion** |  |  |  |
| Dream (from github) | question, choice | answer | train, test |
| **NER** |  |  |  |
| conll 03 | done already |  | train, test |
| mrc-05 | downloaded |  | train, test |

Additional tasks of the SuperGLUE are presented in dataset_meta.jsonl

| Name | Input key | Output key | split |
| --- | --- | --- | --- |
| aps/super_glue/axb | sentence1, sentence2 | label (0,1) entailment, not_entailment | test |
| aps/super_glue/axg | sentence1, sentence2 | label (0,1) entailment, not_entailment | test |
| aps/super_glue/boolq | question, passage | label (0,1) True, False | train, validation |
| aps/super_glue/cb | premise, hypothesis | label (0,1,2) entailment, contradiction, neutral | train, validation |
| aps/super_glue/copa | question, premise, choice1, choice2, (cause/effect) | label(0,1), choice1, choice2 | train, validation |
| aps/super_glue/multirc | paragraph, question, answer | label (0,1), False, True | train, validation |
| aps/super_glue/record | passage, query, entities | answers  | train, validation |
| aps/super_glue/rte | premise, hypothesis | label (0,1) entailment, not_entailment | train, validation |
| aps/super_glue/wic | word, sentence1, sentence2 | label (0,1), False, True | train, validation |
| aps/super_glue/wsc | text, span1_text, span2_text | label (0,1), False, True | train, validation |
|  |  |  |  |

## How it works

1. Generate predictions with run_test_all.py using the metadata files shown above. 
2. Evaluate with eval_all.py. Each evaluation metrics are specified in the metadata. 
