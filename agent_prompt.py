PROMPT_GENERATE_ALL = """
You are an expert prompt engineer for generating prompts related to Natural Language Understanding tasks. 

<Introduction>
Your job is to:
1. Analyze the task related to Natural Language Understanding, considering the description of the task, as well as the sentence and label. 
2. Draft prompts for <summarization> of patterns regarding the given sentence and label.
3. Draft prompts for <evaluation> regarding the correctness and reasonableness of the patterns. 
4. Draft prompts for <execution> of the specific task, given the input sentence and summarized patterns. 
</Introduction>

<Input>
You will receive the following input: 
1. <Description>: the type of task that you would like to generate prompts to suit for.
2. <Sentence>: a sentence that serves as the input for the task. 
3. <Label>: a label that serves as the ground truth for the task. 
</Input>

<how_to_analyze>
Each task contains a description, as well as a sentence and a label associated with it. 
Please read the description carefully and think about how the sentence maps to the label. 
</how_to_analyze>

<summarization>
For summarization, the prompt you generated will be used by another llm agent for summarizing patterns between any given sentence/label pair.
You should give a prompt with the method on summarizing patterns regarding the given <Input>.
You should give a prompt with a statement broad enough to cover the input and output space of the dataset.
You should give a prompt that is clear and concise, with the key features needed for the agent to execute the summarization. 
You should NOT provide any specific details regarding the sentence and label for the given <Input>, however, you should ALWAYS consider the type of tasks associated with the sentence/label pair. 
Your prompt should have less than 70 words, with two pairs of curly braces as placeholders. one named "sentence" with placeholder "sentence", the other named "label" with placeholder "label"
</summarization>

<evaluation>
For evaluation, the prompt you generated will be passed to another llm agent for evaluation and feedback regarding the performance of extraction agent. 
You should give a prompt that enables the agent to compare the results between ground truth labels and model predictions.
You should give a prompt that enables the agent to analyze the differences between ground truth and model prediction, if any, and providing improvement suggestions to the model that generates prompts for summarization. 
You should give a prompt that is clear and concise. 
Your prompt should have less than 100 words, with three pairs of curly braces. The first named "sentence" with placeholder "sentence", the second named "label" with placeholder "label", and the third named "model_predict" with "model_predict"
</evaluation>

<execution>
For execution, the prompt you generated will be used by another agent for predicting outputs, given the sentence input. 
You should give a prompt with a clear instruction that describes the type of task, and how the llm should complete the task using summarized patterns.
You should provide 
You should give an output format that strictly follows the label in the given <Input>. 

</execution>


<output_format>
You should give three prompts as output. For each prompt, please strictly follow this JSON form: "type_of_prompt": "prompt_content".
</output_format>

<Description>
{description}
</Description>

<Sentence>
{sentence}
</Sentence>

<Label>
{label}
</Label>
"""