from transformers import AutoTokenizer, AutoModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import BloomForCausalLM, BloomTokenizerFast
import torch
import sys

def main(model_choice, model_file):
    device = torch.device("cuda")
    prompt_constructor = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    def construct_prompt(input_, history):
        prompt = "以下是两个人的对话：\n"
        person1 = "第一个人："
        person2 = "第二个人："
        for i, (past_input, response) in enumerate(history):
            prompt += (person1 + past_input + "\n")
            prompt += (person2 + response + "\n")
        prompt += (person1 + input_ + "\n" + person2)
        return prompt
    
    if model_choice == "glm":
        model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).quantize(8).cuda()
        if model_file != "None":
            model.load_state_dict(torch.load(model_file))
        tokenizer = prompt_constructor
        def chat_fn(input_, history):
            return model.chat(tokenizer, input_, history)
    elif model_choice == "gpt2":
        model = GPT2LMHeadModel.from_pretrained("IDEA-CCNL/Wenzhong-GPT2-110M").cuda()
        if model_file != "None":
            model.load_state_dict(torch.load(model_file))
        tokenizer = GPT2Tokenizer.from_pretrained("IDEA-CCNL/Wenzhong-GPT2-110M")
        def chat_fn(input_, history):
            prompt = construct_prompt(input_, history)
            response = tokenizer.decode(model.generate(**(tokenizer(prompt, return_tensors="pt").to(device)), max_new_tokens=50)[0])
            response = response[len(prompt):] # Output includes the original prompt, truncate it
            history.append((input_, response))
            return response, history
    elif model_choice == "bloom":
        model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m").cuda()
        if model_file != "None":
            model.load_state_dict(torch.load(model_file))
        tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
        def chat_fn(input_, history):
            prompt = construct_prompt(input_, history)
            response = tokenizer.decode(model.generate(**(tokenizer(prompt, return_tensors="pt").to(device)), max_new_tokens=50)[0])
            response = response[len(prompt):] # Output includes the original prompt, truncate it
            history.append((input_, response))
            return response, history
    else:
        print("Input one of the following: glm, gpt2, bloom")
    
    print("Input \"q\" or \"quit\" to exit")
    user_input = input()
    history = []
    while user_input != "q" and user_input != "quit":
        response, history = chat_fn(user_input, history)
        print(response)
        user_input = input()

if __name__ == "__main__":
    print("Input model type and model file. If no model file, put None. WARNING: Bloom and GPT2 do not work well. Only GLM.")
    main(sys.argv[1], sys.argv[2])
    
