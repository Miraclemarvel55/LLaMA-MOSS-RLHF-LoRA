import importlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import torch


def generate_inputs(tokenizer, format_qa_f=lambda i, q, a:f"问：{q}\n答：{a}", query='', history=[]):
    assert query or history, "query and history cannot both empty"
    if not history:
        prompt = query
    else:
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            if i==len(history)-1 and query == "":
                prompt += format_qa_f(i, old_query)
            else:
                prompt += format_qa_f(i, old_query, response)
        if query:
            prompt += format_qa_f(len(history), query)
    inputs = tokenizer([prompt], return_tensors="pt")
    gen_len = 0
    if query=="":
        # query为空代表history的最后一个回答是目标答案
        last_response_encode = tokenizer.encode(history[-1][1], return_tensors="pt", add_special_tokens=False)
        if last_response_encode[0, 0] == 5:
            last_response_encode = last_response_encode[:, 1:]
            # TODO batch化
        eops = torch.zeros_like(last_response_encode[:, :1])+tokenizer.eos_token_id
        # TODO 后续用scatter来放到可能多个句子的带padding的正确位置，暂时先放到最后，因为现在只有一句
        last_response_encode = torch.cat([last_response_encode, eops], dim=-1)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], last_response_encode], dim=-1)
        gen_len = last_response_encode.shape[1]
    return inputs, gen_len

def get_class_in_module(class_name, module_path):
    """
    Import a module on the cache directory for modules and extract a class from it.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        module_dir = os.path.dirname(module_path)
        module_file_name = module_path.split(os.path.sep)[-1] + ".py"

        # Copy to a temporary directory. We need to do this in another process to avoid strange and flaky error
        # `ModuleNotFoundError: No module named 'transformers_modules.[module_dir_name].modeling'`
        shutil.copy(f"{module_dir}/{module_file_name}", tmp_dir)
        # On Windows, we need this character `r` before the path argument of `os.remove`
        cmd = f'import os; os.remove(r"{module_dir}{os.path.sep}{module_file_name}")'
        # We don't know which python binary file exists in an environment. For example, if `python3` exists but not
        # `python`, the call `subprocess.run(["python", ...])` gives `FileNotFoundError` (about python binary). Notice
        # that, if the file to be removed is not found, we also have `FileNotFoundError`, but it is not raised to the
        # caller's process.
        try:
            subprocess.run(["python", "-c", cmd])
        except FileNotFoundError:
            try:
                subprocess.run(["python3", "-c", cmd])
            except FileNotFoundError:
                pass

        # copy back the file that we want to import
        shutil.copyfile(f"{tmp_dir}/{module_file_name}", f"{module_dir}/{module_file_name}")

        # import the module
        module_path = module_path.replace(os.path.sep, ".")
        module = importlib.import_module(module_path)

        return getattr(module, class_name)


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    inputs = generate_inputs(tokenizer, query="", history=[["你好", "你好"]])
    print(inputs)
    inputs2 = generate_inputs(tokenizer, query="你好", history=[["你好", "你好"]])
    print(inputs2)

