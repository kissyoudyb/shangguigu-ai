from datasets import load_dataset

def convert_to_qwen_format(examples):
    conversations = []
    # 遍历每个对话样本,注意开启batch时，会自动套一层list
    for conv_list in examples["conversation"]:
        # 重建符合Qwen3标准的消息结构
        for conv in conv_list:
            conversations.append([
                {"role": "user", "content": conv['human'].strip()},
                {"role": "assistant", "content": conv['assistant'].strip()}
            ])

    return {"conversations": conversations}

if __name__ == '__main__':

    dataset = load_dataset("json", data_files="data/keywords_data_train.jsonl", split="train")
    # 格式化数据为 Chatgpt 格式
    dataset = dataset.map(
        convert_to_qwen_format,
        batched=True,
        remove_columns=dataset.column_names
    )
    dataset.to_json("data/keywords_data_sharegpt.json",force_ascii=False)

