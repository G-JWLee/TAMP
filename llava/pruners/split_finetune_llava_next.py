import json

home_path="home_path"

if __name__ == '__main__':

    llava_finetune_list = json.load(open(f'{home_path}/TAMP/playground/LLaVA-NeXT-Data/llava_next_raw_format/llava_next_raw_format_processed.json','r'))

    unique_tasks = []
    task_idx_log = []
    task_indices_dict = {}

    for idx, item in enumerate(llava_finetune_list):
        assert 'image' in item
        task = item['image'].split("/")[0]
        if task not in unique_tasks:
            unique_tasks.append(task)
            task_idx_log.append(idx)
    
    for task_idx, task_name in enumerate(unique_tasks):
        task_start = task_idx_log[task_idx]
        if task_idx != len(unique_tasks)-1:
            task_end = task_idx_log[task_idx+1]
        else:
            task_end = len(llava_finetune_list) - 1
        task_indices_dict[task_name] = [task_start, task_end]

    with open(f'{home_path}/TAMP/playground/LLaVA-NeXT-Data/llava_next_raw_format/task_split.json','w') as f:
        json.dump(task_indices_dict, f)
        
    print("Done!")