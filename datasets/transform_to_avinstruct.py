# Based on https://github.com/rikeilong/Bay-CAT/tree/main/AVinstruct json files and 
# https://github.com/GeWu-Lab/MUSIC-AVQA/tree/main AVQA-Music dataset, we change this into AVinstruct for videollama2 format.

import os
import json
import argparse


def video_exists(video_dir, video_id):
    video_path = os.path.join(video_dir, video_id + '.mp4')
    return os.path.exists(video_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--dataset_path1', type=str, required=True)
    parser.add_argument('--dataset_path2', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)

    args = parser.parse_args()

    with open(args.dataset_path1, 'r') as f1:
        avinstruct = json.load(f1)
    
    with open(args.dataset_path2, 'r') as f2:
        avinstruct2 = json.load(f2)
    
    avinstruct.extend(avinstruct2)

    transformed_avinstruct = []
    num_samples = 0

    for idx, sample in enumerate(avinstruct):
        video_id = sample.get('video_id', "")
        if video_exists(args.video_dir, video_id):
            updated_conversations = []
            for conv_idx, conv in enumerate(sample["conversations"]):
                # Place <video> token to the front
                if conv_idx == 0 and conv["from"] == "human":
                    conv["value"] = conv["value"].replace("<video>", "").strip()
                    temp = conv["value"]
                    conv["value"] = f"<video>\n{temp}"
                # Remove <Q> token
                if conv["from"] == "human":
                    conv["value"] = conv["value"].replace("<Q>", "").strip()
                updated_conversations.append(conv)
                
                if conv_idx >= 2:
                    print("check")

            transformed_avinstruct.append(
                {"id": num_samples,
                 "audio": video_id + '.mp4',
                 "video": video_id + '.mp4',
                 "conversations": updated_conversations})
            num_samples += 1
    
    with open(args.save_path, 'w') as f:
        json.dump(transformed_avinstruct, f)
    
    print("finished!")
