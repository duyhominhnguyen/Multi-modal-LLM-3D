import json
import os
import tarfile

def compress_directory_to_tar(directory_path):
    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    os.makedirs('replicate_mmc4', exist_ok=True)
    
    for i in range(0, len(json_files), 20):
        batch_files = json_files[i:i+20]
        tar_file_path = os.path.join('replicate_mmc4', f"{i//20:09d}.tar")
        
        with tarfile.open(tar_file_path, "w:gz") as tar:
            for file in batch_files:
                tar.add(os.path.join(directory_path, file), arcname=file)
        
        print(f"Batch {i//20} compressed to {tar_file_path}")

def convert_adni_to_mmc4(input_json_path, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load the large JSON file
    with open(input_json_path, 'r') as f:
        data = json.load(f)
        
    matched_text_index = 0

    # Iterate over each item in the list and save it as a separate JSON file
    for idx, item in enumerate(data):
        # Ensure compatibility with the structure of f9773b9c866145c28fe0b701dde8dfbe.json
        
        # Handle text list:
        conversations = item.get("conversations", None)
        if conversations is not None:
            text_list = []
            for conversation in conversations:
                text_list.append(conversation["value"])
                
            # Check for <image> tag in the first element of conversations list
            first_convo = conversations[0]["value"]
            if "<image>" in first_convo:
                if first_convo.startswith("<image>"):
                    matched_text_index = 0
                elif first_convo.endswith("<image>"):
                    matched_text_index = 1
                
            item["text_list"] = text_list
            
        # Handle image's base64 content:
        with open('./sample_base64.txt', 'r') as f:
            sample_img_base64_data = f.read()
            
        # Handle image info:
        img_info = []
        images_list = item.get("image", None)
        if images_list is not None:
            for img in images_list:
                img_obj = {}
                img_obj["image_name"] = img
                img_obj["raw_url"] = "https://example.com/{}".format(img)
                img_obj["matched_text_index"] = matched_text_index
                img_obj["matched_sim"] = 0.75
                img_obj["image_base64"] = sample_img_base64_data
                img_info.append(img_obj)
                
        # Create similarity_matrix
        similarity_matrix = []
        for img in img_info:
            for _ in range(len(text_list)):
                inner_list = [0] * len(text_list)
                inner_list[matched_text_index] = 1
            similarity_matrix.append(inner_list)

        # item["similarity_matrix"] = similarity_matrix
        
        output_item = {
            "id": item.get("id", None),
            "url": "https://example.com",
            "text_list": item.get("text_list", None),
            "image_info": img_info,
            "similarity_matrix": similarity_matrix,
            "could_have_url_duplicate": 0
        }

        # Save the item as a separate JSON file
        output_path = os.path.join(output_folder, f"{idx:05d}.json")
        with open(output_path, 'w') as out_f:
            json.dump(output_item, out_f)

if __name__ == "__main__":
    print("GOOO")
    input_json_path = "/home/azureuser/maund/open_flamingo/modifications/VLM_ADNI_DATA/AD_caption-flamingo_3D_version.json"
    output_folder = "/home/azureuser/maund/open_flamingo/modifications/VLM_ADNI_DATA/dummy"
    convert_adni_to_mmc4(input_json_path, output_folder)
    compress_directory_to_tar(output_folder)