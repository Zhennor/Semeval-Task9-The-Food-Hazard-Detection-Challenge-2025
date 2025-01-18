import google.generativeai as genai
import pandas as pd
import json
import os
import re 
import time
from google.generativeai.types import HarmCategory, HarmBlockThreshold

def extract_square_bracket_content(text):
    match = re.search(r'(?<=\[)(.*?)(?=\])', text, re.DOTALL)

    if match:
        return (match.group(0))  
    else:
        print("Không tìm thấy đoạn văn bản khớp.")

def log_progress(index, file_path):
    try:
        with open(file_path, 'a') as f:  
            f.write(f"Processed index: {index}\n")  
        print(f"Logged index {index} to {file_path}.")
    except Exception as e:
        print(f"Error while logging index {index}: {e}")

def append_to_json(sample, file_path):
    try:
        if isinstance(sample, pd.Series):
            sample = sample.to_dict()

        with open(file_path, 'r+') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []  

            if isinstance(data, dict):
                data = [data]  

            data.append(sample)

            f.seek(0)
            json.dump(data, f, indent=4)
        
        print(f"Sample appended to {file_path}.")
    except Exception as e:
        print(f"Error while appending sample to {file_path}: {e}")

data = pd.read_json('/kaggle/input/data-clean-final/translate/data_clean.json')

genai.configure(api_key="your_api_key")
model = genai.GenerativeModel("gemini-2.0-flash-exp")

progress_txt_path = '/kaggle/working/progress.txt'
with open(progress_txt_path, 'w') as txt_file:
    txt_file.write('')

aug_label_hazard = [
    "foreign bodies",
    "chemical",
    "fraud",
    "other hazard",
    "packaging defect",
    "migration",
    "organoleptic aspects",
    "food additives and flavourings"
]

aug_label_product = [
    "cereals and bakery products",
    "fruits and vegetables",
    "prepared dishes and snacks",
    "nuts, nut products and seeds",
    "soups, broths, sauces and condiments",
    "seafood",
    "cocoa and cocoa preparations, coffee and tea",
    "confectionery",
    "ices and desserts",
    "herbs and spices",
    "non-alcoholic beverages",
    "dietetic foods, food supplements, fortified foods",
    "food contact materials",
    "fats and oils",
    "alcoholic beverages",
    "pet feed",
    "other food product / mixed",
    "honey and royal jelly",
    "food additives and flavourings",
    "feed materials",
    "sugars and syrups"
]

#Augument data follow product-category 
for label in aug_label_product:
    count_aug=0
    while(count_aug<=100):
        filtered_data = data[data['product-category'] == label]
        sampled_data = filtered_data.sample(n=min(50, len(filtered_data)), replace=False)
        prompt=f'''
                Giả sử bạn là chuyên gia  nghiên cứu nguy hiểm gia vị, 
                hãy tổng hợp thông tin về text và title có label dựa vào hazard-category và product-category rồi tạo ra 20 mẫu dữ liệu mới 
                về text và title mô phỏng chính xác phong cách, ngôn ngữ, và nội dung dưới dạng theo mẫu, tuyệt đối không được tạo mới 
                hay thay đổi label, trong mục text và title chỉ ghi nội dung của text và title không ghi gì thêm và giữ giá trị label giống 
                với mẫu với dạng data mẫu như sau ,tuyệt đối tạo đủ các columns chứa đày đủ các thuộc tính,
                 và sau đó hãy gửi tôi file json chứa đầy đủ 20 mẫu: 
        sample: 
                {sampled_data}
                '''
        response = model.generate_content(prompt,safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        })
        print(response.text)
                #Gọi hàm
        extracted_text=extract_square_bracket_content(response.text)
        if extracted_text:
            print("OK")
        else:
            print("Keyword not found.")
        try:
            with open(progress_txt_path, 'a') as f:
                f.write(f"{extracted_text},\n")
            print("Ghi thành công")
        except Exception as e:
            print(f"Lỗi")
        count_aug+=20


#Augument data following hazard-category
for label in aug_label_hazard:
    count_aug=0
    while(count_aug<=100):
        filtered_data = data[data['hazard-category'] == label]
        sampled_data = filtered_data.sample(n=min(50, len(filtered_data)), replace=False)
        prompt=f'''
                Giả sử bạn là chuyên gia  nghiên cứu nguy hiểm gia vị, 
                hãy tổng hợp thông tin về text và title có label dựa vào hazard-category và product-category rồi tạo ra 20 mẫu dữ liệu mới 
                về text và title mô phỏng chính xác phong cách, ngôn ngữ, và nội dung dưới dạng theo mẫu, tuyệt đối không được tạo mới 
                hay thay đổi label, trong mục text và title chỉ ghi nội dung của text và title không ghi gì thêm và giữ giá trị label giống 
                với mẫu với dạng data mẫu như sau ,tuyệt đối tạo đủ các columns chứa đày đủ các thuộc tính,
                 và sau đó hãy gửi tôi file json chứa đầy đủ 20 mẫu: 
        sample: 
                {sampled_data}
                '''
        response = model.generate_content(prompt,safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        })
        print(response.text)
        extracted_text=extract_square_bracket_content(response.text)
        if extracted_text:
            print("OK")
        else:
            print("Keyword not found.")
        try:
            with open(progress_txt_path, 'a') as f:  
                f.write(f"{extracted_text},\n")  
            print("Ghi thành công")  
        except Exception as e:
            print(f"Lỗi")
        count_aug+=20

print('All are completed')