from torchvision import transforms
import re

def preprocess_question(question):
    question = question.lower().strip().split()
    return ["<sos>"] + question + ["<eos>"]

def preprocess_answer(answer):
    answer = re.sub(" ", "_", answer.strip()).lower()
    return answer

def get_transform(target_size):
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])