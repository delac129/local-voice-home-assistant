from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import threading
import time

# Load model
print("Loading GPT-2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")
print("GPT-2 loaded!")

#Training data
training_texts = [
    "Command: turn off the lights -> Action: lights_off",
    "Command: kill the lights -> Action: lights_off",
    "Command: lights off -> Action: lights_off",
    "Command: turn on the lights -> Action: lights_on",
    "Command: lights on -> Action: lights_on",
    "Command: it's too cold -> Action: increase_temp",
    "Command: I'm freezing -> Action: increase_temp",
    "Command: turn up the heat -> Action: increase_temp",
    "Command: it's too hot -> Action: decrease_temp",
    "Command: cool it down -> Action: decrease_temp",
    "Command: lock the door -> Action: lock_door",
    "Command: lock up -> Action: lock_door",
    "Command: unlock the door -> Action: unlock_door",
    "Command: good night -> Action: lights_off, lock_door, decrease_temp",
    "Command: goodnight -> Action: lights_off, lock_door, decrease_temp",
    "Command: I'm going to sleep -> Action: lights_off, lock_door, decrease_temp",
    "Command: good morning -> Action: lights_on, increase_temp, open_blinds",
    "Command: wake up -> Action: lights_on, increase_temp, open_blinds",
    "Command: I'm leaving -> Action: lights_off, lock_door",
    "Command: I'm heading out -> Action: lights_off, lock_door",
    "Command: bye -> Action: lights_off, lock_door",
    "Command: I'm home -> Action: lights_on, unlock_door",
    "Command: I'm back -> Action: lights_on, unlock_door",
    "Command: make it darker -> Action: dim_lights",
    "Command: dim the lights -> Action: dim_lights",
    "Command: it's too bright -> Action: dim_lights",
    "Command: brighten up -> Action: brighten_lights",
    "Command: more light -> Action: brighten_lights",
    "Command: close the blinds -> Action: close_blinds",
    "Command: open the blinds -> Action: open_blinds",
    "Command: movie time -> Action: dim_lights, close_blinds",
    "Command: bedtime -> Action: lights_off, lock_door, decrease_temp",
    "Command: I'm cold -> Action: increase_temp",
    "Command: turn everything off -> Action: lights_off, lock_door",
    "Command: shut everything down -> Action: lights_off, lock_door",
    "Command: party mode -> Action: brighten_lights, increase_temp",
    "Command: it's stuffy -> Action: decrease_temp, open_blinds",
]

#Tokenize
def tokenize_data(texts, tokenizer, max_length=128):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    encodings["labels"] = encodings["input_ids"].clone()
    return encodings

tokenized = tokenize_data(training_texts, tokenizer)

class HomeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings["input_ids"])
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

dataset = HomeDataset(tokenized)

#Train
print("Training...")
training_args = TrainingArguments(
    output_dir="./gpt2_home",
    num_train_epochs=30,
    per_device_train_batch_size=4,
    logging_steps=10,
    use_cpu=True,
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
print("Fine-tuning complete!")

#Gateway
app = Flask(__name__)
CORS(app)

def run_command(command):
    prompt = f"Command: {command} -> Action:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=15,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("-> Action:")[-1].strip()

@app.route('/command', methods=['POST'])
def handle_command():
    data = request.json
    if not data or 'command' not in data:
        return jsonify({"error": "No command provided"}), 400
    command = data.get('command', '')
    start = time.time()
    action = run_command(command)
    latency = round((time.time() - start) * 1000, 1)
    return jsonify({"command": command, "action": action, "latency_ms": latency, "status": "success"})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "online", "model": "gpt2-finetuned"})

if __name__ == '__main__':
    print("Gateway running on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000)