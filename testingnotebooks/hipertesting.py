from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.append('/content/drive/MyDrive/ARCAGI3/models/')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = '/content/drive/MyDrive/ARCAGI3/models/qwen_coder_1.5b_instruct/'

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16, device_map="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)


messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Please prove why there is a n in variance from a gaussian distribution"}
    ]


text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors='pt').to(model.device)

outputs = model.generate(**inputs, max_new_tokens=500, temperature=0.2)
response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print(f'Response: {response}')

messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Please prove why there is a n in variance from a gaussian distribution"}
    ]


text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors='pt').to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.2)
response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print(f'Response: {response}')

#env code

import numpy as np
import pandas as pd



class grid_env:
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.grid = self.grid_creation()
        self.goal_pos = (grid_size-1, grid_size-1)
        self.start_pos = (0, 0)
        self.key_pos = (1, grid_size-1)
        self.p_plate_pos = [
            (grid_size - 1, 0), (grid_size - 1, 1),
            (grid_size - 2, 0), (grid_size - 2, 1)
        ]
        self.obstacles_pos = [
            (grid_size-2, grid_size-1), (grid_size-2, grid_size-2),
            (grid_size-1, grid_size-2)
        ]
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.key_obtained = 0
        self.char_pos = (0, 0)
        self.step_count = 0
        self.env_clear = False
    def grid_creation(self) -> dict:
      table = {}
      for j in range(self.grid_size):
        for i in range(self.grid_size):
          table[(i, j)] = 0
        table[self.start_pos] = 1
        table[self.key_pos] = 2
        for i in range(len(self.p_plate_pos)):
            table[self.p_plate_pos[i]] = 3
        for i in range(len(self.obstacles_pos)):
            table[self.obstacles_pos[i]] = 4
        table[self.goal_pos] = 5

        return table
    def move(self, x, y, action):
      if action not in self.actions:
        print(f'INVALID ACTION: {action}')
        return (x, y)
      self.step_count += 1
      org = (x, y)
      if action == 'UP':
          nx, ny = x, y + 1
      elif action == 'DOWN':
          nx, ny = x, y - 1
      elif action == 'LEFT':
          nx, ny = x - 1, y
      elif action == 'RIGHT':
          nx, ny = x + 1, y
      if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
          if self.grid[(nx, ny)] == 4:
              return org
          if (nx, ny) == self.goal_pos:
            self.env_clear = True
            self.char_pos = (nx, ny)
            print(f"Goal reached at step: {self.step_count}")
            return self.char_pos
          if (nx, ny) == self.key_pos and self.key_obtained == 0:
            self.key_obtained = 1
            self.grid[(self.key_pos)] = 8
            print(f'Key obtained at step: {self.step_count}')
          if (nx, ny) in self.p_plate_pos and self.key_obtained == 1:
            if self.grid[(self.p_plate_pos[0])] == 0:
              pass
            for pos in self.p_plate_pos:
              self.grid[(pos)] = 0
            for pos in self.obstacles_pos:
              self.grid[(pos)] = 0
            print(f"Plate activated at ({nx}, {ny}) | step: {self.step_count}")
          self.char_pos = (nx, ny)
          return self.char_pos
      else:
          return org

    def reward_function(self, oldx, oldy, x, y):
      if (oldx, oldy) == (x, y):
        return 0
      if (x, y) == self.goal_pos:
        return 10
      return 0
    def reset(self):
      self.key_obtained = 0
      self.char_pos = (0, 0)
      self.step_count = 0
      self.grid = self.grid_creation()
      return self.start_pos
    def get_grid_array(self) -> np.ndarray:
        """Convert dict grid to numpy array."""
        grid_array = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for (i, j), value in self.grid.items():
            grid_array[j, i] = value
        return grid_array

#helper functions
def grid_brief(grid):
    bg = np.bincount(grid.flatten()).argmax()
    items = []
    for c in range(16):
        if c == bg:
            continue
        ys, xs = np.where(grid == c)
        if len(ys) > 0:
            items.append(f"c{c}:{len(ys)}px@({xs.min()},{ys.min()})")
    return " ".join(items) if items else "empty"

def parse_action(response: str) -> str:
    response = response.strip().upper()


    for action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
        if action in response:
            return action

    if 'U' in response:
        return 'UP'
    elif 'D' in response:
        return 'DOWN'
    elif 'L' in response:
        return 'LEFT'
    elif 'R' in response:
        return 'RIGHT'

    print(f"Could not parse action from: {response}, PLEASE RETRY")
    return False

#testing
curr_pos = (0, 0)
max_steps = 100
grid_size = 8
current_h = ''
env = grid_env(8)
history = []

while env.step_count <= 100 and env.env_clear == False:
  grid_array = env.get_grid_array()
  grid_with_agent = grid_array.copy()
  (y, x) = env.char_pos #np arrays use (row, column) for some reason ;-;
  grid_with_agent[y, x] = 9
  brief = grid_brief(grid_with_agent)
  history_map_str = "".join(
            f"{h['action']} -> {h['result']}\n" for h in history[-10:]
        )
  if env.step_count % 10 == 0 and env.step_count != 0:
    #hypothesis creation
    messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"You are an expert environment analyzer.\n"
                            f"Observe the trajectory of actions and their effects on objects over the last 10 steps:\n"
                            f"Previous hypothesis: {current_h}\n"
                            f"Current grid state: {brief}"
                            f"{history_map_str}\n"
                            "Generate a 1 sentence sub-task for the next 10 steps\n"
                        ),
                    },
                ],
            }
        ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt').to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.2)
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    print(f'Response: {response}')
    current_h = response
  approved = False
  while not approved:
    messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"You are an expert sequential puzzle solver.\n"
                            f"Observe the trajectory of actions and their effects on objects over the last 10 steps:\n"
                            f"Current Hypothesis: {current_h}\n"
                            f"{history_map_str}\n"
                            "AVAILABLE ACTIONS: UP, DOWN, LEFT, RIGHT\n"
                            "Only ouput the action:"
                        ),
                    },
                ],
            }
        ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt').to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.2)
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    print(f'Response: {response}')
    parsed_action = parse_action(response)
    if parsed_action is not False:
      approved = True
