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

#env code

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches



class grid_env:
    def __init__(self, grid_size: int):
        self.start_pos = (0, 0)
        self.grid_size = grid_size
        self.goal_pos = (grid_size-1, grid_size-1)
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
        self.grid = self.grid_creation()
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
    def move(self, action):
      (x, y) = self.char_pos
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
    def plot_grid(self, figsize=(8, 8)):
      grid_array = self.get_grid_array()
      x, y = self.char_pos
      grid_array[y, x] = 9

      fig, ax = plt.subplots(figsize=figsize)

      color_map = {
            0: 'white',       # Empty
            1: '#90EE90',     # Start (light green)
            2: '#FFD700',     # Key (gold)
            3: '#87CEEB',     # Pressure plate (light blue)
            4: '#2F2F2F',     # Obstacle (dark gray)
            5: '#FF4444',     # Goal (red)
            8: '#FF8C00',     # Key collected (dark orange)
            9: '#4169E1'      # Agent (royal blue)
        }

      #
      for i in range(self.grid_size):  # y (row)
            for j in range(self.grid_size):  # x (col)
                # Get cell value
                if (j, i) == self.char_pos:
                    value = 9
                else:
                    value = self.grid.get((j, i), 0)

                # Draw rectangle
                color = color_map.get(value, 'white')
                rect = patches.Rectangle(
                    (j, i), 1, 1,  # (x, y, width, height)
                    facecolor=color,
                    edgecolor='gray',
                    linewidth=1
                )
                ax.add_patch(rect)

      symbols = {
          1: 'S', #start
          2: 'K', #key
          3: 'P', #p_plate
          4: 'X', #obs
          5: 'G', #goal
          8: 'W', #key_ob
          9: 'A' #agent
      }

      for i in range(self.grid_size): #y
        for j in range(self.grid_size): #x
          if (j, i) == self.char_pos:
            value = 9
          else:
            value = self.grid.get((j, i), 0)
          if value in symbols:
            ax.text(j + 0.5, i + 0.5, symbols[value], ha='center', va='center', fontsize=16, fontweight='bold')
      ax.set_xticks(range(self.grid_size + 1))
      ax.set_yticks(range(self.grid_size + 1))
      ax.set_xticklabels(range(self.grid_size + 1))
      ax.set_yticklabels(range(self.grid_size + 1))
      ax.grid(True, color='gray', linewidth=0.5)

      status = f'step: {self.step_count} | key obtained: {self.key_obtained} | Clear: {self.env_clear}'
      ax.set_title(f'Grid env: ({self.grid_size} * {self.grid_size})\n{status}', fontsize=12)

      ax.invert_yaxis()

      plt.tight_layout()
      plt.show()

def test_environment():
    env = grid_env(grid_size=5)

    print('Initial Grid')
    print(pd.DataFrame(env.get_grid_array()))
    print()

    # Plot initial grid
    env.plot_grid()

    # Test some moves
    moves = ['UP', 'UP', 'UP', 'UP', 'LEFT', 'LEFT', 'LEFT', 'DOWN', 'RIGHT']

    for action in moves:
        if env.env_clear:
            break
        print(f'\nAction: {action}')
        new_pos = env.move(action)
        print(f'Position: {env.char_pos}')
        print(f'Key obtained: {env.key_obtained}')
        print(f'Environment clear: {env.env_clear}')

        # Plot after significant events
        if env.key_obtained or env.env_clear or env.step_count % 5 == 0:
            env.plot_grid()

    print('\nFinal Grid')
    print(pd.DataFrame(env.get_grid_array()))
    env.plot_grid()

if __name__ == "__main__":
    test_environment()

#helper functions
def grid_hash(grid):
    """Create a hash of the grid for state tracking."""
    return hash(grid.tobytes())

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

def grid_diff_text(
    old, new, norm_action_str, state_action_memory, seen_grid_hashes
):
    mask = old != new
    if not np.any(mask):
        return "no change (0px)"

    ys, xs = np.where(mask)
    changed_pixels = len(ys)
    old_h = grid_hash(old)
    new_h = grid_hash(new)

    if (old_h, norm_action_str) in state_action_memory:
        return f"{changed_pixels}px changed [REPEAT MOVE FROM THIS STATE]"

    if new_h in seen_grid_hashes:
        return f"{changed_pixels}px changed [REVISITED PREVIOUS STATE]"

    if changed_pixels > 8:
        return f"{changed_pixels}px changed [NEW STATE]"

    changes = [f"({x},{y}):{old[y,x]}->{new[y,x]}" for y, x in zip(ys, xs)]
    return f"{', '.join(changes)} [NEW STATE]"

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
env = grid_env(grid_size)
history = []
total_reward = 0
state_action_memory = {}
seen_grid_hashes = set()

print('initial grid:')
env.plot_grid()

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
        "content":
                            f"You are an expert environment analyzer.\n"
                            f"Observe the trajectory of actions and their effects on objects over the last 10 steps:\n"
                            f"{history_map_str}\n"
                            f"Previous hypothesis: {current_h}\n"
                            f"Current grid state: {brief}\n"
                            f"Environment clear: {env.env_clear}\n"
                            "Generate a 2 sentence task for the next 10 steps."
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
        "content": (
                            f"You are an expert sequential puzzle solver.\n"
                            f"Observe the trajectory of actions and their effects on objects over the last 10 steps:\n"
                            f"Current Hypothesis: {current_h}\n"
                            f"{history_map_str}\n"
                            f"Environment clear: {env.env_clear}\n"
                            "AVAILABLE ACTIONS: UP, DOWN, LEFT, RIGHT\n"
                            "Only output the action:"
        )
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
      action = parsed_action

  #execution
  old_pos = env.char_pos
  old_grid = env.get_grid_array().copy()
  new_pos = env.move(action)
  new_grid = env.get_grid_array().copy()
  reward = env.reward_function(old_pos[0], old_pos[1], new_pos[0], new_pos[1])
  total_reward += reward

  grid_diff = grid_diff_text(
        old_grid,
        new_grid,
        action,
        state_action_memory,
        seen_grid_hashes
    )

  history.append({
        'step': env.step_count,
        'action': action,
        'result': f"{old_pos} → {new_pos}",
        'reward': reward,
        'grid_diff': grid_diff,
        'key_obtained': env.key_obtained,
        'env_clear': env.env_clear
    })
  seen_grid_hashes.add(grid_hash(new_grid))
  state_action_memory[(grid_hash(old_grid), action)] = True

  print(f'ACTION: {parsed_action}')
  print(f"  {old_pos} to {new_pos} | Reward: {reward:.2f} | Total: {total_reward:.2f}")
  print(f'Current position: {env.char_pos}')
  print(f'Key obtained: {env.key_obtained}')
  print(f'Plot at step: {env.step_count}')
  env.plot_grid()

  if env.env_clear:
        print(f"\n🎉 SUCCESS! Environment cleared in {env.step_count} steps!")
        break
