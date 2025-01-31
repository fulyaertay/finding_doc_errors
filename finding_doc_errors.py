# A project that uses Q-learning algorithm to detect incorrect information in a file
# Import required libraries
import numpy as np
import random

# Q-learning parameters
LEARNING_RATE = 0.1  # Learning rate
DISCOUNT_FACTOR = 0.9  # Future rewards consideration factor
EPSILON = 0.1  # Random exploration probability

# Create sample format file
def create_format_file(format_filename):
    with open(format_filename, "w") as f:
        f.write("name=STRING\n")
        f.write("age=INTEGER\n")
        f.write("email=STRING\n")
        f.write("address=STRING\n")

# Create sample data file (contains valid and invalid lines)
def create_sample_file(sample_filename):
    with open(sample_filename, "w") as f:
        f.write("name=Fulya\n")  # Correct format
        f.write("age=30\n")  # Correct format
        f.write("email=fulya@ai.com\n")  # Correct format
        f.write("invalid_line\n")  # Invalid format (missing '=')
        f.write("address=\n")  # Invalid: Value cannot be empty
        f.write("age=thirty\n")  # Invalid: Age must be INTEGER

# Function to determine line state
def get_state(line, format_dict):
    if "=" not in line:
        return "INVALID_FORMAT"  # If '=' is missing, wrong format
    key, value = line.split("=", 1)
    if key not in format_dict:
        return "UNKNOWN_KEY"  # Undefined key used
    if format_dict[key] == "INTEGER" and not value.isdigit():
        return "INVALID_TYPE"  # INTEGER expected but string entered
    if value.strip() == "":
        return "EMPTY_VALUE"  # Value cannot be empty
    return "VALID"  # Valid format

# Function to return errors as messages
def take_action(state, line):
    if state == "INVALID_FORMAT":
        return "Error: Missing '='"
    elif state == "UNKNOWN_KEY":
        return "Error: Unknown key"
    elif state == "INVALID_TYPE":
        return "Error: Invalid data type"
    elif state == "EMPTY_VALUE":
        return "Error: Value cannot be empty"
    return "Valid"

# Function implementing Q-learning algorithm
def q_learning(sample_filename, format_filename, episodes=1000):
    states = ["INVALID_FORMAT", "UNKNOWN_KEY", "INVALID_TYPE", "EMPTY_VALUE", "VALID"]
    actions = ["Fix", "Delete", "Leave"]
    q_table = np.zeros((len(states), len(actions)))  # Initialize Q table
    
    # Read format rules
    format_dict = {}
    with open(format_filename, "r") as f:
        for line in f:
            key, value_type = line.strip().split("=")
            format_dict[key] = value_type
    
    # Q-learning training
    for _ in range(episodes):
        with open(sample_filename, "r") as f:
            for line in f:
                line = line.strip()
                state = get_state(line, format_dict)
                state_idx = states.index(state)
                
                if random.uniform(0, 1) < EPSILON:
                    action_idx = random.randint(0, len(actions) - 1)  # Random selection (exploration)
                else:
                    action_idx = np.argmax(q_table[state_idx])  # Best known action
                
                reward = 10 if state == "VALID" else -5  # Reward for valid lines, penalty for errors
                next_state_idx = state_idx if state == "VALID" else states.index("VALID")
                
                # Q-table update
                q_table[state_idx, action_idx] = (1 - LEARNING_RATE) * q_table[state_idx, action_idx] + \
                    LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(q_table[next_state_idx]))
    return q_table

# Function to validate user's file
def validate_file(sample_filename, format_filename):
    format_dict = {}
    with open(format_filename, "r") as f:
        for line in f:
            key, value_type = line.strip().split("=")
            format_dict[key] = value_type
    
    with open(sample_filename, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            state = get_state(line, format_dict)
            result = take_action(state, line)
            if "Error" in result:
                print(f"Line {line_no}: {result} -> {line}")  # Print errors to screen

# Program execution
format_filename = "format_rules.txt"
sample_filename = "sample_data.txt"
create_format_file(format_filename)  # Create format rules
create_sample_file(sample_filename)  # Create sample data file
q_learning(sample_filename, format_filename)  # Q-learning training
validate_file(sample_filename, format_filename)  # Validate file and print errors 