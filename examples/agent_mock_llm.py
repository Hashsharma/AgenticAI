import random

# ----- Functions (capabilities/tools) -----
def func_a():
    return "Output A"

def func_b():
    return "Output B"

def func_c():
    return "Output C"

def func_d():
    return "Output D"

def func_e():
    return "Output E"

# ----- Mock LLM -----
def mock_llm_observe(observation):
    """
    Simulates what an LLM would decide.
    Returns the name of the function to call next or 'FINISH'.
    """
    # For demonstration, randomly choose a function or finish
    options = ["func_a", "func_b", "func_c", "func_d", "func_e", "FINISH"]
    return random.choice(options)


class Agent:
    def __init__(self, goal, functions):
        self.goal = goal
        self.functions = functions
        self.memory = []
        self.done = False
    
    def run(self):
        print(f'Agent started with goal: {self.goal}')
        observation = None

        while not self.done:
            # Deciding next action (mock LLM)
            action_name = mock_llm_observe(observation)
            if action_name == 'FINISH':
                self.done = True
                print("Agent finished")
                break

            # 2. Execute the Function

            func = self.functions[action_name]
            output = func()

            # 3. Storing the Observation
            self.memory.append((action_name, output))
            observation = output

            print(f"Agent chose {action_name}, output = {output} === Agent Memory {self.memory}")


# ---- Running the Agent

if __name__ == "__main__":
    functions = {
        "func_a": func_a,
        "func_b": func_b,
        "func_c": func_c,
        "func_d": func_d,
        "func_e": func_e,
    }

    agent = Agent("Agentic AI with single agent using MOCK LLM", functions)
    agent.run()