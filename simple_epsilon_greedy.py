import numpy as np


class Action:
    def __init__(self, action_id):
        self.action_no = action_id
        self.estimated_reward_Q = 0                       # Estimated value of action
        self.step_size_N = 0                              # Step size of action

    def choose_action(self):
        return self.estimated_reward_Q + np.random.randn()

    def update_action(self, actual_reward_R):
        self.step_size_N += 1
        self.estimated_reward_Q = self.estimated_reward_Q + (1/self.step_size_N)*(actual_reward_R - self.estimated_reward_Q)
        
        
def launch(a1, a2, a3, a4, a5, epsilon_, iter_length):

    actions = [Action(a1), Action(a2), Action(a3), Action(a4), Action(a5)]   # Initialize actions

    print("\n---Starting values of actions---")
    for i in range(len(actions)):
        print("Action " + str(actions[i].action_no) + ": " + str(actions[i].estimated_reward_Q))

    print("\n---Experiment---")
    for i in range(iter_length):

        prob_ = np.random.random()
        if prob_ < epsilon_:
            curr_act_A = np.random.choice(len(actions))                       # Explore
        else:
            curr_act_A = np.argmax([a.estimated_reward_Q for a in actions])   # Exploit

        actual_reward_R = actions[curr_act_A].choose_action()
        actions[curr_act_A].update_action(actual_reward_R)
        print("Action: " + str(curr_act_A+1) + "\n" + "Reward: " + str(actual_reward_R) + "\n")

    print("---End values of actions---")
    for i in range(len(actions)):
        print("Action " + str(actions[i].action_no) + ": " + str("{:.8f}".format(actions[i].estimated_reward_Q)) + 
              " - Step Size: " + str(actions[i].step_size_N))


if __name__ == '__main__':
    
    epsilon_ = 0.1
    iter_length = 1000

    launch(1, 2, 3, 4, 5, epsilon_, iter_length)
