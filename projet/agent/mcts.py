import numpy as np
from pettingzoo.classic import connect_four_v3
from projet.agent.base_agent import Agent



class MCTSNode:
    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.children = []
        self.action = action
        self.visit_count = 0
        self.total_reward = 0
        self.average_reward = 0


class MCTS_Agent(Agent):
    def __init__(self, space, name, agent_name, n_simulations=1000, c_puct=1.0):
        super().__init__(space, name)
        self.agent_name = agent_name
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.root_node = MCTSNode()

    def get_best_action(self, state):
        for _ in range(self.n_simulations):
            leaf_node, leaf_state = self.select(self.root_node, state)
            reward = self.rollout(leaf_state)
            self.backpropagate(leaf_node, reward)

        best_child = self.get_best_child(self.root_node)
        return best_child.action

    def reset(self):
        self.root_node = MCTSNode()

    def select(self, node: MCTSNode, state: np.ndarray) -> tuple:
        env = connect_four_v3.env()
        env.reset()
        env.state = state.copy()

        done = False
        while not done:
            if len(node.children) == 0:
                self.expand(node, env)
            node = self.get_best_child(node)
            _, _, done, _, _ = env.last()  # Call env.last() to get the done value
            if not done:
                env.step(node.action)

        return node, env.state

    def rollout(self, state: np.ndarray) -> float:
        env = connect_four_v3.env()
        env.reset()
        env.state = state.copy()

        done = False
        while not done:
            legal_actions = self.get_legal_actions(env)
            action = np.random.choice(legal_actions)
            env.step(action)
            _, _, done, _, _ = env.last()  # Call env.last() to get the done value

        return self.get_reward(env)

    def backpropagate(self, node: MCTSNode, reward: float) -> None:
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node.average_reward = node.total_reward / node.visit_count
            node = node.parent

    def expand(self, node: MCTSNode, env) -> None:
        legal_actions = self.get_legal_actions(env)
        for action in legal_actions:
            child_node = MCTSNode(parent=node, action=action)
            node.children.append(child_node)

    def get_legal_actions(self, env) -> list:
        action_mask = env.state['action_mask']
        legal_actions = [i for i, is_legal in enumerate(action_mask) if is_legal]
        return legal_actions

    def get_best_child(self, node: MCTSNode) -> MCTSNode:
        return max(node.children, key=lambda child: child.average_reward)

    def get_reward(self, env) -> float:
        _, rewards, done, _, _ = env.last()
        if done:
            return rewards
        return 0
    
    def update(self, obs, action, reward, terminated, next_obs):
        pass
   
    def load(self):
        pass

    def save(self):
        pass