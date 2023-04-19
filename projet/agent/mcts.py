import numpy as np
import random
from typing import List
from projet.agent.base_agent import Agent


class MCTSNode:
    def __init__(self, parent=None, action=None, reward=0.0):
        self.parent = parent
        self.children = []
        self.action = action
        self.reward = reward
        self.visits = 0
        self.total_reward = 0.0

    def add_child(self, child):
        self.children.append(child)


class MCTS(Agent):
    def __init__(self, action_space, agent, num_simulations=500, exploration_constant=1.0, **kwargs):
        super().__init__(action_space, agent, **kwargs)
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant

    def get_best_action(self, obs: dict) -> int:
        root = MCTSNode()

        for _ in range(self.num_simulations):
            selected_node = self.tree_policy(root, obs)
            reward, next_obs, terminated = self.simulate(selected_node, obs)
            self.backpropagate(selected_node, reward)

        best_child = self.best_child(root, 0.0)
        return best_child.action

    def tree_policy(self, root: MCTSNode, obs: dict) -> MCTSNode:
        current_node = root

        while not self.is_leaf_node(current_node, obs):
            if self.has_unexplored_actions(current_node, obs):
                return self.expand(current_node, obs)
            else:
                current_node = self.best_child(current_node, self.exploration_constant)

        return current_node

    def expand(self, node: MCTSNode, obs: dict) -> MCTSNode:
        available_actions = self.get_available_actions(obs)
        action = random.choice(available_actions)
        child = MCTSNode(parent=node, action=action)
        node.add_child(child)
        return child

    def simulate(self, node: MCTSNode, obs: dict) -> tuple:
        # Implement simulation logic 
        # need to define how the simulation will be performed based on the problem
        # Return the reward, next observation, and a boolean indicating if the simulation has terminated
        pass

    def backpropagate(self, node: MCTSNode, reward: float) -> None:
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def is_leaf_node(self, node: MCTSNode, obs: dict) -> bool:
        # Define  leaf node condition here based on the problem
        pass

    def has_unexplored_actions(self, node: MCTSNode, obs: dict) -> bool:
        available_actions = self.get_available_actions(obs)
        return len(available_actions) > len(node.children)

    def get_available_actions(self, obs):
        action_mask = obs['action_mask']
        available_actions = [i for i, is_valid in enumerate(action_mask) if is_valid]
        return available_actions

    def best_child(self, node: MCTSNode, c: float) -> MCTSNode:
        def uct_score(n: MCTSNode) -> float:
            if n.visits == 0:
                return float('inf')
            exploitation = n.total_reward / n.visits
            exploration = c * np.sqrt(np.log(node.visits) / n.visits)
            return exploitation + exploration

        return max(node.children.values(), key=uct_score)
    
    def load(self):
        pass

    def reset(self):
        pass

    def save(self):
        pass

    def update(self, last_observation, action, reward, termination, observation):
        pass