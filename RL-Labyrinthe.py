import numpy as np
import matplotlib.pyplot as plt
import time

# --- 1. Créer le labyrinthe ---
# 0: vide, 1: mur, 2: départ, 3: but
maze = np.array([
    [0, 0, 0, 1, 3],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [2, 0, 0, 0, 0]])

# --- 2. Environnement RL ---
class MazeEnv:
    def __init__(self, maze):
        self.maze = maze.copy()
        self.start_pos = tuple(np.argwhere(maze == 2)[0])
        self.goal_pos = tuple(np.argwhere(maze == 3)[0])
        self.reset()

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos
    
    def step(self, action):
        moves = [(-1,0), (1,0), (0,-1), (0,1)] # haut, bas, gauche, droite
        move = moves[action]
        new_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])
        
        if not (0 <= new_pos[0] < self.maze.shape[0] and 0 <= new_pos[1] < self.maze.shape[1]):
            return self.agent_pos, -1, False

        if self.maze[new_pos] == 1:
            return self.agent_pos, -1, False
        
        self.agent_pos = new_pos

        if self.agent_pos == self.goal_pos:
            return self.agent_pos, 10, True
        
        return self.agent_pos, -0.1, False
    
# --- 3. Q-learning setup ---
env = MazeEnv(maze)
q_table = np.zeros((5, 5, 4)) # états (5x5), 4 actions
alpha = 0.3 # Taux d'apprentissage : Témoin = 0.1
gamma = 0.7 # Facteur de discount : Témoin = 0.9
epsilon = 0.2 # Taux d'exploration : Témoin = 0.2
episodes = 50 # Nombre d'épisodes : Témoin = 10

# --- 4. Entraînement ---
reward_list = []
for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    steps = 0
    done = False
    print(f"\nÉpisode {ep+1}")
    while not done and steps < 100:
        if np.random.rand() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(q_table[state[0], state[1]])

        next_state, reward, done = env.step(action)

        q_old = q_table[state[0], state[1], action]
        q_next = np.max(q_table[next_state[0], next_state[1]])
        q_table[state[0], state[1], action] = q_old + alpha * (reward + gamma * q_next - q_old)
        
        state = next_state
        total_reward += reward
        steps += 1     
        

    print(f" Terminé en {steps} étapes | Reward total = {round(total_reward, 2)}")
    reward_list.append(total_reward)

# --- 5. Politique finale ---
print("\n Politique finale (meilleure action par case) :")
action_map = ['↑', '↓', '←', '→']
policy = np.full((5, 5), ' ')

for i in range(5):
    for j in range(5):
        if maze[i, j] == 1:
            policy[i, j] = '█'
        elif maze[i, j] == 3:
            policy[i, j] = '○'
        elif maze[i, j] == 2:
            policy[i, j] = 'S'
        else:
            best_a = np.argmax(q_table[i, j])
            policy[i, j] = action_map[best_a]
print(policy)

# --- 6. Tracer la courbe Rewards/Episodes ---
fig = plt.figure(figsize=(10, 6))
grid = plt.GridSpec(3, 3, hspace=0.4, wspace=0.3)

# Subplot 1: Rewards/Episodes (2/3 of the height)
ax1 = fig.add_subplot(grid[:, :2])
ax1.plot(reward_list)
ax1.set_title("Récompenses cumulées par épisode")
ax1.set_xlabel("Épisode")
ax1.set_ylabel("Récompense cumulée")
ax1.grid()

# Subplot 2: Politique finale (1/3 of the height)
ax2 = fig.add_subplot(grid[:, 2])
ax2.set_xticks(np.arange(5))
ax2.set_yticks(np.arange(5))
ax2.set_xlim(-0.5, 4.5)
ax2.set_ylim(-0.5, 4.5)
ax2.set_aspect('equal')

for i in range(5):
    for j in range(5):
        ax2.text(j, 4-i, policy[i, j], ha='center', va='center', fontsize=20)

ax2.set_title("Politique finale")
plt.show()



# fig, ax = plt.subplots()
# fig.set_size_inches(5, 5)
# ax.set_xticks(np.arange(5))
# ax.set_yticks(np.arange(5))
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_xlim(-0.5, 4.5)
# ax.set_ylim(-0.5, 4.5)
# ax.set_aspect('equal')

# for i in range(5):
#     for j in range(5):
#             ax.text(j, 4-i, policy[i, j], ha='center', va='center', fontsize=20)
# plt.title("Politique finale")
# plt.show()





