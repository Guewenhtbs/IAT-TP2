import os
import cv2
import numpy as np
import random
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Charger le dataset depuis Kaggle
def load_dataset(folder_path, label):
    dataset = []
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Convertir en niveaux de gris
        img = cv2.resize(img, (64, 64)) # Redimensionner
        features = extract_hog_features(img) # Extraire les caractéristiques HOG
        dataset.append({'features': features, 'label': label})
    return dataset
    
# Extraction des descripteurs HOG
def extract_hog_features(image): 
    features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9, visualize=True, block_norm='L2-Hys')
    return features

# Charger les images des deux classes
vehicle_images = load_dataset("C:/Users/kergu/Documents/INSA/4TC/IAT/TP2/data/train/vehicles", label=1)
non_vehicle_images = load_dataset("C:/Users/kergu/Documents/INSA/4TC/IAT/TP2/data/train/non-vehicles", label=0)
dataset = vehicle_images + non_vehicle_images

test_vehicle_images = load_dataset("C:/Users/kergu/Documents/INSA/4TC/IAT/TP2/data/test/vehicles", label=1)
test_non_vehicle_images = load_dataset("C:/Users/kergu/Documents/INSA/4TC/IAT/TP2/data/test/non-vehicles", label=0)
test_dataset = test_vehicle_images + test_non_vehicle_images

# Normaliser les caractéristiques
scaler = StandardScaler()
features_matrix = np.array([data['features'] for data in dataset])
scaled_features = scaler.fit_transform(features_matrix)
test_features_matrix = np.array([data['features'] for data in test_dataset])
scaled_test_features = scaler.transform(test_features_matrix)

# Mettre à jour le dataset avec les caractéristiques normalisées
for i, data in enumerate(dataset): dataset[i]['features'] = tuple(scaled_features[i])
for i, data in enumerate(test_dataset): test_dataset[i]['features'] = tuple(scaled_test_features[i])
# Convertir en tuple pour l'utiliser comme clé dans la Q-Table
# Définition de l'environnement

class Environment:
    def __init__(self, dataset):
        self.dataset = dataset
        self.current_index = 0
    def reset(self):
        self.current_index = 0
        return self.dataset[self.current_index]['features']
    def step(self, action):
        reward = 1 if action == self.dataset[self.current_index]['label'] else -1
        self.current_index += 1
        done = self.current_index >= len(self.dataset)
        next_state = self.dataset[self.current_index]['features'] if not done else None
        return next_state, reward, done
    
# Agent Q-Learning
class QLearningAgent:
    def __init__(self, action_space):
        self.q_table = {}
        self.action_space = action_space
        self.learning_rate = 0.1 # Taux d'apprentissage : Témoin = 0.1
        self.gamma = 0.7 # Facteur de discount : Témoin = 0.9   
    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space)
        return np.argmax(self.q_table[state]) if random.random() > 0.1 else random.randint(0, self.action_space - 1)
    def update_q_value(self, state, action, reward, next_state):
        if next_state is None:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table.get(next_state,
        np.zeros(self.action_space)))
        self.q_table[state][action] += self.learning_rate * (target -
        self.q_table[state][action])

# Initialiser les listes pour stocker les rewards et l'accuracy
reward_per_episode = []
accuracy_per_episode = []

# Simulation
env = Environment(dataset)
agent = QLearningAgent(action_space=2)
episodes = 10 # Nombre d'épisodes : Témoin = 10
for episode in range(episodes):  # Nombre d'épisodes : Témoin = 10
    state = env.reset()
    done = False
    total_reward = 0
    correct_predictions = 0
    total_predictions = 0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update_q_value(state, action, reward, next_state)
        state = next_state

        total_reward += reward
        total_predictions += 1
        if action == env.dataset[env.current_index-1]["label"] :  # Si l'action est correcte
            correct_predictions += 1

    # Calculer l'accuracy pour cet épisode
    accuracy = correct_predictions / total_predictions
    accuracy_per_episode.append(accuracy)
    reward_per_episode.append(total_reward)

    print(f"Épisode {episode + 1}: Reward = {total_reward}, Accuracy = {accuracy:.2f}")
# Afficher la Q-Table (partielle)
# print("Q-Table (partielle):", list(agent.q_table.items())[:5])

# Test de l'agent sur le dataset de test
env_test = Environment(test_dataset)
test_correct_predictions = 0
test_total_predictions = 0
test_state = env_test.reset()
test_done = False
test_total_reward = 0
while not test_done:
    action = agent.choose_action(test_state)
    next_state, reward, test_done = env_test.step(action)
    test_state = next_state
    test_total_predictions += 1
    test_total_reward += reward
    if action == env_test.dataset[env_test.current_index-1]["label"] :  # Si l'action est correcte
        test_correct_predictions += 1

accuracy_test = test_correct_predictions / test_total_predictions
print(f"Test: Reward = {test_total_reward}, Accuracy = {accuracy_test:.2f}")

test_total_reward_list = [test_total_reward] * episodes
test_accuracy_list = [accuracy_test] * episodes

# Tracer les courbes
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Courbe des rewards par épisode
axes[0].plot(reward_per_episode, label="train")
axes[0].plot(test_total_reward_list, label="test", linestyle='--', color="red")   
axes[0].set_title("Reward par épisode")
axes[0].set_xlabel("Épisode")
axes[0].set_ylabel("Reward total")
axes[0].grid()
axes[0].legend()

# Courbe de l'accuracy par épisode
axes[1].plot(accuracy_per_episode, label="train", color="orange")
axes[1].plot(test_accuracy_list, label="test", linestyle='--', color="red")
axes[1].set_title("Accuracy par épisode")
axes[1].set_xlabel("Épisode")
axes[1].set_ylabel("Accuracy")
axes[1].grid()
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"Accuracy finale : {accuracy_per_episode[-1]:.2f}, Reward total : {reward_per_episode[-1]}")

