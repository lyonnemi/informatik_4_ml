import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt



def initialize_q_table(state_space, action_space):

    # Create and return the Q-table.
    # TODO:
    # - Initialize a Q-table of shape (state_space, action_space)
    # - Start with all values equal to zero

    q_table = np.zeros((state_space, action_space))
    return q_table

def choose_action(state, q_table, epsilon, action_space, training=True):

    ## Select an action using the epsilon-greedy strategy.
    ## TODO
    ## - if training:
    ##     * with probability epsilon -> choose a random action (exploration)
    ##     * otherwise -> choose the best action from the Q-table (exploitation)
    ## - if testing:
    ##     * always choose the best action from the Q-table
    ##
    if training:
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, action_space - 1)  # exploration
        else:
            action = np.argmax(q_table[state, :])          # exploitation
    else:
        action = np.argmax(q_table[state, :])              # always exploit during testing
    return action



def compute_discounted_return(rewards, gamma):
    G = 0.0
    for t, reward in enumerate(rewards):
        G += (gamma ** t) * reward
    return G


def update_q_table(q_table, state, action, reward, next_state, alpha, gamma, done):

    # Update the Q-table using the Q-learning rule.

    # Bellman-based Q-learning update:
    #     Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

    # If the episode is done, there is no future reward term.

    # TODO:
    # - Implement the Q-learning update rule
    if done:
        future_q = 0.0
    else:
        future_q = gamma * np.max(q_table[next_state, :])

    q_table[state, action] += alpha * (reward + future_q - q_table[state, action])


def run_q_learning(
    env_name="Taxi-v3",
    train_episodes=4000,
    test_episodes=1000,
    max_steps=100,
    alpha=0.35,
    gamma=0.75,
    epsilon=1.0,
    epsilon_min=0.001,
    epsilon_decay=0.9999
):

    env = gym.make(env_name)

    state_space = env.observation_space.n
    action_space = env.action_space.n

    q_table = initialize_q_table(state_space, action_space)

    total_episodes = train_episodes + test_episodes

    episode_rewards = []
    episode_steps = []
    episode_returns = []

    for episode in range(total_episodes):
        state, info = env.reset()
        training = episode < train_episodes

        total_reward = 0
        rewards_this_episode = []

        for step in range(max_steps):

            action = choose_action(
                state=state,
                q_table=q_table,
                epsilon=epsilon,
                action_space=action_space,
                training=training
            )

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            rewards_this_episode.append(reward)

            if training:
                update_q_table(
                    q_table=q_table,
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    alpha=alpha,
                    gamma=gamma,
                    done=done
                )

            # Move to next state
            state = next_state

            if done:
                break

        # Store episode statistics
        episode_rewards.append(total_reward)
        episode_steps.append(step + 1)


        # Compute the discounted return of the episode
        G = compute_discounted_return(rewards_this_episode, gamma)
        episode_returns.append(G)

        # Reduce epsilon after each training episode
        if training:
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

    env.close()

    return q_table, episode_rewards, episode_steps, episode_returns


def plot_results(episode_rewards, episode_steps, episode_returns, train_episodes):
    episodes = np.arange(len(episode_rewards))

    plt.figure(figsize=(8, 4))
    plt.plot(episodes, episode_rewards)
    plt.axvline(x=train_episodes, color='r', linestyle='--', label='Start of testing')
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Reward per episode")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(episodes, episode_steps)
    plt.axvline(x=train_episodes, color='r', linestyle='--', label='Start of testing')
    plt.xlabel("Episode")
    plt.ylabel("Number of steps")
    plt.title("Steps per episode")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(episodes, episode_returns)
    plt.axvline(x=train_episodes, color='r', linestyle='--', label='Start of testing')
    plt.xlabel("Episode")
    plt.ylabel("Discounted return")
    plt.title("Discounted return per episode")
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_demo(q_table, env_name="Taxi-v3", max_steps=100):

    env = gym.make(env_name, render_mode="human")
    state, info = env.reset()

    total_reward = 0

    for step in range(max_steps):
        action = np.argmax(q_table[state, :])
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        state = next_state

        if done:
            break

    env.close()
    print("Demo episode reward:", total_reward)



def main():
    q_table, rewards, steps, returns = run_q_learning()
    plot_results(rewards, steps, returns, train_episodes=4000)

    # Optional visual demonstration.
    # Uncomment only if it works on your system.
    # run_demo(q_table)


if __name__ == "__main__":
    main()