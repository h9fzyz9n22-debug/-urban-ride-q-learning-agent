"""
Q-Learning Intelligent Agent for Urban Ride Demand

Author: Masis Mkrtchian
Class: Intelligent Agents and Reinforcement Learning

The agent learns whether to remain in its current Los Angeles
operating zone or reposition to a neighboring zone. Its objective
is to maximize net reward during a simulated six-hour shift.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------

ZONES = [
    "LAX",
    "Santa Monica",
    "Hollywood",
    "Downtown Los Angeles",
    "Glendale",
    "Burbank",
]

NUMBER_OF_ZONES = len(ZONES)

# Twelve 30-minute periods represent a six-hour shift.
SHIFT_LENGTH = 12

# Expected trip values for every time period and operating zone.
# The values are synthetic and are used only for this simulation.
DEMAND_VALUES = np.array(
    [
        [19, 16, 18, 17, 14, 15],
        [18, 17, 19, 18, 15, 16],
        [17, 18, 21, 20, 16, 17],
        [16, 19, 23, 22, 17, 18],
        [16, 20, 25, 23, 18, 18],
        [17, 21, 24, 24, 19, 19],
        [18, 20, 22, 25, 21, 20],
        [20, 19, 21, 26, 23, 21],
        [23, 18, 20, 24, 25, 23],
        [28, 17, 19, 22, 24, 26],
        [34, 16, 18, 21, 22, 29],
        [38, 15, 17, 20, 21, 31],
    ],
    dtype=float,
)

# A very large value represents an unavailable connection.
UNAVAILABLE = 1_000_000_000

# Estimated cost of remaining in or moving between zones.
MOVEMENT_COSTS = np.array(
    [
        [0, 5, UNAVAILABLE, UNAVAILABLE, UNAVAILABLE, UNAVAILABLE],
        [5, 0, 5, UNAVAILABLE, UNAVAILABLE, UNAVAILABLE],
        [UNAVAILABLE, 5, 0, 4, 4, UNAVAILABLE],
        [UNAVAILABLE, UNAVAILABLE, 4, 0, 4, 6],
        [UNAVAILABLE, UNAVAILABLE, 4, 4, 0, 3],
        [UNAVAILABLE, UNAVAILABLE, UNAVAILABLE, 6, 3, 0],
    ],
    dtype=float,
)


# ---------------------------------------------------------
# Environment functions
# ---------------------------------------------------------

def available_actions(current_zone):
    """Return all zones that the agent can legally select."""

    return np.flatnonzero(
        MOVEMENT_COSTS[current_zone] < UNAVAILABLE
    )


def environment_step(current_zone, time_period, action, random_generator):
    """
    Perform one action and return the outcome.

    The agent receives positive reward when a trip is obtained.
    Movement and unsuccessful waiting reduce the reward.
    """

    movement_cost = MOVEMENT_COSTS[current_zone, action]
    expected_trip_value = DEMAND_VALUES[time_period, action]

    pickup_probability = np.clip(
        0.46 + (expected_trip_value - 15) * 0.017,
        0.45,
        0.86,
    )

    if random_generator.random() < pickup_probability:
        trip_value = max(
            6.0,
            random_generator.normal(expected_trip_value, 4.5),
        )

        reward = trip_value - movement_cost
        trip_completed = 1
    else:
        # The driver receives an idle penalty and still pays
        # the cost of repositioning.
        reward = -3.0 - movement_cost
        trip_completed = 0

    next_zone = action

    return next_zone, reward, trip_completed, movement_cost


# ---------------------------------------------------------
# Q-learning training
# ---------------------------------------------------------

def train_agent(
    seed=7,
    training_episodes=18_000,
    learning_rate=0.12,
    discount_factor=0.93,
):
    """Train the intelligent agent using tabular Q-learning."""

    random_generator = np.random.default_rng(seed)

    q_table = np.zeros(
        (SHIFT_LENGTH, NUMBER_OF_ZONES, NUMBER_OF_ZONES),
        dtype=float,
    )

    training_rewards = []

    for episode in range(training_episodes):
        current_zone = int(
            random_generator.integers(NUMBER_OF_ZONES)
        )

        episode_reward = 0.0

        # Exploration gradually decreases during training.
        epsilon = max(
            0.03,
            0.90 * (1 - episode / training_episodes),
        )

        for time_period in range(SHIFT_LENGTH):
            legal_actions = available_actions(current_zone)

            if random_generator.random() < epsilon:
                action = int(
                    random_generator.choice(legal_actions)
                )
            else:
                legal_q_values = q_table[
                    time_period,
                    current_zone,
                    legal_actions,
                ]

                action = int(
                    legal_actions[np.argmax(legal_q_values)]
                )

            next_zone, reward, _, _ = environment_step(
                current_zone,
                time_period,
                action,
                random_generator,
            )

            if time_period == SHIFT_LENGTH - 1:
                future_value = 0.0
            else:
                next_actions = available_actions(next_zone)

                future_value = np.max(
                    q_table[
                        time_period + 1,
                        next_zone,
                        next_actions,
                    ]
                )

            # Standard Q-learning update.
            temporal_difference = (
                reward
                + discount_factor * future_value
                - q_table[time_period, current_zone, action]
            )

            q_table[time_period, current_zone, action] += (
                learning_rate * temporal_difference
            )

            episode_reward += reward
            current_zone = next_zone

        training_rewards.append(episode_reward)

    return q_table, np.array(training_rewards)


# ---------------------------------------------------------
# Baseline and learned policies
# ---------------------------------------------------------

def select_action(policy, q_table, current_zone, time_period,
                  random_generator):
    """Select an action using one of the three tested policies."""

    legal_actions = available_actions(current_zone)

    if policy == "Q-Learning":
        q_values = q_table[
            time_period,
            current_zone,
            legal_actions,
        ]

        return int(legal_actions[np.argmax(q_values)])

    if policy == "Random":
        return int(random_generator.choice(legal_actions))

    if policy == "Greedy":
        immediate_values = []

        for action in legal_actions:
            expected_trip_value = DEMAND_VALUES[
                time_period,
                action,
            ]

            pickup_probability = np.clip(
                0.46 + (expected_trip_value - 15) * 0.017,
                0.45,
                0.86,
            )

            expected_reward = (
                pickup_probability * expected_trip_value
                + (1 - pickup_probability) * -3.0
                - MOVEMENT_COSTS[current_zone, action]
            )

            immediate_values.append(expected_reward)

        return int(
            legal_actions[np.argmax(immediate_values)]
        )

    raise ValueError("Unknown policy: " + policy)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_policy(
    q_table,
    policy,
    seed=2026,
    evaluation_episodes=1_000,
):
    """Evaluate a policy over previously unseen episodes."""

    random_generator = np.random.default_rng(seed)

    total_rewards = []
    completed_trips = []
    total_movement_costs = []

    for _ in range(evaluation_episodes):
        current_zone = int(
            random_generator.integers(NUMBER_OF_ZONES)
        )

        episode_reward = 0.0
        episode_trips = 0
        episode_movement_cost = 0.0

        for time_period in range(SHIFT_LENGTH):
            action = select_action(
                policy,
                q_table,
                current_zone,
                time_period,
                random_generator,
            )

            current_zone, reward, trip, movement_cost = (
                environment_step(
                    current_zone,
                    time_period,
                    action,
                    random_generator,
                )
            )

            episode_reward += reward
            episode_trips += trip
            episode_movement_cost += movement_cost

        total_rewards.append(episode_reward)
        completed_trips.append(episode_trips)
        total_movement_costs.append(
            episode_movement_cost
        )

    total_rewards = np.array(total_rewards)

    return {
        "Mean Reward": total_rewards.mean(),
        "Reward Standard Deviation": total_rewards.std(ddof=1),
        "Success Rate": np.mean(total_rewards >= 125.0),
        "Average Trips": np.mean(completed_trips),
        "Average Movement Cost": np.mean(total_movement_costs),
    }


# ---------------------------------------------------------
# Results and visualization
# ---------------------------------------------------------

def create_training_graph(training_rewards):
    """Create and save the training-progress graph."""

    moving_average_window = 500

    moving_average = np.convolve(
        training_rewards,
        np.ones(moving_average_window) / moving_average_window,
        mode="valid",
    )

    episode_numbers = np.arange(
        moving_average_window,
        len(training_rewards) + 1,
    )

    plt.figure(figsize=(9, 5))

    plt.plot(
        episode_numbers,
        moving_average,
        color="#1F4E79",
        linewidth=1.8,
    )

    plt.title(
        "Learning Progress Using a 500-Episode Moving Average"
    )
    plt.xlabel("Training Episode")
    plt.ylabel("Mean Net Reward")
    plt.grid(axis="y", alpha=0.35)
    plt.tight_layout()

    plt.savefig(
        "training_progress.png",
        dpi=200,
    )

    plt.show()


def print_results(results):
    """Print evaluation results in a readable format."""

    print("\nPOLICY EVALUATION RESULTS")
    print("-" * 72)

    header = (
        f"{'Policy':<14}"
        f"{'Reward':>11}"
        f"{'SD':>10}"
        f"{'Success':>11}"
        f"{'Trips':>10}"
        f"{'Move Cost':>13}"
    )

    print(header)
    print("-" * 72)

    for policy, metrics in results.items():
        print(
            f"{policy:<14}"
            f"{metrics['Mean Reward']:>11.2f}"
            f"{metrics['Reward Standard Deviation']:>10.2f}"
            f"{metrics['Success Rate'] * 100:>10.1f}%"
            f"{metrics['Average Trips']:>10.2f}"
            f"{metrics['Average Movement Cost']:>13.2f}"
        )

    print("-" * 72)


# ---------------------------------------------------------
# Run the project
# ---------------------------------------------------------

def main():
    print("Training the Q-learning agent...")

    q_table, training_rewards = train_agent()

    first_training_average = np.mean(
        training_rewards[:1_000]
    )

    final_training_average = np.mean(
        training_rewards[-1_000:]
    )

    print(
        f"Average reward during first 1,000 episodes: "
        f"{first_training_average:.2f}"
    )

    print(
        f"Average reward during final 1,000 episodes: "
        f"{final_training_average:.2f}"
    )

    results = {}

    for policy in ["Q-Learning", "Greedy", "Random"]:
        results[policy] = evaluate_policy(
            q_table,
            policy,
        )

    print_results(results)
    create_training_graph(training_rewards)


if __name__ == "__main__":
    main()
