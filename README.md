# Agent Implementations RL
Here I am implementing the agents as I read through them in the Sutton and Bartow book: Reinforcement Learning An Introduction 2nd edition. Unless stated otherwise these are trained to find q* using &epsilon;-greedy policy then evaluated using a greedy policy on the q* obtained through training. 

## Agents Implemented:

#### Chapter 5 - Monte Carlo Methods
- On-policy first-visit Monte Carlo Control
- Off-policy Monte Carlo Control

#### Chapter 6 - Temporal-Difference Learning
- SARSA(0)
- Q-Learning
- Expected SARSA
- Double Q-Learning

#### Chapter 7 - n-step Bootstrapping
- n-step SARSA
- Off-Policy n-step SARSA

#### Chapter 12 - Eligibility Traces
- SARSA(&lambda;)




## Testing:
I am testing each of these using different gymnasium environments.