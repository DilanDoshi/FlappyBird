Project Description: This is a object oriented version of the game Flappy Bird implemented with NEAT (NeuroEvolution of Augmenting Topologies) algorithms. Each bird is linked to an individual neural network linked to an overall generation. The algorithm succesfully learns how to play the game by created multiple generations with the "fittest" birds.

To Run: Clone the repository and run FlappBird.py file. You may have to install pygame and neat-pyton libraries. You can do so by typing pip install pygame and pip install neat-python.

Note/Disclaimer: NEAT Algorithm and config-feedforward implemented personally using algorithm and library documentation on NEAT website. Graphics completed with tutorial from YouTube.

NEAT Description: NEAT is an evolutionary algorithm that creates artificial neural networks that evolves both the structure and weights of neural networks to solve complex tasks. NEAT begins by creating a population of simple neural networks, often just an input and an output layer, with minimal connections. Each network (or “genome”) is randomly initialized. Each network is test in the environemnt (ie. bird playing the game) and is assigned a "fitness" score based on its performance (ie. how long the bird survives, and how many pipes it passes). The networks with higher fitness scores are more likely to be selected to “reproduce,” creating the next generation. Poor-performing networks are removed from the pool. Additionally, similar to genetics, crossovers and mutations happen in the creation of new generations. NEAT also clusters similar netwroks into "species" to preserve innovation. This process is then repeated.

More can be found at: https://neat-python.readthedocs.io/en/latest/neat_overview.html# 
