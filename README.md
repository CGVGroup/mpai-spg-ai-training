# mpai-spg-ai-training

In this repository are stored all the code used for training the models for the MPAI-SPG prediction system and the dataset over which the models trained on.

## Dataset
The dataset "Database\GamesDatabased.txt" was created by saving the car's info during  multiples races played by autonomous drivers. Each race was chracterized by 2 laps and only one player.
Every 0.02 seconds a new row was added to the dataset.
The dataset contains the following columns:

1. Players ID
2. Car's Global Position x-axis
3. Car's Global Position z-axis
4. Car's Velocity x-axis
5. Car's Velocity z-axis
6. Car's Rotation y-axis
7. Car's Angular Velocity y-axis
8. Car's Acceleration x-axis
9. Car's Acceleration z-axis
10. Tile ID
11. Tile Ranking (e.g. first, second, last tile of the track)
12. Car's Relative Position x-axis concerning the Tile's Center
13. Car's Relative Position z-axis concerning the Tile's Center
14. Time passed from the start of the race

A new race is indicated by the '_' character. Whenever the "Player ID" has the '_' character at the start of the name, it means a new race has started.
The "Tile ID" field specifies the type of tile (turn left, turn right etc.). Each unique tile is associated with a different integer number that creates the "Tile ID".
The race track can be seen as an array of tiles. "Tile Ranking" correspond to the index of the race track array associated with the tile.


## Analysis

Before creating the input, we transformed the dataset's values into relative ones. By subtracting two rows distant D rows between each other, we evaluate how each columns changes its values after (D+1)*0.02 seconds. The D values is called Discard.
In the file "Notebook\Generators.py" we computes the relative values of each columns except for the columns 10,11,12 and 13. Then, by the means of a python Generator, we evaluate at runtime the input batch for the training and validation set.
The batch is created by collecting Sequence Length (SL) elements each distant D rows from the next one. Then the groundtruth is equal to the row distant D from the last element of the time sequence.
This process is done multiple times, based on the Batch Size parameter.

## Training 
"Notebook\Models.py" contains all the code we used for creating the different models and how the training was implemented.
Lastly the file "NotebookTraining.ipynb" contains the notebook were we can train the different models. By changing the Depths and number of hidden units of the LSTMs and MLP we can train different architectures.
