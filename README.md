# Canabalt-ML
Using a neural network to play Canabalt ( a 1 button endless runner) from player examples.

requirements:
  - windows 
  - canabalt in windowed mode, 800x600 in top left corner of screen
  - python 3.6
  - numpy
  - pandas
  - opencv3
  - pywin32
  - sklearn
  
there are 2 neural networks running, one to play the game, and one to detect game over situations. 
in the current game over detection there is a specific issue where it detects a game over when the player is too high over the landscape so no landscape can be seen. data for this neural network was gathered simply by holding space while game over to mark the data, using the normal data gathering script.

the neural network to play the game is built from data of me playing the game, and some extra data gathering done by the script playing the game. currently the neural network plays slightly worse than i do myself, but manages to get reasonably decent scores. 
please note that i am not great at this game either. If someone better at this game would like to play and record data (at least 100k samples) i would love to see the improvement.

all of this was developed using anaconda.
  
