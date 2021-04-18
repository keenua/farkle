A bot playing Farkle (a dice game) in the `Kingdom Come: Deliverance`

It uses the (ostensibly) optimal strategy, which is a simplified variation of Markov Decision Process (MDP), see [mdp.py](src/farkle/visual/recognition.py).
The dice position detection is done using `OpenCV` and some image processing / contour detection algorithms exposed by the library, see [detection.py](src/farkle/visual/detection.py).
The dice value recognition is handled by a simple convolutional neural network, see [recognition.py](src/farkle/visual/recognition.py).

# Running locally

The instructions are written for Windows command line, but it's almost identical for bash

- Clone the project
`git clone https://github.com/keenua/farkle`
- Switch to 
- (Optional) Create a virtual environment with Python 3.7 (using `venv` only as an example)
`python -m venv env`
- (Optional) Activate the environment
`.\env\Scripts\activate`
- Install the dependencies
`pip install -r requirements.txt`
- Switch to `src` folder
`cd src`
- Train the MDP model. It'll probably take a lot of time depending on your specs. The output file (`4000_parallel.pkl.npy`) would be ~1GB.
`python -m farkle -a train`
- Run the bot
`python -m farkle`
- Launch the `Kingdom Come: Deliverance` game and sit at the Farkle table. The bot should start printing suggestions as soon as it's your move. Until then it should be showing `Waiting for hero's turn` message

# Issues

If you're having troubles running it or understanding something, just create a github issue in this repo