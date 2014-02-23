From Data With Love
=====================================

This is a collection of code and utilities that I have developed whilst playing with data without sufficient adult supervision.

Companion code to my tech blog: http://fromdatawithlove.thegovans.us/

Requirements:
-------------------------------------
### Old Faithful
To install within a virtualenv: `pip install -r oldfaithful_requirements.txt`
* numpy
* matplotlib
* scikit-learn
* nose (for unit tests)

### Chess Social
To install within a virtualenv: `pip install -r chess_requirements.txt`
* numpy
* matplotlib
* scipy
* networkx
* nose (for unit tests)
* mock (for unit tests)

To Run the Code:
-------------------------------------
### Old Faithful
`python old_faithful.py ../data/faithful.csv`
There are also other command line options:
* --iterations: Number of iterations for the Gibbs sampler (default 500)
* --save_diagnostics: Whether to save the diagnostic images (default False)
* --output_dir: The directory to save the images to (default '.')
* --burnin: The number of burnin iterations (default 0)

### Chess Social
To download the TWIC chess dataset:
`python twic_scrape.py`
To run using the downloaded TWIC chess dataset:
`python run_community_detection.py path/to/twic_chess_data.pgn`
There are also other command line options:
* --iterations: Number of iterations for the Gibbs sampler (default 100)
* --output_dir: The directory to save the images to (default '.')
* --burnin: The number of burnin iterations (default 0)
* --min_elo: The minimum elo rating for players to be included (default 2500)
* --p_in: The initial value for the 'IN' edge probabilities (default 0.8)
* --p_out: The initial value for the 'OUT' edge probabilities (default 0.2)
