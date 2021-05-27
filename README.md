# Finetune GPT2 model to generate Ruby code
## Steps to run:
* generate API access token for GitHub API
* Save the access token in a file called token in the same folder as the python code
* Run the RepoCLone.py (you can adjust how many days to search, this will affect how much code it downloads)
* Run the PrepData.py file to convert the downloaded code to a txt file to use for training
* Run the Tokenizer.py file (with FIRST_RUN = True and RUN_TRAINER = False )  to train the Tokenizer and save it (should be very fast no GPU needed)
* Run the Tokenizer.py file (with FIRST_RUN = False and RUN_TRAINER = True )  to train GPT2 model this takes time and require GPU (depending on the number of 
  samples, for the values i used in the files it took roughly 18 hours on my RTX 3090 GPU) 
