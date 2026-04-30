#!/bin/sh

echo "Host: $(hostname)"                                                                                                                                          
echo "User: $(whoami)"                                                                                                                                            
echo "Working dir: $(pwd)"                                                                                                                                        
echo "HF_HOME: $HF_HOME"                                                                                                                                          
echo "HF_HUB_CACHE: $HF_HUB_CACHE"                                                                                                                                



# Set up environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medjargone-gpu

echo "Python: $(which python)"                                                                                                                                    

export HF_HUB_OFFLINE=1

# Run code
python -m src.run_inference "$@"
