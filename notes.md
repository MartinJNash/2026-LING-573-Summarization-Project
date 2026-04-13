Tried to run BART with "summarization" type, but new versions of the library removed "summarization" and there are new recommendations.

https://discuss.huggingface.co/t/summarization-task-is-not-recognized-in-pipeline/173156/2

---

We might need to install a different version of pytorch which enables CUDA to use on GPU machines.

---

Create a copy of the conda environment

`conda env create -f environment.yml`

Use it:

`conda activate medjargone`
