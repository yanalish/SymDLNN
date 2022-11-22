# SymDLNN
Code base acompanying the paper Discrete Lagrangian Neural Networks with Automatic Symmetry Discovery

## Prerequisites

In order to run the code we require several packages, conda users can use the `environment.yml` to generate the necessary virtual environment from terminal

Use the terminal or an Anaconda Prompt (for windows users) for the following steps:

1. Create the environment from the environment.yml file:

```bash
conda env create -f environment.yml
```
The first line of the yml file sets the new environment's name. This process can take a while despite the number of dependencies being low, because of the dependency checker in conda. Go grab a coffee.

2. Activate the new environment once environment is created: 

```bash
conda activate symdlnn
```

3. This is your own environment, feel free to install any other dependencies

## Citation

If you use any of the code for your own projects, please consider citing
```
@misc{lishkova2022symdlnn,
      title={Discrete Lagrangian Neural Networks with Automatic Symmetry Discovery}, 
      author={Yana Lishkova and Paul Scherer and Steffen Ridderbusch and Mateja Jamnik and Pietro Liò and Sina Ober-Blöbaum and Christian Offen},
      year={2022},
      eprint={2211.10830},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```