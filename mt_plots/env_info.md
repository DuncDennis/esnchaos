# Info on how to reproduce the exact environment used to create the MT plots 

To exactly reproduce the masters-thesis plots follow these steps: 

1. Install the conda environment from inside this folder by running: 
    ```bash
   conda env create -f mt_plots_conda_env.yml
    ```

2. Activate the conda environment by running: 
    ```bash
   conda activate esnchaos
    ```

3. From the ``esnchaos`` root directory install ``esnchaos`` without dependencies:
    ```bash
   pip install --no-deps .
    ```
   
4. From the activated ``esnchaos`` environment run the python files under 
``mt_plots`` to recreate the plots. 
