# iTRADE: **i**mage-based **TRA**nsfer learning for **D**rug **E**ffects
<!-- Thanks to https://acronymify.com! -->

This repository holds code supporting _Berker et al. (2022) IEEE Trans Med Imaging_, <https://doi.org/10.1109/TMI.2022.3205554>. It uses the `iTRADE-Data` dataset published on Zenodo (<https://doi.org/10.5281/zenodo.5885481>) from where it is downloaded when required.

## Prerequisites

- A computer equipped with a Graphical Processing Unit (GPU). We used a Linux workstation (openSUSE Leap 15.2) with a 10-core i9-7900X (Intel) processor, 32 gigabytes of random access memory (RAM), and a Quadro RTX 6000 (NVIDIA) GPU with 24 gigabytes of memory. (A GeForce RTX 2060 (NVIDIA) GPU with 6 gigabytes of memory is also sufficient.)
- An installation of Python (≥ v3.10; we used v3.10.4) including the `pip` package manager (see [python.org](https://www.python.org/downloads/)).
- Everything required to install TensorFlow (≥ v2.8; we used v2.8.0) with GPU support, such as CUDA, CUPTI, and cuDNN (see [tensorflow.org](https://www.tensorflow.org/install/gpu) for details).
- Depending on the desired postprocessing steps, additional dependencies as listed [below](#postprocessing-prerequisites).
- Note: basic operation of the code has also been tested on Windows 10, and it might also work on macOS. For v1.0.1, CNN training has also been tested on Windows 11 using Python v3.10.11 and TensorFlow v2.10.1, as well as Python v3.11.3 and TensorFlow v2.12.0 and v2.13.0rc1 (without GPU support).

## Installation

On the terminal, run

```shell
pip install https://github.com/yannickberker/iTRADE/archive/main.zip
```

For GPU support on Windows (compare [tensorflow.org](https://www.tensorflow.org/install/pip#windows-native)), you may need to use Python v3.10 and run

```shell
pip install https://github.com/yannickberker/iTRADE/archive/main.zip "tensorflow<2.11"
```

## Testing GPU access

To verify that TensorFlow can access the GPU, run

```shell
itrade-test-gpu
```

## Testing CNN operations

To verify that CNN training works, run

```shell
itrade-run-cnns --fast-try
```

## Running CNN training

CNN training can be started using

```shell
itrade-run-cnns
```

If that does not work, one may try

```shell
python -m itrade.run_cnns
```

or the Python prompt:

```python
from itrade.run_all import run_cnns

run_cnns()
```

## Running postprocessing

Individual postprocessing steps (each depending on the result of the previous step, respectively) can be started using, `itrade-run-itrex` (driving iTReX), and `itrade-run-plots` (generating plots), respectively.

### Postprocessing prerequisites

- `itrade-run-itrex`: driving iTReX for drug-sensitivity scoring requires an installation of Chrome/Chromium ([chromium.org](https://www.chromium.org)) and a compatible version of ChromeDriver ([chromedriver.chromium.org](https://chromedriver.chromium.org/downloads)).
- `itrade-run-plots`: generating plots requires an installation of R (v4.1, with Cairo support; [cran.r-project.org](https://cran.r-project.org)) and `perl` ([perl.org](https://www.perl.org/get.html)). You will also need to install the package with the `R` set of extra dependencies:

   ```shell
   pip install "iTRADE[R] @ https://github.com/yannickberker/iTRADE/archive/main.zip"
   ```

## Running CNN training and postprocessing

For a full run of all training and postprocessing steps (including driving iTReX and generating plots - see requirements [above](#postprocessing-prerequisites)!), issue

```shell
itrade-run-all
```

This may take many hours, so on a remote Linux workstation with `nohup`, `bash` and `ts`, one may prefer

```shell
nohup bash -c 'itrade-run-all 0< /dev/null \
    1> >(ts "%F %.T" > itrade-run-all.out.log) \
    2> >(ts "%F %.T" > itrade-run-all.err.log)' \
    0< /dev/null 1> itrade-run-all.nohup.log 2>&1 &
```

and check `itrade-run-all.err.log` for errors.

## Visualizing CNN training

While the experiments are running, progress can be viewed using TensorBoard:

```shell
tensorboard --logdir itrade-results/Board --reload_multifile=true --reload_multifile_inactive_secs 3600
```

Then, launch <http://localhost:6006> in your web browser to visualize live training progress. Useful search expressions include:

- Comparison of different replicates of a cell line (e.g., BT-40):
  - [BT-40.*/train](http://localhost:6006/#scalars&tagFilter=epoch_loss&regexInput=BT-40.*%2Ftrain)
  - [BT-40.*/val](http://localhost:6006/#scalars&tagFilter=epoch_loss&regexInput=BT-40.*%2Fval)

- Impact of transfer learning:
  - [/TransferStudy.*/val](http://localhost:6006/#scalars&tagFilter=epoch_loss&regexInput=Transfer.*val)

- Different network structures:
  - [/(StructureStudy-)?Phase1-INF_R_153_CE.*/val](http://localhost:6006/#scalars&tagFilter=epoch_auc&regexInput=%2F(StructureStudy-)%3FPhase1-INF_R_153_CE.*/val)
  - [/(StructureStudy-)?Phase2-INF_R_153_V2_DS1.*/val](http://localhost:6006/#scalars&tagFilter=epoch_auc&regexInput=%2F(StructureStudy-)%3FPhase2-INF_R_153_V2_DS1.*/val)

## Running your own experiments

You can run individual CNN trainings using the following commands:

- On the shell:

  ```shell
  itrade-run-cnn
  ```
  
  ```shell
  python -m itrade.run_cnn
  ```

- On the Python prompt:

  ```python
  from itrade.run_cnn import run_cnn
  
  run_cnn()
  ```

All variants will print detailed usage information - check [run_all.py](./itrade/run_all.py) for further inspiration.

## Further development

You may browse the code using the github.dev web-based editor at <https://github.dev/yannickberker/iTRADE>.

To continue development using [Visual Studio Code](https://code.visualstudio.com/) and the [`git` command-line client](https://git-scm.com/downloads):

```shell
git clone https://github.com/yannickberker/iTRADE
pip install -e iTRADE[dev] # and maybe [R,dev], see above
code iTRADE
```
