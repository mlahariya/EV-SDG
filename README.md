# EV-SDG
Electric vehicle (EV) - Synthetic data generator(SDG): Parametric models to generate synthetic samples of EV sessions data.
 
This repository has
(i) Trained models that reflect real world dataset, which can be used to generate a synthetic sample of EV sessions, and
(ii) Code to train SDG models from new data.

## References
<a id="1">[1]</a> 
[Lahariya, Manu and Benoit, Dries and Develder, Chris (2020). 
Defining a Synthetic Data Generator for Realistic Electric Vehicle Charging Sessions](https://dl.acm.org/doi/10.1145/3396851.3403509). 
Proceedings of the Eleventh ACM International Conference on Future Energy Systems, pp. 406–407.

<a id="1">[2]</a> 
[Lahariya, Manu and Benoit, Dries and Develder, Chris (2020). 
Synthetic data generator for electric vehicle charging sessions: Modeling and evaluation using real-world data.](https://www.mdpi.com/1996-1073/13/16/4211)
Energies 2020, vol. 13.

#### Citation
```
@inproceedings{10.1145/3396851.3403509,
author = {Lahariya, Manu and Benoit, Dries and Develder, Chris},
    title = {Defining a Synthetic Data Generator for Realistic Electric Vehicle Charging Sessions},
    year = {2020},
    publisher = {Association for Computing Machinery},
    doi = {10.1145/3396851.3403509},
    booktitle = {Proceedings of the Eleventh ACM International Conference on Future Energy Systems},
    pages = {406–407}
}

@article{lahariya2020synthetic,
    title = {Synthetic data generator for electric vehicle charging sessions: Modeling and evaluation using real-world data},
    journal = {Energies},
    author = {Lahariya, Manu and Benoit, Dries and Develder, Chris},
    volume={13},
    number={16},
    pages={4211},
    year={2020},
    doi = {10.3390/en13164211},
    publisher={Multidisciplinary Digital Publishing Institute}
}
```
<img src="https://idlab.technology/assets/img/logo.jpg" width="100"> <img src="https://styleguide.ugent.be/files/uploads/logo_UGent_EN_RGB_2400_kleur_witbg.png" width="50">

Developed at [UGent, IDlab](https://www.ugent.be/ea/idlab/en). 



### EV sessions data generation
A sample of EV sessions data can be generated using the python script `SDG_sample_generate.py`. Generate_sample() can be used 
to generate and save samples to /res/generated samples. 


```python
from modeling.generate_sample import generate_sample

AM,MMc,MMe = SDG[0],SDG[1],SDG[2]   # SDG = synthetic data generator

generate_sample(AM=AM, MMc=MMc, MMe=MMe,
                horizon_start=horizon_start,horizon_end=horizon_end)
```

AM, MMc and MMe are the SDG models that needs to be passed as inputs. 
Models fitted on a real world data will be used as default SDG models for generation.
These models are saved in [modeling/default_models/saved_models](https://github.com/mlahariya/EV-SDG/tree/master/modeling/default_models/saved_models). 
Generated EV sessions data will be saved on the res/generated_samples folder. Please refer to  `SDG_sample_generate.py` for further details. 

##### Command line arguments for SDG_sample_generate.py

```
optional arguments:
  -h, --help            Show this help message and exit
  -start_date START_DATE
                        First date of the horizon for data generation format:
                        dd/mm/YYYY.
  -end_date END_DATE    Last date of the horizon for data generation format:
                        dd/mm/YYYY.
  -use USE              Which kind of models to use. "default" for using the
                        default models "latest" for using the lastest trained
                        models.
  -model MODEL          Modeling method to be used for modeling arrival times:
                        * AC for arrival count models;
                        * IAT for inter-arrival time models.
  -lambdamod LAMBDAMOD  Method to be used for modeling lambda:
                        * AC: has two options, poisson_fit/neg_bio_reg;
                        * IAT: has three options, mean/loess/poly.
  -verbose VERBOSE      0 to print nothing; > 0 values for printing more
                        information. Possible values: 0, 1, 2, 3. (integer)

```


Default model specifications:

* SDG:
    * AM: Arrival models can be either inter-arrival time models (IAT) or arrival count (AC) models.
        * IAT models: possible modeling methods for lambda are (i) mean, (ii) poly, and (iii) loess.
        * AC models: possible modeling methods for lambda are (i) poisson_fit (ii) neg_bio_reg
        
    * MMc: Connected time mixture model.
        * Mixtures = GMMs
        * Optimization = EM (MLE)
    * MMe: Energy required mixture model.
        * Mixtures = GMMs
        * Optimization = EM (MLE)

Default models for all possible methods of modeling for AM are provided and can be used.

### Model fitting

To train the models, a (real world) dataset is required. Following columns are necessary in this EV sessions data.

Column name | Description | Data format
--- | --- | ---
Started | Starting date and time of the EV session | datetime (dd/mm/YYY HH:MM:SS)
ConnectedTime | Connection time of the EV session | Hours (float)
TotalEnergy | Requested energy of the EV session | kWh (float)
ChargePoint | Charging station | Categorical (str) 

#### Preprocessing

We clean the data and prepare preprocessed data using [preprocess](preprocess) module. 
The command line script `SDG_preprocessing.py` can be used to create the preprocessed datasets. 
Running this script will generate a 'slotted data' and a 'preprocessed data' file. 

* Save the raw data file in a folder
* Run `SDG_preprocessing.py` (supporting module `./preprocess` )
* Please see /res/preprocess for generated plots and pre processed data.

##### Command line arguments for SDG_preprocessing.py

```
optional arguments:
  -h, --help            Show this help message and exit
  -Year YEAR            Year for modeling (integer)
  -Slotmins SLOTMINS    Minutes in each timeslot (integer)
  -create_plots CREATE_PLOTS
                        Indicator for creating plots
  -Sessions_filename SESSIONS_FILENAME
                        Name of the file contaning raw data. This file must be
                        present in the res_folder (str)
  -res_folder RES_FOLDER
                        Location for raw data file. Default is "./res". Inside
                        this directory EV session files must be present. 
                        (string)
  -verbose VERBOSE      0 to print nothing; > 0 values for printing more
                        information. Possible values: 0, 1, 2, 3. (integer)
```
 
NOTE: Please do not forget to give the file name and file location while calling this script

NOTE: This file should be run before `./SDG_fit.py` (used for training models)

NOTE: Don't forget to install the packages in requirements.txt 
(`pip install -r requirements.txt`)
 
#### Training SDG models

`SDG_fit.py` can be used to fit the SDG models. 

##### Command line arguments for SDG_fit.py
```
optional arguments:
  -h, --help            Show this help message and exit.
  -model MODEL          Modeling method to be used for modeling arrival times:
                        * AC for arrival count model;
                        * IAT for inter-arrival time model.
  -lambdamod LAMBDAMOD  Method to be used for modeling lambda, depending on MODEL: 
                        * AC: has two options, poisson_fit/neg_bio_reg;
                        * IAT: has three options, mean/loess/poly.
  -verbose VERBOSE      0 to print nothing; > 0 values for printing more
                        information. Possible values: 0, 1, 2, 3. (integer)
```
