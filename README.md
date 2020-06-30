# EV-SDG
Electric vehicle (EV) - Synthetic data generator(SDG): Parametric models to generate synthetic samples of EV sessions data.
 
This repository has.
(i) Trained models that reflect real world dataset. They can be used to generate a synthetic sample of EV sessions dataset, and
(ii) Code to train SDG models from new data.

## References
<a id="1">[1]</a> 
Lahariya, Manu and Benoit, Dries and Develder, Chris (2020). 
Defining a Synthetic Data Generator for Realistic Electric Vehicle Charging Sessions. 
Proceedings of the Eleventh ACM International Conference on Future Energy Systems, 406–407.

<a id="1">[2]</a> 
Lahariya, Manu and Benoit, Dries and Develder, Chris (2020). 
Title
Publication

<img src="https://idlab.technology/assets/img/logo.jpg" width="100">
<img src="https://styleguide.ugent.be/files/uploads/logo_UGent_EN_RGB_2400_kleur_witbg.png" width="50">

Developed in [UGent, IDlab](https://www.ugent.be/ea/idlab/en). 


## Citation
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

@inproceedings{,
author = {Lahariya, Manu and Benoit, Dries and Develder, Chris},
    title = {},
    year = {},
    isbn = {},
    publisher = {},
    doi = {},
    booktitle = {},
    pages = {}
}
```




### Session data generation
A sample of EV sessions data can be generated using the python script `/SDG_sample_generate.py`. Generate_sample() can be used 
to generate and save samples to [/res/generated samples](/res/generated samples). 


```python
from modeling.generate_sample import generate_sample

AM,MMc,MMe = SDG[0],SDG[1],SDG[2]   # SDG = synthetic data generator

generate_sample(AM=AM,MMc=MMc, MMe = MMe,
                horizon_start=horizon_start,horizon_end=horizon_end)
```

AM, MMc and MMe are the SDG models that needs to be passed as inputs. 
Models fitted on a real world data will be used as default SDG models for generation.
These models are saved in [modeling/default_models/saved_models]([modeling/default_models/saved_models]). 
Generated EV sessions data will be saved on the [res/generated_samples]([res/generated_samples]) folder. Please refer to  `SDG_sample_generate.py` for futher details. 

#### Command line arguments for SDG_sample_generate.py

```
optional arguments:
  -h, --help            show this help message and exit
  -start_date START_DATE
                        first date of the horizon for data generation format:
                        dd/mm/YYYY
  -end_date END_DATE    last date of the horizon for data generation format:
                        dd/mm/YYYY
  -use USE              which kind of models to use. "default" for using the
                        default models "latest" for using the lastest trained
                        models
  -model MODEL          modeling method to be used for modeling arrival times:
                        AC for arrival count models IAT for inter-arrival time
                        models
  -lambdamod LAMBDAMOD  Method to be used for modeling lambda: AC: has two
                        options, poisson_fit/neg_bio_reg IAT: has three
                        options, mean/loess/poly
  -verbose VERBOSE      0 to print nothing; >0 values for printing more
                        information. Possible values:0,1,2,3 (integer)

```


Default model specifications:

* SDG:
    * AM: Arrival models can be either inter-arrival time models(IAT) or arrival count(AV) models.
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

To train the models, a Transactions.csv file needs to be saved in the 
root directory. This file must have the raw data that is required to train the model.
The required columns in the this file are *Meter Start, Meter stop, Connection time, Energy, Charge point, lat (optional) and lon (optional)*.

This data will be used to fit an SDG model. `/SDG_fit.py` has the code for fitting the model.
We also the models in [/res/modeling/models](/res/modeling/models) with a timestamp for future use. We save 
the SDG model as a list. 

```python
SDG_model = list([AM,MMc,MMe])
SDG_model = [remove_raw_data(m) for m in SDG_model]
```

**Important:** We save the SDG after removing all the training data that was used to train model,
this is done to protect the confidentiality of the data. (Implemented using *remove_raw_data()* function)


### Pre-processing and analysis of models.

*Pre processing* : 

We study the raw data. This can be found in the `/preprocessing.py`. We create
the session clusters and pole clusters. plot the important distributions, generate the slotted data,
generate time series data from the raw data file that needs to be placed in the root directory. 

* save the transactions.csv in root directory
* run `/preprocessing.py` (supporting module `/preprocess` )
* Please see [res/preprocess](res/preprocess) for generated plots and pre processed data.
 
*Analysis of model parameters* :

A complete analysis of the build models is done using `/Arr_time_model.py` and
`/dep_time_model.py`. We study modeling methods for lambda for arrival models, and
mixture types and optimization methods for mixture models.

* Please see [res/modeling](res/modeling) for further details and generated samples



## References 


 
for preprocessing and fitting the real world data, and also code for generating data using the SDG.  Each model is made 
of the following three components

pip install -r requirements.txt

* Arrival model (AM) - An exponential process/poisson process based model
to generate arrivals of EVs for a given horizon
* Connected time model (MMc)  - a GMMs based model for connected times
* Energy required model (MMe) - a GMMs based model for required energy

All the models are implemented in [modeling/stat](modeling/stat) directory. 
The analysis of different parameters of models is also performed. This is present in 
`/arr_time_model.py` and `/dep_time_model.py`.

We also perform perprocessing on the transactions data to study the statistical 
properties of the real world data. Including generating session clusters/pole clusters.
These processes are implemented in [preprocess](preprocess) directory.
For further details, please see `/preprocessing.py`

 