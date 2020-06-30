# SDG-EV-sessions-data
Synthetic data generator(SDG) for EV sessions data. This repository has the code 
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



### Session data generation
A sample of EV sessions data can be generated using the code in file `/SDG_sample_generate.py`. Generate_sample() can be used 
to generate and save samples to [/res/generated samples](/res/generated samples). 


```python
from modeling.generate_sample import run_script as generate_sample

AM,MMc,MMe = SDG[0],SDG[1],SDG[2]   # SDG = synthetic data generator

generate_sample(AM=AM,MMc=MMc, MMe = MMe,
                horizon_start=horizon_start,horizon_end=horizon_end)
```

AM, MMc and MMe are the SDG models that needs to be passed as inputs. We have a fitted SDG model
that is saved in [res/modeling/models]([res/modeling/models]). We use this to generate the data
in this repository. Please refer to  `SDG_sample_generate.py` for futher details. 


Horizon is the time period for which the data sample has to be generated. This can be updated in `/SDG_sample_generate.py`.

```python

# here we can specify the horizon for data generation
horizon_start= datetime.datetime(2015,1 ,1)
horizon_end= datetime.datetime(2015,1,31)

```


Model Specifications:
latest file - res/modeling/models/SDG Model DT=2020-04-23 19-50
* SDG:
    * AM: Arrival model is exponential process.
        * lambda model = poly
        * Log normal data before modeling = TRUE
        * Combined time slots = TRUE (1-6)
        
    * MMc: Connected time mixture model.
        * Mixtures = GMMs
        * Optimization = EM (MLE)
    * MMe: Energy required mixture model.
        * Mixtures = GMMs
        * Optimization = EM (MLE)

These parameters can be changed and new models can be created using a raw data file.

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

<img src="https://idlab.technology/assets/img/logo.jpg" width="100">
<img src="https://styleguide.ugent.be/files/uploads/logo_UGent_EN_RGB_2400_kleur_witbg.png" width="50">

Developed in [UGent, IDlab](https://www.ugent.be/ea/idlab/en). 

 
 