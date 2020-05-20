# NO2 Detection/ Prediction Analysis from TROPOMI / Landsat
The Repository is for NO2 Prediction Analysis Project from TROPOMI/ Landsat satellite images
## 1. Background. Why NO2 is important?
### 1.1. How serious is the problem of Nitrogen pollution?
Nitrogen oxide is a powerful greenhouse gas responsible for about 6% of human-induced warming to date.  
It also damages the ozone layer, that protects life on earth by absorbing some of  the harmful ultraviolet radiation coming from the sun.
The World Health Organisation (WHO) estimates that about 3 million people die each year from ailments caused by air pollution, and that more than 80% of people living in urban areas are exposed to air quality levels that exceed safe limits. The situation is worse in low-income countries, where 98% of cities fail to meet WHO air quality standards.
### 1.2 Source of NO2:
exhaust stacks.
emissions from individual power plants,
plumbing.
enlarge chemical factories.
NO2 is short-lived and primary emitted from fossil fuel combustion (cars, power plants), so most NO2 is found near the surface. 
### 1.3. Remote Sensing Detection 
On 13 October 2017, the TROPOspheric Monitoring Instrument (TROPOMI)  on the European Space Agency’s (ESA’s)
Sentinel-5 Precursor (S-5P) mission was launched, with spatial resolution of up to 3.5 × 7 km2 and a high signal-to-noise ratio.

## 2. Methods of Machine Learning (ML)
### 2.1.  Why ML is important in Earth Science?

Recent intencive developments in Machine Learning (ML) have expanded appication of artificial intellegence to different areas of our life: urban monitoring, fire detection or flood prediction (Fayyad et al., 1996.). The ML-based methods have been widely applied to the science and engineering problems for near two decades. This is while the application of these techniques in the geosciences and remote sensing area is fairly new and limited (David J.Lary, 2016).
Machine learning algorithms allowed the use of increasently available ‘big data’ like Remote Sensing: multispectral or radar satelite images, LiDAR high resolution data in automatization process of processing and preparing for future analyses.
A machine learning algorithm is a process that is used to fit a model to a dataset, through training or learning. The learned model is subsequently used against an independent dataset, in order to determine how well the learned model can generalise against the unseen data, a process called testing. In general, machine learning algorithms can be divided into two main groups (supervised- and unsupervised-learning; Fig. 1). Supervised-learning algorithms use predefined input-output pairs and learn how to derive outputs from inputs. The user specifies which variables (i.e., outputs) are considered dependent on others (i.e., inputs). The machine learning toolbox includes several linear and non-linear supervised learners, predicting either numeric outputs (regressors) or nominal outputs (classifiers) (Table 1). (SimonWillcock et al., 2018)

### 2.2  Workflow:
Machine learning processes automatically provide estimates of uncertainty.
Uncertainty information enables decision-makers to assign their own thresholds.
Machine learning algorithms can help scientists make use of ‘big data’.
