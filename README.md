# New York air quality study

This project aims to study the air quality around New York neighborhoods and the health outcomes related to it. 
The goal is to investigate the impact of various external factors on air quality and understand their contribution and importance.

## Installation and Usage

1. Clone the repository:
```git clone https://github.com/HindFaris/nyc-air-quality-study```

2. Navigate to the project directory:
```cd nyc-air-quality-study```

3. Install the required dependencies using the requirements.txt file. Make sure you have Python and pip installed. Create a virtual environment if necessary.
```pip install -r requirements.txt```
This will install the necessary packages and libraries required for the project.

## Project Structure

data_preparation
  - data: Folder containing datasets as downloaded from different sources.
  - UHF42: Libraries allowing transformation of geolocation data into UHF42 neighborhoods.
  - data_preparation: Jupyter notebook containing all transformations done on datasets to result in one single dataset merging all data.

data_exploration
  - data_exploration: Notebook containing the core of the study.
  - data_preprocessing: Some aggregations are done to simplify the study.
  - final-dataset: The final dataset that will be used in the study.
  - tools: Functions and tools used in the project.

## Reference

[NYC Government Open Data](https://data.cityofnewyork.us/Environment/Air-Quality/c3uy-2p5r): Air Quality, Automated Traffic Volume Counts, Parks Properties Map, DOHMH New York City Restaurant Inspection Results
Neighborhood poverty

[Kaggle dataset](https://www.kaggle.com/datasets/claytonmiller/new-york-city-buildings-energy-consumption-survey?fbclid=IwAR26gcwjnmJIlwTzyQ9WDNeG0bRFUjtOM2r594mZRD_2WUVkVXXPiryrP_w): New York City Buildings Energy Consumption Survey

Yuan Ren, Zelong Qu, Yuanyuan Du, Ronghua Xu, Danping Ma, Guofu Yang, Yan Shi, Xing Fan, Akira Tani, Peipei Guo, Ying Ge, Jie Chang,
[Air quality and health effects of biogenic volatile organic compounds emissions from urban green spaces and the mitigation strategies,
Environmental Pollution](https://www.sciencedirect.com/science/article/abs/pii/S0269749117309491)
