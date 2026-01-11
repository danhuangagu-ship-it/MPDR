# MPDR (Machine Learning-based Personalized Dietary Recommendations to Achieve Desired Gut Microbial Compositions)
This is a Pytorch implementation of DKI, as described in our paper: 

Wang, X.W.#, Huang D.#, Yu P.F, Weiss, S.T. and Liu, Y.Y. [Machine Learning-based Personalized Dietary Recommendations to Achieve Desired Gut Microbial Compositions ].


<img width="975" height="714" alt="image" src="https://github.com/user-attachments/assets/cf1269a9-d522-4ad0-a8d6-d298ba27f9be" />


# Contents
Overview
Environment
Repo Contents
How the use the MPDR framework

# Overview
Dietary intervention is an effective way to alter the gut microbiome to promote human health. Yet, due to our limited knowledge of diet-microbe interactions and the highly personalized gut microbial compositions, an efficient method to prescribe personalized dietary recommendations to achieve desired health gut microbial compositions is still lacking. Here, we propose a machine learning framework to resolve this challenge. Our key idea is to implicitly learn the diet-microbe interactions by training a machine learning model using paired gut microbiome and dietary intake data from a population-level cohort. The well-trained machine learning model enables us to predict the microbial composition of any given species collection and dietary intake. Next, we prescribe personalized dietary recommendations by solving an optimization problem to achieve the desired microbial compositions. We systematically validated this Machine learning-based Personalized Dietary Recommendation (MPDR) framework using synthetic data generated from an established microbial consumer-resource model. We then validated MPDR using real data collected from a diet-microbiome association study. The presented MPDR framework demonstrates the potential of machine learning for personalized nutrition.

# Environment
We have tested this code for Python 3.9.7 and Pytorch 2.1.0.

# Repo Contents
(1) Python code to predict the species composition using species assemblage and dietary profile (MLP).

(2) A synthetic dataset to test the Machine Learning-based Personalized Dietary Recommendations (MPDR) framework.

(3) A real dataset to test the Machine Learning-based Personalized Dietary Recommendations (MPDR) framework.

# How the use the MPDR framework

Users can directly replace the input CSV files with their own datasets, as long as the formats are consistent, with each row representing one sample. Run the Python script MPDR_simulated_community.py in "code" folder：The model is trained using p_train.csv, z_train.csv, and q_train.csv to learn a mapping for microbiome composition prediction. For diet recommendation, p_desired.csv, z_start.csv, and q_start.csv are provided as inputs to optimize personalized diets. Example: python MPDR_simulated_community.py --p_train ./data/p_healthy_0.1_0.01_0.2_0_5_1.csv --z_train data/z_healthy_0.1_0.01_0.2_0_5_1.csv --q_train ./data/q_healthy_0.1_0.01_0.2_0_5_1.csv --p_target  ./data/p_disease_0.1_0.01_0.2_0_5_1.csv --z_start ./data/z_disease_0.1_0.01_0.2_0_5_1.csv --q_start  ./data/q_disease_perm_0.1_0.01_0.2_0_5_1.csv --out_dir ./results --tag MPDR_test

Run Python code in "code" folder: "MPDR.py" by taking p_desired.csv, z_start.csv, and q_start.csv as input will output the optimize personalized diets. Example: python MPDR.py python MPDR.py --model_path ./MPDR_model.pt --z_start ./data/DMAS_z_disease.csv --q_start ./data/DMAS_q_disease_random.csv --p_target ./data/DMAS_p_disease_desired.csv --out_dir ./results
