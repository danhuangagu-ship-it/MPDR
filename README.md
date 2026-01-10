# MPDR (Machine Learning-based Personalized Dietary Recommendations to Achieve Desired Gut Microbial Compositions)
This is a Pytorch implementation of DKI, as described in our paper: 

Wang, X.W.#, Huang D.#, Yu P.F, Weiss, S.T. and Liu, Y.Y. [Machine Learning-based Personalized Dietary Recommendations to Achieve Desired Gut Microbial Compositions ].


<img width="975" height="714" alt="image" src="https://github.com/user-attachments/assets/cf1269a9-d522-4ad0-a8d6-d298ba27f9be" />


# Contents
Overview
Environment
Repo Contents
Data type for MPDR
How the use the MPDR framework

# Overview
Dietary intervention is an effective way to alter the gut microbiome to promote human health. Yet, due to our limited knowledge of diet-microbe interactions and the highly personalized gut microbial compositions, an efficient method to prescribe personalized dietary recommendations to achieve desired health gut microbial compositions is still lacking. Here, we propose a machine learning framework to resolve this challenge. Our key idea is to implicitly learn the diet-microbe interactions by training a machine learning model using paired gut microbiome and dietary intake data from a population-level cohort. The well-trained machine learning model enables us to predict the microbial composition of any given species collection and dietary intake. Next, we prescribe personalized dietary recommendations by solving an optimization problem to achieve the desired microbial compositions. We systematically validated this Machine learning-based Personalized Dietary Recommendation (MPDR) framework using synthetic data generated from an established microbial consumer-resource model. We then validated MPDR using real data collected from a diet-microbiome association study. The presented MPDR framework demonstrates the potential of machine learning for personalized nutrition.

# Environment
We have tested this code for Python 3.9.7 and Pytorch 2.1.0.

# Repo Contents
(1) Python code to predict the species composition using species assemblage and dietary profile (MLP).
(2) A synthetic dataset to test the Machine Learning-based Personalized Dietary Recommendations (MPDR) framework.
(3) A real dataset to test the Machine Learning-based Personalized Dietary Recommendations (MPDR) framework.

# Data type for MPDR

# How the use the MPDR framework
Run Python code in "code" folder: "MPDR_mapping.py" by taking p.csv, z.csv and q.csv as input will output the predicted microbiome composition. Example: python DPDR_mapping.py --perturbation $perturbation --'sparsity' $sp --'connectivity' $C --noise $ep --ratio $ratio --fold $fold

Run Python code in "code" folder: "MPDR_simulation.py" by taking p.csv, z.csv and q.csv as input will output the predicted microbiome composition. Example: python DPDR_mapping.py --perturbation $perturbation --'sparsity' $sp --'connectivity' $C --noise $ep --ratio $ratio --fold $fold

Run Python code in "code" folder: "MPDR_real.py" by taking p.csv, z.csv and q.csv as input will output the predicted microbiome composition. Example: python DPDR_mapping.py --perturbation $perturbation --'sparsity' $sp --'connectivity' $C --noise $ep --ratio $ratio --fold $fold
