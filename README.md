# Deduplicated Monte Carlo Reward Generation for Large Language Models

## Please direct inquiries about this project to dasilva [dot] 30 [at] osu [dot] edu

--------

## Motivation
### - Monte Carlo Reward Estimation in Lanuage Models shows promising results on math reasoning tasks such as GSM8k and MATH.
### - Standard practice employs temperature as a solitary method for obtaining multiple diverse reasoning paths from a seed prompt.
### - Basic analysis of open source data (e.g. Math Shepherd) shows that solutions to GSM8k and MATH questions contain many duplicates.

## Implemetation
### - I propose Monte Carlo Deduplicated, MC-dd, a method for generating deduplicated datasets
### - As generations are built, semantic similarity between steps makes sure that only unique steps are chosen as the next candidate in a rollout.
### - This will augment the number of reasoning paths a model can traverse, creating a more diverse and complete reward dataset based on the models' on-policy generations.

## Results
### - MC-dd creates synthetic datasets with more samples and more diverse solutions
### - Final evals pending