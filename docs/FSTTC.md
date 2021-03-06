# Calculating forward-spike time tiling coefficents in SimBA

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/FSTTC_1.png" />
</p>


When classifying mutiple behaviors in SimBA, we may be interested in causal sequences - for example, does behaviour A cause the expression of behaviour B, or does behavior B cause the expression of behavior A (... and so one for all the different behaviors being classified in SimBA).

Several statistical solutions to get to such answers have been presented (e.g., [Haccou et al. 1988](https://www.tandfonline.com/doi/abs/10.1080/00949658808811102)), and [Lee et al. (2019)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0220596) proposed the Forward Spike Time Tiling Coefficient (FSTTC), an adaptation of the Spike Time Tiling Coefficient ([Cutts and Eglen, 2014](https://www.jneurosci.org/content/34/43/14288.short)), to detect how behaviors initiated by one animal can trigger behavioral responses of a second animal during dyadic encounters.  

The FSTTC may be helpful to answer questions such as: 

* Does resident attack behavior predominantly cause intruder defensive behavior, or intruder escape behavior? 

* Does attack behavior cause defensive behavior or does defensive behavior cause attack behavior?

Note that SimBA will calculate the FSTTC for all the behaviors selected by the user, and SimBA does **not** require there to be two or more tracked animals.  

>Note: Although  transitional relationships of behavioral events often are evaluated and visualized using Markov decision processes, such techniques may require mutually exclusive states and this introduces stastistical challenges for multi-individual environments. We recognise that the more sophisticated approach to explore casuse and effect in multi-individual environments would be [multi-agent reinforcement learning](https://medium.com/swlh/the-gist-multi-agent-reinforcement-learning-767b367b395f) techniques but.. yeah.. we don't have time to get into that :)


## Part 1: Generate a dataset.

First, SimBA needs classified data to calculate the FSTTC. SimBA will look in the `project_folder/csv/machine_results` directory for files, and calculate the FSTTC scores for all the files in this folder. Thus, before calculating the FSTTC, make sure that you have run your classifiers on your data as documented [HERE](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-8-run-machine-model). In other words, make sure you have processed your data as documented in the [Scenario 1 Tutorial](https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md), up to and including [Step 8](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-8-run-machine-model).






