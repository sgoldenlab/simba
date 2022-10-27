| FEATURE                                       | DESCRIPTION                                                                                            |
|:----------------------------------------------|:-------------------------------------------------------------------------------------------------------|
| Euclidean_distance_bodypart1_bodypart2        | Distance between body-part 1 and body-part 2 within the current frame expressed in millimeter          |
| Movement_bodypart1                            | Movement of body-part 1 from location in preceeding frame expressed in millimeter                      |
| All_bp_movements_Animal1_sum                  | The sum of all Animal1 body-part movements from preceeding frame expressed in millimeter               |
| All_bp_movements_Animal1_mean                 | The mean of all Animal1 body-part movements from preceeding frame expressed in millimeter              |
| All_bp_movements_Animal1_min                  | The minimum Animal1 body-part movement from preceeding frame expressed in millimeter                   |
| All_bp_movements_Animal1_max                  | The maximum Animal1 body-part movement from preceeding frame expressed in millimeter                   |
| Mean_N_Euclidean_distance_bodypart1_bodypart2 | The mean distance between body-part 1 and body-part 2 in the preeding N frames expressed in millimeter |
| Sum_N_Euclidean_distance_bodypart1_bodypart2  | The sum of distances between body-part 1 and body-part 2 in the preeding N frames                      |
| Mean_N_All_bp_movements_Animal1_sum           | The mean of the total movement of Animal 1 in the preceeding N frames.                                 |
| Sum_N_All_bp_movements_Animal1_sum            | The aggregated sum of the of the total movement of Animal 1 in the preceeding N frames.                |
| Sum_probabilities                             | The sum of all body-part pose-estimation confidence scores within the current frame                    |
| Mean_probabilities                            | The mean body-part pose-estimation confidence scores within the current frame                          |
| Low_prob_detections_N                         | The count of body-part confidence scores below N within the cirrent frame                              |