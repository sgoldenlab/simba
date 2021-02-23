# Appending BORIS annotations in SimBA

BORIS has options for saving human annotations in various formats - and we need to make sure that the BORIS output files are saved in a format that is compatible with SimBA. The screenshot below shows the format of the tabular BORIS CSV format that SimBA expects:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/BORIS_1.png" />
</p>


To generate BORIS data in this file format, begin by creating a new project with **seconds** time format:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/BORIS_2.png" />
</p>

Next, create a new observation in BORIS:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/BORIS_3.png" />
</p>


When you set up your ethogram, name your behaviors the same way as they are named in your simba project. If that is not possible (because historical data), rename your simba classifier names to match the names in the BORIS annotations. We need a way of knowing what behaviors in SimBA matches the behaviors in Boris.

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/BORIS_4.png" />
</p>

Once done, export your annotations as **Tabular events**:

<p align="center">
<img src="https://github.com/sgoldenlab/simba/blob/master/images/BORIS_5.png" />
</p>


Now you have your data in the SimBA-required format - head to the  tutorial on [Appending third-party annotations in SimBA](https://github.com/sgoldenlab/simba/edit/master/docs/third_party_annot.md) to learn how to append it to your feature and pose data. 
