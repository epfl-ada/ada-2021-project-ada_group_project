Title: Project Applied Data Analysis: Analysis of LGBTQIA+ related quotes between 2015 and 2020

Abstract: Through this project, we will use the great potential of Quotebank to analyse the evolution of opinions regarding LGBTQIA+ community in the USA: what are the opinions of politicians in different states? How did this topic evolve across time and political state borders? What is the citizens behavior with respect to this topic? 
One could say that it is sufficiant to read on the internet what is the situation related to this topic and then wonder what is the purpose of focusing our project on this subject? As it is kind of a controversial topic with various opinions, changes across time and different actors, it requires an analysis taking into account many parameters. We chose this topic because we thought that it would be more interesting to focus on a controversial theme that would thus refer to different actors (speakers) with meaningful statistics. Moreover, basing our analysis on a comparison between politicians representing different states should give more valuable results than looking at the entire country without partitioning it.     

Research questions: 
We will ask ourselves several questions such as: 
In which way was there a change of the credits given to this topic between 2015 and 2020? 
Are there real differences in opinion between states politicians? Are there some prominent speakers? 
In which state do we find the maximum number of quotes? 
Which jobs occupy the top in the speakers ranking?
Are there some states where the topic is into the spotlight in comparison to other states?
Are there states with more hate crime related to sexual orientation?
How quotes about gay rights classed by locations are relevent about the mentality and general politic of a state?

Methods:
To answer these questions we added a few datasets to make the Quotebank dataset even more relevant regarding our subject. We added The list of politicans by states under the file politicans, the list of hate crimes in the united states in 2015 and 2019, we got the percentage of lgbt in the adult population in each state for 2015-2016 in the file lgbtsummary and also various informations about the lgbt community in the file lgbtpopulation2021.
1/. Create a proper dataset mainly by searching for key words in the quote repository (refer to the code line 'words = ['lesbian', 'gay', 'homosexual', 'gender', 'bisexual', 'sexuality', 'same sex', ...]') and by grouping the politicians in different states to search for pertinent speakers.
2/. Filter if our datasets appears to be noisy (by removing useless information, cleaning the data, create proper dataframes, normalize...)
3/. Aggregate our data in a meaningful way:
we will generate various dataframe to fragment the information in function of the year for the analysis across time. Dataframes should contain the names of the speakers, their States, and the corresponding number of quotes. 
4/. Create relevant visualizations to analyze the data: this is a necessary step to check for correlation, for specific patterns or for unexpected trends if there are some. 
5/. Think critically about the data: concluing part of the project, with creation of the data story. At this point the research question should all be answered. 

Link to the datastory: https://meghanharrington.github.io/

Organization of the files:
* The project is divided into three notebooks:
    - new_dataframes: creation of the dataframes. It deals with heavy data processing, which are not working well on every
    computer and are too heavy for GitHub. Thus, the code is not to be runned. All the necessary dataframes created in here are
    in './data'.
    - exploration_file: it contains all the pre-processing steps data and the exploration with analysis of politician speakers. 
    - statistics: it contains an analysis of a broader spectrum (not onlu politicians) including american citizens. 
    - lda_functions.py contains necessary functions for the lda topic modeling
* The folder 'data' contains csv files:
    - americans: american speakers from Quotebank and their information
    - df{year} : quotes per speaker with additional features we won't use
    - hatecrime{year} : information about hate crimes in the USA for the given year
    - term{..} : files with congress people
    - us senators
    - annual population : population size per state and per year
    - lgbtsummary : statistics about lgbtqia+ people
    - lgbtpopulation : statistics about lgbtqia+ people
