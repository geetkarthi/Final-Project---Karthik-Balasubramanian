# Final-Project---Karthik-Balasubramanian
# Project URL: https://final-project---karthik-balasubramanian-c6tpjdmg3rk9a4vnommxhq.streamlit.app/

# Abstract
The main idea of this project was to show the different statistics for each team over the years. These statistics include - xG, xGA, npxG, npxGA, deep, deep_allowed, scored, missed and many more. These stats are all shown with the help of bar charts, with the options given to the user to choose what stats they want to see. There are 6 options. 
These options are:
1. League Table
2. Team performance per season - this includes 2 options in it. They are a. xG performance and b. passes per action
3. PPDA stats
4. League Position over an entire season
5. Regression for team - an extra regression model which gives us predicted result for a team of your choosing
6. Multiple Visualizations - this includes 3 options - a. PPDA vs Points, b. deep passes vs points and c. xG metrics vs Points

# Data Description
The data for this project was taken from Kaggle, and has the league results for any team in the top 5 leagues and Russian Premier League. The results include:
1. Season
2. Home/Away
3. xG stats - for, against, non-penalty, goal-difference, actual difference
4. Goals for and against
5. Points stats - expected, actual and difference
6. PPDA stats - for, against, attack and defence
7. 20 yard passes stats - for and against
8. Match Date
9. Wins, Draws, Losses and Points
10. Team Name and League Name

# Algorithm Description:
The algorithm used for each choice is mentione below
1. Ask the user for the league and the season. Filter the dataset with these conditions. Group by team, and get the sum of all the relevant stats - Wins, Draws, Losses, Points, Goals for, Goals against, Expected Points. Sort the grouped table by points to get the final league table.
2. There are 2 choices here: First the user will choose the year and then choose the team.
   a. For the xG performance - The dataset is filtered with the above conditions. Group by Team and Year, get the sum and average of the relevant stats - xG, xGA, npxG, npxGA, Goals for and against, xG_diff, xGA_diff, xpts and xpts_diff.
   b. For the PPDA performance - The dataset is filtered with the above conditions. Group by Team and Year, get the sum and average of the relevant stats - ppda/oppda coef, att, def.
3. The user is askef to choose the League and then the Team. The dataset is filtered with these conditions. Group by Year/Season, get the sum and average of the relevant stats - deep, deep_allowed, oppda_coef, ppda_coef. There will be multiple rows for each team, with a minimum of 1 and a maximum of 6. This is because some teams get relegated and are not in the top flight each year.
4. The user is asked to the choose the Season and the League. The dataset is filtered with these conditions. Now, a new table is formed with "number of teams" rows and  "total number of matches" columns. This table shows the league table at the end of each matchweek. From this table, we create another table which has "number of teams" rows and "number of matches columns". This new table will record the league position for each team over the entire season. Now the user will choose which team's league positions they want to see. The corresponding row will be chosen and displayed.
5. For this, the user will just choose the team they are interested in. The dataset is filtered for this team alone, and it is split in a 4:1 ratio for train-test split. The feature variables are chosen as - Home/Away, xG, goals scored, ppda coef, att, def. The target variable is chosen as result. We now employ a decision tree classifier on the training feature dataset, and evaluate the RMSE with the training target dataset. The model is then used on the testing feature dataset, and the result is published alongside the testing target dataset. Finally, the user is aksed to input their own choice of input features, to generate a new result for the team.
6. The user is asked to choose the season first, then the team and then finally the metric they want to see. For any metric chosen, the relevant stats are plotted against the points scored by the team. The relevant stats are as follows:
   a. PPDA metrics - ppda/oppda coef/att/def.
   b. Passes in third metrics - deep. deep_allowed
   c. xG metrics - xG, xGA, npxG, npxGA, xpts and npxGD.

# Tools Used
The following Python packages were used - Numpy, Pandas, Mayplotlib, Seaborn, Streamlit, Sklearn, plotly, itertools
The following python functions were used - st.title. st.selectbox, st.caption, st.pyplot, st.plotly_chart, pd.DataFrame.unique,  pd.DataFrame.group_by, pd.DataFrame.sort_values, pd.DataFrame.reset_index, pd.DataFrame, np.append, lambda functions, matplotlib.plot, train_test_split, fit, predict, np.reshape, px.bar, px.scatter, dataframe filtering.
   
