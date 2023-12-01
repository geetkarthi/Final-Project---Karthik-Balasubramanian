import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
import streamlit as st
import itertools
import plotly.express as px
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz 
from sklearn import preprocessing
df = pd.read_csv(r"C:\Users\karth\Downloads\understat_per_game.csv")
st.title("Welcome to Football Statistics")
#the options available
array = ['Choose Value', 'League Table', 'Team Performance per Season', 'ppda stats', 'League position through a season', 'Regression for Team', 'Multiple Visualizations']
selectbox_a = st.selectbox(
  "What do you want to see?",
  [i for i in array]
)
if selectbox_a == 'Choose Value':
  st.write('Choose an option')

elif selectbox_a == 'League Table':
    selectbox1 = st.selectbox(
        "Select the Season",
        [i for i in df['year'].unique()]
    )

    selectbox2 = st.selectbox(
        "Select the League",
        [i for i in df['league'].unique()]
    )
    df_new = df[df['year'] == selectbox1]
    df_new = df_new[df_new['league'] == selectbox2]
    df_actual_table = df_new.groupby(['year', 'league', 'team'])['wins', 'draws', 'loses', 'scored', 'xG', 'missed', 'xGA', 'xpts', 'pts'].sum()
    df_actual_table.sort_values('pts', ascending = False, inplace = True)
    df_actual_table.reset_index(inplace=True)
    st.write(df_actual_table)
    st.caption("The league table for the season")
    df_x = df_actual_table.sort_values('xpts', ascending = False, inplace = True)
    st.write(df_x)
    st.caption("The xG table for the season")
    st.title("Total points table for" + ' ' + str(selectbox2) + ' ' + "for the season" + ' ' + str(selectbox1))
    fig = plt.figure(figsize=(50, 4))
    a = sns.barplot(df_actual_table, x = 'team', y = 'pts')
    st.pyplot(fig)
    b = px.bar(df_actual_table, x = 'team', y = ['wins', 'draws', 'loses'], barmode = 'group')
    c = px.bar(df_actual_table, x = 'team', y = ['scored', 'missed'], barmode = 'group')
    st.title("Wins - draws - losses for" + ' ' + str(selectbox2) + ' ' + "for the season" + ' ' + str(selectbox1))
    st.plotly_chart(b)
    st.title("Goals scored and against")
    st.plotly_chart(c)

elif selectbox_a == 'Team Performance per Season':
    selectbox1 = st.selectbox(
        "Select the Team",
        [i for i in df['team'].unique()]
    )
    a = ['xG performance', 'passes per action']
    selectbox2 = st.selectbox(
        "Select the metric",
        [i for i in a]
    )
    selectbox3 = st.selectbox(
        "Select the Season",
        [i for i in df['year'].unique()]
    )

    if selectbox2 == 'xG performance':
        df_new = df[df['team'] == selectbox1]
        df_new = df_new[df_new['year'] == selectbox3]
        df_sum = df_new.groupby(['team', 'year'])['xG', 'xGA', 'npxG', 'npxGA', 'scored', 'missed', 'xG_diff', 'xGA_diff', 'xpts', 'xpts_diff'].sum()
        df_mean = df_new.groupby(['team', 'year'])['xG', 'xGA', 'npxG', 'npxGA', 'scored', 'missed', 'xG_diff', 'xGA_diff', 'xpts', 'xpts_diff'].mean()
        df_sum.reset_index(inplace=True)
        df_mean.reset_index(inplace=True)
        st.write(df_sum)
        st.caption("Sum of all xG metrics across a season")
        st.write(df_mean)
        st.caption("Mean of all xG metrics across a season")
        st.title("xG performance visualized for" + ' ' + str(selectbox1) + ' ' + "in the season" + ' ' + str(selectbox3))
        fig = plt.figure(figsize=(50, 4))
        a = px.bar(df_sum, y = ['xG', 'xGA', 'npxG', 'npxGA'], x = 'team', barmode = 'group')
        st.plotly_chart(a)
        st.title("Goals scored and Conceded")
        fig = plt.figure(figsize=(50, 4))
        b = px.bar(df_sum, y = ['scored', 'missed'], x = 'team', barmode = 'group')
        st.plotly_chart(b)

    elif selectbox2 == 'passes per action':
        df_new = df[df['team'] == selectbox1]
        df_new = df_new[df_new['year'] == selectbox3]
        df_pass_sum = df_new.groupby(['team', 'year'])['ppda_coef', 'ppda_att', 'ppda_def', 'oppda_coef', 'oppda_att', 'oppda_def'].sum()
        df_pass_avg = df_new.groupby(['team', 'year'])['ppda_coef', 'ppda_att', 'ppda_def', 'oppda_coef', 'oppda_att', 'oppda_def'].mean()
        df_pass_sum.reset_index(inplace=True)
        df_pass_avg.reset_index(inplace=True)
        st.write(df_pass_sum)
        st.caption('Sum of Passes per defensive action for an entire season')
        st.write(df_pass_avg)
        st.caption('Mean of Passes per defensive action for an entire season')
        st.title("Passes per defensive action for" + ' ' + str(selectbox1))
        a = px.bar(df_pass_sum, y = ['ppda_coef', 'oppda_coef'], x = 'team', barmode = 'group')
        st.plotly_chart(a)

elif selectbox_a == 'ppda stats':
    selectbox1 = st.selectbox(
        'Select the team',
        [i for i in df['team'].unique()]
    )

    selectbox2 = st.selectbox(
        'Select the league',
        [i for i in df['league'].unique()]
    )

    df_new = df[df['team'] == selectbox1]
    df_new = df_new[df_new['league'] == selectbox2]
    df_actual_sum = df_new.groupby(['year'])['deep', 'deep_allowed', 'ppda_coef', 'oppda_coef'].sum()
    df_actual_mean = df_new.groupby(['year'])['deep', 'deep_allowed', 'ppda_coef', 'oppda_coef'].mean()
    df_actual_sum.reset_index(inplace=True)
    df_actual_mean.reset_index(inplace=True)
    st.write(df_actual_sum)
    st.write(df_actual_mean)
    st.caption('Evolution of the team over the years')
    st.title("Passes near both 20 yard boxes for" + ' ' + str(selectbox1))
    fig = plt.figure(figsize=(50, 4))
    a = px.bar(df_actual_sum, y = ['deep', 'deep_allowed'], x = 'year', barmode = 'group')
    st.plotly_chart(a)

elif selectbox_a == 'League position through a season':
    selectbox1 = st.selectbox(
        "Select the Season",
        [i for i in df['year'].unique()]
    )

    selectbox2 = st.selectbox(
        "Select the League",
        [i for i in df['league'].unique()]
      )
  
    df_new = df[df['year'] == selectbox1]
    df_new = df_new[df_new['league'] == selectbox2]
    df_new['difference'] = df_new['scored'] - df_new['missed']
    if selectbox2 == 'Bundesliga':
        a = [[(0, 0, 0, 0) for i in range(34)] for i in range(18)]
        x = 0
        n = [i for i in df_new['team'].unique()]
        q = []
        for i in range(len(n)):
            q.append(i+1)
        for i in range(18):
            a[i][0] = (df_new['pts'].values[x], df_new['difference'].values[x], df_new['scored'].values[x], q[i])  
            for j in range(1, 34):
                a[i][j] = (df_new['pts'].values[x + j] + a[i][j-1][0], df_new['difference'].values[x + j] + a[i][j-1][1], df_new['scored'].values[x + j] + a[i][j-1][2], q[i])
            x += 34
        m = []
        for i in range(34):
            m.append("Week" + ' ' + str(i + 1))
        new_df = pd.DataFrame(a, columns = m)
        for i in new_df.columns:
            new_df[i] = sorted(new_df[i], key = lambda sub: (-sub[0], -sub[1], -sub[2]))
            new_df[i] = sorted(new_df[i], key = lambda sub: (-sub[0], -sub[1], -sub[2]))
            new_df[i] = sorted(new_df[i], key = lambda sub: (-sub[0], -sub[1], -sub[2]))
        c = []
        for i in new_df.columns:
            new_df[i + ' ' + "positions"] = new_df[i].apply(lambda row: row[3])
            c.append(i + ' ' + "positions")
        new = new_df[c]
        b = [[0 for i in range(34)] for j in range(18)]
        for i in range(18):
            for j in range(34):
                b[i][j] = 18 - new.index[new.iloc[:, j] == i + 1].to_list()[0]
        new_one = pd.DataFrame(b, columns=c)
        st.write("The teams and their corresponding numbers are below")
        st.write([x for x in itertools.chain.from_iterable(itertools.zip_longest(n,q)) if x])
        selectbox3 = st.selectbox(
            "Please select the team number",
            [i for i in range(1, 19)]
        )
        l = new_one.iloc[[selectbox3 - 1]]
        l = l.T
        fig = l.plot()
        st.pyplot(fig.figure)
        
    elif (selectbox2 == 'EPL') or (selectbox2 == 'Ligue 1') or (selectbox2 == 'Serie A') or (selectbox2 == 'La Liga'):
        a = [[(0, 0, 0, 0) for i in range(38)] for i in range(20)]
        x = 0
        n = [i for i in df_new['team'].unique()]
        q = []
        for i in range(len(n)):
            q.append(i+1)
        for i in range(20):
            a[i][0] = (df_new['pts'].values[x], df_new['difference'].values[x], df_new['scored'].values[x], q[i])  
            for j in range(1, 38):
                a[i][j] = (df_new['pts'].values[x + j] + a[i][j-1][0], df_new['difference'].values[x + j] + a[i][j-1][1], df_new['scored'].values[x + j] + a[i][j-1][2], q[i])
            x += 38
        m = []
        for i in range(38):
            m.append("Week" + ' ' + str(i + 1))
        new_df = pd.DataFrame(a, columns = m)
        for i in new_df.columns:
            new_df[i] = sorted(new_df[i], key = lambda sub: (-sub[0], -sub[1], -sub[2]))
            new_df[i] = sorted(new_df[i], key = lambda sub: (-sub[0], -sub[1], -sub[2]))
            new_df[i] = sorted(new_df[i], key = lambda sub: (-sub[0], -sub[1], -sub[2]))
        c = []
        for i in new_df.columns:
            new_df[i + ' ' + "positions"] = new_df[i].apply(lambda row: row[3])
            c.append(i + ' ' + "positions")
        new = new_df[c]
        b = [[0 for i in range(38)] for j in range(20)]
        for i in range(20):
            for j in range(38):
                b[i][j] = 20 - new.index[new.iloc[:, j] == i + 1].to_list()[0]
        new_one = pd.DataFrame(b, columns=c)
        st.write("The teams and their corresponding numbers are below")
        st.write([x for x in itertools.chain.from_iterable(itertools.zip_longest(n,q)) if x])
        selectbox3 = st.selectbox(
            "Please select the team number",
            [i for i in range(1, 21)]
        )
        l = new_one.iloc[[selectbox3 - 1]]
        l = l.T
        fig = l.plot()
        st.pyplot(fig.figure)

    elif selectbox2 == 'RFPL':
        a = [[(0, 0, 0, 0) for i in range(30)] for i in range(16)]
        x = 0
        n = [i for i in df_new['team'].unique()]
        q = []
        for i in range(len(n)):
            q.append(i+1)
        for i in range(16):
            a[i][0] = (df_new['pts'].values[x], df_new['difference'].values[x], df_new['scored'].values[x], q[i])  
            for j in range(1, 30):
                a[i][j] = (df_new['pts'].values[x + j] + a[i][j-1][0], df_new['difference'].values[x + j] + a[i][j-1][1], df_new['scored'].values[x + j] + a[i][j-1][2], q[i])
            x += 30
        m = []
        for i in range(30):
            m.append("Week" + ' ' + str(i + 1))
        new_df = pd.DataFrame(a, columns = m)
        for i in new_df.columns:
            new_df[i] = sorted(new_df[i], key = lambda sub: (-sub[0], -sub[1], -sub[2]))
            new_df[i] = sorted(new_df[i], key = lambda sub: (-sub[0], -sub[1], -sub[2]))
            new_df[i] = sorted(new_df[i], key = lambda sub: (-sub[0], -sub[1], -sub[2]))
        c = []
        for i in new_df.columns:
            new_df[i + ' ' + "positions"] = new_df[i].apply(lambda row: row[3])
            c.append(i + ' ' + "positions")
        new = new_df[c]
        b = [[0 for i in range(30)] for j in range(16)]
        for i in range(16):
            for j in range(30):
                b[i][j] = 16 - new.index[new.iloc[:, j] == i + 1].to_list()[0]
        new_one = pd.DataFrame(b, columns=c)
        st.write("The teams and their corresponding numbers are below")
        st.write([x for x in itertools.chain.from_iterable(itertools.zip_longest(n,q)) if x])
        selectbox3 = st.selectbox(
            "Please select the team number",
            [i for i in range(1, 17)]
        )
        l = new_one.iloc[[selectbox3 - 1]]
        l = l.T
        fig = l.plot()
        st.pyplot(fig.figure) 
    
elif selectbox_a == 'Regression for Team':
    selectbox1 = st.selectbox(
        "Select the Team",
        [i for i in df['team'].unique()]
    )
    
    df_new = df[df['team'] == selectbox1]
    df_new['h_a'] = df_new['h_a'].map({'h': 1, 'a': 0})
    df_new['result'] = df_new['result'].map({'w': 2, 'd': 1, 'l': 0})
    X = df_new[['h_a', 'xG', 'scored', 'ppda_coef', 'ppda_att', 'ppda_def']]
    y = df_new['result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 28)
    regressor = DecisionTreeRegressor() 
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    st.write(y_test, y_pred)
    rmse = format(np.sqrt(mean_squared_error(y_test, y_pred)), '.3f')
    st.write("\nRMSE: ", rmse)
    export_graphviz(regressor, out_file ='tree_structure_2.dot', 
               feature_names =['h_a', 'xG', 'scored', 'ppda_coef', 'ppda_att', 'ppda_def'])
    
elif selectbox_a == 'Multiple Visualizations':
    selectbox1 = st.selectbox(
        "Select the team",
        [i for i in df['team'].unique()]
    )
    
    selectbox2 = st.selectbox(
        "Select the Season",
        [i for i in df['year'].unique()]
    )
    
    m = ['PPDA metrics', 'Passes in thirds metric', 'xG metrics']
    selectbox3 = st.selectbox(
        "Choose the metric",
        [i for i in m]
    )
    
    df_new = df[df['team'] == selectbox1]
    df_new = df_new[df_new['year'] == selectbox2]
    
    if selectbox3 == 'PPDA metrics':
        a = px.scatter(df_new, x = 'ppda_coef', y = 'pts')
        b = px.scatter(df_new, x = 'ppda_att', y = 'pts')
        c = px.scatter(df_new, x = 'ppda_def', y = 'pts')
        d = px.scatter(df_new, x = 'oppda_coef', y = 'pts')
        e = px.scatter(df_new, x = 'oppda_att', y = 'pts')
        f = px.scatter(df_new, x = 'oppda_def', y = 'pts')
        st.title("PPDA stats versus points")
        st.plotly_chart(a)
        st.plotly_chart(b)
        st.plotly_chart(c)
        st.plotly_chart(d)
        st.plotly_chart(e)
        st.plotly_chart(f)
        
    if selectbox3 == 'Passes in thirds metric':
        a = px.scatter(df_new, x = 'deep', y = 'pts')
        b = px.scatter(df_new, x = 'deep_allowed', y = 'pts')
        st.title("Passes in thirds versus points")
        st.plotly_chart(a)
        st.plotly_chart(b)
        
    if selectbox3 == 'xG metrics':
        a = px.scatter(df_new, x = 'xG', y = 'pts')
        b = px.scatter(df_new, x = 'xGA', y = 'pts')
        c = px.scatter(df_new, x = 'npxG', y = 'pts')
        d = px.scatter(df_new, x = 'npxGA', y = 'pts')
        e = px.scatter(df_new, x = 'xpts', y = 'pts')
        f = px.scatter(df_new, x = 'npxGD', y = 'pts')
        st.title("xG stats versus points")
        st.plotly_chart(a)
        st.plotly_chart(b)
        st.plotly_chart(c)
        st.plotly_chart(d)
        st.plotly_chart(e)
        st.plotly_chart(f)
