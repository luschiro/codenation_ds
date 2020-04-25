import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as pl

#EXPLORATION FUNCTIONS
def df_properties(df):
    st.write('**A quick look into our data: **')
    n_lines = st.slider('Number of rows', min_value = 5 , max_value = 50)
    st.write(df.head(n_lines))
    st.write('Our dataset has **{}** rows and **{}** columns/features.'.format(str(df.shape[0]), str(df.shape[1])))

    check_missing = st.checkbox('Show data types and missing values')
    if(check_missing):
        missing_values = [df[col].isnull().sum() for col in df.columns]
        aux = pd.DataFrame({'data_types' : df.dtypes, 'missing_values' : missing_values})
        aux['missing_values_%'] = aux['missing_values'] / len(df) * 100
        st.table(aux)

    st.write('**Looking at specific features**')
    features = st.multiselect('Select features:', options =  df.columns, default = None)
    n_lines = st.number_input('How many rows: ', min_value = 5, max_value = 50, step = 1)
    if(features): st.write(df[features].head(n_lines))

def descriptive_stats(df):
    st.write('**Univariate Descriptive Statistics**')
    selected_stats = False
    numeric = [value for value in df.columns if df[value].dtypes == np.int64 or df[value].dtypes == np.float64]
    selected_stats = st.multiselect('Choose feature', numeric)
    if(selected_stats): st.table(df[selected_stats].describe())

    check_corr = st.radio('Check correlation between numerical features?', ('No', 'Yes'))
    if(check_corr == 'Yes'):
        corr_type = st.selectbox('Type', ('Table', 'Heatmap'))
        corr = df[selected_stats].corr()
        if(corr_type is not None):
            if(corr_type == 'Table'): st.table(corr)
            elif(corr_type == 'Heatmap'): sns.heatmap(corr, annot = True)
            else: corr = None
        st.pyplot()
    return selected_stats

def plots(df):
    selected_plot = False
    st.write('**Plots**')
    selected_stats = False
    numeric = [value for value in df.columns if df[value].dtypes == np.int64 or df[value].dtypes == np.float64]
    selected_stats = st.multiselect('Choose feature(s) to plot (X,Y): ', numeric)
    if(selected_stats): st.write(df[selected_stats])
    plots = ['distribution plot', 'boxplot', 'scatterplot']
    selected_plot = st.radio('Choose plot type', plots)
    if(selected_plot):
        if(selected_plot == 'distribution plot' and len(selected_stats) == 1): 
            sns.distplot(df[selected_stats], axlabel = selected_stats)
        if(selected_plot == 'boxplot' and len(selected_stats) == 1): 
            sns.boxplot(df[selected_stats], width = 0.5)
        if(selected_plot == 'scatterplot'and len(selected_stats) == 2):
            sns.scatterplot(x = df[selected_stats[0]], y = df[selected_stats[1]])
            st.write('Correlation between these features is about: **{:.2f}**'.format(df[selected_stats].corr().iloc[0][1]))
        
        st.pyplot()

#PAGES
def about():
    st.title('A simple NBA stats app')
    st.header('Exploring the functionalites of Streamlit with data from the best basketball league in the world.')
    st.subheader('About the datasets')
    st.markdown('Datasets were compiled by user Nathan Lauga in [kaggle](https://www.kaggle.com/nathanlauga/nba-games) using the official NBA stats [website](https://stats.nba.com/) API. \nIt has data about NBA teams, players and games.')
    st.markdown('This work is part of the AceleraDev Data Science program from [Codenation](https://www.codenation.dev/) and our tutor is [Túlio Vieira](https://www.linkedin.com/in/tuliovieira/).')
    st.image('https://cutt.ly/QyolEka', width=600)
    st.write('**Author:** Lucas Rodrigues Schiavetti')
    st.write('[LinkedIn](https://www.linkedin.com/in/lucas-schiavetti/)')
    st.write('[GitHub](https://github.com/Lurvetti)')

def exploration():
    #new dropdown menu
    df_prop_check = st.sidebar.checkbox('DataFrame Properties')
    desc_check = st.sidebar.checkbox('Descriptive Statistics')
    plot_check = st.sidebar.checkbox('Plotting')

    st.title('Data Exploration')
    st.markdown('NBA data is divided into 4 different datasets: teams.csv, players.csv, games.csv and games_detail.csv.')
    st.markdown('We will start with **games.csv** but the plan is to improve the app with the others.')
    st.image('https://cutt.ly/FypYcdk', width = 600)
    df = pd.read_csv('games.csv')
    #st.markdown('* The only preprocessing was to replace TEAM IDs (i.e, 1610612759) for actual TEAM_NAMEs from the teams.csv: ')

    if(df_prop_check == True):
        df_properties(df)
    if(desc_check == True):
        descriptive_stats(df)
    if(plot_check == True):
        plots(df)
        
def players():
    st.title('Players')
    st.write('under construction :)')
    pass

#MAIN
def main():
    st.sidebar.image('https://cutt.ly/vyol9CN', width = 300)
    st.sidebar.image('https://cutt.ly/BypUgIL', use_column_width=True)

    page = st.sidebar.selectbox('Menu', ('About','Exploration', 'Players Stats'), index = 0)
    if (page == 'About'):
        about()
    if(page == 'Exploration'):
        exploration()
    if (page == 'Players Stats'):
        players()

if __name__ == "__main__":
    main()