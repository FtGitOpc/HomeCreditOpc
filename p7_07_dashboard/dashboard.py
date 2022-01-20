#streamlit run dashboard.py

import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import re
import math
import base64
from zipfile import ZipFile
from lightgbm import LGBMClassifier
from streamlit_echarts import st_echarts



#Configuration application
st.set_page_config('Accord Crédit')
original_title = '<p style="font-family:Courier; color:Red; font-size: 50px;text-align: center;">Accord prêt bancaire: Analyse détaillée</p>'
st.markdown(original_title, unsafe_allow_html=True)
st.markdown("***")
st.write("""Cette application prédit la probabilité qu'un client de la banque "Prêt à dépenser" ne rembourse pas son prêt.
""")
#Data obtained from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data).


main_bg = "color/main_bg.jpeg"
main_bg_ext = "jpeg"
sns.set_theme(style="darkgrid")

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 700px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 500px;
        margin-left: -100px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

#fonctions
def plot_stat(data, feature, title, size) :
    '''
    plot_stat(): Ajoute le pourcentage auquel correspond chaque hue de chaque catégorie sur la totalité du dataset
    '''
    ax, fig = plt.subplots(figsize=size)
    ax = sns.countplot(y=feature, data=data, order=data[feature].value_counts(ascending=False).index)
    ax.set_title(title)

    for p in ax.patches:
                percentage = '{:.1f}%'.format(100 * p.get_width()/len(data[feature]))
                x = p.get_x() + p.get_width()
                y = p.get_y() + p.get_height()/2
                ax.annotate(percentage, (x, y), fontsize=20, fontweight='bold')

    plt.show()

def barPerc(df,xVar,ax):
    '''
    barPerc(): Add percentage depenfor hues to bar plots
    args:
        df: pandas dataframe
        xVar: (string) X variable
        ax: Axes object (for Seaborn Countplot/Bar plot or
                         pandas bar plot)
    '''
    # 1. how many X categories
    ##   check for NaN and remove
    numX=len([x for x in df[xVar].unique() if x==x])

    # 2. The bars are created in hue order, organize them
    bars = ax.patches
    ## 2a. For each X variable
    for ind in range(numX):
        ## 2b. Get every hue bar
        ##     ex. 8 X categories, 4 hues =>
        ##    [0, 8, 16, 24] are hue bars for 1st X category
        hueBars=bars[ind:][::numX]
        ## 2c. Get the total height (for percentages)
        total = sum([x.get_height() for x in hueBars])

        # 3. Print the percentage on the bars
        for bar in hueBars:
            ax.text(bar.get_x() + bar.get_width()/2.,
                    bar.get_height(),
                    f'{bar.get_height()/total:.0%}',
                    ha="center",va="bottom", fontsize=16)


def load_knn(sample):
    knn = knn_training(sample)
    return knn

def knn_training(sample):
    knn = KMeans(n_clusters=3).fit(sample)
    return knn

def load_kmeans(sample, id, mdl):
    index = sample[sample.index == int(id)].index.values
    index = index[0]
    data_client = pd.DataFrame(sample.loc[sample.index, :])
    df_neighbors = pd.DataFrame(knn.fit_predict(data_client), index=data_client.index)
    #df_neighbors = pd.concat([df_neighbors, data], axis=1)
    return df_neighbors.iloc[:,1:].sample(10)


#Mise en place du model, des données et du seuil
lgbm = pickle.load(open('model/best_final_prediction.pickle', 'rb'))

z = ZipFile("data/df_test_imputed.zip")
data_clean = pd.read_csv(z.open('df_test_imputed.csv'), index_col='SK_ID_CURR', encoding ='utf-8')
all_id_client = data_clean.index

z = ZipFile("data/real_data_clean_test.zip")
data_origin = pd.read_csv(z.open('real_data_clean_test.csv'), index_col='SK_ID_CURR', encoding ='utf-8')

z = ZipFile("data/columns_description_modif.zip")
description = pd.read_csv(z.open('columns_description_modif.csv'),
                                  usecols=['Row', 'Description'], index_col=0, encoding='mac_roman')

z = ZipFile("data/test_imputed_without_standardisation.zip")
data_clean_without_standard = pd.read_csv(z.open('test_imputed_without_standardisation.csv'), index_col='SK_ID_CURR', encoding ='utf-8')



seuil= 0.6
original_title = '<p style="font-family:Courier; color:Blue; font-size: 18px;">La probabilité maximale de défaut de remboursement autorisée par la banque est de : {}</p>'.format(seuil)
st.markdown(original_title, unsafe_allow_html=True)


####################################################################################
####################################################################################
####################################################################################
#sidebar : Analyse descriptive
st.sidebar.markdown(' ', unsafe_allow_html=True)

ori_title = '<p style="font-family:Courier; color:Red; font-size: 50px;text-align: center;">Analyse générale</p>'
st.sidebar.markdown(ori_title, unsafe_allow_html=True)

graph=['Quelle variable voulez-vous voir ?', 'Type de contrat', 'Revenus', 'Montant du crédit', 'Age']
choix = st.sidebar.selectbox("Analyses possibles", graph)
#*------------------------------------------------------------------------------------------*
if choix == 'Type de contrat':
    sub_graph=['Par rapport à quelle variable ?', 'Sans filtre (all)', 'Sexe']
    sub_choix = st.sidebar.selectbox("", sub_graph)

    if sub_choix == 'Sans filtre (all)':
        values_train = data_origin['NAME_CONTRACT_TYPE'].value_counts()
        labels_train = data_origin['NAME_CONTRACT_TYPE'].value_counts().index

        fig = plt.figure(figsize=(10, 10))
        plt.pie(values_train, labels=labels_train,
                autopct='%.1f%%', textprops={'fontsize': 20}, colors=['darkcyan',"plum"])
        plt.title("Répartition des types de prêt", fontsize=20)
        st.sidebar.pyplot(fig)
    elif sub_choix == 'Sexe':
        fig, ax = plt.subplots(figsize=(8, 10))
        ax = sns.countplot(x="NAME_CONTRACT_TYPE", hue="CODE_GENDER", data=data_origin, palette=['violet',"blueviolet"])
        title="Répartition du genre des clients selon le type de prêt"
        ax.set_title(title, fontsize=16)
        plt.xticks(size=14)
        ax.set_xlabel("")
        ax.set_ylabel("")
        barPerc(data_origin, 'NAME_CONTRACT_TYPE', ax)
        st.sidebar.pyplot(fig)
    sub_choix = None
#*------------------------------------------------------------------------------------------*
if choix == 'Revenus':
    df_income = pd.DataFrame(data_origin)
    df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
    df_income['INCOME_BINNED'] = pd.cut(df_income['AMT_INCOME_TOTAL'], bins=list(range(20000,200001,20000)))
    df_income['AGE_BINNED'] = pd.cut(df_income['AGE'], bins=list(range(20,71,10)))

    sub_graph=['Par rapport à quelle variable ?', 'Sans filtre (all)', 'Sexe', 'Age', 'Count', 'Montant du crédit', 'Status familial', 'Type de revenus', 'Education','Possesion voiture', 'Possession logement', 'Type de contrat', 'Nombre enfant', 'Type de logement']
    sub_choix = st.sidebar.selectbox("", sub_graph)
    if sub_choix == 'Sans filtre (all)':
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.histplot(df_income["AMT_INCOME_TOTAL"], color="violet", bins=10)
        ax.set(title='Revenu des clients', xlabel='Revenu (USD)', ylabel='')
        st.sidebar.pyplot(fig)
    elif sub_choix == 'Sexe':
        fig, ax = plt.subplots(figsize=(12, 10))
        ax = sns.countplot(x="INCOME_BINNED", hue="CODE_GENDER", data=df_income, palette=['violet',"blueviolet"])
        title="Répartition des revenus des clients selon leur genre"
        ax.set_title(title, fontsize=16)
        plt.xticks(size=18, rotation=75)
        ax.set_xlabel("Revenus")
        ax.set_ylabel("")
        plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='18') # for legend title
        barPerc(df_income, 'INCOME_BINNED', ax)
        st.sidebar.pyplot(fig)
    elif sub_choix == 'Age':
        graph=['Choix', 'Scatterplot', 'Barplot']
        choice = st.sidebar.selectbox("Type de graphe", graph)
        if choice == 'Scatterplot':
            subset = df_income.sample(n=2000)
            fig, ax = plt.subplots(figsize=(12, 10))
            ax = sns.regplot(x="AGE", y="AMT_INCOME_TOTAL", data=subset, scatter_kws={"color": "mediumseagreen"}, line_kws={"color": "red"})
            ax.annotate('Ligne de régression', xy=(400, 30), xycoords='axes points',
                        size=20, ha='right', va='top',
                        bbox=dict(boxstyle='round', fc='w'), color='red')
            title="Relation entre les revenus globaux et l'âge des clients"
            ax.set_title(title, fontsize=18)
            plt.xticks(size=18)
            ax.set_xlabel("Age des clients", fontsize=16)
            ax.set_ylabel("Revenus total", fontsize=16)
            #ax.legend()
            st.sidebar.pyplot(fig)

        elif choice == 'Barplot':
            age_data = df_income[['AGE', 'INCOME_BINNED']]
            income_groups = age_data.groupby("INCOME_BINNED").mean().round(1)

            fig = plt.figure(figsize=(12, 10))
            ax1 = plt.subplot(3, 1, 1)
            sns.barplot(data = age_data, x = 'INCOME_BINNED', y = 'AGE', palette = 'rocket', ci = 'sd')
            title="Age moyen des clients selon leur tranche de revenus"
            ax1.set_title(title, fontsize=18)
            plt.xticks(rotation = 75, size=18)
            ax1.set_xlabel("Tranches de revenus", fontsize=16)
            ax1.set_ylabel("Age moyen des clients", fontsize=16)

            ax2 = plt.subplot(3, 1, 3)
            sns.lineplot(data = income_groups, x = income_groups.index.astype(str), y = 'AGE', ci = None)
            title="Age moyen des clients selon leur tranche de revenus"
            ax2.set_title(title, fontsize=18)
            plt.xticks(rotation = 75, size=18)
            ax2.set_xlabel("Tranches de revenus", fontsize=16)
            ax2.set_ylabel("Age moyen des clients", fontsize=16)

            st.sidebar.pyplot(fig)

    elif sub_choix == 'Count':
        age_data = df_income[['AGE', 'INCOME_BINNED']]
        income_groups = age_data.groupby("INCOME_BINNED").count()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(data = income_groups, x = income_groups.index.astype(str), y = income_groups['AGE'], palette = 'rocket')
        title="Nombre de clients selon leur tranche de revenus"
        ax.set_title(title, fontsize=18)
        plt.xticks(rotation = 75, size=18)
        ax.set_xlabel("Tranches de revenus", fontsize=16)
        ax.set_ylabel("Nombre des clients", fontsize=16)
        st.sidebar.pyplot(fig)


    elif sub_choix == 'Montant du crédit':
        subset = df_income.sample(n=2000)
        fig, ax = plt.subplots(figsize=(12, 10))
        ax = sns.regplot(x="AMT_INCOME_TOTAL", y="AMT_CREDIT", data=subset, scatter_kws={"color": "lightcoral"}, line_kws={"color": "red"})
        title="Relation entre les revenus globaux et le montant des prêts"
        ax.annotate('Ligne de régression', xy=(400, 30), xycoords='axes points',
                    size=20, ha='right', va='top',
                    bbox=dict(boxstyle='round', fc='w'), color='red')
        ax.set_title(title, fontsize=18)
        plt.xticks(size=18)
        ax.set_ylabel("Montant crédit")
        ax.set_xlabel("Revenus total")
        st.sidebar.pyplot(fig)
    elif sub_choix == 'Status familial':
        fig, ax = plt.subplots(figsize=(12, 10))
        ax = sns.countplot(x="INCOME_BINNED", hue="NAME_FAMILY_STATUS", data=df_income)
        title="Répartition des revenus des clients selon leur status familial"
        ax.set_title(title, fontsize=18)
        plt.xticks(size=18, rotation=75)
        ax.set_xlabel("Revenus")
        ax.set_ylabel("")
        plt.setp(ax.get_legend().get_texts(), fontsize='16') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='14') # for legend title
        barPerc(df_income, 'INCOME_BINNED', ax)
        st.sidebar.pyplot(fig)
    elif sub_choix == 'Type de revenus':
        fig, ax = plt.subplots(figsize=(12, 10))
        ax = sns.countplot(x="INCOME_BINNED", hue="NAME_INCOME_TYPE", data=df_income)
        title="Répartition des revenus des clients selon le type de revenus"
        ax.set_title(title, fontsize=18)
        plt.xticks(size=18, rotation=75)
        ax.set_xlabel("Revenus")
        ax.set_ylabel("")
        plt.setp(ax.get_legend().get_texts(), fontsize='16') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='14') # for legend title
        #barPerc(df_income, 'INCOME_BINNED', ax)
        st.sidebar.pyplot(fig)
    elif sub_choix == 'Education':
        fig, ax = plt.subplots(figsize=(12, 10))
        ax = sns.countplot(x="INCOME_BINNED", hue="NAME_EDUCATION_TYPE", data=df_income)
        title="Répartition des revenus des clients selon leur niveau d'éducation"
        ax.set_title(title, fontsize=18)
        plt.xticks(size=18, rotation=75)
        plt.legend(loc='upper left')
        ax.set_xlabel("Revenus")
        ax.set_ylabel("")
        plt.setp(ax.get_legend().get_texts(), fontsize='16') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='14') # for legend title
        barPerc(df_income, 'INCOME_BINNED', ax)
        st.sidebar.pyplot(fig)
    elif sub_choix == 'Possesion voiture':
        fig, ax = plt.subplots(figsize=(12, 10))
        ax = sns.countplot(x="INCOME_BINNED", hue="FLAG_OWN_CAR", data=df_income, palette=['blue',"green"])
        title="Répartition des revenus des clients selon si ils possèdent une voiture"
        ax.set_title(title, fontsize=16)
        plt.xticks(size=18, rotation=75)
        ax.set_xlabel("Revenus")
        ax.set_ylabel("")
        plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='18') # for legend title
        barPerc(df_income, 'INCOME_BINNED', ax)
        st.sidebar.pyplot(fig)
    elif sub_choix == 'Possession logement':
        fig, ax = plt.subplots(figsize=(12, 10))
        ax = sns.countplot(x="INCOME_BINNED", hue="FLAG_OWN_REALTY", data=df_income, palette=['yellowgreen',"skyblue"])
        title="Répartition des revenus des clients selon si ils possèdent leur logement"
        ax.set_title(title, fontsize=16)
        plt.xticks(size=18, rotation=75)
        plt.legend(loc='upper left')
        ax.set_xlabel("Revenus")
        ax.set_ylabel("")
        plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='18') # for legend title
        barPerc(df_income, 'INCOME_BINNED', ax)
        st.sidebar.pyplot(fig)
    elif sub_choix == 'Type de contrat':
        fig, ax = plt.subplots(figsize=(12, 10))
        ax = sns.countplot(x="INCOME_BINNED", hue="NAME_CONTRACT_TYPE", data=df_income, palette=['burlywood',"cornflowerblue"])
        title="Répartition des revenus des clients selon le type de contrat"
        ax.set_title(title, fontsize=16)
        plt.xticks(size=18, rotation=75)
        plt.legend(loc='upper left')
        ax.set_xlabel("Revenus")
        ax.set_ylabel("")
        plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='18') # for legend title
        barPerc(df_income, 'INCOME_BINNED', ax)
        st.sidebar.pyplot(fig)
    elif sub_choix == 'Nombre enfant':
        fig, ax = plt.subplots(figsize=(12, 10))
        ax = sns.countplot(x="INCOME_BINNED", hue="CNT_CHILDREN", data=df_income)
        title="Répartition des revenus des clients selon le nombre d'enfant"
        ax.set_title(title, fontsize=16)
        plt.xticks(size=18, rotation=75)
        plt.legend(loc='upper left')
        ax.set_xlabel("Revenus")
        ax.set_ylabel("")
        plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='18') # for legend title
        #barPerc(df_income, 'INCOME_BINNED', ax)
        st.sidebar.pyplot(fig)
    elif sub_choix == 'Type de logement':
        fig, ax = plt.subplots(figsize=(12, 10))
        ax = sns.countplot(x="INCOME_BINNED", hue="NAME_HOUSING_TYPE", data=df_income)
        title="Répartition des revenus des clients selon le type de logement"
        ax.set_title(title, fontsize=16)
        plt.xticks(size=18, rotation=75)
        plt.legend(loc='upper left')
        ax.set_xlabel("Revenus")
        ax.set_ylabel("")
        plt.setp(ax.get_legend().get_texts(), fontsize='20') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='18') # for legend title
        barPerc(df_income, 'INCOME_BINNED', ax)
        st.sidebar.pyplot(fig)
    sub_choix = None
#*------------------------------------------------------------------------------------------*
if choix == 'Montant du crédit':
    df_credit = pd.DataFrame(data_origin)
    #df_credit = df_credit.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
    df_credit['CREDIT_BINNED'] = pd.cut(df_credit['AMT_CREDIT'], bins=list(range(40000,2160000,150000)))
    df_credit['AGE_BINNED'] = pd.cut(df_credit['AGE'], bins=list(range(20,71,10)))

    sub_graph=['Par rapport à quelle variable ?', 'Sans filtre (all)', 'Sexe', 'Age', 'Status familial', 'Montant des revenus', 'Education', 'Type de logement','Possesion voiture', 'Possession logement', 'Type de contrat', 'Nombre enfant']
    sub_choix = st.sidebar.selectbox("", sub_graph)
    if sub_choix == 'Sans filtre (all)':
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.histplot(df_credit["AMT_CREDIT"], color="violet", bins=10)
        ax.set(title='Crédit des clients', xlabel='Credit (USD)', ylabel='')
        st.sidebar.pyplot(fig)
    elif sub_choix == 'Sexe':
        fig, ax = plt.subplots(figsize=(12, 10))
        ax = sns.countplot(x="CREDIT_BINNED", hue="CODE_GENDER", data=df_credit, palette=['violet',"blueviolet"])
        title="Répartition des crédits des clients selon leur genre"
        ax.set_title(title, fontsize=16)
        plt.xticks(size=18, rotation=80)
        ax.set_xlabel("Credit")
        ax.set_ylabel("")
        plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='18') # for legend title
        barPerc(df_credit, 'CREDIT_BINNED', ax)
        st.sidebar.pyplot(fig)
    elif sub_choix == 'Age':
        graph=['Choix', 'Scatterplot', 'Barplot']
        choice = st.sidebar.selectbox("Type de graphe", graph)
        if choice == 'Scatterplot':
            subset = df_credit.sample(n=2000)
            fig, ax = plt.subplots(figsize=(12, 10))
            ax = sns.regplot(x="AGE", y="AMT_CREDIT", data=subset, scatter_kws={"color": "lightcoral"}, line_kws={"color": "red"})
            ax.annotate('Ligne de régression', xy=(400, 30), xycoords='axes points',
                        size=20, ha='right', va='top',
                        bbox=dict(boxstyle='round', fc='w'), color='red')
            title="Relation entre les crédits globaux et l'âge des clients"
            ax.set_title(title, fontsize=18)
            plt.xticks(size=18)
            ax.set_xlabel("Age des clients", fontsize=16)
            ax.set_ylabel("Crédit total", fontsize=16)
            st.sidebar.pyplot(fig)

        elif choice == 'Barplot':
            age_data = df_credit[['AGE', 'CREDIT_BINNED']]
            income_groups = age_data.groupby("CREDIT_BINNED").mean().round(1)

            fig = plt.figure(figsize=(12, 10))
            ax1 = plt.subplot(3, 1, 1)
            sns.barplot(data = age_data, x = 'CREDIT_BINNED', y = 'AGE', palette = 'rocket', ci = 'sd')
            title="Age moyen des clients selon leur tranche de crédit"
            ax1.set_title(title, fontsize=18)
            plt.xticks(rotation = 75, size=18)
            ax1.set_xlabel("Tranches de crédit", fontsize=16)
            ax1.set_ylabel("Age moyen des clients", fontsize=16)

            ax2 = plt.subplot(3, 1, 3)
            sns.lineplot(data = income_groups, x = income_groups.index.astype(str), y = 'AGE', ci = None)
            title="Age moyen des clients selon leur tranche de crédit"
            ax2.set_title(title, fontsize=18)
            plt.xticks(rotation = 75, size=18)
            ax2.set_xlabel("Tranches de crédit", fontsize=16)
            ax2.set_ylabel("Age moyen des clients", fontsize=16)
            st.sidebar.pyplot(fig)

    elif sub_choix == 'Status familial':
        fig, ax = plt.subplots(figsize=(12, 10))
        ax = sns.countplot(x="CREDIT_BINNED", hue="NAME_FAMILY_STATUS", data=df_credit)
        title="Répartition des crédits des clients selon leur status familial"
        ax.set_title(title, fontsize=18)
        plt.xticks(size=18, rotation=75)
        ax.set_xlabel("Credit")
        ax.set_ylabel("")
        plt.setp(ax.get_legend().get_texts(), fontsize='16') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='14') # for legend title
        barPerc(df_credit, 'CREDIT_BINNED', ax)
        st.sidebar.pyplot(fig)
    elif sub_choix == 'Montant des revenus':
        subset = df_credit.sample(n=2000)
        fig, ax = plt.subplots(figsize=(12, 10))
        ax = sns.regplot(x="AMT_CREDIT", y="AMT_INCOME_TOTAL", data=subset, scatter_kws={"color": "lightcoral"}, line_kws={"color": "red"})
        ax.annotate('Ligne de régression', xy=(400, 30), xycoords='axes points',
                    size=20, ha='right', va='top',
                    bbox=dict(boxstyle='round', fc='w'), color='red')
        title="Relation entre le montant des prêts et les revenus globaux"
        ax.set_title(title, fontsize=18)
        plt.xticks(size=18)
        ax.set_xlabel("Montant crédit")
        ax.set_ylabel("Revenus total")
        st.sidebar.pyplot(fig)
    elif sub_choix == 'Education':
        fig, ax = plt.subplots(figsize=(12, 10))
        ax = sns.countplot(x="CREDIT_BINNED", hue="NAME_EDUCATION_TYPE", data=df_credit)
        title="Répartition des crédits des clients selon leur niveau d'éducation"
        ax.set_title(title, fontsize=18)
        plt.xticks(size=18, rotation=75)
        plt.legend(loc='upper right')
        ax.set_xlabel("Credit")
        ax.set_ylabel("")
        plt.setp(ax.get_legend().get_texts(), fontsize='16') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='14') # for legend title
        barPerc(df_credit, 'CREDIT_BINNED', ax)
        st.sidebar.pyplot(fig)
    elif sub_choix == 'Type de logement':
        fig, ax = plt.subplots(figsize=(12, 10))
        ax = sns.countplot(x="CREDIT_BINNED", hue="NAME_HOUSING_TYPE", data=df_credit)
        title="Répartition des crédits des clients selon le type de logement"
        ax.set_title(title, fontsize=16)
        plt.xticks(size=18, rotation=75)
        plt.legend(loc='upper right')
        ax.set_xlabel("Credit")
        ax.set_ylabel("")
        plt.setp(ax.get_legend().get_texts(), fontsize='20') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='18') # for legend title
        barPerc(df_credit, 'CREDIT_BINNED', ax)
        st.sidebar.pyplot(fig)
    elif sub_choix == 'Possesion voiture':
        fig, ax = plt.subplots(figsize=(12, 10))
        ax = sns.countplot(x="CREDIT_BINNED", hue="FLAG_OWN_CAR", data=df_credit, palette=['blue',"green"])
        title="Répartition des crédits des clients selon si ils possèdent une voiture"
        ax.set_title(title, fontsize=16)
        plt.xticks(size=18, rotation=75)
        ax.set_xlabel("Crédit")
        ax.set_ylabel("")
        plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='18') # for legend title
        barPerc(df_credit, 'CREDIT_BINNED', ax)
        st.sidebar.pyplot(fig)
    elif sub_choix == 'Possession logement':
        fig, ax = plt.subplots(figsize=(12, 10))
        ax = sns.countplot(x="CREDIT_BINNED", hue="FLAG_OWN_REALTY", data=df_credit, palette=['yellowgreen',"skyblue"])
        title="Répartition des crédits des clients selon si ils possèdent leur logement"
        ax.set_title(title, fontsize=16)
        plt.xticks(size=18, rotation=75)
        plt.legend(loc='upper right')
        ax.set_xlabel("Crédit")
        ax.set_ylabel("")
        plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='18') # for legend title
        barPerc(df_credit, 'CREDIT_BINNED', ax)
        st.sidebar.pyplot(fig)
    elif sub_choix == 'Type de contrat':
        fig, ax = plt.subplots(figsize=(12, 10))
        ax = sns.countplot(x="CREDIT_BINNED", hue="NAME_CONTRACT_TYPE", data=df_credit, palette=['burlywood',"cornflowerblue"])
        title="Répartition des crédits des clients selon le type de contrat"
        ax.set_title(title, fontsize=16)
        plt.xticks(size=18, rotation=75)
        plt.legend(loc='upper right')
        ax.set_xlabel("Crédit")
        ax.set_ylabel("")
        plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='18') # for legend title
        barPerc(df_credit, 'CREDIT_BINNED', ax)
        st.sidebar.pyplot(fig)
    elif sub_choix == 'Nombre enfant':
        fig, ax = plt.subplots(figsize=(12, 10))
        ax = sns.countplot(x="CREDIT_BINNED", hue="CNT_CHILDREN", data=df_credit)
        title="Répartition des crédits des clients selon le nombre d'enfant"
        ax.set_title(title, fontsize=16)
        plt.xticks(size=18, rotation=75)
        plt.legend(loc='upper right')
        ax.set_xlabel("Crédit")
        ax.set_ylabel("")
        plt.setp(ax.get_legend().get_texts(), fontsize='14') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='12') # for legend title
        #barPerc(df_credit, 'CREDIT_BINNED', ax)
        st.sidebar.pyplot(fig)
    sub_choix = None
#*------------------------------------------------------------------------------------------*
if choix == 'Age':
    df_age = pd.DataFrame(data_origin)
    #df_credit = df_credit.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
    df_age['AGE_BINNED'] = pd.cut(df_age['AGE'], bins=list(range(20,71,10)))

    sub_graph=['Par rapport à quelle variable ?', 'Sans filtre (all)', 'Nombre enfant', 'Status familial', 'Type de contrat','Possesion voiture', 'Possession logement', 'Type de revenus', 'Education', 'Count']
    sub_choix = st.sidebar.selectbox("", sub_graph)
    if sub_choix == 'Sans filtre (all)':
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.histplot(df_age["AGE"], color="violet", bins=20)
        ax.set(title='Age des clients', xlabel='Age (ans)', ylabel='')
        st.sidebar.pyplot(fig)
    elif sub_choix == 'Nombre enfant':
        fig, ax = plt.subplots(figsize=(12, 10))
        ax = sns.countplot(x="AGE_BINNED", hue="CNT_CHILDREN", data=df_age)
        title="Répartition des âges des clients selon le nombre d'enfant"
        ax.set_title(title, fontsize=16)
        plt.xticks(size=18, rotation=75)
        plt.legend(loc='upper left')
        ax.set_xlabel("Age des clients (ans)")
        ax.set_ylabel("")
        plt.setp(ax.get_legend().get_texts(), fontsize='12') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='10') # for legend title
        #barPerc(df_age, 'AGE_BINNED', ax)
        st.sidebar.pyplot(fig)
    elif sub_choix == 'Status familial':
        fig, ax = plt.subplots(figsize=(12, 10))
        ax = sns.countplot(x="AGE_BINNED", hue="NAME_FAMILY_STATUS", data=df_age)
        title="Répartition des âges des clients selon leur status familial"
        ax.set_title(title, fontsize=18)
        plt.xticks(size=18, rotation=75)
        ax.set_xlabel("Age des clients (ans)")
        ax.set_ylabel("")
        plt.setp(ax.get_legend().get_texts(), fontsize='16') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='14') # for legend title
        barPerc(df_age, 'AGE_BINNED', ax)
        st.sidebar.pyplot(fig)
    elif sub_choix == 'Type de contrat':
        fig, ax = plt.subplots(figsize=(12, 10))
        ax = sns.countplot(x="AGE_BINNED", hue="NAME_CONTRACT_TYPE", data=df_age, palette=['burlywood',"cornflowerblue"])
        title="Répartition des âges des clients selon le type de contrat"
        ax.set_title(title, fontsize=16)
        plt.xticks(size=18, rotation=75)
        #plt.legend(loc='upper right')
        ax.set_xlabel("Age des clients (ans)")
        ax.set_ylabel("")
        plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='18') # for legend title
        barPerc(df_age, 'AGE_BINNED', ax)
        st.sidebar.pyplot(fig)
    elif sub_choix == 'Possesion voiture':
        fig, ax = plt.subplots(figsize=(12, 10))
        ax = sns.countplot(x="AGE_BINNED", hue="FLAG_OWN_CAR", data=df_age, palette=['blue',"green"])
        title="Répartition des âges des clients selon si ils possèdent une voiture"
        ax.set_title(title, fontsize=16)
        plt.xticks(size=18, rotation=75)
        ax.set_xlabel("Age des clients (ans)")
        ax.set_ylabel("")
        plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='18') # for legend title
        barPerc(df_age, 'AGE_BINNED', ax)
        st.sidebar.pyplot(fig)
    elif sub_choix == 'Possession logement':
        fig, ax = plt.subplots(figsize=(12, 10))
        ax = sns.countplot(x="AGE_BINNED", hue="FLAG_OWN_REALTY", data=df_age, palette=['yellowgreen',"skyblue"])
        title="Répartition des âges des clients selon si ils possèdent leur logement"
        ax.set_title(title, fontsize=16)
        plt.xticks(size=18, rotation=75)
        #plt.legend(loc='upper right')
        ax.set_xlabel("Age des clients (ans)")
        ax.set_ylabel("")
        plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='18') # for legend title
        barPerc(df_age, 'AGE_BINNED', ax)
        st.sidebar.pyplot(fig)
    elif sub_choix == 'Type de revenus':
        fig, ax = plt.subplots(figsize=(12, 10))
        ax = sns.countplot(x="AGE_BINNED", hue="NAME_INCOME_TYPE", data=df_age)
        title="Répartition des revenus des clients selon le type de revenus"
        ax.set_title(title, fontsize=18)
        plt.xticks(size=18, rotation=75)
        ax.set_xlabel("Age des clients (ans)")
        ax.set_ylabel("")
        plt.legend(loc='upper right')
        plt.setp(ax.get_legend().get_texts(), fontsize='16') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='14') # for legend title
        #barPerc(df_age, 'AGE_BINNED', ax)
        st.sidebar.pyplot(fig)
    elif sub_choix == 'Education':
        fig, ax = plt.subplots(figsize=(12, 10))
        ax = sns.countplot(x="AGE_BINNED", hue="NAME_EDUCATION_TYPE", data=df_age)
        title="Répartition des âges des clients selon leur niveau d'éducation"
        ax.set_title(title, fontsize=18)
        plt.xticks(size=18, rotation=75)
        #plt.legend(loc='upper left')
        ax.set_xlabel("Age des clients (ans)")
        ax.set_ylabel("")
        plt.setp(ax.get_legend().get_texts(), fontsize='16') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='14') # for legend title
        barPerc(df_age, 'AGE_BINNED', ax)
        st.sidebar.pyplot(fig)

    elif sub_choix == 'Count':
        age_data = df_age[['AGE', 'AGE_BINNED']]
        income_groups = age_data.groupby("AGE_BINNED").count()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(data = income_groups, x = income_groups.index.astype(str), y = income_groups['AGE'], palette = 'rocket')
        title="Nombre de clients selon leur tranche d'âge"
        ax.set_title(title, fontsize=18)
        plt.xticks(rotation = 75, size=18)
        ax.set_xlabel("Tranches d'âge", fontsize=16)
        ax.set_ylabel("Nombre des clients", fontsize=16)
        st.sidebar.pyplot(fig)

    sub_choix = None
choix = None


####################################################################################
####################################################################################
####################################################################################
#Analyse détaillée pour chaque client
def classify_client(model, ID, df, seuil):
    ID = int(ID)
    X = df[df.index == ID]
    #X = X.drop(['TARGET'], axis=1) #if df_train_imputes.csv
    probability_default_payment = model.predict_proba(X)[:, 1]
    if probability_default_payment >= seuil:
        prediction = "Prêt NON Accordé"
    else:
        prediction = "Prêt Accordé"
    return probability_default_payment, prediction


#Main Page
st.selectbox("Pour information : Liste des identifiants possibles", all_id_client)
st.text("")
id_client = st.text_input("Veuillez entrer l'Identifiant d'un client")


if id_client != "":
    id_client = int(id_client)
    if id_client not in all_id_client:
          st.write("Ce client n'est pas répertorié")
          st.text("")
    else :
          identite_client = data_origin[data_origin.index == int(id_client)]
          predict_client = data_clean[data_clean.index == int(id_client)]
          client_without_standard = data_clean_without_standard[data_clean_without_standard.index == int(id_client)]

          st.text("")
          st.text("")
          st.text("")
          original_title = '<p style="font-size: 20px; text-align: center "><u>Informations générales dont nous disposons : </u> </p>'

          st.markdown(original_title, unsafe_allow_html=True)


          st.write("**Genre : **", identite_client["CODE_GENDER"].values[0])
          st.write("**Age : ** {:.0f}".format(identite_client["AGE"].values[0]), 'ans')

          #Age distribution plot
          fig, ax = plt.subplots(figsize=(10, 5))
          sns.histplot(data_origin["AGE"], color="orchid", bins=20)
          ax.axvline(int(identite_client["AGE"]), color="red", linestyle='dashed')
          ax.set(title='Age des clients', xlabel='Age (ans)', ylabel='')
          st.pyplot(fig)


          st.write("**Status Familial : **", identite_client["NAME_FAMILY_STATUS"].values[0])
          st.write("**Nombre d'enfant : **{:.0f}".format(identite_client["CNT_CHILDREN"].values[0]))
          st.write("**Possession d'une voiture : **", identite_client["FLAG_OWN_CAR"].values[0])
          st.write("**Possession de votre propre logement : **", identite_client["FLAG_OWN_REALTY"].values[0])
          st.write("**Type de logement habité : **", identite_client["NAME_HOUSING_TYPE"].values[0])
          st.write("**Revenu total (USD) : **{:.2f}".format(identite_client["AMT_INCOME_TOTAL"].values[0]))

          df_income = pd.DataFrame(data_origin["AMT_INCOME_TOTAL"])
          df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
          #Age distribution plot
          fig, ax = plt.subplots(figsize=(10, 5))
          sns.histplot(df_income["AMT_INCOME_TOTAL"], color="orchid", bins=20)
          ax.axvline(int(identite_client["AMT_INCOME_TOTAL"].values[0]), color="red", linestyle='dashed')
          ax.set(title='Revenu des clients', xlabel='Revenu (USD)', ylabel='')
          st.pyplot(fig)


          st.write("**Type de revenus : ** ", identite_client["NAME_INCOME_TYPE"].values[0])

          st.write("**Nombre de demandes de prêt : ** {:.0f}" .format(identite_client["PREVIOUS_APPLICATION_COUNT"].values[0]))
          st.write("**Type de prêt le plus demandé : **", identite_client["MOST_CREDIT_TYPE"].values[0])
          st.write("**Nombre de prêts accordés précédents : **{:.0f}".format(identite_client["PREVIOUS_LOANS_COUNT"].values[0]))
          st.write("**Montant total à payer pour les crédits :** {:.2f}".format(identite_client["AMT_CREDIT"].values[0]))
          st.write("**Pourcentage remboursement crédit sur les revenus total : **{:.2f}".format(identite_client["ANNUITY_CREDIT_PERCENT_INCOME"].values[0]*100), "%")
          st.write("**Montant du crédit remboursé par an : **{:.2f}".format(identite_client["AMT_ANNUITY"].values[0]))
          st.write("**Durée remboursement crédit en année : **{:.2f}".format(identite_client["CREDIT_REFUND_TIME"].values[0]))

          st.text("")
          st.text("")
          st.text("")


          probability_default_payment, prediction = classify_client(lgbm, id_client, data_clean, seuil)
          original_title = '<p style="font-size: 20px;text-align: center;"> <u>Probabilité d\'être en défaut de paiement : </u> </p>'
          st.markdown(original_title, unsafe_allow_html=True)
#####################################################
          options = {
                    "tooltip": {"formatter": "{a} <br/>{b} : {c}%"},
                    "series": [
                        {
                            "name": "défaut de paiement",
                            "type": "gauge",
                            "axisLine": {
                                "lineStyle": {
                                    "width": 10,
                                },
                            },
                            "progress": {"show": "true", "width": 10},
                            "detail": {"valueAnimation": "true", "formatter": "{value}"},
                            "data": [{"value": (probability_default_payment[0]*100).round(2), "name": "Score"}],
                        }
                    ],
                }
        
        
          st_echarts(options=options, width="100%", key=0)

#####################################################

          original_title = '<p style="font-size: 20px;text-align: center;"> <u>Conclusion : </u> </p>'
          st.markdown(original_title, unsafe_allow_html=True)

          if prediction == "Prêt Accordé":
              original_title = '<p style="font-family:Courier; color:GREEN; font-size:70px; text-align: center;">{}</p>'.format(prediction)
              st.markdown(original_title, unsafe_allow_html=True)
          else :
              original_title = '<p style="font-family:Courier; color:red; font-size:70px; text-align: center;">{}</p>'.format(prediction)
              st.markdown(original_title, unsafe_allow_html=True)


          st.text("")
          st.text("")

         #Feature importance / description
          original_title = '<p style="font-size: 20px;text-align: center;"> <u>Quelles sont les informations les plus importantes dans la prédiction ?</u> </p>'
          st.markdown(original_title, unsafe_allow_html=True)
          feature_imp = pd.DataFrame(sorted(zip(lgbm.booster_.feature_importance(importance_type='gain'), data_clean.columns)), columns=['Value','Feature'])

          fig, ax = plt.subplots(figsize=(10, 5))
          sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(5))
          ax.set(title='Importance des informations', xlabel='', ylabel='')
          st.pyplot(fig)

          st.write("Notre modèle de prédiction se base essentiellement sur les éléments suivants :")
          st.write("Les éléments sont ici classés selon leur importance. C'est à dire que le résultat du premier éléments a un poids plus important que les autres dans la décision d'accord du prêt.")
          data=feature_imp.sort_values(by="Value", ascending=False).head(5)
          features_list = list(data['Feature'])
          for feature in features_list:
              st.table(description.loc[description.index == feature][:1])

              #, figsize=(10, 5) showfliers=False, showcaps=True, showmeans=True
              if feature == 'DAYS_BIRTH':
                  st.write("Cette caractéristique correspond au nombre de jours depuis votre naissance ce qui est analogue à votre âge. Les plus jeunes clients ont dû mal à rembourser les prêts.")

                  fig, axes = plt.subplots(ncols=2, figsize=(5, 5)) #bins=10,
                  sns.histplot(data_clean_without_standard["AGE"], color="orchid", bins=10, ax=axes[0])
                  axes[0].axvline(int(client_without_standard["AGE"].values[0]), color="red", linestyle='dashed')
                  axes[0].set(title='', xlabel='', ylabel='')
                  sns.boxplot(y="AGE", data=data_clean_without_standard, ax=axes[1], color='tan', showfliers=False, showcaps=True, showmeans=True, whiskerprops={'linestyle': '-', 'linewidth': 2, 'color' : 'blue'}, capprops={'linestyle': '-', 'linewidth': 2, 'color':'blue'}, meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"black","markersize":"10"})
                  axes[1].axhline(int(client_without_standard["AGE"].values[0]), color="red", linestyle='dashed')
                  axes[1].set(title='', xlabel='', ylabel='')
                  st.pyplot(fig)

                  colData=data_clean_without_standard["AGE"]
                  Q1 = np.percentile(colData, 25)

                  st.write("**25% des client ont un âge inférieur ou égal à : **{:.0f}".format(Q1))
                  st.write("**Votre Age : **{:.0f}".format(client_without_standard["AGE"].values[0]))

                  if client_without_standard["AGE"].values[0] > Q1 :
                      st.write("Votre âge est supérieur à 25% des clients pour cette caractéristique ce qui joue en votre faveur pour l'accord de prêt. ")
                      st.write(' ')
                  else :
                      st.write("Votre âge est inférieur ou égal à 25% des clients pour cette caractéristique ce qui joue en votre défaveur pour l'accord de prêt. ")
                      st.write(' ')
              elif feature == 'CREDIT_REFUND_TIME':

                  fig, axes = plt.subplots(ncols=2, figsize=(5, 5)) #bins=10,
                  sns.histplot(data_clean_without_standard[feature], color="orchid", bins=10, ax=axes[0])
                  axes[0].axvline(client_without_standard[feature].values[0].round(2), color="red", linestyle='dashed')

                  axes[0].set(title='', xlabel='', ylabel='')
                  sns.boxplot(y=feature, data=data_clean_without_standard, ax=axes[1], color='tan', showfliers=False, showcaps=True, showmeans=True, whiskerprops={'linestyle': '-', 'linewidth': 2, 'color' : 'blue'}, capprops={'linestyle': '-', 'linewidth': 2, 'color':'blue'}, meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"black","markersize":"10"})

                  axes[1].axhline(client_without_standard[feature].values[0].round(2), color="red", linestyle='dashed')
                  axes[1].set(title='', xlabel='', ylabel='')
                  st.pyplot(fig)


                  colData=data_clean_without_standard['CREDIT_REFUND_TIME']
                  #med = np.percentile(colData, 50)
                  mean = np.mean(colData)

                  st.write("**Le score moyen (point rouge) pour cette caractéristique est de : **{:.2f}".format(mean))
                  st.write("**Votre score pour cette caractéristique : **{:.2f}".format(client_without_standard[feature].values[0]))


                  if client_without_standard[feature].values[0] > mean :
                      st.write("Votre score est supérieur à la moyenne de cette caractéristique ce qui joue à priori en votre défaveur pour l'accord de prêt. ")
                      st.write(' ')
                  else :
                      st.write("Votre score est inférieur ou égal à la moyenne de cette caractéristique ce qui joue à priori en votre faveur pour l'accord de prêt. ")
                      st.write(' ')

              else:

                fig, axes = plt.subplots(ncols=2, figsize=(5, 5)) #bins=10,
                sns.histplot(data_clean_without_standard[feature], color="orchid", bins=10, ax=axes[0])
                axes[0].axvline(client_without_standard[feature].values[0].round(2), color="red", linestyle='dashed')

                axes[0].set(title='', xlabel='', ylabel='')
                sns.boxplot(y=feature, data=data_clean_without_standard, ax=axes[1], color='tan', showfliers=False, showcaps=True, showmeans=True, whiskerprops={'linestyle': '-', 'linewidth': 2, 'color' : 'blue'}, capprops={'linestyle': '-', 'linewidth': 2, 'color':'blue'}, meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"black","markersize":"10"})

                axes[1].axhline(client_without_standard[feature].values[0].round(2), color="red", linestyle='dashed')
                axes[1].set(title='', xlabel='', ylabel='')
                st.pyplot(fig)


                colData=data_clean_without_standard[feature]
                #med = np.percentile(colData, 50)
                mean = np.mean(colData)

                st.write("**Le score moyen (point rouge) pour cette caractéristique est de : **{:.2f}".format(mean))
                st.write("**Votre score pour cette caractéristique : **{:.2f}".format(client_without_standard[feature].values[0]))


                if client_without_standard[feature].values[0] > mean :
                    st.write("Votre score est supérieur à la moyenne de cette caractéristique ce qui joue à priori en votre faveur pour l'accord de prêt. ")
                    st.write(' ')
                else :
                    st.write("Votre score est inférieur ou égal à la moyenne de cette caractéristique ce qui joue à priori en votre défaveur pour l'accord de prêt. ")
                    st.write(' ')

          chk_voisins = st.checkbox("Voir des clients similaires ?")
          if chk_voisins:
              df=data_clean.copy()
              def find_similiar_items(item_id):
                    tmp_df = df.sub(df.loc[item_id], axis='columns')
                    tmp_series = tmp_df.apply(np.square).apply(np.sum, axis=1)
                    #tmp_series.sort()
                    return tmp_series

              voisins = data_origin.loc[find_similiar_items(id_client).index].head(6)
              st.dataframe(voisins)
              chk_voisins = None
          else:
              st.markdown("<i>…</i>", unsafe_allow_html=True)

          id_client = None
