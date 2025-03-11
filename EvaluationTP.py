# Importation des bibliothèques de base nécessaires
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
#from sklearn.feature_selection import mutual_info_classif

#Importation du DataFrame
df = pd.read_csv('MathEdataset.csv',encoding='cp1252',delimiter=";")

import streamlit as st

#from sklearn.model_selection import train_test_split
st.sidebar.title("Menu")
st.sidebar.header("Analyse des données")
#st.sidebar.text("selectionner un des menus suivants pour continuer")
st.title("TP 1 IA ENS")
st.header("Partie 1")

#Ajout d'un message defilant
st.markdown(
    """
    <marquee behavior="scroll" direction="left" scrollamount="5">
          Bienvenue sur notre application de traitement de données !  
    </marquee>
    """,
    unsafe_allow_html=True
)

#Ajouter un élément à la page d'accueil
#st.write("Hello world !")

# Ici je vais inserer les menus interactifs dans ma barre de menu
menu = st.sidebar.radio(
    "Sélectionnez une page:", 
    ["Accueil", "Description des données", "Visualisation des données", "Interogation des données", "Contacts"]
)

# Affichage du contenu selon le choix
if menu == "Accueil":
    st.subheader("Bienvenue dans le module d'analyse des données")
    st.write("Pour effectuer une tache, retourner sur le menu correspondant")

if menu == "Description des données":
    st.subheader("Bienvenu dans le menu de description des données")
    st.write("Ici vous pouvez afficher ou calculer les caracteristiques de de dispersion des données")

elif menu == "Visualisation des données":
    st.subheader("Bienvenue dans la page de visualisation des données")
    st.write("Cette page vous offre la possibilité de faire des representations ")

elif menu == "Interogation des données":
    st.subheader("Bienvenue dans la page d'Interrogation des données")
    st.write("Je vous offre ici la possibilité d'interroger des données à votre convenancer")
    st.write("Selectionne la longueur de sepal")
    longueurSepal = st.slider(
        "Choisissez une valeur :", 
        min_value=0, 
        max_value=20,                
        value=50, 
        step=5
    )
    st.write(f"La longueur du sepal sélectionnée : {longueurSepal}")

if menu == "Contacts":
    st.subheader("Contactez-nous:")
    st.write("Téléphone: +237 697 848 562")
    st.write("WhatsApp : +237 697 848 562")
    st.write("Email    : tezemfrederic@gmail.com")
#VISUALISATION GLOBALE DE DONNEES

#Afficharge des premières lignes de notre DataFrame
print(df.head())

#Suppression de la colonne "Keywords" dans notre DataFrame
df=df.drop('Keywords',axis=1)

#Affichage des colonnes de notre DataFrame après la suppression de la colonne "Keywords"
print(df.columns)

#Affichage des Premières lignes de notre DataFrame après la suppression de la colonne "Keywords"
print(df.describe())

#ANALYSE DE LA COLONNE Student ID

#Effectif des categories de Student ID
effectifsStudentId=df['Student ID'].value_counts()
print(effectifsStudentId)

#Affichage du Diagramme en Baton
plt.bar(effectifsStudentId.index, effectifsStudentId.values, color=['blue','red'])
plt.title("Effectif des Student")
plt.xlabel("Student ID")
plt.ylabel("effectifsStudentId")
plt.show()

#données representées en secteurs
plt.figure(figsize=(8,8))
effectifsStudentId.plot.pie(autopct='%1.1f%%',startangle=90, colors=['green','red'], labels=effectifsStudentId.index)
plt.ylabel('effectifsStudentId')
plt.show()

#Diagrammes à barres groupés
group_donnees=df.groupby('Student ID').mean()
group_donnees.plot(kind='bar',figsize=(10,6))
plt.title('Moyenne des caracteristiques par élève')
plt.xlabel('Student ID')
plt.ylabel('Valeur moyenne')
#Afficher la legende
plt.legend(title='Caracteristiques')

#Affichage
plt.tight_layout()
plt.show()

#Analyse du Students Country

#Affichage des effectifs des categories des questions
effectifsStudentCountry=df['Student Country'].value_counts()
print(effectifsStudentCountry)

#Affichage du Diagramme en Baton
plt.bar(effectifsStudentCountry.index, effectifsStudentCountry.values, color=['blue','red'])
plt.title("Effectif des Student Country")
plt.xlabel("Student Country")
plt.ylabel("effectifsStudentCountry")
plt.show()

#données representées en secteurs
plt.figure(figsize=(8,8))
effectifsStudentCountry.plot.pie(autopct='%1.1f%%',startangle=0, colors=['green','red'], labels=effectifsStudentCountry.index)
plt.ylabel('effectifsStudentCountry')
plt.show()

#Diagrammes à barres groupés
group_donnees=df.groupby('Student Country').mean()
group_donnees.plot(kind='bar',figsize=(10,6))
plt.title('Moyenne des caracteristiques par élève')
plt.xlabel('Student Country')
plt.ylabel('Valeur moyenne')
#Afficher la legende
plt.legend(title='Caracteristiques')

#Affichage
plt.tight_layout()
plt.show()

#Anamyse des categories de question

#Affichage des effectifs des categories des questions
effectifsQuestionId=df['Question ID'].value_counts()
print(effectifsQuestionId)

#Affichage du Diagramme en Baton
plt.bar(effectifsQuestionId.index, effectifsQuestionId.values, color=['blue','red'])
plt.title("Effectif des Types de question")
plt.xlabel("Question Id")
plt.ylabel("effectifsQuestionId")
plt.show()

#données representées en secteurs
plt.figure(figsize=(8,8))
effectifsQuestionId.plot.pie(autopct='%1.1f%%',startangle=0, colors=['green','red'], labels=effectifsQuestionId.index)
plt.ylabel('effectifsQuestionId')
plt.show()

#Diagrammes à barres groupés
group_donnees=df.groupby('Question ID').mean()
group_donnees.plot(kind='bar',figsize=(10,6))
plt.title('Moyenne des caracteristiques par élève')
plt.xlabel('Question ID')
plt.ylabel('Valeur moyenne')
#Afficher la legende
plt.legend(title='Caracteristiques')

#Affichage
plt.tight_layout()
plt.show()

#Analyse des types de reponse

#Affichage des effectifs de type de reponse
effectifsTypeofAnswer=df['Type of Answer'].value_counts()
print(effectifsTypeofAnswer)

#Affichage du Diagramme en Baton
plt.bar(effectifsTypeofAnswer.index, effectifsTypeofAnswer.values, color=['blue','red'])
plt.title("Effectif des Types de question")
plt.xlabel("Typeof Answer")
plt.ylabel("effectifsTypeofAnswer")
plt.show()

#données representées en secteurs
plt.figure(figsize=(8,8))
effectifsTypeofAnswer.plot.pie(autopct='%1.1f%%',startangle=0, colors=['green','red'], labels=effectifsTypeofAnswer.index)
plt.ylabel('Typeof Answer')
plt.show()

#Diagrammes à barres groupés
group_donnees=df.groupby('Type of Answer').mean()
group_donnees.plot(kind='bar',figsize=(10,6))
plt.title('Moyenne des caracteristiques par élève')
plt.xlabel('Type of Answer')
plt.ylabel('Valeur moyenne')
#Afficher la legende
plt.legend(title='Caracteristiques')

#Affichage
plt.tight_layout()
plt.show()

# Séparer les caractéristiques et la cible
X = df.drop('Type of Answer', axis=1)
y = df['Type of Answer']

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

#labelencoder=LabelEncoder()
#X_train=labelencoder.fit_transform(X_train['Type of Answer'])
#print(X_train.dtypes)

# Normaliser les caractéristiques
scaler = StandardScaler()
X_train=X_train.select_dtypes(include=['number'])
X_test=X_test.select_dtypes(include=['number'])
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Créer le modèle KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Entraîner le modèle
knn.fit(X_train, y_train)

# Prédire les classes de l'ensemble de test
y_pred = knn.predict(X_test)

# Afficher la matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=df['Type of Answer'].unique(), yticklabels=df['Type of Answer'].unique())
plt.title('Matrice de confusion')
plt.xlabel('Prédictions')
plt.ylabel('Vraies classes')
plt.show()

# Calculer l'exactitude
accuracy = accuracy_score(y_test, y_pred)
print(f"Exactitude du modèle : {accuracy * 100:.2f}%")

# Afficher le rapport de classification
print("Rapport de classification :\n", classification_report(y_test,y_pred))
