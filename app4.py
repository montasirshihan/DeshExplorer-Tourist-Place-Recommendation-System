import streamlit as st ##for building the UI of the website
import pickle
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity # Added missing import
from nltk.stem.porter import PorterStemmer

# 1. Initialize Stemmer 
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

# Load the saved models
maindf = joblib.load('main_df.joblib')
##similarity = joblib.load('similarity.joblib')
tidf=joblib.load("tfidf.joblib")


##recommendation function

def recommendation(district,p_type,budget):
    df=maindf.copy()

    
    user_query = stem(f"{p_type} {budget}")
    vector_query=tidf.transform([user_query])
    all_vectors=tidf.transform(df['tags'])

    scores = cosine_similarity(vector_query, all_vectors).flatten()
    df['score']=scores

    ##now APPLY FILTERS ------

    if district.lower()!="all":
        df = df[df['district'] == district.lower()]
    
    # Filter by Budget
    df = df[df['budget'] == budget.lower()]

    # Filter by Type (using contains for flexibility)
    df = df[df['type'].str.contains(p_type.lower())]

    if df.empty:
        return pd.DataFrame()
    
    # Sort by score and return top 20
    return df.sort_values(by='score', ascending=False).head(20)



#no 2: UI design

st.set_page_config(page_title="DeshExplorer", layout="wide")
st.title("DeshExplorer: Tourist Place Recommendation System")
st.markdown("Find Your Next Dream Destination Based on Your City, Budget and Taste!!")


##user input
col1,col2,col3= st.columns(3)

with col1:
    district_input= st.selectbox("Select District (Optional)", ["All", "Chattogram", "Dhaka", "Sylhet","Rajshahi", "Rangpur", "Khulna", "Cumilla", "Barisal"])
with col2:
    type_input= st.selectbox("Type of Place", ["beach", "hill", "historical", "nature", "museum", "religious"])
with col3:
    budget_input=st.selectbox("Budget Range", ["low", "medium", "high"])

##members=st.sidebar.number_input("Number of Members",min_value=1,value=1)

if st.button("EXPLORE!!"):

    results=recommendation(district_input,type_input,budget_input)


#Output Logic

    if results.empty or results['score'].max() == 0:
        st.warning("No exact matches found. Try changing the budget or type!")

    else :## show 20 best suggestion with googlemap link

        st.success(f"Showing top results for {type_input} in {district_input}")

        for i ,(index, row) in enumerate(results.iterrows(),start=1):
           with st.container():
                st.markdown(f"### {i}.{row['place'].title()}")
                st.write(f"**Description:** {row['description']}")

                col_left,col_right=st.columns(2)
                with col_left:
                    st.info(f"**Security:** {row['security']}")
                    st.write(f"**Route:** {row['route']}")
                with col_right:
                    st.warning(f"**Best Time:** {row['best_time']}")
                    st.write(f"**Avg Cost:** {row['avg_cost']} BDT")

                ##restuerent
                st.success(f"**Nearby Restaurants:** {row['nearby restaurants']}")
                 # Using a clean URL with no spaces
                st.markdown(f"[View on Google Maps]({row['map_link']})")
               
                st.divider()




