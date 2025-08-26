import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.title('Location recommendation')


user_profiles = pd.read_csv("C:\\Users\\User\\Desktop\\user profiles.csv")
cities = pd.read_csv("C:\\Users\\User\\Desktop\cities dataset.csv")

minmax = MinMaxScaler()
cities_norm = minmax.fit_transform(cities)
cities_norm = pd.DataFrame(cities_norm, columns=cities.columns)

scaled_features = pd.DataFrame()
scaled_features[['Immigrants', 'Crime severity index', 'Housing cost index', 'Transportation cost index', 'Postsecondary certificate or diploma or degree', 'Employment rate', 'Population density per square kilometre', 'Commute by Public transit']] = minmax.fit_transform(cities[['Immigrants', 'Crime severity index', 'Housing cost index', 'Transportation cost index', 'Postsecondary certificate or diploma or degree', 'Employment rate', 'Population density per square kilometre', 'Commute by Public transit']])
scaled_features['Crime severity index'] = 1 - scaled_features['Crime severity index']
scaled_features['Housing cost index'] = 1 - scaled_features['Housing cost index']
scaled_features['Transportation cost index'] = 1 - scaled_features['Transportation cost index']

cities['diversity_score'] = scaled_features['Immigrants']
cities['safety_score'] = scaled_features['Crime severity index']
cities['living cost_score'] = scaled_features[['Housing cost index', 'Transportation cost index']].mean(axis=1)
cities['economic opp_score'] = scaled_features[['Employment rate', 'Postsecondary certificate or diploma or degree']].mean(axis=1)
cities['population density_score'] = scaled_features['Population density per square kilometre']
cities['public transit_score'] = scaled_features['Commute by Public transit']

merged = cities.merge(user_profiles,how='cross')
merged.fillna(1, inplace=True)

merged['match score'] = (
        (merged['safety preference']*(merged['safety_score']**10) )+
        (merged['economic_opportunity preference']*(merged['economic opp_score']**10) ) +
        (merged['living_cost preference']*(merged['living cost_score']**10) ) +
        (merged['diversity preference']*(merged['diversity_score']**10) ) +
        (merged['population_density preference']*(merged['population density_score'] **10) ) +
        (merged['public_transit preference']*(merged['public transit_score'] **10) )
)/6
merged['match score'] = minmax.fit_transform(merged[['match score']])
merged.drop(['diversity_score', 'safety_score', 'living cost_score', 'economic opp_score', 'public transit_score', 'population density_score'], axis=1, inplace=True)

x = merged.drop(['UserID', 'match score', 'Immigrants','Crime severity index', 'Housing cost index', 'Transportation cost index', 'Employment rate', 'Postsecondary certificate or diploma or degree','Population density per square kilometre', 'Commute by Public transit'], axis=1)
y = merged['match score']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

forest = RandomForestRegressor()
forest.fit(x_train, y_train)
y_pred = forest.predict(x_test)

#user inputs
safety = st.slider("Safety importance:", min_value=0.0, max_value=1.0, step=0.01)
economic = st.slider("Economic opportunity importance:", min_value=0.0, max_value=1.0, step=0.01)
cost = st.slider("Living cost importance:", min_value=0.0, max_value=1.0, step=0.01)
diversity = st.slider("Diversity importance:", min_value=0.0, max_value=1.0, step=0.01)
pop_density = st.slider("Population density importance:", min_value=0.0, max_value=1.0, step=0.01)
public_transit = st.slider("Public transit importance:", min_value=0.0, max_value=1.0, step=0.01)

user_pref = {"safety preference": safety,
    "economic_opportunity preference": economic,
    "living_cost preference": cost,
    "diversity preference": diversity,
    "population_density preference": pop_density,
    "public_transit preference": public_transit
}
user_df = pd.DataFrame([user_pref])

x = cities.drop(['Immigrants','Crime severity index', 'Housing cost index', 'Transportation cost index', 'Employment rate', 'Postsecondary certificate or diploma or degree', 'Population density per square kilometre', 'Commute by Public transit'], axis=1)
combined = x.merge(user_df, how='cross')

combined['score'] = forest.predict(combined)

recomendation = combined['score'].sort_values(ascending=False).head()

locations = pd.read_csv("C:\\Users\\User\\Desktop\\location names.csv")

for i in recomendation.index:
  st.write("top 5 matches for you are:", locations.loc[i, "precise location"])



