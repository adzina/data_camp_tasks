import datadotworld as dw
import matplotlib as plt

# Import the city council votes dataset
dataset = dw.load_dataset("stephen-hoover/chicago-city-council-votes")

# Use describe() to review all the metadata that is downloaded with the dataset. 
# Print it to the screen using pp.pprint().
pp.pprint(dataset.describe())

# Use describe() again to get a description of a specific resource: alderman_votes. Print it to the screen.
alderman_votes = dataset.describe('alderman_votes')
print(alderman_votes)


# Use the dataframes property to assign the alderman_votes table to the variable votes_dataframe.
votes_dataframe = dataset.dataframes['alderman_votes']

# Use the pandas shape property to get rows/columns size for the `votes_dataframe` dataframe.
pp.pprint(votes_dataframe.shape)

# Use the pandas head function to print the first 3 rows of the `votes_dataframe` dataframe.
pp.pprint(votes_dataframe.head(3))


# We've loaded two datasets to use 'int_dataset' and 'fipsCodes_dataset'
int_dataset = dw.load_dataset('https://data.world/jonloyens/intermediate-data-world')
fipsCodes_dataset = dw.load_dataset('https://data.world/uscensusbureau/fips-state-codes')

## Create two dataframes: police_shootings from the 'fatal_police_shootings_data' table of int_dataset and state_abbrvs, from the 'statesfipscodes' table of fipsCodes_dataset
police_shootings = int_dataset.dataframes['fatal_police_shootings_data']
state_abbrvs = fipsCodes_dataset.dataframes['statesfipscodes']

## Merge the two datasets together on the state and stusab fields. Assign to a merged_dataframe variable.
merged_dataframe = police_shootings.merge(state_abbrvs, how = 'left', left_on = 'state', right_on='stusab')

## Add a 'citystate' column to the merged_dataframe dataframe, populating it with the concatinated values from the 'city' and 'state_name' columns, separated by ', '. 
merged_dataframe['citystate']=merged_dataframe['city']+','+merged_dataframe['state_name']

## Print first 5 rows of merged_dataframe
pp.pprint(merged_dataframe.head())





## Complete the SQL query to select all rows from the `unhcr_all` table where `Year` equals 2010. Assign the query string to a `sql_query` variable.
sql_query = "SELECT * FROM `unhcr_all` WHERE Year = 2010"

## Use the `query` method of the datadotworld module to run the `sql_query` against the `https://data.world/nrippner/refugee-host-nations` dataset. Assign the results to a `query2010` variable.
query2010 = dw.query('https://data.world/nrippner/refugee-host-nations',query = sql_query)

## Use the dataframe property of the resulting query to create a dataframe variable named `unhcr2010`
unhcr2010 = query2010.dataframe

## Print the first 5 rows using the head method.
pp.pprint(unhcr2010.head())


## Complete the SQL query to select state, the count of farmers markets (fmid), and average obesity rate from agriculture.`national-farmers-markets`.export, LEFT JOINED against health.`obesity-by-state-2014`.adult_obese on state and location
sql_query = "SELECT state, count(FMID) as count, Avg(obesity.value) as obesityAvg FROM export LEFT JOIN health.`obesity-by-state-2014`.`adult_obese` as obesity ON state = obesity.location GROUP BY state ORDER BY count desc"

## Use the `query` method of the datadotworld module to run the `sql_query` against the `https://data.world/agriculture/national-farmers-markets` dataset. Assign the results to a `queryResults` variable.
queryResults = dw.query("https://data.world/agriculture/national-farmers-markets",sql_query)

## Use the dataframes property of the resulting query to create a dataframe variable named `stateStats`
stateStats = queryResults.dataframe

## Plot the stateStats results using state as the x-axis (matplotlib is already imported)

stateStats.plot(x=stateStats["state"])

plt.show()

# We've written a SPARQL query for you and assigned it to the `sparql_query` variable: 
sparql_query = "PREFIX GOT: <https://tutorial.linked.data.world/d/sparqltutorial/> SELECT ?FName ?LName WHERE {?person GOT:col-got-house \"Stark\" . ?person GOT:col-got-fname ?FName . ?person GOT:col-got-lname ?LName .}"

# Use the pre-defined SPARQL query to query dataset http://data.world/tutorial/sparqltutorial and return the results to a queryResults variable
queryResults = dw.query("https://data.world/tutorial/sparqltutorial",sparql_query, query_type='sparql')

# Use the dataframe property of the resulting query to create a dataframe variable named `houseStark`
houseStark = queryResults.dataframe
# Use pp.pprint() to print the dataframe to the screen.
pp.pprint(houseStark)