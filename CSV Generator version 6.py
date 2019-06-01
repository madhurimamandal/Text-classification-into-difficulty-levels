#Importing modules
import pandas as pd
import features as f

#Creating feature sets
f2 = f.features() 
Dataframe_Easy = f2.create_dataframe()
Dataframe_Medium = f2.create_dataframe()
Dataframe_Tough = f2.create_dataframe()
    
#Easy
value1 = []
for i in range(0,33):
    value1.append(1)
df1 = pd.DataFrame({'Class':value1})
Dataframe_Easy = Dataframe_Easy.join(df1)

#Medium
value2 = []
for i in range(0,33):
    value2.append(2)
df2 = pd.DataFrame({'Class':value2})
Dataframe_Medium = Dataframe_Medium.join(df2)
 
#Hard
value3 = []
for i in range(0,33):
    value3.append(3)
df3 = pd.DataFrame({'Class':value3})
Dataframe_Tough = Dataframe_Tough.join(df3)
 
frames = [Dataframe_Easy, Dataframe_Medium, Dataframe_Tough]
 
#Final
Dataframe_Final = pd.concat(frames).reset_index(drop=True)

#Shuffling rows and storing
Dataframe_Final = Dataframe_Final.sample(frac=1).reset_index(drop=True)
Dataframe_Final.to_csv('Dataset(final).csv', encoding = 'utf-8', index = False)