## EXNO-3-DS

### NAME: Surya SK

### REG.NO: 212222100052

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```

       import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df

```

<img width="1018" height="570" alt="image" src="https://github.com/user-attachments/assets/02a24da3-6816-4c0d-8473-4bf6d13959e9" />

```

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])

```

<img width="740" height="352" alt="Screenshot 2025-10-06 214134" src="https://github.com/user-attachments/assets/836460ca-dd66-47bf-bd63-5c7c4b683351" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df

```

<img width="722" height="528" alt="image" src="https://github.com/user-attachments/assets/29a82f7e-80ae-4ffc-bbeb-7062c1fbdbaf" />

```

le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc

```

<img width="644" height="569" alt="image" src="https://github.com/user-attachments/assets/f38834e2-b854-4176-a678-fb3358836f95" />


```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2

```

<img width="550" height="441" alt="image" src="https://github.com/user-attachments/assets/c17f158c-a0b2-4da1-824a-4775176838cc" />


```
pd.get_dummies(df2,columns=["nom_0"])

```
<img width="828" height="456" alt="image" src="https://github.com/user-attachments/assets/2a66ef17-3c8f-4244-a140-74b65ee20615" />

```

pip install --upgrade category_encoders

```
<img width="1382" height="428" alt="image" src="https://github.com/user-attachments/assets/45a48117-8a9f-4c1d-b351-88dcc09975eb" />

```

from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df

```
<img width="622" height="439" alt="image" src="https://github.com/user-attachments/assets/59829014-be57-4100-9276-ccca72f0ebc8" />

```

be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df

```
<img width="620" height="444" alt="image" src="https://github.com/user-attachments/assets/91b729d7-dca9-450e-a95b-a88fae2927f9" />

```

dfb=pd.concat([df,nd],axis=1)
dfb

```
<img width="877" height="441" alt="image" src="https://github.com/user-attachments/assets/9019cb39-498c-46bb-ab40-6e9e32a4302a" />

```

from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC

```

<img width="709" height="445" alt="image" src="https://github.com/user-attachments/assets/75c0700e-6e84-4156-95bc-5a79cb1709f1" />

```

import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df

```

<img width="986" height="498" alt="image" src="https://github.com/user-attachments/assets/2e36cb4a-071b-4f6d-8171-2f9622b94200" />


```

df.skew()

```

<img width="464" height="243" alt="image" src="https://github.com/user-attachments/assets/ca10078b-2ba5-4422-b4ae-997b7e98710a" />

```

np.log(df["Highly Positive Skew"])

```

<img width="470" height="557" alt="image" src="https://github.com/user-attachments/assets/959413ee-7b3f-4a04-bc67-bb563a946050" />

```
np.reciprocal(df["Moderate Positive Skew"])

```

<img width="536" height="586" alt="image" src="https://github.com/user-attachments/assets/89faf6a7-67d8-49de-96e3-20c708b14fd0" />

```
np.sqrt(df["Highly Positive Skew"])

```
<img width="597" height="577" alt="image" src="https://github.com/user-attachments/assets/1126de13-80a2-4dfd-b639-56fa451ca9a8" />


```

np.square(df["Highly Positive Skew"])

```
<img width="651" height="567" alt="image" src="https://github.com/user-attachments/assets/88576a1d-8c4c-48cf-a194-19df5b75ed6b" />


```

df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df

```

<img width="1276" height="517" alt="image" src="https://github.com/user-attachments/assets/8061f77f-d9fb-49db-ac6d-2cd0f74fb1be" />


```
df.skew()

```
<img width="515" height="294" alt="image" src="https://github.com/user-attachments/assets/dbeb41a1-8179-4c6d-84ba-39d5b3ec7f2c" />

```

df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()

```
<img width="615" height="354" alt="image" src="https://github.com/user-attachments/assets/14eac6a0-541d-4184-a130-4fa5b267928b" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df

```

<img width="1343" height="556" alt="image" src="https://github.com/user-attachments/assets/0591f458-a0b9-4a37-8a0c-0fd6a1502783" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

```
<img width="910" height="561" alt="image" src="https://github.com/user-attachments/assets/6d66241d-f888-4fe8-91fb-d4427534757d" />

```

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()

```
<img width="929" height="562" alt="image" src="https://github.com/user-attachments/assets/d34d2025-9c02-4321-9f2f-8b4a5b58c89d" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

```

<img width="1062" height="557" alt="image" src="https://github.com/user-attachments/assets/3a611137-67f3-4238-82be-b40499983f47" />

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()

```

<img width="828" height="546" alt="image" src="https://github.com/user-attachments/assets/f6e43487-6092-47ce-b7fb-dd693e7592c1" />

```

dt=pd.read_csv("/content/titanic_dataset.csv")
dt

```

<img width="1360" height="620" alt="image" src="https://github.com/user-attachments/assets/5c5cc5fb-2342-474a-be94-5cf27c1259f8" />


```

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()

```
<img width="1090" height="558" alt="image" src="https://github.com/user-attachments/assets/f46beb80-7722-4163-b3d2-ae3a6e37b38f" />


```

sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()

```

<img width="831" height="559" alt="image" src="https://github.com/user-attachments/assets/e1ba819e-443c-491e-bc8a-dfe1cd49b92f" />


### RESULT

   Thus the given data, Feature Encoding, Transformation process and save the data to a file
was performed successfully.    
